'''
@author: Y.J. Lee
'''


import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from .layers import Network
from tqdm import tqdm
import numpy as np


class GAIN(nn.Module):

    def __init__(self, embed_dim:int, hidden_dim:int, num_enc_layers:int, num_dec_layers:int, 
                 enc_dropout:float, dec_dropout:float,
                 batch_size:int, epochs:int, lr:float, seed:int=None):
        super(GAIN, self).__init__()

        if seed is not None:
            torch.manual_seed(int(seed))

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        self.enc_rate = enc_dropout
        self.dec_rate = dec_dropout

        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def fit(self, dataset):

        self.net = Network(emb_dim=self.embed_dim,
                           attr_dims=dataset.attribute_dims,
                           hidden_dim=self.hidden_dim,
                           num_enc_layers=self.num_enc_layers,
                           num_dec_layers=self.num_dec_layers,
                           encoder_dropout=self.enc_rate,
                           decoder_dropout=self.dec_rate).to(self.device)
        
        loader = DataLoader(
            dataset=dataset.DataChunks, batch_size=self.batch_size,
            shuffle=True, follow_batch=['x','seq'], pin_memory=True, num_workers=8,
            prefetch_factor=2, persistent_workers=True)

        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=self.lr, weight_decay=0.)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.lr*0.1)

        criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=0)        

        for epoch in range(self.epochs):
            self.net.train()
            epoch_loss = 0.0
            for batch in tqdm(loader, desc="Epoch {}".format(epoch+1)):

                batch = batch.to(self.device)

                Xg, Xs, Xa = batch.x, batch.seq, batch.act_origin
                edge_index = batch.edge_index
                Act_pos, batch_g = batch.act_pos, batch.x_batch

                logits = self.net(Xg=Xg, Xs=Xs, Xa=Xa,
                    edge_index=edge_index, Act_pos=Act_pos, batch_g=batch_g)

                TrueSeq = torch.cat([Xa.unsqueeze(2), Xs], dim=2)   # Shape(batch_size, seq_len, num_attr)

                batch_loss = 0.0
                for idx in range(len(logits)):
                    ground_truth = TrueSeq[:, :, idx]               # Shape(batch_size, seq_len)
                    pred = logits[idx].permute(0,2,1)       # Shape(batch_size, num_classes, seq_len)
                    mask = ground_truth > 0     # mask pad token  Shape(batch_size, seq_len)
                    mask[:, 0] = False  # mask start token

                    loss = criterion(pred, ground_truth) * mask.float()
                    loss = loss.sum(dim=1) / mask.float().sum(dim=1)        # Shape(batch_size)
                    batch_loss += loss.mean()
                    
                optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
                optimizer.step()

                epoch_loss += batch_loss.item()

            scheduler.step()
            print("Epoch {} - Loss = {:.4f}".format(epoch+1, epoch_loss/len(loader)))


    def detect(self, dataset):
        
        loader = DataLoader(
            dataset=dataset.DataChunks, batch_size=self.batch_size,
            shuffle=False, follow_batch=['x','seq'])
        
        self.net.eval()
        with torch.no_grad():
            attr_result = []
            for i, batch in enumerate(loader):
                batch = batch.to(self.device)

                TrueSeq = torch.cat([batch.act_origin.unsqueeze(2), batch.seq], dim=2)   # Shape(batch_size, seq_len, num_attr)
                pad_mask = TrueSeq[:,:,0] != 0 # Shape(batch_size, seq_len), pad위치는 True
                pad_mask[:, 0] = False  # <start> token은 pad가 아님
                pad_mask = pad_mask.unsqueeze(2)  # Shape(batch_size, seq_len, 1)

                logits = self.net(Xg=batch.x, Xs=batch.seq, Xa=batch.act_origin,
                    edge_index=batch.edge_index, Act_pos=batch.act_pos, batch_g=batch.x_batch)
                
                current_batch_res = []
                for idx, logit in enumerate(logits):
                    true_idx = TrueSeq[:, :, idx].unsqueeze(2)              # Shape(batch_size, seq_len, 1)
                    pred = torch.softmax(logit, dim=2)                      # Shape(batch_size, seq_len, num_classes)

                    true_proba = pred.gather(dim=2, index=true_idx)         # Shape(batch_size, seq_len, 1)
                    
                    pred_temp = pred.clone()
                    pred_temp[pred_temp <= true_proba] = 0                  # true_proba보다 작거나 같으면 0으로 설정
                    anomaly_score = pred_temp.sum(dim=2, keepdim=True)      # 나머지 확률들의 합
                    
                    anomaly_score = anomaly_score * pad_mask.float()        # 패딩 마스크 적용
                    current_batch_res.append(anomaly_score)
                attr_result.append(torch.cat(current_batch_res, dim=2)) # Shape(batch_size, seq_len, num_attr)
        
        attr_level_anomaly_score = np.array(torch.cat(attr_result, dim=0).detach().cpu())
        event_level_anomaly_score = attr_level_anomaly_score.max((2))  # Shape(num_cases, seq_len)
        trace_level_anomaly_score = attr_level_anomaly_score.max((1,2)) # Shape(num_cases,)
        return trace_level_anomaly_score, event_level_anomaly_score, attr_level_anomaly_score

#TODO: 뭔가 하나만 더 추가하고 싶은데..

# 실제 인덱스에 해당하는 logits에 sigmoid적용 --> 0~1 : Anomaly이면 확률이 작음. 정상이면 확률이 높음.
# 실제 인덱스 외에 나머지 인덱스 확률의 합 --> 0~1 : Anomaly이면 합이 큼.