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

    def __init__(self, hidden_dim:int, num_enc_layers:int, num_dec_layers:int, 
                 enc_dropout:float, dec_dropout:float,
                 batch_size:int, epochs:int, lr:float, seed:int):
        super(GAIN, self).__init__()

        torch.manual_seed(int(seed))

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

        self.net = Network(attr_dims=dataset.attribute_dims,
                           hidden_dim=self.hidden_dim,
                           num_enc_layers=self.num_enc_layers,
                           num_dec_layers=self.num_dec_layers,
                           encoder_dropout=self.enc_rate,
                           decoder_dropout=self.dec_rate)

        self.net = self.net.to(self.device)
        
        loader = DataLoader(
            dataset=dataset.DataChunks, batch_size=self.batch_size,
            shuffle=True, follow_batch=['x','seq'])
        
        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.lr, weight_decay=1e-5)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=len(loader), eta_min=0.)
        
        criterion = nn.CrossEntropyLoss(reduction='none', label_smoothing=0.05)

        for epoch in range(self.epochs):

            self.net.train()
            epoch_loss = 0.0
            
            for batch in tqdm(loader, desc="Epoch {}".format(epoch+1)):

                optimizer.zero_grad()

                batch = batch.to(self.device)

                Xg, Xs, Xa = batch.x, batch.seq, batch.act_origin
                edge_index = batch.edge_index
                Act_pos = batch.act_pos
                batch_g = batch.x_batch
                
                logits = self.net(Xg=Xg, Xs=Xs, Xa=Xa,
                    edge_index=edge_index, Act_pos=Act_pos, batch_g=batch_g)

                TrueSeq = torch.cat([Xa.unsqueeze(2), Xs], dim=2)   # Shape(batch_size, seq_len, num_attr)

                batch_loss = 0.0
                for idx in range(len(logits)):
                    true = TrueSeq[:, :, idx]               # Shape(batch_size, seq_len)
                    pred = logits[idx].permute(0,2,1)       # Shape(batch_size, num_classes, seq_len)

                    mask = true > 0     # mask pad token
                    mask[:, 0] = False  # mask start token
                    loss = criterion(pred, true) * mask.float()   # Shape(batch_size, seq_len)
                    loss = loss.sum(dim=1) / mask.float().sum(dim=1)       # Shape(batch_size)
                    batch_loss += loss.mean()

                batch_loss.backward()                
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
                optimizer.step()
                epoch_loss += batch_loss.item() / len(dataset.attribute_dims)
        
            scheduler.step()
            print("Epoch {} - Loss = {:.4f}".format(epoch+1, epoch_loss/len(loader)))


    def detect(self, dataset):
        self.net.eval()
        
        loader = DataLoader(
            dataset=dataset.DataChunks, batch_size=self.batch_size,
            shuffle=False, follow_batch=['x','seq'])
        
        with torch.no_grad():
            proba_res = []

            for batch in loader:

                batch = batch.to(self.device)
                TrueSeq = torch.cat([batch.act_origin.unsqueeze(2), batch.seq], dim=2)   # Shape(batch_size, seq_len, num_attr)

                logits = self.net(Xg=batch.x, Xs=batch.seq, Xa=batch.act_origin,
                    edge_index=batch.edge_index, Act_pos=batch.act_pos, batch_g=batch.x_batch)

                batch_res = []
                for idx in range(len(logits)):
                    true = TrueSeq[:, :, idx].unsqueeze(2)                                   # Shape(batch_size, seq_len, 1)
                    pred = torch.nn.functional.softmax(logits[idx], dim=2)      # Shape(batch_size, seq_len, num_classes)

                    proba = pred.gather(dim=2, index=true)  # Shape(batch_size, seq_len, 1)
                    batch_res.append(proba)
                
                proba_res.append(torch.cat(batch_res, dim=2))  # Shape(batch_size, seq_len, num_attr)

        return np.array(torch.cat(proba_res, dim=0).detach().cpu())     # Shape(num_cases, seq_len, num_attr)
