'''
@original code from: Guan et al., 2024
@refactor by: Y.J. Lee
'''

import torch
import numpy as np
import torch.utils.data as Data
from torch.utils.data import DataLoader


from layers.encoder import Encoder
from layers.decoder import Decoder
import itertools

class COMB():
    def __init__(self, d_model:int = 64, n_layers_agg:int = 2, n_layers:int = 2, n_heads:int = 4,
                  ffn_hidden:int = 128, drop_prob:float = 0.1, n_epochs:int = 20, batch_size:int = 64, lr:float = 0.0002,
                  b1:float=0.5, b2:float=0.999):
        
        self.d_model = d_model
        self.n_layers_agg = n_layers_agg
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ffn_hidden = ffn_hidden
        self.drop_prob = drop_prob
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.b1, self.b2 = b1, b2

        self.name = 'COMB'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def fit(self, dataset):
        
        attr_dims = dataset.attribute_dims
        max_len = dataset.max_len
        
        self.encoder = Encoder(attr_dims, max_len, self.d_model, self.ffn_hidden,
                               self.n_heads, self.n_layers, self.n_layers_agg, self.drop_prob,
                               self.device)
        self.decoder = Decoder(attr_dims, max_len, self.d_model, self.ffn_hidden,
                               self.n_heads, self.n_layers, self.drop_prob, self.device)        
        self.encoder.to(self.device)
        self.decoder.to(self.device)

        Xs = []
        for i, _ in enumerate(attr_dims):
            Xs.append(torch.LongTensor(dataset.features[i]))
        mask = torch.BoolTensor(dataset.mask)
        train_dataset = Data.TensorDataset(*Xs, mask)
        self.test_dataset = Data.TensorDataset(*Xs, mask)

        train_loader = DataLoader(train_dataset, self.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)


        optimizer = torch.optim.Adam(itertools.chain(self.encoder.parameters(),self.decoder.parameters()),lr=self.lr, betas=(self.b1, self.b2))
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.n_epochs//2, gamma=0.1)
        
        for epoch in range(self.n_epochs):

            train_loss = 0.0
            train_num = 0
            for i, Xs in enumerate(train_loader):

                mask = Xs[-1]
                Xs = Xs[:-1]
                mask = mask.to(self.device)
                for k, X in enumerate(Xs):
                    Xs[k] = X.to(self.device)

                optimizer.zero_grad()

                enc_output = self.encoder(Xs, mask)
                fake_X = self.decoder(Xs, enc_output, mask)

                loss = 0.0
                for idx in range(len(attr_dims)):
                    pred = torch.softmax(fake_X[idx][:, :-1, :], dim=2).flatten(0, -2)
                    true = Xs[idx][:, 1:].flatten()

                    corr_pred = pred.gather(1, true.view(-1, 1)).flatten().to(self.device).reshape(-1, fake_X[0].shape[1] - 1)

                    ce = -torch.log(corr_pred)
                    loss += ce.masked_select((mask[:, 1:])).mean()
                
                train_loss += loss.item() * Xs[0].shape[0]
                train_num += Xs[0].shape[0]
                loss.backward()
                optimizer.step()
            
            train_loss_epoch = train_loss / train_num
            print(f"[Epoch {epoch+1:{len(str(self.n_epochs))}}/{self.n_epochs}] "
                  f"[loss: {train_loss_epoch:3f}]")
            scheduler.step()
    

    def detect(self, dataset):
        
        attr_shape=(dataset.num_cases,dataset.max_len,dataset.num_attributes)
        attr_dims = dataset.attribute_dims

        detect_loader = DataLoader(self.test_dataset, self.batch_size,
                                   shuffle=False, num_workers=4, pin_memory=True)
        
        self.encoder.eval()
        self.decoder.eval()
        pos = 0

        with torch.no_grad():
            attr_level_abnormal_scores = np.zeros(attr_shape)

            for Xs in detect_loader:
                mask = Xs[-1]
                Xs = Xs[:-1]
                for k, tempX in enumerate(Xs):
                    Xs[k] = tempX.to(self.device)
                mask = mask.to(self.device)

                enc_output = self.encoder(Xs, mask)
                fake_X = self.decoder(Xs, enc_output, mask)

                for attr_index in range(len(attr_dims)):
                    fake_X[attr_index] = torch.softmax(fake_X[attr_index][:, :-1, :], dim=2)

                # Calculate abnormal scores
                for attr_index in range(len(attr_dims)):
                    truepos = Xs[attr_index][:, 1:].flatten()
                    p = fake_X[attr_index].reshape((truepos.shape[0], -1)).gather(1, truepos.view(-1, 1)).squeeze()
                    p_distribution = fake_X[attr_index].reshape((truepos.shape[0], -1))

                    p_distribution = p_distribution + 1e-8

                    attr_level_abnormal_scores[pos: pos + Xs[attr_index].shape[0], 1: ,attr_index] = \
                        ((torch.sum(torch.log(p_distribution) * p_distribution, 1) - torch.log(p)).reshape((Xs[attr_index].shape[0],-1))*(mask[:,1:])).detach().cpu()

                pos += Xs[attr_index].shape[0]
            
            trace_level_abnormal_scores = attr_level_abnormal_scores.max((1, 2))
            event_level_abnormal_scores = attr_level_abnormal_scores.max((2))
            return trace_level_abnormal_scores, event_level_abnormal_scores, attr_level_abnormal_scores



        
        
