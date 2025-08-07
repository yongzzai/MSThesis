'''
@author: Y.J. Lee
'''


import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from .layers import Network
from utils.dataset import Dataset

class GAIN(nn.Module):
    def __init__(self, hidden_dim:int, num_enc_layers:int, num_dec_layers:int, batch_size:int, epochs:int, 
                 lr:float, seed:int):
        super(GAIN, self).__init__()

        torch.manual_seed(seed)
        
        self.hidden_dim = hidden_dim
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr 

    def train(self, datachunk):

        self.net = Network(attr_dims=self.attribute_dims,
                           hidden_dim=self.hidden_dim,
                           num_enc_layers=self.num_enc_layers,
                           num_dec_layers=self.num_dec_layers)

        loader = DataLoader(
            dataset=datachunk,
            batch_size=self.batch_size,
            shuffle=True,
            follow_batch=['x','seq']
        )

        optimizer = torch.optim.AdamW(
            list(self.net.parameters()),
            lr=self.lr, weight_decay=1e-5)

        print("=========================")
        print(self.attribute_dims)
        for epoch in range(self.epochs):
            for idx, batch in enumerate(loader):

                optimizer.zero_grad()

                s0, s = self.net(data=batch)
                                    

                #TODO: 여기까진 잘 돌아가는거 확인함.

    def fit(self, dataset):
        
        self.attribute_dims = dataset.attribute_dims

        self.train(datachunk=dataset.DataChunks)


