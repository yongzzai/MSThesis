'''
@author: Y.J. Lee
'''


import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from .layers import GraphEncoder, EventSeqEncoder, FeatureMixer
from utils.dataset import Dataset

class GAIN(nn.Module):
    def __init__(self, hidden_dim:int, num_gru_layer:int, batch_size:int, epochs:int, 
                 lr:float, seed:int):
        super(GAIN, self).__init__()

        torch.manual_seed(seed)
        self.hidden_dim = hidden_dim
        self.num_gru_layer = num_gru_layer
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr 

    def train(self, datachunk):
        self.graph_encoder = GraphEncoder(attr_dims=self.attribute_dims,
                                          hidden_dim=self.hidden_dim)
        self.event_encoder = EventSeqEncoder(attr_dims=self.attribute_dims,
                                             hidden_dim=self.hidden_dim,
                                             num_layers=self.num_gru_layer)
        self.feature_mixer = FeatureMixer(hidden_dim=self.hidden_dim)

        loader = DataLoader(
            dataset=datachunk,
            batch_size=self.batch_size,
            shuffle=True,
            follow_batch=['x','seq']
        )

        optimizer = torch.optim.AdamW(
            list(self.graph_encoder.parameters()) +
            list(self.event_encoder.parameters()) +
            list(self.feature_mixer.parameters()),
            lr=self.lr, weight_decay=1e-5
        )

        for epoch in range(self.epochs):
            for idx, batch in enumerate(loader):

                optimizer.zero_grad()
                Xg, edge_index = batch.x, batch.edge_index 
                Xs, Apos, Aorigin = batch.seq, batch.act_pos, batch.act_origin 
                batch_g, batch_s = batch.x_batch, batch.seq_batch

                Hg, Zg = self.graph_encoder(Xg, edge_index, batch_g)
                Zs, Hf, Hb = self.event_encoder(Xs)
                
                #TODO: 여기까진 잘 돌아가는거 확인함.

    def fit(self, dataset):
        
        self.attribute_dims = dataset.attribute_dims

        self.train(datachunk=dataset.DataChunks)


