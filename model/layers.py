'''
@author: Y.J. Lee
'''

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class GraphEncoder(nn.Module):
    def __init__(self, attr_dims:list, hidden_dim:int):
        super(GraphEncoder, self).__init__()

        self.acts_embedding = nn.Embedding(attr_dims[0], hidden_dim)
        self.conv1 = GATConv(in_channels=hidden_dim,
                            out_channels=hidden_dim,
                            heads=2, dropout=0.3)
        self.act1 = nn.ELU()
        self.do1 = nn.Dropout(p=0.3)    

        self.conv2 = GATConv(in_channels=hidden_dim * 2,
                            out_channels=hidden_dim,
                            heads=2, dropout=0.3)
        self.act2 = nn.ELU()
        self.do2 = nn.Dropout(p=0.3)
        self.projection = nn.Linear(hidden_dim * 2, hidden_dim)     # Shape(A, H)

    def forward(self, x, edge_index):
        x = self.acts_embedding(x)
        x = self.conv1(x, edge_index)
        x = self.act1(x)
        x = self.do1(x)

        x = self.conv2(x, edge_index)
        x = self.act2(x)
        x = self.do2(x)

        x = self.projection(x)
        return x                        # Shape(A, H)


class EventSeqEncoder(nn.Module):
    def __init__(self, attr_dims:list, hidden_dim:int, num_layers:int):
        super(EventSeqEncoder, self).__init__()

        self.attr_embs = nn.ModuleList([nn.Embedding(dim, hidden_dim) for dim in attr_dims[1:]])
        self.gru = nn.GRU(input_size=hidden_dim*len(attr_dims[1:]),
                          hidden_size=hidden_dim,
                          num_layers=num_layers, 
                          dropout=0.3, batch_first=True, bidirectional=True)
        
    
    def forward(self, x):
        pass
        # x: Shape(B, L, A)



