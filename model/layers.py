'''
@author: Y.J. Lee
'''

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool

class GraphEncoder(nn.Module):
    def __init__(self, attr_dims:list, hidden_dim:int):
        super(GraphEncoder, self).__init__()

        self.acts_embedding = nn.Embedding(attr_dims[0]+1, hidden_dim)
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

    def forward(self, x, edge_index, batch_idx):
        x = self.acts_embedding(x).squeeze(1)  # Shape(A, H)
        h = self.conv1(x, edge_index)
        h = self.act1(h)
        h = self.do1(h)

        h = self.conv2(h, edge_index)
        h = self.act2(h)
        h = self.do2(h)

        h = self.projection(h)              # Shape(A, H)
        z = global_mean_pool(h, batch_idx)  # Shape(B, H)
        return h, z


class EventSeqEncoder(nn.Module):
    def __init__(self, attr_dims:list, hidden_dim:int, num_layers:int):
        super(EventSeqEncoder, self).__init__()

        self.attr_embs = nn.ModuleList([nn.Embedding(dim+1, hidden_dim) for dim in attr_dims[1:]])
        self.gru = nn.GRU(input_size=hidden_dim*len(attr_dims[1:]),
                          hidden_size=hidden_dim,
                          num_layers=num_layers, 
                          dropout=0.3, batch_first=True, bidirectional=True)


    def forward(self, seq):
        self.gru.flatten_parameters()
        # seq: (Batch_size, Seq_len, Attr_num)
        embs = []
        for i, m in enumerate(self.attr_embs):
            embs.append(m(seq[:, :, i]))
        
        seq_embs = torch.cat(embs, dim=2)
        # seq_embs: (batch_size, seq_len, num_attrs * hidden_dim)

        out, hidden = self.gru(seq_embs)

        fwd, bwd = hidden[-2, :, :], hidden[-1, :, :]   # Shape(batch_size, hidden)
        return out, fwd, bwd


class FeatureMixer(nn.Module):
    def __init__(self, hidden_dim):
        super(FeatureMixer, self).__init__()

        self.state_mixer = nn.Sequential(
            nn.LayerNorm(hidden_dim*3),
            nn.Linear(hidden_dim*3, hidden_dim*3),
            nn.ELU(), nn.Dropout(p=0.3)
            )

    def forward(self, z, fwd, bwd):
        
        # Initial Input of decoder gru cell
        vector = torch.cat([z, fwd, bwd], dim=1)  # Shape(B, H*3)

