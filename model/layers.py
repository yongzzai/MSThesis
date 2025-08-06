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
        self.act1 = nn.LeakyReLU()
        self.do1 = nn.Dropout(p=0.3)

        self.conv2 = GATConv(in_channels=hidden_dim * 2,
                            out_channels=hidden_dim,
                            heads=2, dropout=0.3)
        self.act2 = nn.LeakyReLU()
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

        self.context_mixer = nn.Sequential(
            nn.LayerNorm(hidden_dim*3),
            nn.Linear(hidden_dim*3, hidden_dim*3),
            nn.LeakyReLU(), nn.Dropout(p=0.3))

        self.vector_mixer = nn.Sequential(
            nn.LayerNorm(hidden_dim*3),
            nn.Linear(hidden_dim*3, hidden_dim*2),
            nn.LeakyReLU(), nn.Dropout(p=0.3))

    def forward(self, h, z):
        s0 = self.context_mixer(z)  # Shape(batch_size, H*3)
        s = self.vector_mixer(h)    # Shape(batch_size, seq_len, H*2)
        return s0, s


class Decoder(nn.Module):
    def __init__(self, attr_dims:list, hidden_dim:int, num_dec_layers:int):
        super(Decoder, self).__init__()

        self.gru = nn.ModuleList([
            nn.GRU(input_size=hidden_dim*3,
                   hidden_size=hidden_dim,
                   num_layers=num_dec_layers,
                   dropout=0.3, batch_first=True) for _ in range(len(attr_dims))])
        
        self.projection = nn.ModuleList([
            nn.Linear(hidden_dim, attr_dims[i]+1) for i in range(len(attr_dims))
        ])
    
    def forward(self):
        pass

from torch_geometric.nn import PositionalEncoding

class Network(nn.Module):
    def __init__(self, attr_dims:list, hidden_dim:int, num_layers:int):
        super(Network, self).__init__()

        self.GraphEnc = GraphEncoder(attr_dims, hidden_dim)
        self.EventSeqEnc = EventSeqEncoder(attr_dims, hidden_dim, num_layers)
        self.FeatureMixer = FeatureMixer(hidden_dim)
        self.PosEnc = PositionalEncoding(hidden_dim)

    def forward(self, data):

        Xg, edge_index = data.x, data.edge_index 
        Xs, Act_pos = data.seq, data.act_pos 
        batch_g, batch_s = data.x_batch, data.seq_batch

        out_g, z_g = self.GraphEnc(Xg, edge_index, batch_g)         # Shape(num_nodes, H), Shape(batch_size, H)
        out_s, h_fwd, h_bwd = self.EventSeqEnc(Xs)      # Shape(batch_size, seq_len, H*2), Shape(batch_size, H), Shape(batch_size, H)

        z = torch.cat([z_g, h_fwd, h_bwd], dim=1)  # Shape(batch_size, H*3)

        batch_size, seq_len, _ = out_s.shape
        hidden_dim = out_g.shape[1]
        
        mapped_g = torch.zeros(batch_size, seq_len, hidden_dim, device=out_g.device)
        
        valid_mask = Act_pos >= 0
        valid_act_pos = Act_pos[valid_mask]
        
        batch_indices = torch.arange(batch_size, device=Act_pos.device).unsqueeze(1).expand(-1, seq_len)[valid_mask]
        seq_indices = torch.arange(seq_len, device=Act_pos.device).unsqueeze(0).expand(batch_size, -1)[valid_mask]
            
        mapped_g[batch_indices, seq_indices] = out_g[valid_act_pos]
        
        h = torch.cat([out_s, mapped_g], dim=2)  # Shape(batch_size, seq_len, H*3)

        # positional encoding
        temp = torch.arange(seq_len, device=out_s.device).unsqueeze(0).expand(batch_size, -1)  # Shape(batch_size, seq_len)
        pos = self.PosEnc(temp) # Shape(batch_size, seq_len, hidden_dim)

        # exclude the pe at padding position
        pos = pos * valid_mask.unsqueeze(2).float()  # Shape(batch_size, seq_len, hidden_dim)

        h = torch.cat([h, pos], dim=2)

        s0, s = self.FeatureMixer(h, z) 
        # Shape(batch_size, H*3), Shape(batch_size, seq_len, H*3)

        return s0, s

        #TODO: Positional Encoding 수정

        #TODO: Decoder 구현 필요
        #TODO: s는 encoder 초기의 embedding을 가져와서 concat해야함 (Teacher Forcing)
        #TODO: s0는 decoder gru의 첫번째 cell input으로 사용함.
