'''
@author: Y.J. Lee
'''

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool, PositionalEncoding

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
        x_emb = self.acts_embedding(x).squeeze(1)  # Shape(A, H)
        h = self.conv1(x_emb, edge_index)
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


class Mixer(nn.Module):
    def __init__(self, hidden_dim):
        super(Mixer, self).__init__()

        self.context_mixer = nn.Sequential(
            nn.LayerNorm(hidden_dim*3),
            nn.Linear(hidden_dim*3, hidden_dim*3),
            nn.LeakyReLU(), nn.Dropout(p=0.3))

        self.vector_mixer = nn.Sequential(
            nn.LayerNorm(hidden_dim*4),
            nn.Linear(hidden_dim*4, hidden_dim*2),
            nn.LeakyReLU(), nn.Dropout(p=0.3))

    def forward(self, h, z):
        context = self.context_mixer(z)  # Shape(batch_size, H*3)
        rep = self.vector_mixer(h)    # Shape(batch_size, seq_len, H*2)
        rep = rep[:, :-1, :]  # Shape(batch_size, seq_len-1, H*2)
        return context, rep


class DecoderBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_dec_layers):
        super(DecoderBlock, self).__init__()

        self.gru = nn.GRU(input_size=hidden_dim*3,
                          hidden_size=hidden_dim,
                          num_layers=num_dec_layers, 
                          dropout=0.3, batch_first=True)
        self.lin = nn.Linear(hidden_dim, input_dim)

    def forward(self, seq):

        self.gru.flatten_parameters()
        out, _ = self.gru(seq)
        return self.lin(out)  # Shape(batch_size, seq_len, input_dim)


class Network(nn.Module):

    def __init__(self, attr_dims:list, hidden_dim:int, num_enc_layers:int, num_dec_layers:int):
        super(Network, self).__init__()

        self.GraphEnc = GraphEncoder(attr_dims, hidden_dim)
        self.EventSeqEnc = EventSeqEncoder(attr_dims, hidden_dim, num_enc_layers)
        self.PosEnc = PositionalEncoding(hidden_dim)
        self.Mixer = Mixer(hidden_dim)

        self.Embedder = nn.ModuleList([
            nn.Embedding(dim+1, hidden_dim) for dim in attr_dims
        ])

        self.Decoder = nn.ModuleList([
            DecoderBlock(input_dim=int(dim+1), hidden_dim=hidden_dim,
                         num_dec_layers=num_dec_layers) for dim in attr_dims])
        
    def forward(self, Xg, Xs, Xa, edge_index, Act_pos, batch_g):

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
        temp = torch.arange(seq_len, device=out_s.device).reshape(-1, 1) # Shape(seq_len, 1)
        pos = self.PosEnc(temp) # Shape(seq_len, hidden_dim)
        pos = pos.unsqueeze(0).expand(batch_size, -1, -1)   # Shape(batch_size, seq_len, hidden_dim)

        # exclude the pe at padding position
        pos = pos * valid_mask.unsqueeze(2).float()  # Shape(batch_size, seq_len, hidden_dim)

        h = torch.cat([h, pos], dim=2)  # Shape(batch_size, seq_len, H*4)

        context, dec_input = self.Mixer(h, z)
        # context: Shape(batch_size, H*3)
        # dec_input: Shape(batch_size, seq_len-1, H*2)

        output = [] 

        for i, dec in enumerate(self.Decoder):

            if i < 1:
                Xa_emb = self.Embedder[0](Xa)[:, :-1, :]                        # Shape(batch_size, seq_len-1, hidden_dim)
                input0 = torch.cat([dec_input, Xa_emb], dim=2)                  # Shape(batch_size, seq_len-1, H*3)
                gru_input = torch.cat([context.unsqueeze(1), input0], dim=1)    # Shape(batch_size, seq_len, H*3)
                dec_output = dec(gru_input)                                     # Shape(batch_size, seq_len, out_dim)
                output.append(dec_output) 

            else:
                attr_emb = self.Embedder[i](Xs[:, :, i-1])[:, :-1, :]           # Shape(batch_size, seq_len-1, hidden_dim)
                input0 = torch.cat([dec_input, attr_emb], dim=2)                # Shape(batch_size, seq_len-1, H*3)
                gru_input = torch.cat([context.unsqueeze(1), input0], dim=1)    # Shape(batch_size, seq_len, H*3)
                dec_output = dec(gru_input)                                     # Shape(batch_size, seq_len, out_dim)
                output.append(dec_output)

        return output
