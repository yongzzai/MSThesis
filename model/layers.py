'''
@author: Y.J. Lee
'''

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool, PositionalEncoding

class GraphEncoder(nn.Module):
    def __init__(self, attr_dims:list, hidden_dim:int, dropout:float=0.3):
        super(GraphEncoder, self).__init__()

        self.acts_embedding = nn.Embedding(attr_dims[0]+1, hidden_dim)
        
        self.conv1 = GATConv(in_channels=hidden_dim,
                            out_channels=hidden_dim,
                            heads=2, dropout=dropout)
        self.post1 = nn.Sequential(
            nn.LayerNorm(hidden_dim*2),
            nn.GELU(), nn.Dropout(p=dropout))

        self.conv2 = GATConv(in_channels=hidden_dim * 2,
                            out_channels=hidden_dim,
                            heads=2, dropout=dropout)
        self.post2 = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x, edge_index, batch_idx):

        x_emb = self.acts_embedding(x).squeeze(1)  # Shape(A, H)
        
        h = self.conv1(x_emb, edge_index)
        h = self.post1(h)

        h = self.conv2(h, edge_index)   # Shape(A, 2H)
        h = self.post2(h)

        z = global_mean_pool(h, batch_idx)  # Shape(B, H)
        return h, z


class EventSeqEncoder(nn.Module):
    def __init__(self, attr_dims:list, hidden_dim:int, num_layers:int, dropout:float=0.3):
        super(EventSeqEncoder, self).__init__()

        self.attr_embs = nn.ModuleList([nn.Embedding(dim+1, hidden_dim) for dim in attr_dims[1:]])
        
        self.input_dim = hidden_dim * len(attr_dims[1:])
        self.input_norm = nn.LayerNorm(self.input_dim)
        
        self.gru = nn.GRU(input_size=self.input_dim,
                          hidden_size=hidden_dim,
                          num_layers=num_layers, 
                          dropout=dropout if num_layers > 1 else 0.0, 
                          batch_first=True, bidirectional=True)
                        
        self._flattened = False

    def forward(self, seq):
        # Only flatten parameters when necessary (e.g., after model.cuda() or DataParallel)
        if not self._flattened or not self.gru._flat_weights:
            self.gru.flatten_parameters()
            self._flattened = True
        # seq: (Batch_size, Seq_len, Attr_num)
        
        batch_size, seq_len, _ = seq.shape
        num_attrs = len(self.attr_embs)
        hidden_dim = self.attr_embs[0].embedding_dim
        
        if num_attrs == 1:
            seq_embs = self.attr_embs[0](seq[:, :, 0])
        else:
            seq_embs = torch.empty(batch_size, seq_len, num_attrs * hidden_dim, 
                                 device=seq.device, dtype=self.attr_embs[0].weight.dtype)
            
            for i, emb_layer in enumerate(self.attr_embs):
                start_idx = i * hidden_dim
                end_idx = (i + 1) * hidden_dim
                seq_embs[:, :, start_idx:end_idx] = emb_layer(seq[:, :, i])
        
        seq_embs = self.input_norm(seq_embs)
        out, hidden = self.gru(seq_embs)        # seq_embs: (batch_size, seq_len, num_attrs * hidden_dim)

        fwd, bwd = hidden[-2, :, :], hidden[-1, :, :]   # Shape(batch_size, hidden)
        return out, fwd, bwd


class Mixer(nn.Module):
    def __init__(self, hidden_dim, dropout:float=0.3):
        super(Mixer, self).__init__()

        self.context_mixer = nn.Sequential(
            nn.LayerNorm(hidden_dim*3),
            nn.Linear(hidden_dim*3, hidden_dim*3),
            nn.GELU(), nn.Dropout(p=dropout))

        self.vector_mixer = nn.Sequential(
            nn.LayerNorm(hidden_dim*4),
            nn.Linear(hidden_dim*4, hidden_dim*2),
            nn.GELU(), nn.Dropout(p=dropout))

    def forward(self, h, z):
        context = self.context_mixer(z)  # Shape(batch_size, H*3)
        rep = self.vector_mixer(h)    # Shape(batch_size, seq_len, H*2)
        rep = rep[:, :-1, :]  # Shape(batch_size, seq_len-1, H*2)
        return context, rep


class DecoderBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_dec_layers, dropout:float=0.3):
        super(DecoderBlock, self).__init__()

        self.input_norm = nn.LayerNorm(hidden_dim * 3)
        
        self.gru = nn.GRU(input_size=hidden_dim*3,
                          hidden_size=hidden_dim,
                          num_layers=num_dec_layers, 
                          dropout=dropout, 
                          batch_first=True)

        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_dim*2),
            nn.Linear(hidden_dim*2, input_dim)  # input_dim is the number of attributes
        )
        # self.fc = nn.Linear(hidden_dim*2, input_dim)
        self._flattened = False

    def forward(self, seq, emb):

        if not self._flattened or not self.gru._flat_weights:
            self.gru.flatten_parameters()
            self._flattened = True
            
        seq = self.input_norm(seq)
        gru_out, _ = self.gru(seq)  # Shape(batch_size, seq_len, hidden_dim)

        emb = torch.nn.functional.pad(emb, (0, 0, 1, 0), value=0.0)  # Shape(batch_size, seq_len, hidden_dim)
        out = torch.cat([gru_out, emb], dim=2)  # Shape(batch_size, seq_len, hidden_dim*2)

        return self.fc(out)  # Shape(batch_size, seq_len, input_dim)


class Network(nn.Module):

    def __init__(self, attr_dims:list, hidden_dim:int, num_enc_layers:int, num_dec_layers:int, 
                 encoder_dropout:float=0.3, decoder_dropout:float=0.3):
        super(Network, self).__init__()

        self.GraphEnc = GraphEncoder(attr_dims, hidden_dim, dropout=encoder_dropout)
        self.EventSeqEnc = EventSeqEncoder(attr_dims, hidden_dim, num_enc_layers, dropout=encoder_dropout)
        self.PosEnc = PositionalEncoding(hidden_dim)
        self.Mixer = Mixer(hidden_dim, dropout=encoder_dropout)

        self.Embedder = nn.ModuleList([
            nn.Embedding(dim+1, hidden_dim)
            for dim in attr_dims
        ])

        self.Decoder = nn.ModuleList([
            DecoderBlock(input_dim=int(dim+1), hidden_dim=hidden_dim,
                         num_dec_layers=num_dec_layers, dropout=decoder_dropout) for dim in attr_dims])
                
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
        
        rep = torch.cat([out_s, mapped_g], dim=2)  # Shape(batch_size, seq_len, H*3)

        # positional encoding
        temp = torch.arange(seq_len, device=out_s.device).reshape(-1, 1) # Shape(seq_len, 1)
        pos = self.PosEnc(temp) # Shape(seq_len, hidden_dim)
        pos = pos.unsqueeze(0).expand(batch_size, -1, -1)   # Shape(batch_size, seq_len, hidden_dim)

        # exclude the pe at padding position
        pos = pos * valid_mask.unsqueeze(2).float()  # Shape(batch_size, seq_len, hidden_dim)

        rep = torch.cat([rep, pos], dim=2)  # Shape(batch_size, seq_len, H*4)

        context, dec_input = self.Mixer(rep, z)
        # context: Shape(batch_size, H*3)
        # dec_input: Shape(batch_size, seq_len-1, H*2)

        context_expanded = context.unsqueeze(1)  # Shape(batch_size, 1, H*3)
        
        embeddings = []
        for i in range(len(self.Decoder)):
            if i < 1:
                emb = self.Embedder[0](Xa)[:, :-1, :]  # Shape(batch_size, seq_len-1, hidden_dim)
            else:
                emb = self.Embedder[i](Xs[:, :, i-1])[:, :-1, :]  # Shape(batch_size, seq_len-1, hidden_dim)
            embeddings.append(emb)

        output = []
        for i, (dec, emb) in enumerate(zip(self.Decoder, embeddings)):
            # Combine dec_input and embedding efficiently
            input0 = torch.cat([dec_input, emb], dim=2)  # Shape(batch_size, seq_len-1, H*3)
            gru_input = torch.cat([context_expanded, input0], dim=1)  # Shape(batch_size, seq_len, H*3)
            dec_output = dec(gru_input, emb)  # Shape(batch_size, seq_len, out_dim)
            output.append(dec_output)

        return output
