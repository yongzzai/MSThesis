'''
@author: Y.J. Lee
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, AttentionalAggregation


class GraphEncoder(nn.Module):
    def __init__(self, emb_dim:int,attr_dims:list, hidden_dim:int, dropout:float):
        super(GraphEncoder, self).__init__()

        self.p = dropout

        self.acts_embedding = nn.Embedding(attr_dims[0]+1, emb_dim, padding_idx=0)
        
        self.conv1 = GATConv(in_channels=emb_dim,
                            out_channels=hidden_dim,
                            heads=2, dropout=dropout)
        
        self.conv2 = GATConv(in_channels=hidden_dim*2,
                            out_channels=hidden_dim,
                            heads=2, dropout=dropout)
        
        self.aggr = AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_dim*2, hidden_dim),
                nn.LeakyReLU(negative_slope=0.05),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, 1)))
        
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim*2),
            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim*2, hidden_dim))

    def forward(self, x, edge_index, batch_idx):

        x_emb = self.acts_embedding(x).squeeze(1)

        out_conv = F.leaky_relu(self.conv1(x_emb, edge_index), negative_slope=0.05)
        out_conv = F.dropout(out_conv, p=self.p, training=self.training)
        out_conv = self.conv2(out_conv, edge_index)     # Shape(num_nodes, 2H)

        hidden = self.aggr(x=out_conv, index=batch_idx)
        hidden = self.ffn(hidden)
        return out_conv, hidden     # Shape(num_nodes, 2H), Shape(num_graphs, H)


class EventSeqEncoder(nn.Module):
    def __init__(self, emb_dim:int, attr_dims:list, hidden_dim:int, num_layers:int, dropout:float=0.3):
        super(EventSeqEncoder, self).__init__()
        self._flattened = False
        self.in_dim = emb_dim * len(attr_dims[1:])
        
        self.attrEmbedder = nn.ModuleList([
            nn.Embedding(dim+1, emb_dim, padding_idx=0) for dim in attr_dims[1:]])
        
        self.gru = nn.GRU(input_size=self.in_dim, hidden_size=hidden_dim,
                   num_layers=num_layers, dropout=dropout if num_layers > 1 else 0.0,
                   batch_first=True, bidirectional=True)
        
        self.pool = nn.MultiheadAttention(embed_dim=hidden_dim*2, num_heads=1, batch_first=True)

        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim * 2, hidden_dim))

    def forward(self, seq, mask):
        if not self._flattened or not self.gru._flat_weights:
            self.gru.flatten_parameters()
            self._flattened = True
        
        B, S, _ = seq.shape
        emb_dim = self.attrEmbedder[0].embedding_dim
        lengths = (~mask).sum(dim=1)
        
        seq_embs = torch.empty(B, S, self.in_dim, device=seq.device,
                               dtype=self.attrEmbedder[0].weight.dtype)
        for i, embedder in enumerate(self.attrEmbedder):
            start_idx = i * emb_dim
            end_idx = (i + 1) * emb_dim
            seq_embs[:, :, start_idx:end_idx] = embedder(seq[:, :, i])

        seq_packed = nn.utils.rnn.pack_padded_sequence(
            seq_embs, lengths.cpu(), batch_first=True, enforce_sorted=False)

        out_packed, h = self.gru(seq_packed)
        out_gru, _ = nn.utils.rnn.pad_packed_sequence(
            out_packed, batch_first=True, total_length=S)

        attn_out, _ = self.pool(out_gru, out_gru, out_gru, key_padding_mask=mask)
        attn_out.masked_fill_(mask.unsqueeze(-1), 0.)
        attn_out = attn_out.sum(dim=1) + torch.cat([h[-2,:,:], h[-1, :, :]], dim=1)  # Shape(batch_size, hidden*2)

        hidden = self.ffn(attn_out)      # Shape(batch_size, hidden)

        return out_gru, hidden
        # Shape(batch_size, seq_len, hidden*2), Shape(batch_size, hidden)


import math
class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * 
                           -(math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class ActivityDecoder(nn.Module):
    def __init__(self, emb_dim:int, input_dim, hidden_dim, num_dec_layers, dropout:float=0.3):
        super(ActivityDecoder, self).__init__()

        self.nlayers = num_dec_layers
        self.p = dropout

        self.tfEmbedder = nn.Embedding(input_dim, emb_dim, padding_idx=0)
        self.pe = PositionalEncoding(hidden_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=1, batch_first=True,
            kdim=hidden_dim*2, vdim=hidden_dim*2)

        self.norm = nn.LayerNorm(hidden_dim)

        self.gru = nn.GRU(input_size=hidden_dim+emb_dim,
                          hidden_size=hidden_dim,
                          num_layers=num_dec_layers,
                          dropout=dropout if num_dec_layers > 1 else 0.0,
                          batch_first=True)

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim+emb_dim, input_dim)    # tf, output
        )

        self._flattened = False

    def forward(self, Xa, hg, map_g, pad_mask):
        if not self._flattened or not self.gru._flat_weights:
            self.gru.flatten_parameters()
            self._flattened = True

        mask = pad_mask[:, :-1]      # Shape(batch_size, seq_len-1)
        lengths = (~mask).sum(dim=1)

        emb = self.tfEmbedder(Xa)[:, :-1, :]     # Shape(batch_size, seq_len-1, emb_dim)

        query = self.pe(hg.unsqueeze(1).expand(-1, emb.size(1), -1))  # Shape(batch_size, seq_len-1, hidden_dim)
        kv = map_g[:, :-1, :]    # Shape(batch_size, seq_len-1, hidden_dim*2)
        
        causal_mask = torch.triu(torch.ones(query.size(1), query.size(1), dtype=torch.bool, device=emb.device), diagonal=1)
        
        attn_out, _ = self.cross_attn(query=query, key=kv, value=kv,
                                      attn_mask=causal_mask,
                                      key_padding_mask=mask)  # Shape(batch_size, seq_len-1, hidden_dim)
        attn_out = self.norm(attn_out + query)
        gru_input = F.dropout(torch.cat([attn_out, emb], dim=2), p=self.p, training=self.training)

        gru_input_packed = nn.utils.rnn.pack_padded_sequence(
            gru_input, lengths.cpu(), batch_first=True, enforce_sorted=False)

        gru_out_packed, _ = self.gru(gru_input_packed, hg.repeat(self.nlayers, 1, 1))  # Shape(batch_size, seq_len-1, hidden_dim)
        gru_out, _ = nn.utils.rnn.pad_packed_sequence(
            gru_out_packed, batch_first=True, total_length=emb.size(1))
        
        autoreg_input = gru_out.detach()     # Shape(batch_size, seq_len-1)

        proj_input = torch.cat([gru_out, emb], dim=2)
        logits = self.proj(proj_input)
        return autoreg_input, logits


class AttrDecoder(nn.Module):
    def __init__(self, emb_dim:int, input_dim, hidden_dim, num_dec_layers, dropout:float=0.3):
        super(AttrDecoder, self).__init__()

        self.nlayers = num_dec_layers
        self.p = dropout

        self.pe = PositionalEncoding(hidden_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=1, batch_first=True,
            kdim=hidden_dim*2, vdim=hidden_dim*2)

        self.norm = nn.LayerNorm(hidden_dim)

        self.tfEmbedder = nn.Embedding(input_dim, emb_dim, padding_idx=0)
        self.gru = nn.GRU(input_size=hidden_dim*2 + emb_dim,      # tfembedding, attn_out, autoregressive_act
                          hidden_size=hidden_dim,
                          num_layers=num_dec_layers,
                          dropout=dropout if num_dec_layers > 1 else 0.0,
                          batch_first=True)
        
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim*2 + emb_dim, input_dim)    # teacher forcing + gru_out + act_dec autoregressive
        )

        self._flattened = False
    
    def forward(self, reg_input, xs, hs, out_s, pad_mask):
        if not self._flattened or not self.gru._flat_weights:
            self.gru.flatten_parameters()
            self._flattened = True

        mask = pad_mask[:, :-1]
        lengths = (~mask).sum(dim=1)

        emb = self.tfEmbedder(xs)[:, :-1, :]        # Shape(batch_size, seq_len-1, emb_dim)

        query = self.pe(hs.unsqueeze(1).expand(-1, emb.size(1), -1)) # Shape(batch_size, seq_len-1, hidden_dim)
        kv = out_s[:, :-1, :]    # Shape(batch_size, seq_len-1, hidden_dim*2)
        
        causal_mask = torch.triu(torch.ones(query.size(1), query.size(1), dtype=torch.bool, device=emb.device), diagonal=1)
        attn_out, _ = self.cross_attn(query=query, key=kv, value=kv,
                                      attn_mask=causal_mask,
                                      key_padding_mask=mask)
        attn_out = self.norm(attn_out + query)      # Residual

        reg_input = reg_input + torch.randn_like(reg_input, device=reg_input.device) * 0.02  # Add noise to reg_input
        input0 = F.dropout(torch.cat([attn_out, emb], dim=2), p=self.p, training=self.training)
        gru_input = torch.cat([input0, reg_input], dim=2)

        gru_input_packed = nn.utils.rnn.pack_padded_sequence(
            gru_input, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        gru_out_packed, _ = self.gru(gru_input_packed, hs.repeat(self.nlayers, 1, 1))

        gru_out, _ = nn.utils.rnn.pad_packed_sequence(
            gru_out_packed, batch_first=True, total_length=emb.size(1))

        # input1 = F.dropout(torch.cat([gru_out, emb], dim=2), p=self.p, training=self.training)
        proj_input = torch.cat([gru_out, emb, reg_input], dim=2)
        logits = self.proj(proj_input)
        return logits
    

class Network(nn.Module):

    def __init__(self, emb_dim:int, attr_dims:list, hidden_dim:int, num_enc_layers:int, num_dec_layers:int, 
                 encoder_dropout:float=0.3, decoder_dropout:float=0.3):
        super(Network, self).__init__()

        self.GraphEnc = GraphEncoder(emb_dim, attr_dims, hidden_dim, dropout=encoder_dropout)
        self.EventSeqEnc = EventSeqEncoder(emb_dim, attr_dims, hidden_dim, num_enc_layers, dropout=encoder_dropout)

        self.Decoder = nn.ModuleList(
            [ActivityDecoder(emb_dim=emb_dim, input_dim=int(attr_dims[0]+1), 
                            hidden_dim=hidden_dim, num_dec_layers=num_dec_layers, dropout=decoder_dropout)] +
            [AttrDecoder(emb_dim=emb_dim, input_dim=int(dim+1), hidden_dim=hidden_dim, num_dec_layers=num_dec_layers, dropout=decoder_dropout)
            for dim in attr_dims[1:]])
                
    def forward(self, Xg, Xs, Xa, edge_index, Act_pos, batch_g):

        pad_mask = (Xs == 0).all(dim=2) # Shape(batch_size, seq_len)

        out_g, hg = self.GraphEnc(Xg, edge_index, batch_g)     # Shape(num_nodes, 2H), Shape(num_graphs, H)
        out_s, hs = self.EventSeqEnc(Xs, pad_mask)    # Shape(B, S, 2H), Shape(B, H)

        valid_mask = Act_pos >= 0
        valid_act_pos = Act_pos[valid_mask]        
        batch_indices = torch.arange(out_s.shape[0], device=Act_pos.device).unsqueeze(1).expand(-1, out_s.shape[1])[valid_mask]
        seq_indices = torch.arange(out_s.shape[1], device=Act_pos.device).unsqueeze(0).expand(out_s.shape[0], -1)[valid_mask]  

        map_g = torch.zeros_like(out_s, device=out_g.device)
        map_g[batch_indices, seq_indices] = out_g[valid_act_pos]

        output = []
        for idx, dec in enumerate(self.Decoder):
            if idx == 0:
                autoreg_input, logits = dec(Xa, hg, map_g, pad_mask)
                logits = torch.cat([torch.zeros((logits.size(0), 1, logits.size(2)), device=logits.device), logits], dim=1)
                output.append(logits)  # Shape(batch_size, seq_len, input_dim)
            else:
                logits = dec(autoreg_input, Xs[:, :, idx-1], hs, out_s, pad_mask)
                logits = torch.cat([torch.zeros((logits.size(0), 1, logits.size(2)), device=logits.device), logits], dim=1)
                output.append(logits)  # Shape(batch_size, seq_len, input_dim)

        return output
                        