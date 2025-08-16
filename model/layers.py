'''
@author: Y.J. Lee
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, AttentionalAggregation, global_mean_pool


class GraphEncoder(nn.Module):
    def __init__(self, emb_dim:int,attr_dims:list, hidden_dim:int, dropout:float=0.3):
        super(GraphEncoder, self).__init__()

        self.p = dropout
        self.dim_conv = hidden_dim*2

        self.acts_embedding = nn.Embedding(attr_dims[0]+1, emb_dim, padding_idx=0)
        self.pre = nn.Sequential(
            nn.LayerNorm(normalized_shape=emb_dim),
            nn.Dropout(p=dropout))
        
        self.conv1 = GATConv(in_channels=emb_dim,
                            out_channels=hidden_dim,
                            heads=2, dropout=dropout)
        
        self.conv2 = GATConv(in_channels=self.dim_conv,
                            out_channels=hidden_dim,
                            heads=2, dropout=dropout)
        
        self.aggr = AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(self.dim_conv, hidden_dim),
                nn.LeakyReLU(negative_slope=0.05),
                nn.Linear(hidden_dim, 1)))
        
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim*2),
            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim*2, hidden_dim))

    def forward(self, x, edge_index, batch_idx):

        x_emb = self.acts_embedding(x).squeeze(1)
        x_emb = self.pre(x_emb)

        # Inter-Attribute Local Information Aggregation
        out_conv = F.leaky_relu(self.conv1(x_emb, edge_index), negative_slope=0.05)
        out_conv = F.dropout(out_conv, p=self.p, training=self.training)
        out_conv = self.conv2(out_conv, edge_index)     # Shape(num_nodes, 2H)

        # Global Information Aggregation
        hidden = self.aggr(x=out_conv, index=batch_idx)
        # hidden = hidden + global_mean_pool(out_conv, batch_idx)    #!
        hidden = self.ffn(hidden)
        return out_conv, hidden     # Shape(num_nodes, 2H), Shape(num_graphs, H)


class EventSeqEncoder(nn.Module):
    def __init__(self, emb_dim:int, attr_dims:list, hidden_dim:int, num_layers:int, dropout:float=0.3):
        super(EventSeqEncoder, self).__init__()
        self._flattened = False
        self.in_dim = emb_dim * len(attr_dims[1:])
        
        self.attrEmbedder = nn.ModuleList([
            nn.Embedding(dim+1, emb_dim, padding_idx=0) for dim in attr_dims[1:]])
        
        self.gru_pre = nn.LayerNorm(normalized_shape=self.in_dim)

        self.gru = nn.GRU(input_size=self.in_dim, hidden_size=hidden_dim,
                   num_layers=num_layers, dropout=dropout if num_layers > 1 else 0.0,
                   batch_first=True, bidirectional=True)
        
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
        
        seq_embs = torch.empty(B, S, self.in_dim, 
                             device=seq.device, dtype=self.attrEmbedder[0].weight.dtype)
        for i, embedder in enumerate(self.attrEmbedder):
            start_idx = i * emb_dim
            end_idx = (i + 1) * emb_dim
            seq_embs[:, :, start_idx:end_idx] = embedder(seq[:, :, i])

        seq_embs = self.gru_pre(seq_embs)
        seq_packed = nn.utils.rnn.pack_padded_sequence(
            seq_embs, lengths.cpu(), batch_first=True, enforce_sorted=False)

        out_packed, hidden = self.gru(seq_packed)

        out_gru, _ = nn.utils.rnn.pad_packed_sequence(
            out_packed, batch_first=True, total_length=S)
        out_gru = out_gru.masked_fill(mask.unsqueeze(-1), 0.0)

        h_fwd, h_bwd = hidden[-2, :, :], hidden[-1, :, :]   # Shape(batch_size, hidden)
        z = self.ffn(torch.cat([h_fwd, h_bwd], dim=1))  # Shape(batch_size, hidden)

        return out_gru, z
        # Shape(batch_size, seq_len, hidden*2), Shape(batch_size, hidden)


class ActivityDecoder(nn.Module):
    def __init__(self, emb_dim:int, input_dim, hidden_dim, num_dec_layers, dropout:float=0.3):
        super(ActivityDecoder, self).__init__()

        self.nlayers = num_dec_layers
        self.p = dropout
        # query: hg.repeat(), key/value: mapped_g
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=1, batch_first=True,
            kdim=hidden_dim*2, vdim=hidden_dim*2)

        self.tfEmbedder = nn.Embedding(input_dim, emb_dim, padding_idx=0)
        self.ln = nn.LayerNorm(normalized_shape=hidden_dim + emb_dim)

        self.gru = nn.GRU(input_size=hidden_dim+emb_dim,
                          hidden_size=hidden_dim,
                          num_layers=num_dec_layers,
                          dropout=self.p if num_dec_layers > 1 else 0.0,
                          batch_first=True)
        
        self.proj = nn.Linear(hidden_dim+emb_dim, input_dim)    # tf, output

        self._flattened = False

    def forward(self, Xa, hg, map_g, pad_mask):
        if not self._flattened or not self.gru._flat_weights:
            self.gru.flatten_parameters()
            self._flattened = True

        mask = pad_mask[:, :-1]      # Shape(batch_size, seq_len-1)
        emb = self.tfEmbedder(Xa)[:, :-1, :]     # Shape(batch_size, seq_len-1, emb_dim)
        
        query = hg.unsqueeze(1).expand(-1, emb.size(1), -1)    # Shape(batch_size, seq_len-1, hidden_dim)
        kv = map_g[:, :-1, :]    # Shape(batch_size, seq_len-1, hidden_dim*2)
        
        seq_len = query.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=emb.device), diagonal=1)
        
        attn_out, _ = self.cross_attn(query=query, key=kv, value=kv,
                                      attn_mask=causal_mask, key_padding_mask=mask)  # Shape(batch_size, seq_len-1, hidden_dim)
        
        gru_input = F.dropout(torch.cat([attn_out, emb], dim=2), p=self.p, training=self.training)
        gru_input = self.ln(gru_input)  # Layer normalization #!
        gru_out, _ = self.gru(gru_input, hg.repeat(self.nlayers, 1, 1))  # Shape(batch_size, seq_len-1, hidden_dim)
        gru_out = gru_out.masked_fill(mask.unsqueeze(-1), 0.0)  # Masking the padded positions
        
        autoreg_input = gru_out.detach()     # Shape(batch_size, seq_len-1)
        proj_input = torch.cat([emb, gru_out], dim=2)
        logits = self.proj(proj_input)
        return autoreg_input, logits


class AttrDecoder(nn.Module):
    def __init__(self, emb_dim:int, input_dim, hidden_dim, num_dec_layers, dropout:float=0.3):
        super(AttrDecoder, self).__init__()

        self.nlayers = num_dec_layers
        self.p = dropout

        self.htproj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim)
          )  # Hidden state projection

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=1, batch_first=True,
            kdim=hidden_dim*2, vdim=hidden_dim*2)

        self.tfEmbedder = nn.Embedding(input_dim, emb_dim, padding_idx=0)

        self.ln = nn.LayerNorm(normalized_shape=hidden_dim*2 + emb_dim)
        self.gru = nn.GRU(input_size=hidden_dim*2 + emb_dim,      # tfembedding, attn_out, autoregressive_act
                          hidden_size=hidden_dim,
                          num_layers=num_dec_layers,
                          dropout=dropout if num_dec_layers > 1 else 0.0,
                          batch_first=True)
        
        self.proj = nn.Linear(hidden_dim*2 + emb_dim, input_dim)    # teacher forcing + gru_out + act_dec autoregressive

        self._flattened = False
    
    def forward(self, reg_input, xs, hs, ht, out_s, pad_mask):
        if not self._flattened or not self.gru._flat_weights:
            self.gru.flatten_parameters()
            self._flattened = True

        mask = pad_mask[:, :-1]
        emb = self.tfEmbedder(xs)[:, :-1, :]        # Shape(batch_size, seq_len-1, emb_dim)

        query = hs.unsqueeze(1).expand(-1, emb.size(1), -1)    # Shape(batch_size, seq_len-1, hidden_dim)
        kv = out_s[:, :-1, :]    # Shape(batch_size, seq_len-1, hidden_dim*2)
        
        # Causal mask 생성
        seq_len = query.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=emb.device), diagonal=1)

        attn_out, _ = self.cross_attn.forward(query=query, key=kv, value=kv,
                                              attn_mask=causal_mask, key_padding_mask=mask)

        ht = self.htproj(ht)
        gru_input = F.dropout(
            torch.cat([attn_out, emb, reg_input], dim=2),
            p=self.p, training=self.training)
        
        gru_input = self.ln(gru_input)  # Layer normalization
        gru_out, _ = self.gru(gru_input, ht.repeat(self.nlayers, 1, 1))
        gru_out = gru_out.masked_fill(mask.unsqueeze(-1), 0.0)

        proj_input = torch.cat([emb, gru_out, reg_input], dim=2)
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
        ht = torch.cat([hs, hg], dim=1)  # Shape(batch_size, H*2)

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
                logits = dec(autoreg_input, Xs[:, :, idx-1], hs, ht, out_s, pad_mask)
                logits = torch.cat([torch.zeros((logits.size(0), 1, logits.size(2)), device=logits.device), logits], dim=1)
                output.append(logits)  # Shape(batch_size, seq_len, input_dim)

        return output
                        