'''
@author: Y.J. Lee
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


# class GraphEncoder(nn.Module):
#     def __init__(self, attr_dims:list, hidden_dim:int, dropout:float=0.3):
#         super(GraphEncoder, self).__init__()

#         self.acts_embedding = nn.Embedding(attr_dims[0]+1, hidden_dim, padding_idx=0)

#         self.conv1 = GATConv(in_channels=hidden_dim,
#                             out_channels=hidden_dim,
#                             heads=2, dropout=dropout)
#         self.post1 = nn.Sequential(
#             nn.LeakyReLU(negative_slope=0.1),
#             nn.Dropout(p=dropout))

#         self.conv2 = GATConv(in_channels=hidden_dim * 2,
#                             out_channels=hidden_dim,
#                             heads=2, dropout=dropout)

#         self.ffn = nn.Sequential(
#                         nn.LayerNorm(hidden_dim * 2),
#                         nn.Linear(hidden_dim * 2, hidden_dim))

#     def forward(self, x, edge_index, batch_idx):

#         x_emb = self.acts_embedding(x).squeeze(1)  # Shape(A, H)

#         out1 = self.conv1(x_emb, edge_index)
#         out1 = self.post1(out1)     # Shape(A, 2H)

#         h = self.conv2(out1, edge_index)   # Shape(A, 2H)
#         h = self.ffn(h)

#         z = global_mean_pool(h, batch_idx)
#         return h, z


# class EventSeqEncoder(nn.Module):
#     def __init__(self, attr_dims:list, hidden_dim:int, num_layers:int, dropout:float=0.3):
#         super(EventSeqEncoder, self).__init__()

#         self.input_dim = hidden_dim * len(attr_dims[1:])

#         self.attr_embs = nn.ModuleList([nn.Embedding(dim+1, hidden_dim, padding_idx=0) for dim in attr_dims[1:]])
#         # self.input_norm = nn.LayerNorm(self.input_dim)
        
#         self.gru = nn.GRU(input_size=self.input_dim,
#                           hidden_size=hidden_dim,
#                           num_layers=num_layers, 
#                           dropout=dropout, 
#                           batch_first=True, bidirectional=True)
#         self.out_norm = nn.LayerNorm(hidden_dim * 2)

#         self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=1, 
#                                                dropout=dropout,batch_first=True)
        
#         self.ffn = nn.Sequential(
#             nn.LayerNorm(hidden_dim*2),
#             nn.Linear(hidden_dim*2, hidden_dim*2),
#             nn.LeakyReLU(negative_slope=0.1),
#             nn.Dropout(p=dropout),
#             nn.Linear(hidden_dim*2, hidden_dim*2))

#         self.last_norm = nn.LayerNorm(hidden_dim * 2)

#         self._flattened = False

#     def forward(self, seq, mask):
#         if not self._flattened or not self.gru._flat_weights:
#             self.gru.flatten_parameters()
#             self._flattened = True
        
#         num_attrs = len(self.attr_embs)
#         batch_size, seq_len, _ = seq.shape
#         hidden_dim = self.attr_embs[0].embedding_dim
        
#         if num_attrs == 1:
#             seq_embs = self.attr_embs[0](seq[:, :, 0])
#         else:
#             seq_embs = torch.empty(batch_size, seq_len, num_attrs * hidden_dim, 
#                                  device=seq.device, dtype=self.attr_embs[0].weight.dtype)
            
#             for i, emb_layer in enumerate(self.attr_embs):
#                 start_idx = i * hidden_dim
#                 end_idx = (i + 1) * hidden_dim
#                 seq_embs[:, :, start_idx:end_idx] = emb_layer(seq[:, :, i])
        
#         # seq_embs = self.input_norm(seq_embs)

#         lengths = (~mask).sum(dim=1)
#         packed = nn.utils.rnn.pack_padded_sequence(
#             seq_embs, lengths.cpu(), batch_first=True, enforce_sorted=False)

#         gru_out_packed, hidden = self.gru(packed)

#         gru_out, _ = nn.utils.rnn.pad_packed_sequence(
#             gru_out_packed, batch_first=True, total_length=seq_len)

#         attn_in = self.out_norm(gru_out)  # Shape(batch_size, seq_len, hidden_dim * 2)
#         attn_out, _ = self.attention(attn_in, attn_in, attn_in, key_padding_mask=mask)  # Self-Attention

#         gru_out = attn_out + gru_out
#         enc_out = self.ffn(gru_out)

#         enc_out = enc_out + gru_out
#         enc_out = self.last_norm(enc_out)

#         enc_out = enc_out.masked_fill(mask.unsqueeze(-1), 0.0)

#         fwd, bwd = hidden[-2, :, :], hidden[-1, :, :]   # Shape(batch_size, hidden)
#         return enc_out, fwd, bwd


# class Mixer(nn.Module):
#     def __init__(self, hidden_dim, m=4, dropout:float=0.3):
#         super(Mixer, self).__init__()

#         self.rep_dim = hidden_dim * 3
#         self.m = m

#         self.input_norm = nn.LayerNorm(self.rep_dim)
#         self.kv_norm = nn.LayerNorm(self.rep_dim)
#         self.kv_proj = nn.Linear(self.rep_dim, m*self.rep_dim)
#         self.cross_attn = nn.MultiheadAttention(self.rep_dim, num_heads=1, 
#                                                 dropout=dropout, batch_first=True)

#         self.residual = nn.Linear(self.rep_dim, hidden_dim*2)
        
#         self.ffn = nn.Sequential(
#             nn.LayerNorm(self.rep_dim),
#             nn.Linear(self.rep_dim, self.rep_dim),
#             nn.LeakyReLU(negative_slope=0.1),
#             nn.Dropout(p=dropout),
#             nn.Linear(self.rep_dim, hidden_dim*2))

#     def forward(self, rep, z, mask):

#         context = self.kv_norm(z)
#         kv = self.kv_proj(context).view(rep.shape[0], self.m, self.rep_dim)

#         attn_in = self.input_norm(rep)
#         attn_in = attn_in.masked_fill(mask.unsqueeze(-1), 0.0)  # Mask padding tokens
#         attn_out, _ = self.cross_attn(attn_in, kv, kv)

#         rep = attn_out + rep
#         ffn_out = self.ffn(rep)
#         mixer_out = ffn_out + self.residual(rep)
#         mixer_out = mixer_out.masked_fill(mask.unsqueeze(-1), 0.0)  # Mask padding tokens

#         mixer_out = mixer_out[:, :-1, :]    # Shape(batch_size, seq_len-1, H*2)

#         return mixer_out, context

# class DecoderBlock(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_dec_layers, dropout:float=0.3):
#         super(DecoderBlock, self).__init__()

#         # self.input_norm = nn.LayerNorm(hidden_dim * 3)
        
#         self.gru = nn.GRU(input_size=hidden_dim*3,
#                           hidden_size=hidden_dim,
#                           num_layers=num_dec_layers, 
#                           dropout=dropout if num_dec_layers > 1 else 0, 
#                           batch_first=True)
#         self.fc = nn.Sequential(
#             nn.LayerNorm(hidden_dim * 2),
#             nn.Linear(hidden_dim * 2, input_dim))

#         self.do = nn.Dropout(p=dropout)

#         self._flattened = False

#     def forward(self, seq, emb, pad_mask):

#         if not self._flattened or not self.gru._flat_weights:
#             self.gru.flatten_parameters()
#             self._flattened = True

#         # seq = self.input_norm(seq)

#         pad_mask = pad_mask.to(torch.bool)
#         seq = seq.masked_fill(pad_mask.unsqueeze(-1), 0.0)  # Mask padding tokens
#         lengths = (~pad_mask).sum(dim=1).clamp_min(1)

#         packed_seq = nn.utils.rnn.pack_padded_sequence(
#                 seq, lengths.cpu(), batch_first=True, enforce_sorted=False)

#         gru_out_packed, _ = self.gru(packed_seq)  # Shape(batch_size, seq_len, hidden_dim)
#         gru_out, _ = nn.utils.rnn.pad_packed_sequence(
#             gru_out_packed, batch_first=True, total_length=seq.size(1))
        
#         emb = torch.nn.functional.pad(emb, (0, 0, 1, 0), value=0.0)  # Shape(batch_size, seq_len, hidden_dim)
#         out = torch.cat([gru_out, self.do(emb)], dim=2)  # Shape(batch_size, seq_len, hidden_dim*2)
#         out = self.fc(out)  # Shape(batch_size, seq_len, input_dim)

#         out = out.masked_fill(pad_mask.unsqueeze(-1), 0.0)  # Mask padding tokens
#         return out


# class Network(nn.Module):

#     def __init__(self, attr_dims:list, hidden_dim:int, num_enc_layers:int, num_dec_layers:int, 
#                  encoder_dropout:float=0.3, decoder_dropout:float=0.3):
#         super(Network, self).__init__()

#         self.GraphEnc = GraphEncoder(attr_dims, hidden_dim, dropout=encoder_dropout)
#         self.EventSeqEnc = EventSeqEncoder(attr_dims, hidden_dim, num_enc_layers, dropout=encoder_dropout)
#         # self.PosEnc = PositionalEncoding(hidden_dim)
#         self.Mixer = Mixer(hidden_dim, dropout=encoder_dropout)

#         self.tfEmbedder = nn.ModuleList([
#             nn.Embedding(dim+1, hidden_dim, padding_idx=0)
#             for dim in attr_dims])
        
#         self.Decoder = nn.ModuleList([
#             DecoderBlock(input_dim=int(dim+1), hidden_dim=hidden_dim,
#                          num_dec_layers=num_dec_layers, dropout=decoder_dropout) 
#                          for dim in attr_dims])
                
#     def forward(self, Xg, Xs, Xa, edge_index, Act_pos, batch_g):
        
#         pad_mask = (Xs == 0).all(dim=2)
#         dec_pad_mask = torch.cat([torch.zeros_like(pad_mask[:, :1]), pad_mask[:, :-1]], dim=1)

#         out_g, z_g = self.GraphEnc(Xg, edge_index, batch_g)         # Shape(num_nodes, H), Shape(batch_size, H)
#         out_s, h_fwd, h_bwd = self.EventSeqEnc(Xs, pad_mask)      # Shape(batch_size, seq_len, H*2), Shape(batch_size, H), Shape(batch_size, H)

#         batch_size, seq_len, _ = out_s.shape
#         hidden_dim = out_g.shape[1]

#         mapped_g = torch.zeros(batch_size, seq_len, hidden_dim, device=out_g.device)    # mapping activity embeddings

#         valid_mask = Act_pos >= 0
#         valid_act_pos = Act_pos[valid_mask]
        
#         batch_indices = torch.arange(batch_size, device=Act_pos.device).unsqueeze(1).expand(-1, seq_len)[valid_mask]
#         seq_indices = torch.arange(seq_len, device=Act_pos.device).unsqueeze(0).expand(batch_size, -1)[valid_mask]
            
#         mapped_g[batch_indices, seq_indices] = out_g[valid_act_pos]
        
#         rep = torch.cat([out_s, mapped_g], dim=2)  # Shape(batch_size, seq_len, H*3)
#         z = torch.cat([z_g, h_fwd, h_bwd], dim=1)  # Shape(batch_size, H*3)
        
#         dec_input, context = self.Mixer(rep, z, pad_mask)
#         # context: Shape(batch_size, H*3)
#         # dec_input: Shape(batch_size, seq_len-1, H*2)
        
#         embeddings = []
#         for i in range(len(self.Decoder)):
#             if i == 0:
#                 emb = self.tfEmbedder[0](Xa)[:, :-1, :]  # Shape(batch_size, seq_len-1, hidden_dim)
#             else:
#                 emb = self.tfEmbedder[i](Xs[:, :, i-1])[:, :-1, :]  # Shape(batch_size, seq_len-1, hidden_dim)
#             embeddings.append(emb)

#         output = []
#         for i, (dec, emb) in enumerate(zip(self.Decoder, embeddings)):
#             input0 = torch.cat([dec_input, F.dropout(emb, p=0.3, training=self.training)], dim=2)  # Shape(batch_size, seq_len-1, H*3)
#             gru_input = torch.cat([context.unsqueeze(1), input0], dim=1)  # Shape(batch_size, seq_len, H*3)
#             dec_output = dec(gru_input, emb, dec_pad_mask)  # Shape(batch_size, seq_len, out_dim)
#             output.append(dec_output)

#         return output


class GraphEncoder(nn.Module):
    def __init__(self, attr_dims:list, hidden_dim:int, dropout:float=0.3):
        super(GraphEncoder, self).__init__()

        self.p = dropout
        self.acts_embedding = nn.Embedding(attr_dims[0]+1, hidden_dim, padding_idx=0)

        self.conv = GATConv(in_channels=hidden_dim,
                            out_channels=hidden_dim,
                            heads=2, dropout=self.p)
        
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=self.p))

    def forward(self, x, edge_index, batch_idx):

        x_emb = self.acts_embedding(x).squeeze(1)  # Shape(A, H)
        x_emb = F.dropout(x_emb, p=self.p, training=self.training)

        h = self.conv(x_emb, edge_index)
        h = self.ffn(h)     # Shape(A, 2H)

        z = global_mean_pool(h, batch_idx)
        return h, z


class EventSeqEncoder(nn.Module):
    def __init__(self, attr_dims:list, hidden_dim:int, num_layers:int, dropout:float=0.3):
        super(EventSeqEncoder, self).__init__()

        self.input_dim = hidden_dim * len(attr_dims[1:])

        self.attr_embs = nn.ModuleList([nn.Embedding(dim+1, hidden_dim, padding_idx=0) for dim in attr_dims[1:]])
        self.gru = nn.GRU(input_size=self.input_dim,
                          hidden_size=hidden_dim,
                          num_layers=num_layers, 
                          dropout=dropout, 
                          batch_first=True, bidirectional=True)

        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim*2),
            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout))

        self._flattened = False

    def forward(self, seq, mask):
        if not self._flattened or not self.gru._flat_weights:
            self.gru.flatten_parameters()
            self._flattened = True
        
        num_attrs = len(self.attr_embs)
        batch_size, seq_len, _ = seq.shape
        hidden_dim = self.attr_embs[0].embedding_dim
        
        seq_embs = torch.empty(batch_size, seq_len, num_attrs * hidden_dim, 
                                 device=seq.device, dtype=self.attr_embs[0].weight.dtype)
            
        for i, emb_layer in enumerate(self.attr_embs):
            start_idx = i * hidden_dim
            end_idx = (i + 1) * hidden_dim
            seq_embs[:, :, start_idx:end_idx] = F.dropout(emb_layer(seq[:, :, i]), p=0.3, training=self.training)

        lengths = (~mask).sum(dim=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            seq_embs, lengths.cpu(), batch_first=True, enforce_sorted=False)

        gru_out_packed, hidden = self.gru(packed)

        gru_out, _ = nn.utils.rnn.pad_packed_sequence(
            gru_out_packed, batch_first=True, total_length=seq_len)

        out = gru_out.masked_fill(mask.unsqueeze(-1), 0.0)
        out = self.ffn(out)

        fwd, bwd = hidden[-2, :, :], hidden[-1, :, :]   # Shape(batch_size, hidden)
        return out, fwd, bwd

#class Mixer(nn.Module):
#
#    def __init__(self, hidden_dim:int, dropout:float=0.3):
#        super().__init__()
#        self.rep_dim = hidden_dim * 3
#
#        self.input_norm = nn.LayerNorm(self.rep_dim)
#        self.gate_gen   = nn.Linear(self.rep_dim, self.rep_dim)   # z → gate(3H)
#        self.proj       = nn.Linear(self.rep_dim, hidden_dim * 2) # (3H → 2H)
#        self.dropout    = nn.Dropout(dropout)
#
#    def forward(self, rep, z, mask):
#        # rep: (B, L, 3H), z: (B, 3H), mask: (B, L) True=pad
#        gate = torch.sigmoid(self.gate_gen(z))         # (B, 3H)
#        rep2 = self.input_norm(rep) * gate.unsqueeze(1)
#        rep2 = self.dropout(rep2)
#        rep2 = rep2.masked_fill(mask.unsqueeze(-1), 0.0)
#
#        dec_input = self.proj(rep2)[:, :-1, :]
#        context = z
#        return dec_input, context


class Mixer(nn.Module):

    def __init__(self, hidden_dim:int, dropout:float=0.3):
        super().__init__()
        H = hidden_dim
        self.rep_dim = 3 * H
        self.out_dim = 2 * H

        self.input_norm = nn.LayerNorm(self.rep_dim)
        self.r_proj = nn.Linear(self.rep_dim, self.rep_dim, bias=False)
        self.z_proj = nn.Linear(self.rep_dim, self.rep_dim, bias=True)
        self.dw_conv = nn.Conv1d(self.rep_dim, self.rep_dim, kernel_size=3,
                                 padding=1, groups=self.rep_dim, bias=False)

        self.proj = nn.Linear(self.rep_dim, self.out_dim)

        self.to_affine = nn.Linear(self.rep_dim, 2 * self.rep_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, rep, z, mask):

        rep_n = self.input_norm(rep)  # (B, L, 3H)

        r_term = self.r_proj(rep_n)                            # (B, L, 3H)
        z_term = self.z_proj(z).unsqueeze(1)                   # (B, 1, 3H)
        pre = r_term + z_term                                  # (B, L, 3H)

        pre = pre.transpose(1, 2)                          # (B, 3H, L)
        pre = self.dw_conv(pre)                            # (B, 3H, L)
        pre = pre.transpose(1, 2)                          # (B, L, 3H)

        gamma, beta = self.to_affine(pre).chunk(2, dim=-1) # (B, L, 3H) x2
        gamma = 1.0 + torch.tanh(gamma)
        rep2 = rep_n * gamma + beta                        # (B, L, 3H)

        rep2 = self.dropout(rep2)
        rep2 = rep2.masked_fill(mask.unsqueeze(-1).bool(), 0.0)

        dec_input = self.proj(rep2)[:, :-1, :]                 # (B, L-1, 2H)
        return dec_input, z

class SharedDecoder(nn.Module):

    def __init__(self, attr_dims, hidden_dim, num_layers, dropout: float = 0.3):
        super().__init__()

        H = hidden_dim

        self.gru = nn.GRU(
            input_size=3*H, hidden_size=H,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True)
        
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2*H, int(dim + 1))
            ) for dim in attr_dims
        ])
        self.drop = nn.Dropout(dropout)
        self._flattened = False

    def forward(self, dec_input, context, emb_list, pad_mask):
        outputs = []

        for i, emb in enumerate(emb_list):
            input0 = torch.cat([dec_input, emb], dim=2)
            gru_in = torch.cat([context.unsqueeze(1), input0], dim=1)

            mask = pad_mask.to(torch.bool)                     # (B, L)
            gru_in = gru_in.masked_fill(mask.unsqueeze(-1), 0.0)
            lengths = (~mask).sum(dim=1).clamp_min(1)

            if not self._flattened or not self.gru._flat_weights:
                self.gru.flatten_parameters()
                self._flattened = True

            packed = nn.utils.rnn.pack_padded_sequence(
                gru_in, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.gru(packed)
            gru_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=gru_in.size(1))

            emb_padded = F.pad(emb, (0, 0, 1, 0), value=0.0)
            feat = torch.cat([gru_out, self.drop(emb_padded)], dim=2)

            # 속성별 head
            out = self.heads[i](feat)                          # (B, L, dim_i+1)
            out = out.masked_fill(mask.unsqueeze(-1), 0.0)
            outputs.append(out)

        return outputs


class Network(nn.Module):
    def __init__(self, attr_dims:list, hidden_dim:int, num_enc_layers:int, num_dec_layers:int, 
                 encoder_dropout:float=0.3, decoder_dropout:float=0.3):
        super(Network, self).__init__()

        self.GraphEnc = GraphEncoder(attr_dims, hidden_dim, dropout=encoder_dropout)
        self.EventSeqEnc = EventSeqEncoder(attr_dims, hidden_dim, num_enc_layers, dropout=encoder_dropout)
        self.Mixer = Mixer(hidden_dim, dropout=encoder_dropout)

        self.tfEmbedder = nn.ModuleList([
            nn.Embedding(dim+1, hidden_dim, padding_idx=0)
            for dim in attr_dims])

        self.Decoder = SharedDecoder(attr_dims=attr_dims, hidden_dim=hidden_dim,
                                     num_layers=num_dec_layers, dropout=decoder_dropout)

    def forward(self, Xg, Xs, Xa, edge_index, Act_pos, batch_g):
        pad_mask = (Xs == 0).all(dim=2)
        dec_pad_mask = torch.cat([torch.zeros_like(pad_mask[:, :1]), pad_mask[:, :-1]], dim=1)

        out_g, z_g = self.GraphEnc(Xg, edge_index, batch_g)            # (N, H), (B, H)
        out_s, h_fwd, h_bwd = self.EventSeqEnc(Xs, pad_mask)           # (B, L, 2H), (B, H), (B, H)

        B, L, _ = out_s.shape
        H = out_g.shape[1]

        mapped_g = torch.zeros(B, L, H, device=out_g.device)
        valid_mask = (Act_pos >= 0)
        batch_indices = torch.arange(B, device=Act_pos.device).unsqueeze(1).expand(-1, L)[valid_mask]
        seq_indices   = torch.arange(L, device=Act_pos.device).unsqueeze(0).expand(B, -1)[valid_mask]
        mapped_g[batch_indices, seq_indices] = out_g[Act_pos[valid_mask]]

        rep = torch.cat([out_s, mapped_g], dim=2)                       # (B, L, 3H)
        z   = torch.cat([z_g, h_fwd, h_bwd], dim=1)                     # (B, 3H)

        dec_input, context = self.Mixer(rep, z, pad_mask)               # (B, L-1, 2H), (B, 3H)

        emb_list = []
        for i in range(len(self.tfEmbedder)):
            if i == 0:
                emb = self.tfEmbedder[0](Xa)[:, :-1, :]                 # (B, L-1, H)
            else:
                emb = self.tfEmbedder[i](Xs[:, :, i-1])[:, :-1, :]      # (B, L-1, H)
            emb_list.append(emb)

        output = self.Decoder(dec_input, context, emb_list, dec_pad_mask)
        return output
