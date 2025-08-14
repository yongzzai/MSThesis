'''
@author: Y.J. Lee
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, AttentionalAggregation


class GraphEncoder(nn.Module):
    def __init__(self, attr_dims:list, hidden_dim:int, dropout:float=0.3):
        super(GraphEncoder, self).__init__()
        '''
        Graph Encoder for Activity embedding
        
        returns:
        h | Shape(num_nodes, hidden_dim*2)
        z | Shape(num_graphs, hidden_dim*2)
        '''

        self.p = dropout
        self.dim_conv = hidden_dim*2

        self.acts_embedding = nn.Embedding(attr_dims[0]+1, 32, padding_idx=0)
        self.pre = nn.Sequential(
            nn.LayerNorm(normalized_shape=32),
            nn.Dropout(p=dropout))
        
        self.conv1 = GATConv(in_channels=32,
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

    def forward(self, x, edge_index, batch_idx):

        x_emb = self.acts_embedding(x).squeeze(1)  # Shape(A, H)
        x_emb = self.pre(x_emb)

        h = F.leaky_relu(self.conv1(x_emb, edge_index), negative_slope=0.05)
        h = F.dropout(h, p=self.p, training=self.training)
        h = self.conv2(h, edge_index)

        z = self.aggr.forward(x=h, index=batch_idx)
        return h, z     # Shape(num_nodes, hidden_dim*2), Shape(num_graphs, hidden_dim*2)


class EventSeqEncoder(nn.Module):
    def __init__(self, attr_dims:list, hidden_dim:int, num_layers:int, dropout:float=0.3):
        super(EventSeqEncoder, self).__init__()
        '''
        Event Sequence Encoder for Event Attributes embedding
        input: Shape(B, S, H)
        '''
        self._flattened = False
        self.in_dim = 32 * len(attr_dims[1:])
        
        self.attrEmbedder = nn.ModuleList([
            nn.Embedding(dim+1, 32, padding_idx=0) for dim in attr_dims[1:]])
        
        self.gru_pre = nn.LayerNorm(normalized_shape=self.in_dim)

        self.gru = nn.GRU(input_size=self.in_dim, hidden_size=hidden_dim,
                   num_layers=num_layers, dropout=dropout if num_layers > 1 else 0.0,
                   batch_first=True, bidirectional=True)
        
        self.attention = nn.MultiheadAttention(hidden_dim*2, num_heads=1,
                                               batch_first=True, dropout=dropout)

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
        packed = nn.utils.rnn.pack_padded_sequence(
            seq_embs, lengths.cpu(), batch_first=True, enforce_sorted=False)

        gru_out_packed, hidden = self.gru(packed)

        gru_out, _ = nn.utils.rnn.pad_packed_sequence(
            gru_out_packed, batch_first=True, total_length=S)
        gru_out = gru_out.masked_fill(mask.unsqueeze(-1), 0.0)

        fwd, bwd = hidden[-2, :, :], hidden[-1, :, :]   # Shape(batch_size, hidden)
        return gru_out, fwd, bwd



class Mixer(nn.Module):
    def __init__(self, hidden_dim, dropout:float=0.3):
        super(Mixer, self).__init__()

        self.H = hidden_dim*4
        
        self.local_ffn = nn.Sequential(
            nn.LayerNorm(self.H),
            nn.Linear(self.H, self.H),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Dropout(p=dropout),
            nn.Linear(self.H, hidden_dim))
        
        self.global_ffn = nn.Sequential(
            nn.LayerNorm(self.H),
            nn.Linear(self.H, self.H),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Dropout(p=dropout),
            nn.Linear(self.H, hidden_dim))

    def forward(self, local_context, global_context, mask):

        local_mixed = self.local_ffn(local_context).masked_fill(mask.unsqueeze(-1), 0.0)  # Shape(batch_size, seq_len, hidden_dim)
        local_mixed = local_mixed[:,:-1,:]      # Shape(batch_size, seq_len-1, hidden_dim), Remove last token
        global_mixed = self.global_ffn(global_context)   # Shape(batch_size, hidden_dim)
        return local_mixed, global_mixed
    

class DecoderBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_dec_layers, dropout:float=0.3):
        super(DecoderBlock, self).__init__()

        self.num_layers = num_dec_layers

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=1, batch_first=True)

        self.norm = nn.LayerNorm(hidden_dim * 2)

        self.gru = nn.GRU(input_size=hidden_dim*2,  # attn_out + emb
                          hidden_size=hidden_dim,
                          num_layers=self.num_layers, 
                          dropout=dropout if self.num_layers > 1 else 0.0,
                          batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, input_dim))

        self._flattened = False

    def forward(self, mixed_out, s0, emb, mask):
        if not self._flattened or not self.gru._flat_weights:
            self.gru.flatten_parameters()
            self._flattened = True

        B, S, H = mixed_out.shape   # batch_size, seq_len - 1, hidden_dim

        query = s0.unsqueeze(1).repeat(1, S, 1)  # [B, S, H]
        key, value = mixed_out, mixed_out

        attn_out, _ = self.cross_attn(
            query=query, key=key, value=value,
            key_padding_mask=mask)
        attn_out = attn_out + query 
        attn_out = attn_out.masked_fill(mask.unsqueeze(-1), 0.0)  # [B, S, H]

        gru_input = torch.cat([attn_out, emb], dim=2)  # [B, S, H*2]
        gru_input = self.norm(gru_input)
        gru_input = gru_input.masked_fill(mask.unsqueeze(-1), 0.0)
        
        lengths = (~mask).sum(dim=1).clamp_min(1)
        packed_input = nn.utils.rnn.pack_padded_sequence(
            gru_input, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        h0 = s0.unsqueeze(0).repeat(self.num_layers, 1, 1)  # [num_layers, B, H]
        gru_out_packed, _ = self.gru(packed_input, h0)
        
        gru_out, _ = nn.utils.rnn.pad_packed_sequence(
            gru_out_packed, batch_first=True, total_length=S)
        
        # Final output --> <start> token reconstruction X
        out = self.fc(torch.cat([gru_out, emb], dim=2))  # Shape(batch_size, seq_len - 1, input_dim)
        return out


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

        self.Decoder = nn.ModuleList([
            DecoderBlock(input_dim=int(dim+1), hidden_dim=hidden_dim,
                         num_dec_layers=num_dec_layers, dropout=decoder_dropout) 
                         for dim in attr_dims])
                
    def forward(self, Xg, Xs, Xa, edge_index, Act_pos, batch_g):
        
        pad_mask = (Xs == 0).all(dim=2)

        out_g, z_g = self.GraphEnc(Xg, edge_index, batch_g)     # Shape(num_nodes, 2H), Shape(num_graphs, 2H)
        out_s, h_fwd, h_bwd = self.EventSeqEnc(Xs, pad_mask)    # Shape(B, S, 2H), Shape(B, H), Shape(B, H)

        B, S, H = out_s.shape
        mapped_g = torch.zeros(B, S, H, device=out_g.device)    # mapping activity embeddings

        valid_mask = Act_pos >= 0
        valid_act_pos = Act_pos[valid_mask]
        
        batch_indices = torch.arange(B, device=Act_pos.device).unsqueeze(1).expand(-1, S)[valid_mask]
        seq_indices = torch.arange(S, device=Act_pos.device).unsqueeze(0).expand(B, -1)[valid_mask]  
        mapped_g[batch_indices, seq_indices] = out_g[valid_act_pos]
        
        enc_out = torch.cat([out_s, mapped_g], dim=2)  # Shape(batch_size, seq_len, H*4)
        enc_context = torch.cat([h_fwd, h_bwd, z_g], dim=1)  # Shape(batch_size, H*4) 

        mixed_out, s0 = self.Mixer(enc_out, enc_context, pad_mask)
        # Shape(Batch_size, Seq_len-1, Hidden), Shape(Batch_size, hidden)

        pad_mask = pad_mask[:, :-1]  # Remove <start> token mask

        embeddings = []
        for i in range(len(self.Decoder)):
            if i == 0:
                emb = self.tfEmbedder[0](Xa)[:, :-1, :]  # Shape(batch_size, seq_len-1, hidden_dim)
            else:
                emb = self.tfEmbedder[i](Xs[:, :, i-1])[:, :-1, :]  # Shape(batch_size, seq_len-1, hidden_dim)
            embeddings.append(emb)
        
        output = []
        for i, (dec, tfembs) in enumerate(zip(self.Decoder, embeddings)):
            out = dec(mixed_out, s0, tfembs, pad_mask)  # Shape(batch_size, seq_len - 1, input_dim)
            out = torch.cat([torch.zeros((out.size(0), 1, out.size(2)), device=out.device), out], dim=1)    # Shape(batch_size, seq_len, input_dim)
            output.append(out)  # Shape(batch_size, seq_len, input_dim)

        return output