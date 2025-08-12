import math
import os
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from .scheduler import WarmupScheduler
from .layers import Network
from tqdm import tqdm
import numpy as np


class GAIN(nn.Module):

    def __init__(self, hidden_dim:int, num_enc_layers:int, num_dec_layers:int, 
                 enc_dropout:float, dec_dropout:float,
                 batch_size:int, epochs:int, lr:float, seed:int=None, alpha:float=0.5):
        super(GAIN, self).__init__()

        if seed is not None:
            torch.manual_seed(int(seed))

        self.hidden_dim = hidden_dim
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        self.enc_rate = enc_dropout
        self.dec_rate = dec_dropout

        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.alpha = alpha  # anomaly score 가중치: s = alpha*(1-p_true) + (1-alpha)*H_norm

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def fit(self, dataset):

        self.net = Network(attr_dims=dataset.attribute_dims,
                           hidden_dim=self.hidden_dim,
                           num_enc_layers=self.num_enc_layers,
                           num_dec_layers=self.num_dec_layers,
                           encoder_dropout=self.enc_rate,
                           decoder_dropout=self.dec_rate).to(self.device)

        loader = DataLoader(
            dataset=dataset.DataChunks, batch_size=self.batch_size,
            shuffle=True, follow_batch=['x','seq'],
            pin_memory=True, num_workers=max(0, os.cpu_count() // 4)
        )
        
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        
        total_steps = max(1, len(loader) * self.epochs)
        warmup_steps = max(3, int(0.05 * total_steps))  # 전체 스텝의 5% 정도 권장

        scheduler = WarmupScheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            max_lr=self.lr,
            min_lr=0.0
        )
        
        criterion = nn.CrossEntropyLoss(reduction='none', label_smoothing=0.1)

        for epoch in range(self.epochs):

            self.net.train()
            epoch_loss = 0.0
            
            for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):

                optimizer.zero_grad()
                batch = batch.to(self.device)

                Xg, Xs, Xa = batch.x, batch.seq, batch.act_origin
                edge_index = batch.edge_index
                Act_pos = batch.act_pos
                batch_g = batch.x_batch
                
                logits = self.net(
                    Xg=Xg, Xs=Xs, Xa=Xa,
                    edge_index=edge_index, Act_pos=Act_pos, batch_g=batch_g
                )

                TrueSeq = torch.cat([Xa.unsqueeze(2), Xs], dim=2)   # (B, S, num_attr)

                batch_loss = 0.0
                for idx in range(len(logits)):
                    true = TrueSeq[:, :, idx]               # (B, S)
                    pred = logits[idx].permute(0, 2, 1)     # (B, C, S)

                    mask = (true > 0)                       # pad=0 무시
                    loss_tok = criterion(pred, true)        # (B, S)
                    # 유효 토큰 평균(분모 0 방지)
                    denom = mask.float().sum(dim=1).clamp_min(1.0)
                    loss_seq = (loss_tok * mask.float()).sum(dim=1) / denom  # (B,)
                    batch_loss += loss_seq.mean()

                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
                optimizer.step()

                current_lr = scheduler.step()
                epoch_loss += batch_loss.item() / len(dataset.attribute_dims)
        
            print(f"Epoch {epoch+1} - Loss = {epoch_loss/len(loader):.4f}, LR = {current_lr:.6f}")

    @torch.no_grad()
    def detect(self, dataset):
        """
        반환:
          - trace_level_anomaly_score: (N,)
          - event_level_anomaly_score: (N, S)
          - attr_level_anomaly_score:  (N, S, num_attr)
        """
        loader = DataLoader(
            dataset=dataset.DataChunks, batch_size=self.batch_size,
            shuffle=False, follow_batch=['x','seq'],
            pin_memory=True, num_workers=max(0, os.cpu_count() // 4)
        )
        
        self.net.eval()
        eps = 1e-12
        attr_chunks = []

        for batch in loader:
            batch = batch.to(self.device)

            # (B, S, num_attr)
            TrueSeq = torch.cat([batch.act_origin.unsqueeze(2), batch.seq], dim=2)
            valid_mask = (TrueSeq[:, :, 0] != 0)     # 패딩이 아닌 위치 True  (B, S)
            valid_mask_f = valid_mask.float().unsqueeze(-1)  # (B, S, 1)

            logits = self.net(
                Xg=batch.x, Xs=batch.seq, Xa=batch.act_origin,
                edge_index=batch.edge_index, Act_pos=batch.act_pos, batch_g=batch.x_batch
            )  # 리스트: 각 attr마다 (B, S, C)

            current_batch_res = []
            # 각 속성별 anomaly score 계산
            for idx, logit in enumerate(logits):
                # 확률
                pred = logit.softmax(dim=-1)                                   # (B, S, C)
                true_idx = TrueSeq[:, :, idx].unsqueeze(-1)                    # (B, S, 1)

                # p_true
                p_true = pred.gather(dim=-1, index=true_idx).squeeze(-1)       # (B, S)
                p_true = torch.clamp(p_true, min=eps)

                # 전체 엔트로피 H = -Σ p log p, 정규화: log(C)
                C = pred.size(-1)
                H = -(pred * torch.log(pred + eps)).sum(dim=-1)                # (B, S)
                H_norm = H / math.log(max(2, C))                                # (B, S)

                # 최종 스코어: s = α*(1-p_true) + (1-α)*H_norm
                s = self.alpha * (1.0 - p_true) + (1.0 - self.alpha) * H_norm  # (B, S)
                s = s.unsqueeze(-1) * valid_mask_f                              # (B, S, 1)
                current_batch_res.append(s)

            # (B, S, num_attr)
            attr_chunks.append(torch.cat(current_batch_res, dim=-1))

        # (N, S, num_attr)
        attr_level_anomaly_score = torch.cat(attr_chunks, dim=0).cpu().numpy()
        # 이벤트/트레이스 집계(기본: max)
        event_level_anomaly_score = attr_level_anomaly_score.max(axis=2)       # (N, S)
        trace_level_anomaly_score = attr_level_anomaly_score.max(axis=(1, 2))  # (N,)

        return trace_level_anomaly_score, event_level_anomaly_score, attr_level_anomaly_score
