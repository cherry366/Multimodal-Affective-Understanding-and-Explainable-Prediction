# -*- coding: utf-8 -*-
"""模块二 模型（MoDE-Former+ v2）

符号（见方案）
  X^(m) ∈ R^{B×50×d_m}   标准化且缺失处置 0
  M^(m) ∈ {0,1}^{B×50}   1=有效
  H^(m) = Transformer(W_m X^(m) + PE),  key_padding_mask = ¬M^(m)
  Z^(m←m') = MHA(Q=H^(m), K=V=H^(m'), mask=¬M^(m'))
  α = softmax( g([h̄; c]) + β ⊙ log(c + ε) )          # v2 覆盖率先验
  z = Σ_m α_m (W_o h̄^(m) + mean_{m'≠m} W_1 z̄^(m←m') + Σ_k β_k E_k(h̄^(m)))

v2 相对 v1 的改动：
  1) 删除模态类型嵌入 e^(m)（消融证明无贡献）
  2) 门控增加 β ⊙ log(c + ε) 覆盖率先验（use_cov_prior）
  3) MoE 路由器输入增加缺失率/缺失段数（use_moe_feats）
  4) 融合层 dropout（fusion_dropout）
  5) forward 输出 z（供 InfoNCE 对比式不变性损失使用）
"""
import math
import torch
import torch.nn as nn

import config_q2 as C


def sinusoidal_pe(L, d):
    pe = torch.zeros(L, d)
    pos = torch.arange(0, L, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)          # (1,L,d)


def masked_mean(H, M, eps=1e-8):
    w = M.unsqueeze(-1)
    return (H * w).sum(1) / (w.sum(1) + eps)


def _mask_feats(M, eps=1e-8):
    """(B,L) -> (B,2)：有效区 [1, leff) 内的缺失率与缺失段数。"""
    B, L = M.shape
    dev = M.device
    pos = torch.arange(L, device=dev).float().unsqueeze(0)      # (1,L)
    last = torch.where(M > 0.5, pos, torch.full_like(pos, -1.0)).max(1).values
    leff = torch.clamp(last + 1.0, min=2.0)                     # (B,)
    in_span = (pos >= 1.0) & (pos <= (leff - 1.0).unsqueeze(1))
    miss = in_span & (M <= 0.5)
    span_len = in_span.float().sum(1)
    miss_rate = miss.float().sum(1) / (span_len + eps)
    prev = torch.cat([torch.ones(B, 1, device=dev), miss.float()[:, :-1]], dim=1)
    n_seg = (miss & (prev <= 0.5)).float().sum(1)
    return torch.stack([miss_rate, n_seg], dim=1)


class Q2Net(nn.Module):
    def __init__(self, dims=None, hid=C.HID, nhead=C.NHEAD, nlayer=C.NLAYER,
                 ffn=C.FFN, dropout=C.DROPOUT, use_key_mask=True, use_gate=True,
                 use_cross=True, use_recon=False, n_cls=3,
                 use_mod_emb=False, use_moe=False, use_nll=False,
                 n_experts=C.N_EXPERTS, use_cov_prior=False, cov_prior_init=1.0,
                 use_moe_feats=False, fusion_dropout=0.0):
        super().__init__()
        dims = dims or C.DIM
        self.mods = list(C.MODS)
        self.use_key_mask = use_key_mask
        self.use_gate = use_gate
        self.use_cross = use_cross
        self.use_recon = use_recon
        self.use_mod_emb = use_mod_emb
        self.use_moe = use_moe
        self.use_nll = use_nll
        self.use_cov_prior = use_cov_prior
        self.use_moe_feats = use_moe_feats

        self.proj = nn.ModuleDict({m: nn.Linear(dims[m], hid) for m in self.mods})
        self.register_buffer("pe", sinusoidal_pe(C.L, hid))
        if use_mod_emb:
            self.mod_emb = nn.ParameterDict(
                {m: nn.Parameter(torch.zeros(hid)) for m in self.mods})
        enc_layer = nn.TransformerEncoderLayer(d_model=hid, nhead=nhead,
                                               dim_feedforward=ffn, dropout=dropout,
                                               batch_first=True, norm_first=True,
                                               activation="gelu")
        self.enc = nn.ModuleDict({m: nn.TransformerEncoder(enc_layer, num_layers=nlayer)
                                  for m in self.mods})
        if use_cross:
            self.cross = nn.ModuleDict({
                f"{m}<-{mp}": nn.MultiheadAttention(hid, nhead, dropout=dropout, batch_first=True)
                for m in self.mods for mp in self.mods if m != mp})
            self.w1 = nn.Linear(hid, hid)
        if use_moe:
            rdim = hid + 3 if use_moe_feats else hid + 1
            self.experts = nn.ModuleDict({
                m: nn.ModuleList([nn.Sequential(nn.Linear(hid, hid), nn.GELU())
                                  for _ in range(n_experts)])
                for m in self.mods})
            self.router = nn.ModuleDict({
                m: nn.Linear(rdim, n_experts) for m in self.mods})
        if use_cov_prior:
            self.cov_beta = nn.Parameter(torch.full((3,), float(cov_prior_init)))
        self.wo = nn.ModuleDict({m: nn.Linear(hid, hid) for m in self.mods})
        self.gate = nn.Sequential(nn.Linear(3 * hid + 3, hid), nn.GELU(),
                                  nn.Linear(hid, 3))
        self.norm = nn.LayerNorm(hid)
        self.fusion_drop = nn.Dropout(fusion_dropout) if fusion_dropout > 0 else nn.Identity()
        self.head_cls = nn.Linear(hid, n_cls)
        self.head_reg = nn.Linear(hid, 1)
        if use_nll:
            self.head_var = nn.Linear(hid, 1)
        if use_recon:
            self.rec_enc = nn.ModuleDict({m: nn.Linear(hid, hid) for m in self.mods})
            self.rec_dec = nn.ModuleDict({m: nn.Linear(hid, dims[m]) for m in self.mods})

    def forward(self, X, M):
        Msafe = {}
        for m in self.mods:
            Mx = M[m].clone()
            empty = Mx.sum(1) <= 0
            if bool(empty.any()):
                Mx[empty.nonzero(as_tuple=True)[0], 0] = 1.0
            Msafe[m] = Mx

        hs, pool = {}, {}
        for m in self.mods:
            z = self.proj[m](X[m]) + self.pe
            if self.use_mod_emb:
                z = z + self.mod_emb[m]
            z = torch.where(Msafe[m].unsqueeze(-1).bool(), z, torch.zeros_like(z))
            kpm = (~Msafe[m].bool()) if self.use_key_mask else None
            h = self.enc[m](z, src_key_padding_mask=kpm)
            h = torch.where(Msafe[m].unsqueeze(-1).bool(), h, torch.zeros_like(h))
            hs[m] = h
            pool[m] = masked_mean(h, Msafe[m])

        enriched = {}
        betas = {}
        for m in self.mods:
            acc, cnt = 0.0, 0
            if self.use_cross:
                for mp in self.mods:
                    if mp == m:
                        continue
                    kpm = (~Msafe[mp].bool()) if self.use_key_mask else None
                    zc, _ = self.cross[f"{m}<-{mp}"](hs[m], hs[mp], hs[mp], key_padding_mask=kpm)
                    zc = torch.where(Msafe[m].unsqueeze(-1).bool(), zc, torch.zeros_like(zc))
                    acc = acc + self.w1(masked_mean(zc, Msafe[m]))
                    cnt += 1
            cross_contrib = acc / cnt if cnt else 0.0
            moe_contrib = 0.0
            if self.use_moe:
                cov_m = M[m].mean(1, keepdim=True)
                if self.use_moe_feats:
                    rin = torch.cat([pool[m], cov_m, _mask_feats(M[m])], dim=1)
                else:
                    rin = torch.cat([pool[m], cov_m], dim=1)
                beta = torch.softmax(self.router[m](rin), dim=1)
                betas[m] = beta
                for k, expert in enumerate(self.experts[m]):
                    moe_contrib = moe_contrib + beta[:, k:k + 1] * expert(pool[m])
            enriched[m] = self.wo[m](pool[m]) + cross_contrib + moe_contrib

        if self.use_gate:
            cov = torch.stack([M[m].mean(1) for m in self.mods], dim=1)
            feat = torch.cat([pool[m] for m in self.mods] + [cov], dim=1)
            g = self.gate(feat)
            if self.use_cov_prior:
                # 有效区内缺失率 -> 有效覆盖率（clean=1，真实缺失才会 <1）
                cov_eff = torch.stack(
                    [1.0 - _mask_feats(M[m])[:, 0] for m in self.mods], dim=1)
                g = g + self.cov_beta * torch.log(cov_eff + 1e-6)
            alpha = torch.softmax(g, dim=1)
        else:
            B = M[self.mods[0]].shape[0]
            alpha = torch.full((B, 3), 1.0 / 3, device=M[self.mods[0]].device)

        z = 0.0
        for i, m in enumerate(self.mods):
            z = z + alpha[:, i:i + 1] * enriched[m]
        z = self.norm(z)
        zh = self.fusion_drop(z)

        out = {
            "logits": self.head_cls(zh),
            "reg": 3.0 * torch.tanh(self.head_reg(zh)).squeeze(-1),
            "alpha": alpha,
            "z": z,
        }
        if self.use_nll:
            out["log_var"] = self.head_var(zh).squeeze(-1)
        if self.use_moe:
            out["beta"] = betas
        if self.use_recon:
            rec = {}
            for m in self.mods:
                s, cnt = 0.0, 0
                for mp in self.mods:
                    if mp == m:
                        continue
                    s = s + self.rec_enc[mp](hs[mp])
                    cnt += 1
                rec[m] = self.rec_dec[m](s / cnt)
            out["recon"] = rec
        return out
