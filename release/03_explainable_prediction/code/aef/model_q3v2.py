# -*- coding: utf-8 -*-
"""模块三 新方案（v2）模型：原生证据加权融合 Q3V2Net。

在 Q2 骨干（final_v2）之上做一处机制替换：
  均值池化 masked_mean(h)  ->  可学习证据加权池化 sum_t ebar_m(t) * h_m(t)
其中 ebar_m = softmax(EvScore_m(h))（对有效位置），e_m(t) 即"原生局部证据"。
因此关键证据定位在训练中直接参与决策（前验/可解释），而非 v1 的推理期后验遮挡。
"""
import torch
import torch.nn as nn

from model_q2 import Q2Net, _mask_feats
import config_q2 as C


class Q3V2Net(Q2Net):
    def __init__(self):
        super().__init__(use_key_mask=True, use_gate=True, use_cross=True, use_recon=False,
                         use_mod_emb=False, use_moe=True, use_nll=True, use_cov_prior=True,
                         use_moe_feats=True, fusion_dropout=0.0)
        # 每模态一个证据打分器（token 级）
        self.ev = nn.ModuleDict({m: nn.Linear(C.HID, 1) for m in self.mods})

    def evidence_pool(self, h, Msafe, m):
        logit = self.ev[m](h)                          # (B,50,1)
        masked_logit = logit.masked_fill(~Msafe[m].unsqueeze(-1).bool(), float("-inf"))
        ebar = torch.softmax(masked_logit, dim=1)      # 归一化证据权重
        e = torch.sigmoid(logit) * Msafe[m].unsqueeze(-1)   # 原始证据(0~1，用于可视化)
        pool = (h * ebar).sum(1)                       # (B,hid) 证据加权池化
        return pool, ebar.squeeze(-1), e.squeeze(-1)

    def forward(self, X, M):
        Msafe = {}
        for m in self.mods:
            Mx = M[m].clone()
            empty = Mx.sum(1) <= 0
            if bool(empty.any()):
                Mx[empty.nonzero(as_tuple=True)[0], 0] = 1.0
            Msafe[m] = Mx

        hs, pool, ev_bar, ev_raw = {}, {}, {}, {}
        for m in self.mods:
            z = self.proj[m](X[m]) + self.pe
            z = torch.where(Msafe[m].unsqueeze(-1).bool(), z, torch.zeros_like(z))
            kpm = (~Msafe[m].bool()) if self.use_key_mask else None
            h = self.enc[m](z, src_key_padding_mask=kpm)
            h = torch.where(Msafe[m].unsqueeze(-1).bool(), h, torch.zeros_like(h))
            hs[m] = h
            pool[m], ev_bar[m], ev_raw[m] = self.evidence_pool(h, Msafe, m)

        enriched, betas = {}, {}
        for m in self.mods:
            acc, cnt = 0.0, 0
            if self.use_cross:
                for mp in self.mods:
                    if mp == m:
                        continue
                    kpm = (~Msafe[mp].bool()) if self.use_key_mask else None
                    zc, _ = self.cross[f"{m}<-{mp}"](hs[m], hs[mp], hs[mp], key_padding_mask=kpm)
                    zc = torch.where(Msafe[m].unsqueeze(-1).bool(), zc, torch.zeros_like(zc))
                    acc = acc + self.w1((zc * ev_bar[m].unsqueeze(-1)).sum(1))
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
                cov_eff = torch.stack([1.0 - _mask_feats(M[m])[:, 0] for m in self.mods], dim=1)
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
            "evidence": {m: ev_raw[m] for m in self.mods},
            "evidence_bar": {m: ev_bar[m] for m in self.mods},
        }
        if self.use_nll:
            out["log_var"] = self.head_var(zh).squeeze(-1)
        if self.use_moe:
            out["beta"] = betas
        return out
