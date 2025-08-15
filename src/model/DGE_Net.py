# src/model/DGE_Net.py
# -*- coding: utf-8 -*-
from typing import Optional
from omegaconf import DictConfig

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv
    from torch_geometric.utils import dense_to_sparse
except Exception as e:
    raise ImportError(
        f"""DGE_Net requires PyTorch Geometric.

Please install it manually by finding the correct command for your PyTorch and CUDA version.
Visit the official installation guide: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

A typical command looks like (EXAMPLE ONLY, CHECK YOUR VERSIONS):
pip install torch_geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.13.1+cu117.html

Original error: {e}
"""
    )

class SpatioTemporalBlock(nn.Module):
    def __init__(self, feature_dim: int, hybrid_cfg: DictConfig):
        super().__init__()
        self.feature_dim = feature_dim
        self.hybrid_cfg = hybrid_cfg
        self.topk = hybrid_cfg.topk
        self.gat_bchunk = hybrid_cfg.get("gat_bchunk", 4)

        temporal_enc_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=hybrid_cfg.get("time_heads", 4),
            dim_feedforward=feature_dim * 4,
            dropout=hybrid_cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.temporal_processor = nn.TransformerEncoder(temporal_enc_layer, num_layers=1)
        self.temporal_norm = nn.LayerNorm(feature_dim)

        dk = max(1, feature_dim // 2)
        self.q_proj = nn.Linear(feature_dim, dk)
        self.k_proj = nn.Linear(feature_dim, dk)
        self.scale = dk ** -0.5

        self.spatial_processor = GATConv(
            in_channels=feature_dim,
            out_channels=feature_dim,
            heads=hybrid_cfg.get("space_heads", 4),
            concat=False,
            dropout=hybrid_cfg.dropout,
            add_self_loops=False,
            bias=True,
        )
        self.spatial_norm = nn.LayerNorm(feature_dim)

    @torch.no_grad()
    def _build_topk_edges(self, A: torch.Tensor, k: int) -> torch.Tensor:
        C = A.size(0)
        k = int(max(1, min(k, C - 1)))
        idx = torch.topk(A, k=k, dim=-1).indices
        mask = A.new_zeros((C, C), dtype=torch.bool)
        mask.scatter_(1, idx, True)
        mask.fill_diagonal_(False)
        A_topk = torch.where(mask, A, A.new_zeros(()))
        edge_index, _ = dense_to_sparse(A_topk)
        return edge_index

    def forward(self, H_in: torch.Tensor, static_adj: torch.Tensor) -> torch.Tensor:
        B, C, Np, D = H_in.shape
        device = H_in.device

        H_reshaped = H_in.reshape(B * C, Np, D)
        H_temp_processed = self.temporal_processor(H_reshaped)
        H_temp = self.temporal_norm(H_reshaped + H_temp_processed).reshape(B, C, Np, D)

        H_summary = H_temp.mean(dim=2)
        q = self.q_proj(H_summary)
        k = self.k_proj(H_summary)
        sim = torch.einsum("bcd,bed->bce", q, k) * self.scale
        A_dyn = torch.tanh(sim)
        A = A_dyn + static_adj.unsqueeze(0)

        H_out = torch.empty_like(H_temp)
        for p in range(Np):
            X_p = H_temp[:, :, p, :]
            X_p_out = torch.empty_like(X_p)
            for b0 in range(0, B, self.gat_bchunk):
                b1 = min(B, b0 + self.gat_bchunk)
                Xb, Ab, bs = X_p[b0:b1], A[b0:b1], b1 - b0
                batch_out = [self.spatial_processor(Xb[i], self._build_topk_edges(Ab[i], self.topk).to(device)) for i in range(bs)]
                X_p_out[b0:b1] = torch.stack(batch_out, dim=0)
            H_out[:, :, p, :] = self.spatial_norm(X_p_out + X_p)

        return H_out

class DGE_Net(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg.model
        mlla_cfg = model_cfg.MLLA
        hybrid_cfg = model_cfg.hybrid

        self.C = model_cfg.in_channels
        self.feature_dim = mlla_cfg.out_dim

        self.initial_projection = nn.Linear(mlla_cfg.patch_size, self.feature_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, 2000, self.feature_dim))

        self.static_adj = nn.Parameter(torch.randn(self.C, self.C) * 0.02)

        self.blocks = nn.ModuleList()
        for _ in range(hybrid_cfg.num_blocks):
            self.blocks.append(SpatioTemporalBlock(self.feature_dim, hybrid_cfg))

        self.head = nn.Sequential(nn.Dropout(hybrid_cfg.dropout), nn.Linear(self.feature_dim, self.feature_dim))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear): nn.init.trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        Np = patches.size(2)
        
        H = self.initial_projection(patches)
        H = H + self.pos_embedding[:, :, :Np, :]

        for block in self.blocks:
            H = block(H, self.static_adj)

        Hc = H.mean(dim=1)
        Hc = self.head(Hc)
        return Hc.permute(0, 2, 1).unsqueeze(2), H

__all__ = ["DGE_Net"]