# -*- coding: utf-8 -*-
"""
Dynamic graph modules for spatio-temporal iterative encoder.

Author: you & ChatGPT
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynGraphBlock(nn.Module):
    """
    动态图构建 + 图传播（单步）
    - 从当前特征 x 估计功能连接 A（时变）：基于时间维标准化相关
    - Top-K 稀疏化 + 对称化 + 自环 + 归一化
    - 传播：x <- x + gamma * (Â @ x) （沿“通道/脑区”维传播）
    """
    def __init__(
        self,
        topk: int = 8,
        ema_alpha: float | None = 0.8,
        self_loop: bool = True,
        sym: bool = True,
        normalize: str = "row",      # "row" | "sym"
        detach_affinity: bool = True # True 更稳 & 省显存；False 可端到端
    ):
        super().__init__()
        self.topk = topk
        self.ema_alpha = ema_alpha
        self.self_loop = self_loop
        self.sym = sym
        self.normalize = normalize
        self.detach_affinity = detach_affinity

        # 残差门控参数（可学习）
        self.gamma = nn.Parameter(torch.tensor(0.1))

    @staticmethod
    def _corr_affinity(x: torch.Tensor) -> torch.Tensor:
        """
        x: [N, C, T]
        return A: [N, C, C]  (非负相关)
        """
        x = x - x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + 1e-6
        x = x / std
        A = torch.einsum("nct,nkt->nck", x, x) / x.size(-1)
        A = torch.relu(A)
        return A

    def _topk(self, A: torch.Tensor) -> torch.Tensor:
        N, C, _ = A.shape
        k = min(self.topk, C)
        vals, idx = torch.topk(A, k=k, dim=-1)
        mask = torch.zeros_like(A)
        mask.scatter_(-1, idx, 1.0)
        return A * mask

    def _normalize(self, A: torch.Tensor) -> torch.Tensor:
        if self.sym:
            A = 0.5 * (A + A.transpose(1, 2))
        if self.self_loop:
            I = torch.eye(A.size(1), device=A.device, dtype=A.dtype).unsqueeze(0).expand_as(A)
            A = A + I

        A = torch.clamp(A, min=0)

        if self.normalize == "sym":
            deg = A.sum(-1) + 1e-6
            d_inv_sqrt = torch.pow(deg, -0.5)
            D_inv_sqrt = torch.diag_embed(d_inv_sqrt)
            A = torch.bmm(torch.bmm(D_inv_sqrt, A), D_inv_sqrt)
        else:
            deg = A.sum(-1, keepdim=True) + 1e-6
            A = A / deg
        return A

    def forward(self, x: torch.Tensor, A_prev: torch.Tensor | None = None):
        # x: [N, C, T]
        x_feat = x.detach() if self.detach_affinity else x
        A = self._corr_affinity(x_feat)
        A = self._topk(A)
        A = self._normalize(A)

        if (A_prev is not None) and (self.ema_alpha is not None):
            A = self.ema_alpha * A_prev + (1.0 - self.ema_alpha) * A

        # 图传播（对每个时间步并行）
        z = torch.einsum("nij,njt->nit", A, x)  # [N, C, T]
        x = x + self.gamma * z
        return x, A


class IterativeSTEncoder(nn.Module):
    """
    N 步“时空混合”迭代编码器：每步：动态图估计 -> 归一化 -> 图传播 -> 残差
    """
    def __init__(self, n_steps: int = 1, **block_kwargs):
        super().__init__()
        self.n_steps = n_steps
        self.block = DynGraphBlock(**block_kwargs)

    def forward(self, x: torch.Tensor):
        # x: [N, C, T]
        A = None
        for _ in range(self.n_steps):
            x, A = self.block(x, A_prev=A)
        return x, A
