# -*- coding: utf-8 -*-
"""
CNN + Attention encoder with Dynamic Graph (Spatio-Temporal Iterative Encoder)

Author: you & ChatGPT
"""

from __future__ import annotations
import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- dynamic graph ----------------
try:
    from .graph.dyn_graph import IterativeSTEncoder
except Exception:
    # 支持从同目录导入（如果你把 dyn_graph.py 放在同目录）
    from graph.dyn_graph import IterativeSTEncoder


# ---------------- Attention modules ----------------
class SEAttention(nn.Module):
    """
    Channel SE 注意力：先在时间维做全局池化 -> [N, C, 1, 1] -> MLP -> [N,C,1,1]
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, 1, T] or [N, C, H, W]
        s = x.mean(dim=-1, keepdim=True)  # 全局池化时间维 -> [N,C,1,1]
        s = self.fc2(F.relu(self.fc1(s), inplace=True)).sigmoid()
        return x * s


class ConvSegmentAttention(nn.Module):
    """
    轻量卷积注意力：使用 depthwise-separable conv2d 在时间轴上聚合
    *为避免你之前的通道维 kernel 超界问题，本实现仅对时间轴做卷积（kh=1），不跨通道滑动*
    """
    def __init__(self, channels: int, seg_att_time: int = 3, pw_hidden: Optional[int] = None):
        super().__init__()
        seg_att_time = max(1, int(seg_att_time))
        self.dw = nn.Conv2d(
            channels, channels, kernel_size=(1, seg_att_time),
            padding=(0, seg_att_time // 2), groups=channels, bias=False
        )
        self.pw1 = nn.Conv2d(channels, pw_hidden or channels, kernel_size=1, bias=False)
        self.pw2 = nn.Conv2d(pw_hidden or channels, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        att = self.dw(x)
        att = F.relu(self.pw1(att), inplace=True)
        att = torch.sigmoid(self.pw2(att))
        return x * att


# ---------------- Backbone（时间建模） ----------------
class DepthwiseTemporalConv(nn.Module):
    """
    只在时间轴上进行 depthwise temporal conv，再做 pointwise 融合。
    更省显存，避免 (kh, kw) 中 kh>通道数的问题。
    输入/输出形状： [N, C, 1, T] -> [N, C, 1, T]
    """
    def __init__(self, channels: int, k_list=(7, 15, 31), pw_out: Optional[int] = None, dropout=0.0):
        super().__init__()
        self.branches = nn.ModuleList()
        for k in k_list:
            k = max(1, int(k))
            pad = k // 2
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=(1, k),
                              padding=(0, pad), groups=channels, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True)
                )
            )
        out_ch = channels * len(self.branches)
        self.pw = nn.Sequential(
            nn.Conv2d(out_ch, pw_out or channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(pw_out or channels),
            nn.ReLU(inplace=True)
        )
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [b(x) for b in self.branches]
        x = torch.cat(feats, dim=1)
        x = self.pw(x)
        x = self.drop(x)
        return x


# ---------------- 主干：cnn_MLLA ----------------
class cnn_MLLA(nn.Module):
    """
    兼容你现有 MultiModel_PL 的 Encoder：
      - 输入 x: [N, C, 1, T] 或 [N, C, T] 或 [N, 1, C, T]
      - 输出 out: [N, D, 1, T]（D == C 或配置的 pw_out）
    支持：
      - use_dyn_graph: True/False
      - att_type: "se" (默认) 或 "conv"
    """
    def __init__(self, cfg=None, in_channels: Optional[int] = None, time_length: Optional[int] = None):
        super().__init__()
        self.cfg = cfg

        # ------- 从 cfg 读取必要参数（有兜底） -------
        mcfg = getattr(cfg, "model", None)
        mlla = getattr(mcfg, "MLLA", None) if mcfg is not None else None

        self.att_type = "se"
        self.seg_att = 3                  # 这里只用于 conv 注意力的时间核宽
        self.reduction = 8
        self.dropout = 0.1
        self.pw_out = None
        self.k_list = (7, 15, 31)

        if mlla is not None:
            self.att_type = str(getattr(mlla, "att_type", self.att_type)).lower()
            self.seg_att = int(getattr(mlla, "seg_att", self.seg_att))
            self.reduction = int(getattr(mlla, "se_reduction", self.reduction))
            self.dropout = float(getattr(mlla, "dropout", self.dropout))
            self.pw_out = getattr(mlla, "pw_out", self.pw_out)
            if hasattr(mlla, "k_list"):
                self.k_list = tuple(int(k) for k in mlla.k_list)

        # 动态图设置
        self.use_dyn_graph = bool(getattr(mlla, "use_dyn_graph", False)) if mlla is not None else False
        gcfg = getattr(mlla, "graph", {}) if mlla is not None else {}
        self.st_iter_encoder = None
        if self.use_dyn_graph:
            self.st_iter_encoder = IterativeSTEncoder(
                n_steps=int(gcfg.get("layers", 1)),
                topk=int(gcfg.get("topk", 8)),
                ema_alpha=float(gcfg.get("ema_alpha", 0.8)),
                self_loop=bool(gcfg.get("self_loop", True)),
                sym=bool(gcfg.get("sym", True)),
                normalize=str(gcfg.get("normalize", "row")),
                detach_affinity=bool(gcfg.get("detach_affinity", True)),
            )

        # 我们不在 __init__ 里强行依赖 in_channels，避免 Lazy 参数问题。
        # backbone/attention 用 Lazy 模块做通道对齐。
        self.backbone = None
        self.att = None
        self._lazy_infer_built = False

    # ---------- 工具：把各种形状统一成 [N,C,1,T] ----------
    @staticmethod
    def _to_NC1T(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:        # [N, C, T]
            x = x.unsqueeze(2)  # -> [N, C, 1, T]
        elif x.dim() == 4:
            # 允许 [N, 1, C, T]
            if x.size(1) == 1 and x.size(2) > 1:
                x = x.transpose(1, 2)  # [N,C,1,T]
        else:
            raise RuntimeError(f"cnn_MLLA expects 3D/4D, got {x.shape}")
        return x

    def _build_lazy(self, x: torch.Tensor):
        """
        基于真实 batch 动态构建/初始化 backbone 与 attention，避免 UninitializedParameter 报错。
        """
        if self._lazy_infer_built:
            return
        x = self._to_NC1T(x)
        N, C, _, _ = x.shape

        # Backbone：depthwise temporal multi-branch
        self.backbone = DepthwiseTemporalConv(
            channels=C, k_list=self.k_list, pw_out=self.pw_out, dropout=self.dropout
        ).to(x.device)

        # Attention
        if self.att_type == "se":
            self.att = SEAttention(channels=self.pw_out or C, reduction=self.reduction).to(x.device)
        elif self.att_type == "conv":
            # 卷积注意力只在时间维卷积（更稳更省），seg_att 作为时间核宽
            self.att = ConvSegmentAttention(channels=self.pw_out or C, seg_att_time=max(1, self.seg_att)).to(x.device)
        else:
            warnings.warn(f"[Warn] Unknown att_type '{self.att_type}', fallback to 'se'.")
            self.att_type = "se"
            self.att = SEAttention(channels=self.pw_out or C, reduction=self.reduction).to(x.device)

        self._lazy_infer_built = True

    # ----------------- forward -----------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        返回特征张量，保持形状 [N, D, 1, T]，以兼容你现有下游。
        """
        x = self._to_NC1T(x)
        self._build_lazy(x)

        # 1) 时间卷积主干
        out = self.backbone(x)            # [N, D, 1, T] （D= C 或 pw_out）

        # 2) 动态图时空迭代编码（注意这里需要 [N, D, T]）
        if self.st_iter_encoder is not None:
            dyn, _A = self.st_iter_encoder(out.squeeze(2))  # [N,D,T]
            out = dyn.unsqueeze(2)

        # 3) 注意力
        out = self.att(out)               # [N, D, 1, T]

        return out
