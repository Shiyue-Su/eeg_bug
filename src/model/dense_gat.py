import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseGAT(nn.Module):
    """
    基于稠密邻接 A 的图注意力（多头、点积式），适配批量：
      X: [B, C, D]，A: [B, C, C] -> out: [B, C, D]
    做法：Q/K/V 多头线性 -> masked softmax（用 A 做加性 mask + log 权重）-> 注意力加权 V -> 线性融合
    """
    def __init__(self, d_model: int, n_heads: int = 4, attn_dropout: float = 0.0, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=True)

        self.attn_drop = nn.Dropout(attn_dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, D], A: [B, C, C]（可为 0-1 或权重）
        """
        B, C, D = x.shape
        H, Dh = self.n_heads, self.d_head

        q = self.w_q(x).view(B, C, H, Dh).transpose(1, 2)   # [B, H, C, Dh]
        k = self.w_k(x).view(B, C, H, Dh).transpose(1, 2)   # [B, H, C, Dh]
        v = self.w_v(x).view(B, C, H, Dh).transpose(1, 2)   # [B, H, C, Dh]

        # 注意力分数： [B, H, C, C]
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)

        # 用 A 做 mask + log 权重
        #   - 未连接处置 -inf
        #   - 若 A 为权重，等价于在 logits 上加 log(A_clamped)
        mask = (A > 0).unsqueeze(1)  # [B,1,C,C]
        attn = attn.masked_fill(~mask, float("-inf"))

        A_clamped = torch.clamp(A, min=1e-6).unsqueeze(1)  # [B,1,C,C]
        attn = attn + torch.log(A_clamped)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)  # [B, H, C, Dh]
        out = out.transpose(1, 2).contiguous().view(B, C, D)  # [B, C, D]
        out = self.w_o(out)
        out = self.drop(out)
        return out
