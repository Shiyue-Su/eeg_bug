import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DGEBlock(nn.Module):
    """
    Dynamic Graph Encoder Block
    对通道维（节点）做多头自注意力；输入 X: [B, C, D]
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        ff_ratio: float = 4.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # QKV & 输出投影
        self.qkv = nn.Linear(d_model, d_model * 3, bias=True)
        self.proj = nn.Linear(d_model, d_model, bias=True)

        # 归一化和 MLP
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * ff_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(d_model * ff_ratio), d_model),
            nn.Dropout(dropout),
        )

        self.attn_drop = nn.Dropout(attn_dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_head)

    def forward(self, X: torch.Tensor):
        """
        X: [B, C, D]
        return: X_out: [B, C, D], A: [B, C, C]（多头平均后的注意力图）
        """
        B, C, D = X.shape
        device, dtype = X.device, X.dtype

        # 保险：如果模块权重不在当前设备，把整个模块搬过来（DDP 下也安全）
        if self.qkv.weight.device != device:
            self.to(device)

        # 层归一 + QKV
        X_norm = self.norm1(X)
        qkv = self.qkv(X_norm)  # [B, C, 3D]
        q, k, v = torch.chunk(qkv, 3, dim=-1)  # 各 [B, C, D]

        # 拆多头 -> [B, C, H, d_head] -> [B, H, C, d_head]
        def split_heads(t):
            return t.view(B, C, self.n_heads, self.d_head).permute(0, 2, 1, 3)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        # 注意力得分 [B, H, C, C]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn_scores, dim=-1)
        attn = self.attn_drop(attn)

        # 上下文
        ctx = torch.matmul(attn, v)  # [B, H, C, d_head]
        # 合并多头 -> [B, C, D]
        ctx = ctx.permute(0, 2, 1, 3).contiguous().view(B, C, D)

        # 输出投影 + 残差
        X = X + self.resid_drop(self.proj(ctx))

        # FFN
        X = X + self.mlp(self.norm2(X))

        # 返回多头平均的 A 方便调试/可视化
        A = attn.mean(dim=1)  # [B, C, C]
        return X, A
