# src/model/spatiotemporal_block.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SpatioTemporalBlock(nn.Module):
    """
    时空混合块（静态图关闭版）
    Input:  H: [B, C, Np, D]
    Output: H_new: [B, C, Np, D]

    逻辑：
      1) 时间摘要：Hc = mean_t(H) -> [B,C,D]
      2) 生成 Q/K，相似度 -> ΔA (softmax+topk)，不使用静态图（Ak = ΔA）
      3) 图传播：对每个时间片 t，H[:,:,t,:] <- Ak @ H[:,:,t,:]
      4) 残差 + FFN
    """
    def __init__(
        self,
        d_model: int,
        n_heads_gat: int = 4,            # 预留，不在本实现中显式使用（做简单的线性图传播）
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        save_adj: bool = False,
        use_A_static: bool = False,      # 置 False：忽略静态图
        topk: Optional[int] = 8,
        tau: float = 0.2,
        ema_delta: float = 0.0           # 0 代表关闭 EMA
    ):
        super().__init__()
        self.d_model = d_model
        self.attn_dropout = attn_dropout
        self.dropout_p = dropout
        self.save_adj = save_adj

        # 关闭静态图
        self.use_A_static = False if use_A_static is None else use_A_static
        self.use_A_static = False  # 强制关闭

        self.topk = topk
        self.tau = tau
        self.ema_delta = ema_delta

        # Q/K 生成
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

        self.drop = nn.Dropout(dropout)

        # runtime cache for visualization
        self.last_Ak: Optional[torch.Tensor] = None
        self.register_buffer("deltaA_ema", None, persistent=False)

    @torch.no_grad()
    def _sparsify_topk(self, A: torch.Tensor, k: int) -> torch.Tensor:
        """对 A 做 TopK 稀疏化，按最后一维保留 TopK，再按行归一化。"""
        if k is None or k <= 0 or k >= A.size(-1):
            return A
        v, idx = torch.topk(A, k=k, dim=-1)
        mask = torch.zeros_like(A).scatter_(-1, idx, 1.0)
        A = A * mask
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)
        return A

    def forward(self, H: torch.Tensor, A_static: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        H: [B, C, Np, D]
        A_static: [B, C, C] or None（此版本被忽略）
        """
        assert H.dim() == 4, f"H must be [B,C,Np,D], got {tuple(H.shape)}"
        B, C, Np, D = H.shape
        device = H.device

        # 1) 时间摘要
        Hc = H.mean(dim=2)  # [B, C, D]

        # 2) ΔA
        Q = self.q_proj(Hc)  # [B, C, D]
        K = self.k_proj(Hc)  # [B, C, D]
        S = torch.matmul(Q, K.transpose(-2, -1)) / (D ** 0.5)  # [B, C, C]
        A_delta = torch.softmax(S / self.tau, dim=-1)          # [B, C, C]
        if self.attn_dropout > 0:
            A_delta = F.dropout(A_delta, p=self.attn_dropout, training=self.training)

        if self.topk is not None and self.topk > 0:
            A_delta = self._sparsify_topk(A_delta, self.topk)

        # EMA（可选）
        if self.ema_delta > 0 and self.training:
            if self.deltaA_ema is None or self.deltaA_ema.shape != A_delta.shape:
                self.deltaA_ema = A_delta.detach().clone()
            else:
                self.deltaA_ema = self.ema_delta * self.deltaA_ema + (1 - self.ema_delta) * A_delta.detach()
            Ak = self.deltaA_ema
        else:
            Ak = A_delta

        # === 关闭静态图 ===
        # if self.use_A_static and A_static is not None:
        #     Ak = Ak + A_static
        # else:
        #     Ak = Ak
        # 这里直接使用 Ak = ΔA
        # =================

        if self.save_adj:
            self.last_Ak = Ak.detach().cpu()
        else:
            self.last_Ak = None

        # 3) 图传播：对每个时间片 t，Ak @ H[:,:,t,:]
        # H_t: [B, Np, C, D]
        H_t = H.permute(0, 2, 1, 3).contiguous()
        # out_t[b, n] = Ak[b] @ H_t[b, n]
        out_t = torch.einsum('bij,bnjd->bnid', Ak, H_t)  # [B, Np, C, D]
        out = out_t.permute(0, 2, 1, 3).contiguous()     # [B, C, Np, D]

        # 4) 残差 + FFN
        H1 = H + self.drop(out)
        H2 = H1 + self.ffn(H1)
        return H2
