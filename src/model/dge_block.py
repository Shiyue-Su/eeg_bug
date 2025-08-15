import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# 我们直接复用你工程里的 MLLA_new.channel_MLLA 作为时间处理器
from src.model.MLLA_new import channel_MLLA


def _symmetrize_and_zero_diag(A: torch.Tensor) -> torch.Tensor:
    """对称化并置零对角线。A: [B, C, C] 或 [C, C]"""
    if A.dim() == 2:
        A = (A + A.t()) * 0.5
        A = A - torch.diag(torch.diag(A))
        return A
    elif A.dim() == 3:
        A = (A + A.transpose(1, 2)) * 0.5
        A = A - torch.diag_embed(torch.diagonal(A, dim1=1, dim2=2))
        return A
    else:
        raise ValueError(f"A must be 2D or 3D, got {A.dim()}D")


def _norm_adj(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """对称归一化  A_hat = D^{-1/2} A D^{-1/2}，对每个 batch 独立处理。A: [B, C, C]"""
    d = torch.clamp(A.sum(-1), min=eps)              # [B, C]
    d_inv_sqrt = d.pow(-0.5)                         # [B, C]
    D_inv_sqrt = torch.diag_embed(d_inv_sqrt)        # [B, C, C]
    return D_inv_sqrt @ A @ D_inv_sqrt


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DGEBlock(nn.Module):
    """
    动态图编码块：
      - 第1次接收 [B,1,C,T]，利用 channel_MLLA 生成 [B,C,N,D]
      - 后续接收 [B,C,N,D] 直接进行图传播
      - A = A_static(可训练) + ΔA(QK^T)，并做对称化、对角置零与归一化
    """
    def __init__(
        self,
        n_channels: int,
        context_window: int,
        patch_size: int,
        patch_stride: int,
        hidden_dim: int,
        out_dim: int,
        depth: int,
        n_heads: int,
        dk: int,
        dropout: float = 0.1,
        use_shared_time_encoder: Optional[channel_MLLA] = None,
    ) -> None:
        super().__init__()
        self.C = n_channels
        self.D = out_dim
        self.dk = dk

        # 时间处理器：可共享（推荐）
        if use_shared_time_encoder is None:
            self.time_encoder = channel_MLLA(
                context_window=context_window,
                patch_size=patch_size,
                hidden_dim=hidden_dim,
                out_dim=out_dim,
                depth=depth,
                patch_stride=patch_stride,
                n_heads=n_heads
            )
        else:
            self.time_encoder = use_shared_time_encoder

        # 学习到的静态图 A_static（随机初始化，可训练）
        A0 = torch.empty(n_channels, n_channels)
        nn.init.xavier_uniform_(A0)
        A0 = _symmetrize_and_zero_diag(A0)
        self.A_static = nn.Parameter(A0)  # [C,C]

        # Q/K 投影 (基于时间特征汇聚后的 [B,C,D])
        self.q_proj = nn.Linear(out_dim, dk, bias=False)
        self.k_proj = nn.Linear(out_dim, dk, bias=False)

        # 图传播后的 FFN + 残差 + LN
        self.ffn = FeedForward(out_dim, dropout=dropout)
        self.ln1 = nn.LayerNorm(out_dim)
        self.ln2 = nn.LayerNorm(out_dim)

        # 可调温度（稳定早期训练）
        self.register_buffer("tau", torch.tensor(1.0))

    def _make_time_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        将输入规整为 [B,C,N,D] 的时间特征：
          - 若 x 是 [B,1,C,T]，跑 MLLA；
          - 若 x 是 [B,C,N,D]，直接返回。
        """
        if x.dim() == 4 and x.size(1) == 1:  # [B,1,C,T]
            H = self.time_encoder(x)          # [B,C,N,D]
            return H
        elif x.dim() == 4 and x.size(1) == self.C:
            # 假设是 [B,C,N,D]
            return x
        else:
            raise ValueError(f"Unsupported input shape for DGEBlock: {x.shape}")

    def _build_delta_A(self, H: torch.Tensor) -> torch.Tensor:
        """
        基于时间特征 H:[B,C,N,D] 生成 ΔA：
           1) 先在 N 上做平均 -> [B,C,D]
           2) Q=Linear(D→dk), K=Linear(D→dk)
           3) ΔA = tanh( (Q K^T) / sqrt(dk) / tau )
        """
        B, C, N, D = H.shape
        Hc = H.mean(dim=2)                  # [B,C,D]
        Q = self.q_proj(Hc)                 # [B,C,dk]
        K = self.k_proj(Hc)                 # [B,C,dk]
        scale = (self.dk ** 0.5) * (float(self.tau.item()) if isinstance(self.tau, torch.Tensor) else self.tau)
        att = torch.matmul(Q, K.transpose(1, 2)) / scale  # [B,C,C]
        dA = torch.tanh(att)
        return dA

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          x: [B,1,C,T] 或 [B,C,N,D]
        Returns:
          H_out: [B,C,N,D]    # 图传播 + FFN 后
          H_time: [B,C,N,D]   # 仅时间处理器输出（供上层可视化/复用）
          A: [B,C,C]          # 本层的最终图（静态+动态）
        """
        # 1) 时间特征
        H_time = self._make_time_features(x)  # [B,C,N,D]
        B, C, N, D = H_time.shape

        # 2) 构造图 A = A_static + ΔA
        dA = self._build_delta_A(H_time)      # [B,C,C]
        A = self.A_static.unsqueeze(0).expand(B, -1, -1) + dA  # [B,C,C]
        A = _symmetrize_and_zero_diag(A)
        A = _norm_adj(A)                      # 归一化，稳定训练

        # 3) 图传播：对每个 patch n 独立做 A @ H[:, :, n, :]
        Hn = H_time.permute(0, 2, 1, 3).reshape(B * N, C, D)  # [B*N, C, D]
        Ab = A.unsqueeze(1).expand(B, N, C, C).reshape(B * N, C, C)  # [B*N, C, C]
        Hprop = torch.bmm(Ab, Hn)                                # [B*N, C, D]
        Hprop = Hprop.reshape(B, N, C, D).permute(0, 2, 1, 3)    # [B,C,N,D]

        # 4) 残差 + FFN
        H_res1 = self.ln1(H_time + Hprop)
        H_ffn = self.ffn(H_res1)
        H_out = self.ln2(H_res1 + H_ffn)

        return H_out, H_time, A
