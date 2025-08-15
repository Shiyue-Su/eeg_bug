# src/model/DGE.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from src.model.MLLA_new import channel_MLLA, MLLA_BasicLayer  # 复用你已有的实现

class GraphBuilder(nn.Module):
    """
    动态图构建器：
      输入 H ∈ [B, C, T, D]
      h_summary = mean_T(H) ∈ [B, C, D]
      Q = h_summary @ WQ, K = h_summary @ WK  -> [B, C, d_k]
      sim = (Q @ K^T)/sqrt(d_k)               -> [B, C, C]
      ΔA = tanh(sim)
      A_k = sym(A_static) + ΔA                -> [B, C, C]，A_static 是可学习参数（静态图）
    """
    def __init__(self, n_channels: int, dim: int, d_k: int):
        super().__init__()
        self.WQ = nn.Linear(dim, d_k, bias=False)
        self.WK = nn.Linear(dim, d_k, bias=False)
        # 可学习静态图，用小值初始化；训练更稳
        self.A_static = nn.Parameter(0.01 * torch.randn(n_channels, n_channels))
        self.n_channels = n_channels
        self.d_k = d_k

    def forward(self, h_temp: torch.Tensor) -> torch.Tensor:
        # h_temp: [B, C, T, D]
        B, C, T, D = h_temp.shape
        h_summary = h_temp.mean(dim=2)           # [B, C, D]
        Q = self.WQ(h_summary)                   # [B, C, d_k]
        K = self.WK(h_summary)                   # [B, C, d_k]
        sim = torch.einsum('bcd,bkd->bck', Q, K) / math.sqrt(self.d_k)  # [B, C, C]
        delta_A = torch.tanh(sim)
        A_static = (self.A_static + self.A_static.t()) / 2.0            # 对称化
        A_k = delta_A + A_static.unsqueeze(0)                            # [B, C, C]
        return A_k


class SpatialGAT(nn.Module):
    """
    GAT 风格的空间传播（不依赖外部图库）：
      对每个时间步 t：
        Q_t = X_t Wq, K_t = X_t Wk, V_t = X_t Wv
        Scores = (Q_t K_t^T)/sqrt(d) + α * A_k
        Attn   = softmax(Scores)  (沿最后一维 C)
        Out_t  = Attn @ V_t
      残差 + LayerNorm
    输入 X: [B, C, T, D], 图 A_k: [B, C, C]
    """
    def __init__(self, dim: int, d_attn: int, alpha_adj: float = 1.0):
        super().__init__()
        self.Wq = nn.Linear(dim, d_attn, bias=False)
        self.Wk = nn.Linear(dim, d_attn, bias=False)
        self.Wv = nn.Linear(dim, dim,   bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.alpha = nn.Parameter(torch.tensor(alpha_adj, dtype=torch.float32))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, A_k: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, D], A_k: [B, C, C]
        B, C, T, D = x.shape
        x_btcd = x.permute(0, 2, 1, 3)  # [B, T, C, D]

        Q = self.Wq(x_btcd)             # [B, T, C, d_attn]
        K = self.Wk(x_btcd)             # [B, T, C, d_attn]
        V = self.Wv(x_btcd)             # [B, T, C, D]

        scores = torch.einsum('btcd,btkd->btck', Q, K) / math.sqrt(Q.shape[-1])  # [B,T,C,C]
        scores = scores + self.alpha * A_k.unsqueeze(1)                           # 广播到 T

        attn = F.softmax(scores, dim=-1)                                          # [B,T,C,C]
        out  = torch.einsum('btck,btkd->btcd', attn, V)                           # [B,T,C,D]
        out  = self.proj(out).permute(0, 2, 1, 3)                                 # [B,C,T,D]
        out  = self.norm(out + x)
        return out


class TemporalRefiner(nn.Module):
    """
    轻量级时序细化：沿 T 用你项目里的 MLLA_BasicLayer 再编码一次。
    输入/输出: [B, C, T, D]
    """
    def __init__(self, T_short: int, dim: int, depth: int = 1, n_heads: int = 4):
        super().__init__()
        self.encoder = MLLA_BasicLayer(
            q_len=T_short, in_dim=dim, hidden_dim=dim, out_dim=dim,
            depth=depth, num_heads=n_heads
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, D = x.shape
        x_seq = x.reshape(B * C, T, D)
        x_seq = self.encoder(x_seq)
        x = x_seq.view(B, C, T, D)
        return self.norm(x)


class DGEBlock(nn.Module):
    """
    一个时空混合块：
      (可选)TemporalRefiner -> 动态图构建 -> 空间传播
    """
    def __init__(self, n_channels: int, T_short: int, dim: int, d_k_graph: int = 64,
                 use_temporal_refiner: bool = True, temporal_depth: int = 1, n_heads: int = 4):
        super().__init__()
        self.graph_builder = GraphBuilder(n_channels, dim, d_k=d_k_graph)
        self.spatial = SpatialGAT(dim, d_attn=d_k_graph)
        self.temporal = TemporalRefiner(T_short, dim, depth=temporal_depth, n_heads=n_heads) \
                        if use_temporal_refiner else nn.Identity()
        # 供外部可视化：保留最近一次 forward 的 A_k
        self._last_A = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, D]
        x = self.temporal(x)            # 时序细化
        A_k = self.graph_builder(x)     # 动态图 [B,C,C]
        self._last_A = A_k.detach()     # 挂钩给可视化
        x = self.spatial(x, A_k)        # 空间传播
        return x

    @torch.no_grad()
    def get_last_A(self) -> Optional[torch.Tensor]:
        return self._last_A


class DGENet(nn.Module):
    """
    整体结构：
      1) 前端：channel_MLLA（来自 MLLA_new），把原始 EEG -> X0 ∈ [B, C, T_short, D]
      2) 堆叠若干 DGEBlock 做时空迭代融合
    """
    def __init__(
        self,
        n_channels: int,
        context_window: int,
        patch_size: int,
        patch_stride: int,
        hidden_dim: int,
        out_dim: int,
        depth_mlla: int,
        n_heads_mlla: int,
        num_blocks: int = 1,
        use_temporal_refiner: bool = True,
        temporal_depth: int = 1,
        n_heads_refiner: int = 4,
        d_k_graph: int = 64,
    ):
        super().__init__()
        # 1) 时序入口：复用 MLLA_new.channel_MLLA
        self.mlla = channel_MLLA(
            context_window=context_window,
            patch_size=patch_size,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            depth=depth_mlla,
            patch_stride=patch_stride,
            n_heads=n_heads_mlla,
        )
        # T_short 为 patch 数
        self.T_short = int((context_window - patch_size) / patch_stride + 1)
        self.n_channels = n_channels
        self.dim = out_dim

        # 2) 迭代式时空混合块
        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                DGEBlock(
                    n_channels=n_channels,
                    T_short=self.T_short,
                    dim=out_dim,
                    d_k_graph=d_k_graph,
                    use_temporal_refiner=use_temporal_refiner,
                    temporal_depth=temporal_depth,
                    n_heads=n_heads_refiner,
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        """
        x_raw: [B, 1, C, T_raw]
        return: X ∈ [B, C, T_short, D]
        """
        # MLLA_new 输出: [B, C, T_short, D]
        X = self.mlla(x_raw)
        assert X.dim() == 4 and X.shape[1] == self.n_channels and X.shape[3] == self.dim, \
            f"MLLA 输出维度异常: got {X.shape}, expect [B,{self.n_channels},T,{self.dim}]"
        for blk in self.blocks:
            X = blk(X)
        return X

    @torch.no_grad()
    def get_last_A(self) -> Optional[torch.Tensor]:
        """从最后一个 block 拿最近一次的 A_k（用于可视化）"""
        if len(self.blocks) == 0:
            return None
        return self.blocks[-1].get_last_A()
