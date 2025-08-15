import torch
import torch.nn as nn
from .dge_blocks import DGEBlock


class DGEEncoder(nn.Module):
    """
    堆叠多个 DGEBlock
    输入 H: [B, C, D]，输出 z: [B, C, D]，以及各层的 A（注意力图）列表
    """
    def __init__(
        self,
        d_model: int,
        depth: int = 2,
        n_heads: int = 4,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        ff_ratio: float = 4.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            DGEBlock(
                d_model=d_model,
                n_heads=n_heads,
                attn_dropout=attn_dropout,
                dropout=dropout,
                ff_ratio=ff_ratio,
            )
            for _ in range(depth)
        ])

    def forward(self, H: torch.Tensor):
        """
        H: [B, C, D]
        """
        device = H.device
        # 保险：如果整个 encoder 没被 Lightning 正确迁移，这里跟随输入设备
        try:
            p = next(self.parameters())
            if p.device != device:
                self.to(device)
        except StopIteration:
            pass

        A_list = []
        X = H
        for blk in self.blocks:
            X, A = blk(X)   # X: [B,C,D], A: [B,C,C]
            A_list.append(A)
        return X, A_list
