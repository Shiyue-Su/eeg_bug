# 文件路径: src/model/adaptive_encoder.py (MLLA融合版)

import torch
import torch.nn as nn
from src.model.hybrid_block import HybridBlock 
from src.model.PatchTST_layers import positional_encoding

class AdaptiveEncoder(nn.Module):
    """
    顶层模型：迭代式的自适应编码器。
    这个版本使用 MLLAEncoderLayer 作为时间处理器。
    """
    def __init__(self, num_channels, patch_len, patch_stride, max_T_short,
                 num_blocks=4, feature_dim=128, gnn_heads=8, dropout=0.1, top_k=8):
        super().__init__()
        self.feature_dim = feature_dim
        self.patch_len = patch_len
        self.patch_stride = patch_stride

        self.patch_embedding = nn.Linear(patch_len, feature_dim)
        self.pos_encoder = positional_encoding(pe='sincos', learn_pe=True, q_len=max_T_short, d_model=feature_dim)
        
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            # --- 核心修改：将 max_T_short 传递给 HybridBlock ---
            self.blocks.append(
                HybridBlock(
                    num_channels=num_channels,
                    feature_dim=feature_dim,
                    gnn_heads=gnn_heads,
                    dropout=dropout,
                    top_k=top_k,
                    max_T_short=max_T_short # <--- 传递此参数
                )
            )

    def _patchify(self, x):
        return x.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)

    def forward(self, x_patched):
        T_short = x_patched.shape[2]
        x_embedded = self.patch_embedding(x_patched)
        pos_code = self.pos_encoder[:T_short].unsqueeze(0).unsqueeze(0)
        x_0 = x_embedded + pos_code

        delta_A_list = []
        x_k = x_0
        for block in self.blocks:
            x_k, delta_A = block(x_k) 
            delta_A_list.append(delta_A)
        
        x_N = x_k
        
        return x_0, x_N, delta_A_list