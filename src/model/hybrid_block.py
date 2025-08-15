# 文件路径: src/model/hybrid_block.py (MLLA + DGL 融合版)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data, Batch
import torch_geometric
from typing import Optional
from torch import Tensor

# =================================================================== #
#  Part 1: 从 MLLA_new.py 移植过来的辅助模块和核心MLLA层
# =================================================================== #

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

def get_activation_fn(activation):
    if callable(activation): return activation()
    elif activation.lower() == "relu": return nn.ReLU()
    elif activation.lower() == "gelu": return nn.GELU()
    raise ValueError(f'{activation} is not available. You can use "relu", "gelu", or a callable') 

class _LinearAttention(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        bs, n_head, L, dim = q.shape
        eps = 1e-5
        k = k.permute(0, 1, 3, 2)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        phi_Q = F.elu(q) + 1
        phi_K = F.elu(k) + 1
        KV = torch.einsum('blhd,blhm->bhdm', phi_K, v)
        K_sum = phi_K.sum(dim=1, keepdim=True)
        Z = 1.0 / (torch.einsum('blhd,bkhd->blh', phi_Q, K_sum) + eps)
        V_new = torch.einsum('blhd,bhdm->blhm', phi_Q, KV) * Z.unsqueeze(-1)
        return V_new.permute(0, 2, 1, 3), None

class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)
        self.res_attention = res_attention
        self.lnr_attn = _LinearAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)
        if self.res_attention:
            output, attn_weights, attn_scores = self.lnr_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.lnr_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)
        output = self.to_out(output)
        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights

class MLLAEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                nn.GELU(),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))
        self.cpe1 = nn.Conv1d(d_model, d_model, 3, padding=1, groups=d_model)
        self.cpe2 = nn.Conv1d(d_model, d_model, 3, padding=1, groups=d_model)
        self.dwc = nn.Conv1d(d_model, d_model, 3, padding=1, groups=d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.act_proj = nn.Linear(d_model, d_model)
        self.in_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.act = nn.SiLU()

    def forward(self, src:Tensor):
        src = src + self.cpe1(src.permute(0, 2, 1)).permute(0, 2, 1)
        shortcut = src
        src = self.norm1(src)
        act_res = self.act(self.act_proj(src))
        src = self.in_proj(src)
        src = self.act(self.dwc(src.permute(0, 2, 1))).permute(0, 2, 1)
        src2, attn = self.self_attn(src, src, src)
        src2 = self.out_proj(src2 * act_res)
        src2 = shortcut + src2
        src2 = src2 + self.cpe2(src2.permute(0, 2, 1)).permute(0, 2, 1)
        src2 = src2 + self.ff(src2)
        if self.res_attention:
            return src2, attn # Placeholder for potential future use
        else:
            return src2

# =================================================================== #
#  Part 2: 您的 SpatialBlock 保持不变 (Top-K版本)
# =================================================================== #

class SpatialBlock(nn.Module):
    def __init__(self, num_channels, feature_dim, gnn_heads=8, key_dim=32, top_k=8):
        super().__init__()
        self.num_channels = num_channels
        self.key_dim = key_dim
        self.static_adj = nn.Parameter(torch.randn(num_channels, num_channels))
        self.top_k = top_k 
        self.W_q = nn.Linear(feature_dim, key_dim, bias=False)
        self.W_k = nn.Linear(feature_dim, key_dim, bias=False)
        self.gnn = GATConv(feature_dim, feature_dim, heads=gnn_heads, concat=False)

    def forward(self, x):
        B, C, T_short, D = x.shape
        h_summary = torch.mean(x, dim=2)
        Q = self.W_q(h_summary)
        K = self.W_k(h_summary)
        Sim_k = torch.bmm(Q, K.transpose(-2, -1)) / (self.key_dim ** 0.5)
        delta_A = torch.tanh(Sim_k)
        adaptive_A = self.static_adj.unsqueeze(0) + delta_A
        top_k_values, top_k_indices = torch.topk(adaptive_A.abs(), k=self.top_k, dim=-1)
        sparse_A = torch.zeros_like(adaptive_A)
        sparse_A.scatter_(dim=-1, index=top_k_indices, src=adaptive_A.gather(dim=-1, index=top_k_indices))
        node_features = x.permute(0, 2, 1, 3).contiguous().view(B * T_short * C, D)
        edge_indices = [dense_to_sparse(adj)[0] for adj in sparse_A]
        full_data_list = [Data(num_nodes=C, edge_index=edge_indices[b]) for b in range(B) for _ in range(T_short)]
        final_batched_data = Batch.from_data_list(full_data_list).to(x.device)
        edge_index_final = final_batched_data.edge_index
        g_out = self.gnn(node_features, edge_index_final)
        x_updated = g_out.view(B, T_short, C, D).permute(0, 2, 1, 3)
        return x_updated, delta_A

# =================================================================== #
#  Part 3: 最终的 HybridBlock (融合版)
# =================================================================== #

class MllaTemporalBlock(nn.Module):
    """
    一个包装器，用于在我们的4D数据流中使用 MLLAEncoderLayer。
    """
    def __init__(self, d_model, n_heads, q_len):
        super().__init__()
        self.encoder = MLLAEncoderLayer(q_len=q_len, d_model=d_model, n_heads=n_heads)

    def forward(self, x):
        B, C, T_short, D = x.shape
        x_reshaped = x.contiguous().view(B * C, T_short, D)
        output = self.encoder(x_reshaped)
        return output.view(B, C, T_short, D)

class HybridBlock(nn.Module):
    """
    时空混合块 (MLLA + DGL 融合版)。
    """
    def __init__(self, num_channels, feature_dim, gnn_heads, dropout, top_k, max_T_short):
        super().__init__()
        self.spatial_module = SpatialBlock(
            num_channels, feature_dim, gnn_heads=gnn_heads, 
            key_dim=32, top_k=top_k
        )
        self.temporal_module = MllaTemporalBlock(
            d_model=feature_dim, 
            n_heads=gnn_heads,
            q_len=max_T_short
        )
        
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        x_norm = self.norm1(x)
        x_spatial, delta_A = self.spatial_module(x_norm)
        x = residual + self.dropout(x_spatial)
        
        residual = x
        x_norm = self.norm2(x)
        x_temporal = self.temporal_module(x_norm)
        output = residual + self.dropout(x_temporal)
        
        return output, delta_A