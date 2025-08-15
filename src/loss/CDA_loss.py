import torch
import torch.nn as nn
import torch.nn.functional as F


class CDALoss(nn.Module):
    """
    Cross-Dataset Alignment (CDA) Loss.

    输入：
        - fea_list: List[Tensor]，长度 = #datasets。
          每个 Tensor 形状为 [B, D, C, T]：
            B: batch size
            D: feature dim（例如 CNN/Transformer 输出的通道数）
            C: 通道数（EEG导联）
            T: 时间步（或 patch 数）
        也兼容以下误用：
            - 单个 Tensor [B, D, C, T]，会自动包成 list
            - 单个 Tensor [B, C, T]，自动补 D=1 -> [B, 1, C, T]

    计算：
        1) 对每个数据集，先把 [B, D, C, T] -> 计算每个样本的通道协方差 cov ∈ R^{C×C}
           方法：先把特征 reshape 为 X ∈ R^{B×C×(D*T)}，做去均值，再 X X^T/(N-1)
        2) 对每个数据集得到其 batch 平均协方差中心（可选 log-Euclidean）
        3) 对所有数据集中心做两两 Frobenius 距离并平均，作为 LCDA

    可选：
        - cfg.train.loss.to_riem = True 时，对 SPD 做对数映射（log-Euclidean mean）
    """
    def __init__(self, cfg):
        super().__init__()
        loss_cfg = getattr(cfg.train, 'loss', None)
        self.to_riem = bool(getattr(loss_cfg, 'to_riem', False)) if loss_cfg is not None else False
        self.eps = 1e-6

    @staticmethod
    def _ensure_list_4d(x):
        """
        将输入标准化为 List[Tensor[B, D, C, T]]
        """
        if isinstance(x, (list, tuple)):
            out = []
            for t in x:
                if not torch.is_tensor(t):
                    raise TypeError("CDA_loss expects a list of Tensors.")
                if t.ndim == 4:
                    out.append(t)
                elif t.ndim == 3:
                    # 补一个 D=1 维度
                    out.append(t.unsqueeze(1))  # [B, 1, C, T]
                else:
                    raise ValueError(f"CDA_loss tensor must be 3D or 4D, got {t.ndim}D.")
            return out

        # 单个 Tensor
        if not torch.is_tensor(x):
            raise TypeError("CDA_loss expects a Tensor or a list of Tensors.")
        if x.ndim == 4:
            return [x]
        if x.ndim == 3:
            return [x.unsqueeze(1)]  # [B, 1, C, T]
        raise ValueError(f"CDA_loss tensor must be 3D or 4D, got {x.ndim}D.")

    def cov_mat(self, data: torch.Tensor) -> torch.Tensor:
        """
        data: [B, D, C, T]  ->  返回 covs: [B, C, C]
        """
        if data.ndim == 3:
            data = data.unsqueeze(1)  # [B, 1, C, T]
        if data.ndim != 4:
            raise ValueError(f"cov_mat expects 4D tensor [B, D, C, T], got {data.ndim}D.")

        B, D, C, T = data.shape
        # [B, D, C, T] -> [B, C, D*T]
        x = data.permute(0, 2, 1, 3).reshape(B, C, D * T)  # (B, C, N)
        x = x - x.mean(dim=2, keepdim=True)
        N = x.shape[-1]
        denom = (N - 1) if N > 1 else 1.0
        cov = torch.matmul(x, x.transpose(1, 2)) / denom  # [B, C, C]
        # 数值稳定
        cov = cov + self.eps * torch.eye(C, device=cov.device, dtype=cov.dtype).unsqueeze(0)
        return cov

    def _spd_log(self, covs: torch.Tensor) -> torch.Tensor:
        """
        对 SPD 矩阵做 log 映射（逐样本），输入 [B, C, C] -> 输出 [B, C, C]
        """
        # 用特征分解做矩阵对数：log(U diag(lam) U^T) = U diag(log(lam)) U^T
        # clamp 特征值确保正
        lam, U = torch.linalg.eigh(covs)
        lam = torch.clamp(lam, min=self.eps)
        log_lam = torch.log(lam)
        # 重建
        # U * diag(log_lam) * U^T
        out = torch.matmul(U, torch.matmul(torch.diag_embed(log_lam), U.transpose(-1, -2)))
        return out

    def forward(self, fea_list):
        """
        fea_list: List[Tensor[B, D, C, T]] 或 单个 Tensor (3D/4D)
        返回：标量 loss
        """
        tensors = self._ensure_list_4d(fea_list)
        if len(tensors) < 2:
            # 只有一个数据集就没有跨域对齐的意义，返回 0
            return torch.zeros([], device=tensors[0].device if tensors else 'cpu')

        centers = []
        for t in tensors:
            covs = self.cov_mat(t)           # [B, C, C]
            if self.to_riem:
                covs = self._spd_log(covs)   # log-Euclidean
            center = covs.mean(dim=0)        # [C, C]
            centers.append(center)

        # 两两 Frobenius 距离并平均
        loss = 0.0
        n = len(centers)
        cnt = 0
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                loss = loss + torch.norm(diff, p='fro') ** 2
                cnt += 1
        loss = loss / max(cnt, 1)
        return loss
