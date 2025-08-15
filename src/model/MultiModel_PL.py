# -*- coding: utf-8 -*-
import os
import numpy as np
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import OmegaConf

try:
    from src.model.DGE_Net import DGE_Net as Encoder
except ImportError:
    Encoder = None

try:
    from src.loss.simclr import SimCLRLoss as ProjectSimCLRLoss
except ImportError:
    ProjectSimCLRLoss = None

try:
    from src.loss.cda import CDALoss as ProjectCDALoss
except ImportError:
    ProjectCDALoss = None

class _NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.2):
        super().__init__()
        self.tau = float(temperature)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        B, D = z1.shape
        z = torch.cat([z1, z2], dim=0)
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / self.tau
        sim.fill_diagonal_(-1e9)
        labels = torch.arange(B, device=z.device)
        labels = torch.cat([labels + B, labels])
        return F.cross_entropy(sim, labels)

class SimpleDecoder(nn.Module):
    def __init__(self, feature_dim: int, patch_len: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, patch_len)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)

class MultiModel_PL(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        if Encoder is None:
            raise RuntimeError("Failed to import DGE_Net. Please ensure it exists and is correct.")
        
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.mlla_cfg = cfg.model.MLLA

        self.encoder = Encoder(cfg=cfg)
        self.decoder = SimpleDecoder(self.mlla_cfg.out_dim, self.mlla_cfg.patch_size)

        self.clisa_loss = (ProjectSimCLRLoss or _NTXentLoss)(temperature=cfg.train.loss.temp)
        self.cda_loss = ProjectCDALoss(cfg) if ProjectCDALoss is not None else None
        self.recon_loss = nn.MSELoss(reduction='none')

        self.w_sim = cfg.train.loss.w_sim
        self.w_rec = cfg.train.loss.w_rec
        self.w_cda = cfg.train.loss.w_cda

        self.channel_interpolate = np.load('channel_interpolate.npy').astype(int)
        self.uni_channelname = self.mlla_cfg.uni_channels

    def channel_project(self, data, cha_source):
        device = data.device
        if data.dim() == 5:
            data = data.reshape(-1, 1, data.size(3), data.size(4))
        elif data.dim() != 4:
            raise ValueError(f"Expected 4D or 5D tensor, but got {data.dim()}D")

        batch_size, _, n_channel_source, n_timepoint = data.shape
        n_channel_standard = len(self.uni_channelname)
        source_ch_map = {name.upper(): idx for idx, name in enumerate(cha_source)}
        result = torch.zeros((batch_size, 1, n_channel_standard, n_timepoint), device=device, dtype=data.dtype)
        
        for std_idx, std_name in enumerate(self.uni_channelname):
            std_name_upper = std_name.upper()
            if std_name_upper in source_ch_map:
                src_idx = source_ch_map[std_name_upper]
                result[:, :, std_idx] = data[:, :, src_idx]
            else:
                neighbor_std_indices = self.channel_interpolate[std_idx]
                valid_src_indices = [source_ch_map[self.uni_channelname[i.item()].upper()] for i in neighbor_std_indices if self.uni_channelname[i.item()].upper() in source_ch_map]
                if len(valid_src_indices) > 0:
                    neighbor_data = data[:, :, valid_src_indices, :]
                    result[:, :, std_idx] = neighbor_data.mean(dim=2)
        return result

    def _patchify(self, x):
        # x: [B, C, T]
        x_unf = x.unfold(dimension=2, size=self.mlla_cfg.patch_size, step=self.mlla_cfg.patch_stride)
        return x_unf # [B, C, Np, P]

    def _mask_input(self, patches):
        B, C, Np, P = patches.shape
        mask_ratio = self.cfg.train.loss.get("mask_ratio", 0.25)
        num_to_mask = int(Np * mask_ratio)
        masked_patches = patches.clone()
        mask = torch.zeros(B, C, Np, dtype=torch.bool, device=patches.device)
        for b in range(B):
            for c in range(C):
                perm = torch.randperm(Np, device=patches.device)
                idx_to_mask = perm[:num_to_mask]
                mask[b, c, idx_to_mask] = True
                masked_patches[b, c, idx_to_mask, :] = 0.0
        return masked_patches, mask

    def training_step(self, batch: Tuple[List[torch.Tensor], List[torch.Tensor]], batch_idx: int):
        x_list, _ = batch
        total_loss = 0
        all_summaries_for_cda = []
        
        for i, x_raw in enumerate(x_list):
            if x_raw is None or x_raw.numel() == 0: continue

            source_channels = self.cfg[f'data_{i}'].channels
            x_proj = self.channel_project(x_raw, source_channels)
            x = x_proj.squeeze(1)

            original_patches = self._patchify(x)
            masked_patches, mask = self._mask_input(original_patches)

            summary_vec, feat_map = self.encoder(masked_patches)
            summary_vec = summary_vec.squeeze()

            loss_sim, loss_rec = 0, 0
            if self.w_sim > 0:
                z1, z2 = torch.chunk(summary_vec, 2, dim=0)
                loss_sim = self.clisa_loss(z1, z2)
                self.log(f'train/loss_sim_ds{i}', loss_sim, on_step=True, on_epoch=True)

            if self.w_rec > 0:
                reconstructed_patches = self.decoder(feat_map)
                loss_rec_per_patch = self.recon_loss(reconstructed_patches, original_patches)
                loss_rec = loss_rec_per_patch[mask].mean()
                self.log(f'train/loss_rec_ds{i}', loss_rec, on_step=True, on_epoch=True)

            if self.w_cda > 0 and self.cda_loss is not None:
                all_summaries_for_cda.append(summary_vec)

            domain_loss = (self.w_sim * loss_sim) + (self.w_rec * loss_rec)
            total_loss += domain_loss

        loss_cda = 0
        if self.w_cda > 0 and self.cda_loss is not None and len(all_summaries_for_cda) > 1:
            loss_cda = self.cda_loss(all_summaries_for_cda)
            self.log('train/loss_cda', loss_cda, on_step=False, on_epoch=True)
        
        final_loss = total_loss + (self.w_cda * loss_cda)
        self.log('train/total_loss', final_loss, prog_bar=True, on_step=True, on_epoch=True)
        
        return final_loss

    def configure_optimizers(self):
        lr = self.hparams.train.lr
        wd = self.hparams.train.wd
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)
        return optimizer