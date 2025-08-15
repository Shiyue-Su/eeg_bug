# =========================
# train_multi.py
# =========================
import os
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("WORLD_SIZE", "1")

import multiprocessing as mp
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.model.MultiModel_PL import MultiModel_PL
from src.data.multi_dataloader import MultiDataModule

import logging
log = logging.getLogger(__name__)


def _ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


@hydra.main(config_path="cfgs_multi", config_name="config_multi", version_base="1.3")
def run_pipeline(cfg: DictConfig):
    # ===================================================================
    # GEMINI DEBUG: Print the loaded configuration to diagnose the issue
    # ===================================================================
    print("\n" + "="*50)
    print("--- GEMINI DEBUG: LOADED CONFIGURATION ---")
    print(OmegaConf.to_yaml(cfg))
    print("--- END GEMINI DEBUG ---")
    print("="*50 + "\n")

    # Explicitly check for the problematic key before doing anything else
    if 'hybrid' not in cfg.model:
        print("FATAL ERROR: The loaded configuration is MISSING the 'model.hybrid' section.")
        print("Please ensure you are running with 'config_multi.yaml' and that the file is correct.")
        return # Stop execution
    else:
        print("SUCCESS: 'model.hybrid' section was found in the configuration.")

    # reproducibility
    pl.seed_everything(cfg.train.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    run_name = cfg.log.run_name
    save_dir = os.path.join("log", run_name)
    _ensure_dir(save_dir)
    cfg_path = os.path.join(save_dir, f"{run_name}.yaml")
    OmegaConf.save(config=cfg, f=cfg_path)
    print(f"Config saved to: {cfg_path}")

    cfg.data_cfg_list = [cfg.data_0, cfg.data_1, cfg.data_2, cfg.data_3, cfg.data_4]
    cfg.data_cfg_list = [d for d in cfg.data_cfg_list if d is not None and d.dataset_name != "None"]
    print(f"Using {len(cfg.data_cfg_list)} datasets to pretrain\n")

    n_folds = int(cfg.train.n_fold)

    for fold in range(n_folds):
        print(f"========== fold {fold} / {n_folds} ==========")

        dm = MultiDataModule(
            data_cfg_list=cfg.data_cfg_list,
            fold=fold,
            n_folds=n_folds,
            num_workers=int(cfg.train.num_workers),
            n_pairs=int(cfg.train.n_pairs),
        )
        dm.setup("fit")

        model = MultiModel_PL(cfg)

        total_params = sum(p.numel() for p in model.parameters())
        total_size = sum(p.numel() * p.element_size() for p in model.parameters())
        log.info(f"Total number of parameters: {total_params}")
        log.info(f"Model size: {total_size} bytes ({total_size / (1024 ** 2):.2f} MB)")

        fold_dir = os.path.join(save_dir, str(fold))
        _ensure_dir(fold_dir)
        cp_dir = os.path.join(save_dir, "ckpt")
        _ensure_dir(cp_dir)

        checkpoint_callback = ModelCheckpoint(
            dirpath=cp_dir,
            filename="{epoch:02d}",
            every_n_epochs=int(cfg.train.save_interval),
            save_top_k=-1,
        )

        limit_val_batches = 0.0 if n_folds == 1 else 1.0

        trainer = pl.Trainer(
            logger=TensorBoardLogger(save_dir=fold_dir, name=run_name),
            callbacks=[checkpoint_callback],
            max_epochs=int(cfg.train.max_epochs),
            min_epochs=int(cfg.train.min_epochs),
            accelerator="gpu",
            devices=int(cfg.train.n_gpu_use),
            strategy="ddp_find_unused_parameters_true",
            limit_val_batches=limit_val_batches,
            num_sanity_val_steps=0,
        )

        trainer.fit(model, dm)


if __name__ == "__main__":
    run_pipeline()