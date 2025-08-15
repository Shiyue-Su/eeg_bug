import hydra
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["WORLD_SIZE"] = "1"

from omegaconf import DictConfig
from src.model.valMLP import simpleNN3, MLPModel
import numpy as np
from src.data.dataset import PDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import torch
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

@hydra.main(config_path="cfgs_multi", config_name="config_multi", version_base="1.3")
def train_mlp(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.val.mlp.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    val_subs_all = cfg.data_val.val_subs_all
    if cfg.val.n_fold == "loo":
        val_subs_all = [[i] for i in range(cfg.data_val.n_subs)]
    n_folds = len(val_subs_all)

    for fold in range(n_folds):
        print(f"=== Fold {fold} ===")
        val_subs = val_subs_all[fold]
        train_subs = list(set(range(cfg.data_val.n_subs)) - set(val_subs))
        if cfg.val.extractor.reverse:
            train_subs, val_subs = val_subs, train_subs

        # feature file (must match ext_fea.py naming)
        save_dir = os.path.join(cfg.data_val.data_dir, 'ext_fea')
        epoch_str = f"{(cfg.val.extractor.ckpt_epoch-1):02d}"
        ckpt_tag = getattr(cfg.val.extractor, "ckpt_tag", None)
        if ckpt_tag:
            epoch_str = epoch_str + f"-{ckpt_tag}"
        if not cfg.val.extractor.normTrain:
            file_name = f"{cfg.log.run_name}_all_fea_epoch={epoch_str}_{cfg.val.extractor.fea_mode}.npy"
        else:
            file_name = f"{cfg.log.run_name}_f{fold}_fea_epoch={epoch_str}_{cfg.val.extractor.fea_mode}.npy"
        save_path = os.path.join(save_dir, file_name)
        print(f"Loading features from: {save_path}")
        data = np.load(save_path)
        data = np.nan_to_num(data)
        data = data.reshape(cfg.data_val.n_subs, -1, data.shape[-1])

        onesub_label = np.load(os.path.join(save_dir, 'onesub_label.npy'))
        train_labels = np.tile(onesub_label, len(train_subs))
        val_labels   = np.tile(onesub_label, len(val_subs))

        trainset = PDataset(data[train_subs].reshape(-1, data.shape[-1]), train_labels)
        valset   = PDataset(data[val_subs].reshape(-1, data.shape[-1]), val_labels)

        trainLoader = DataLoader(trainset, batch_size=cfg.val.mlp.batch_size, shuffle=True,  num_workers=cfg.train.num_workers, pin_memory=True)
        valLoader   = DataLoader(valset,   batch_size=cfg.val.mlp.batch_size, shuffle=False, num_workers=cfg.train.num_workers, pin_memory=True)

        fea_dim = data.shape[-1]
        base_model = simpleNN3(fea_dim, cfg.val.mlp.hidden_dim, cfg.val.mlp.out_dim, cfg.val.mlp.dropout)
        lightning_module = MLPModel(base_model, cfg.val.mlp)

        cp_dir = os.path.join(cfg.log.mlp_cp_dir, cfg.log.run_name)
        os.makedirs(cp_dir, exist_ok=True)
        checkpoint_callback = ModelCheckpoint(monitor="mlp/val/acc", mode="max", dirpath=cp_dir, filename=f"{cfg.data_val.dataset_name}_mlp_f{fold}"+"_{epoch}", save_top_k=1)

        trainer = pl.Trainer(max_epochs=cfg.val.mlp.max_epochs, min_epochs=cfg.val.mlp.min_epochs,
                             accelerator='gpu', devices=1, callbacks=[checkpoint_callback])
        trainer.fit(lightning_module, trainLoader, valLoader)

        # eval best
        best = MLPModel.load_from_checkpoint(checkpoint_callback.best_model_path, model=base_model, cfg=cfg.val.mlp)
        best.eval(); best.freeze()
        y_true, y_pred, y_prob = [], [], []
        for xb, yb in valLoader:
            logits = best.model(xb.to(best.device))
            probs  = torch.softmax(logits, dim=1).detach().cpu().numpy()
            y_true.append(yb.numpy())
            y_pred.append(np.argmax(probs, axis=1))
            y_prob.append(probs)
        y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred); y_prob = np.concatenate(y_prob)
        acc = (y_pred == y_true).mean()
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall    = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1        = f1_score(y_true, y_pred, average='macro', zero_division=0)

        n_classes = len(np.unique(y_true))
        if n_classes == 2:
            auroc = roc_auc_score(y_true, y_prob[:,1])
            auprc = average_precision_score(y_true, y_prob[:,1])
        elif n_classes > 2:
            y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
            auroc = roc_auc_score(y_true_bin, y_prob, multi_class='ovo', average='macro')
            auprc = average_precision_score(y_true_bin, y_prob, average='macro')
        else:
            auroc, auprc = 0.0, 0.0

        print(f"Fold {fold}: Acc={acc*100:.2f}%, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}, AUROC={auroc:.4f}, AUPRC={auprc:.4f}")

if __name__ == '__main__':
    train_mlp()
