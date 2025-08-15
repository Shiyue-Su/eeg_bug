import torch.nn as nn
import torch.nn.functional as F
import torch
import pytorch_lightning as pl

class simpleNN3(nn.Module):
    def __init__(self, inp_dim, hidden_dim, out_dim, dropout=0.5, bn='bn2'):
        super(simpleNN3, self).__init__()
        self.fc1 = nn.Linear(inp_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim, affine=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2, affine=False)
        self.fc3 = nn.Linear(hidden_dim // 2, out_dim)
        self.bn = bn
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        if (self.bn == 'bn1') or (self.bn == 'bn2'):
            out = self.bn1(out)
        out = self.drop(out)
        out = F.relu(self.fc2(out))
        if self.bn == 'bn2':
            out = self.bn2(out)
        out = self.drop(out)
        out = self.fc3(out)
        return out

class MLPModel(pl.LightningModule):
    def __init__(self, model, cfg) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.cfg = cfg
        self.lr = cfg.lr
        self.wd = cfg.wd
        ls = getattr(cfg, 'label_smoothing', 0.0)
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=ls)
        self.metric = accuracy

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        use_plateau = getattr(self.cfg, 'use_plateau', True)
        if use_plateau:
            factor   = getattr(self.cfg, 'lr_factor', 0.5)
            patience = getattr(self.cfg, 'lr_patience', 3)
            min_lr   = getattr(self.cfg, 'min_lr', 1e-6)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode='min', factor=factor, patience=patience, min_lr=min_lr
            )
            return {"optimizer": opt,
                    "lr_scheduler": {"scheduler": sched, "monitor": "mlp/val/loss"}}
        return opt

    def training_step(self, batch, batch_idx):
        data, labels = batch
        logits = self.model(data)
        loss = self.criterion(logits, labels)
        top1 = self.metric(logits, labels, topk=(1,))
        self.log_dict({'mlp/train/loss': loss, 'mlp/train/acc': top1[0]},
                      on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, labels = batch
        logits = self.model(data)
        loss = self.criterion(logits, labels)
        top1 = self.metric(logits, labels, topk=(1,))
        self.log_dict({'mlp/val/loss': loss, 'mlp/val/acc': top1[0]},
                      on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        data, labels = batch
        logits = self(data)
        return logits.argmax(dim=1)

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
