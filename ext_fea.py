import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["WORLD_SIZE"] = "1"

import glob
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import mne

from src.data.io_utils import load_finetune_EEG_data
from src.data.data_process import running_norm_onesubsession, LDS_gpu
from src.data.dataset import ext_Dataset
from src.model.MultiModel_PL import MultiModel_PL
from src.utils import video_order_load, reorder_vids_sepVideo, reorder_vids_back

torch.set_float32_matmul_precision('high')

def cal_fea_from_pred(pred_np: np.ndarray, mode: str):
    """
    pred_np: [N, Dp, 1, Tshort]  (our predict_step returns a 4D tensor)
    mode: 'de' or 'me'
    """
    if mode == 'de':
        fea = 0.5 * np.log(2 * np.pi * np.e * (np.var(pred_np, axis=3)) + 1.0).squeeze()
        fea[fea < -40] = -40
    elif mode == 'me':
        fea = np.mean(pred_np, axis=3).squeeze()
    else:
        raise ValueError(f"Unsupported fea_mode: {mode}")
    return fea

def resolve_ckpt_path(run_name: str, epoch_idx: int, ckpt_tag: str | None) -> str:
    cp_dir = os.path.join('log', run_name, 'ckpt')
    os.makedirs(cp_dir, exist_ok=True)
    epoch_str = f"{epoch_idx:02d}"
    name = f"epoch={epoch_str}.ckpt" if not ckpt_tag else f"epoch={epoch_str}-{ckpt_tag}.ckpt"
    cp_path = os.path.join(cp_dir, name)
    if os.path.exists(cp_path):
        return cp_path
    cands = sorted(glob.glob(os.path.join(cp_dir, f"epoch={epoch_str}-*.ckpt")))
    if cands:
        return cands[-1]
    all_ckpts = sorted(glob.glob(os.path.join(cp_dir, "epoch=*.ckpt")), key=os.path.getmtime, reverse=True)
    if all_ckpts:
        return all_ckpts[0]
    raise FileNotFoundError(f"No checkpoint found in {cp_dir} for epoch={epoch_str}, tag={ckpt_tag}")

@hydra.main(config_path="cfgs_multi", config_name="config_multi", version_base="1.3")
def ext_fea(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.train.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    load_dir = os.path.join(cfg.data_val.data_dir, 'processed_data')
    print('data loading...')
    data, onesub_label, n_samples_onesub, n_samples_sessions = load_finetune_EEG_data(load_dir, cfg.data_val)
    print('data loaded')
    print(f'data ori shape:{data.shape}')
    print(f'n_samples_onesub shape:{n_samples_onesub.shape}')
    print(f'n_samples_sessions shape:{n_samples_sessions.shape}')

    # reshape to [subs, trials, chans, points]
    data = data.reshape(cfg.data_val.n_subs, -1, data.shape[-2], data.shape[-1])

    save_dir = os.path.join(cfg.data_val.data_dir, 'ext_fea')
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'onesub_label.npy'), onesub_label)

    # init model&trainer
    Extractor = None
    trainer = None
    if cfg.val.extractor.use_pretrain:
        print('Use pretrain model (DGE):')
        ckpt_epoch = cfg.val.extractor.ckpt_epoch
        ckpt_tag = getattr(cfg.val.extractor, "ckpt_tag", None)
        cp_path = resolve_ckpt_path(cfg.log.run_name, ckpt_epoch - 1, ckpt_tag)
        print(f'checkpoint load from: {cp_path}')

        cfg.data_cfg_list = [cfg.data_0, cfg.data_1, cfg.data_2, cfg.data_3, cfg.data_4, cfg.data_val]
        cfg.data_cfg_list = [cfg_i for cfg_i in cfg.data_cfg_list if cfg_i.dataset_name != 'None']

        Extractor = MultiModel_PL.load_from_checkpoint(checkpoint_path=cp_path, cfg=cfg, strict=False)
        trainer = pl.Trainer(accelerator='gpu', devices=1)

    # folds
    if cfg.val.extractor.normTrain:
        val_subs_all = cfg.data_val.val_subs_all
        if cfg.val.n_fold == "loo":
            val_subs_all = [[i] for i in range(cfg.data_val.n_subs)]
        n_folds = len(val_subs_all)
    else:
        n_folds = 1

    for fold in tqdm(range(n_folds), desc='Extracting feature......'):
        if cfg.val.extractor.normTrain:
            val_subs = cfg.data_val.val_subs_all[fold]
            train_subs = list(set(range(cfg.data_val.n_subs)) - set(val_subs))
        else:
            val_subs = list(range(cfg.data_val.n_subs))
            train_subs = list(range(cfg.data_val.n_subs))

        if cfg.val.extractor.reverse:
            train_subs, val_subs = val_subs, train_subs
        print(f'train_subs:{train_subs}')
        print(f'val_subs:{val_subs}')

        data_fold = data
        if cfg.val.extractor.normTrain:
            # normalize by train subjects stats (time-wise)
            temp = np.transpose(data[train_subs], (0,1,3,2)).reshape(-1, data.shape[-2])
            mean = np.mean(temp, axis=0)
            var  = np.var(temp, axis=0)
            data_fold = (data - mean.reshape(1,1,-1,1)) / (np.sqrt(var + 1e-5).reshape(1,1,-1,1))

        if cfg.val.extractor.use_pretrain:
            flat = data_fold.reshape(-1, data_fold.shape[-2], data_fold.shape[-1])  # [N, C, T]
            label_fold = np.tile(onesub_label, cfg.data_val.n_subs)
            foldset = ext_Dataset(flat, label_fold)
            loader = DataLoader(foldset, batch_size=cfg.val.extractor.batch_size, shuffle=False,
                                num_workers=cfg.train.num_workers, pin_memory=True)
            pred_list = trainer.predict(Extractor, loader)
            pred = torch.cat(pred_list, dim=0).cpu().numpy()  # [N, Dp, 1, Tshort]
            fea = cal_fea_from_pred(pred, cfg.val.extractor.fea_mode)  # [N, Dp]
            fea = fea.reshape(cfg.data_val.n_subs, -1, fea.shape[-1])
        else:
            # direct DE features from raw EEG (fallback)
            n_subs, n_samples, n_chans, sfreqs = data_fold.shape
            freqs = [[1,4], [4,8], [8,14], [14,30], [30,47]]
            de_data = np.zeros((n_subs, n_samples, n_chans, len(freqs)))
            n_samples_onesub_cum = np.concatenate((np.array([0]), np.cumsum(n_samples_onesub)))
            for idx, band in enumerate(freqs):
                for sub in tqdm(range(n_subs)):
                    for vid in tqdm(range(len(n_samples_onesub)), desc=f'DE sub: {sub}', leave=False):
                        d = data_fold[sub, n_samples_onesub_cum[vid]:n_samples_onesub_cum[vid+1]]
                        d = d.transpose(1,0,2).reshape(n_chans, -1)
                        filt = mne.filter.filter_data(d, sfreqs, l_freq=band[0], h_freq=band[1], verbose=False)
                        filt = filt.reshape(n_chans, -1, sfreqs)
                        de_onevid = 0.5*np.log(2*np.pi*np.e*(np.var(filt, 2))).T
                        de_data[sub, n_samples_onesub_cum[vid]:n_samples_onesub_cum[vid+1], :, idx] = de_onevid
            fea = de_data.reshape(n_subs, n_samples, -1)

        # post-processing as original
        fea_train = fea[list(set(range(cfg.data_val.n_subs)) - set(val_subs)) if cfg.val.extractor.normTrain else list(range(cfg.data_val.n_subs))]
        data_mean = np.mean(np.mean(fea_train, axis=1), axis=0)
        data_var  = np.mean(np.var(fea_train, axis=1), axis=0)

        # FACED reorder (if used)
        if cfg.data_val.dataset_name == 'FACED':
            vid_order = video_order_load(cfg.data_val.n_vids)
            n_vids = 28
            vid_inds = np.arange(n_vids)
            fea, vid_play_order_new = reorder_vids_sepVideo(fea, vid_order, vid_inds, n_vids)

        # running norm by session
        n_sample_sum_sessions = np.sum(n_samples_sessions, 1)
        cum = np.concatenate((np.array([0]), np.cumsum(n_sample_sum_sessions)))
        for sub in tqdm(range(cfg.data_val.n_subs), desc='Running norm'):
            for s in range(len(n_sample_sum_sessions)):
                st, ed = cum[s], cum[s+1]
                fea[sub, st:ed] = running_norm_onesubsession(fea[sub, st:ed], data_mean, data_var, cfg.val.extractor.rn_decay)

        if cfg.data_val.dataset_name == 'FACED':
            fea = reorder_vids_back(fea, len(vid_inds), vid_play_order_new)

        if cfg.val.extractor.LDS:
            n_samples_onesub_cum = np.concatenate((np.array([0]), np.cumsum(n_samples_onesub)))
            for sub in tqdm(range(cfg.data_val.n_subs), desc='LDS......'):
                for vid in range(len(n_samples_onesub)):
                    st, ed = n_samples_onesub_cum[vid], n_samples_onesub_cum[vid+1]
                    fea[sub, st:ed] = LDS_gpu(fea[sub, st:ed])

        fea = fea.reshape(-1, fea.shape[-1])

        ckpt_tag = getattr(cfg.val.extractor, "ckpt_tag", None)
        epoch_str = f"{(cfg.val.extractor.ckpt_epoch-1):02d}" + (f"-{ckpt_tag}" if ckpt_tag else "")
        if not cfg.val.extractor.normTrain:
            save_path = os.path.join(save_dir, f"{cfg.log.run_name}_all_fea_epoch={epoch_str}_{cfg.val.extractor.fea_mode}.npy")
        else:
            save_path = os.path.join(save_dir, f"{cfg.log.run_name}_f{fold}_fea_epoch={epoch_str}_{cfg.val.extractor.fea_mode}.npy")
        np.save(save_path, fea)
        print(f"Feature saved to {save_path}")

        if not cfg.val.extractor.normTrain:
            break

if __name__ == "__main__":
    ext_fea()
