import os
import random
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

from src.data.dataset import EEG_Dataset


def get_train_subs(n_subs, fold, n_folds):
    """按折数切分被试；当 n_folds == 1 时，全部作为 train，val 为空"""
    n_per = round(n_subs / n_folds) if n_folds > 0 else n_subs
    if n_folds == 1:
        val_subs = []
    elif fold < n_folds - 1:
        val_subs = list(range(n_per * fold, n_per * (fold + 1)))
    else:
        val_subs = list(range(n_per * fold, n_subs))
    train_subs = list(sorted(set(range(n_subs)) - set(val_subs)))
    return [train_subs, val_subs]


class EEGSampler:
    """
    基于“子会话两两配对”的索引采样器。
    - 支持多数据集（datasets 为 EEG_Dataset 的列表）
    - 保持你原有的 get_sample / pair 构造逻辑
    - 额外的健壮性处理：若某数据集没有可配对样本，跳过该数据集的采样
    """
    def __init__(self, datasets, n_pairs):
        self.n_pairs = int(n_pairs)
        self.datasets = datasets
        self.num_datasets = len(datasets)

        # 记录每个数据集的被试 / 会话数
        self.subs_list = []
        self.n_subs_list = []
        self.n_sessions_list = []

        # 为每个数据集构造 subsession-pairs
        self.pairs_list = []
        self.n_pairs_list = []
        self.max_n_pairs = 0

        # 预加载一些需要的元数据（来自切片目录）
        self.save_dirs = []
        self.sliced_data_dirs = []
        self.n_samples_session_list = []
        self.n_vids_list = []
        self.batch_sizes = []
        self.n_sessions = []
        self.n_per_session_list = []
        self.n_per_session_cum_list = []
        self.n_samples_per_trial_list = []
        self.n_samples_cum_session_list = []

        # 初始化
        for dataset in self.datasets:
            # 以 train_subs 为主（val sampler 会传 valsets 进来）
            subs = dataset.train_subs if dataset.train_subs is not None else dataset.val_subs
            self.subs_list.append(subs)
            self.n_subs_list.append(len(subs))
            self.n_sessions_list.append(dataset.n_session)

        for idx, dataset in enumerate(self.datasets):
            n_subs = self.n_subs_list[idx]
            n_sessions = self.n_sessions_list[idx]
            print(f'dataset {idx}: n_subs={n_subs} n_sessions={n_sessions}')

            pairs = []
            # 如果该数据集没有样本，直接记录空对
            if n_subs == 0 or n_sessions == 0:
                self.pairs_list.append([])
                self.n_pairs_list.append(0)
                continue

            # 构造不同被试但相同 session 的子会话对
            # 将 (sub, session) 展开为 range(n_subs * n_sessions)，
            # 跨被试同 session 的索引差是 n_sessions 的倍数
            for i in range(n_subs * n_sessions):
                for j in range(i + n_sessions, n_subs * n_sessions, n_sessions):
                    pairs.append((i, j))

            random.shuffle(pairs)
            self.pairs_list.append(pairs)
            self.n_pairs_list.append(len(pairs))
            self.max_n_pairs = max(self.max_n_pairs, len(pairs))

            # 预加载切片元信息
            save_dir = os.path.join(dataset.cfg.data_dir, 'sliced_data')
            sliced_data_dir = os.path.join(
                save_dir, f'sliced_len{dataset.cfg.timeLen}_step{dataset.cfg.timeStep}'
            )
            n_samples_session = np.load(
                os.path.join(sliced_data_dir, 'metadata', 'n_samples_sessions.npy')
            )
            n_vid = dataset.cfg.n_vids
            batch_size = n_vid
            n_session = dataset.cfg.n_session
            n_per_session = np.sum(n_samples_session, 1).astype(int)
            n_per_session_cum = np.concatenate((np.array([0]), np.cumsum(n_per_session)))
            n_samples_per_trial = int(n_vid / n_samples_session.shape[1])
            n_samples_cum_session = np.concatenate(
                (np.zeros((n_session, 1)), np.cumsum(n_samples_session, 1)), 1
            )

            self.save_dirs.append(save_dir)
            self.sliced_data_dirs.append(sliced_data_dir)
            self.n_samples_session_list.append(n_samples_session)
            self.n_vids_list.append(n_vid)
            self.batch_sizes.append(batch_size)
            self.n_sessions.append(n_session)
            self.n_per_session_list.append(n_per_session)
            self.n_per_session_cum_list.append(n_per_session_cum)
            self.n_samples_per_trial_list.append(n_samples_per_trial)
            self.n_samples_cum_session_list.append(n_samples_cum_session)

        print('Dataloader Lengths:')
        print(self.n_pairs_list)

    def get_sample(self, dataset_idx, subsession_pair):
        n_per_session = self.n_per_session_list[dataset_idx]
        n_per_session_cum = self.n_per_session_cum_list[dataset_idx]
        n_samples_cum_session = self.n_samples_cum_session_list[dataset_idx]
        n_session = self.n_sessions[dataset_idx]
        n_samples_per_trial = self.n_samples_per_trial_list[dataset_idx]
        batch_size = self.batch_sizes[dataset_idx]

        subsession1, subsession2 = subsession_pair

        # 要求同一个 session
        cur_session = int(subsession1 % n_session)
        assert cur_session == int(subsession2 % n_session), "Subsessions are from different sessions"

        cur_sub1 = int(subsession1 // n_session)
        cur_sub2 = int(subsession2 // n_session)

        ind_abs_list = []
        n_trials = len(n_samples_cum_session[cur_session]) - 2

        for i in range(n_trials):
            start = int(n_samples_cum_session[cur_session][i])
            end = int(n_samples_cum_session[cur_session][i + 1])
            ind_one = np.random.choice(np.arange(start, end), n_samples_per_trial, replace=False)
            ind_abs_list.append(ind_one)

        # 剩余拼满 batch
        i = n_trials
        start = int(n_samples_cum_session[cur_session][i])
        end = int(n_samples_cum_session[cur_session][i + 1])
        remaining_samples = int(batch_size - n_samples_per_trial * n_trials)
        ind_one = np.random.choice(np.arange(start, end), remaining_samples, replace=False)
        ind_abs_list.append(ind_one)

        ind_abs = np.concatenate(ind_abs_list)

        ind_this1 = ind_abs + np.sum(n_per_session) * cur_sub1 + n_per_session_cum[cur_session]
        ind_this2 = ind_abs + np.sum(n_per_session) * cur_sub2 + n_per_session_cum[cur_session]

        return ind_this1, ind_this2

    def __len__(self):
        return int(self.n_pairs)

    def __iter__(self):
        # 如果所有数据集都没有可配对样本，直接停止
        if self.max_n_pairs == 0:
            return
            yield  # 生成器语义兼容（不会执行到）

        for _ in range(self.n_pairs):
            index = random.randint(0, self.max_n_pairs - 1)
            idx_list = []
            for idx in range(self.num_datasets):
                pairs = self.pairs_list[idx]
                n_pairs = self.n_pairs_list[idx]

                # 如果该数据集没有可配对样本，给一个空 list 占位
                if n_pairs == 0:
                    idx_list.append([])
                    continue

                pair = pairs[index % n_pairs]
                idx_1, idx_2 = self.get_sample(dataset_idx=idx, subsession_pair=pair)
                idx_combined = np.concatenate((idx_1, idx_2))
                idx_list.append(list(idx_combined.astype(int)))

            yield idx_list


class MultiDataset(Dataset):
    """
    将多个 EEG_Dataset 组合在一起。采样器传入的 idx_list
    是“每个数据集的一组索引列表”。
    """
    def __init__(self, datasets):
        self.datasets = datasets
        self.lens = [len(dataset) for dataset in datasets]
        print('Dataset lengths:', self.lens)

    def __len__(self):
        # 对齐为最大长度即可（采样器控制选取）
        return max(self.lens) if len(self.lens) else 0

    def __getitem__(self, idx_list):
        data_list, label_list = [], []
        for idxs, dataset in zip(idx_list, self.datasets):
            if not idxs:  # 该数据集无样本，占位
                data_list.append(torch.empty(0))
                label_list.append(torch.empty(0, dtype=torch.long))
                continue
            data = [dataset[i][0] for i in idxs]
            label = [dataset[i][1] for i in idxs]
            data = torch.stack(data)
            label = torch.stack(label)
            data_list.append(data)
            label_list.append(label)
        return data_list, label_list


class MultiDataModule(pl.LightningDataModule):
    """
    多数据集 DataModule
    - n_folds == 1 时，不创建 val_dataloader（Lightning 会跳过验证）
    - DataLoader 使用 spawn & 非持久化 worker，退出更干净
    """
    def __init__(self, data_cfg_list, fold, n_folds, n_pairs=256, num_workers=8, device='cpu', sub_list_pre=None):
        super().__init__()
        self.device = device
        self.n_pairs = int(n_pairs)
        self.num_workers = int(num_workers)
        self.data_list = data_cfg_list
        self.fold = int(fold)
        self.n_folds = int(n_folds)
        self.sub_list_pre = [None] * len(data_cfg_list) if sub_list_pre is None else sub_list_pre

        self.train_subs_list = []
        self.val_subs_list = []
        self.trainsets = []
        self.valsets = []
        self.trainset = None
        self.valset = None

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # 先确定各数据集的 train/val 被试划分
        self.train_subs_list.clear()
        self.val_subs_list.clear()

        for index, data_cfg in enumerate(self.data_list):
            if self.sub_list_pre[index] is None:
                train_subs, val_subs = get_train_subs(data_cfg.n_subs, self.fold, self.n_folds)
            else:
                train_subs, val_subs = self.sub_list_pre[index]
            self.train_subs_list.append(train_subs)
            self.val_subs_list.append(val_subs)
            print(f'Dataset {index} \nTrain subs: {train_subs} \nVal subs: {val_subs}\n')

        # 组装 train/val 数据集
        if stage in ("fit", None):
            self.trainsets = []
            self.valsets = []
            for idx, data_cfg in enumerate(self.data_list):
                trainset = EEG_Dataset(data_cfg, train_subs=self.train_subs_list[idx], mods='train', sliced=False)
                valset = EEG_Dataset(data_cfg, val_subs=self.val_subs_list[idx], mods='val', sliced=False)
                self.trainsets.append(trainset)
                self.valsets.append(valset)
            self.trainset = MultiDataset(self.trainsets)
            self.valset = MultiDataset(self.valsets)

        if stage == "validate":
            self.valsets = []
            for idx, data_cfg in enumerate(self.data_list):
                valset = EEG_Dataset(data_cfg, val_subs=self.val_subs_list[idx], mods='val', sliced=False)
                self.valsets.append(valset)
            self.valset = MultiDataset(self.valsets)

    # --- DataLoader 通用参数（spawn + 非持久化） ---
    def _dl_kwargs(self):
        kw = dict(
            batch_size=1,                     # 采样器每次返回“一组索引列表”，所以这里就是 1
            shuffle=False,                    # 有 sampler，不需要 shuffle
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False,         # 关闭，退出更干净
            multiprocessing_context="spawn",  # 关键：spawn 上下文
            drop_last=False,
        )
        if self.num_workers and self.num_workers > 0:
            # 只有多进程才设置 prefetch_factor，避免 PyTorch 警告
            kw["prefetch_factor"] = 2
        return kw

    def train_dataloader(self):
        # 构造训练采样器（各数据集都有 train_subs）
        self.trainsampler = EEGSampler(datasets=self.trainsets, n_pairs=self.n_pairs)
        return DataLoader(self.trainset, sampler=self.trainsampler, **self._dl_kwargs())

    def val_dataloader(self):
        # 若所有数据集的 val_subs 都为空（如 n_folds == 1），直接返回 []，Lightning 会跳过
        total_val_subs = sum(len(vs) for vs in self.val_subs_list)
        if total_val_subs == 0:
            return []

        # 否则正常构造验证采样器（可适当减少配对数）
        n_pairs_val = max(1, self.n_pairs // 4)
        self.valsampler = EEGSampler(datasets=self.valsets, n_pairs=n_pairs_val)

        # 如果 val 数据集里**确实**也没有任何可配对样本（极端情况），也返回 []
        if self.valsampler.max_n_pairs == 0:
            return []

        return DataLoader(self.valset, sampler=self.valsampler, **self._dl_kwargs())
