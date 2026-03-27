# src/utils/split.py — 按log10(Ctr)分层划分训练/验证/测试集
import logging

import h5py
import numpy as np

RAFT_TYPES = ['dithioester', 'trithiocarbonate', 'xanthate', 'dithiocarbamate']


def build_stratified_indices(h5_paths, val_frac=0.10, test_frac=0.10, seed=42):
    """按log10(Ctr)分层划分数据集，返回(train_idx, val_idx, test_idx)。

    每个索引元素为 (file_idx, sample_idx, class_id) 元组。
    D-09: 按0.5单位一档分层; D-10: 80/10/10划分; D-11: 测试集覆盖全Ctr范围。

    Args:
        h5_paths: HDF5文件路径列表，顺序对应 RAFT_TYPES 索引
        val_frac: 验证集比例（默认0.10）
        test_frac: 测试集比例（默认0.10）
        seed: 随机种子（默认42）

    Returns:
        (train_idx, val_idx, test_idx) — 各为 (file_idx, sample_idx, class_id) 元组列表
    """
    all_indices = []
    all_log10_ctr = []

    for file_idx, path in enumerate(h5_paths):
        with h5py.File(path, 'r') as f:
            log10_ctr = f['labels'][:, 0]   # 确认列0 = log10_Ctr
            n = len(log10_ctr)
            all_indices.extend([(file_idx, i, file_idx) for i in range(n)])
            all_log10_ctr.extend(log10_ctr.tolist())

    all_log10_ctr = np.array(all_log10_ctr)
    bins = np.arange(-2.0, 4.5, 0.5)   # 12档: [-2,-1.5), ..., [3.5,4.0)
    bin_ids = np.digitize(all_log10_ctr, bins) - 1

    train_idx, val_idx, test_idx = [], [], []
    rng = np.random.default_rng(seed)

    for bin_id in range(len(bins) - 1):
        mask = np.where(bin_ids == bin_id)[0]
        if len(mask) == 0:
            logging.debug(
                f"分层划分: 第{bin_id}档 (log10Ctr [{bins[bin_id]:.1f}, {bins[bin_id+1]:.1f})) 为空 — 跳过"
            )
            continue
        rng.shuffle(mask)
        n_val = max(1, int(len(mask) * val_frac))
        n_test = max(1, int(len(mask) * test_frac))
        val_idx.extend([all_indices[i] for i in mask[:n_val]])
        test_idx.extend([all_indices[i] for i in mask[n_val:n_val + n_test]])
        train_idx.extend([all_indices[i] for i in mask[n_val + n_test:]])

    return train_idx, val_idx, test_idx
