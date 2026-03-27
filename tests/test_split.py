# tests/test_split.py — TRN-02测试覆盖：分层划分和CombinedHDF5Dataset
"""
验证:
- 分层划分产生不重叠的train/val/test索引集
- 划分比例约为80/10/10
- CombinedHDF5Dataset每项返回 (2,64,64) float32张量和 (3,) float32标签
"""

import numpy as np
import pytest
import h5py
import torch

from utils.split import build_stratified_indices, RAFT_TYPES
from dataset import CombinedHDF5Dataset


# ============================================================
# Fixture: 临时HDF5文件（4个RAFT类型 × 200样本）
# ============================================================

@pytest.fixture
def tmp_h5_paths(tmp_path):
    """创建4个临时HDF5文件，每个含200个合成样本。

    标签[:, 0] = log10_Ctr，均匀分布在 [-2.0, 4.0] 加小噪声。
    指纹形状 (200, 64, 64, 2)，随机float32。
    """
    rng = np.random.default_rng(42)
    paths = []

    for i, raft_type in enumerate(RAFT_TYPES):
        path = tmp_path / f"{raft_type}.h5"
        with h5py.File(path, 'w') as f:
            n = 200
            # 指纹：随机float32，形状(n, 64, 64, 2)
            fingerprints = rng.random((n, 64, 64, 2), dtype=np.float32)
            f.create_dataset('fingerprints', data=fingerprints)

            # 标签：log10_Ctr均匀分布 + 小噪声
            log10_ctr = np.linspace(-2.0, 4.0, n) + rng.normal(0, 0.05, n)
            inhibition = rng.random(n).astype(np.float32) * 0.1
            retardation = rng.random(n).astype(np.float32) * 0.1 + 0.9
            labels = np.column_stack([
                log10_ctr.astype(np.float32),
                inhibition,
                retardation,
            ])
            f.create_dataset('labels', data=labels)

            # 元数据
            f.attrs['label_names'] = ['log10_Ctr', 'inhibition_period', 'retardation_factor']
            f.attrs['raft_type'] = raft_type

        paths.append(str(path))

    return paths


# ============================================================
# TRN-02: 分层划分无重叠
# ============================================================

def test_no_index_overlap(tmp_h5_paths):
    """分层划分产生不重叠的train/val/test索引集。"""
    train_idx, val_idx, test_idx = build_stratified_indices(
        tmp_h5_paths, val_frac=0.10, test_frac=0.10, seed=42
    )

    train_set = set(map(tuple, train_idx))
    val_set = set(map(tuple, val_idx))
    test_set = set(map(tuple, test_idx))

    # 三组互不重叠
    assert train_set & val_set == set(), f"train/val重叠: {len(train_set & val_set)}个样本"
    assert val_set & test_set == set(), f"val/test重叠: {len(val_set & test_set)}个样本"
    assert train_set & test_set == set(), f"train/test重叠: {len(train_set & test_set)}个样本"


# ============================================================
# TRN-02: 划分比例近似 80/10/10
# ============================================================

def test_split_ratio(tmp_h5_paths):
    """分层划分的比例应近似 80/10/10（允许±5%浮动）。"""
    train_idx, val_idx, test_idx = build_stratified_indices(
        tmp_h5_paths, val_frac=0.10, test_frac=0.10, seed=42
    )

    total = 4 * 200  # 800样本
    actual_total = len(train_idx) + len(val_idx) + len(test_idx)

    # 总样本数应覆盖全部（允许小幅偏差，因为空档的min(1)舍入）
    assert actual_total <= total, f"样本数 {actual_total} 超过总数 {total}"
    assert actual_total >= total * 0.90, f"样本数 {actual_total} 损失超过10%"

    train_frac = len(train_idx) / total
    val_frac = len(val_idx) / total
    test_frac = len(test_idx) / total

    assert 0.75 <= train_frac <= 0.85, f"训练集比例 {train_frac:.3f} 不在 [0.75, 0.85] 内"
    assert 0.08 <= val_frac <= 0.12, f"验证集比例 {val_frac:.3f} 不在 [0.08, 0.12] 内"
    assert 0.08 <= test_frac <= 0.12, f"测试集比例 {test_frac:.3f} 不在 [0.08, 0.12] 内"


# ============================================================
# CombinedHDF5Dataset: 每项形状正确
# ============================================================

def test_combined_dataset_item_shapes(tmp_h5_paths):
    """CombinedHDF5Dataset每项返回正确形状和dtype的张量。"""
    train_idx, _, _ = build_stratified_indices(
        tmp_h5_paths, val_frac=0.10, test_frac=0.10, seed=42
    )

    dataset = CombinedHDF5Dataset(h5_paths=tmp_h5_paths, indices=train_idx[:10])

    fp, lbl = dataset[0]

    # 指纹形状: (2, 64, 64) channel-first
    assert fp.shape == (2, 64, 64), f"指纹形状期望 (2, 64, 64), 实际 {fp.shape}"
    assert fp.dtype == torch.float32, f"指纹dtype期望 float32, 实际 {fp.dtype}"

    # 标签形状: (3,)
    assert lbl.shape == (3,), f"标签形状期望 (3,), 实际 {lbl.shape}"
    assert lbl.dtype == torch.float32, f"标签dtype期望 float32, 实际 {lbl.dtype}"
