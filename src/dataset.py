# src/dataset.py — 封装4个HDF5文件的PyTorch Dataset（延迟打开文件句柄）
import h5py
import torch
from torch.utils.data import Dataset


class CombinedHDF5Dataset(Dataset):
    """封装4个HDF5文件的PyTorch Dataset。

    文件句柄延迟打开（lazy），保证在num_workers>0时的fork安全性（Pitfall 1）。
    每个DataLoader worker进程在首次调用__getitem__时独立打开自己的文件句柄。

    Args:
        h5_paths: HDF5文件路径列表，顺序对应 RAFT_TYPES 索引
        indices: 样本索引列表，每项为 (file_idx, sample_idx, ...) 元组
    """

    def __init__(self, h5_paths, indices):
        self.h5_paths = h5_paths
        self.indices = indices
        self._handles = None   # 延迟打开，每个worker进程独立持有

    def __len__(self):
        return len(self.indices)

    def _get_handles(self):
        """延迟打开HDF5文件句柄（每个worker进程调用一次）。"""
        if self._handles is None:
            self._handles = [h5py.File(p, 'r') for p in self.h5_paths]
        return self._handles

    def __getitem__(self, idx):
        entry = self.indices[idx]
        file_idx, sample_idx = entry[0], entry[1]
        handles = self._get_handles()
        fp = handles[file_idx]['fingerprints'][sample_idx]   # (64, 64, 2) float32
        lbl = handles[file_idx]['labels'][sample_idx]        # (3,) float32
        # (64,64,2) channel-last → (2,64,64) channel-first（Conv2d必需）
        fp_tensor = torch.from_numpy(fp.transpose(2, 0, 1).copy())
        return fp_tensor, torch.from_numpy(lbl.copy())
