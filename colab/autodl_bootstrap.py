#!/usr/bin/env python3
"""
AutoDL Bootstrap一键脚本 — 零外部依赖，自包含全部逻辑。

仅依赖: torch, numpy, scipy, h5py (AutoDL预装)。
不需要 src/ 目录，不需要 import 项目代码。

用法:
    # === 一键全流程 (推荐在tmux中运行) ===
    tmux new -s bootstrap
    cd /root/autodl-tmp/ViT-Ctr
    python colab/autodl_bootstrap.py
    # Ctrl+B, D 断开; tmux attach -t bootstrap 重连

    # === 断点续训 (自动检测已完成的迭代数) ===
    python colab/autodl_bootstrap.py --resume

    # === 仅校准+验证 (bootstrap已完成时) ===
    python colab/autodl_bootstrap.py --calibrate_only

产物:
    checkpoints/bootstrap_heads.pth   — 200个Bootstrap头权重
    checkpoints/calibration.json      — 校准因子和覆盖率
    checkpoints/bootstrap_summary.json — 运行摘要报告
    checkpoints/bootstrap_results.tar.gz — 打包好的产物（方便下载）
"""
import argparse
import copy
import json
import logging
import os
import sys
import tarfile
import time
from datetime import datetime

import h5py
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import f as fdist
from torch.utils.data import TensorDataset, DataLoader


# ═══════════════════════════════════════════════════════
# 常量
# ═══════════════════════════════════════════════════════

N_BOOTSTRAP = 200
N_EPOCHS_PER_HEAD = 5
BOOTSTRAP_LR = 1e-3
BOOTSTRAP_BATCH_SIZE = 256
SEED = 42
P_OUTPUTS = 3

DEFAULT_H5_DIR = '/root/autodl-tmp/data'
DEFAULT_CKPT_DIR = '/root/autodl-tmp/checkpoints'
RAFT_TYPES = ['dithioester', 'trithiocarbonate', 'xanthate', 'dithiocarbamate']

# ═══════════════════════════════════════════════════════
# 模型定义 (逐字复制自 src/model.py)
# ═══════════════════════════════════════════════════════

class SimpViT(nn.Module):
    def __init__(self, img_size=64, patch_size=16, num_outputs=3,
                 hidden_size=64, num_layers=2, num_heads=4):
        super(SimpViT, self).__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.patch_embedding = nn.Conv2d(in_channels=2, out_channels=hidden_size,
                                         kernel_size=patch_size, stride=patch_size)
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, hidden_size))
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, num_outputs)

    def forward(self, x):
        patches = self.patch_embedding(x)
        patches = patches.permute(0, 2, 3, 1)
        patches = patches.view(x.size(0), -1, patches.size(-1))
        patches = patches + self.position_embedding[:, :patches.size(1)]
        patches = patches.permute(1, 0, 2)
        encoded_patches = self.transformer_encoder(patches)
        class_token = encoded_patches.mean(dim=0)
        return self.fc(class_token)


# ═══════════════════════════════════════════════════════
# 损失函数 (复制自 src/train.py)
# ═══════════════════════════════════════════════════════

def weighted_mse_loss(pred, target, weights=(2.0, 0.5, 0.5)):
    """加权MSE损失: [log10_Ctr, inhibition, retardation] = (2.0, 0.5, 0.5)"""
    w = torch.tensor(weights, dtype=pred.dtype, device=pred.device)
    sq_err = (pred - target) ** 2
    per_output_mse = sq_err.mean(dim=0)
    return (w * per_output_mse).sum()


# ═══════════════════════════════════════════════════════
# 数据划分 (内联自 src/utils/split.py)
# ═══════════════════════════════════════════════════════

def build_stratified_indices(h5_paths, val_frac=0.10, test_frac=0.10, seed=42):
    """按log10(Ctr)分层划分数据集，返回(train_idx, val_idx, test_idx)。
    每个索引元素为 (file_idx, sample_idx, class_id) 元组。
    """
    all_indices = []
    all_log10_ctr = []
    for file_idx, path in enumerate(h5_paths):
        with h5py.File(path, 'r') as f:
            log10_ctr = f['labels'][:, 0]
            n = len(log10_ctr)
            all_indices.extend([(file_idx, i, file_idx) for i in range(n)])
            all_log10_ctr.extend(log10_ctr.tolist())

    all_log10_ctr = np.array(all_log10_ctr)
    bins = np.arange(-2.0, 4.5, 0.5)
    bin_ids = np.digitize(all_log10_ctr, bins) - 1

    train_idx, val_idx, test_idx = [], [], []
    rng = np.random.default_rng(seed)
    for bin_id in range(len(bins) - 1):
        mask = np.where(bin_ids == bin_id)[0]
        if len(mask) == 0:
            continue
        rng.shuffle(mask)
        n_val = max(1, int(len(mask) * val_frac))
        n_test = max(1, int(len(mask) * test_frac))
        val_idx.extend([all_indices[i] for i in mask[:n_val]])
        test_idx.extend([all_indices[i] for i in mask[n_val:n_val + n_test]])
        train_idx.extend([all_indices[i] for i in mask[n_val + n_test:]])
    return train_idx, val_idx, test_idx


# ═══════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════

def preload_to_ram(h5_paths, indices, label=""):
    """一次性把所有样本加载到内存，返回 (fp_tensor, lbl_tensor)。"""
    desc = f"[preload{' ' + label if label else ''}]"
    print(f"{desc} 预加载 {len(indices)} 个样本到内存...")
    t0 = time.time()

    handles = [h5py.File(p, 'r') for p in h5_paths]
    n = len(indices)
    fps = np.empty((n, 2, 64, 64), dtype=np.float32)
    lbls = np.empty((n, 3), dtype=np.float32)

    for i, entry in enumerate(indices):
        file_idx, sample_idx = entry[0], entry[1]
        fp = handles[file_idx]['fingerprints'][sample_idx]   # (64,64,2)
        fps[i] = fp.transpose(2, 0, 1)                       # (2,64,64)
        lbls[i] = handles[file_idx]['labels'][sample_idx]     # (3,)

    for h in handles:
        h.close()

    elapsed = time.time() - t0
    mem_gb = (fps.nbytes + lbls.nbytes) / 1e9
    print(f"{desc} 完成: {elapsed:.1f}s, 内存占用 {mem_gb:.2f} GB")
    return torch.from_numpy(fps), torch.from_numpy(lbls)


# ═══════════════════════════════════════════════════════
# Bootstrap核心
# ═══════════════════════════════════════════════════════

def freeze_backbone(model):
    """冻结除fc层以外的所有参数。"""
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith('fc')
    return model


def run_bootstrap(model, train_fps, train_lbls, *, n_bootstrap=N_BOOTSTRAP,
                  n_epochs=N_EPOCHS_PER_HEAD, lr=BOOTSTRAP_LR,
                  batch_size=BOOTSTRAP_BATCH_SIZE, device='cpu', seed=SEED,
                  save_dir=None, resume=False, progress_callback=None):
    """轻量Bootstrap: 冻结backbone，n_bootstrap次有放回重采样 + fc头微调。"""
    base_state = copy.deepcopy(model.state_dict())
    n_samples = len(train_fps)
    rng = np.random.default_rng(seed)

    heads = []
    start_iter = 0
    heads_path = os.path.join(save_dir, 'bootstrap_heads.pth') if save_dir else None
    progress_path = os.path.join(save_dir, 'bootstrap_progress.json') if save_dir else None

    if resume and progress_path and os.path.exists(progress_path):
        with open(progress_path, 'r') as fp:
            progress = json.load(fp)
        start_iter = progress['completed']
        if heads_path and os.path.exists(heads_path):
            saved = torch.load(heads_path, map_location='cpu', weights_only=False)
            heads = saved['heads'][:start_iter]
        for _ in range(start_iter):
            rng.choice(n_samples, size=n_samples, replace=True)
        print(f"[bootstrap] 从第 {start_iter}/{n_bootstrap} 次迭代继续")

    remaining = n_bootstrap - start_iter
    if remaining == 0:
        print("[bootstrap] 已全部完成，跳过训练")
        return _build_result(heads, base_state, n_bootstrap)

    use_pin = (device != 'cpu' and str(device) != 'cpu')
    num_workers = 4 if use_pin else 0

    for i in range(start_iter, n_bootstrap):
        t0 = time.time()
        boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
        boot_ds = TensorDataset(train_fps[boot_idx], train_lbls[boot_idx])
        boot_loader = DataLoader(boot_ds, batch_size=batch_size, shuffle=True,
                                 num_workers=num_workers, pin_memory=use_pin)

        model.load_state_dict(copy.deepcopy(base_state))
        freeze_backbone(model)
        model.to(device)
        opt = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=lr)

        model.train()
        epoch_loss, n_steps = 0.0, 0
        for _ in range(n_epochs):
            for fp_batch, lbl_batch in boot_loader:
                fp_batch = fp_batch.to(device, non_blocking=True)
                lbl_batch = lbl_batch.to(device, non_blocking=True)
                opt.zero_grad()
                pred = model(fp_batch)
                loss = weighted_mse_loss(pred, lbl_batch)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                n_steps += 1

        head_state = {
            k: v.clone().cpu()
            for k, v in model.state_dict().items() if k.startswith('fc')
        }
        heads.append(head_state)

        elapsed = time.time() - t0
        avg_loss = epoch_loss / max(n_steps, 1)
        if progress_callback:
            progress_callback(i, n_bootstrap, elapsed, avg_loss)
        if save_dir:
            _save_progress(heads, base_state, n_bootstrap, i + 1,
                           heads_path, progress_path)

    model.load_state_dict(base_state)
    model.to(device)
    return _build_result(heads, base_state, n_bootstrap)


def _build_result(heads, base_state, n_bootstrap):
    return {'heads': heads, 'base_model_state_dict': base_state,
            'n_bootstrap': n_bootstrap}


def _save_progress(heads, base_state, n_bootstrap, completed,
                   heads_path, progress_path):
    torch.save({'heads': heads, 'base_model_state_dict': base_state,
                'n_bootstrap': n_bootstrap}, heads_path)
    with open(progress_path, 'w') as fp:
        json.dump({'completed': completed, 'total': n_bootstrap}, fp)


# ═══════════════════════════════════════════════════════
# 校准
# ═══════════════════════════════════════════════════════

def collect_ensemble_predictions(model, fps, lbls, bootstrap_ckpt, *,
                                 device='cpu', batch_size=256):
    """用全部heads在数据集上批量推理。"""
    base_state = bootstrap_ckpt['base_model_state_dict']
    heads = bootstrap_ckpt['heads']
    n_heads = len(heads)

    use_pin = (device != 'cpu' and str(device) != 'cpu')
    ds = TensorDataset(fps, lbls)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=4 if use_pin else 0, pin_memory=use_pin)

    model.eval()
    all_preds = []
    with torch.no_grad():
        for hi, head_state in enumerate(heads):
            full_state = copy.deepcopy(base_state)
            full_state.update(head_state)
            model.load_state_dict(full_state)
            model.to(device)
            batch_preds = []
            for fp_batch, _ in loader:
                fp_batch = fp_batch.to(device, non_blocking=True)
                batch_preds.append(model(fp_batch).cpu().numpy())
            all_preds.append(np.vstack(batch_preds))
            if (hi + 1) % 50 == 0:
                print(f"  [ensemble] {hi + 1}/{n_heads} heads processed")

    model.load_state_dict(copy.deepcopy(base_state))
    model.to(device)
    all_preds = np.stack(all_preds)
    return {'all_preds': all_preds, 'mean_pred': all_preds.mean(axis=0),
            'y_true': lbls.numpy()}


def compute_jci(cov_matrix, n, p=P_OUTPUTS):
    """F分布95%联合置信区间半宽度。
    注意: src/bootstrap.py 中有此函数的副本（本地推理用）。
    """
    dfd = n - p
    if dfd <= 0:
        raise ValueError(f"dfd={dfd} <= 0. Need n > p (n={n}, p={p}).")
    f_val = fdist.ppf(0.95, dfn=p, dfd=dfd)
    return np.sqrt(np.diag(cov_matrix) * p * f_val / dfd)


def compute_half_widths(all_preds):
    """从集成预测矩阵计算每个样本的JCI半宽度。"""
    n_heads = all_preds.shape[0]
    p = all_preds.shape[2]
    dfd = n_heads - p
    f_val = fdist.ppf(0.95, dfn=p, dfd=dfd)
    per_sample_var = all_preds.var(axis=0, ddof=1)
    return np.sqrt(per_sample_var * p * f_val / dfd)


def compute_coverage(y_true, y_pred_mean, half_widths):
    """计算每个输出的实际CI覆盖率。"""
    within = np.abs(y_pred_mean - y_true) <= half_widths
    return within.mean(axis=0)


def calibrate_coverage(y_true, y_pred_mean, half_widths, target=0.95):
    """事后校准: 二分搜索找最小标量因子使覆盖率 >= target。"""
    p = y_true.shape[1]
    cal_factors = []
    for i in range(p):
        lo, hi = 1.0, 100.0
        for _ in range(50):
            mid = (lo + hi) / 2
            cov = compute_coverage(
                y_true[:, i:i + 1], y_pred_mean[:, i:i + 1],
                half_widths[:, i:i + 1] * mid)
            if cov[0] >= target:
                hi = mid
            else:
                lo = mid
        cal_factors.append(float(hi))
    return cal_factors


def run_calibration(model, val_fps, val_lbls, bootstrap_ckpt, *,
                    device='cpu', batch_size=256):
    """完整校准流程: 集成推理 -> JCI -> 校准。"""
    print(f"[calibration] 在 {len(val_fps)} 个验证样本上计算...")
    t0 = time.time()

    ensemble = collect_ensemble_predictions(
        model, val_fps, val_lbls, bootstrap_ckpt,
        device=device, batch_size=batch_size)

    y_true = ensemble['y_true']
    mean_pred = ensemble['mean_pred']
    half_widths = compute_half_widths(ensemble['all_preds'])

    raw_coverage = compute_coverage(y_true, mean_pred, half_widths)
    cal_factors = calibrate_coverage(y_true, mean_pred, half_widths, target=0.95)
    calibrated_half = half_widths * np.array(cal_factors)
    final_coverage = compute_coverage(y_true, mean_pred, calibrated_half)

    elapsed = time.time() - t0
    print(f"[calibration] 完成: {elapsed:.1f}s")
    print(f"  Raw coverage:        {raw_coverage}")
    print(f"  Calibrated coverage: {final_coverage}")
    print(f"  Cal factors:         {cal_factors}")

    return {
        'cal_factors': cal_factors,
        'empirical_coverage_before': raw_coverage.tolist(),
        'empirical_coverage_after': final_coverage.tolist(),
        'n_val_samples': len(y_true),
        'target_coverage': 0.95,
    }


# ═══════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════

def fmt_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}min"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h{m:02d}m"


def print_banner(title, char='='):
    line = char * 60
    print(f"\n{line}\n  {title}\n{line}")


class ProgressTracker:
    """Bootstrap训练进度追踪器。"""
    def __init__(self, n_total):
        self.n_total = n_total
        self.iter_times = []
        self.wall_start = time.time()

    def __call__(self, iter_idx, n_total, elapsed, avg_loss):
        self.iter_times.append(elapsed)
        left = n_total - iter_idx - 1
        avg_iter_time = sum(self.iter_times) / len(self.iter_times)
        eta_seconds = left * avg_iter_time
        pct = (iter_idx + 1) / n_total * 100
        print(f"[bootstrap {iter_idx + 1:3d}/{n_total}] "
              f"{elapsed:.1f}s | loss={avg_loss:.6f} | "
              f"{pct:.0f}% | ETA: {fmt_time(eta_seconds)}")

    @property
    def total_wall_time(self):
        return time.time() - self.wall_start


# ═══════════════════════════════════════════════════════
# 验证与打包
# ═══════════════════════════════════════════════════════

def predict_with_uncertainty(model, fp_tensor, bootstrap_ckpt, cal_factors, device='cpu'):
    """单样本推理，返回 (mean, calibrated_half_width)。"""
    base_state = bootstrap_ckpt['base_model_state_dict']
    heads = bootstrap_ckpt['heads']
    n = len(heads)

    model.eval()
    fp_tensor = fp_tensor.to(device)
    predictions = []
    with torch.no_grad():
        for head_state in heads:
            full_state = copy.deepcopy(base_state)
            full_state.update(head_state)
            model.load_state_dict(full_state)
            model.to(device)
            pred = model(fp_tensor).cpu().numpy().squeeze()
            predictions.append(pred)

    predictions = np.array(predictions)
    mean_pred = predictions.mean(axis=0)
    cov_matrix = np.cov(predictions, rowvar=False)
    half_width = compute_jci(cov_matrix, n=n, p=P_OUTPUTS)
    calibrated_half_width = half_width * np.array(cal_factors)

    model.load_state_dict(copy.deepcopy(base_state))
    model.to(device)
    return mean_pred, calibrated_half_width


def run_verification(model, test_fps, test_lbls, bootstrap_ckpt, cal_factors, device):
    """在少量测试样本上运行预测验证。"""
    n_demo = min(5, len(test_fps))
    output_names = ['log10_Ctr', 'inhibition_period', 'retardation_factor']
    print(f"[verify] 在 {n_demo} 个测试样本上验证推理...")

    all_ok = True
    for idx in range(n_demo):
        fp = test_fps[idx].unsqueeze(0)
        lbl = test_lbls[idx].numpy()
        mean_pred, half_width = predict_with_uncertainty(
            model, fp, bootstrap_ckpt, cal_factors, device=device)

        if np.any(np.isnan(mean_pred)) or np.any(np.isnan(half_width)):
            print(f"  [FAIL] 样本 {idx}: NaN in predictions!")
            all_ok = False
            continue
        if np.any(half_width <= 0):
            print(f"  [FAIL] 样本 {idx}: negative or zero half_width!")
            all_ok = False
            continue

        within = np.abs(mean_pred - lbl) <= half_width
        status = "OK" if all(within) else "WIDE"
        if idx < 3:
            print(f"  样本 {idx} [{status}]:")
            for i, name in enumerate(output_names):
                in_ci = "v" if within[i] else "x"
                print(f"    {name}: pred={mean_pred[i]:.4f} +/- {half_width[i]:.4f} | "
                      f"true={lbl[i]:.4f} [{in_ci}]")

    if all_ok:
        print(f"[verify] 通过! 所有 {n_demo} 个样本推理正常")
    else:
        print(f"[verify] 警告: 部分样本推理异常，请检查")
    return all_ok


def package_results(ckpt_dir):
    """将所有产物打包为 tar.gz。"""
    tar_path = os.path.join(ckpt_dir, 'bootstrap_results.tar.gz')
    files_to_pack = ['bootstrap_heads.pth', 'calibration.json',
                     'bootstrap_summary.json', 'bootstrap_progress.json']
    print(f"[package] 打包产物到 {tar_path}...")
    with tarfile.open(tar_path, 'w:gz') as tar:
        for fname in files_to_pack:
            fpath = os.path.join(ckpt_dir, fname)
            if os.path.exists(fpath):
                tar.add(fpath, arcname=fname)
                size_mb = os.path.getsize(fpath) / 1e6
                print(f"  + {fname} ({size_mb:.1f} MB)")
            else:
                print(f"  - {fname} (不存在，跳过)")
    tar_size = os.path.getsize(tar_path) / 1e6
    print(f"[package] 完成: {tar_path} ({tar_size:.1f} MB)")
    return tar_path


# ═══════════════════════════════════════════════════════
# CLI入口
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="AutoDL Bootstrap一键脚本 (零依赖自包含)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python colab/autodl_bootstrap.py\n"
               "  python colab/autodl_bootstrap.py --resume\n"
               "  python colab/autodl_bootstrap.py --calibrate_only\n")
    parser.add_argument('--h5_dir', default=DEFAULT_H5_DIR)
    parser.add_argument('--ckpt_dir', default=DEFAULT_CKPT_DIR)
    parser.add_argument('--base_model', default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--calibrate_only', action='store_true')
    parser.add_argument('--batch_size', type=int, default=BOOTSTRAP_BATCH_SIZE)
    parser.add_argument('--skip_verify', action='store_true')
    parser.add_argument('--skip_package', action='store_true')
    args = parser.parse_args()

    total_start = time.time()
    print_banner("ViT-Ctr Bootstrap 一键脚本 (自包含)")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  H5数据: {args.h5_dir}")
    print(f"  Checkpoints: {args.ckpt_dir}")
    mode_str = '仅校准' if args.calibrate_only else '断点续训' if args.resume else '完整流程'
    print(f"  模式: {mode_str}")

    # ── Step 1: 环境和数据检查 ──
    print(f"\n>>> Step 1: 环境和数据检查")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e6:.0f} MB")
    else:
        print("  [WARN] 未检测到GPU! Bootstrap将非常慢。")

    h5_paths = [os.path.join(args.h5_dir, f'{t}.h5') for t in RAFT_TYPES]
    for p in h5_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"HDF5文件不存在: {p}")
    print(f"  HDF5数据: 全部 {len(h5_paths)} 个文件已就绪")

    os.makedirs(args.ckpt_dir, exist_ok=True)
    base_model_path = args.base_model or os.path.join(args.ckpt_dir, 'best_model.pth')
    if not os.path.exists(base_model_path):
        raise FileNotFoundError(f"Base model不存在: {base_model_path}")
    print(f"  Base model: {base_model_path}")

    train_idx, val_idx, test_idx = build_stratified_indices(h5_paths, seed=SEED)
    print(f"  数据划分: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    train_fps, train_lbls = preload_to_ram(h5_paths, train_idx, label="train")
    val_fps, val_lbls = preload_to_ram(h5_paths, val_idx, label="val")
    if not args.skip_verify:
        test_fps, test_lbls = preload_to_ram(h5_paths, test_idx[:20], label="test(20)")

    model = SimpViT(num_outputs=3).to(device)
    ckpt = torch.load(base_model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"  模型加载完成 (epoch {ckpt.get('epoch', '?')}, val_loss={ckpt.get('val_loss', '?')})")

    # ── Step 2: Bootstrap训练 ──
    if not args.calibrate_only:
        print(f"\n>>> Step 2: Bootstrap训练 ({N_BOOTSTRAP}次迭代, 每次{N_EPOCHS_PER_HEAD}epochs)")
        tracker = ProgressTracker(N_BOOTSTRAP)
        bootstrap_ckpt = run_bootstrap(
            model, train_fps, train_lbls,
            n_bootstrap=N_BOOTSTRAP, n_epochs=N_EPOCHS_PER_HEAD,
            lr=BOOTSTRAP_LR, batch_size=args.batch_size,
            device=device, seed=SEED, save_dir=args.ckpt_dir,
            resume=args.resume, progress_callback=tracker)
        print(f"\n[bootstrap] 全部完成! 总耗时: {fmt_time(tracker.total_wall_time)}")
    else:
        print(f"\n>>> Step 2: Bootstrap训练 (跳过 — 使用已有结果)")
        heads_path = os.path.join(args.ckpt_dir, 'bootstrap_heads.pth')
        if not os.path.exists(heads_path):
            raise FileNotFoundError(f"bootstrap_heads.pth不存在: {heads_path}")
        bootstrap_ckpt = torch.load(heads_path, map_location='cpu', weights_only=False)
        print(f"  加载已有bootstrap: {len(bootstrap_ckpt['heads'])} heads")

    # ── Step 3: 校准 ──
    print(f"\n>>> Step 3: 验证集校准")
    cal_result = run_calibration(
        model, val_fps, val_lbls, bootstrap_ckpt,
        device=device, batch_size=args.batch_size)
    cal_path = os.path.join(args.ckpt_dir, 'calibration.json')
    with open(cal_path, 'w') as f:
        json.dump(cal_result, f, indent=2)
    print(f"  保存到 {cal_path}")

    # ── Step 4: 验证推理 (可选) ──
    verify_ok = True
    if not args.skip_verify:
        print(f"\n>>> Step 4: 推理验证")
        verify_ok = run_verification(
            model, test_fps, test_lbls, bootstrap_ckpt,
            cal_result['cal_factors'], device)

    # ── Step 5: 打包 (可选) ──
    tar_path = None
    if not args.skip_package:
        print(f"\n>>> Step 5: 产物打包")
        total_elapsed = time.time() - total_start
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_time_seconds': round(total_elapsed, 1),
            'total_time_human': fmt_time(total_elapsed),
            'device': str(device),
            'gpu_name': torch.cuda.get_device_name() if device.type == 'cuda' else None,
            'n_bootstrap': N_BOOTSTRAP,
            'n_epochs_per_head': N_EPOCHS_PER_HEAD,
            'batch_size': args.batch_size,
            'n_train': len(train_idx),
            'n_val': len(val_idx),
            'calibration': cal_result,
            'verification_passed': verify_ok,
            'mode': mode_str,
        }
        summary_path = os.path.join(args.ckpt_dir, 'bootstrap_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        tar_path = package_results(args.ckpt_dir)

    # ── 最终摘要 ──
    total_elapsed = time.time() - total_start
    print_banner("全流程完成!")
    print(f"  总耗时: {fmt_time(total_elapsed)}")
    print(f"  校准覆盖率: {cal_result['empirical_coverage_after']}")
    print(f"  校准因子:   {cal_result['cal_factors']}")
    if not args.skip_verify:
        print(f"  验证推理:   {'通过' if verify_ok else '异常'}")
    print()
    print("  产物文件:")
    for fname in ['bootstrap_heads.pth', 'calibration.json', 'bootstrap_summary.json']:
        fpath = os.path.join(args.ckpt_dir, fname)
        if os.path.exists(fpath):
            size = os.path.getsize(fpath) / 1e6
            print(f"    {fpath} ({size:.1f} MB)")
    if tar_path and os.path.exists(tar_path):
        size = os.path.getsize(tar_path) / 1e6
        print(f"\n  一键下载包: {tar_path} ({size:.1f} MB)")
    print()
    print("  下载命令 (在本地Windows终端运行):")
    print(f'    scp -P <PORT> root@<HOST>:{args.ckpt_dir}/bootstrap_results.tar.gz .')
    print()

    for i, cov in enumerate(cal_result['empirical_coverage_after']):
        if cov < 0.95:
            names = ['log10_Ctr', 'inhibition', 'retardation']
            print(f"  [WARN] {names[i]} 校准后覆盖率 {cov:.3f} < 0.95!")
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[中断] 用户中断。如需续训，添加 --resume 参数重新运行。")
        sys.exit(1)
    except Exception as e:
        print(f"\n[错误] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
