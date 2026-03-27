# Bootstrap不确定性量化 — 轻量级策略：冻结backbone，微调200个输出头
import copy
import json
import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from scipy.stats import f


def freeze_backbone(model):
    """冻结除fc层以外的所有参数 (D-12)。"""
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith('fc')
    return model


def run_bootstrap(model, train_loader, n_bootstrap=200, n_epochs=5, lr=1e-3, device='cpu', seed=42):
    """
    轻量Bootstrap: 冻结backbone，200次有放回重采样训练集，每次微调fc头5个epoch。
    D-12/D-13: n_bootstrap=200, n_epochs=5
    返回: {'heads': [state_dict × n_bootstrap], 'base_model_state_dict': ..., 'n_bootstrap': n_bootstrap, 'debug_mode': n_bootstrap < 50}
    """
    from train import weighted_mse_loss

    rng = np.random.default_rng(seed)
    base_state = copy.deepcopy(model.state_dict())
    heads = []

    all_items = list(range(len(train_loader.dataset)))

    for i in tqdm(range(n_bootstrap), desc='Bootstrap'):
        # 有放回重采样
        boot_indices = rng.choice(all_items, size=len(all_items), replace=True).tolist()
        from torch.utils.data import Subset, DataLoader as DL
        boot_subset = Subset(train_loader.dataset, boot_indices)
        boot_loader = DL(boot_subset, batch_size=train_loader.batch_size, shuffle=True, num_workers=0)

        # 从base state开始，冻结backbone
        model.load_state_dict(copy.deepcopy(base_state))
        model = freeze_backbone(model)
        model.to(device)
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

        # 微调 n_epochs epochs
        model.train()
        for _ in range(n_epochs):
            for fp, lbl in boot_loader:
                fp, lbl = fp.to(device), lbl.to(device)
                opt.zero_grad()
                pred = model(fp)
                loss = weighted_mse_loss(pred, lbl)
                loss.backward()
                opt.step()

        # 保存fc头的state dict
        head_state = {k: v.clone().cpu() for k, v in model.state_dict().items() if k.startswith('fc')}
        heads.append(head_state)

    # 恢复base state
    model.load_state_dict(base_state)
    model.to(device)

    return {
        'heads': heads,
        'base_model_state_dict': base_state,
        'n_bootstrap': n_bootstrap,
        'debug_mode': n_bootstrap < 50,
    }


def compute_jci(cov_matrix, n, p):
    """
    F分布95%联合置信区间半宽度 (D-15)。
    Direct port from ViT-RR deploy.py lines 157–167, with p=3 instead of p=2.
    cov_matrix: (p, p) covariance matrix from np.cov(predictions, rowvar=False)
    Returns: half_width array of shape (p,)
    """
    dfd = n - p
    if dfd > 0 and np.isfinite(cov_matrix).all():
        f_val = f.ppf(0.95, dfn=p, dfd=dfd)
        return np.sqrt(np.diag(cov_matrix) * p * f_val / dfd)
    else:
        raise ValueError(f"Invalid cov_matrix or dfd={dfd}. Use percentile fallback in caller.")


def compute_coverage(val_true, val_pred_mean, val_pred_half):
    """
    在验证集上计算每个输出的实际CI覆盖率。
    Returns: coverage array of shape (p,), each value in [0, 1].
    """
    within = np.abs(val_pred_mean - val_true) <= val_pred_half
    return within.mean(axis=0)


def calibrate_coverage(val_true, val_pred_mean, val_pred_half, target=0.95):
    """
    事后校准 (D-16): 找到最小标量因子使每个输出的CI覆盖率 >= target。
    Binary search per output. Returns list of 3 floats >= 1.0.
    """
    p = val_true.shape[1]
    cal_factors = []
    for i in range(p):
        lo, hi = 1.0, 100.0
        for _ in range(50):
            mid = (lo + hi) / 2
            coverage = compute_coverage(val_true[:, i:i+1],
                                        val_pred_mean[:, i:i+1],
                                        val_pred_half[:, i:i+1] * mid)
            if coverage[0] >= target:
                hi = mid
            else:
                lo = mid
        cal_factors.append(float(hi))
    return cal_factors


def predict_with_uncertainty(model, fp_tensor, bootstrap_ckpt, cal_factors, device='cpu'):
    """
    用200个Bootstrap头对单个样本预测，返回 (mean, calibrated_half_width)。
    fp_tensor: (1, 2, 64, 64) input
    bootstrap_ckpt: output of run_bootstrap (or loaded from bootstrap_heads.pth)
    cal_factors: list of 3 floats from calibrate_coverage
    Returns: (mean: np.ndarray (3,), half_width: np.ndarray (3,))
    """
    base_state = bootstrap_ckpt['base_model_state_dict']
    heads = bootstrap_ckpt['heads']
    n = len(heads)
    p = 3

    model.load_state_dict(copy.deepcopy(base_state))
    model.to(device)
    model.eval()
    fp_tensor = fp_tensor.to(device)

    predictions = []
    with torch.no_grad():
        for head_state in heads:
            full_state = copy.deepcopy(base_state)
            full_state.update(head_state)
            model.load_state_dict(full_state)
            pred = model(fp_tensor).cpu().numpy().squeeze()
            predictions.append(pred)

    predictions = np.array(predictions)
    mean_pred = predictions.mean(axis=0)
    cov_matrix = np.cov(predictions, rowvar=False)

    half_width = compute_jci(cov_matrix, n=n, p=p)
    cal_factors_arr = np.array(cal_factors)
    calibrated_half_width = half_width * cal_factors_arr

    return mean_pred, calibrated_half_width

