# Bootstrap不确定性量化 — 本地推理接口
# 训练逻辑在 colab/autodl_bootstrap.py（自包含脚本），此文件只保留推理时需要的函数。
#
# 下游消费者:
#   - app.py (Streamlit): from src.bootstrap import predict_with_uncertainty
#   - src/literature_validation.py: from bootstrap import predict_with_uncertainty
import copy
import json

import numpy as np
import torch
from scipy.stats import f as fdist


P_OUTPUTS = 3  # 三个输出: log10_Ctr, inhibition_period, retardation_factor


def compute_jci(cov_matrix, n, p=P_OUTPUTS):
    """F分布95%联合置信区间半宽度 (D-15)。

    Direct port from ViT-RR deploy.py, with p=3 instead of p=2.
    注意: colab/autodl_bootstrap.py 中有此函数的副本（自包含需要）。

    Args:
        cov_matrix: (p, p) 协方差矩阵，来自 np.cov(predictions, rowvar=False)
        n: Bootstrap头数量
        p: 输出维度数 (=3)

    Returns:
        half_width: np.ndarray (p,)
    """
    dfd = n - p
    if dfd <= 0:
        raise ValueError(f"dfd={dfd} <= 0. Need n > p (n={n}, p={p}).")
    if not np.isfinite(cov_matrix).all():
        raise ValueError("cov_matrix contains non-finite values.")

    f_val = fdist.ppf(0.95, dfn=p, dfd=dfd)
    return np.sqrt(np.diag(cov_matrix) * p * f_val / dfd)


def predict_with_uncertainty(model, fp_tensor, bootstrap_ckpt, cal_factors, device='cpu'):
    """用Bootstrap头集成预测，返回 (mean, calibrated_half_width)。

    Args:
        model: SimpViT实例
        fp_tensor: (1, 2, 64, 64) 输入张量
        bootstrap_ckpt: dict with 'heads' and 'base_model_state_dict'
        cal_factors: 校准因子列表 [float x 3]
        device: 推理设备

    Returns:
        (mean_pred, calibrated_half_width): 各为 np.ndarray (3,)
    """
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

    predictions = np.array(predictions)  # (n, 3)
    mean_pred = predictions.mean(axis=0)
    cov_matrix = np.cov(predictions, rowvar=False)  # (3, 3), ddof=1

    half_width = compute_jci(cov_matrix, n=n, p=P_OUTPUTS)
    calibrated_half_width = half_width * np.array(cal_factors)

    # 恢复base state
    model.load_state_dict(copy.deepcopy(base_state))
    model.to(device)

    return mean_pred, calibrated_half_width


def load_calibration(path):
    """读取calibration.json，返回cal_factors列表。

    Args:
        path: calibration.json文件路径

    Returns:
        list[float]: 长度3的校准因子列表

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 格式不正确
    """
    with open(path, 'r') as f:
        data = json.load(f)
    if 'cal_factors' not in data:
        raise ValueError(f"calibration.json missing 'cal_factors' key: {path}")
    factors = data['cal_factors']
    if len(factors) != P_OUTPUTS:
        raise ValueError(f"Expected {P_OUTPUTS} cal_factors, got {len(factors)}")
    return factors
