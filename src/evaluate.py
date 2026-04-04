# src/evaluate.py — 模型评估函数（测试集指标、按RAFT类型分类评估、异常值分析）
# 注意：matplotlib仅在绘图函数内部延迟导入，保证模块本身可在无显示环境下导入
import numpy as np

from utils.metrics import r2_score_np, rmse_np, mae_np
from utils.split import RAFT_TYPES


def compute_test_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """计算测试集的R²、RMSE、MAE指标（每个输出列独立计算）。

    Args:
        y_true: 真实值数组，形状 (N, 3)，列顺序 [log10_Ctr, inhibition, retardation]
        y_pred: 预测值数组，形状 (N, 3)

    Returns:
        dict含 'r2'、'rmse'、'mae'，各为长度3的列表
        对应三个输出: [log10_Ctr, inhibition_period, retardation_factor]
    """
    r2_list, rmse_list, mae_list = [], [], []

    for col in range(3):
        r2_list.append(r2_score_np(y_true[:, col], y_pred[:, col]))
        rmse_list.append(rmse_np(y_true[:, col], y_pred[:, col]))
        mae_list.append(mae_np(y_true[:, col], y_pred[:, col]))

    return {'r2': r2_list, 'rmse': rmse_list, 'mae': mae_list}


def per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_ids: np.ndarray,
) -> dict:
    """按RAFT类型分类计算R²、RMSE、MAE。

    Args:
        y_true: 真实值数组，形状 (N, 3)
        y_pred: 预测值数组，形状 (N, 3)
        class_ids: RAFT类型标识，形状 (N,)，值0-3对应RAFT_TYPES索引

    Returns:
        dict，键为RAFT类型名称字符串，值为含 'r2'/'rmse'/'mae' 的指标字典
        仅包含至少有1个样本的类型
    """
    result = {}

    for class_id in range(len(RAFT_TYPES)):
        mask = class_ids == class_id
        if mask.sum() == 0:
            continue
        metrics = compute_test_metrics(y_true[mask], y_pred[mask])
        result[RAFT_TYPES[class_id]] = metrics

    return result


def compute_outlier_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """计算异常值统计（D-21: |pred - true| > 2*std(residuals)）。

    Args:
        y_true: 真实值数组，形状 (N, 3)
        y_pred: 预测值数组，形状 (N, 3)

    Returns:
        dict含 'outlier_fraction'，为长度3的列表
        对应三个输出 [log10_Ctr, inhibition_period, retardation_factor]
    """
    residuals = y_pred - y_true   # (N, 3)
    outlier_fractions = []

    for col in range(3):
        res = residuals[:, col]
        threshold = 2.0 * np.std(res)
        is_outlier = np.abs(res) > threshold
        outlier_fractions.append(float(is_outlier.mean()))

    return {'outlier_fraction': outlier_fractions}


OUTPUT_NAMES = ['log10_Ctr', 'inhibition_period', 'retardation_factor']


def run_full_evaluation(model, test_loader, device, figures_dir='figures'):
    """运行完整评估：推理+指标+绘图。"""
    import torch
    import os
    os.makedirs(figures_dir, exist_ok=True)

    y_true_list, y_pred_list = [], []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            y_true_list.append(y.cpu().numpy())
            y_pred_list.append(pred.cpu().numpy())

    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)

    # 从dataset的indices提取class_id
    class_ids = np.array([entry[0] for entry in test_loader.dataset.indices])

    # 整体指标
    overall = compute_test_metrics(y_true, y_pred)

    # 按Ctr范围分段
    log10_ctr_true = y_true[:, 0]
    segmented = {
        'low_Ctr': compute_test_metrics(y_true[log10_ctr_true < -2], y_pred[log10_ctr_true < -2]),
        'mid_Ctr': compute_test_metrics(y_true[(log10_ctr_true >= -2) & (log10_ctr_true < 0)],
                                       y_pred[(log10_ctr_true >= -2) & (log10_ctr_true < 0)]),
        'high_Ctr': compute_test_metrics(y_true[log10_ctr_true >= 0], y_pred[log10_ctr_true >= 0]),
    }

    # 按RAFT类型
    by_class = per_class_metrics(y_true, y_pred, class_ids)

    # 绘图
    _plot_parity(y_true, y_pred, figures_dir)
    _plot_parity_by_class(y_true, y_pred, class_ids, figures_dir)
    _plot_residuals(y_true, y_pred, figures_dir)

    return {'overall': overall, 'segmented': segmented, 'by_class': by_class}


def _plot_parity(y_true, y_pred, figures_dir):
    """绘制整体parity图。"""
    import matplotlib.pyplot as plt
    for i, name in enumerate(OUTPUT_NAMES):
        plt.figure(figsize=(6, 6))
        plt.scatter(y_true[:, i], y_pred[:, i], alpha=0.3, s=1)
        plt.plot([y_true[:, i].min(), y_true[:, i].max()],
                 [y_true[:, i].min(), y_true[:, i].max()], 'r--', lw=2)
        plt.xlabel(f'True {name}')
        plt.ylabel(f'Predicted {name}')
        plt.title(f'Parity Plot: {name}')
        plt.tight_layout()
        plt.savefig(f'{figures_dir}/parity_{name}.png', dpi=150)
        plt.close()


def _plot_parity_by_class(y_true, y_pred, class_ids, figures_dir):
    """按RAFT类型绘制parity图。"""
    import matplotlib.pyplot as plt
    import os
    os.makedirs(f'{figures_dir}/parity_by_class', exist_ok=True)
    for class_id, raft_type in enumerate(RAFT_TYPES):
        mask = class_ids == class_id
        if mask.sum() == 0:
            continue
        for i, name in enumerate(OUTPUT_NAMES):
            plt.figure(figsize=(6, 6))
            plt.scatter(y_true[mask, i], y_pred[mask, i], alpha=0.3, s=1)
            plt.plot([y_true[mask, i].min(), y_true[mask, i].max()],
                     [y_true[mask, i].min(), y_true[mask, i].max()], 'r--', lw=2)
            plt.xlabel(f'True {name}')
            plt.ylabel(f'Predicted {name}')
            plt.title(f'{raft_type}: {name}')
            plt.tight_layout()
            plt.savefig(f'{figures_dir}/parity_by_class/{raft_type}_{name}.png', dpi=150)
            plt.close()


def _plot_residuals(y_true, y_pred, figures_dir):
    """绘制残差直方图。"""
    import matplotlib.pyplot as plt
    residuals = y_pred - y_true
    for i, name in enumerate(OUTPUT_NAMES):
        plt.figure(figsize=(8, 5))
        plt.hist(residuals[:, i], bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel(f'Residual ({name})')
        plt.ylabel('Frequency')
        plt.title(f'Residual Distribution: {name}')
        plt.tight_layout()
        plt.savefig(f'{figures_dir}/residuals_{name}.png', dpi=150)
        plt.close()
