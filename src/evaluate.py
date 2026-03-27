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
