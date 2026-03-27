# src/utils/metrics.py — 评估指标函数（无sklearn依赖）
import numpy as np


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算R²分数（决定系数），不依赖sklearn。

    Args:
        y_true: 真实值数组
        y_pred: 预测值数组

    Returns:
        R²值，当ss_tot=0时返回0.0
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def rmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算均方根误差（RMSE）。

    Args:
        y_true: 真实值数组
        y_pred: 预测值数组

    Returns:
        RMSE值
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算平均绝对误差（MAE）。

    Args:
        y_true: 真实值数组
        y_pred: 预测值数组

    Returns:
        MAE值
    """
    return float(np.mean(np.abs(y_true - y_pred)))
