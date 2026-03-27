# src/evaluate.py — 模型评估函数（测试集指标、按RAFT类型分类评估、异常值分析）
# 注意：matplotlib仅在绘图函数内部延迟导入，保证模块本身可在无显示环境下导入
import os
import numpy as np
import torch

from utils.metrics import r2_score_np, rmse_np, mae_np
from utils.split import RAFT_TYPES

OUTPUT_NAMES = ['log10_Ctr', 'inhibition_period', 'retardation_factor']


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


def run_inference(model, loader, device):
    """在DataLoader上运行模型推理，返回(y_true, y_pred, class_ids)。

    Args:
        model: 训练好的模型
        loader: DataLoader实例
        device: torch.device

    Returns:
        y_true: (N, 3) numpy数组
        y_pred: (N, 3) numpy数组
        class_ids: (N,) numpy数组，整数类型
    """
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for fp, lbl in loader:
            fp, lbl = fp.to(device), lbl.to(device)
            pred = model(fp)
            all_true.append(lbl.cpu().numpy())
            all_pred.append(pred.cpu().numpy())
    all_class_ids = np.array([entry[2] for entry in loader.dataset.indices])
    return np.vstack(all_true), np.vstack(all_pred), all_class_ids


def save_parity_plots(y_true, y_pred, out_dir):
    """生成3个总体parity图（每个输出一张），保存到out_dir/parity_{name}.png。"""
    import matplotlib.pyplot as plt
    from utils.visualization import parity_plot
    os.makedirs(out_dir, exist_ok=True)
    for i, name in enumerate(OUTPUT_NAMES):
        fig = parity_plot(y_true[:, i], y_pred[:, i], label=name)
        fig.savefig(os.path.join(out_dir, f'parity_{name}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)


def save_per_class_parity_plots(y_true, y_pred, class_ids, out_dir):
    """生成12个按RAFT类型的parity图（4类型×3输出），保存到out_dir/parity_by_class/。"""
    import matplotlib.pyplot as plt
    from utils.visualization import parity_plot
    class_dir = os.path.join(out_dir, 'parity_by_class')
    os.makedirs(class_dir, exist_ok=True)
    for cid, cname in enumerate(RAFT_TYPES):
        mask = class_ids == cid
        if mask.sum() == 0:
            continue
        for i, name in enumerate(OUTPUT_NAMES):
            fig = parity_plot(y_true[mask, i], y_pred[mask, i], label=f'{cname} {name}')
            fig.savefig(os.path.join(class_dir, f'{cname}_{name}.png'), dpi=150, bbox_inches='tight')
            plt.close(fig)


def save_residual_plots(y_true, y_pred, out_dir):
    """生成3个残差直方图（每个输出一张），保存到out_dir/residuals_{name}.png。"""
    import matplotlib.pyplot as plt
    from utils.visualization import residual_hist
    os.makedirs(out_dir, exist_ok=True)
    for i, name in enumerate(OUTPUT_NAMES):
        residuals = y_pred[:, i] - y_true[:, i]
        fig = residual_hist(residuals, label=name)
        fig.savefig(os.path.join(out_dir, f'residuals_{name}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)


def compute_segmented_metrics(y_true, y_pred):
    """按log10(Ctr)范围分段计算指标（D-18）。

    Args:
        y_true: 真实值数组，形状 (N, 3)
        y_pred: 预测值数组，形状 (N, 3)

    Returns:
        dict，键为 'low'/'mid'/'high'，值为指标字典
        low: log10_ctr ∈ [-2, 0), mid: [0, 2), high: [2, 4]
    """
    log10_ctr = y_true[:, 0]
    segments = {
        'low':  (log10_ctr >= -2.0) & (log10_ctr < 0.0),
        'mid':  (log10_ctr >= 0.0)  & (log10_ctr < 2.0),
        'high': (log10_ctr >= 2.0)  & (log10_ctr <= 4.0),
    }
    result = {}
    for seg_name, mask in segments.items():
        if mask.sum() == 0:
            continue
        result[seg_name] = compute_test_metrics(y_true[mask], y_pred[mask])
    return result


def run_full_evaluation(model, test_loader, device, figures_dir='figures'):
    """完整评估流程入口：计算所有指标并生成所有图表。

    Args:
        model: 训练好的模型
        test_loader: 测试集DataLoader
        device: torch.device
        figures_dir: 图表保存目录（默认'figures'）

    Returns:
        dict含 'overall'/'segmented'/'by_class'/'outliers' 四个键
    """
    y_true, y_pred, class_ids = run_inference(model, test_loader, device)

    overall = compute_test_metrics(y_true, y_pred)
    segmented = compute_segmented_metrics(y_true, y_pred)
    by_class = per_class_metrics(y_true, y_pred, class_ids)
    outliers = compute_outlier_stats(y_true, y_pred)

    save_parity_plots(y_true, y_pred, figures_dir)
    save_per_class_parity_plots(y_true, y_pred, class_ids, figures_dir)
    save_residual_plots(y_true, y_pred, figures_dir)

    results = {
        'overall': overall,
        'segmented': segmented,
        'by_class': by_class,
        'outliers': outliers,
    }

    print("=== Test Set Evaluation ===")
    for i, name in enumerate(OUTPUT_NAMES):
        print(f"  {name}: R²={overall['r2'][i]:.4f}, RMSE={overall['rmse'][i]:.4f}, MAE={overall['mae'][i]:.4f}")
    print(f"\n  Outlier fractions (|pred-true|>2σ): {outliers['outlier_fraction']}")
    print(f"\nFigures saved to: {figures_dir}/")

    print("\n[NOTE] Retardation factor R² near 0 for TTC/xanthate/dithiocarbamate is EXPECTED.")
    print("       These RAFT types have retardation_factor ≈ 1.0 (physically correct trivial prediction).")

    return results
