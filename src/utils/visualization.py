import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.metrics import r2_score_np


def parity_plot(y_true, y_pred, label, figsize=(5, 5)):
    """生成parity plot（预测值 vs 真实值散点图）"""
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(y_true, y_pred, alpha=0.3, s=5, rasterized=True)

    lim = (min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max()))
    ax.plot(lim, lim, 'r--', lw=1)

    r2 = r2_score_np(y_true, y_pred)
    ax.set_xlabel(f"True {label}")
    ax.set_ylabel(f"Predicted {label}")
    ax.set_title(f"{label} R²={r2:.4f}")
    ax.set_xlim(lim)
    ax.set_ylim(lim)

    fig.tight_layout()
    return fig


def residual_hist(residuals, label, bins=50, figsize=(5, 4)):
    """生成残差直方图"""
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(residuals, bins=bins, edgecolor='none', alpha=0.7)
    ax.axvline(0, color='r', linestyle='--', lw=1)

    ax.set_xlabel(f"Residual (pred − true) [{label}]")
    ax.set_ylabel("Count")
    ax.set_title(f"Residuals: {label}")

    fig.tight_layout()
    return fig
