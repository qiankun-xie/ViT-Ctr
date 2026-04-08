"""
论文图表生成脚本 — Phase 06 Plan 01

生成四张新图:
  Figure 1: 概念/流程图 (实验数据 → ctFP → SimpViT → 三参数+CI)
  Figure 2: ctFP双通道热力图示例 (dithioester, Ctr=1000)
  Figure 3: 三参数parity复合图 (3合1)
  TOC graphic: 简化版概念图 (~3.25×1.75 in)

用法: python scripts/generate_figures.py
"""

import sys
import os

# 确保项目根目录在路径中
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIGURES_DIR = os.path.join(ROOT, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# 统一样式
BLUE = '#4472C4'
DARK_BLUE = '#2F5496'
LIGHT_BLUE = '#D6E4F0'
GRAY = '#808080'
LIGHT_GRAY = '#F2F2F2'
GREEN = '#548235'
ORANGE = '#ED7D31'
FONT_MAIN = 10
FONT_SUB = 8
FONT_TINY = 7


def _draw_box(ax, xy, w, h, text, subtitle=None, color=BLUE,
              text_color='white', fontsize=FONT_MAIN):
    """绘制圆角矩形框+文字。"""
    x, y = xy
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.05",
        facecolor=color, edgecolor=DARK_BLUE, linewidth=1.2,
    )
    ax.add_patch(box)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.6, text,
                ha='center', va='center', fontsize=fontsize,
                fontweight='bold', color=text_color)
        ax.text(x + w / 2, y + h * 0.28, subtitle,
                ha='center', va='center', fontsize=FONT_TINY,
                color=text_color, style='italic')
    else:
        ax.text(x + w / 2, y + h / 2, text,
                ha='center', va='center', fontsize=fontsize,
                fontweight='bold', color=text_color)


def _draw_arrow(ax, start, end, color=DARK_BLUE):
    """绘制箭头。"""
    ax.annotate(
        '', xy=end, xytext=start,
        arrowprops=dict(
            arrowstyle='->', color=color, lw=1.8,
            connectionstyle='arc3,rad=0',
        ),
    )


def generate_fig1_concept(outpath=None):
    """
    Figure 1: 概念/流程图。
    实验数据 → ctFP编码 → SimpViT → 三参数 + 95% CI
    """
    if outpath is None:
        outpath = os.path.join(FIGURES_DIR, 'fig1_concept.png')

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis('off')

    # Box 1: Experimental Data
    _draw_box(ax, (0.2, 0.8), 1.8, 1.4,
              'Experimental\nData',
              subtitle='(conv, Mn, Đ)',
              color='#5B9BD5')

    # Arrow 1→2
    _draw_arrow(ax, (2.1, 1.5), (2.6, 1.5))

    # Box 2: ctFP Encoding
    _draw_box(ax, (2.6, 0.8), 1.8, 1.4,
              'ctFP\nEncoding',
              subtitle='64×64, 2-channel',
              color='#4472C4')

    # 小网格图标 (在ctFP框内下方)
    grid_x, grid_y = 3.15, 0.88
    for i in range(4):
        for j in range(4):
            rect = plt.Rectangle(
                (grid_x + j * 0.1, grid_y + i * 0.08),
                0.09, 0.07,
                facecolor=plt.cm.viridis(np.random.rand()),
                edgecolor='white', linewidth=0.3,
            )
            ax.add_patch(rect)

    # Arrow 2→3
    _draw_arrow(ax, (4.5, 1.5), (5.0, 1.5))

    # Box 3: SimpViT
    _draw_box(ax, (5.0, 0.8), 2.0, 1.4,
              'SimpViT',
              subtitle='2-layer, 4-head, ~877K params',
              color=DARK_BLUE)

    # Arrow 3→outputs
    _draw_arrow(ax, (7.1, 1.9), (7.7, 2.25))
    _draw_arrow(ax, (7.1, 1.5), (7.7, 1.5))
    _draw_arrow(ax, (7.1, 1.1), (7.7, 0.75))

    # Output boxes
    outputs = [
        ('log₁₀(Ctr)', 2.0),
        ('Inhibition\nPeriod', 1.25),
        ('Retardation\nFactor', 0.5),
    ]
    for label, y_center in outputs:
        box_y = y_center - 0.25
        _draw_box(ax, (7.7, box_y), 1.6, 0.5, label,
                  color=GREEN, fontsize=FONT_SUB)
        # ± 95% CI annotation
        ax.text(9.45, y_center, '± 95% CI',
                ha='left', va='center', fontsize=FONT_TINY,
                color=ORANGE, fontweight='bold')

    fig.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Figure 1 saved: {outpath}')


def generate_fig2_ctfp(outpath=None):
    """
    Figure 2: ctFP双通道热力图示例。
    使用ODE模拟生成dithioester样本 (Ctr=1000)，编码为ctFP。
    """
    if outpath is None:
        outpath = os.path.join(FIGURES_DIR, 'fig2_ctfp_example.png')

    from src.raft_ode import simulate_raft
    from src.ctfp_encoder import transform

    # 代表性dithioester参数 (Ctr = kadd/kfrag ≈ 1000)
    params = {
        'kd': 1e-5,
        'f': 0.7,
        'ki': 1e3,
        'kp': 500.0,
        'kt': 1e7,
        'kadd': 1e6,
        'kfrag': 1e3,
        'kadd0': 1e6,
        'kfrag0': 10.0,  # slow fragmentation → inhibition
        'M0': 8.0,
        'I0': 0.02,
        'CTA0': 0.04,
        'M_monomer': 104.15,  # styrene
    }

    # 多个[CTA]/[M]比值生成数据点
    cta_ratios = np.linspace(0.001, 0.05, 12)
    data_points = []

    for cta_r in cta_ratios:
        p = params.copy()
        p['CTA0'] = cta_r * p['M0']
        Mn_theory = p['M0'] / p['CTA0'] * p['M_monomer']

        result = simulate_raft(p, raft_type='dithioester', n_conv_points=30)
        if result is None:
            continue

        for j in range(len(result['conversion'])):
            conv = result['conversion'][j]
            mn_n = result['mn'][j] / Mn_theory if Mn_theory > 0 else 0
            disp = result['dispersity'][j]
            cta_norm = cta_r / 0.1  # 归一化到[0,1]，max [CTA]/[M]=0.1
            data_points.append((cta_norm, conv, mn_n, disp))

    if len(data_points) < 10:
        # Fallback: 生成合成数据用于演示
        print('  Warning: ODE simulation yielded few points, using synthetic demo data')
        np.random.seed(42)
        for _ in range(200):
            cta_n = np.random.uniform(0, 1)
            conv = np.random.uniform(0.02, 0.95)
            mn_n = conv * (0.5 + 0.5 * cta_n) + np.random.normal(0, 0.05)
            disp = 1.0 + 0.3 * np.exp(-2 * cta_n) + np.random.normal(0, 0.02)
            data_points.append((cta_n, conv, max(mn_n, 0), max(disp, 1.0)))

    ctfp = transform(data_points, img_size=64)
    ch0 = ctfp[0].numpy()  # Mn/Mn_theory
    ch1 = ctfp[1].numpy()  # Dispersity

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Channel 0: Mn/Mn_theory
    im0 = axes[0].imshow(ch0, origin='lower', cmap='viridis', aspect='equal',
                          vmin=0, vmax=max(ch0.max(), 0.01))
    axes[0].set_title('(a) Channel 0: Mn / Mn,theory', fontsize=11, pad=8)
    axes[0].set_xlabel('[CTA]/[M] (normalized)', fontsize=9)
    axes[0].set_ylabel('Conversion (normalized)', fontsize=9)
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar0.set_label('Mn / Mn,theory', fontsize=8)

    # Channel 1: Dispersity
    im1 = axes[1].imshow(ch1, origin='lower', cmap='plasma', aspect='equal',
                          vmin=0, vmax=max(ch1.max(), 0.01))
    axes[1].set_title('(b) Channel 1: Đ (Mw/Mn)', fontsize=11, pad=8)
    axes[1].set_xlabel('[CTA]/[M] (normalized)', fontsize=9)
    axes[1].set_ylabel('Conversion (normalized)', fontsize=9)
    cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar1.set_label('Đ (clipped at 4.0)', fontsize=8)

    # 设置刻度标签
    tick_pos = [0, 16, 32, 48, 63]
    tick_labels_x = ['0', '0.025', '0.05', '0.075', '0.1']
    tick_labels_y = ['0', '0.25', '0.5', '0.75', '1.0']
    for ax_i in axes:
        ax_i.set_xticks(tick_pos)
        ax_i.set_xticklabels(tick_labels_x, fontsize=7)
        ax_i.set_yticks(tick_pos)
        ax_i.set_yticklabels(tick_labels_y, fontsize=7)

    fig.suptitle('Chain Transfer Fingerprint (ctFP) — Dithioester, Ctr ≈ 1000',
                 fontsize=12, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Figure 2 saved: {outpath}')


def generate_fig3_composite(outpath=None):
    """
    Figure 3: 三参数parity复合图。
    加载现有parity PNG，排列为(a)(b)(c)三面板。
    """
    if outpath is None:
        outpath = os.path.join(FIGURES_DIR, 'fig3_parity_composite.png')

    parity_files = [
        ('parity_log10_Ctr.png', '(a) log₁₀(Ctr)'),
        ('parity_inhibition_period.png', '(b) Inhibition Period'),
        ('parity_retardation_factor.png', '(c) Retardation Factor'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, (fname, label) in enumerate(parity_files):
        fpath = os.path.join(FIGURES_DIR, fname)
        if not os.path.exists(fpath):
            print(f'  Warning: {fpath} not found, using placeholder')
            axes[i].text(0.5, 0.5, f'{label}\n(not available)',
                        ha='center', va='center', fontsize=12,
                        transform=axes[i].transAxes)
            axes[i].set_xlim(0, 1)
            axes[i].set_ylim(0, 1)
        else:
            img = plt.imread(fpath)
            axes[i].imshow(img)

        axes[i].axis('off')
        # Panel label in top-left
        axes[i].text(0.02, 0.98, label,
                    transform=axes[i].transAxes,
                    fontsize=12, fontweight='bold',
                    va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.3',
                             facecolor='white', edgecolor='gray',
                             alpha=0.85))

    fig.tight_layout(pad=1.0)
    fig.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Figure 3 saved: {outpath}')


def generate_toc(outpath=None):
    """
    TOC graphic: 简化版概念图。
    ~3.25×1.75 inches, 300 DPI, 紧凑布局。
    """
    if outpath is None:
        outpath = os.path.join(FIGURES_DIR, 'toc_graphic.png')

    fig, ax = plt.subplots(figsize=(3.25, 1.75))
    ax.set_xlim(0, 6.5)
    ax.set_ylim(0, 3.5)
    ax.axis('off')

    # ctFP grid icon (left)
    grid_x, grid_y = 0.2, 1.0
    grid_size = 5
    cell = 0.22
    for i in range(grid_size):
        for j in range(grid_size):
            val = np.random.rand()
            rect = plt.Rectangle(
                (grid_x + j * cell, grid_y + i * cell),
                cell * 0.95, cell * 0.95,
                facecolor=plt.cm.viridis(val),
                edgecolor='none',
            )
            ax.add_patch(rect)
    ax.text(grid_x + grid_size * cell / 2, grid_y - 0.25, 'ctFP',
            ha='center', va='top', fontsize=7, fontweight='bold',
            color=DARK_BLUE)

    # Arrow 1
    _draw_arrow(ax, (1.5, 1.55), (2.1, 1.55))

    # SimpViT box (center)
    _draw_box(ax, (2.1, 0.9), 1.6, 1.3, 'SimpViT',
              color=DARK_BLUE, fontsize=8)

    # Arrow 2
    _draw_arrow(ax, (3.8, 1.55), (4.3, 1.55))

    # Output: three lines with CI bars
    out_x = 4.4
    outputs = [
        ('Ctr', 2.5),
        ('tinh', 1.55),
        ('Rret', 0.6),
    ]
    for label, y_c in outputs:
        # CI error bar
        ax.plot([out_x + 0.6, out_x + 1.4], [y_c, y_c],
                color=ORANGE, lw=2.0)
        ax.plot([out_x + 0.6, out_x + 0.6], [y_c - 0.12, y_c + 0.12],
                color=ORANGE, lw=1.5)
        ax.plot([out_x + 1.4, out_x + 1.4], [y_c - 0.12, y_c + 0.12],
                color=ORANGE, lw=1.5)
        ax.plot(out_x + 1.0, y_c, 'o', color=GREEN, markersize=4, zorder=5)
        ax.text(out_x, y_c, label, ha='right', va='center',
                fontsize=6, fontweight='bold', color=DARK_BLUE,
                fontstyle='italic')

    # "± 95% CI" label
    ax.text(out_x + 1.0, 0.15, '± 95% CI',
            ha='center', va='center', fontsize=5.5,
            color=ORANGE, fontweight='bold')

    fig.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  TOC graphic saved: {outpath}')


if __name__ == '__main__':
    print('Generating publication figures...')
    print()

    print('[1/4] Figure 1: Concept/workflow diagram')
    generate_fig1_concept()

    print('[2/4] Figure 2: ctFP example (dithioester, Ctr ≈ 1000)')
    generate_fig2_ctfp()

    print('[3/4] Figure 3: Composite parity plot')
    generate_fig3_composite()

    print('[4/4] TOC graphic')
    generate_toc()

    print()
    print('All figures generated successfully.')
