# Phase 4 Re-validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-run Ctr validation with expanded 77-point literature dataset and add qualitative validation for inhibition period and retardation factor.

**Architecture:** Extend `run_validation_pipeline` in `src/literature_validation.py` to capture all 3 model outputs (not just Ctr). Add a new plotting function for inhibition/retardation by RAFT class. Update the notebook and docs.

**Tech Stack:** Python, NumPy, pandas, matplotlib, pytest

**Spec:** `docs/superpowers/specs/2026-04-08-phase4-revalidation-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/literature_validation.py` | Modify | Add inhibition/retardation columns to pipeline; new plot function |
| `tests/test_literature_validation.py` | Create | Tests for new columns and plot function |
| `notebooks/04-validate.ipynb` | Modify | Update for 77 points; add inhibition/retardation cells |
| `.planning/ROADMAP.md` | Modify | Update Phase 4 success criteria count |

---

### Task 1: Add inhibition/retardation columns to validation pipeline

**Files:**
- Modify: `src/literature_validation.py:263-300` (the results-building loop in `run_validation_pipeline`)
- Test: `tests/test_literature_validation.py` (create)

- [ ] **Step 1: Write failing tests**

Create `tests/test_literature_validation.py`:

```python
"""文献验证模块测试 — 验证pipeline输出包含inhibition/retardation列。"""
import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from literature_validation import (
    fold_error_log,
    fold_error_ratio,
    compute_summary_stats,
)


def test_fold_error_log_identity():
    """pred == true时fold-error应为1.0。"""
    assert fold_error_log(100, 100) == pytest.approx(1.0)


def test_fold_error_log_symmetric():
    """fold-error对称: fe(a,b) == fe(b,a)。"""
    assert fold_error_log(10, 100) == pytest.approx(fold_error_log(100, 10))


def test_fold_error_ratio_identity():
    assert fold_error_ratio(100, 100) == pytest.approx(1.0)


def test_compute_summary_stats_perfect():
    """完美预测时R²=1, RMSE=0, fold-error=1。"""
    true = np.array([1, 10, 100, 1000])
    stats = compute_summary_stats(true, true)
    assert stats['r2_log10'] == pytest.approx(1.0)
    assert stats['rmse_log10'] == pytest.approx(0.0, abs=1e-10)
    assert stats['median_fold_error'] == pytest.approx(1.0)
    assert stats['pct_within_2x'] == pytest.approx(100.0)


def test_pipeline_results_have_inhibition_retardation_columns():
    """验证pipeline输出DataFrame包含inhibition和retardation列。"""
    # 构造一个最小的mock results DataFrame来验证列名
    # 实际pipeline测试需要模型checkpoint，这里只测列名约定
    expected_cols = [
        'ml_inhibition', 'ml_retardation',
        'ml_inhibition_std', 'ml_retardation_std',
    ]
    # 这个测试在Task 1 Step 3实现后才能通过
    # 先用一个简单的结构验证
    from literature_validation import EXPECTED_RESULT_COLUMNS
    for col in expected_cols:
        assert col in EXPECTED_RESULT_COLUMNS, f"Missing column: {col}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_literature_validation.py -x -q`
Expected: FAIL — `EXPECTED_RESULT_COLUMNS` not defined

- [ ] **Step 3: Implement changes to `src/literature_validation.py`**

Add the column constant near the top of the file (after `RAFT_COLORS`):

```python
EXPECTED_RESULT_COLUMNS = [
    'id', 'raft_type', 'method',
    'ctr_true', 'log10_ctr_true',
    'ml_log10_ctr', 'mayo_log10_ctr',
    'ml_fold_error', 'mayo_fold_error',
    'ml_ci_low', 'ml_ci_high', 'ml_std',
    'ml_inhibition', 'ml_retardation',
    'ml_inhibition_std', 'ml_retardation_std',
]
```

Modify the results-building loop in `run_validation_pipeline` (around line 287-300). Replace the `results.append({...})` block with:

```python
        results.append({
            'id': row['id'],
            'raft_type': row['raft_type'],
            'method': row['method'],
            'ctr_true': row['ctr'],
            'log10_ctr_true': row['log10_ctr'],
            'ml_log10_ctr': ml_log10_ctr,
            'mayo_log10_ctr': mayo_log10_ctr,
            'ml_fold_error': ml_fold_err,
            'mayo_fold_error': mayo_fold_err,
            'ml_ci_low': ci_low,
            'ml_ci_high': ci_high,
            'ml_std': float(ml_std[0]),
            'ml_inhibition': float(ml_median[1]),
            'ml_retardation': float(ml_median[2]),
            'ml_inhibition_std': float(ml_std[1]),
            'ml_retardation_std': float(ml_std[2]),
        })
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_literature_validation.py -x -q`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/literature_validation.py tests/test_literature_validation.py
git commit -m "feat(phase4): add inhibition/retardation columns to validation pipeline"
```

---

### Task 2: Add inhibition/retardation plot function

**Files:**
- Modify: `src/literature_validation.py` (add new function)
- Modify: `tests/test_literature_validation.py` (add plot test)

- [ ] **Step 1: Write failing test**

Append to `tests/test_literature_validation.py`:

```python
def test_plot_inhibition_retardation_by_class(tmp_path):
    """验证inhibition/retardation分类图生成无报错。"""
    from literature_validation import plot_inhibition_retardation_by_class
    # 构造mock数据
    rng = np.random.default_rng(42)
    rows = []
    for raft_type in ['dithioester', 'trithiocarbonate', 'xanthate', 'dithiocarbamate']:
        for i in range(5):
            inh = rng.uniform(0.05, 0.3) if raft_type == 'dithioester' else rng.uniform(0, 0.02)
            ret = rng.uniform(0.3, 0.8) if raft_type == 'dithioester' else rng.uniform(0.9, 1.0)
            rows.append({
                'raft_type': raft_type,
                'ml_inhibition': inh,
                'ml_retardation': ret,
                'ml_inhibition_std': 0.01,
                'ml_retardation_std': 0.02,
            })
    df = pd.DataFrame(rows)
    plot_inhibition_retardation_by_class(df, str(tmp_path))
    assert (tmp_path / 'inhibition_retardation_by_class.png').exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_literature_validation.py::test_plot_inhibition_retardation_by_class -x -q`
Expected: FAIL — `plot_inhibition_retardation_by_class` not found

- [ ] **Step 3: Implement `plot_inhibition_retardation_by_class`**

Add to `src/literature_validation.py` after `plot_parity_ml_vs_mayo`:

```python
def plot_inhibition_retardation_by_class(results_df, output_dir='figures/validation'):
    """生成inhibition period和retardation factor按RAFT类型分组的strip图。"""
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    raft_order = ['dithioester', 'trithiocarbonate', 'xanthate', 'dithiocarbamate']
    colors = [RAFT_COLORS[rt] for rt in raft_order]

    for ax, col, ylabel, ref_val, title in [
        (axes[0], 'ml_inhibition', 'Predicted Inhibition Period', 0.0,
         'Inhibition Period by RAFT Type'),
        (axes[1], 'ml_retardation', 'Predicted Retardation Factor', 1.0,
         'Retardation Factor by RAFT Type'),
    ]:
        ax.axhline(ref_val, color='gray', ls='--', lw=1, alpha=0.7,
                    label=f'Reference ({ref_val})')
        for i, rt in enumerate(raft_order):
            sub = results_df[results_df['raft_type'] == rt]
            if sub.empty:
                continue
            x = np.full(len(sub), i) + np.random.default_rng(42).uniform(-0.15, 0.15, len(sub))
            ax.scatter(x, sub[col].values, color=RAFT_COLORS[rt], s=40, alpha=0.7, zorder=3)
            mean_val = sub[col].mean()
            ax.plot([i - 0.25, i + 0.25], [mean_val, mean_val],
                    color=RAFT_COLORS[rt], lw=2.5, zorder=4)

        ax.set_xticks(range(len(raft_order)))
        ax.set_xticklabels([rt[:6] for rt in raft_order], fontsize=9)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'inhibition_retardation_by_class.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
```

- [ ] **Step 4: Wire plot into `run_validation_pipeline`**

In `run_validation_pipeline`, after the `plot_parity_ml_vs_mayo(results_df, output_dir)` call (around line 312), add:

```python
    plot_inhibition_retardation_by_class(results_df, output_dir)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_literature_validation.py -x -q`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/literature_validation.py tests/test_literature_validation.py
git commit -m "feat(phase4): add inhibition/retardation strip plot by RAFT class"
```

---

### Task 3: Update notebook for 77-point dataset and inhibition/retardation

**Files:**
- Modify: `notebooks/04-validate.ipynb`

- [ ] **Step 1: Update notebook header cell**

Replace cell `08e56187` markdown:

```markdown
# Phase 4: Literature Validation — ML vs Mayo Baseline

对比SimpViT ML模型与传统Mayo ODE拟合在77个已发表Ctr值上的预测精度。
同时对inhibition period和retardation factor进行定性验证。
```

- [ ] **Step 2: Update display columns cell**

Replace cell `a23e671b`:

```python
display_cols = ["id", "raft_type", "method", "log10_ctr_true",
               "ml_log10_ctr", "mayo_log10_ctr",
               "ml_fold_error", "mayo_fold_error",
               "ml_inhibition", "ml_retardation"]
pd.set_option("display.float_format", "{:.3f}".format)
results_df[display_cols]
```

- [ ] **Step 3: Add inhibition/retardation summary cell after per-type breakdown**

Insert new cell after cell `8b801af0`:

```python
print("\n=== Inhibition Period & Retardation Factor (Qualitative) ===\n")
print("Expected: dithioester → elevated inhibition, retardation < 1.0")
print("          TTC/xanthate/dithiocarbamate → inhibition ≈ 0, retardation ≈ 1.0\n")
for raft_type in ["dithioester", "trithiocarbonate", "xanthate", "dithiocarbamate"]:
    sub = results_df[results_df["raft_type"] == raft_type]
    if sub.empty:
        continue
    inh_mean = sub["ml_inhibition"].mean()
    inh_std = sub["ml_inhibition"].std()
    ret_mean = sub["ml_retardation"].mean()
    ret_std = sub["ml_retardation"].std()
    print(f"{raft_type:20s}  n={len(sub):2d}  "
          f"inhibition={inh_mean:.3f}±{inh_std:.3f}  "
          f"retardation={ret_mean:.3f}±{ret_std:.3f}")
```

- [ ] **Step 4: Add inhibition/retardation figure cell**

Insert new cell after the summary cell:

```python
from IPython.display import Image
Image(filename=os.path.join(OUTPUT_DIR, "inhibition_retardation_by_class.png"), width=800)
```

- [ ] **Step 5: Update notes markdown cell**

Replace cell `1e024d33`:

```markdown
## Notes

**扩展数据集:** 77个文献Ctr值，覆盖4种RAFT剂类型（dithioester, trithiocarbonate, xanthate, dithiocarbamate），来源于Chong 2003, Moad 2009, Moad 2012。

**公平对比设计 (D-07):** ML和Mayo均在相同的ODE模拟数据上运行（加σ=0.03噪声），消除真实实验数据异质性的干扰。

**Mayo有利条件 (D-06):** Mayo拟合时其他动力学参数固定为训练分布中心值，这对Mayo是有利的（理想参数）。

**Bootstrap CI:** 当前无bootstrap检查点，CI使用集成std代替。训练bootstrap后可重新运行以获得95%置信区间。

**Inhibition/Retardation定性验证:** 文献中无定量数值，采用定性验证——检查模型预测是否符合已知RAFT化学行为：
- Dithioester: 应有明显诱导期（slow pre-equilibrium fragmentation），减速因子显著 < 1.0
- TTC/xanthate/dithiocarbamate: 诱导期≈0，减速因子≈1.0（理想RAFT交换）
```

- [ ] **Step 6: Commit**

```bash
git add notebooks/04-validate.ipynb
git commit -m "docs(phase4): update validation notebook for 77 points + inhibition/retardation"
```

---

### Task 4: Update ROADMAP and planning docs

**Files:**
- Modify: `.planning/ROADMAP.md`
- Modify: `.planning/STATE.md`

- [ ] **Step 1: Update ROADMAP Phase 4 description**

In `.planning/ROADMAP.md`, replace the Phase 4 goal line (line ~71):

```
**Goal**: The trained model is validated against 14 published Ctr values spanning 4 RAFT agent classes and 3 measurement methods, with fold-errors reported and a Mayo equation ODE-fitting baseline comparison included
```

with:

```
**Goal**: The trained model is validated against 77 published Ctr values spanning 4 RAFT agent classes and multiple measurement methods, with fold-errors reported, a Mayo equation ODE-fitting baseline comparison, and qualitative validation of inhibition period and retardation factor predictions
```

- [ ] **Step 2: Update ROADMAP success criteria**

Replace Phase 4 success criteria (lines ~74-78):

```
**Success Criteria** (what must be TRUE):
  1. A curated dataset of 77 published Ctr values exists, each annotated with RAFT agent class, measurement method (Mayo / CLD / dispersity / kinetic simulation), temperature, solvent, and monomer
  2. Model predictions on the literature set are compared to published values with fold-error computed for each point, and results are broken down by measurement method
  3. The Mayo equation baseline is implemented and evaluated on the same literature set, enabling a direct accuracy comparison between ML and traditional methods
  4. Paper-ready validation figures (predicted vs. published Ctr, per-class breakdown, inhibition/retardation by RAFT type) are produced
  5. Inhibition period and retardation factor predictions are qualitatively validated against known RAFT chemistry (dithioester shows retardation; others show ideal behavior)
```

- [ ] **Step 3: Update STATE.md context**

In `.planning/STATE.md`, add to the Decisions section:

```
- [Phase 04 revalidation]: Literature dataset expanded from 14 to 77 points (Chong 2003, Moad 2009, Moad 2012)
- [Phase 04 revalidation]: Inhibition/retardation validated qualitatively (no quantitative literature values exist); dithioester expected to show retardation, others ≈ ideal
```

- [ ] **Step 4: Commit**

```bash
git add .planning/ROADMAP.md .planning/STATE.md
git commit -m "docs(phase4): update roadmap and state for 77-point revalidation"
```
