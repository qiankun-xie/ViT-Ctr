# Phase 3: Model Training and Evaluation — Research

**Researched:** 2026-03-27
**Domain:** PyTorch regression training, Vision Transformer adaptation, bootstrap UQ, HDF5 data pipeline
**Confidence:** HIGH (based on direct source inspection of ViT-RR codebase, live HDF5 schema probing, and existing Phase 2 artifacts)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Training Strategy**
- D-01: Colab T4 for full training (16 GB VRAM), local MX350 only for debug/small-scale validation
- D-02: Batch size = 64
- D-03: Initial LR = 3e-4, ReduceLROnPlateau scheduler (factor=0.5, patience=5)
- D-04: Early stopping: monitor val loss, patience=15 epochs, save lowest val-loss checkpoint
- D-05: Max epochs determined by early stopping (estimated 30–100 epochs)

**Loss Function**
- D-06: Manually weighted MSE: `loss = 2.0 * MSE(pred_ctr, true_ctr) + 0.5 * MSE(pred_inh, true_inh) + 0.5 * MSE(pred_ret, true_ret)`
- D-07: Ctr is the core prediction target; inhibition and retardation are secondary
- D-08: Weights are [2.0, 0.5, 0.5] for [log10_Ctr, inhibition, retardation]

**Data Splitting**
- D-09: Stratified split by log10(Ctr) bins (0.5-unit bins over [-2, 4] → 12 bins)
- D-10: 80/10/10 train/val/test within each bin
- D-11: Test set guaranteed to span full Ctr range

**Bootstrap UQ**
- D-12: Lightweight bootstrap: train one base model, freeze backbone, fine-tune output head 200 times
- D-13: 200 bootstrap iterations, 5 fine-tuning epochs per iteration
- D-14: 200 output heads → mean and covariance at inference
- D-15: F-distribution 95% JCI: `half_width = sqrt(F(0.95, p=3, n-p) * p * cov_diag / n)`
- D-16: Post-hoc calibration on validation set; scalar correction factor if raw coverage < 95%

**Evaluation**
- D-17: Per-output R², RMSE, MAE on test set
- D-18: Segmented evaluation by Ctr range (low/mid/high)
- D-19: Per-RAFT-class evaluation (4 types, 4 parity plots per output)
- D-20: Residual histograms (predicted − true) per output
- D-21: Outlier analysis (|pred − true| > 2σ)

### Claude's Discretion
- DataLoader num_workers (recommended 2–4)
- ReduceLROnPlateau specific params (factor, patience, min_lr)
- Checkpoint save strategy (best only vs. top-3)
- HDF5 loading implementation (fully-in-memory vs. lazy)
- Parity plot visual style (colors, markers, legend position)
- Residual histogram bin count
- Training log format and frequency

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| TRN-01 | SimpViT (2-layer Transformer, 4-head attention, hidden=64) trained, output 3 parameters: log10(Ctr), inhibition period, retardation factor | SimpViT architecture from ViT-RR model_utils.py — only change is num_outputs=2→3; full tensor flow documented in Section 2 |
| TRN-02 | Train/val/test split stratified by Ctr range to prevent data leakage | Stratified split algorithm in Section 4 — operates on (file_idx, sample_idx) tuples across 4 HDF5 files |
| TRN-03 | Training uses log-space MSE loss, convergence curves recorded | Weighted MSE loss implementation in Section 3; convergence check functions in Section 10 |
| EVL-01 | Test-set R², RMSE, MAE per output parameter | NumPy-native metrics implementation (no sklearn dependency) in Section 7 |
| EVL-02 | Parity plots (predicted vs. true) per output parameter | Plot code with RAFT-class coloring and rasterized scatter in Section 7 |
| EVL-03 | Per-RAFT-class evaluation, class-specific parity plots | RAFT class labels flow through from HDF5 file index; 4 × 3 plot grid strategy in Section 7 |
| UQ-01 | Bootstrap (200 iterations) + F-distribution joint confidence interval | Backbone-freeze pattern + bootstrap loop + F-distribution formula from ViT-RR deploy.py in Sections 5–6 |
| UQ-02 | Post-hoc calibration on validation set, 95% CI coverage verified | Calibration procedure with scalar correction factor in Section 6 |
</phase_requirements>

---

## Summary

Phase 3 trains a SimpViT model (directly adapted from ViT-RR with `num_outputs=3`) on the four HDF5 files generated in Phase 2. The data pipeline runs: HDF5 files → stratified index split → `CombinedHDF5Dataset` → PyTorch `DataLoader` → training loop. The HDF5 schema is fully known from direct inspection (confirmed 2026-03-27): `fingerprints` shape `(n, 64, 64, 2)` channel-last, `labels` shape `(n, 3)` with column order `[log10_Ctr, inhibition_period, retardation_factor]`, four files of 9–10 samples each (test-scale). Full 1M-sample dataset must be regenerated on Colab before model training.

The lightweight bootstrap UQ strategy (freeze backbone, fine-tune 200 output heads × 5 epochs each) is directly portable from ViT-RR's `predict_model()` in `deploy.py` with `p=3` instead of `p=2`. Post-hoc calibration is required because raw bootstrap coverage systematically underestimates (confirmed in Pitfalls PITFALLS.md M3 and literature). The retardation factor prediction will behave as a near-trivial prediction (~1.0) for TTC/xanthate/dithiocarbamate systems — this is physically correct and must be documented, not treated as a model failure.

**Primary recommendation:** Copy SimpViT from `C:/CodingCraft/DL/ViT-RR/model_utils.py` verbatim, change `num_outputs=2` to `num_outputs=3`, implement the HDF5 Dataset with lazy file-handle opening, run stratified split before constructing Dataset objects, and port the F-distribution CI from `deploy.py` lines 157–167 exactly.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.10.0+cpu (local), 2.10.x+cu121 (Colab) | SimpViT, DataLoader, loss, optimizer, checkpoint | Project constraint; ViT-RR uses same; confirmed locally |
| h5py | (current) | HDF5 Dataset class, lazy loading of fingerprints and labels | Dataset format from Phase 2 is HDF5; no alternative |
| NumPy | 2.x | Stratified split logic, metrics, bootstrap covariance | Core scientific array library |
| scipy.stats.f | (part of SciPy 1.15.x) | F-distribution quantile for JCI: `f.ppf(0.95, dfn=3, dfd=197)` | Direct port from ViT-RR deploy.py line 162 |
| matplotlib | 3.10.x | Parity plots, loss curves, residual histograms | Project constraint |
| tqdm | 4.x | Bootstrap iteration progress bar | Already in project dependencies |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pandas | 2.2.x | Outlier analysis DataFrame, metrics summary CSV | Evaluation only; not needed in training loop |
| copy (stdlib) | — | `copy.deepcopy(model)` for bootstrap iteration | Required in bootstrap loop to avoid shared state |
| json (stdlib) | — | Save calibration factors to `calibration.json` | Post-hoc calibration output |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Manual weighted MSE | `torch.nn.MSELoss` per output + manual sum | Manual is clearer and avoids reduction mode confusion |
| Lazy HDF5 handles | Load all data to RAM (`np.array(f['fingerprints'][:])`) | At 38 samples (test data) this works; at 1M samples (~32 GB) it does not — use lazy for production |
| Custom metrics | `sklearn.metrics.r2_score` | sklearn adds a dependency; 3-line NumPy implementation is equivalent |

**Installation (Colab, additional):**
```bash
pip install h5py gdown  # h5py usually pre-installed; gdown for Drive downloads
```

**Version verification (confirmed 2026-03-27 on local machine):**
```
Python 3.13.9 | PyTorch 2.10.0+cpu | CUDA: False (local)
```
Note: Python 3.13 is newer than CLAUDE.md recommends (3.11). Local environment works; Colab will use its default (3.10 or 3.11 with GPU runtimes).

---

## Architecture Patterns

### Recommended Project Structure
```
src/
  model.py                  # SimpViT with num_outputs=3 (copy from ViT-RR + 1 change)
  train.py                  # Main training script
  bootstrap.py              # Lightweight bootstrap UQ
  evaluate.py               # Test-set metrics, parity plots, residual analysis
  utils/
    split.py                # Stratified train/val/test split
    metrics.py              # R², RMSE, MAE (no sklearn)
    visualization.py        # Parity plots, loss curves, residual histograms

checkpoints/
  best_model.pth            # Best base model (by val loss)
  bootstrap_heads.pth       # 200 fine-tuned fc heads + base model state_dict
  calibration.json          # cal_factors (3,) + empirical coverage report

figures/
  loss_curves.png
  parity_log10ctr.png
  parity_inhibition.png
  parity_retardation.png
  parity_by_class/          # 4 RAFT types × 3 outputs = 12 plots
  residuals_log10ctr.png
  residuals_inhibition.png
  residuals_retardation.png

colab/
  03-train-colab.ipynb      # Full training run on T4
  03-bootstrap-colab.ipynb  # Bootstrap UQ on T4

notebooks/
  03-debug-local.ipynb      # Local debug run (38 samples, 5 epochs)
  03-evaluate.ipynb         # Evaluation and figure generation
```

### Pattern 1: SimpViT Architecture (Verbatim from ViT-RR model_utils.py)

**What:** Copy `SimpViT` class from `C:/CodingCraft/DL/ViT-RR/model_utils.py` lines 7–38. Change `num_outputs=2` to `num_outputs=3`. No other change.

**Exact tensor flow:**
```
Input:  (B, 2, 64, 64) float32
patch_embedding Conv2d(in=2, out=64, kernel=16, stride=16) → (B, 64, 4, 4)
.permute(0, 2, 3, 1)    → (B, 4, 4, 64)
.view(B, 16, 64)         → (B, 16, 64)   # 16 patches
+ position_embedding[:, :16]   # (1, 17, 64) sliced to (1, 16, 64)
.permute(1, 0, 2)        → (16, B, 64)   # seq-first for PyTorch Transformer
TransformerEncoder (2 layers, 4 heads, d=64, batch_first=False)  → (16, B, 64)
.mean(dim=0)             → (B, 64)
fc Linear(64, 3)         → (B, 3)        # [log10_Ctr, inhibition, retardation]
```

**Source:** `C:/CodingCraft/DL/ViT-RR/model_utils.py` lines 7–38, direct inspection — HIGH confidence.

**Critical notes:**
- `position_embedding` shape is `(1, 17, 64)` (num_patches+1 = 17). Only first 16 rows are used. Do NOT modify — it is a harmless legacy artifact.
- `batch_first=False` is the PyTorch default for `TransformerEncoderLayer`. The `permute(1,0,2)` step in `forward()` is required.
- `TransformerEncoderLayer` uses dropout=0.1 by default. This is fine for training; call `model.eval()` during inference to disable it.

### Pattern 2: CombinedHDF5Dataset with Lazy File Handles

**What:** A PyTorch `Dataset` wrapping all 4 HDF5 files, using pre-computed index tuples. File handles are opened lazily inside `__getitem__` to avoid fork-unsafe handles with `num_workers > 0`.

**When to use:** Always for production training. For local debug with 38 samples, can load to RAM instead.

```python
# Source: Phase 2 HDF5 schema + PyTorch multiprocessing best practices
import h5py, torch
from torch.utils.data import Dataset

class CombinedHDF5Dataset(Dataset):
    def __init__(self, h5_paths, indices):
        # indices: list of (file_idx, sample_idx) or (file_idx, sample_idx, class_id)
        self.h5_paths = h5_paths
        self.indices = indices
        self._handles = None   # opened lazily, once per worker

    def __len__(self):
        return len(self.indices)

    def _get_handles(self):
        if self._handles is None:
            self._handles = [h5py.File(p, 'r') for p in self.h5_paths]
        return self._handles

    def __getitem__(self, idx):
        entry = self.indices[idx]
        file_idx, sample_idx = entry[0], entry[1]
        handles = self._get_handles()
        fp = handles[file_idx]['fingerprints'][sample_idx]   # (64, 64, 2) float32
        lbl = handles[file_idx]['labels'][sample_idx]        # (3,) float32
        # Transpose channel-last to channel-first for PyTorch Conv2d
        fp_tensor = torch.from_numpy(fp.transpose(2, 0, 1))  # → (2, 64, 64)
        return fp_tensor, torch.from_numpy(lbl.copy())
```

**DataLoader configuration (Claude's Discretion):**
- `num_workers=2`, `pin_memory=True` on Colab T4
- `num_workers=0` on local Windows (avoids spawn overhead with h5py)
- `shuffle=True` for train, `False` for val/test
- `batch_size=64` (D-02, locked)

### Pattern 3: Stratified Split by log10(Ctr) Bins

**What:** Build index lists for train/val/test before constructing `CombinedHDF5Dataset`. All indices have the form `(file_idx, sample_idx, class_id)`.

**Key facts from direct HDF5 inspection (2026-03-27):**
- `labels[:, 0]` is `log10_Ctr` — confirmed by `f.attrs['label_names']` = `['log10_Ctr', 'inhibition_period', 'retardation_factor']`
- `file_idx` 0→3 maps to `['dithioester', 'trithiocarbonate', 'xanthate', 'dithiocarbamate']`
- RAFT class label is derivable from file_idx (no separate class field needed)
- Current test files have only 9–10 samples each (38 total) — stratified split will work but produce very small splits

```python
# Source: established pattern for multi-file HDF5 + stratified split
import numpy as np, h5py

RAFT_TYPES = ['dithioester', 'trithiocarbonate', 'xanthate', 'dithiocarbamate']

def build_stratified_indices(h5_paths, val_frac=0.10, test_frac=0.10, seed=42):
    all_indices = []      # (file_idx, sample_idx, class_id)
    all_log10_ctr = []

    for file_idx, path in enumerate(h5_paths):
        with h5py.File(path, 'r') as f:
            log10_ctr = f['labels'][:, 0]   # confirmed column 0 = log10_Ctr
            n = len(log10_ctr)
            all_indices.extend([(file_idx, i, file_idx) for i in range(n)])
            all_log10_ctr.extend(log10_ctr.tolist())

    all_log10_ctr = np.array(all_log10_ctr)
    bins = np.arange(-2.0, 4.5, 0.5)        # 12 bins: [-2, -1.5), ..., [3.5, 4.0]
    bin_ids = np.digitize(all_log10_ctr, bins) - 1

    train_idx, val_idx, test_idx = [], [], []
    rng = np.random.default_rng(seed)

    for bin_id in range(len(bins) - 1):
        mask = np.where(bin_ids == bin_id)[0]
        if len(mask) == 0:
            continue
        rng.shuffle(mask)
        n_val  = max(1, int(len(mask) * val_frac))
        n_test = max(1, int(len(mask) * test_frac))
        val_idx.extend([all_indices[i] for i in mask[:n_val]])
        test_idx.extend([all_indices[i] for i in mask[n_val:n_val+n_test]])
        train_idx.extend([all_indices[i] for i in mask[n_val+n_test:]])

    return train_idx, val_idx, test_idx
```

**Edge case — empty bins with small datasets:** The 38-sample test dataset will have many empty bins (samples span about 8 of 12 bins). At 1M samples (~250K per type), all 12 bins will be populated. Log empty bins as warnings, do not raise errors.

### Pattern 4: Weighted MSE Loss (Locked D-06/D-08)

```python
def weighted_mse_loss(pred, target, weights=(2.0, 0.5, 0.5)):
    """
    pred, target: (B, 3) — [log10_Ctr, inhibition, retardation]
    """
    w = torch.tensor(weights, dtype=pred.dtype, device=pred.device)
    sq_err = (pred - target) ** 2       # (B, 3)
    per_output_mse = sq_err.mean(dim=0) # (3,) — average over batch
    return (w * per_output_mse).sum()   # scalar
```

**Device note:** Create the weight tensor inside the function so it moves with `pred` to any device (CPU/CUDA).

### Pattern 5: F-Distribution JCI (Exact from ViT-RR deploy.py lines 157–167)

```python
from scipy.stats import f

n = 200         # bootstrap iterations (locked D-13)
p = 3           # number of outputs (changed from ViT-RR's p=2)
dfd = n - p     # = 197

if dfd > 0 and np.isfinite(cov_matrix).all():
    f_val = f.ppf(0.95, dfn=p, dfd=dfd)           # ≈ 2.65 for p=3, n=200
    jci_half_width = np.sqrt(np.diag(cov_matrix) * p * f_val / dfd)
else:
    lower = np.percentile(predictions, 2.5, axis=0)
    upper = np.percentile(predictions, 97.5, axis=0)
    jci_half_width = (upper - lower) / 2.0
```

**Difference from ViT-RR:** `dfn=p` changes from `2` to `3`. Everything else is identical. `f.ppf(0.95, dfn=3, dfd=197)` ≈ 2.65 (vs. ≈ 3.05 for dfn=2).

### Anti-Patterns to Avoid

- **Opening HDF5 file in `__init__`:** Creates a file handle that cannot be pickled for `num_workers > 0`. Open lazily in `__getitem__` or use `num_workers=0`.
- **Calling `model.train()` during bootstrap inference:** Dropout is active in `train()` mode. Use `model.eval()` for all inference, including during bootstrap fine-tuning's evaluation step.
- **Forgetting `fp.transpose(2, 0, 1)`:** The HDF5 stores `(64, 64, 2)` channel-last. PyTorch Conv2d expects `(2, 64, 64)` channel-first. The transpose is mandatory.
- **Creating weight tensor once outside the loss function:** If model moves to GPU, the weight tensor stays on CPU → device mismatch error. Create it inside the function.
- **Plotting 1M scatter points without rasterization:** Vector scatter at 1M points produces unusable file sizes. Always use `rasterized=True` on scatter calls.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Transformer encoder | Custom attention + MLP | `nn.TransformerEncoderLayer` + `nn.TransformerEncoder` | PyTorch's implementation is optimized, handles masking, dropout; SimpViT already uses it |
| F-distribution quantile | Manual chi-squared ratio computation | `scipy.stats.f.ppf` | One function call; handles edge cases; exact port from ViT-RR |
| Bootstrap resampling | Custom sampling loop | `np.random.default_rng(seed).choice(..., replace=True)` | One line; reproducible; matches ViT-RR pattern |
| Learning rate scheduling | Custom LR decay | `torch.optim.lr_scheduler.ReduceLROnPlateau` | Handles plateau detection automatically; project constraint |
| Early stopping | Custom patience counter | Implement as `EarlyStopper` class (see Section 1) | Keep training loop clean; avoid off-by-one in patience counting |
| R² metric | Custom formula | 3-line NumPy (avoid sklearn dependency) | sklearn is not in pyproject.toml; trivial to implement |

**Key insight:** The entire SimpViT backbone and Bootstrap+F-distribution JCI pattern already exist, verified, and battle-tested in ViT-RR. The only new code needed is the weighted loss, HDF5 dataset, stratified split, calibration, and evaluation utilities.

---

## Dataset: Confirmed Ground Truth

**Confirmed 2026-03-27 by direct HDF5 inspection:**

| File | Samples (current) | Samples (target) | Size (current) |
|------|-------------------|------------------|----------------|
| `data/dithioester.h5` | 9 | ~250K | 298.5 KB |
| `data/trithiocarbonate.h5` | 10 | ~250K | 330.5 KB |
| `data/xanthate.h5` | 9 | ~250K | 298.5 KB |
| `data/dithiocarbamate.h5` | 10 | ~250K | 330.5 KB |

**HDF5 Schema (confirmed):**
- `fingerprints`: shape `(n, 64, 64, 2)`, dtype `float32`, chunk=(1000, 64, 64, 2)
- `labels`: shape `(n, 3)`, dtype `float32`, column order: `[0]=log10_Ctr, [1]=inhibition_period, [2]=retardation_factor`
- `params`: structured array with `(log10_Ctr, kp, kt, kd, I0, f, cta_ratio)`
- `attrs['label_names']` = `['log10_Ctr', 'inhibition_period', 'retardation_factor']`
- `attrs['raft_type']` = RAFT type string

**Label ranges (from current test-scale data, confirmed):**
| Parameter | Min | Max | Notes |
|-----------|-----|-----|-------|
| log10_Ctr | -1.98 | 3.99 | Spans design range [-2, 4] |
| inhibition_period | 0.0002 | 0.1028 | Dimensionless (t_inh / t_end) |
| retardation_factor | 0.997 | 1.000 | Near 1.0 for TTC/xanthate/dithiocarbamate |

**Critical finding:** Only dithioester samples show retardation < 1.0 (min 0.997 in test data, but likely wider range at scale). TTC, xanthate, dithiocarbamate all have retardation = 1.000 in test data. This confirms Pitfall C5 — the retardation head will learn a near-trivial prediction (~1.0) for 3 of 4 RAFT types.

**Current state:** Local HDF5 files exist (`data/*.h5`) with 38 total test samples. Full 1M-sample dataset must be generated by running `python src/dataset_generator.py` on Colab (Phase 2 plan) and uploading to Google Drive. The Google Drive file IDs are not yet in `02-DATASET-INFO.md` (listed as [TBD]).

---

## Common Pitfalls

### Pitfall 1: HDF5 Fork-Unsafe File Handles with num_workers > 0

**What goes wrong:** Opening `h5py.File` in `Dataset.__init__` creates a file handle that cannot be forked into worker processes. `DataLoader` with `num_workers > 0` on Linux/Colab will crash with a segfault or "cannot serialize" error.

**Why it happens:** h5py uses C-level HDF5 file handles that are not picklable.

**How to avoid:** Open file handles lazily inside `__getitem__`. Since each worker process gets its own copy of the Dataset object, `self._handles = None` is reset in each worker and opened fresh on first access.

**Warning signs:** `RuntimeError: Invalid argument` or segfault immediately on first DataLoader iteration on Colab with `num_workers=2`.

### Pitfall 2: Retardation Factor Is Near-Trivial for 3 of 4 RAFT Types

**What goes wrong:** Training loss for retardation converges too quickly (near-trivial prediction of ≈1.0) for TTC, xanthate, and dithiocarbamate systems. Per-class R² for retardation on these types will be near 0.

**Why it happens:** The ODE simulator confirms: at `kfrag=1e4` (fixed) with no pre-equilibrium, the polymerization rate is essentially unretarded (retardation_factor ≈ 1.0). The ctFP carries no information to distinguish retardation from 1.0.

**How to avoid:** This is expected behavior, not a bug. Document in training notes: "Retardation factor prediction is informative only for dithioester systems. For TTC/xanthate/dithiocarbamate, the model predicts ≈1.0, which is physically correct."

**Warning signs:** If retardation_factor for ALL RAFT types (including dithioester) converges to 1.0, then the dithioester pre-equilibrium ODE may not be producing retardation signal — check dataset.

### Pitfall 3: Loss Scale Imbalance → Weak Gradient for Inhibition/Retardation

**What goes wrong:** With weights 2.0/0.5/0.5, the Ctr loss dominates. The inhibition and retardation heads receive ~4× weaker gradient signal. The inhibition period range is [0.0002, 0.1028] (MSE on order 1e-4), while log10_Ctr range is [-2, 4] (MSE on order 1–4). The effective gradient ratio is roughly 200:1 in favor of Ctr.

**How to avoid:** Monitor per-output loss components separately during training. Log `loss_ctr`, `loss_inh`, `loss_ret` in addition to total weighted loss. If `loss_inh` or `loss_ret` plateaus at high values at epochs 10–20, consider temporarily equalizing weights during warmup.

**Warning signs:** `loss_inh` and `loss_ret` do not decrease after epoch 10 while `loss_ctr` continues decreasing.

### Pitfall 4: Dataset Too Small for Meaningful Training (Current State)

**What goes wrong:** The current HDF5 files contain only 38 total samples. Any "training" on this data will massively overfit and produce meaningless metrics. Phase 3 cannot produce valid results without the full ~1M sample dataset.

**How to avoid:** Phase 3 execution depends on Phase 2 completion at scale. The plan must include a wave/step that confirms the full dataset is available before training. The 38-sample debug mode is only valid for pipeline validation.

**Warning signs:** Training R² = 1.0 after 2 epochs on 38 samples — this is expected overfitting, not a real result.

### Pitfall 5: NaN Loss from Corrupt Samples

**What goes wrong:** If any HDF5 sample contains NaN or Inf (from ODE edge cases), one corrupt batch propagates NaN through the entire model.

**How to avoid:** Add NaN check after loss computation:
```python
if torch.isnan(loss):
    print(f"NaN loss at epoch {epoch}")
    break
```
Also validate dataset integrity before training: scan all labels for NaN/Inf.

### Pitfall 6: Bootstrap on 38-Sample Dataset Produces Misleading CIs

**What goes wrong:** With only 38 training samples, bootstrap resampling with replacement will produce many near-identical bootstrap sets (combinatorial exhaustion). The resulting CI will be artificially narrow.

**How to avoid:** Only run full bootstrap (200 iterations) on the full 1M-sample dataset. In debug mode with 38 samples, use `N_BOOTSTRAP=5` and label results as "debug-mode bootstrap, not valid for reporting."

---

## Runtime State Inventory

Step 2.5 is not applicable — this is a new model training phase, not a rename/refactor/migration.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python | Runtime | ✓ | 3.13.9 | — |
| PyTorch | Model training | ✓ | 2.10.0+cpu | — |
| h5py | HDF5 Dataset | ✓ | (installed) | — |
| NumPy | Split, metrics | ✓ | (installed) | — |
| SciPy | F-distribution JCI | ✓ | (installed) | — |
| Matplotlib | Figures | ✓ | (installed) | — |
| HDF5 data at scale | Full training | ✗ | 38 samples only | Debug mode with 38 samples |
| GPU (CUDA) | Fast training | ✗ (local) | — | Colab T4 (D-01) |
| Google Drive ID | Colab data access | ✗ | Not in DATASET-INFO.md | Must upload and record ID |

**Missing dependencies with no fallback:**
- Full ~1M sample HDF5 dataset — Phase 2 must complete at scale before Phase 3 can produce valid results

**Missing dependencies with fallback:**
- GPU: local training will be very slow but pipeline works; Colab T4 is the planned environment (D-01)

---

## Code Examples

### EarlyStopper Helper Class
```python
# Standard early stopping pattern for PyTorch
class EarlyStopper:
    def __init__(self, patience=15, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_state = None

    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
        return self.counter >= self.patience  # True = stop training
```

### Bootstrap Backbone Freeze
```python
# Source: standard PyTorch pattern for selective fine-tuning
def freeze_backbone(model):
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith('fc')
    return model

def get_head_optimizer(model, lr=1e-3):
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(trainable, lr=lr)
```

### Colab HDF5 File Access Pattern
```python
# Standard Colab Google Drive mount pattern
from google.colab import drive
drive.mount('/content/drive')

H5_PATHS = [
    '/content/drive/MyDrive/ViT-Ctr-data/dithioester.h5',
    '/content/drive/MyDrive/ViT-Ctr-data/trithiocarbonate.h5',
    '/content/drive/MyDrive/ViT-Ctr-data/xanthate.h5',
    '/content/drive/MyDrive/ViT-Ctr-data/dithiocarbamate.h5',
]

# Copy to local SSD for faster I/O (optional, recommended)
import shutil, os
os.makedirs('/content/data', exist_ok=True)
for src in H5_PATHS:
    shutil.copy(src, '/content/data/')
```

### Checkpoint Save/Resume Pattern
```python
# Save (called every 5 epochs + on best val loss)
def save_checkpoint(model, optimizer, epoch, val_loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, path)

# Resume
def load_checkpoint(model, optimizer, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return ckpt['epoch'] + 1, ckpt['val_loss']
```

### R² Metric (No sklearn)
```python
def r2_score_np(y_true, y_pred):
    """NumPy R² — no sklearn dependency."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Full bootstrap (200× retrain) | Lightweight: freeze backbone, fine-tune head 200× | ViT-RR deploy.py design | 400 hrs → ~22 hrs for 1M dataset |
| Raw bootstrap std as CI | F-distribution JCI + post-hoc calibration | ViT-RR → ViT-Ctr extension | Calibrated coverage at nominal level |
| p=2 (ViT-RR: r1, r2 reactivity) | p=3 (ViT-Ctr: Ctr, inhibition, retardation) | This project | dfn changes from 2 to 3 in F.ppf |
| sklearn metrics | NumPy-native metrics | This project convention | Fewer dependencies |

**Key difference from ViT-RR bootstrap:** ViT-RR's `predict_model()` in `deploy.py` uses input perturbation (noisy copies of the input data, 3% Gaussian noise on each row). This is inference-time bootstrap, not training-time bootstrap. For ViT-Ctr, the locked decision (D-12 to D-14) is training-time bootstrap (freeze backbone, fine-tune head on resampled training set). The inference pattern for ViT-Ctr is: run the 200 saved heads on the same clean input, not on noisy perturbations.

---

## Open Questions

1. **Full dataset availability on Google Drive**
   - What we know: 02-DATASET-INFO.md lists Drive IDs as [TBD]; Phase 2 execution uploaded to Drive but IDs not recorded
   - What's unclear: Are the full 1M samples already on Drive, or does Phase 2 need to run again?
   - Recommendation: Wave 0 of Phase 3 should include a task to confirm Drive IDs and record them in DATASET-INFO.md before any training

2. **Retardation factor range at scale**
   - What we know: Test data shows retardation_factor ≈ 1.0 for all non-dithioester types; dithioester min is 0.997 in 9 samples
   - What's unclear: At 250K dithioester samples, will there be a meaningful spread of retardation values below 0.9?
   - Recommendation: After generating full dataset, inspect retardation distribution per RAFT type; adjust kfrag0 in dataset_generator.py if dithioester retardation variance is too small

3. **Training time estimate on Colab T4**
   - What we know: SimpViT is ~3.4 MB; 1M samples at batch=64 = ~15,625 steps/epoch
   - What's unclear: At 50 epochs (early stop estimate), total = ~781K steps; T4 @ ~500 steps/sec → ~1,600 sec/epoch → ~22 hours total
   - Recommendation: Implement periodic checkpoint save every 5 epochs to survive Colab session timeout

4. **Bootstrap fine-tuning convergence at 5 epochs**
   - What we know: D-13 locks 5 epochs per bootstrap iteration; fc is `Linear(64, 3)` = 195 parameters
   - What's unclear: Will 5 epochs at LR=1e-3 actually converge for each bootstrap head?
   - Recommendation: During first bootstrap run, log per-epoch loss for the first 10 iterations to confirm convergence; extend to 10 epochs if needed

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (configured in pyproject.toml) |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` |
| Quick run command | `pytest tests/ -m "not slow" -x` |
| Full suite command | `pytest tests/ -x` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| TRN-01 | SimpViT instantiates with num_outputs=3 and forward pass produces (B,3) output | unit | `pytest tests/test_model.py::test_simpvit_forward -x` | ❌ Wave 0 |
| TRN-01 | SimpViT total parameter count matches expected (~3.4M) | unit | `pytest tests/test_model.py::test_simpvit_param_count -x` | ❌ Wave 0 |
| TRN-02 | Stratified split produces no overlap between train/val/test index sets | unit | `pytest tests/test_split.py::test_no_index_overlap -x` | ❌ Wave 0 |
| TRN-02 | Stratified split produces approximately 80/10/10 ratio | unit | `pytest tests/test_split.py::test_split_ratio -x` | ❌ Wave 0 |
| TRN-03 | Weighted MSE loss outputs correct scalar for known input | unit | `pytest tests/test_train.py::test_weighted_mse_loss -x` | ❌ Wave 0 |
| TRN-03 | Training loop runs 2 epochs on 38-sample debug dataset without error | integration | `pytest tests/test_train.py::test_debug_training_loop -x -m "not slow"` | ❌ Wave 0 |
| EVL-01 | R², RMSE, MAE metrics produce correct values on known array | unit | `pytest tests/test_metrics.py::test_metrics_known_values -x` | ❌ Wave 0 |
| EVL-02 | Parity plot function runs without error and returns figure object | unit | `pytest tests/test_visualization.py::test_parity_plot -x` | ❌ Wave 0 |
| EVL-03 | Per-class evaluation produces 4-class breakdown matching input class labels | unit | `pytest tests/test_evaluate.py::test_per_class_eval -x` | ❌ Wave 0 |
| UQ-01 | Bootstrap loop runs 3 iterations (debug N) and produces 3 fc state dicts | unit | `pytest tests/test_bootstrap.py::test_bootstrap_produces_heads -x` | ❌ Wave 0 |
| UQ-01 | F-distribution JCI produces correct half_width for known cov_matrix | unit | `pytest tests/test_bootstrap.py::test_f_dist_jci -x` | ❌ Wave 0 |
| UQ-02 | Calibration function returns scalar factors >= 1.0 when coverage < 0.95 | unit | `pytest tests/test_bootstrap.py::test_calibration_factors -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/ -m "not slow" -x`
- **Per wave merge:** `pytest tests/ -x`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_model.py` — covers TRN-01: SimpViT instantiation and forward pass
- [ ] `tests/test_split.py` — covers TRN-02: stratified split correctness
- [ ] `tests/test_train.py` — covers TRN-03: loss function and debug training loop
- [ ] `tests/test_metrics.py` — covers EVL-01: R², RMSE, MAE metric functions
- [ ] `tests/test_visualization.py` — covers EVL-02: parity plot function
- [ ] `tests/test_evaluate.py` — covers EVL-03: per-class evaluation
- [ ] `tests/test_bootstrap.py` — covers UQ-01, UQ-02: bootstrap loop, F-dist JCI, calibration
- [ ] `src/model.py` — SimpViT with num_outputs=3 (to be tested by test_model.py)
- [ ] `src/utils/split.py` — stratified split function
- [ ] `src/utils/metrics.py` — R², RMSE, MAE
- [ ] `src/utils/visualization.py` — parity plots, residual histograms

---

## Project Constraints (from CLAUDE.md)

| Constraint | Source | Enforcement |
|-----------|--------|-------------|
| PyTorch + Streamlit stack | CLAUDE.md § Constraints | No TensorFlow; no Gradio |
| SimpViT architecture: hidden=64, 2-layer, 4-head, output dim=3 | CLAUDE.md § Model Architecture | Do not change any hyperparameter except num_outputs 2→3 |
| Colab for large training, local MX350 for debug only | CLAUDE.md + D-01 | Training scripts must work in both environments |
| Python/Chinese for docs, English for paper and SI | CLAUDE.md § Language | Comments in code can be Chinese |
| GSD workflow enforcement | CLAUDE.md § GSD Workflow | All file edits through GSD commands |

---

## Sources

### Primary (HIGH confidence)
- `C:/CodingCraft/DL/ViT-RR/model_utils.py` (direct inspection) — SimpViT architecture, forward() tensor flow, position_embedding shape, num_patches calculation
- `C:/CodingCraft/DL/ViT-RR/deploy.py` (direct inspection, lines 121–174) — Bootstrap inference pattern, F-distribution JCI formula (lines 157–167), covariance matrix computation (line 155)
- `C:/CodingCraft/DL/ViT-Ctr/src/dataset_generator.py` (direct inspection) — HDF5 schema (n, 64, 64, 2) channel-last, label column order [log10_Ctr, inhibition, retardation], 4-file organization
- HDF5 file direct inspection (python h5py probe, 2026-03-27) — confirmed n=9–10 per file, label ranges, attrs['label_names']
- `.planning/phases/03-model-training-and-evaluation/03-CONTEXT.md` — all locked decisions D-01 through D-21
- `.planning/research/PITFALLS.md` — C5 (retardation identifiability), M3 (bootstrap CI undercovers), M4 (data leakage)
- `.planning/research/ARCHITECTURE.md` — component boundaries, data flow diagram
- `pyproject.toml` (direct inspection) — pytest configuration, project dependencies

### Secondary (MEDIUM confidence)
- PyTorch documentation — TransformerEncoderLayer batch_first=False default, DataLoader multiprocessing with h5py
- npj Computational Materials 2022 (Calibration after Bootstrap) — confirms bootstrap raw std undercovers; scalar correction approach

### Tertiary (LOW confidence)
- None — all critical claims verified by primary sources

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all versions confirmed locally, HDF5 schema confirmed by direct inspection
- Architecture: HIGH — SimpViT and F-dist JCI verified from ViT-RR source; HDF5 Dataset pattern is standard PyTorch
- Pitfalls: HIGH for retardation triviality (confirmed by data inspection); HIGH for HDF5 fork safety (standard PyTorch knowledge); HIGH for bootstrap CI calibration (PITFALLS.md + literature)

**Research date:** 2026-03-27
**Valid until:** 2026-05-27 (stable stack; PyTorch 2.10.0 released Jan 2026; no breaking changes expected)
