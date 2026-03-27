# Plan 03-02 Summary: Data Pipeline — Stratified Split, Dataset, Per-Class Evaluation

**Phase:** 03-model-training-and-evaluation
**Plan:** 02 of 5
**Wave:** 0
**Completed:** 2026-03-27
**Duration:** ~8 min

## What Was Done

### Task 1: Stratified split and CombinedHDF5Dataset
Created the data pipeline for training: stratified train/val/test split and the lazy-handle HDF5 dataset class.

**Files created:**
- `src/utils/__init__.py` — Python package marker for utils
- `src/utils/metrics.py` — r2_score_np, rmse_np, mae_np (no sklearn dependency)
- `src/utils/split.py` — `build_stratified_indices` with `RAFT_TYPES` constant
- `src/dataset.py` — `CombinedHDF5Dataset` with lazy `_handles` pattern
- `tests/test_split.py` — 3 tests covering TRN-02

**Key implementation decisions:**
- `self._handles = None` in `__init__` (not opened at construction time) — fork-safe for `num_workers > 0` on Colab
- `fp.transpose(2, 0, 1).copy()` converts `(64,64,2)` channel-last → `(2,64,64)` channel-first for Conv2d
- Stratification uses 12 bins of 0.5 units over `[-2.0, 4.0]` log10(Ctr)
- Empty bins log a debug message and are skipped (no exception raised)
- `n_val = max(1, int(len(mask) * val_frac))` ensures at least 1 sample per bin in val/test

### Task 2: Per-class evaluation stub
Created evaluation functions for test-set metrics and per-RAFT-class breakdown.

**Files created:**
- `src/evaluate.py` — `compute_test_metrics`, `per_class_metrics`, `compute_outlier_stats`
- `tests/test_evaluate.py` — 2 tests covering EVL-03

**Key implementation decisions:**
- `compute_test_metrics` computes R²/RMSE/MAE per output column (3 columns)
- `per_class_metrics` groups by `class_id` using `RAFT_TYPES` imported from `utils.split`
- `compute_outlier_stats` (D-21): `|pred - true| > 2 * std(residuals)` per output
- matplotlib NOT imported at module level — lazy import pattern for headless environments
- `evaluate.py` structured to grow in Plan 04 (figure generation added inside functions)

## Test Results

```
tests/test_split.py::test_no_index_overlap          PASSED
tests/test_split.py::test_split_ratio               PASSED
tests/test_split.py::test_combined_dataset_item_shapes  PASSED
tests/test_evaluate.py::test_per_class_eval         PASSED
tests/test_evaluate.py::test_compute_metrics_all_outputs  PASSED

5 passed in 2.46s
```

## Success Criteria Verification

| Criterion | Status |
|-----------|--------|
| `pytest tests/test_split.py tests/test_evaluate.py` exits 0 | ✅ 5/5 passed |
| `grep "_handles = None" src/dataset.py` returns match | ✅ |
| `grep "transpose(2, 0, 1)" src/dataset.py` returns match | ✅ |
| `grep "RAFT_TYPES" src/utils/split.py` returns match | ✅ |
| `src/evaluate.py` contains `per_class_metrics` and `compute_test_metrics` | ✅ |
| `grep "RAFT_TYPES" src/evaluate.py` returns match | ✅ |
| `grep "compute_outlier_stats" src/evaluate.py` returns match | ✅ |

## Commits

1. `feat(03-02/task1): stratified split and CombinedHDF5Dataset`
2. `feat(03-02/task2): per-class evaluation and test metrics (EVL-03)`

## Impact on Next Plans

- Plan 03 (training loop) can now import `CombinedHDF5Dataset` and `build_stratified_indices`
- Plan 03 can construct `DataLoader`s from the split indices
- Plan 04 (evaluation) can import `compute_test_metrics` and `per_class_metrics`
- `src/evaluate.py` ready to receive plotting functions in Plan 04

## Decisions Made

- `src/utils/metrics.py` created here (not in Plan 01 artifact) — needed as evaluate.py dependency; aligns with Plan 01's intent
- `compute_outlier_stats` included now (not tested) to avoid Plan 04 merge conflicts
- `per_class_metrics` returns only classes with ≥1 sample — consistent with edge case handling elsewhere
