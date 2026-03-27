---
phase: 03-model-training-and-evaluation
plan: 05
subsystem: uncertainty-quantification
tags: [bootstrap, f-distribution, calibration, colab]
dependency_graph:
  requires: [03-04-PLAN.md]
  provides: [bootstrap.py, bootstrap_heads.pth, calibration.json]
  affects: [Phase 5 Streamlit inference, Phase 6 paper UQ section]
tech_stack:
  added: [scipy.stats.f]
  patterns: [backbone-freeze, bootstrap-resample, binary-search-calibration]
key_files:
  created:
    - src/bootstrap.py
    - tests/test_bootstrap.py
    - colab/03-bootstrap-colab.ipynb
  modified: []
decisions:
  - Bootstrap uses training-time head fine-tuning (not inference-time input perturbation)
  - F-distribution JCI with p=3, exact port from ViT-RR deploy.py
  - Calibration via binary search per output (50 iterations, precision < 1e-14)
  - Debug mode flag when n_bootstrap < 50
metrics:
  duration_min: 6
  completed_date: "2026-03-27"
  tasks_completed: 2
  files_created: 3
  tests_added: 4
  test_pass_rate: "100% (82/82 tests pass)"
---

# Phase 03 Plan 05: Bootstrap Uncertainty Quantification Summary

**One-liner:** Lightweight bootstrap UQ with backbone-freeze strategy, F-distribution 95% JCI (p=3), and post-hoc calibration on validation set.

## What Was Built

Implemented the complete bootstrap uncertainty quantification pipeline for ViT-Ctr's three-output regression (log10_Ctr, inhibition_period, retardation_factor). The system freezes the SimpViT backbone and fine-tunes 200 output heads on bootstrap-resampled training data, then computes calibrated 95% confidence intervals using F-distribution joint confidence intervals.

**Core components:**
- `src/bootstrap.py`: 5 functions (freeze_backbone, run_bootstrap, compute_jci, calibrate_coverage, predict_with_uncertainty)
- `tests/test_bootstrap.py`: 4 tests covering UQ-01 and UQ-02 requirements
- `colab/03-bootstrap-colab.ipynb`: 13-cell notebook for Colab T4 execution (~20 hours for full 200 iterations)

## Implementation Details

### Bootstrap Strategy (D-12/D-13)
- Freeze all parameters except fc layer: `param.requires_grad = name.startswith('fc')`
- 200 bootstrap iterations, each with:
  - Bootstrap resample (with replacement) of full training set
  - 5 epochs of fc head fine-tuning (lr=1e-3, Adam optimizer)
  - Save fc.weight and fc.bias state dict
- Output: `bootstrap_heads.pth` containing 200 head state dicts + base backbone

### F-Distribution JCI (D-15)
Direct port from ViT-RR deploy.py lines 157-167, adapted for p=3:
```python
f_val = f.ppf(0.95, dfn=p, dfd=n-p)  # p=3, n=200, dfd=197
half_width = sqrt(diag(cov_matrix) * p * f_val / dfd)
```
For n=200, p=3: f_val ≈ 2.65

### Post-Hoc Calibration (D-16)
Binary search per output to find minimum scalar factor achieving 95% empirical coverage on validation set:
- 50 bisection iterations (precision < 1e-14)
- Coverage metric: fraction of samples where |pred_mean - true| <= half_width
- Output: `calibration.json` with 3 calibration factors (one per output)

### Colab Notebook Flow
1. Environment setup + Drive mount
2. Load best_model.pth from Phase 3 training
3. Build train/val DataLoaders with stratified split
4. Debug run (3 iterations) to check convergence
5. Full bootstrap (200 iterations, ~20 hours)
6. Calibration on validation set
7. Demo inference with uncertainty
8. Download bootstrap_heads.pth and calibration.json

## Deviations from Plan

None — plan executed exactly as written.

## Test Coverage

All 4 bootstrap tests pass:
- `test_bootstrap_produces_heads`: Verifies 200 head state dicts saved with correct keys
- `test_f_dist_jci`: Validates F-distribution formula against known covariance matrix
- `test_calibration_factors`: Confirms factors >= 1.0 for narrow CIs
- `test_predict_with_uncertainty`: End-to-end inference with 5 heads

Full suite: 82/82 tests pass (including all Phase 3 Wave 0 tests).

## Known Stubs

None. All functions are fully implemented and tested.

## Verification

```bash
# Test suite
pytest tests/test_bootstrap.py -x -v  # 4/4 pass
pytest tests/ -m "not slow" -x -v     # 82/82 pass

# Pattern verification
grep "f.ppf(0.95, dfn=p, dfd=dfd)" src/bootstrap.py  # ✓ exact formula
grep "startswith.*fc" src/bootstrap.py                # ✓ backbone freeze
grep "calibrate_coverage" colab/03-bootstrap-colab.ipynb  # ✓ in notebook
```

## Next Steps

1. Run `colab/03-bootstrap-colab.ipynb` on Colab T4 after full ~1M sample dataset is uploaded to Google Drive
2. Download `bootstrap_heads.pth` (~50 MB) and `calibration.json` to local `checkpoints/`
3. Verify `calibration.json` shows `empirical_coverage_after` >= 0.95 for all three outputs
4. Use `predict_with_uncertainty()` in Phase 5 Streamlit app for inference with confidence intervals
5. Cite bootstrap UQ methodology and calibration results in Phase 6 paper

## Commits

- `15e5203`: feat(03-05): implement bootstrap UQ with F-distribution JCI
- `590002e`: feat(03-05): add Colab bootstrap notebook with calibration

## Self-Check: PASSED

**Files created:**
- ✓ src/bootstrap.py exists
- ✓ tests/test_bootstrap.py exists
- ✓ colab/03-bootstrap-colab.ipynb exists

**Commits exist:**
- ✓ 15e5203 found in git log
- ✓ 590002e found in git log

**Requirements met:**
- ✓ UQ-01: Bootstrap + F-distribution JCI implemented
- ✓ UQ-02: Post-hoc calibration implemented
- ✓ All 82 tests pass
