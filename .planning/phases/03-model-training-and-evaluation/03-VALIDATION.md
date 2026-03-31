---
phase: 3
slug: model-training-and-evaluation
status: ready
nyquist_compliant: true
wave_0_complete: false
created: 2026-03-27
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (configured in pyproject.toml) |
| **Config file** | `pyproject.toml` `[tool.pytest.ini_options]` |
| **Quick run command** | `pytest tests/ -m "not slow" -x` |
| **Full suite command** | `pytest tests/ -x` |
| **Estimated runtime** | ~30 seconds (unit tests only) |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/ -m "not slow" -x`
- **After every plan wave:** Run `pytest tests/ -x`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 3-01-01 | 01 | 0 | TRN-01 | unit | `pytest tests/test_model.py::test_simpvit_forward -x` | ❌ W0 | ⬜ pending |
| 3-01-02 | 01 | 0 | TRN-01 | unit | `pytest tests/test_model.py::test_simpvit_param_count -x` | ❌ W0 | ⬜ pending |
| 3-01-03 | 01 | 0 | TRN-01 | unit | `pytest tests/test_model.py::test_simpvit_eval_mode -x` | ❌ W0 | ⬜ pending |
| 3-01-04 | 01 | 0 | EVL-01 | unit | `pytest tests/test_metrics.py::test_metrics_known_values -x` | ❌ W0 | ⬜ pending |
| 3-01-05 | 01 | 0 | EVL-01 | unit | `pytest tests/test_metrics.py::test_metrics_perfect_prediction -x` | ❌ W0 | ⬜ pending |
| 3-01-06 | 01 | 0 | EVL-02 | unit | `pytest tests/test_visualization.py::test_parity_plot -x` | ❌ W0 | ⬜ pending |
| 3-01-07 | 01 | 0 | EVL-02 | unit | `pytest tests/test_visualization.py::test_residual_hist -x` | ❌ W0 | ⬜ pending |
| 3-02-01 | 02 | 0 | TRN-02 | unit | `pytest tests/test_split.py::test_no_index_overlap -x` | ❌ W0 | ⬜ pending |
| 3-02-02 | 02 | 0 | TRN-02 | unit | `pytest tests/test_split.py::test_split_ratio -x` | ❌ W0 | ⬜ pending |
| 3-02-03 | 02 | 0 | TRN-02 | unit | `pytest tests/test_split.py::test_combined_dataset_item_shapes -x` | ❌ W0 | ⬜ pending |
| 3-02-04 | 02 | 0 | EVL-03 | unit | `pytest tests/test_evaluate.py::test_per_class_eval -x` | ❌ W0 | ⬜ pending |
| 3-02-05 | 02 | 0 | EVL-03 | unit | `pytest tests/test_evaluate.py::test_compute_metrics_all_outputs -x` | ❌ W0 | ⬜ pending |
| 3-03-01 | 03 | 1 | TRN-03 | unit | `pytest tests/test_train.py::test_weighted_mse_loss -x` | ❌ W0 | ⬜ pending |
| 3-03-02 | 03 | 1 | TRN-03 | unit | `pytest tests/test_train.py::test_weighted_mse_loss_device_safe -x` | ❌ W0 | ⬜ pending |
| 3-03-03 | 03 | 1 | TRN-03 | unit | `pytest tests/test_train.py::test_early_stopper_triggers -x` | ❌ W0 | ⬜ pending |
| 3-03-04 | 03 | 1 | TRN-03 | integration | `pytest tests/test_train.py::test_debug_training_loop -x` | ❌ W0 | ⬜ pending |
| 3-04-01 | 04 | 2 | EVL-01,02,03 | import | `python -c "from evaluate import run_full_evaluation"` | ❌ W2 | ⬜ pending |
| 3-05-01 | 05 | 3 | UQ-01 | unit | `pytest tests/test_bootstrap.py::test_bootstrap_produces_heads -x` | ❌ W0 | ⬜ pending |
| 3-05-02 | 05 | 3 | UQ-01 | unit | `pytest tests/test_bootstrap.py::test_f_dist_jci -x` | ❌ W0 | ⬜ pending |
| 3-05-03 | 05 | 3 | UQ-02 | unit | `pytest tests/test_bootstrap.py::test_calibration_factors -x` | ❌ W0 | ⬜ pending |
| 3-05-04 | 05 | 3 | UQ-01 | unit | `pytest tests/test_bootstrap.py::test_predict_with_uncertainty -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Created by Plan 01 (wave 0):
- [ ] `tests/test_model.py` — TRN-01 (SimpViT forward, param count, eval mode)
- [ ] `tests/test_metrics.py` — EVL-01 (R², RMSE, MAE known values + perfect)
- [ ] `tests/test_visualization.py` — EVL-02 (parity_plot, residual_hist)
- [ ] `src/model.py` — SimpViT with num_outputs=3
- [ ] `src/utils/metrics.py` — r2_score_np, rmse_np, mae_np
- [ ] `src/utils/visualization.py` — parity_plot, residual_hist

Created by Plan 02 (wave 0):
- [ ] `tests/test_split.py` — TRN-02 (no overlap, ratio, dataset item shapes)
- [ ] `tests/test_evaluate.py` — EVL-03 (per_class_eval, compute_metrics_all_outputs)
- [ ] `src/utils/split.py` — build_stratified_indices, RAFT_TYPES
- [ ] `src/dataset.py` — CombinedHDF5Dataset (lazy handles)
- [ ] `src/evaluate.py` — per_class_metrics, compute_test_metrics (stubs)

Created by Plan 03 (wave 1):
- [ ] `tests/test_train.py` — TRN-03 (loss, device_safe, early_stopper, debug_loop)
- [ ] `src/train.py` — full training infrastructure

Created by Plan 05 (wave 3):
- [ ] `tests/test_bootstrap.py` — UQ-01,02 (bootstrap heads, F-dist JCI, calibration, predict)
- [ ] `src/bootstrap.py` — run_bootstrap, compute_jci, calibrate_coverage, predict_with_uncertainty

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Training loss curves show convergence (no divergence, no plateau) | TRN-03 | Requires visual inspection of full training run on Colab T4 | Run `colab/03-train-colab.ipynb`, inspect loss_curves.png — all three output losses must decrease monotonically or plateau at low values |
| Parity plots show tight clustering around identity line (R² reported) | EVL-01, EVL-02 | Requires full 1M-sample trained model | Inspect figures/parity_*.png — R² values reported per output in eval summary |
| Per-RAFT-class parity plots (4 types × 3 outputs = 12 plots) | EVL-03 | Requires full trained model and visual inspection | Inspect figures/parity_by_class/*.png — confirm all 4 RAFT types and all 3 outputs present |
| Bootstrap 95% CI coverage ≥ 95% on calibration set | UQ-02 | Requires full bootstrap run and calibration | Inspect calibration.json — empirical_coverage_after field must be ≥ 0.95 after calibration |
| Wave 0 gate triggers RuntimeError on incomplete HDF5 dataset | TRN-01 (guard) | Requires live test with missing/small HDF5 files | Manually change MIN_SAMPLES_PER_FILE to exceed actual sample count; verify RuntimeError raised |

---

## Nyquist Compliance Check

- [x] All tasks have `<automated>` verify command or Wave 0 dependency documented
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING file references
- [x] No watch-mode flags
- [x] Feedback latency < 30s (unit tests only)
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** ready

---

## Requirement Coverage Map

| Req ID | Plans | Tests |
|--------|-------|-------|
| TRN-01 | 03-01 | test_model.py (3 tests) |
| TRN-02 | 03-02 | test_split.py (3 tests) |
| TRN-03 | 03-03 | test_train.py (4 tests) |
| EVL-01 | 03-01, 03-04 | test_metrics.py (2 tests) |
| EVL-02 | 03-01, 03-04 | test_visualization.py (2 tests) |
| EVL-03 | 03-02, 03-04 | test_evaluate.py (2 tests) |
| UQ-01 | 03-05 | test_bootstrap.py (2 tests) |
| UQ-02 | 03-05 | test_bootstrap.py (2 tests) |
