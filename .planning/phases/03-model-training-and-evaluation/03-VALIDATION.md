---
phase: 3
slug: model-training-and-evaluation
status: draft
nyquist_compliant: false
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
| 3-02-01 | 02 | 0 | TRN-02 | unit | `pytest tests/test_split.py::test_no_index_overlap -x` | ❌ W0 | ⬜ pending |
| 3-02-02 | 02 | 0 | TRN-02 | unit | `pytest tests/test_split.py::test_split_ratio -x` | ❌ W0 | ⬜ pending |
| 3-03-01 | 03 | 0 | TRN-03 | unit | `pytest tests/test_train.py::test_weighted_mse_loss -x` | ❌ W0 | ⬜ pending |
| 3-03-02 | 03 | 0 | TRN-03 | integration | `pytest tests/test_train.py::test_debug_training_loop -x -m "not slow"` | ❌ W0 | ⬜ pending |
| 3-04-01 | 04 | 1 | EVL-01 | unit | `pytest tests/test_metrics.py::test_metrics_known_values -x` | ❌ W0 | ⬜ pending |
| 3-04-02 | 04 | 1 | EVL-02 | unit | `pytest tests/test_visualization.py::test_parity_plot -x` | ❌ W0 | ⬜ pending |
| 3-04-03 | 04 | 1 | EVL-03 | unit | `pytest tests/test_evaluate.py::test_per_class_eval -x` | ❌ W0 | ⬜ pending |
| 3-05-01 | 05 | 2 | UQ-01 | unit | `pytest tests/test_bootstrap.py::test_bootstrap_produces_heads -x` | ❌ W0 | ⬜ pending |
| 3-05-02 | 05 | 2 | UQ-01 | unit | `pytest tests/test_bootstrap.py::test_f_dist_jci -x` | ❌ W0 | ⬜ pending |
| 3-05-03 | 05 | 2 | UQ-02 | unit | `pytest tests/test_bootstrap.py::test_calibration_factors -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_model.py` — stubs for TRN-01 (SimpViT instantiation, forward pass, param count)
- [ ] `tests/test_split.py` — stubs for TRN-02 (stratified split correctness, no overlap, ratio)
- [ ] `tests/test_train.py` — stubs for TRN-03 (weighted MSE loss, debug training loop)
- [ ] `tests/test_metrics.py` — stubs for EVL-01 (R², RMSE, MAE on known arrays)
- [ ] `tests/test_visualization.py` — stubs for EVL-02 (parity plot returns figure object)
- [ ] `tests/test_evaluate.py` — stubs for EVL-03 (per-class evaluation breakdown)
- [ ] `tests/test_bootstrap.py` — stubs for UQ-01, UQ-02 (bootstrap heads, F-dist JCI, calibration)
- [ ] `src/model.py` — SimpViT with num_outputs=3 (tested by test_model.py)
- [ ] `src/utils/split.py` — stratified split function
- [ ] `src/utils/metrics.py` — R², RMSE, MAE (no sklearn)
- [ ] `src/utils/visualization.py` — parity plots, residual histograms

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Training loss curves show convergence (no divergence, no plateau) | TRN-03 | Requires visual inspection of full training run on Colab T4 | Run `colab/03-train-colab.ipynb`, inspect loss_curves.png — all three output losses must decrease monotonically or plateau at low values |
| Parity plots show tight clustering around identity line (R² reported) | EVL-01, EVL-02 | Requires full 1M-sample trained model | Inspect figures/parity_*.png — R² values reported per output in eval summary |
| Per-RAFT-class parity plots (4 types × 3 outputs = 12 plots) | EVL-03 | Requires full trained model and visual inspection | Inspect figures/parity_by_class/*.png — confirm all 4 RAFT types and all 3 outputs present |
| Bootstrap 95% CI coverage ≥ 95% on calibration set | UQ-02 | Requires full bootstrap run and calibration | Inspect calibration.json — empirical_coverage field must be ≥ 0.95 after calibration |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
