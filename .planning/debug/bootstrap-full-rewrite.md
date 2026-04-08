---
status: in_progress
trigger: "Bootstrap subsystem too fragmented and never successfully run end-to-end"
created: 2026-04-05T00:00:00Z
updated: 2026-04-05T00:00:00Z
---

## Current Focus

hypothesis: Bootstrap code is duplicated across 6+ files with overlapping logic, making it fragile and hard to debug. Need a clean rewrite with single source of truth.
test: Rewrite to consolidate, then run tests
expecting: Clean architecture with src/bootstrap.py as single source of truth, autodl_bootstrap.py as thin runner
next_action: Execute rewrite

## Symptoms

expected: Clean bootstrap workflow: upload to AutoDL -> run -> download artifacts
actual: 6+ files with overlapping logic, previous bugs indicate quality issues, never run end-to-end
errors: (1) run_inference import that didn't exist (2) ddof=0 vs ddof=1 inconsistency
reproduction: Full codebase review
started: Phase 3 implementation

## Analysis

### Identified Problems

1. **Code duplication**: `autodl_bootstrap.py` (613 lines) reimplements freeze_backbone, bootstrap training loop, calibration logic, verification — all partially duplicating `src/bootstrap.py`
2. **Import fragility**: `autodl_bootstrap.py` line 260 does `from bootstrap import compute_jci, compute_coverage, calibrate_coverage` — relies on sys.path manipulation. line 334 does `from bootstrap import predict_with_uncertainty`. But it reimplements freeze_backbone and run_bootstrap_autodl locally.
3. **JCI inconsistency**: `run_calibration()` in autodl_bootstrap.py computes JCI per-sample (line 298: `val_pred_half = np.sqrt(val_pred_var * p * f_val / dfd)`) using per-sample variance, while `compute_jci()` in bootstrap.py uses the full covariance matrix. For diagonal case they're equivalent, but the approach is inconsistent.
4. **Two notebooks nearly identical**: 03-bootstrap-colab.ipynb and 03-bootstrap-autodl.ipynb differ only in paths and num_workers. They inline calibration logic instead of calling functions.
5. **predict_with_uncertainty in bootstrap.py is sample-at-a-time**: It loads all 200 heads and runs one sample through each. For calibration (many val samples), autodl_bootstrap.py reimplements a batched version.
6. **src/bootstrap.py run_bootstrap uses CombinedHDF5Dataset (lazy I/O)**: For AutoDL the preload-to-RAM approach in autodl_bootstrap.py is better, but this means autodl can't just call src/bootstrap.py's run_bootstrap.

### Architecture Decision

**Solution**: Rewrite `src/bootstrap.py` as the single source of truth with:
- `freeze_backbone()` — unchanged
- `run_bootstrap()` — accepts tensors OR DataLoader, with resume support, progress callbacks
- `collect_ensemble_predictions()` — batch prediction with all heads (the bottleneck operation)
- `compute_jci()` — unchanged
- `compute_coverage()` — unchanged
- `calibrate_coverage()` — unchanged
- `predict_with_uncertainty()` — single-sample convenience (for deploy/Streamlit)

Then `colab/autodl_bootstrap.py` becomes a thin CLI runner (~200 lines):
- Data loading (preload to RAM)
- Call src/bootstrap.py functions
- Progress printing, packaging, summary

### Files Changed
- `src/bootstrap.py` — Full rewrite (single source of truth)
- `colab/autodl_bootstrap.py` — Rewrite as thin runner importing from bootstrap.py
- `tests/test_bootstrap.py` — Update tests
- `colab/03-bootstrap-colab.ipynb` — DELETE (superseded by autodl_bootstrap.py)
- `colab/03-bootstrap-autodl.ipynb` — DELETE (superseded by autodl_bootstrap.py)
- `scripts/autodl-*.bat` — Keep as-is (they just call autodl_bootstrap.py)

## Resolution

root_cause: Code duplication, fragmented architecture, no single source of truth
fix: Complete rewrite with clean separation of concerns
verification: All tests pass, code review confirms no duplication
