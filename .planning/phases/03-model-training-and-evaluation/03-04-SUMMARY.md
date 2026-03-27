---
phase: 03-model-training-and-evaluation
plan: 04
subsystem: evaluation
tags: [evaluation, metrics, visualization, parity-plots, residuals]
dependency_graph:
  requires: [03-02-dataset-pipeline, 03-03-training-script]
  provides: [evaluation-module, evaluation-notebook, figure-generation]
  affects: [phase-06-paper-writing]
tech_stack:
  added: []
  patterns: [matplotlib-lazy-import, rasterized-scatter]
key_files:
  created:
    - src/evaluate.py (extended)
    - notebooks/03-evaluate.ipynb
    - figures/.gitkeep
  modified:
    - src/evaluate.py
decisions:
  - Matplotlib imports delayed inside plotting functions to allow module import in headless environments
  - Rasterized scatter plots enforced (already in visualization.py) to prevent massive file sizes at 1M points
  - Retardation factor R²≈0 for TTC/xanthate/dithiocarbamate documented as expected behavior
  - OUTPUT_NAMES constant ensures consistent labeling across all outputs
metrics:
  duration_min: 3
  tasks_completed: 2
  files_created: 2
  files_modified: 1
  commits: 2
  tests_added: 0
  completed_date: "2026-03-27"
---

# Phase 03 Plan 04: Evaluation Module and Figures Summary

Complete evaluation infrastructure with figure generation and metrics reporting.

## Tasks Completed

### Task 1: Extended evaluate.py with Figure Generation
Added 7 new functions to src/evaluate.py:
- `run_inference`: Extract predictions and class IDs from test DataLoader
- `save_parity_plots`: Generate 3 overall parity plots (one per output)
- `save_per_class_parity_plots`: Generate 12 class-specific parity plots (4 RAFT types × 3 outputs)
- `save_residual_plots`: Generate 3 residual histograms
- `compute_segmented_metrics`: Metrics by Ctr range (low/mid/high per D-18)
- `run_full_evaluation`: Main entrypoint orchestrating all metrics and figures
- `OUTPUT_NAMES` constant: ['log10_Ctr', 'inhibition_period', 'retardation_factor']

Created figures/.gitkeep placeholder for output directory.

**Commit:** 734bb2f

### Task 2: Evaluation Notebook
Created notebooks/03-evaluate.ipynb with 10 cells:
1. Title and description
2. sys.path setup
3. Configuration (device, checkpoint path, H5 paths)
4. Model loading with weights_only=True
5. Test set DataLoader construction
6. Run full evaluation
7. Metrics summary table
8. Segmented metrics by Ctr range
9. Per-RAFT-type metrics
10. Figure documentation with retardation factor note

**Commit:** cd9be4a

## Deviations from Plan

None - plan executed exactly as written.

## Verification Results

✅ All evaluate functions importable
✅ OUTPUT_NAMES verified: ['log10_Ctr', 'inhibition_period', 'retardation_factor']
✅ figures/.gitkeep exists
✅ Notebook has 10 cells
✅ grep verifications passed:
  - `def run_full_evaluation` found
  - `def save_per_class_parity_plots` found
  - retardation NOTE comment found
✅ pytest tests/ -m "not slow" -x: 78 passed, 0 failed

## Key Implementation Details

**Figure generation pattern:**
- All plotting functions accept numpy arrays, not tensors
- Matplotlib imported inside functions (lazy import for headless compatibility)
- rasterized=True enforced in visualization.py for scatter plots
- Figures saved at 150 DPI with tight bounding boxes

**Metrics structure:**
- Overall: R²/RMSE/MAE for all 3 outputs
- Segmented: Same metrics split by log10(Ctr) range (low/mid/high)
- By-class: Same metrics split by RAFT type (4 types)
- Outliers: Fraction of samples with |pred-true| > 2σ per output

**Expected behavior documented:**
Retardation factor R² near 0 for TTC/xanthate/dithiocarbamate is physically correct - these RAFT types have retardation_factor ≈ 1.0 (trivial prediction). This is NOT a model failure.

## Known Stubs

None - all functions are complete implementations.

## Integration Points

**Upstream dependencies:**
- src/model.py: SimpViT model definition
- src/dataset.py: CombinedHDF5Dataset
- src/utils/split.py: build_stratified_indices, RAFT_TYPES
- src/utils/metrics.py: r2_score_np, rmse_np, mae_np
- src/utils/visualization.py: parity_plot, residual_hist

**Downstream usage:**
- Phase 6 paper writing will use generated figures directly
- Bootstrap UQ (Plan 05) will call run_full_evaluation on bootstrap ensemble
- Colab training notebook can import run_full_evaluation for post-training evaluation

## Next Steps

Plan 03-05: Bootstrap uncertainty quantification
- Lightweight bootstrap: freeze backbone, fine-tune fc layer only
- 200 bootstrap iterations × 5 epochs each
- F-distribution JCI with p=3 (3 outputs)
- Save bootstrap_heads.pth and calibration.json

---

**Self-Check: PASSED**

Files created:
✅ notebooks/03-evaluate.ipynb exists
✅ figures/.gitkeep exists

Files modified:
✅ src/evaluate.py contains run_full_evaluation
✅ src/evaluate.py contains save_per_class_parity_plots
✅ src/evaluate.py contains OUTPUT_NAMES constant

Commits:
✅ 734bb2f: feat(03-04): add figure generation functions to evaluate.py
✅ cd9be4a: feat(03-04): add evaluation notebook

Tests:
✅ 78 tests passed, 0 failed
