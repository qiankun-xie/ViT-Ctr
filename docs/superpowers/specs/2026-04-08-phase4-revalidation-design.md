# Phase 4 Re-validation Design

**Date:** 2026-04-08
**Status:** Approved
**Scope:** Re-run Ctr validation with expanded 77-point dataset; add qualitative validation for inhibition period and retardation factor

## Context

Phase 4 was originally completed with 14 literature Ctr values. The dataset has since been expanded to 77 points (Chong 2003, Moad 2009, Moad 2012). Additionally, the model's two other outputs (inhibition period, retardation factor) were never validated against literature expectations.

## Changes

### A. Ctr Re-validation (77 points)

The existing `run_validation_pipeline` in `src/literature_validation.py` already handles arbitrary CSV sizes. No logic changes needed for Ctr validation itself.

Updates:
- Notebook `04-validate.ipynb`: update title/description from 14 to actual count, re-run all cells
- Parity plot: x-axis range already [-1, 5], sufficient for new data range (log10_Ctr: -1.52 to 3.78)
- Summary docs and ROADMAP success criteria: update point count

### B. Inhibition Period & Retardation Factor Qualitative Validation

**Why qualitative:** No published quantitative values exist for these parameters in the RAFT literature. They are observed as qualitative phenomena (e.g., "significant retardation observed" or "no inhibition period").

**Known chemistry expectations:**
- Dithioester: elevated inhibition period (> 0), retardation factor < 1.0 (significant retardation due to slow fragmentation of intermediate radical)
- Trithiocarbonate: inhibition ≈ 0, retardation ≈ 1.0 (ideal RAFT behavior)
- Xanthate: inhibition ≈ 0, retardation ≈ 1.0
- Dithiocarbamate: inhibition ≈ 0, retardation ≈ 1.0

**Implementation:**

1. `src/literature_validation.py` — `run_validation_pipeline`:
   - Extract `ml_median[1]` and `ml_median[2]` into results DataFrame as `ml_inhibition` and `ml_retardation`
   - Also extract `ml_std[1]` and `ml_std[2]` as `ml_inhibition_std` and `ml_retardation_std`

2. `src/literature_validation.py` — new function `plot_inhibition_retardation_by_class(results_df, output_dir)`:
   - Two-panel figure (1×2): left = inhibition period by RAFT type, right = retardation factor by RAFT type
   - Strip plot with per-point jitter, colored by RAFT type (using existing RAFT_COLORS)
   - Horizontal reference lines: inhibition=0, retardation=1.0
   - Saved as `figures/validation/inhibition_retardation_by_class.png`

3. `notebooks/04-validate.ipynb`:
   - Add cells displaying inhibition/retardation columns in results table
   - Add cell for per-RAFT-type summary statistics (mean ± std)
   - Add cell displaying the new figure
   - Add markdown cell with chemistry interpretation

4. Tests: verify new columns exist in pipeline output and plotting function runs without error

### What does NOT change
- `ml_predict_ensemble` — already returns all 3 outputs
- `bootstrap.py` — inference interface unchanged
- Mayo fitter — only fits Ctr (no Mayo equivalent for inhibition/retardation)
- `ctfp_encoder.py`, `model.py`, `raft_ode.py` — untouched

## File Impact

| File | Change |
|------|--------|
| `src/literature_validation.py` | Add inhibition/retardation to results; new plot function |
| `notebooks/04-validate.ipynb` | Update for 77 points; add inhibition/retardation cells |
| `tests/test_literature_validation.py` | Add tests for new columns and plot function |

## Success Criteria

1. Validation runs on all 77 literature points without error
2. Results DataFrame contains `ml_inhibition`, `ml_retardation` columns
3. Dithioester points show higher inhibition and lower retardation than other RAFT types
4. TTC/xanthate/dithiocarbamate points show retardation ≈ 1.0 and inhibition ≈ 0
5. Parity plot and summary statistics updated for 77-point dataset
