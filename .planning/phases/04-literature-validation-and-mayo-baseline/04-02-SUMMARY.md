# Plan 04-02 Summary

**Status:** Completed (revalidated with 77-point dataset)
**Date:** 2026-04-08 (original: 2026-04-03)

## Deliverables

- `notebooks/04-validate.ipynb` — Interactive validation notebook (valid nbformat v4)
- `figures/validation/parity_ml_vs_mayo.png` — Log-log parity plot with ML circles and Mayo diamonds (77 points)
- `figures/validation/inhibition_retardation_by_class.png` — Strip plot of inhibition/retardation by RAFT class
- `figures/validation/validation_results.csv` — 77-row per-point results table
- `figures/validation/validation_summary.json` — Aggregate comparison statistics

## Validation Results

```json
{
  "ml": {
    "median_fold_error": 1.10,
    "pct_within_2x": 92.2,
    "pct_within_10x": 100.0,
    "rmse_log10": 0.18,
    "r2_log10": 0.97
  },
  "mayo": {
    "median_fold_error": 1.02,
    "pct_within_2x": 85.7,
    "pct_within_10x": 90.9,
    "rmse_log10": 0.72,
    "r2_log10": 0.50
  },
  "n_ensemble": 50
}
```

## Key Findings (77-point revalidation vs 14-point original)

| Metric | ML (14pt) | ML (77pt) | Mayo (14pt) | Mayo (77pt) |
|--------|-----------|-----------|-------------|-------------|
| Median fold-error | 1.17 | 1.10 | 1.01 | 1.02 |
| % within 2x | 92.9% | 92.2% | 85.7% | 85.7% |
| % within 10x | 100% | 100% | 92.9% | 90.9% |
| RMSE(log10) | 0.13 | 0.18 | 0.56 | 0.72 |
| R² | 0.99 | 0.97 | 0.83 | 0.50 |

## Notes

- ML maintains strong performance at 5.5x scale: R²=0.97, 100% within 10x, median fold-error 1.10
- ML decisively outperforms Mayo on R² (0.97 vs 0.50) and within-10x (100% vs 90.9%)
- Mayo R² dropped from 0.83 to 0.50 with expanded dataset — fixed-parameter ODE fitting struggles with diverse RAFT agent structures
- Ensemble approach (n=50) samples random kinetic parameters from training distribution
- Bootstrap CI available via bootstrap_heads.pth + calibration.json
- Full run: 77 points × 50 ensemble samples = ~41 min on CPU
