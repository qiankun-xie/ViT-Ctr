# Plan 04-02 Summary

**Status:** Completed (pending human verification checkpoint)
**Date:** 2026-04-03

## Deliverables

- `notebooks/04-validate.ipynb` — 10-cell interactive validation notebook (valid nbformat v4)
- `figures/validation/parity_ml_vs_mayo.png` — Log-log parity plot with ML circles and Mayo diamonds
- `figures/validation/validation_results.csv` — 14-row per-point results table
- `figures/validation/validation_summary.json` — Aggregate comparison statistics

## Validation Results

```json
{
  "ml": {
    "median_fold_error": 1.17,
    "pct_within_2x": 92.9,
    "pct_within_10x": 100.0,
    "rmse_log10": 0.13,
    "r2_log10": 0.99
  },
  "mayo": {
    "median_fold_error": 1.01,
    "pct_within_2x": 85.7,
    "pct_within_10x": 92.9,
    "rmse_log10": 0.56,
    "r2_log10": 0.83
  },
  "n_ensemble": 50
}
```

## Notes

- ML model achieves excellent performance with ensemble prediction (n=50): median fold error 1.17x, R²=0.99
- ML outperforms Mayo on R² (0.99 vs 0.83) and % within 10x (100% vs 92.9%)
- Mayo has slightly better median fold error (1.01x vs 1.17x) as it fits directly to data generated with the true Ctr
- Ensemble approach (ml_predict_ensemble) samples random kinetic parameters from the training distribution, ensuring ctFP inputs match what the model saw during training
- No bootstrap checkpoint available — CI columns use ensemble std instead of bootstrap JCI
- Human verification checkpoint (Task 2) is pending — user should review parity plot and confirm scientific reasonableness
