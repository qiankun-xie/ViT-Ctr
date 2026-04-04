# Plan 04-01 Summary

**Status:** Completed
**Date:** 2026-04-03

## Deliverables

- `data/literature/literature_ctr.csv` — 14 curated literature Ctr values (4 dithioester, 4 trithiocarbonate, 3 xanthate, 3 dithiocarbamate)
- `src/literature_validation.py` — Mayo ODE fitter, fold-error computation, and end-to-end validation pipeline

## Verification Results

```
CSV validation passed: 14 rows, 4 RAFT types, 3 methods
All exports verified
fold_error math verified
```

## Notes

- Mayo fitter uses `minimize_scalar` with `method='bounded'` for single-parameter Ctr optimization
- Fixed params set to geometric means of training distribution bounds (D-06): kp=1000, kt=10^7.5, kd=1e-5, f=0.65
- `trithiocarbonate` mapped to `'ttc'` for ODE simulator compatibility
- Checkpoint loading handles wrapped format: `ckpt.get('model_state_dict', ckpt)`
- No bootstrap checkpoint found — CI columns will be NaN in validation results
- Pipeline ran successfully: 14/14 points validated in ~2 min on CPU
