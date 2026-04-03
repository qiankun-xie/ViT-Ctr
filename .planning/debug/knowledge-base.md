# GSD Debug Knowledge Base

Resolved debug sessions. Used by `gsd-debugger` to surface known-pattern hypotheses at the start of new investigations.

---

## ml-model-literature-validation-failure — Stale documentation reported 484x fold error but model actually performs well (1.17x)
- **Date:** 2026-04-03
- **Error patterns:** 484x fold error, catastrophically wrong predictions, median fold error, R² negative, literature validation, stale summary
- **Root cause:** The 04-02-SUMMARY.md contained stale validation results (484x fold error) from an earlier pipeline run before the ensemble prediction approach (ml_predict_ensemble) was added to literature_validation.py. The actual validation_summary.json on disk showed the model performs well: median fold error 1.17x, R²=0.991.
- **Fix:** Updated 04-02-SUMMARY.md to reflect the actual validation results from validation_summary.json. The bug was in the documentation, not the model.
- **Files changed:** .planning/phases/04-literature-validation-and-mayo-baseline/04-02-SUMMARY.md, notebooks/04-validate.ipynb
---
