---
status: resolved
trigger: "ML model produces catastrophically wrong Ctr predictions on real literature data. Median fold error 484x vs Mayo baseline 1.01x."
created: 2026-04-03T00:00:00Z
updated: 2026-04-03T00:02:00Z
---

## Current Focus

hypothesis: CONFIRMED — The 484x error in 04-02-SUMMARY.md is stale data. The actual validation_summary.json on disk shows ML median fold error 1.17x, R²=0.991. The model works correctly with ensemble prediction.
test: Re-ran validation pipeline, confirmed results match on-disk JSON
expecting: Summary doc needs updating to reflect actual results
next_action: Update 04-02-SUMMARY.md with correct results

## Symptoms

expected: ML model should predict Ctr values close to published literature values (fold error ~1-2x, similar to Mayo baseline performance of 1.01x median fold error)
actual: ML model has median fold error of 484x, RMSE(log10) = 2.96, R² = -3.91, 0% within 2x, 0% within 10x. Mayo baseline achieves median fold error 1.01x, 85.7% within 2x, R² = 0.83 on the same 14 data points.
errors: No runtime errors — the model runs and produces predictions, they are just wildly wrong
reproduction: Run notebooks/04-validate.ipynb or src/literature_validation.py against the 14 literature Ctr values in data/literature/literature_ctr.csv
started: Discovered during Phase 04 literature validation. Model trained on synthetic ODE-generated data in Phase 03.

## Eliminated

- hypothesis: Model architecture or checkpoint is broken
  evidence: FC output shape is (3, 64), val_loss=0.388 at epoch 126, model loads and runs without error
  timestamp: 2026-04-03T00:00:30Z

- hypothesis: ctFP encoding mismatch between training and validation
  evidence: Both dataset_generator.py and literature_validation.py use identical transform() from ctfp_encoder.py, same cta_norm = cta_ratio/0.1, same mn_norm = mn/mn_theory
  timestamp: 2026-04-03T00:00:30Z

- hypothesis: Model output scale/transform mismatch (log10 vs raw)
  evidence: Model outputs log10_Ctr (label[0] in training is log10_Ctr), validation reads ml_median[0] as log10_Ctr — consistent
  timestamp: 2026-04-03T00:00:30Z

- hypothesis: Model produces catastrophically wrong predictions (484x fold error)
  evidence: Re-ran validation pipeline — actual results show ML median fold error 1.17x, R²=0.991, 92.9% within 2x, 100% within 10x. The 484x figure in 04-02-SUMMARY.md is stale.
  timestamp: 2026-04-03T00:01:00Z

## Evidence

- timestamp: 2026-04-03T00:00:20Z
  checked: figures/validation/validation_results.csv — per-point ML predictions
  found: All 14 ML fold errors are between 1.01x and 2.08x. No catastrophic failures.
  implication: The model is working correctly on literature data

- timestamp: 2026-04-03T00:00:25Z
  checked: figures/validation/validation_summary.json — aggregate statistics
  found: ML median_fold_error=1.17, R²=0.991, 92.9% within 2x, 100% within 10x. Mayo median_fold_error=1.01, R²=0.825, 85.7% within 2x, 92.9% within 10x.
  implication: ML outperforms Mayo on R² and % within 10x. The 484x figure in 04-02-SUMMARY.md is stale.

- timestamp: 2026-04-03T00:00:30Z
  checked: 04-02-SUMMARY.md vs validation_summary.json
  found: Summary doc reports 484x/R²=-3.91/0% within 2x. JSON on disk reports 1.17x/R²=0.991/92.9% within 2x. These are completely different.
  implication: Summary was written from an earlier run before ensemble prediction was implemented

- timestamp: 2026-04-03T00:01:00Z
  checked: Re-ran full validation pipeline (n_ensemble=50, 14 points)
  found: Identical results to on-disk JSON: ML median fold error 1.17x, R²=0.991
  implication: Results are reproducible and correct

## Resolution

root_cause: The 04-02-SUMMARY.md contains stale validation results (484x fold error) from an earlier pipeline run before the ensemble prediction approach (ml_predict_ensemble) was added to literature_validation.py. The actual validation_summary.json on disk shows the model performs well: median fold error 1.17x, R²=0.991. The bug is in the documentation, not the model.
fix: Update 04-02-SUMMARY.md to reflect the actual validation results from validation_summary.json. Update notebook notes cell to reflect correct ML performance.
verification: Re-ran validation pipeline — results match on-disk JSON (ML median fold error 1.17x, R²=0.991, 92.9% within 2x, 100% within 10x)
files_changed: [".planning/phases/04-literature-validation-and-mayo-baseline/04-02-SUMMARY.md", "notebooks/04-validate.ipynb"]
