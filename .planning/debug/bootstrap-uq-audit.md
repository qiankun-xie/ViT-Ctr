---
status: awaiting_human_verify
trigger: "Full systematic audit of Phase 3 bootstrap UQ implementation"
created: 2026-04-05T00:00:00Z
updated: 2026-04-05T00:00:00Z
---

## Current Focus

hypothesis: Two bugs found and fixed; audit complete
test: All 4 existing tests pass; grep confirms no stale references remain
expecting: User confirms fixes are correct
next_action: Await human verification

## Symptoms

expected: Bootstrap UQ correctly implements freeze backbone + fine-tune fc, with-replacement resampling, F-distribution 95% JCI, post-hoc calibration — faithful to ViT-RR
actual: Unknown — preventive audit before 20-hour training run
errors: None reported
reproduction: N/A — code review
started: Phase 3 development

## Eliminated

- hypothesis: freeze_backbone incorrectly freezes/unfreezes params
  evidence: Runtime test confirms only fc.weight and fc.bias have requires_grad=True after freeze. All 39 backbone params frozen, 2 fc params unfrozen.
  timestamp: 2026-04-05

- hypothesis: base_state corrupted during bootstrap iterations
  evidence: load_state_dict copies values, doesn't take references. base_state preserved across all iterations. Verified with runtime test.
  timestamp: 2026-04-05

- hypothesis: head state dict misses fc params
  evidence: k.startswith('fc') correctly captures fc.weight and fc.bias — the only two fc keys in SimpViT.
  timestamp: 2026-04-05

- hypothesis: compute_jci formula incorrect vs ViT-RR
  evidence: Formula matches exactly: sqrt(diag(cov) * p * f_val / dfd) with dfd=n-p, f_val=f.ppf(0.95, dfn=p, dfd=dfd). Verified numerically.
  timestamp: 2026-04-05

- hypothesis: calibrate_coverage binary search has edge case bugs
  evidence: Tested wide CI (factor~1.0), exact match (factor~1.0), and noisy predictions (factor>1.0). All produce correct calibrated coverage >= 0.95.
  timestamp: 2026-04-05

- hypothesis: RNG resume in autodl produces different sequences
  evidence: Fast-forwarding np.random.default_rng by calling choice() produces identical sequences as continuous generation. Verified.
  timestamp: 2026-04-05

- hypothesis: device mismatch between base_state and head_state
  evidence: head_state always .cpu() (bootstrap.py line 58). bootstrap_ckpt loaded with map_location='cpu'. No mismatch possible.
  timestamp: 2026-04-05

- hypothesis: predict_with_uncertainty cov_matrix calculation wrong
  evidence: np.cov(predictions, rowvar=False) with shape (n, 3) -> (3, 3) is correct. Uses ddof=1 by default, matching the F-distribution formula derivation.
  timestamp: 2026-04-05

- hypothesis: weighted_mse_loss used inconsistently
  evidence: Both bootstrap.py and autodl_bootstrap.py import and use weighted_mse_loss from train.py. Consistent.
  timestamp: 2026-04-05

## Evidence

- timestamp: 2026-04-05
  checked: freeze_backbone runtime behavior
  found: Correctly freezes all 39 non-fc params, leaves fc.weight and fc.bias trainable
  implication: Backbone freeze logic is correct

- timestamp: 2026-04-05
  checked: SimpViT parameter structure
  found: 41 total params, 2 fc params. Note: transformer_encoder_layer has unused duplicate params (PyTorch TransformerEncoder deep-copies the template layer). Not a bug.
  implication: No hidden fc-like params missed by startswith('fc')

- timestamp: 2026-04-05
  checked: compute_jci formula vs ViT-RR reference
  found: Exact match. sqrt(diag(cov) * p * f_val / dfd), dfd=n-p, f_val=f.ppf(0.95, dfn=p, dfd=dfd)
  implication: JCI formula is correct

- timestamp: 2026-04-05
  checked: Colab cell 9 JCI calculation vs compute_jci
  found: BUG — cell 9 uses val_pred_std = val_preds_all.std(axis=0) which defaults to ddof=0, then val_pred_std**2. But np.cov uses ddof=1. For n=200, this causes ~0.25% underestimate of half-width.
  implication: Minor numerical inconsistency between cell 9/autodl and predict_with_uncertainty

- timestamp: 2026-04-05
  checked: autodl_bootstrap.py run_calibration JCI calculation
  found: Same ddof=0 bug as Colab cell 9 (line 208: val_pred_std**2 instead of variance with ddof=1)
  implication: Both Colab and autodl have the same minor bug

- timestamp: 2026-04-05
  checked: Colab cell 9 import of run_inference
  found: BUG — `from evaluate import run_inference` but run_inference does not exist in evaluate.py. Only run_full_evaluation exists. Cell 9 will crash at runtime.
  implication: Colab notebook cell 9 is broken — cannot run calibration

- timestamp: 2026-04-05
  checked: 03-bootstrap-autodl.ipynb cell 7
  found: Same two bugs as colab notebook: run_inference import and ddof=0
  implication: Three files affected total

- timestamp: 2026-04-05
  checked: calibrate_coverage edge cases
  found: Binary search correctly handles wide CI (factor~1.0), exact match, and noisy predictions (factor>1.0). Returns hi (smallest factor achieving >= target).
  implication: Calibration logic is correct

- timestamp: 2026-04-05
  checked: RNG resume in autodl
  found: Fast-forwarding default_rng by calling choice() produces identical sequences
  implication: Resume logic is correct

- timestamp: 2026-04-05
  checked: predict_with_uncertainty post-call model state
  found: Model is left with last head's state after function returns. Not a correctness bug for return values, but model state is "dirty".
  implication: Minor — callers should be aware model state is modified

- timestamp: 2026-04-05
  checked: All tests pass after fixes
  found: 4/4 tests pass. No stale run_inference or val_pred_std references in source code.
  implication: Fixes are clean

## Resolution

root_cause: Two bugs found across three files:
  1. MODERATE: Both Colab notebooks import `run_inference` from evaluate.py, but this function was never implemented (replaced by run_full_evaluation with different signature). Calibration cells crash at runtime.
  2. MINOR: Colab notebooks and autodl_bootstrap.py use np.std(ddof=0)**2 instead of np.var(ddof=1) for JCI calculation, causing ~0.25% underestimate of half-width vs the correct ddof=1 used by np.cov in compute_jci/predict_with_uncertainty.

fix: |
  1. Replaced run_inference import with inline val label collection loop in both notebooks
  2. Changed .std(axis=0)**2 to .var(axis=0, ddof=1) in all three files

verification: All 4 bootstrap tests pass. Grep confirms no stale references remain in source code.
files_changed:
  - colab/03-bootstrap-colab.ipynb
  - colab/03-bootstrap-autodl.ipynb
  - colab/autodl_bootstrap.py
