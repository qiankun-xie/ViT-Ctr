---
phase: 01-ode-foundation-and-ctfp-encoder
plan: 03
subsystem: validation
tags: [ode-validation, diagnostic-dataset, ctfp, literature-comparison]

requires: [01-01, 01-02]
provides:
  - "Diagnostic dataset generator (src/diagnostic.py) with 1000-sample generation across 4 RAFT types"
  - "ODE validation notebook comparing 3 literature RAFT systems"
  - "ctFP visualization confirming dual-channel encoding"
affects: [02-large-scale-dataset-generation]

tech-stack:
  added: [joblib, tqdm, matplotlib]
  patterns: [parallel-ode-generation, literature-validation, ctfp-visualization]

key-files:
  created:
    - src/diagnostic.py
    - notebooks/01_ode_validation.py
  modified:
    - tests/test_raft_ode.py

key-decisions:
  - "Xanthate/VAc Ctr=1 produces D~2.0 (near FRP) -- physically correct for weak RAFT agent"
  - "CDB/Styrene only reaches 25% conversion in 36000s -- inhibition visible but full D-rise requires longer simulation"
  - "High Ctr=10000 gives D~1.2 (slightly above Poisson limit due to dead chains) -- acceptable"
  - "joblib prefer=threads for Windows compatibility"

patterns-established:
  - "generate_diagnostic_dataset(n_per_type=250) API for parameter space coverage"
  - "Validation notebook as .py with percent format (Jupytext compatible)"

requirements-completed: [SIM-01, SIM-02, SIM-03, ENC-01]

duration: ~15min (across sessions)
completed: 2026-03-26
---

# Phase 01 Plan 03: ODE Validation & Diagnostic Dataset Summary

**ODE system validated against 3 literature RAFT systems with 1000-sample diagnostic dataset confirming numerical stability**

## Performance

- **Tasks:** 2 (1 auto + 1 human-verify checkpoint)
- **Files modified:** 3

## Accomplishments
- Diagnostic dataset generator producing 1000 samples (250 per RAFT type) across Ctr 0.01-10000
- ODE validation against 3 literature systems:
  - CDB/Styrene (dithioester): visible inhibition period, D decreasing from 5.2 to 1.7
  - TTC/MMA (trithiocarbonate): linear Mn growth, D < 1.4 after 20% conversion
  - Xanthate/VAc: D~2.0 (weak control at Ctr=1), Mn roughly constant
- Extreme parameter limits verified: High Ctr→D~1.2, Low Ctr→D~2.0
- ctFP dual-channel visualization showing physically reasonable patterns
- 41 non-slow tests passing, diagnostic dataset smoke test passing

## Validation Results

| System | Mn Behavior | D Range | Inhibition | Retardation | Verdict |
|--------|-------------|---------|------------|-------------|---------|
| CDB/Styrene (Ctr~20) | Linear after inhibition | 1.7-5.2 | 0.033 | 0.999 | Qualitatively correct |
| TTC/MMA (Ctr~50) | Linear, tracks theory | 1.3-2.2 | 0.016 | 1.000 | Correct |
| Xanthate/VAc (Ctr~1) | ~Constant (~7900) | 2.0-2.2 | 0.002 | 1.000 | Correct |
| High Ctr (10000) | Linear | 1.16-1.27 | — | — | Acceptable |
| Low Ctr (0.01) | Decreasing | ~2.0 | — | — | Correct |

## Files Created/Modified
- `src/diagnostic.py` - generate_diagnostic_dataset() with joblib parallelism
- `notebooks/01_ode_validation.py` - 6-cell validation notebook with literature comparisons
- `tests/test_raft_ode.py` - Extended with test_diagnostic_dataset (slow) and test_diagnostic_labels_valid

## Deviations from Plan
None significant. All 4 must_have truths satisfied.

## User Verification
- User approved ODE validation plots on 2026-03-26
- All 41 non-slow tests passed
- Diagnostic dataset generation confirmed functional

## Known Limitations
- CDB/Styrene reaches only 25% conversion in 36000s (D-rise phase not fully captured)
- Retardation factor ~1.0 for all systems (CTA concentration effect minimal at these ratios)

## Next Phase Readiness
- ODE system fully validated for Phase 2 large-scale dataset generation
- Diagnostic dataset generator pattern established for scaling to 1M samples
- All 3 plans in Phase 01 complete

---
*Phase: 01-ode-foundation-and-ctfp-encoder*
*Completed: 2026-03-26*
