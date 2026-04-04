---
phase: 05-streamlit-web-application
plan: 01
subsystem: ui
tags: [app_utils, validation, normalization, pandas, openpyxl, numpy]

requires:
  - phase: 03-bootstrap-uncertainty
    provides: "predict_with_uncertainty() return format (mean ndarray(3,), half_width ndarray(3,))"
  - phase: 02-ctfp-encoder
    provides: "transform() signature — expects (cta_ratio_norm, conversion, mn_norm, dispersity) tuples"
provides:
  - "validate_input() — drops NaN rows, checks >= 3 points, per-row validation with exact UI-SPEC error messages"
  - "prepare_ctfp_input() — normalizes [CTA]/[M]/0.1 and Mn/Mn_theory per-row; bridges raw user input to ctFP encoder format"
  - "generate_template() — returns BytesIO xlsx with 4 columns and 3 example rows"
  - "format_results() — back-transforms log10(Ctr) to 10^x with asymmetric CI; symmetric CI for inhibition/retardation"
affects: [05-02-PLAN]

tech-stack:
  added: []
  patterns:
    - "Normalization bridge: cta_ratio_norm = [CTA]/[M] / 0.1; mn_theory = m_monomer / [CTA]/[M]; mn_norm = Mn / mn_theory"
    - "Asymmetric CI for Ctr in original scale: 10^(log±hw), NOT ctr ± 10^hw"
    - "Validation returns (df_valid, errors) — caller decides whether to proceed on non-fatal errors"

key-files:
  created:
    - src/app_utils.py
    - tests/test_app_utils.py
  modified: []

key-decisions:
  - "Row numbering in error messages uses df_valid.index + 1 (original index + 1), not sequential position — preserves consistency with data_editor row numbers visible to user"
  - "validate_input returns (df_valid, errors) for per-row validation errors (non-fatal); returns (None, errors) only when < 3 valid points (fatal)"
  - "Column names use ASCII 'D' internally; Unicode Đ appears only in error messages via \\u0110 escape (RESEARCH Pitfall 6)"

requirements-completed: [APP-01, APP-02, APP-04]

duration: 20min
completed: 2026-04-04
---

# Phase 05 Plan 01: App Utilities Summary

**Pure-Python bridge module `src/app_utils.py` with 4 tested functions: input validation (per-row error messages matching UI-SPEC), normalization bridge (cta_ratio/0.1 + Mn/Mn_theory), xlsx template generator, and Ctr back-transform with asymmetric CI**

## Performance

- **Duration:** 20 min
- **Started:** 2026-04-04T02:40:28Z
- **Completed:** 2026-04-04T03:01:22Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 2

## Accomplishments
- 4 utility functions with complete validation, normalization, template generation, and result formatting
- 14 unit tests covering all behaviors from the plan's `<behavior>` spec (including NaN handling, asymmetric CI)
- All tests pass; no regressions in 45 fast tests

## Task Commits

1. **RED — Failing tests** - `d3025ac` (test(05-01))
2. **GREEN — Implementation** - `1cebd99` (feat(05-01))

## Files Created/Modified
- `src/app_utils.py` — 4 exported utility functions with docstrings (176 lines)
- `tests/test_app_utils.py` — 14 unit tests covering all specified behaviors (240 lines)

## Decisions Made
- Row numbering uses original DataFrame index + 1 to stay consistent with what the user sees in data_editor (matches RESEARCH.md code example)
- `validate_input` returns the clean DataFrame even when there are per-row validation errors — callers decide whether to proceed (only fatal case is < 3 valid points)
- ASCII `D` column name internally; `Đ` (U+0110) only in error message strings

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None — all 14 tests passed green on first implementation attempt.

## Next Phase Readiness
- `src/app_utils.py` is ready for import in Plan 02 (`app.py`)
- All 4 exports verified importable: `from src.app_utils import validate_input, prepare_ctfp_input, generate_template, format_results`
- Plan 02 can proceed immediately

## Self-Check: PASSED
- `src/app_utils.py` exists on disk ✓
- `tests/test_app_utils.py` exists on disk ✓
- `git log --oneline --grep="05-01"` returns 2 commits ✓
- 14 tests pass, 0 fail ✓

---
*Phase: 05-streamlit-web-application*
*Completed: 2026-04-04*
