---
phase: 06-paper-and-supporting-information
plan: 02
status: complete
started: 2026-04-08T18:50:00+08:00
completed: 2026-04-08T19:15:00+08:00
---

## Summary

Created complete English manuscript (`paper/manuscript.tex`) using achemso template targeting Macromolecules (ACS), plus BibTeX bibliography (`paper/references.bib`).

## What Was Built

- **paper/manuscript.tex** (242 lines): Full IMRAD manuscript with Abstract, Introduction, Computational Methods, Results and Discussion, Conclusions
- **paper/references.bib** (173 lines): 18 BibTeX entries covering all cited works

## Key Decisions

- Used `\section{Computational Methods}` instead of `\section{Methods}` per ACS convention
- Route A (SMILES → Ctr) positioned exclusively in Conclusions as future directions (PAP-03)
- Mayo baseline framed as validation evidence with note about idealized fixed kinetic parameters
- Three-parameter claim scoped to dithioester where retardation is meaningful
- All 4 limitations from D-15 covered in dedicated subsection

## Key Metrics Referenced

- ML: R²=0.968, RMSE=0.181, median fold-error=1.10, 92.2% within 2-fold, 100% within 10-fold
- Mayo: R²=0.502, RMSE=0.715
- Bootstrap: 200 heads, calibration factors [100.0, 53.7, 3.5]
- Coverage after calibration: [69.2%, 95.0%, 95.0%]

## Key Files

### Created
- paper/manuscript.tex
- paper/references.bib

## Self-Check: PASSED

- [x] achemso mamobx document class
- [x] Abstract, Introduction, Methods, Results, Conclusions sections
- [x] 6 includegraphics references
- [x] R²=0.968 cited
- [x] Route A in Conclusions only
- [x] Retardation scoped to dithioester
- [x] 18 BibTeX entries with Chong, Moad, Keddie

## Checkpoint Status

Task 2 (human-verify) pending — manuscript requires chapter-by-chapter review.
