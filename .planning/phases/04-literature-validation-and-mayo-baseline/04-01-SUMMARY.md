# Plan 04-01 Summary

**Status:** Completed (revalidated with 77-point dataset)
**Date:** 2026-04-08 (original: 2026-04-03)

## Deliverables

- `data/literature/literature_ctr.csv` — 77 curated literature Ctr values (33 dithioester, 30 trithiocarbonate, 4 xanthate, 10 dithiocarbamate)
- `src/literature_validation.py` — Mayo ODE fitter, fold-error computation, ensemble ML prediction, and end-to-end validation pipeline

## Verification Results

```
CSV validation passed: 77 rows, 4 RAFT types, 2 methods (Kinetic simulation, Mayo)
12 monomers covered (Styrene, MMA, MA, BA, AA, NIPAm, GMA, PEGMA, DADMAC, NVC, VAc, NVP)
All exports verified
fold_error math verified
```

## Notes

- Revalidation expanded dataset from 14 → 77 points (Chong 2003, Moad 2009, Moad 2012, and additional sources)
- Added 6 new monomer MWs to MONOMER_MW: AA, NIPAm, GMA, PEGMA, DADMAC, NVC
- New CSV columns: `ctr_type` (Ctr vs Ctr_app), `source_location` (page/table reference)
- Mayo fitter uses `minimize_scalar` with `method='bounded'` for single-parameter Ctr optimization
- Fixed params set to geometric means of training distribution bounds (D-06): kp=1000, kt=10^7.5, kd=1e-5, f=0.65
- `trithiocarbonate` mapped to `'ttc'` for ODE simulator compatibility
- Pipeline ran successfully: 77/77 points validated in ~41 min on CPU
