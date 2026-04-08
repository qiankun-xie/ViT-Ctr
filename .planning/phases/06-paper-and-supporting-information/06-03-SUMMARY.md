---
phase: 06-paper-and-supporting-information
plan: 03
status: complete
started: 2026-04-08T18:50:00+08:00
completed: 2026-04-08T19:15:00+08:00
---

## Summary

Created complete Supporting Information document (`paper/supporting_information.tex`) with full ODE derivation matching `src/raft_ode.py`, dataset construction details, training hyperparameters, bootstrap UQ protocol, and all evaluation figures.

## What Was Built

- **paper/supporting_information.tex** (582 lines): Complete SI with 5 sections

## Sections

1. **S1: RAFT Kinetic ODE Derivation** — 14-var single-eq model (TTC/xanthate/dithiocarbamate) + 16-var pre-eq model (dithioester), moment exchange model, observable quantities
2. **S2: Dataset Construction** — Parameter ranges table matching PARAM_BOUNDS, 4 RAFT types with fixed params, noise (σ=0.03), HDF5 storage
3. **S3: Training Hyperparameters** — Table with Adam, lr=3e-4, batch=256, 142 epochs (best at 126)
4. **S4: Bootstrap UQ Protocol** — 200 heads, 5 epochs/head, F-dist JCI (p=3, n=200), calibration table
5. **S5: Complete Evaluation Figures** — 12 per-class parity plots + 3 residual distributions

## Key Decisions

- ODE equations transcribed directly from `src/raft_ode.py` with consistent notation
- Pre-equilibrium model clearly separated with additional variables CTA₀ and Int_pre
- Calibration table includes before/after coverage values from calibration.json
- All 15 SI figures referenced via includegraphics

## Key Files

### Created
- paper/supporting_information.tex

## Self-Check: PASSED

- [x] 33 frac/dfrac instances (ODE equations present)
- [x] μ, ν, λ moment notation
- [x] k_add and k_frag rate constants
- [x] Pre-equilibrium model for dithioester
- [x] Latin Hypercube sampling documented
- [x] 200 bootstrap heads documented
- [x] 12 parity_by_class figure references
- [x] 3 residual figure references
- [x] 582 lines (above 300 min_lines requirement)

## Checkpoint Status

Task 2 (human-verify) pending — ODE derivation requires verification against code.
