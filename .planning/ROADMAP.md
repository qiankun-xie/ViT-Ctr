# Roadmap: ViT-Ctr

## Overview

Build a deep learning system that extracts three RAFT kinetic parameters simultaneously (chain transfer constant Ctr, inhibition period, retardation factor) from standard Mn and dispersity vs. conversion data. The build follows a strict sequential dependency: validate the ODE physics first, then generate a million-sample synthetic dataset, then train and evaluate SimpViT, then validate against published literature, then deploy as a Streamlit web app, then assemble the paper. Any shortcut in the early phases corrupts all downstream artifacts.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: ODE Foundation and ctFP Encoder** - Validate RAFT kinetic ODE and freeze the shared fingerprint encoder
- [x] **Phase 2: Large-Scale Dataset Generation** - Generate ~1M synthetic ctFP samples in chunked HDF5 format
- [x] **Phase 3: Model Training and Evaluation** - Train SimpViT with 3-parameter output; evaluate with bootstrap UQ
- [x] **Phase 4: Literature Validation and Mayo Baseline** - Curate published Ctr values; compare model vs. Mayo equation
- [x] **Phase 5: Streamlit Web Application** - Deploy interactive app with manual/Excel input, CI display, ctFP preview
- [ ] **Phase 6: Paper and Supporting Information** - Write manuscript and SI using finalized figures and validated results

## Phase Details

### Phase 1: ODE Foundation and ctFP Encoder
**Goal**: The RAFT kinetic ODE is validated against published curves and the shared ctFP encoder is frozen — the two root dependencies that all downstream phases build on
**Depends on**: Nothing (first phase)
**Requirements**: SIM-01, SIM-02, SIM-03, ENC-01, ENC-02
**Success Criteria** (what must be TRUE):
  1. The ODE simulator reproduces published Mn-conversion and D-conversion curves for at least three RAFT agent classes (dithioester, trithiocarbonate, xanthate) to within acceptable error
  2. The two-stage pre-equilibrium mechanism for dithioesters produces visibly distinct Mn/D curves from the single-equilibrium model, confirming the physically correct branch is implemented
  3. The ctFP encoder produces a 64×64×2 tensor from raw (CTA ratio, conversion, Mn, D) input, and the same function imported in both training and web-app contexts produces byte-identical outputs
  4. A 1,000-sample diagnostic dataset can be generated covering the full Ctr range (log10 = -1 to 4) across all four RAFT agent classes without numerical failures
**Plans:** 3 plans
Plans:
- [ ] 01-01-PLAN.md — RAFT ODE system (single-eq + pre-eq models) with tests
- [ ] 01-02-PLAN.md — ctFP encoder with tests
- [ ] 01-03-PLAN.md — ODE validation against literature + diagnostic dataset

### Phase 2: Large-Scale Dataset Generation
**Goal**: A validated ~1M sample ctFP dataset is stored in chunked HDF5 on Google Drive, ready for Colab training, stratified across all RAFT agent classes and the full Ctr parameter space
**Depends on**: Phase 1
**Requirements**: SIM-04
**Success Criteria** (what must be TRUE):
  1. The full dataset (~1M samples) generates without crash or data corruption, using joblib parallelism on the local CPU
  2. Generated samples are stored in chunked HDF5 files that can be loaded incrementally without exceeding 16 GB RAM
  3. Ctr values are log-uniformly distributed across the specified range, with no cluster gaps visible in the histogram
  4. Dataset files are accessible from Google Colab for Phase 3 training
**Plans**: 2 plans
Plans:
- [x] 02-01-PLAN.md — LHS parameter sampling and parallel dataset generation
- [x] 02-02-PLAN.md — Dataset validation and Google Drive upload

### Phase 3: Model Training and Evaluation
**Goal**: A trained SimpViT model with a 3-parameter output head achieves strong test-set performance on all three parameters, with calibrated bootstrap confidence intervals, and is ready for deployment
**Depends on**: Phase 2
**Requirements**: TRN-01, TRN-02, TRN-03, EVL-01, EVL-02, EVL-03, UQ-01, UQ-02
**Success Criteria** (what must be TRUE):
  1. SimpViT converges on all three outputs (log10(Ctr), inhibition period, retardation factor) with loss curves showing no divergence or plateau artifacts
  2. Test-set parity plots for all three parameters show tight clustering around the identity line, with R2 reported per output
  3. Per-RAFT-agent-class evaluation shows acceptable performance across all four RAFT classes
  4. Bootstrap 95% CI coverage on a held-out calibration set reaches the nominal level (or a scalar correction is documented and applied)
**Plans**: 5 plans
Plans:
- [x] 03-01-PLAN.md — SimpViT model (num_outputs=3) + metrics and visualization utilities
- [x] 03-02-PLAN.md — Data pipeline: stratified split, CombinedHDF5Dataset, per-class evaluation stub
- [x] 03-03-PLAN.md — Training loop: weighted MSE, EarlyStopper, Colab training notebook
- [x] 03-04-PLAN.md — Evaluation: parity plots, per-class metrics, residual analysis
- [x] 03-05-PLAN.md — Bootstrap UQ: freeze backbone, 200 heads, F-dist JCI, post-hoc calibration

### Phase 4: Literature Validation and Mayo Baseline
**Goal**: The trained model is validated against 14 published Ctr values spanning 4 RAFT agent classes and 3 measurement methods, with fold-errors reported and a Mayo equation ODE-fitting baseline comparison included
**Depends on**: Phase 3
**Requirements**: VAL-01, VAL-02, VAL-03, EVL-04
**Success Criteria** (what must be TRUE):
  1. A curated dataset of 14 published Ctr values exists, each annotated with RAFT agent class, measurement method (Mayo / CLD / dispersity), temperature, solvent, and monomer
  2. Model predictions on the literature set are compared to published values with fold-error computed for each point, and results are broken down by measurement method
  3. The Mayo equation baseline is implemented and evaluated on the same literature set, enabling a direct accuracy comparison between ML and traditional methods
  4. Paper-ready validation figures (predicted vs. published Ctr, per-class breakdown) are produced
**Plans**: 2 plans
Plans:
- [x] 04-01-PLAN.md — Literature CSV dataset + Mayo fitter + fold-error module
- [x] 04-02-PLAN.md — Validation notebook, pipeline execution, and human verification

### Phase 5: Streamlit Web Application
**Goal**: A deployed Streamlit app lets researchers input experimental RAFT kinetic data and receive simultaneous predictions for Ctr, inhibition period, and retardation factor with confidence intervals
**Depends on**: Phase 3
**Requirements**: APP-01, APP-02, APP-03, APP-04, APP-05, APP-06
**Success Criteria** (what must be TRUE):
  1. A user can manually enter ([CTA]/[M], conversion, Mn, D) rows and receive three predicted parameters with 95% CI displayed
  2. A user can upload an Excel/CSV file and download a pre-formatted Excel template; predictions are generated from the upload without manual re-entry
  3. Invalid input (conversion outside (0,1), Mn <= 0, D < 1, fewer than 3 data points) is caught before inference and an explicit error message is shown
  4. The ctFP fingerprint is displayed as a dual-channel heatmap so the user can visually verify the encoding before accepting results
  5. Model weights load once per session via st.cache_resource with no repeated loading on reruns
**Plans**: 2 plans
Plans:
- [x] 05-01-PLAN.md — App utility module (validation, normalization bridge, template, result formatting) + tests
- [x] 05-02-PLAN.md — Streamlit app (app.py) wiring UI to inference pipeline + visual verification

### Phase 6: Paper and Supporting Information
**Goal**: A complete draft manuscript and Supporting Information exist, with all figures generated from finalized model results, ODE derivations consistent with code, and Route A framed as future work
**Depends on**: Phase 4
**Requirements**: PAP-01, PAP-02, PAP-03
**Success Criteria** (what must be TRUE):
  1. The paper's Introduction, Methods, Results, Discussion, and Conclusion sections are drafted in English, with the three-parameter simultaneous prediction claim scoped to the RAFT agent classes where retardation is reliably identifiable
  2. The Supporting Information contains the LaTeX ODE derivation, and the equations in the SI match the implemented ODE system in raft_ode.py
  3. Route A (molecular structure -> Ctr via SMILES) is included as a Discussion/Conclusion future-directions section, not as a standalone results claim
**Plans**: 4 plans
Plans:
- [ ] 06-01-PLAN.md — Figure generation (concept diagram, ctFP example, composite parity, TOC graphic)
- [ ] 06-02-PLAN.md — LaTeX manuscript (IMRAD + Route A as future work)
- [ ] 06-03-PLAN.md — Supporting Information (ODE derivation matching code + evaluation details)
- [ ] 06-04-PLAN.md — Word versions (English + Chinese .docx)

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5 -> 6

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. ODE Foundation and ctFP Encoder | 3/3 | Complete | 2026-03-26 |
| 2. Large-Scale Dataset Generation | 2/2 | Complete | 2026-03-27 |
| 3. Model Training and Evaluation | 5/5 | Complete | 2026-04-03 |
| 4. Literature Validation and Mayo Baseline | 2/2 | Complete | 2026-04-04 |
| 5. Streamlit Web Application | 2/2 | Complete | 2026-04-06 |
| 6. Paper and Supporting Information | 0/4 | In progress | - |

---
*Roadmap created: 2026-03-24*
*Last updated: 2026-04-06 — Phase 6 planned (4 plans, 3 waves)*
