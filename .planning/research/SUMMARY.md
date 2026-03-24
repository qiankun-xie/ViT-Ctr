# Project Research Summary

**Project:** ViT-Ctr — RAFT Chain Transfer Constant Prediction via Vision Transformer
**Domain:** Scientific ML for polymer chemistry — ODE simulation, Vision Transformer training, uncertainty quantification, web deployment
**Researched:** 2026-03-24
**Confidence:** HIGH

## Executive Summary

ViT-Ctr is a scientific ML system that extracts three RAFT kinetic parameters (chain transfer constant Ctr, inhibition period, retardation factor) from standard Mn and dispersity vs. conversion kinetic data, using a ctFP (chain-transfer fingerprint) image encoding fed to a SimpViT Vision Transformer. The system is a direct extension of the peer-reviewed ViT-RR paradigm (Angew. Chem. 2025): the ctFP encoding, SimpViT model architecture, bootstrap uncertainty quantification, and Streamlit deployment are all carried forward from ViT-RR. The primary changes are (1) adapting the ODE simulation to RAFT chemistry (replacing copolymerization kinetics), (2) expanding the output head from 2 to 3 parameters, and (3) covering all four major RAFT agent classes rather than a single chemistry type. The proven reference codebase makes the stack and architecture decisions essentially settled; the main intellectual challenges are in the RAFT ODE fidelity and the statistical robustness of the three-parameter simultaneous prediction claim.

The recommended approach is a strict sequential build: ODE foundation first (RAFT moment equations validated against published conversion-dispersity curves), then large-scale dataset generation in chunks using joblib parallelism and chunked HDF5 storage, then SimpViT training on Google Colab (MX350 is insufficient for 1M sample training), then rigorous evaluation and literature validation, then Streamlit web app deployment. Route A (structure-to-Ctr via SMILES) is explicitly deferred as a future-work direction — it does not block the primary deliverable and adds architectural complexity. The critical path has no shortcuts: ODE errors corrupt all downstream artifacts and would require regenerating millions of samples.

The central risk cluster is at the ODE and validation layers. RAFT kinetics are stiff (requiring `solve_ivp` with BDF/Radau) and mechanistically nuanced: dithioesters require a two-stage pre-equilibrium model that trithiocarbonates do not; using a single simplified ODE produces plausible-looking but chemically wrong training data that the model will silently memorize. The second major risk is that "three simultaneous parameters" is a strong claim that reviewers will probe: retardation factor is identifiable only for dithioester-class systems where retardation is non-trivial, and Ctr literature values are measurement-method-dependent. Both risks are addressable with careful ODE validation before data generation and annotated literature provenance tracking before writing the validation section.

## Key Findings

### Recommended Stack

The stack is inherited directly from ViT-RR and is mature, well-documented, and verified at the required scale. PyTorch 2.6+ with raw training loops (no Lightning), SciPy `solve_ivp` with `method='BDF'` or `'Radau'` for stiff RAFT ODEs, NumPy/pandas/openpyxl for data handling, joblib for parallel ODE generation, and Streamlit for web deployment. The only non-trivial storage decision is whether to use HDF5 (h5py) or chunked `.pt` files for the dataset: at 1M samples × 64×64×2 float32, uncompressed size is ~32 GB, making chunked storage mandatory. HDF5 is preferred for training. The hardware split is clear: ODE generation runs on CPU (i5-10210U + joblib), model training runs on Colab T4, inference runs on Streamlit Community Cloud CPU. See `.planning/research/STACK.md` for full version pins and installation instructions.

**Core technologies:**
- PyTorch 2.6+: model definition, training, DataLoader — proven in ViT-RR, SimpViT is ~3.4 MB
- SciPy 1.15.x (`solve_ivp`, BDF/Radau): RAFT ODE integration — required for stiff kinetics; `odeint` is inadequate
- NumPy 2.2.x: ctFP array construction, bootstrap statistics — central shared data type
- joblib 1.4.x: CPU-parallel ODE simulation across parameter grid — essential for million-sample generation
- Streamlit 1.40+: web app deployment — direct constraint from PROJECT.md; matches ViT-RR deploy.py
- h5py: chunked dataset storage for 32 GB training data — HDF5 required once dataset exceeds ~8 GB RAM limit
- pandas 2.2.x + openpyxl: Excel I/O for web app — established ViT-RR pattern; pin `<3.0` during dev

### Expected Features

See `.planning/research/FEATURES.md` for complexity estimates and dependency chain.

**Must have (table stakes — paper-critical):**
- ODE-based RAFT kinetic simulator with parameter sweep (all four RAFT agent classes; stiff solver required)
- ctFP encoding: 64×64 dual-channel (Mn, D) image, shared module used by both training pipeline and web app
- SimpViT with 3-output head: log10(Ctr), inhibition period, retardation factor
- Train/val/test split stratified by Ctr range (not random — data leakage risk)
- Loss in log10(Ctr) space; normalize inhibition and retardation targets separately
- Test set metrics: R², RMSE, MAE per output + parity plots (one per output)
- Literature validation against 10+ published Ctr values, multiple RAFT agent classes, annotated by method
- Bootstrap UQ: 200 iterations, F-distribution JCI (exact ViT-RR pattern)
- Streamlit app: manual input + Excel upload, three-parameter output with CI, input validation, Excel template download

**Should have (strengthens paper):**
- Comparison against Mayo equation baseline on the same literature validation set
- ctFP heatmap visualization for users and SI figures
- Per-RAFT-agent-class validation breakdown in paper
- User registration to Google Sheets (usage tracking for paper narrative)
- OOD detection flag in web app when ctFP has fewer than 3 activated pixels

**Defer to future work:**
- Route A (SMILES → Ctr): independent branch, different architecture, limited data — position as paper conclusion future direction
- Attention map visualization: noisy on sparse ctFP; misleads rather than informs
- Generative/inverse design, REST API, multi-language UI, mobile-responsive layout

### Architecture Approach

The system uses the same three-layer architecture as ViT-RR: (1) data generation layer (ODE simulation → ctFP encoding → chunked dataset), (2) model training and evaluation layer (SimpViT + bootstrap UQ → trained weights + paper figures), (3) web deployment layer (Streamlit app consuming shared encoder module + model weights). The critical architectural decision is to extract `ctfp_encoder.py` as a standalone importable module from the start — any encoding divergence between training and inference silently invalidates all predictions. The 6-phase build order maps directly to this layer structure, with Phase 1 (ODE + encoder foundation) being the highest-leverage investment because errors there cascade to all downstream phases.

**Major components:**
1. `raft_ode.py` + `param_sampler.py` — two-stage RAFT moment equations (pre-equilibrium + main equilibrium), parameter space sampling log-scaled by RAFT agent class
2. `ctfp_encoder.py` (shared module) — converts (CTA ratio, conversion, Mn, D) rows to 64×64×2 tensor; imported identically by dataset builder and Streamlit app
3. `dataset_builder.py` — joblib-parallel ODE runs, chunked HDF5 output, ~1M samples
4. `model.py` — SimpViT: Conv2d patch embedding (2→64, kernel=16), 2× TransformerEncoderLayer (d=64, heads=4), mean pool, Linear(64, 3)
5. `train.py` + `evaluate.py` — DataLoader with block train/test split, Adam lr=3e-4, MSELoss in log space; bootstrap CI, parity plots, literature validation
6. `deploy.py` (Streamlit) — data input (manual + Excel), ctFP encoding, model inference, bootstrap UQ display, input validation, ctFP preview

### Critical Pitfalls

Full pitfall catalog with phase warnings in `.planning/research/PITFALLS.md`. Top five by severity:

1. **Wrong RAFT kinetic model for dithioesters (C1)** — Using a single-equilibrium ODE that omits the pre-equilibrium stage produces plausible but wrong Mn/D curves for dithioesters and xanthates. The model trains on corrupted fingerprints, memorizes the corruption, and fails on real experimental data. Prevention: implement two-stage ODE; validate against published curves for CDB/St (dithioester), a TTC system, and EtXan/VAc (xanthate) before generating any training data. Write the ODE in LaTeX first, then implement from LaTeX.

2. **Parameter space coverage gap — silent extrapolation (C2)** — Model silently extrapolates outside training parameter ranges; bootstrap CI does not detect distributional mismatch. Prevention: sample Ctr on log scale across [-1, 4] (log10 units); include deliberate boundary cases; add OOD detection flag (count non-zero ctFP pixels) in the web app.

3. **Incommensurable Ctr measurement methods in literature validation (C3)** — Mayo, CLD slope, and dispersity-based Ctr values are not directly comparable; mixing them inflates RMSE for non-physical reasons. Prevention: annotate every validation point with measurement method, [CTA]/[M], temperature, solvent; report separate RMSEs per method; define precisely what Ctr the ODE represents and state it explicitly in the paper.

4. **ctFP pixel collision for single-[CTA]/[M] experimental data (C4)** — Users running a single-[CTA]/[M] kinetic run place all data points in one pixel column; later entries overwrite earlier ones silently. Prevention: add pre-encoding deduplication warning; display ctFP preview in web app before inference; document minimum data requirements (at least 2 distinct [CTA]/[M] values, or use 1D mode).

5. **Bootstrap CI underestimates true uncertainty without post-hoc calibration (M3)** — Raw bootstrap std. dev. systematically underestimates coverage (empirical coverage can be 30–50% below nominal). Prevention: hold out a calibration set after training; measure empirical coverage vs. nominal; apply scalar correction; report calibration curve in SI.

## Implications for Roadmap

Based on research, the dependency graph from ARCHITECTURE.md directly dictates a 6-phase structure. Phases cannot be reordered without creating rework risk.

### Phase 1: ODE Foundation and ctFP Encoder

**Rationale:** The ODE is the root dependency for every other artifact. Errors discovered in Phase 4 (evaluation) require regenerating the entire dataset. Investing in ODE validation before dataset generation is the highest-leverage risk mitigation in the project. The ctFP encoder belongs here because it is a shared dependency of both Phase 2 and Phase 5 and must be frozen before either begins.
**Delivers:** `raft_ode.py` (validated two-stage RAFT moment equations), `param_sampler.py` (log-scale parameter grid across all four RAFT agent classes), `ctfp_encoder.py` (shared encoding module, tested in isolation), and a small diagnostic dataset (1,000 samples) used to validate ODE output against published curves.
**Addresses:** Table-stakes features — ODE simulator, parameter space sweep, ctFP encoding.
**Avoids:** C1 (wrong kinetic model), C2 (parameter space gaps), C4 (encoding ambiguity), M1 (ODE numerical instability), m5 (SI vs. code divergence).

### Phase 2: Large-Scale Dataset Generation

**Rationale:** Once the ODE and encoder are validated, dataset generation is a parallelizable CPU task. Doing it as a separate phase lets the generation run on the i5-10210U overnight or over multiple sessions while other planning proceeds. The chunked HDF5 strategy must be decided here before storage commitments are made.
**Delivers:** ~1M sample dataset in chunked HDF5 (`dataset_chunk_000.pt` through `dataset_chunk_009.pt`), stratified by RAFT agent class, stored to Google Drive for Colab access.
**Uses:** joblib (parallel ODE), h5py (chunked storage), tqdm (progress tracking).
**Avoids:** C2 (log-scale Ctr sampling, boundary case inclusion), M4 (block train/val/test splits by Ctr range, not random).

### Phase 3: Model Training and Evaluation

**Rationale:** Training requires the complete dataset and a fixed encoder. Evaluation (test metrics, parity plots, bootstrap CI) is grouped with training because they share the same model checkpoint and the results feed directly into the paper. All three model outputs must be validated per RAFT agent class here before proceeding to deployment.
**Delivers:** Trained SimpViT weights (`.pth`), bootstrap ensemble statistics, test-set R²/RMSE/MAE per output, parity plots, ctFP heatmap figures for SI.
**Implements:** `model.py` (SimpViT, output_dim=3), `train.py` (Adam, MSELoss in log space, block splits), `evaluate.py` (bootstrap UQ, F-distribution JCI, calibration check).
**Avoids:** M2 (log-scale target normalization), M3 (bootstrap calibration), M4 (data leakage), C5 (per-class retardation identifiability assessment).

### Phase 4: Literature Validation and Mayo Baseline

**Rationale:** Literature validation is the paper's primary empirical claim beyond synthetic performance. It requires a separately curated dataset of published Ctr values annotated by measurement method — this curation effort is non-trivial and belongs in its own phase. The Mayo equation baseline comparison belongs here because it uses the same literature validation set.
**Delivers:** Literature validation dataset (10+ Ctr values, annotated by method/RAFT class/conditions), model predictions vs. literature with fold-error and per-class breakdown, Mayo equation baseline comparison, paper-ready figures.
**Avoids:** C3 (incommensurable Ctr methods — annotation required), m3 (correct metric selection: fold-error, per-class RMSE, not just log-scale RMSE).

### Phase 5: Streamlit Web Application

**Rationale:** The web app depends on frozen model weights and the shared ctFP encoder, but has no dependency on the literature validation dataset. It can proceed in parallel with Phase 4 if time allows, but the natural sequencing after evaluation ensures the deployed model is the validated one. Input validation and the ctFP preview are non-negotiable for preventing silent errors.
**Delivers:** Deployed Streamlit app on Community Cloud: manual entry + Excel upload, three-parameter output with JCI, ctFP preview, input validation with explicit errors, Excel template download, user registration to Google Sheets.
**Uses:** Streamlit 1.40+, `st.cache_resource`, gspread, shared `ctfp_encoder.py`, trained `.pth`.
**Avoids:** C4 (ctFP preview + deduplication warning), M5 (strict input validation with unit checks), m2 (reduce bootstrap to 50 iterations in deployed app to stay within Streamlit Community Cloud 1 GB memory limit).

### Phase 6: Paper and Supporting Information Writing

**Rationale:** All quantitative results from Phases 3 and 4 must be finalized before the paper's Results section can be written. Phase 6 is not a "write-up at the end" phase — figures and SI derivations should be drafted throughout, but the final paper assembly happens here.
**Delivers:** Draft manuscript (Introduction, Methods, Results, Discussion), Supporting Information with LaTeX ODE derivation, SI consistency test passing, per-class validation tables, ctFP heatmap figures, Route A positioned as future work in Discussion.
**Avoids:** m4 (accurate framing of three-parameter claim), m5 (ODE in LaTeX matches code), C3 (measurement method provenance documented in SI), C5 (explicit statement of which RAFT classes allow reliable retardation prediction).

### Phase Ordering Rationale

- ODE correctness is the root of the dependency graph; no later phase can proceed safely without it validated. This alone justifies the Phase 1 focus.
- Large-scale generation (Phase 2) is deliberately isolated from training (Phase 3) because generation is hardware-bound to local CPU while training is hardware-bound to Colab GPU. Separating them allows asynchronous execution and clear checkpoints.
- Evaluation and training are grouped (Phase 3) because model selection (choosing the best checkpoint) requires evaluation metrics, and the bootstrap ensemble is trained alongside the primary model.
- Literature validation (Phase 4) is separated from synthetic evaluation because it requires externally sourced data curation that has its own timeline, and it is the empirical heart of the paper's claims.
- Web app (Phase 5) is last in the ML pipeline but does not block paper writing; it can overlap with Phase 6 if the validated model weights are available.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 1 (ODE Foundation):** The two-stage pre-equilibrium RAFT ODE for dithioesters is specialized — requires literature diving (Macromolecules 2022 dispersity model paper is the primary reference) to verify moment equations and identify which rate constants are independently estimable vs. must be fixed from literature.
- **Phase 4 (Literature Validation):** Ctr measurement method taxonomy and which literature sources to use requires domain expertise review. The retardation mechanism debate (IRT vs. SFM) must be settled before drafting the paper's Methods section.

Phases with standard patterns (research-phase likely unnecessary):
- **Phase 2 (Dataset Generation):** joblib + HDF5 chunking is a well-documented pattern; no research needed.
- **Phase 3 (Model Training):** SimpViT architecture and training loop are directly ported from ViT-RR; Adam + MSELoss in log space is standard.
- **Phase 5 (Streamlit App):** Web app pattern is a direct port from ViT-RR deploy.py with known modifications; no novel patterns needed.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Inherited directly from ViT-RR (peer-reviewed). All versions verified via PyPI/official sources. Hardware split (CPU generation, Colab training, Community Cloud deployment) is clear. |
| Features | MEDIUM-HIGH | ML pipeline features HIGH — established paradigm. RAFT-specific features (retardation factor identifiability, inhibition period from ctFP) MEDIUM — requires per-class validation to confirm. |
| Architecture | HIGH | Three-layer architecture verified from ViT-RR source code. Component boundaries, data flow, and shared encoder pattern are concrete and code-backed. |
| Pitfalls | HIGH | Chemical pitfalls (C1–C5) are grounded in RAFT literature; ML pitfalls (M1–M5) in calibration and generalization literature. Severity assessments are conservative. |

**Overall confidence:** HIGH

### Gaps to Address

- **RAFT ODE moment equations for pre-equilibrium:** The exact ODE system (which moments to track, how to handle CTA radical termination in the pre-equilibrium) must be verified against the Macromolecules 2022 dispersity model paper before any code is written. Do this before Phase 1 implementation begins.
- **Retardation factor target definition:** It is not yet decided whether retardation is expressed as a dimensionless rate ratio (r = kp_RAFT / kp_conventional) or a fractional value, or what range to expect for the four RAFT classes. This must be resolved in Phase 1 to set the output normalization strategy for Phase 3 training.
- **Bootstrap ensemble strategy for deployment:** The conflict between 200 models needed for accurate CI and the Streamlit Community Cloud ~1 GB memory limit must be resolved before Phase 5. Options are: (a) serialize ensemble statistics rather than model files, (b) reduce to 50 models in the deployed app, or (c) precompute a lookup table. Recommend deciding during Phase 3 once model size is known.
- **Literature validation dataset scope:** The 10+ validation points needed for the paper's literature validation section must be curated during Phase 4. A preliminary literature survey of available published Ctr values across all four RAFT agent classes should be done early (even in Phase 1) to confirm that sufficient validation data exists before training is complete.

## Sources

### Primary (HIGH confidence)
- ViT-RR reference codebase (`model_utils.py`, `deploy.py`, `requirements.txt`) — ctFP encoding, SimpViT architecture, bootstrap UQ pattern, Streamlit deployment
- ViT-RR paper: Angew. Chem. 2025 — [doi:10.1002/anie.202513086] — paradigm validation
- [Development and Experimental Validation of a Dispersity Model for In Silico RAFT Polymerization — Macromolecules 2022](https://pubs.acs.org/doi/10.1021/acs.macromol.2c01798) — ODE moment equations, stiffness
- [50th Anniversary Perspective: RAFT Polymerization — A User Guide, Macromolecules 2017](https://pubs.acs.org/doi/10.1021/acs.macromol.7b00767) — authoritative RAFT mechanism reference
- [Rate Retardation Trends in RAFT — Polymer Chemistry RSC 2024](https://pubs.rsc.org/en/content/getauthorversionpdf/d3py01332d) — retardation mechanism classification
- [Calibration after Bootstrap — npj Computational Materials 2022](https://www.nature.com/articles/s41524-022-00794-8) — bootstrap CI calibration
- [Evaluation Guidelines for ML in Chemical Sciences — Nature Reviews Chemistry 2022](https://www.nature.com/articles/s41570-022-00391-9) — validation standards
- [Probing OOD Generalization in ML for Materials — Communications Materials 2024](https://www.nature.com/articles/s43246-024-00731-w) — OOD failure modes
- [Ten Problems in Polymer Reactivity Prediction — Macromolecules 2025](https://pubs.acs.org/doi/10.1021/acs.macromol.4c02582) — domain perspective
- PyTorch 2.10.0, SciPy 1.15.0, NumPy 2.2.2, Streamlit 1.52.x — official release pages

### Secondary (MEDIUM confidence)
- [Kinetic Analysis of RAFT: Inhibition, Retardation, Optimum Living Polymerization — Monash](https://research.monash.edu/en/publications/kinetic-analysis-of-reversible-addition-fragmentation-chain-trans/) — parameter range reference
- [When Mayo Falls Short (Ctr >> 1) — Polymer Chemistry 2020](https://pubs.rsc.org/en/content/articlelanding/2020/py/d0py00348d) — Mayo equation baseline comparison context
- [Assessing Uncertainty in ML for Polymer Property Prediction — JCIM 2025](https://pubs.acs.org/doi/10.1021/acs.jcim.5c00550) — UQ best practices
- [AI-Based Forecasting with kMC-generated training data — ACS Polym. Au 2024](https://pubs.acs.org/doi/10.1021/acspolymersau.4c00047) — synthetic-to-real transfer precedent
- [Leveraging Molecular Descriptors and Explainable ML for PET-RAFT — PMC 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12894868/) — Route A precedent

### Tertiary (LOW confidence)
- [A Versatile Flow Reactor Platform for ML-Guided RAFT Synthesis — PubMed 2025](https://pubmed.ncbi.nlm.nih.gov/40198808/) — context only, not directly applicable

---
*Research completed: 2026-03-24*
*Ready for roadmap: yes*
