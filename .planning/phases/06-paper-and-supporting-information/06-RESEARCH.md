# Phase 6: Paper and Supporting Information — Research

**Researched:** 2026-04-06
**Phase Goal:** A complete draft manuscript and Supporting Information exist, with all figures generated from finalized model results, ODE derivations consistent with code, and Route A framed as future work

## 1. Available Assets Inventory

### 1.1 Figures (Ready to Use)

| Figure | File | Paper Location |
|--------|------|----------------|
| Parity: log10(Ctr) | `figures/parity_log10_Ctr.png` | Fig 3 (composite) |
| Parity: inhibition period | `figures/parity_inhibition_period.png` | Fig 3 (composite) |
| Parity: retardation factor | `figures/parity_retardation_factor.png` | Fig 3 (composite) |
| ML vs Mayo validation | `figures/validation/parity_ml_vs_mayo.png` | Fig 4 |
| Training loss curves | `checkpoints/loss_curves.png` | Fig 5 |
| Residuals (3 plots) | `figures/residuals_*.png` | SI |
| Per-class parity (12 plots) | `figures/parity_by_class/*.png` | SI + Fig 6 (representative) |

### 1.2 Figures That Need Creation

| Figure | Description | Source Data |
|--------|-------------|-------------|
| Figure 1: Concept/workflow | Experimental data → ctFP → SimpViT → 3 params + CI | Schematic (manual design) |
| Figure 2: ctFP example | Dual-channel heatmap showing encoding | Generate from `src/ctfp_encoder.py` with representative sample |
| TOC graphic | ~3.25×1.75 in summary graphic | Derived from Fig 1 |

### 1.3 Data Files

| File | Content | Paper Use |
|------|---------|-----------|
| `figures/validation/validation_results.csv` | 14-point literature validation with ML and Mayo predictions | Results Table / Fig 4 |
| `figures/validation/validation_summary.json` | ML: R²=0.991, median fold-error=1.17; Mayo: R²=0.825, median fold-error=1.01 | Results text |
| `checkpoints/calibration.json` | Bootstrap calibration factors [100.0, 53.7, 3.5] | Methods/SI |
| `checkpoints/bootstrap_summary.json` | 200 heads, 5 epochs/head, RTX 4090, 9h17m | Methods/SI |
| `checkpoints/training_log.json` | 142 epochs, final val_loss=0.389 | Results/SI |
| `data/literature/literature_ctr.csv` | 14 published Ctr values with references | Results Table |

### 1.4 Source Code (Methods Description Basis)

| Module | Key Details for Paper |
|--------|---------------------|
| `src/raft_ode.py` | 14-var single-eq ODE + 16-var pre-eq ODE; moment method with mu/nu/lam populations |
| `src/ctfp_encoder.py` | 64×64×2 tensor; Ch0=Mn/Mn_theory, Ch1=Đ (clipped at 4.0); x=[CTA]/[M], y=conversion |
| `src/model.py` | SimpViT: patch_size=16, hidden=64, 2 layers, 4 heads, ~877K params, num_outputs=3 |
| `src/bootstrap.py` | 200 heads, freeze backbone, F-dist JCI with p=3, post-hoc calibration |
| `src/dataset_generator.py` | 7-dim LHS + 4 RAFT types; log10_Ctr∈[-2,4]; ~1M samples in chunked HDF5 |
| `src/literature_validation.py` | Mayo ODE fitting baseline; 50-ensemble ML prediction; fold-error metric |
| `src/evaluate.py` | R², RMSE, MAE per output; per-class metrics; outlier stats |

## 2. Key Quantitative Results

### 2.1 Model Performance (Test Set)
- Training: 142 epochs, Adam lr=3e-4 with decay, batch_size=256
- Dataset: ~973K samples (778K train / 97K val / 97K test), 4 RAFT types
- Final val_loss: 0.389

### 2.2 Literature Validation (14 Points)
- **ML model:** R²=0.991, RMSE(log10)=0.126, median fold-error=1.17, 93% within 2×, 100% within 10×
- **Mayo baseline:** R²=0.825, RMSE(log10)=0.558, median fold-error=1.01, 86% within 2×, 93% within 10×
- ML outperforms Mayo on RMSE and R² despite Mayo having lower median fold-error (Mayo benefits from fixed "ideal" kinetic params)

### 2.3 Bootstrap UQ
- 200 bootstrap heads, 5 epochs each, backbone frozen
- F-distribution JCI: p=3, dfd=197, f_val≈2.65
- Calibration factors: [100.0, 53.7, 3.5] for [log10_Ctr, inhibition, retardation]
- Post-calibration coverage: log10_Ctr=69.2%, inhibition=95.0%, retardation=95.0%
- **Note:** log10_Ctr coverage at 69.2% (not 95%) even after calibration — this is a known limitation that must be discussed honestly

### 2.4 Literature Dataset Composition
- 14 published Ctr values from 8 references
- 4 RAFT types: dithioester (4), trithiocarbonate (4), xanthate (3), dithiocarbamate (3)
- 3 measurement methods: Mayo (7), Dispersity (4), CLD (3)
- All at 60°C, bulk polymerization

## 3. Manuscript Structure Research

### 3.1 Macromolecules (ACS) Requirements
- **Template:** achemso LaTeX class (`\documentclass[journal=mamobx]{achemso}`)
- **Length:** 10-15 pages typical for full articles
- **Figures:** Max ~6 in main text; additional in SI
- **Abstract:** ~250 words, structured
- **References:** ACS style (superscript numbers)
- **SI:** Separate document, referenced from main text
- **TOC graphic:** Required, ~3.25×1.75 in

### 3.2 Proposed Section Outline

**Main Manuscript:**
1. **Abstract** (~250 words) — Three-parameter simultaneous extraction claim, key metrics
2. **Introduction** — RAFT background, Ctr measurement challenge (3 separate experiments), ViT-RR precedent, our contribution
3. **Methods**
   - 3.1 RAFT Kinetic ODE Model (moment method, 4 agent types, pre-eq for dithioester)
   - 3.2 Chain Transfer Fingerprint (ctFP) Encoding
   - 3.3 Dataset Generation (LHS sampling, parameter ranges, ~1M samples)
   - 3.4 SimpViT Architecture
   - 3.5 Training Protocol (weighted MSE, early stopping, stratified split)
   - 3.6 Bootstrap Uncertainty Quantification
   - 3.7 Literature Validation Protocol
4. **Results and Discussion**
   - 4.1 Model Performance on Synthetic Test Set
   - 4.2 Literature Validation: ML vs Mayo
   - 4.3 Per-RAFT-Agent-Class Analysis
   - 4.4 Uncertainty Quantification Assessment
   - 4.5 Limitations (retardation scope, parameter boundaries, simulation vs reality)
   - 4.6 Web Application for Community Use
5. **Conclusions and Future Directions** — Route A (SMILES → Ctr) as future work
6. **Associated Content** — SI description, data/code availability
7. **References**

**Supporting Information:**
1. Complete ODE Derivation (moment equations matching raft_ode.py)
2. Dataset Construction Details (parameter ranges, LHS, noise, failure rates)
3. Training Hyperparameters Table
4. Bootstrap UQ Detailed Protocol
5. Complete Per-Class Evaluation (12 parity plots + 3 residual plots)
6. Literature Validation Full Table

## 4. Technical Considerations

### 4.1 ODE-to-LaTeX Mapping
The SI ODE derivation must match `src/raft_ode.py` exactly. Key mapping:
- `raft_ode_single_eq`: 14 state variables → 14 coupled ODEs
- `raft_ode_preequilibrium`: 16 state variables → 16 coupled ODEs (adds CTA_0, Int_pre)
- Moment exchange term: `d(mu_k)/dt_exchange = kadd * (mu0 * nu_k - mu_k * nu0)`
- Termination model: disproportionation-dominant (2 dead chains per event)
- Mn calculation: `mu1 + nu1 + lam1` / `mu0 + nu0 + lam0` × M_monomer
- Đ calculation: from second moments

### 4.2 Figure Composition Strategy
- **Figure 3 (3-in-1 parity):** Combine 3 existing parity PNGs into a single composite figure with (a), (b), (c) panels
- **Figure 6 (per-class):** Select dithioester as representative (most interesting — has retardation/inhibition effects)
- **Figure 1 & 2:** Need Python scripts to generate (matplotlib/tikz)
- **TOC graphic:** Simplified version of Figure 1

### 4.3 Output Format Strategy
Per CONTEXT.md D-02/D-03, three output files needed:
1. **LaTeX (.tex):** achemso template, publication-ready
2. **Word English (.docx):** For collaborator review
3. **Word Chinese (.docx):** For internal use

Practical approach: Write LaTeX first (canonical), then convert/adapt to Word versions. Use pandoc for initial conversion, then manual cleanup.

### 4.4 Honest Framing Requirements
Per CONTEXT.md D-15, must discuss:
1. Simulation vs reality gap (ODE doesn't capture side reactions, non-ideal mixing)
2. Retardation ≈ 1.0 for TTC/xanthate/dithiocarbamate — prediction value limited
3. Parameter range boundaries (log10_Ctr∈[-2,4], [CTA]/[M]∈[0.001,0.1])
4. Uncovered conditions (fixed T=60°C, no solvent effects, only 4 RAFT types)
5. Bootstrap calibration: log10_Ctr coverage only 69.2% post-calibration (cal_factor=100 is extreme)

### 4.5 Mayo Baseline Framing
Per CONTEXT.md specifics: Mayo comparison positioned as validation evidence, not the paper's selling point. Note that Mayo baseline uses fixed "mean" kinetic parameters — this is actually favorable to Mayo (real researchers must estimate these), so ML outperforming is more convincing.

## 5. Dependencies and Risks

### 5.1 Available Tooling
- **No LaTeX compiler** (pdflatex/xelatex/lualatex not installed)
- **pandoc 3.8** — available for format conversion
- **python-docx 1.2.0** — available for programmatic Word generation
- **jinja2 3.1.6** — available for template rendering
- **matplotlib 3.10.6** — available for figure generation
- Best model at epoch 126, val_loss=0.388

### 5.2 Tooling Strategy
Since no LaTeX compiler is available, the practical approach is:
1. Write manuscript content as structured Markdown (canonical source)
2. Generate Word (.docx) versions directly via python-docx or pandoc
3. LaTeX .tex file can be written but compilation deferred to a machine with TeX Live
4. Alternatively: use Overleaf for LaTeX compilation (upload .tex + figures)

### 5.3 Dependencies
- All Phase 1-5 artifacts are complete and available
- All figures exist except concept diagram (Fig 1), ctFP example (Fig 2), and TOC graphic
- Validation data is finalized (14 points)

### 5.4 Risks
- **No local LaTeX:** Cannot compile .tex locally — must use Overleaf or install TeX Live
- **Figure quality:** Existing PNGs may need DPI adjustment for print (300+ DPI required)
- **Word conversion:** pandoc may not perfectly preserve complex formatting
- **ODE derivation length:** Full 14-variable moment equations are extensive — SI could be very long
- **Iterative review:** D-10/D-11 require chapter-by-chapter user review — plans should include checkpoints

## 6. Validation Architecture

### 6.1 Verification Strategy
- **ODE consistency:** Every equation in SI LaTeX must have a corresponding line in `raft_ode.py` — verify by side-by-side comparison
- **Figure accuracy:** All figures in paper must match the actual PNG files in `figures/`
- **Metrics accuracy:** All numbers cited in text must match `validation_summary.json` and `training_log.json`
- **Reference completeness:** All 14 literature entries must appear in bibliography
- **Three-parameter claim scope:** Must be qualified to dithioester systems where retardation is meaningful

### 6.2 Acceptance Criteria
- LaTeX compiles without errors
- All 6 main figures present and correctly labeled
- SI contains complete ODE derivation matching code
- Route A appears only in future directions
- Word versions readable and complete

## RESEARCH COMPLETE

Research covers: asset inventory, quantitative results, manuscript structure, technical mapping, output strategy, and validation approach. Ready for planning.
