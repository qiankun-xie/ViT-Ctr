# Feature Landscape

**Domain:** Scientific ML system for RAFT kinetic parameter prediction (Ctr, inhibition period, retardation factor)
**Researched:** 2026-03-24
**Confidence:** MEDIUM-HIGH (ML pipeline patterns HIGH; domain-specific RAFT ML features MEDIUM from literature)

---

## Table Stakes

Features users (polymer chemists publishing papers) will expect. Missing = product feels unfinished or untrustworthy.

### ML Pipeline (Data Generation + Training + Evaluation)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| ODE-based RAFT kinetic simulator | Training data cannot come from real experiments at million-scale; synthetic generation via ODE is standard practice for this paradigm | High | Must handle all RAFT agent classes (dithioester, trithiocarbonate, xanthate, dithiocarbamate); stiff ODE solver required |
| Parameter space sweep for data generation | ML models trained on narrow parameter ranges fail to generalize; reviewers will probe edge cases | Medium | Cover Ctr ~0.01–1000, kadd/kfrag ratio, [CTA]/[M] ratio, initiator concentration, temperature |
| ctFP encoding (64x64 dual-channel image) | Core representation from ViT-RR paradigm; Mn channel + D channel encodes sufficient information to extract three parameters | Medium | x-axis = [CTA]/[M], y-axis = conversion; must normalize consistently |
| SimpViT architecture (3-output head) | Follows validated ViT-RR paradigm; SimpViT ~3.4MB is deployable on CPU | Low | Change output dim from 2 to 3: log10(Ctr), inhibition period, retardation factor |
| Train/val/test split with held-out test set | Reviewers will check for data leakage; a separate test set is non-negotiable for a methods paper | Low | Random split is insufficient — stratify by Ctr range to ensure coverage |
| Loss function with log-scale for Ctr | Ctr spans orders of magnitude; MSE on linear scale will neglect small values | Low | Train on log10(Ctr); transform back for display |
| Training convergence curves (loss vs. epoch) | Expected in any ML methods paper; reviewers use these to check overfitting | Low | Log to file; plot in paper |
| Test set R² / RMSE / MAE per output | Standard evaluation metrics for regression; missing = paper rejected | Low | Report separately for each of the three outputs |
| Parity plots (predicted vs. true) on test set | Standard figure in every ML methods paper | Low | One per output parameter; include on synthetic data |
| Literature validation against published Ctr values | Distinguishes a publishable result from a demo; proves real-world applicability | High | At minimum 10–20 literature data points covering multiple RAFT agent types |

### Web Application

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Manual data entry (conversion-time + Mn/D vs. conversion) | Primary use case: researcher has a few experimental data points and wants Ctr | Medium | Input: multiple rows of ([CTA]/[M], conversion, Mn, D); validate bounds |
| Excel/CSV file upload | Researchers generate data in spreadsheets; manual entry for >10 points is painful | Low | Provide downloadable template; column validation required |
| Excel template download | Without a template, users guess column names and fail silently | Low | Match ViT-RR template pattern |
| Three-parameter output display (Ctr, inhibition period, retardation factor) | Core value proposition; all three must be shown | Low | Display with uncertainty; explain what each means |
| Uncertainty / confidence interval display | Scientific ML tools without UQ are not trusted by chemists; Bootstrap CI is the established approach in this paradigm | Medium | 200 bootstrap iterations + F-distribution JCI (matching ViT-RR approach) |
| Model loading with caching | Without caching, each prediction reloads the model → unacceptable latency | Low | st.cache_resource pattern (already in ViT-RR) |
| Input validation with clear error messages | Users will enter negative values, out-of-range conversions, empty rows; silent failure destroys trust | Medium | Check: conversion in (0,1), Mn > 0, D ≥ 1, at least 3 data points |
| Result display with parameter explanations | Non-expert users need to understand what Ctr = 3.2 means | Low | Tooltip or sidebar explanation of each parameter |

---

## Differentiators

Features that distinguish this tool from manual Mayo-equation fitting and from generic ML polymer tools.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Simultaneous 3-parameter extraction | Core paper narrative: one input, three outputs replaces three independent experimental workflows (Mayo equation, conversion-time inhibition fitting, rate comparison) | Medium | Requires careful multi-task loss weighting; verify that the model does not ignore any output |
| Coverage of all major RAFT agent classes | Existing tools (where they exist) are agent-specific; a universal model has broader impact | High | Must validate per-class; paper should include per-RAFT-agent-type parity plots |
| Bootstrap uncertainty via F-distribution JCI | More statistically rigorous than simple percentile CI; already validated in ViT-RR (Angew. Chem.); reviewers familiar with the paradigm | Medium | 200 iterations, F.ppf(0.95, dfn=p, dfd=n-p); matches ViT-RR implementation exactly |
| ctFP fingerprint visualization | Shows users what data structure the model sees; increases trust and interpretability for reviewers | Medium | Plot the 64x64 Mn and D channels as heatmaps in the SI |
| Route A (structure → Ctr, exploratory) | Adds a forward-looking "future direction" section with structural interpretability; differentiates from pure kinetics tools | High | SMILES → molecular descriptors or GNN → Ctr; data limited; position as proof-of-concept / future work, not primary claim |
| Comparison with Mayo equation baseline | Validates the method against the gold standard; "better than Mayo" is the key benchmark claim | Medium | Implement Mayo fitting on the same literature validation set; compare accuracy |
| RAFT agent type selector in web app | Allows the model to condition on agent class (if separate models or branches trained per class) | Low-Medium | Low if single universal model; Medium if separate models |
| User registration tracking (Google Sheets) | Provides usage analytics for paper narrative ("tool used by N researchers worldwide"); already implemented in ViT-RR | Low | Reuse ViT-RR approach; optional but high ROI for future citations |

---

## Anti-Features

Features to explicitly NOT build, with rationale.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Real-time online learning / model updates | Requires infrastructure (model versioning, retraining pipelines, data collection consent); out of scope per PROJECT.md; breaks reproducibility for cited predictions | Fix model weights at paper submission; version the model file |
| ATRP / NMP / other CRP methods | Scope creep; different kinetics require different ODE models and encodings; dilutes the RAFT-focused narrative | Explicitly note in paper that extension to other CRP methods is future work |
| Full kadd/kfrag deconvolution | These parameters are not independently resolvable from standard Mn/D vs. conversion data without additional information; claiming to predict them would be scientifically indefensible | Predict only parameters that are identifiable from the ctFP: Ctr, inhibition period, retardation factor |
| Mobile-responsive design | Chemists use lab computers; mobile optimization adds CSS complexity with zero scientific value | Keep Streamlit default layout; use wide mode |
| User accounts with password authentication | Research tools don't need auth beyond lightweight usage logging; full auth (OAuth, hashing, session management) is weeks of work | Keep ViT-RR's lightweight email registration to Google Sheets |
| Generative / inverse design (given target Ctr, suggest RAFT agent) | Inverse design requires a different model class (generative or optimization-based); out of scope for a Ctr prediction paper | Note as future direction in paper conclusion |
| Real experimental data storage / database | Privacy, institutional approval, GDPR complexity; the tool is stateless by design | Process client-side; do not store uploaded experimental data |
| Multi-language interface (Chinese/English toggle) | Target audience for the web app is international researchers; English-only is correct for an Angew. Chem. companion tool | English UI only; Chinese documentation for internal project docs |
| Batch prediction API (REST endpoint) | Only needed for high-throughput screening pipelines; current use case is single-experiment analysis | Streamlit's upload-and-predict workflow is sufficient |
| Model explainability / attention visualization in the web app | Attention maps from SimpViT are noisy with sparse ctFP inputs; including them would confuse rather than inform | Include ctFP heatmap visualization only (shows input structure, not model internals) |

---

## Feature Dependencies

```
ODE simulator → synthetic dataset → SimpViT training → trained model weights
trained model weights → web app deployment
Bootstrap UQ → needs trained model + multiple forward passes
Literature validation → needs trained model + collected Ctr literature data
Mayo equation baseline → needs same literature validation dataset
ctFP encoding → feeds both training pipeline AND web app input preparation
Excel template → depends on finalizing input column format
User registration (Google Sheets) → independent of ML pipeline; deploy at same time as web app
Route A (structure → Ctr) → independent branch; does NOT block Route B delivery
```

Key blocking chain: ODE simulator → dataset → training → evaluation → app. Route A is non-blocking.

---

## MVP Recommendation

### Must Ship (paper-critical)

1. ODE RAFT kinetic simulator with parameter sweep
2. ctFP encoding (dual-channel 64x64)
3. SimpViT training with 3-output head
4. Bootstrap UQ (200 iterations, F-distribution JCI)
5. Test set evaluation: R², RMSE, MAE per output + parity plots
6. Literature validation (10+ published Ctr values, multiple RAFT agent classes)
7. Streamlit web app: manual input + Excel upload + 3-parameter output + CI display

### Should Ship (strengthens paper)

8. Comparison against Mayo equation baseline on same literature set
9. ctFP heatmap visualization (for SI)
10. Per-RAFT-agent-type validation breakdown
11. User registration + usage tracking
12. Excel template download

### Defer

- Route A (structure → Ctr): exploratory; position as future work in conclusion section; build only if time permits after Route B is complete and paper draft is in progress
- Attention map visualization: not scientifically justified for sparse ctFP inputs; omit

---

## Complexity Reference

Low = < 1 day implementation
Medium = 1–3 days implementation
High = 3+ days, iteration likely required

---

## Sources

- ViT-RR reference implementation (C:/CodingCraft/DL/ViT-RR/deploy.py, model_utils.py) — HIGH confidence
- [Development and Experimental Validation of a Dispersity Model for In Silico RAFT Polymerization](https://pubs.acs.org/doi/10.1021/acs.macromol.2c01798) — MEDIUM confidence
- [Kinetic Analysis of RAFT Polymerizations: Conditions for Inhibition, Retardation, and Optimum Living Polymerization](https://research.monash.edu/en/publications/kinetic-analysis-of-reversible-addition-fragmentation-chain-trans/) — HIGH confidence
- [When Mayo Falls Short (Ctr >> 1)](https://pubs.rsc.org/en/content/articlelanding/2020/py/d0py00348d) — HIGH confidence
- [Assessing Uncertainty in ML for Polymer Property Prediction (JCIM 2025)](https://pubs.acs.org/doi/10.1021/acs.jcim.5c00550) — MEDIUM confidence
- [PolUQBench: UQ for Polymer Property Prediction (NeurIPS 2025)](https://openreview.net/forum?id=PAp7ZyD2Yd) — MEDIUM confidence
- [AI-Based Forecasting of Polymer Properties with kMC-generated training data](https://pubs.acs.org/doi/10.1021/acspolymersau.4c00047) — MEDIUM confidence
- [Evaluation Guidelines for ML Tools in the Chemical Sciences](https://www.nature.com/articles/s41570-022-00391-9) — HIGH confidence
- [Leveraging Molecular Descriptors and Explainable ML for PET-RAFT](https://pmc.ncbi.nlm.nih.gov/articles/PMC12894868/) — MEDIUM confidence
- [A Versatile Flow Reactor Platform for ML-Guided RAFT Synthesis (2025)](https://pubmed.ncbi.nlm.nih.gov/40198808/) — LOW confidence (not directly applicable)
