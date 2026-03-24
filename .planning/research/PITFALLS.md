# Domain Pitfalls: ViT-Ctr

**Domain:** Scientific ML for RAFT polymerization kinetics — ODE simulation, Vision Transformer training, literature validation, web deployment, academic paper writing
**Researched:** 2026-03-24
**Overall confidence:** HIGH for domain-specific chemical pitfalls; MEDIUM for ML architecture pitfalls (general ViT literature applied to this specific encoding)

---

## Critical Pitfalls

Mistakes that cause rewrites, invalid results, or paper rejection.

---

### Pitfall C1: RAFT Kinetic Model Mechanistic Mismatch

**What goes wrong:** The ODE system is built on the wrong or oversimplified mechanistic model for RAFT kinetics. Specifically, omitting the pre-equilibrium stage (consumption of the initial CTA before main-equilibrium propagation begins) generates conversion and dispersity profiles that look plausible for trithiocarbonates but are systematically wrong for dithioesters. The model then trains on incorrect Mn and D curves, so ctFP images contain corrupted structural information that cannot possibly encode the true Ctr.

**Why it happens:** The "unified" RAFT mechanism is often simplified to a single equilibrium. The pre-equilibrium is rate-determining for dithioesters and xanthates but negligible for trithiocarbonates. Papers covering "all RAFT agent types" frequently use one universal ODE system without per-class mechanism switching.

**Consequences:** The model learns a mapping from corrupted fingerprints to parameters — it may still achieve low training loss (because the noise is systematic and the model memorizes it) but will fail catastrophically on real experimental data where the true pre-equilibrium dynamics are present. Reviewers with RAFT expertise will immediately question the kinetic model derivation in SI.

**Prevention:**
- Implement two-stage ODE: pre-equilibrium (R-group radical addition) + main equilibrium, with the pre-equilibrium rate governed by k_add,0 / k_frag,0.
- Validate ODE output against known analytical limits before generating any training data: at low [CTA]/[M], Đ should approach that of conventional radical polymerization; at high Ctr (>10), Đ at 50% conversion should be <1.2.
- Per-RAFT-class model validation: generate a curve for one dithioester (e.g., CDB/St, Ctr≈22), one TTC (e.g., RAFT2/MMA), one xanthate (e.g., EtXan/VAc) and compare shape with published conversion-time + Đ-conversion plots before training.

**Detection (warning signs):**
- ODE-generated Đ is monotonically decreasing for all RAFT types, even dithioesters with LAMs — real data shows Đ dip then rise.
- Inhibition period length is zero for all dithioesters regardless of R-group activity.
- Simulated Mn vs. conversion deviates from theoretical Mn at early conversion for active CTAs.

**Phase:** Data Generation (Phase 1 in roadmap)

---

### Pitfall C2: Parameter Space Coverage Gap — Silent Extrapolation

**What goes wrong:** The training parameter grid (Ctr, kadd/kfrag, [CTA]/[M], conversion range) does not cover the experimental regime. When the web app is used with real data, the model is silently extrapolating rather than interpolating. Predictions appear confident (bootstrap CI looks narrow) but are wrong.

**Why it happens:** It is tempting to choose a "reasonable" parameter range from landmark papers. But literature covers only the most popular systems; exotic RAFT agents (high-Ctr dithiocarbamates, low-Ctr xanthates for LAMs) lie outside typical training grids. The ViT-RR analogy also uses a fixed 64×64 grid, so if Mn and D values in a ctFP fall outside the normalized range, they are silently clipped.

**Consequences:** Out-of-distribution failures are invisible without explicit OOD detection. The bootstrap CI underestimates uncertainty for OOD inputs (bootstrap resampling of a trained model does not detect distributional mismatch). Published results will be challenged if validation is attempted on edge cases.

**Prevention:**
- Define training parameter ranges BEFORE data generation based on a literature survey of all four RAFT agent classes: log10(Ctr) from -1 to 4 covers essentially all known RAFT systems.
- Include deliberate boundary cases in the training set (Ctr = 0.1, Ctr = 10000, very high [CTA]/[M] = 0.1, very low = 0.0005).
- Implement OOD detection at inference time: compute the fraction of non-zero pixels in the ctFP and flag inputs where fewer than 3 data points are present as "insufficient data."
- Log-scale sampling for Ctr parameter (not linear) — linear sampling will massively oversample high-Ctr space.

**Detection (warning signs):**
- Model predictions for Ctr > 100 all collapse to the same value (boundary clipping).
- CI width does not increase monotonically as input data becomes sparser.
- Literature xanthate/dithiocarbamate Ctr values (typically 0.5–5) fall far from predicted range.

**Phase:** Data Generation (Phase 1), Model Training (Phase 2), Validation (Phase 3)

---

### Pitfall C3: Conflation of Apparent and True Ctr in Literature Validation

**What goes wrong:** Literature Ctr values are compared directly to model output without accounting for the fact that experimental Ctr depends on measurement method. The Mayo equation, CLD slope, and dispersity-based methods yield systematically different values for the same system. Dithiobenzoate Ctr measured by Mayo plot is concentration-dependent (apparent Ctr decreases with increasing [CTA]) — the "true" value is the infinite-dilution extrapolation.

**Why it happens:** Review papers tabulate Ctr values without labeling the measurement method. A researcher building a validation set will mix Mayo, CLD, and model-fit values without realizing they are incommensurable.

**Consequences:** Validation RMSE appears large not because the model is wrong but because the ground truth is inconsistent. Reviewers who know the field will immediately notice this; the paper cannot be defended without careful provenance annotation.

**Prevention:**
- When collecting literature Ctr values for validation, record: (a) measurement method, (b) [CTA]/[M] used, (c) temperature, (d) solvent, (e) whether it is Ctr,0 (pre-equilibrium) or Ctr (main equilibrium).
- Prefer dispersity-based Ctr over Mayo for active RAFT agents (trithiocarbonates, dithioesters).
- Segment validation by measurement method and report separate RMSEs — this is defensible and actually strengthens the paper.
- The model is trained on simulated Ctr (as used in ODE), so define precisely what Ctr the ODE model represents and make this explicit in the paper.

**Detection (warning signs):**
- Literature values for nominally the same system (same CTA, monomer, temperature) differ by more than a factor of 2.
- Validation error is dramatically worse for dithiobenzoate systems than trithiocarbonate systems.
- Model consistently under- or over-predicts for high-Ctr dithioesters.

**Phase:** Validation (Phase 3), Paper Writing (Phase 5)

---

### Pitfall C4: Fingerprint Encoding Ambiguity — Sparse Data and Coordinate Normalization

**What goes wrong:** The ctFP image is a 64×64 sparse matrix where most pixels are zero. When experimental data has only 5–8 data points, fewer than 10 pixels out of 4096 contain signal. The ViT patch tokenization (4×4 patches of 16×16 pixels each) means that a single data point influences at most one patch token. If data points cluster near identical coordinates, they overwrite each other in the pixel grid (last-write-wins, as in ViT-RR's `transform()` function), silently discarding information.

**Why it happens:** The ViT-RR encoding maps (f1, total_conv) to a discrete pixel. If two experiments have the same f1 at different conversions but round to the same pixel, only the last one is written. This was acceptable for ViT-RR because copolymerization data is typically spread across f1 and conversion; for RAFT data, if users fix [CTA]/[M] = 0.01 and vary conversion, all data points land on the same column.

**Consequences:** Information loss during encoding is silent — the model receives a less-informative fingerprint than expected and produces wider CI (bootstrap detects higher variance) or confidently wrong predictions.

**Prevention:**
- For ctFP, the x-axis is [CTA]/[M] which varies experimentally only if the user performed a dilution series. Users who perform a single-[CTA]/[M] kinetic run will have all points in one column. Design the encoding to handle this explicitly: either use a 1D representation for single-[CTA]/[M] runs or require a minimum of 2 distinct [CTA]/[M] values.
- Add a pre-encoding deduplication check: if any two input points map to the same pixel, warn the user.
- In the web app, display the ctFP image before prediction so users can see what information was actually encoded.
- Consider sub-pixel interpolation: instead of integer-floor indexing, use a Gaussian splat (deposit signal in a 3×3 neighborhood weighted by distance) to reduce hard clipping.

**Detection (warning signs):**
- Predicted CI is unusually wide even with many data points.
- The displayed ctFP image shows only a single column of pixels activated.
- Model predictions do not improve as user adds more points with the same [CTA]/[M].

**Phase:** Data Generation (Phase 1), Model Training (Phase 2), Deployment (Phase 4)

---

### Pitfall C5: Inhibition Period and Retardation Factor Are Not Independently Identifiable from All Experimental Configurations

**What goes wrong:** The three-parameter output (Ctr, inhibition period, retardation factor) assumes that these parameters are independently estimable from Mn and D vs. conversion data. In practice, inhibition period and retardation are only resolvable if the user provides data at early conversion (<5%) with sufficient time resolution, and if the retardation is strong enough to be non-trivial in D. For trithiocarbonate/MAM systems, retardation is near zero and the model will attempt to predict zero from near-zero signal — bootstrap will give wide CI, but the model may still output a point estimate that looks precise.

**Why it happens:** The ODE generates data where all three parameters are known by construction. The ML model learns correlations in the training data. But in training data with Ctr > 100 and retardation ≈ 0, the ctFP contains almost no retardation signal — the model predicts based on Ctr and Đ structure alone, not true retardation information.

**Consequences:** For the dominant use case (trithiocarbonate + acrylate), the retardation factor prediction will be near-uninformative. Reviewers will challenge this and the paper will need to clearly distinguish which RAFT systems allow reliable three-parameter extraction.

**Prevention:**
- Generate training data stratified by RAFT agent class: include dithioester-dominated subsets with large retardation (retardation factor 0.1–0.5) and trithiocarbonate subsets with near-zero retardation.
- Validate retardation factor prediction separately on each RAFT class.
- In the paper and web app, explicitly state which agent classes are expected to give reliable retardation estimates vs. which are informative only for Ctr + inhibition period.
- Use the bootstrap CI width for each parameter separately to communicate reliability.

**Detection (warning signs):**
- Bootstrap CI for retardation factor is 10× wider than for Ctr on TTC systems.
- Retardation factor predictions cluster near 1.0 for all TTC inputs regardless of input data quality.
- Training loss for retardation head converges to a near-trivial value (predicting mean ≈ 1.0 for all TTC samples).

**Phase:** Data Generation (Phase 1), Model Training (Phase 2), Validation (Phase 3), Paper Writing (Phase 5)

---

## Moderate Pitfalls

---

### Pitfall M1: ODE Numerical Instability at Extreme Parameter Values

**What goes wrong:** The RAFT ODE system is stiff — especially during the pre-equilibrium where kadd can be 10^6-10^8 L/mol/s while kfrag is orders of magnitude smaller. Naive integration with `scipy.integrate.solve_ivp` using default tolerances and RK45 solver will fail or return incorrect results for extreme parameter combinations that are valid for training data boundary coverage.

**Prevention:**
- Use BDF (Backward Differentiation Formula) solver for all RAFT ODE integrations: `solve_ivp(..., method='BDF', rtol=1e-6, atol=1e-9)`.
- Add a post-integration validity check: final monomer conversion must be within [0, 1], Mn must be positive and finite, D must be in [1.0, 5.0].
- Log any parameter combinations that fail integration with failure reason — do not silently discard; they signal parameter boundary issues.
- Generate a small diagnostic run (1000 samples) before full dataset generation to identify failure rate.

**Detection (warning signs):** `solve_ivp` returns `success=False` for >1% of parameter samples; Đ values returned as >10 or <1.

**Phase:** Data Generation (Phase 1)

---

### Pitfall M2: Log-Scale Target Normalization Inconsistency

**What goes wrong:** Ctr spans 5+ orders of magnitude (0.1 to 10000). If the model is trained to predict raw Ctr (not log10(Ctr)), the loss function is dominated by high-Ctr samples and the model performs poorly on low-Ctr systems (xanthates, dithiocarbamates with Ctr < 5). The PROJECT.md specifies `log10(Ctr)` output, which is correct — but the inhibition period and retardation factor may also require log or normalized-log treatment.

**Prevention:**
- Confirm log10(Ctr) is the model output target (correct per PROJECT.md).
- For inhibition period: if measured in fractional conversion units [0, ~0.1], linear normalization is fine. If measured in seconds/minutes, log-scale is needed.
- For retardation factor: values near 1 are most common; a sigmoid-transformed output or direct [0,1] normalization is cleaner than raw values.
- Verify loss weighting does not accidentally re-weight back to linear scale during training.

**Detection (warning signs):** Training loss for low-Ctr samples (Ctr < 5) remains high throughout training while high-Ctr loss converges; predicted Ctr for xanthate systems is consistently biased upward.

**Phase:** Model Training (Phase 2)

---

### Pitfall M3: Bootstrap CI Calibration Overconfidence

**What goes wrong:** The ViT-RR approach uses bootstrap ensembles (200 models, each trained on a resampled subset of training data). The raw standard deviation across 200 predictions is presented as the uncertainty. As established in calibration literature (npj Computational Materials 2022), bootstrap raw std. dev. systematically underestimates true predictive uncertainty — the CI coverage is often 30–50% lower than the nominal confidence level.

**Prevention:**
- After training the 200 bootstrap models, perform post-hoc calibration: hold out 5–10% of training data as a calibration set, compute empirical coverage at 95% CI level, and apply a scalar correction factor to the raw std. dev.
- Report calibration metrics (empirical coverage vs. nominal coverage) in the Supporting Information.
- The F-distribution CI mentioned in PROJECT.md is a second approach — implement both and cross-validate on the literature set.

**Detection (warning signs):** For validation data points where the true Ctr is known from literature, the fraction of points falling within the claimed 95% CI is <70%.

**Phase:** Model Training (Phase 2), Validation (Phase 3)

---

### Pitfall M4: Data Leakage via Correlated Parameter Sampling

**What goes wrong:** If training data is generated by sampling parameter combinations on a grid (e.g., 100 values of Ctr × 50 values of kadd/kfrag × 20 values of [CTA]/[M]), then train/test splitting by random sample will put near-identical parameter combinations in both train and test sets. The model memorizes the grid and test accuracy looks excellent but the model has not generalized.

**Prevention:**
- Split training/validation/test sets by parameter block, not random sample. Reserve entire Ctr ranges (e.g., all samples with Ctr in [3.0, 5.0]) for validation.
- Report both in-distribution and out-of-distribution test set performance in the paper.
- Use multiple noise levels of Gaussian noise added to ctFP pixels during training to simulate experimental uncertainty in Mn and Đ measurements.

**Detection (warning signs):** Train R² > 0.99 and test R² > 0.98 immediately suggests data leakage — real generalization gaps are larger.

**Phase:** Data Generation (Phase 1), Model Training (Phase 2)

---

### Pitfall M5: Web App Input Validation Blind Spots

**What goes wrong:** The Streamlit app accepts Excel uploads or manual entry of Mn, Đ, [CTA]/[M], and conversion data. Users will upload data in non-standard formats (Mn in g/mol as integer, Đ as comma-decimal in European locale, conversion as percentage rather than fraction). Silently wrong numerical parsing produces ctFP images with pixel coordinates in the wrong region, and the model returns confidently wrong predictions.

**Prevention:**
- Implement strict input validation with explicit error messages: conversion must be in [0, 1] (detect if user entered percentages); Đ must be in [1.0, 5.0]; Mn must be positive.
- Display the parsed ctFP image in the app before running inference — the user can visually confirm data was read correctly.
- Add unit labels and example values to every input field.
- Test with the exact Excel format example files shown to users (provide downloadable templates).

**Detection (warning signs):** All ctFP pixels appear in the bottom-left corner of the image (conversion values entered as percentages map to x-axis index 50–100, which clips to 63); model returns physically impossible Ctr values (< 0.001 or > 100000).

**Phase:** Deployment (Phase 4)

---

### Pitfall M6: Retardation Mechanism Debate Makes Literature Comparison Fraught

**What goes wrong:** The ODE kinetic model must choose a retardation mechanism: Intermediate Radical Termination (IRT) or Slow Fragmentation Model (SFM). These give different relationships between K_RAFT (retardation constant) and observable rate retardation. Published K_RAFT values exist only for a few systems. If the ODE uses IRT but the literature value was derived assuming SFM, the retardation factor cannot be compared.

**Prevention:**
- Choose one mechanism (IRT is more experimentally supported for dithiobenzoates per Barner-Kowollik group) and state it explicitly.
- Do not claim to "predict retardation factor" in an absolute sense; instead frame it as predicting "the observable rate deceleration relative to conventional radical polymerization," which is mechanism-agnostic.
- Separate the retardation factor from K_RAFT in the paper framing — one is a dimensionless observable, the other is a mechanistic rate constant.

**Phase:** Data Generation (Phase 1), Paper Writing (Phase 5)

---

## Minor Pitfalls

---

### Pitfall m1: MX350 GPU Memory Ceiling for Training

**What goes wrong:** The MX350 has 2GB VRAM. A batch of 200 bootstrap models trained sequentially will succeed, but loading all 200 model checkpoints simultaneously for inference will exceed RAM if not managed. Training with large batch sizes or float32 gradients may be slower than expected.

**Prevention:** Train bootstrap models sequentially, saving each checkpoint to disk. At inference, load one model at a time and accumulate predictions, then discard. Use float32 throughout (the MX350 does not benefit from float16). If generation of 1M samples creates memory pressure, generate in chunks of 100K.

**Phase:** Data Generation (Phase 1), Model Training (Phase 2)

---

### Pitfall m2: Streamlit Community Cloud Free Tier Limitations

**What goes wrong:** Streamlit Community Cloud free tier has a memory limit (~1GB) and will sleep the app after inactivity. Loading 200 bootstrap model checkpoints on cold start will take 30–60 seconds; if each checkpoint is ~3MB, total is ~600MB — pushing toward the limit.

**Prevention:** Merge bootstrap predictions at training time: instead of saving 200 separate model files, save only the ensemble statistics (mean and std of weights) or quantize to int8. Alternatively, reduce bootstrap count to 50 for the deployed app (use 200 only for research validation). Precompute prediction intervals on a held-out grid and serve from a lookup table for common cases.

**Phase:** Deployment (Phase 4)

---

### Pitfall m3: Academic Paper Metric Selection

**What goes wrong:** Reporting only R² and RMSE on a log10(Ctr) scale will obscure errors on low-Ctr systems that are more chemically significant. log10(3) ≈ 0.48 vs. log10(30) ≈ 1.48 — an error of ±0.2 log units means a factor-of-1.6 error in Ctr, which is essentially meaningless at Ctr=100 but critical at Ctr=1.

**Prevention:**
- Report in the paper: RMSE in log10 units, mean absolute percentage error in linear Ctr space, and per-RAFT-class breakdown.
- Use parity plots (predicted vs. true in log scale) stratified by RAFT agent type.
- For the literature validation section, report mean fold-error (predicted / literature value) since this is scale-invariant.

**Phase:** Validation (Phase 3), Paper Writing (Phase 5)

---

### Pitfall m4: Paper Narrative Overselling the Simultaneous Three-Parameter Claim

**What goes wrong:** The paper's core claim is "one experiment, three parameters." Reviewers will ask: (a) why not just use three separate well-established methods?, (b) are all three parameters accurate or only Ctr? The answer — that the ML approach requires only a standard kinetic run with Mn and Đ tracking, which is routinely done in any RAFT lab — is strong, but the framing must be precise.

**Prevention:**
- Acknowledge explicitly in the paper that inhibition period and retardation factor are only reliably predicted for specific RAFT agent classes.
- Frame the three-parameter simultaneous output as "parameter extraction from existing kinetic data" not "three new experimental measurements" — the user is not doing three experiments, they are getting three insights from one experiment they would have done anyway.
- Provide a worked example showing the traditional workflow (three papers, three experimental setups) vs. the ViT-Ctr workflow (one kinetic run, one model query).

**Phase:** Paper Writing (Phase 5)

---

### Pitfall m5: SI Kinetic Derivation Errors

**What goes wrong:** The Supporting Information must contain the ODE derivation. Errors in stoichiometric factors, missing termination terms, or inconsistent notation between SI and code will be caught by reviewers who reproduce the ODE from the SI. The code and SI may diverge during development if the kinetic model is revised but SI is written only at the end.

**Prevention:**
- Write the ODE system in LaTeX in SI format before implementing it in code — the code is a translation of the SI, not the reverse.
- Include a "SI consistency test" in the codebase: a unit test that verifies the implemented ODE matches the SI equation at a specific parameter set with a hand-calculated reference value.

**Phase:** Data Generation (Phase 1), Paper Writing (Phase 5)

---

## Phase-Specific Warning Summary

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| RAFT ODE implementation | Wrong mechanism for pre-equilibrium (C1) | Two-stage ODE, per-class validation against published curves |
| ODE numerical integration | Stiffness failures at extreme parameters (M1) | BDF solver, post-integration validity checks |
| Parameter space design | Extrapolation in deployment (C2) | Log-scale sampling, boundary case inclusion, OOD detection |
| ctFP encoding | Silent pixel overwrites for single-[CTA]/[M] data (C4) | Deduplication check, display encoded image to user |
| Dataset splitting | Correlated parameter leakage inflates test R² (M4) | Block splits by Ctr range |
| Bootstrap training | Raw std. dev. underestimates true CI (M3) | Post-hoc calibration on held-out set |
| Output scale | Ctr range spans 5 orders — MSE dominated by high values (M2) | Predict log10(Ctr), verify inhibition/retardation normalization |
| Literature validation | Incommensurable Ctr measurement methods (C3) | Annotate method + conditions for every validation point |
| Retardation prediction | Mechanism debate makes comparison fraught (M6) | Frame as observable rate deceleration, not mechanistic K_RAFT |
| Three-parameter identifiability | Retardation unresolvable for TTC/MAM (C5) | Per-class validation, explicit claim scoping |
| Streamlit deployment | 200 models exceed memory ceiling (m2) | Reduce to 50 models in deployed app, or serialize ensemble statistics |
| Input validation | Wrong units silently corrupt ctFP (M5) | Strict validation + display of encoded image |
| Paper metrics | RMSE on log scale obscures low-Ctr errors (m3) | Report fold-error and per-class breakdown |
| Paper narrative | Overselling three-parameter claim (m4) | Frame as extraction from existing data, acknowledge per-class limits |
| SI vs. code consistency | Kinetic derivation diverges from implementation (m5) | Write ODE in LaTeX first, implement from LaTeX |

---

## Sources

- [RAFT Polymerization Kinetics: Combination of Apparently Conflicting Models — Macromolecules](https://pubs.acs.org/doi/abs/10.1021/ma800388c) — MEDIUM confidence
- [Development and Experimental Validation of a Dispersity Model for In Silico RAFT Polymerization — Macromolecules 2022](https://pubs.acs.org/doi/10.1021/acs.macromol.2c01798) — HIGH confidence (directly relevant ODE model validation)
- [Rate Retardation Trends in RAFT — Polymer Chemistry RSC 2024](https://pubs.rsc.org/en/content/getauthorversionpdf/d3py01332d) — HIGH confidence
- [50th Anniversary Perspective: RAFT Polymerization — A User Guide, Macromolecules 2017](https://pubs.acs.org/doi/10.1021/acs.macromol.7b00767) — HIGH confidence (authoritative reference)
- [A novel method for the measurement of degenerative chain transfer coefficients — Polymer Chemistry RSC](https://pubs.rsc.org/en/content/articlelanding/2016/py/c5py02004b/unauth) — MEDIUM confidence
- [Kinetic Modeling of RAFT via Dithiobenzoate Agents Considering the Missing Step Theory — ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1385894717310033) — MEDIUM confidence
- [Machine Learning in Polymer Research — Ge et al., Advanced Materials 2025](https://advanced.onlinelibrary.wiley.com/doi/10.1002/adma.202413695) — HIGH confidence
- [Best Practices in Machine Learning for Chemistry — Nature Chemistry 2021](https://www.nature.com/articles/s41557-021-00716-z) — HIGH confidence
- [Evaluation Guidelines for ML Tools in the Chemical Sciences — Nature Reviews Chemistry 2022](https://www.nature.com/articles/s41570-022-00391-9) — HIGH confidence
- [Calibration after Bootstrap for Accurate Uncertainty Quantification — npj Computational Materials 2022](https://www.nature.com/articles/s41524-022-00794-8) — HIGH confidence
- [Probing Out-of-Distribution Generalization in ML for Materials — Communications Materials 2024](https://www.nature.com/articles/s43246-024-00731-w) — HIGH confidence (2024, directly addresses OOD failure modes)
- [Vision Transformers in Domain Adaptation and Generalization — arXiv 2024](https://arxiv.org/html/2404.04452v2) — MEDIUM confidence
- [Machine Learning with Enormous Synthetic Datasets: Predicting Glass Transition Temperature — ACS Omega](https://pubs.acs.org/doi/10.1021/acsomega.2c04649) — MEDIUM confidence (synthetic-to-real transfer learning precedent)
- [Ten Problems in Polymer Reactivity Prediction — Macromolecules 2025](https://pubs.acs.org/doi/10.1021/acs.macromol.4c02582) — HIGH confidence (directly relevant domain perspective)
- ViT-RR source code (`model_utils.py`) — HIGH confidence (direct inspection of pixel-write logic)
