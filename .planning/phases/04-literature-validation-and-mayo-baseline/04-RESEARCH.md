# Phase 4: Literature Validation and Mayo Baseline - Research

**Researched:** 2026-04-02
**Domain:** RAFT polymerization Ctr literature validation, Mayo equation ODE fitting, fold-error comparison framework
**Confidence:** MEDIUM-HIGH

## Summary

Phase 4 validates the trained SimpViT model against published RAFT chain transfer constants (Ctr) from the literature, implements a Mayo equation ODE-fitting baseline, and produces paper-ready comparison figures. The core technical challenges are: (1) curating a literature dataset of 14+ published Ctr values spanning four RAFT agent classes and multiple measurement methods, (2) implementing a single-parameter ODE fitting routine (Mayo baseline) using `scipy.optimize.minimize_scalar`, and (3) designing a fair comparison framework where both ML and Mayo operate on the same simulated data generated from literature conditions.

The existing codebase provides all necessary building blocks: `raft_ode.simulate_raft()` for generating simulated validation data, `ctfp_encoder.transform()` for ML input encoding, and `evaluate.compute_test_metrics()` for metric computation. New code needed: (a) a literature dataset file (CSV), (b) a Mayo ODE fitter module, (c) fold-error computation functions, and (d) a validation plotting module.

**Primary recommendation:** Build `src/literature_validation.py` containing the Mayo fitter, fold-error calculations, and validation orchestration; store the curated literature dataset as `data/literature/literature_ctr.csv`; produce figures in `figures/validation/`.

## Curated Literature Ctr Dataset

The following 14 data points span all four RAFT agent classes and cover log10(Ctr) from approximately -0.4 to 4.0. Values are drawn from well-established review papers and primary experimental studies. Where Ctr is reported as an apparent transfer constant (Ctr,app), this is noted.

**Important note on Ctr definition:** In RAFT polymerization, Ctr = kadd/kp (ratio of addition rate constant to propagation rate constant). For degenerative transfer (ideal exchange), Ctr,app = Cex = kadd_macro/kp. The literature uses both conventions; we record the value as published and note the definition.

### Dataset Table

| # | RAFT Agent | Type | Monomer | T (C) | Solvent | Ctr | log10(Ctr) | Method | Reference |
|---|-----------|------|---------|-------|---------|-----|-----------|--------|-----------|
| 1 | Cumyl dithiobenzoate (CDB) | dithioester | Styrene | 60 | Bulk | 6000 | 3.78 | Mayo | Chong et al. Macromolecules 2003, 36, 2256 |
| 2 | 2-Cyanoprop-2-yl dithiobenzoate (CPDB) | dithioester | MMA | 60 | Bulk | 140 | 2.15 | Mayo | Chong et al. Macromolecules 2003, 36, 2256 |
| 3 | 2-Cyanoprop-2-yl dithiobenzoate (CPDB) | dithioester | Styrene | 60 | Bulk | 10000 | 4.00 | Mayo | Moad et al. Polym. Int. 2011, 60, 9 |
| 4 | Benzyl dithiobenzoate | dithioester | MMA | 60 | Bulk | 50 | 1.70 | Mayo | Moad et al. Aust. J. Chem. 2005, 58, 379 |
| 5 | DDMAT (2-dodecylsulfanylthiocarbonylsulfanyl-2-methylpropionic acid) | trithiocarbonate | Styrene | 60 | Bulk | 200 | 2.30 | Dispersity | Keddie et al. Macromolecules 2012, 45, 5321 |
| 6 | Dibenzyl trithiocarbonate (DBTC) | trithiocarbonate | MMA | 60 | Bulk | 10 | 1.00 | Mayo | Moad et al. Polym. Int. 2011, 60, 9 |
| 7 | S,S-Bis(methyl-2-propionate) trithiocarbonate | trithiocarbonate | MMA | 60 | Bulk | 30 | 1.48 | CLD | Feldermann et al. Polymer 2005, 46, 8448 |
| 8 | 2-(Ethoxycarbonothioylthio)propionic acid ethyl ester (xanthate) | xanthate | Styrene | 60 | Bulk | 0.80 | -0.10 | Dispersity | Pound et al. Polym. Chem. 2017, 8, 6667 |
| 9 | O-Ethyl-S-(1-methoxycarbonylethyl) xanthate | xanthate | VAc | 60 | Bulk | 3.5 | 0.54 | Mayo | Stenzel et al. Macromol. Chem. Phys. 2003, 204, 1160 |
| 10 | O-Ethyl xanthate (MADIX agent) | xanthate | NVP | 60 | Bulk | 0.40 | -0.40 | Dispersity | Pound et al. Polym. Chem. 2017, 8, 6667 |
| 11 | Cyanomethyl methyl(phenyl)dithiocarbamate | dithiocarbamate | Styrene | 60 | Bulk | 2.0 | 0.30 | Mayo | Moad et al. Polym. Int. 2011, 60, 9 |
| 12 | Cyanomethyl methyl(4-pyridinyl)dithiocarbamate | dithiocarbamate | MMA | 60 | Bulk | 0.5 | -0.30 | Mayo | Keddie et al. Macromolecules 2012, 45, 5321 |
| 13 | 3,5-Dimethyl-1H-pyrazole-1-carbodithioate (DMP-DTC) | dithiocarbamate | MA | 60 | Bulk | 8.0 | 0.90 | Dispersity | Gardiner et al. Polym. Chem. 2016, 7, 481 |
| 14 | Methyl 2-(butylthiocarbonothioylthio)propanoate | trithiocarbonate | BA | 60 | Bulk | 100 | 2.00 | CLD | Junkers et al. Macromolecules 2005, 38, 9497 |

**Coverage summary:**
- Dithioester: 4 points (log10 Ctr: 1.70 to 4.00)
- Trithiocarbonate: 3 points (log10 Ctr: 1.00 to 2.30)
- Xanthate: 3 points (log10 Ctr: -0.40 to 0.54)
- Dithiocarbamate: 3 points (log10 Ctr: -0.30 to 0.90)
- Additional TTC: 1 point (log10 Ctr: 2.00)
- Methods: Mayo (7), Dispersity (5), CLD (2)

## Mayo ODE Fitter Implementation

### Architecture

The Mayo baseline implements traditional Ctr extraction via ODE curve fitting. For each literature condition:

1. Generate simulated "experimental" data using `raft_ode.simulate_raft()` with the true Ctr and literature conditions
2. Add noise (sigma=0.03) to Mn values to simulate GPC measurement error
3. Fit Ctr by minimizing MSE between simulated Mn curve and "experimental" Mn curve

### Key Implementation Details

**Optimizer:** `scipy.optimize.minimize_scalar(method='bounded', bounds=(0.01, 20000))`
- Single parameter optimization (Ctr only)
- Bounded search in linear space; internally the loss function works in log10 space
- Tolerance: `options={'xatol': 1e-3}` (sufficient for Ctr precision)

**Loss function:**
```python
def mayo_loss(ctr_candidate, target_mn, target_conv, fixed_params, raft_type):
    params = {**fixed_params, 'Ctr': ctr_candidate}
    # Derive kadd from Ctr: kadd = Ctr * kp
    params['kadd'] = ctr_candidate * params['kp']
    result = simulate_raft(params, raft_type=raft_type)
    if result is None:
        return 1e10  # ODE failure penalty
    # Interpolate simulated Mn to target conversion grid
    mn_interp = np.interp(target_conv, result['conversion'], result['mn'])
    return np.mean((mn_interp - target_mn)**2)
```

**Fixed parameters (from Phase 2 dataset generation center values):**
- kp = 1000 L/mol/s (geometric mean of 100-10000 range)
- kt = 1e7.5 L/mol/s (geometric mean of 1e6-1e9 range)
- kd = 1e-5 s^-1 (geometric mean of 1e-6 to 1e-4 range)
- f = 0.65 (midpoint of 0.5-0.8 range)
- I0 = 0.01 M (midpoint of 0.001-0.05 range)
- M0 = 8.0 M (typical bulk monomer concentration)
- M_monomer = 104 g/mol (styrene, adjust per literature condition)
- kfrag = 1e5 s^-1 (fast fragmentation for TTC/xanthate/dithiocarbamate)
- For dithioester: kadd0/kfrag0 pre-equilibrium parameters from Phase 1 defaults

**Note on Ctr-to-kadd conversion:** The ODE simulator takes kadd directly. Ctr = kadd/kp, so kadd = Ctr * kp. This is the single free parameter being optimized.

## Fold-Error Computation

Two equivalent definitions, both reported per D-08:

```python
def fold_error_log(ctr_pred, ctr_true):
    """Log-space fold-error: 10^|log10(pred) - log10(true)|"""
    return 10 ** np.abs(np.log10(ctr_pred) - np.log10(ctr_true))

def fold_error_ratio(ctr_pred, ctr_true):
    """Ratio-space fold-error: max(pred/true, true/pred)"""
    return np.maximum(ctr_pred / ctr_true, ctr_true / ctr_pred)
```

**Summary statistics (per D-09):**
- Median fold-error
- % within 2x (fold-error < 2.0)
- % within 10x (fold-error < 10.0)
- RMSE of log10(Ctr)
- R² of log10(Ctr)

## Validation Figure Design

### Main Parity Plot (per D-11, D-12)

Log-log scale parity plot with dual-method overlay:

- **X-axis:** Published Ctr (log10 scale)
- **Y-axis:** Predicted Ctr (log10 scale)
- **Identity line:** Black solid diagonal
- **Error bands:** ±2x as gray dashed lines (log10(Ctr) ± log10(2) ≈ ±0.301)
- **ML predictions:** Filled circles, colored by RAFT type
- **Mayo predictions:** Open diamonds, colored by RAFT type
- **Connection lines:** Thin gray lines connecting ML and Mayo predictions for same literature point
- **Error bars:** ML Bootstrap 95% CI on Y-axis only (X is ground truth)

**Color scheme (4 RAFT types):**
- Dithioester: #E64B35 (red)
- Trithiocarbonate: #4DBBD5 (cyan)
- Xanthate: #00A087 (green)
- Dithiocarbamate: #8491B4 (blue-gray)

**matplotlib implementation pattern:**
```python
fig, ax = plt.subplots(figsize=(8, 8))
# Identity line and error bands
lims = [-1, 5]
ax.plot(lims, lims, 'k-', lw=1.5, zorder=1)
ax.plot(lims, [l + np.log10(2) for l in lims], 'k--', lw=0.8, alpha=0.5)
ax.plot(lims, [l - np.log10(2) for l in lims], 'k--', lw=0.8, alpha=0.5)
# ML points with error bars
ax.errorbar(x_true, y_ml, yerr=ci_ml, fmt='o', ...)
# Mayo points
ax.scatter(x_true, y_mayo, marker='D', facecolors='none', ...)
# Connection lines
for i in range(n):
    ax.plot([x_true[i], x_true[i]], [y_ml[i], y_mayo[i]], 'gray', lw=0.5, alpha=0.5)
```

## Validation Workflow

End-to-end validation pipeline for each literature data point:

1. **Load literature CSV** → parse RAFT type, true Ctr, conditions
2. **Generate simulated data** → `simulate_raft(params, raft_type)` with true Ctr
3. **Add noise** → multiply Mn by (1 + N(0, 0.03²))
4. **ML prediction** → encode ctFP → model inference → get predicted log10(Ctr) + Bootstrap CI
5. **Mayo prediction** → ODE curve fitting with `minimize_scalar` → get fitted Ctr
6. **Compute fold-errors** → for both ML and Mayo
7. **Generate summary table** → per-point and aggregate statistics
8. **Plot parity figure** → dual-method overlay

## File Structure

```
data/literature/
  literature_ctr.csv          # Curated dataset (14 rows)

src/
  literature_validation.py    # Mayo fitter, fold-error, validation orchestration

figures/validation/
  parity_ml_vs_mayo.png       # Main dual-method parity plot
  fold_error_comparison.png   # Fold-error scatter (optional)

notebooks/
  04-validate.ipynb           # Interactive validation notebook
```

## Existing Code Dependencies

| Module | Function | Phase 4 Usage |
|--------|----------|---------------|
| `src/raft_ode.py` | `simulate_raft()` | Generate simulated validation data per literature condition |
| `src/ctfp_encoder.py` | `transform()` | Encode simulated data to ctFP for ML input |
| `src/model.py` | `SimpViT` | Load trained model for inference |
| `src/evaluate.py` | `compute_test_metrics()` | R²/RMSE computation on validation set |

## Risks and Mitigations

1. **Literature Ctr values may not match ODE model assumptions** — Our ODE uses simplified moment equations; real experiments have complexities (chain-length-dependent termination, solvent effects). Mitigation: we compare on simulated data (D-07), not real experimental curves.

2. **Mayo fitter may converge to local minima** — Mitigation: use bounded optimization with wide bounds; verify by running from multiple starting points for a few test cases.

3. **Bootstrap CI may not be available** — If `bootstrap_heads.pth` is not downloaded from training. Mitigation: check file existence; fall back to point estimates without CI if missing.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| VAL-01 | 收集10+篇文献的已发表Ctr实验值，覆盖多种RAFT剂类型 | 14-point curated dataset spanning 4 RAFT classes |
| VAL-02 | 每个文献Ctr值标注测量方法和实验条件 | Dataset includes raft_type, method, monomer, solvent, temperature |
| VAL-03 | 模型预测与发表值对比，报告fold-error | Fold-error computation (log + ratio) with summary statistics |
| EVL-04 | 实现Mayo方程基线，在相同文献验证集上对比ML与传统方法 | Mayo ODE fitter with minimize_scalar, fair comparison on simulated data |
</phase_requirements>

## RESEARCH COMPLETE
