# Phase 1: ODE Foundation and ctFP Encoder - Research

**Researched:** 2026-03-25
**Domain:** RAFT polymerization kinetics ODE modeling, method of moments, fingerprint image encoding
**Confidence:** HIGH

## Summary

Phase 1 requires building two root dependencies: (1) a RAFT polymerization ODE simulator based on the method of moments that tracks Mn and dispersity as a function of conversion, and (2) a dual-channel ctFP encoder that converts ODE output trajectories into 64x64x2 image tensors. Both must be validated before any downstream data generation.

The ODE system is well-established in the polymer chemistry literature. The method of moments reduces infinite chain-length distribution balances to a tractable set of ~10-15 ODEs by tracking zeroth, first, and second moments of living and dead chain distributions. Two key papers provide the mathematical framework: Wilding et al. (Macromolecules 2023) for dispersity modeling with experimental validation, and Moad (Polymers 2022) for the partial moments method specifically for RAFT. The critical implementation decision is the two-stage pre-equilibrium model for dithioesters (slow fragmentation of the initial RAFT intermediate), which produces the characteristic inhibition period and rate retardation absent from simplified single-equilibrium models.

For ODE validation, three well-documented RAFT systems have published Mn-conversion and dispersity-conversion curves: (1) CDB/styrene at 60-110C (dithioester, Moad 2000, Arita/Buback/Vana 2005), (2) dodecyl trithiocarbonate/MMA at 60-90C (Moad et al., multiple publications), and (3) O-ethyl xanthate/vinyl acetate at 60C (Ray et al. 2012). The ctFP encoder is a direct adaptation of ViT-RR's `transform()` function with changed axis semantics.

**Primary recommendation:** Implement the ODE system in two variants (pre-equilibrium for dithioesters, single-equilibrium for others), validate against 3 published systems, freeze the ctFP encoder as a shared importable module, then generate a 1000-sample diagnostic dataset to confirm numerical stability across the full parameter space.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Complete moment equation system tracking [M], [I], [CTA], [P.], [Int], lambda_0/lambda_1/lambda_2 (living + dead chains). ~10-15 ODE variables.
- **D-02:** Mn = lambda_1/lambda_0 * M_monomer, D = lambda_2*lambda_0/lambda_1^2. No steady-state approximations for radicals.
- **D-03:** Per-type ODE: dithioester uses two-stage pre-equilibrium (slow fragmentation); trithiocarbonate, xanthate, dithiocarbamate use simplified equilibrium. kadd/kfrag ratio distinguishes behavior.
- **D-04:** Ctr output in log10(Ctr) scale.
- **D-05:** Inhibition period = t_inh/t_total (dimensionless 0-1), where t_inh is time to reach 1% conversion.
- **D-06:** Retardation factor = Rp(RAFT)/Rp(no CTA) at 50% conversion (dimensionless 0-1).
- **D-07:** Ctr range: 0.01-10000 (log10: -2 to 4), log-uniform sampling.
- **D-08:** [CTA]_0/[M]_0 range: 0.001-0.1, log-uniform sampling.
- **D-09:** Temperature fixed, not a variable.
- **D-10:** Four RAFT agent types equally sampled (~250K each for 1M total).
- **D-11:** Validation: extreme-parameter limit checks + literature curve comparison for >= 3 RAFT types (dithioester, trithiocarbonate, xanthate).

### Claude's Discretion
- ODE solver configuration (Radau parameters, tolerance settings)
- ctFP normalization strategy (Mn and D value-to-pixel mapping)
- Diagnostic dataset (1000 samples) parameter grid distribution

### Deferred Ideas (OUT OF SCOPE)
None
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SIM-01 | RAFT ODE model for forward simulation of Mn-conversion and D-conversion curves at arbitrary Ctr | Method of moments ODE system from Wilding 2023 / Moad 2022; solve_ivp with Radau; moment-to-Mn/D conversion formulas |
| SIM-02 | ODE supports all RAFT agent types with two-stage pre-equilibrium for dithioesters | Per-type branching: dithioester adds pre-equilibrium species (Int_pre, [CTA_0]) with separate kadd_0/kfrag_0; others use single equilibrium |
| SIM-03 | Parameter space covers Ctr~0.01-10000, [CTA]/[M]~0.001-0.1 | Log-uniform sampling; kadd/kfrag ratios derive Ctr; published rate constants from Moad/Barner-Kowollik reviews provide realistic ranges |
| ENC-01 | 64x64x2 ctFP image (Ch1=Mn, Ch2=D), x=[CTA]/[M], y=conversion | Direct adaptation of ViT-RR transform(); normalization needed for Mn channel |
| ENC-02 | Shared ctFP encoder between training and web-app | Single ctfp_encoder.py module with no framework dependencies; import in both contexts |
</phase_requirements>

## Standard Stack

### Core (Phase 1 specific)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| SciPy | 1.16.3 (installed) | `solve_ivp(method='Radau')` for stiff RAFT ODE integration | Canonical Python ODE solver; Radau IIA order-5 implicit method handles stiffness from fast RAFT equilibrium vs slow propagation |
| NumPy | 2.3.5 (installed) | Moment equation arrays, ctFP image construction | Foundation for all numerical operations |
| Matplotlib | 3.10.6 (installed) | ODE validation plots (Mn vs conversion, D vs conversion), ctFP visualization | Standard scientific plotting for validation comparison |
| pytest | 8.4.2 (installed) | Unit and integration tests for ODE and encoder | Standard Python test framework |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| PyTorch | 2.10.0 (installed) | ctFP encoder output as torch.Tensor | Only for the final tensor conversion in ctfp_encoder.py |
| joblib | 1.5.2 (installed) | Parallel 1000-sample diagnostic generation | Only for the diagnostic dataset; full-scale parallelism is Phase 2 |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Radau | BDF | BDF is also good for stiff problems; Radau is order-5 vs BDF max order-5, but Radau is generally more accurate for smooth stiff problems; both viable |
| Radau | RK45 | RK45 fails on stiff RAFT systems (kadd >> kp); do not use |
| Full moments | PSSA (pseudo-steady-state approximation) | PSSA removes stiffness but assumes constant radical concentration, which eliminates the inhibition and retardation signals we need for the 3-parameter output |

**Installation:** All required packages are already installed. No additional installation needed.

## Architecture Patterns

### Recommended Project Structure
```
src/
  raft_ode.py          # ODE system definition and integration
  ctfp_encoder.py      # Shared ctFP encoding function
tests/
  test_raft_ode.py     # ODE unit tests and validation
  test_ctfp_encoder.py # Encoder unit tests
  conftest.py          # Shared fixtures (parameter sets, expected outputs)
notebooks/
  01_ode_validation.ipynb  # Interactive validation against literature curves
```

### Pattern 1: Method of Moments ODE System
**What:** Track zeroth (lambda_0), first (lambda_1), and second (lambda_2) moments for both living radical chains and dead polymer chains, plus small-molecule species [M], [I], [CTA], [P.], [Int].
**When:** Always -- this is the core of the data generation pipeline.
**Key equations:**
```python
# State vector (for single-equilibrium model):
# y = [M, I, CTA, P_dot, Int, mu0, mu1, mu2, lam0, lam1, lam2]
# where mu = living chain moments, lam = dead chain moments

# Mn and D from moments:
# total_lam0 = mu0 + lam0  (living + dead zeroth moment)
# total_lam1 = mu1 + lam1
# total_lam2 = mu2 + lam2
# Mn = total_lam1 / total_lam0 * M_monomer
# D  = total_lam2 * total_lam0 / total_lam1**2

# Conversion:
# conversion = 1 - [M] / [M]_0
```

### Pattern 2: Per-Type ODE Branching for Dithioester Pre-equilibrium
**What:** Dithioesters require tracking additional species: the initial RAFT agent (CTA_0), the pre-equilibrium intermediate (Int_pre), and separate kadd_0/kfrag_0 rate constants for the initial chain transfer step. The pre-equilibrium governs the inhibition period.
**When:** Only for dithioester RAFT agent type.
**State vector expansion:**
```python
# Dithioester state vector adds:
# y = [..., CTA_0, Int_pre]
# Pre-equilibrium: P. + CTA_0 -> Int_pre -> R. + macro-CTA
# with kadd_0 (addition to initial CTA) and kfrag_0 (fragmentation releasing R-group)
# R. then reinitiates polymerization
# Main equilibrium proceeds with macro-CTA as normal
```

### Pattern 3: Conversion-Spaced Output Sampling
**What:** Use `dense_output=True` from solve_ivp to interpolate the ODE solution at evenly-spaced conversion values (not time values). This ensures uniform ctFP pixel density along the y-axis (conversion).
**When:** Always -- time-spaced sampling wastes pixel resolution at early times where conversion changes slowly.
```python
from scipy.integrate import solve_ivp

sol = solve_ivp(ode_rhs, t_span, y0, method='Radau',
                dense_output=True, rtol=1e-8, atol=1e-10)

# Interpolate at uniform conversion grid
target_conversions = np.linspace(0.02, 0.95, 50)
# Find time for each target conversion via root-finding on dense output
```

### Pattern 4: Shared Encoder Module
**What:** ctfp_encoder.py is a pure function with no Streamlit or training framework dependencies. It accepts raw data and returns a torch.Tensor.
**When:** Always -- this is the critical shared interface.
```python
# ctfp_encoder.py
import numpy as np
import math
import torch

def transform(data, img_size=64):
    """
    Encode RAFT experimental data into a ctFP image tensor.

    Args:
        data: list of (cta_ratio, conversion, mn_normalized, dispersity) tuples
              cta_ratio: [CTA]/[M] in [0, max_ratio], normalized to [0, 1]
              conversion: monomer conversion in [0, 1]
              mn_normalized: Mn / Mn_theory (dimensionless, typically 0-2)
              dispersity: Mw/Mn (typically 1.0 - 3.0)
        img_size: pixel dimensions (default 64)

    Returns:
        torch.Tensor of shape (2, img_size, img_size)
        Channel 0: normalized Mn
        Channel 1: dispersity
    """
    img = np.zeros((2, img_size, img_size))
    for cta_ratio_norm, conv, mn_norm, disp in data:
        col = min(math.floor(cta_ratio_norm * img_size), img_size - 1)
        row = min(math.floor(conv * img_size), img_size - 1)
        img[0][row][col] = mn_norm
        img[1][row][col] = disp
    return torch.tensor(img, dtype=torch.float32)
```

### Anti-Patterns to Avoid
- **Steady-state radical assumption (PSSA):** Removes stiffness but also removes the inhibition period and retardation signals. Decision D-02 explicitly forbids this.
- **Time-uniform ODE output sampling:** Wastes pixel resolution; use conversion-spaced sampling instead.
- **Unnormalized Mn in ctFP channel:** Raw Mn spans 1,000-200,000 g/mol. Normalize by Mn_theory = [M]_0/[CTA]_0 * MW_monomer to get values near 1.0 for well-controlled RAFT.
- **Single ODE function for all RAFT types:** Dithioester pre-equilibrium adds 2 species and 2 rate constants; forcing all types through the same state vector wastes computation and risks masking bugs. Use separate ODE functions that share common sub-equations.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Stiff ODE integration | Custom Euler/RK4 stepper | `scipy.integrate.solve_ivp(method='Radau')` | RAFT equilibrium creates stiffness ratios of 10^6+; explicit methods require step sizes < 10^-8 s, making generation impossibly slow |
| Root-finding for conversion targets | Manual bisection on ODE output | `scipy.optimize.brentq` on dense_output | Robust bracketed root-finding; handles non-monotonic edge cases |
| Parallel diagnostic generation | Manual multiprocessing.Pool | `joblib.Parallel(n_jobs=-1)(delayed(fn)(p) for p in params)` | Handles serialization, error propagation, and progress tracking |

## Common Pitfalls

### Pitfall 1: Wrong Mechanism for Dithioester Pre-equilibrium (C1 from PITFALLS.md)
**What goes wrong:** Using a single-equilibrium model for dithioesters produces no inhibition period and incorrect Mn evolution at early conversion.
**Why it happens:** The "unified" RAFT mechanism is often simplified to one equilibrium. Pre-equilibrium is rate-determining only for dithioesters.
**How to avoid:** Implement two-stage ODE for dithioesters with separate kadd_0/kfrag_0 for initial CTA consumption. Validate by checking that dithioester curves show visible inhibition at early conversion.
**Warning signs:** ODE-generated D is monotonically decreasing for dithioesters; inhibition period = 0 for all dithioesters regardless of R-group.

### Pitfall 2: ODE Numerical Instability at Extreme Parameters (M1)
**What goes wrong:** solve_ivp with default tolerances fails or returns incorrect results at extreme kadd/kfrag ratios (10^6+).
**Why it happens:** RAFT kinetics are stiff; fast radical reactions coupled with slow polymer growth create timescale separation.
**How to avoid:** Use Radau with tight tolerances (rtol=1e-8, atol=1e-10). Use per-component atol since species concentrations span orders of magnitude ([M]~1 mol/L vs [P.]~10^-8 mol/L). Add post-integration validity checks: conversion in [0,1], Mn > 0, D in [1.0, 5.0].
**Warning signs:** solve_ivp returns success=False for > 1% of samples; D values < 1.0 or > 10.

### Pitfall 3: ctFP Pixel Overwrite for Single-[CTA]/[M] Data (C4)
**What goes wrong:** When multiple ODE output points at the same [CTA]/[M] map to the same pixel column, later values overwrite earlier ones silently.
**Why it happens:** The transform() function uses last-write-wins for duplicate pixel coordinates.
**How to avoid:** For training data, this is acceptable because each ODE run uses one fixed [CTA]/[M] and varies conversion (different rows). For web-app input, users may provide multiple data points at the same [CTA]/[M] and similar conversion -- add a deduplication warning. For training, the y-axis (conversion) naturally spreads points across rows.
**Warning signs:** ctFP image shows only 1-2 columns activated.

### Pitfall 4: Mn Normalization Scale Mismatch
**What goes wrong:** Raw Mn values (1,000 - 200,000 g/mol) dominate pixel intensity; model learns only high-Mn systems.
**Why it happens:** No normalization applied to Mn before encoding.
**How to avoid:** Normalize Mn by Mn_theory = [M]_0/[CTA]_0 * MW_monomer. Well-controlled RAFT gives Mn/Mn_theory near 1.0; poorly controlled gives values >> 1 at early conversion. This produces a consistent scale across all parameter regimes.
**Warning signs:** Channel 0 of ctFP has extreme value range (0 to 200,000).

### Pitfall 5: Retardation Factor Computation Ambiguity
**What goes wrong:** Rp(RAFT)/Rp(no CTA) at "50% conversion" is ambiguous if the RAFT system never reaches 50% conversion in the integration window.
**Why it happens:** High-Ctr dithioesters with strong retardation may reach only 20% conversion in the same time frame.
**How to avoid:** Run two ODE integrations per parameter set: one with CTA (full RAFT), one without CTA (conventional FRP). Compute retardation at the lower of {50% conversion, max conversion reached by RAFT system}. If RAFT system reaches < 5% conversion, flag as "strong inhibition" and set retardation factor to a near-zero sentinel value.
**Warning signs:** Many samples have retardation_factor = NaN or undefined.

## Code Examples

### Example 1: RAFT ODE Right-Hand Side (Single Equilibrium Model)
```python
# Source: Adapted from Wilding et al. Macromolecules 2023 + Moad Polymers 2022
# This is the simplified single-equilibrium model for TTC/xanthate/dithiocarbamate

def raft_ode_single_eq(t, y, kd, f, ki, kp, kt, kadd, kfrag):
    """
    RAFT polymerization ODE with method of moments (single equilibrium).

    State vector y:
      [0] M   - monomer concentration
      [1] I   - initiator concentration
      [2] CTA - RAFT agent concentration (macro-CTA after first transfer)
      [3] P   - propagating radical concentration (sum of all chain lengths)
      [4] Int - RAFT intermediate radical concentration
      [5] mu0 - zeroth moment of living chains
      [6] mu1 - first moment of living chains
      [7] mu2 - second moment of living chains
      [8] lam0 - zeroth moment of dead chains
      [9] lam1 - first moment of dead chains
      [10] lam2 - second moment of dead chains
    """
    M, I, CTA, P, Int, mu0, mu1, mu2, lam0, lam1, lam2 = y

    # Derived rates
    R_init = 2 * f * kd * I           # initiation rate
    R_prop = kp * P * M               # propagation rate
    R_add  = kadd * P * CTA           # addition to RAFT agent
    R_frag = kfrag * Int              # fragmentation of intermediate
    R_term = kt * P**2                # bimolecular termination

    # Species balances
    dM_dt   = -R_prop
    dI_dt   = -kd * I
    dCTA_dt = -R_add + R_frag         # consumed by addition, regenerated by fragmentation
    dP_dt   = R_init + R_frag - R_add - 2*R_term  # born by init+frag, consumed by add+term
    dInt_dt = R_add - R_frag          # born by addition, consumed by fragmentation

    # Living chain moments (propagating radicals)
    dmu0_dt = R_init - 2*kt*mu0*P    # created by initiation, destroyed by termination
    dmu1_dt = kp*M*mu0 + R_init - 2*kt*mu1*P  # growth by propagation
    dmu2_dt = kp*M*(2*mu1 + mu0) + R_init - 2*kt*mu2*P

    # Dead chain moments (terminated + transferred chains)
    dlam0_dt = kt*mu0*P + R_add      # termination products + transfer products
    dlam1_dt = kt*mu1*P + kadd*mu1*CTA
    dlam2_dt = kt*mu2*P + kadd*mu2*CTA

    return [dM_dt, dI_dt, dCTA_dt, dP_dt, dInt_dt,
            dmu0_dt, dmu1_dt, dmu2_dt, dlam0_dt, dlam1_dt, dlam2_dt]
```

### Example 2: ODE Integration with solve_ivp
```python
# Source: SciPy docs (https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)
from scipy.integrate import solve_ivp
import numpy as np

def simulate_raft(params, t_end=36000, n_conv_points=50):
    """
    Run RAFT ODE simulation and return Mn, D at evenly-spaced conversions.
    """
    y0 = initial_conditions(params)

    # Per-component absolute tolerances (species concentrations vary by 10+ orders)
    atol = np.array([
        1e-6,   # M (~1 mol/L)
        1e-10,  # I (~0.01 mol/L)
        1e-8,   # CTA (~0.01 mol/L)
        1e-14,  # P. (~1e-8 mol/L) -- very small!
        1e-14,  # Int (~1e-8 mol/L)
        1e-14, 1e-10, 1e-6,  # mu0, mu1, mu2
        1e-10, 1e-6, 1e-2,   # lam0, lam1, lam2
    ])

    sol = solve_ivp(
        raft_ode_single_eq,
        t_span=(0, t_end),
        y0=y0,
        method='Radau',
        rtol=1e-8,
        atol=atol,
        dense_output=True,
        args=(params['kd'], params['f'], params['ki'],
              params['kp'], params['kt'], params['kadd'], params['kfrag'])
    )

    if not sol.success:
        return None  # log failure, don't silently discard

    # Sample at uniform conversion grid
    M0 = y0[0]
    target_conv = np.linspace(0.02, 0.95, n_conv_points)
    # ... (root-find times for each target conversion using sol.sol)

    return results  # dict with conversion, Mn, D arrays
```

### Example 3: ctFP Encoder (from ViT-RR transform, adapted)
```python
# Source: ViT-RR model_utils.py transform(), adapted for ctFP semantics
import numpy as np
import math
import torch

def transform(data, img_size=64):
    """
    Encode RAFT kinetic data into chain-transfer fingerprint (ctFP).

    Args:
        data: array-like of (cta_ratio_norm, conversion, mn_norm, dispersity)
              cta_ratio_norm: [CTA]/[M] normalized to [0, 1] range
              conversion: in [0, 1]
              mn_norm: Mn / Mn_theory (dimensionless)
              dispersity: Mw/Mn >= 1.0
        img_size: fingerprint resolution (default 64)

    Returns:
        torch.Tensor shape (2, img_size, img_size)
    """
    img = np.zeros((2, img_size, img_size), dtype=np.float32)
    for cta_norm, conv, mn_n, disp in data:
        col = min(int(cta_norm * img_size), img_size - 1)
        row = min(int(conv * img_size), img_size - 1)
        img[0, row, col] = mn_n
        img[1, row, col] = disp
    return torch.tensor(img, dtype=torch.float32)
```

## Literature Validation Targets

Three published RAFT systems for ODE validation (per D-11):

### System 1: Dithioester -- CDB/Styrene at 60-110C
**References:**
- Moad et al., Polym. Int. 2000 -- foundational data, CDB concentrations from 0.0001-0.0029 M
- Arita, Buback, Vana, Macromolecules 2005, 38, 7935 -- high-temperature (120-180C) kinetic study
- Barner-Kowollik et al., J. Polym. Sci. Part A 2001, 39, 1353 -- rate coefficient assessment

**Expected behavior:**
- Visible inhibition period (delayed onset of polymerization)
- Rate retardation vs conventional FRP
- D dip then rise (not monotonically decreasing)
- Mn linear with conversion after inhibition period
- Typical Ctr: 10-50 for CDB/styrene systems

### System 2: Trithiocarbonate -- Dodecyl TTC/MMA at 60-90C
**References:**
- Moad, Rizzardo, Thang -- multiple Aust. J. Chem. reviews with tabulated data
- Wilding et al., Macromolecules 2023 -- dispersity model validated against TTC/acrylate data
- Sigma-Aldrich technical documents -- S-dodecyl S-(2-cyano-4-carboxy)but-2-yl TTC with MMA

**Expected behavior:**
- Minimal or no inhibition period
- Minimal rate retardation (retardation factor near 1.0)
- Linear Mn growth with conversion
- D < 1.2-1.4 throughout
- Typical Ctr: 10-100 for TTC/MMA systems

### System 3: Xanthate -- O-Ethyl Xanthate/Vinyl Acetate at 60C
**References:**
- Ray et al., J. Appl. Polym. Sci. 2012, 126 -- X1/X2 xanthate-mediated VAc polymerization
- RAFT/MADIX miniemulsion polymerization studies (Oliveira et al. 2018)

**Expected behavior:**
- Pseudo-first order kinetics up to ~85% conversion
- Linear Mn growth with conversion
- D ~ 1.2 up to 65% conversion, then gradually increases
- Typical Ctr: 0.5-5 for xanthate/VAc systems
- Less controlled than TTC/MAM systems (D higher)

## RAFT Rate Constant Ranges for Parameter Space

Based on Moad/Rizzardo/Thang reviews and Barner-Kowollik et al.:

| Parameter | Range | Units | Notes |
|-----------|-------|-------|-------|
| kp (propagation) | 10^2 - 10^4 | L/mol/s | Monomer-dependent: styrene ~340 (60C), MMA ~650 (60C), VAc ~6700 (60C) |
| kt (termination) | 10^6 - 10^9 | L/mol/s | Chain-length dependent; use average value |
| kd (initiator decomposition) | 10^-6 - 10^-4 | 1/s | AIBN: ~1.5e-5 (60C) |
| f (initiator efficiency) | 0.3 - 0.8 | - | Typically 0.5-0.7 |
| kadd (RAFT addition) | 10^4 - 10^7 | L/mol/s | Dithioester > TTC > xanthate |
| kfrag (RAFT fragmentation) | 10^-2 - 10^5 | 1/s | Controversial for dithioesters (SF vs IRT debate) |
| Ctr = kadd*kfrag/(k_{-add}*kp) | 0.01 - 10000 | - | The target prediction parameter |

**Key insight for parameter sampling:** Ctr is the physically meaningful parameter, not kadd and kfrag independently. Sample log10(Ctr) uniformly, then derive kadd/kfrag combinations that produce that Ctr. This avoids correlated parameter sampling and ensures uniform coverage of the prediction target space.

## ODE Solver Configuration (Claude's Discretion)

**Recommended settings for solve_ivp:**
```python
solve_ivp(
    fun=raft_ode_rhs,
    t_span=(0, t_end),
    y0=y0,
    method='Radau',       # Implicit Runge-Kutta, order 5, handles stiffness
    rtol=1e-8,            # 8 significant digits relative accuracy
    atol=per_component,   # Array of per-variable absolute tolerances
    dense_output=True,    # Enable interpolation at arbitrary time points
    max_step=100.0,       # Prevent stepping over fast pre-equilibrium dynamics
)
```

**Why rtol=1e-8:** Default (1e-3) is too loose for moment equations where small errors in mu2 propagate to large errors in D. Literature ODE studies for RAFT use rtol=1e-6 to 1e-10.

**Why per-component atol:** Species concentrations span 10+ orders of magnitude. Radical concentration [P.] ~ 1e-8 mol/L requires atol ~ 1e-14 to maintain meaningful relative accuracy, while [M] ~ 1 mol/L only needs atol ~ 1e-6.

**Why max_step=100:** The pre-equilibrium for dithioesters can complete within seconds to minutes. Without max_step, the solver may step over the entire pre-equilibrium in one step, missing the inhibition dynamics.

## ctFP Normalization Strategy (Claude's Discretion)

**Channel 0 (Mn):** Normalize by Mn_theory = [M]_0/[CTA]_0 * MW_monomer * conversion. This produces values near 1.0 for ideal RAFT control. Alternative: normalize by Mn_theory at full conversion (fixed denominator per run). Use the fixed-denominator version for simplicity -- it gives Mn_norm in range [0, ~1.5] for well-controlled RAFT, with values > 1 indicating poor control.

**Channel 1 (Dispersity D):** D is already dimensionless and bounded (1.0 to ~3.0 for RAFT). Use raw D values in the ctFP -- no normalization needed. Clip at D=4.0 for safety (any D > 4 indicates uncontrolled polymerization).

**Coordinate normalization:**
- x-axis ([CTA]/[M]): Normalize to [0, 1] by dividing by max([CTA]/[M]) in the dataset. For training data, since [CTA]/[M] is in [0.001, 0.1], normalize by 0.1.
- y-axis (conversion): Already in [0, 1].

## Diagnostic Dataset Design (Claude's Discretion)

**1000 samples distributed as:**
- 250 per RAFT agent type (dithioester, trithiocarbonate, xanthate, dithiocarbamate)
- Within each type, sample a 10x25 grid:
  - 10 log-uniform values of Ctr from agent-appropriate range
  - 25 log-uniform values of [CTA]/[M] from 0.001 to 0.1
  - Other kinetic parameters (kp, kt, kd, f) fixed at literature values for one representative monomer per type

**Purpose:** Verify numerical stability across the full parameter space before committing to 1M-sample generation. Success criterion: < 2% failure rate (< 20 of 1000 samples return solve_ivp failure or physically invalid outputs).

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| odeint (LSODA wrapper) | solve_ivp (Radau/BDF) | SciPy 1.0+ (2017) | Modern API with dense_output, events, per-component tolerances |
| Single RAFT equilibrium | Two-stage pre-equilibrium | Barner-Kowollik 2006+ | Required for correct dithioester inhibition modeling |
| Chain-length distribution ODE | Method of moments | Standard since 2000s | Reduces ODE system from infinite to ~15 equations |
| Slow fragmentation (SF) only | SF + IRT + IRTO models | Ongoing debate | Project uses observable retardation factor to sidestep mechanism debate |

**Deprecated/outdated:**
- `scipy.integrate.odeint`: Legacy wrapper for LSODA; use `solve_ivp` instead
- Single-equilibrium RAFT ODE for all agent types: Incorrect for dithioesters; per D-03

## Open Questions

1. **Exact moment equations for RAFT intermediate termination**
   - What we know: Cross-termination of RAFT intermediates (Int + P. -> dead) affects retardation. The IRT model includes this; SF does not.
   - What's unclear: Whether to include IRT terms in the moment equations or only model retardation through slow fragmentation.
   - Recommendation: Start with the simpler SF model (no cross-termination). If validation against dithioester literature curves shows poor retardation fit, add IRT terms as a second iteration. This aligns with the project's framing of retardation as an "observable rate deceleration" rather than a mechanistic parameter.

2. **Dithiocarbamate validation data**
   - What we know: D-11 requires validation for 3 types (dithioester, TTC, xanthate). Dithiocarbamates are included in the training data (D-10) but not required for validation.
   - What's unclear: Limited published Mn-conversion curves for dithiocarbamate-mediated RAFT.
   - Recommendation: Treat dithiocarbamate as using the same single-equilibrium model as xanthate (both are less-active RAFT agents with low kadd). Validate indirectly by confirming low-Ctr behavior matches xanthate validation.

3. **Conversion-to-time mapping for retardation factor**
   - What we know: D-06 defines retardation as Rp(RAFT)/Rp(no CTA) at 50% conversion.
   - What's unclear: Rp at a specific conversion requires dConversion/dt at that conversion, which means we need to know the time at which 50% conversion is reached -- this requires a parallel no-CTA simulation.
   - Recommendation: Run two ODE integrations per sample (with and without CTA). Compute Rp = d(conversion)/dt at the time point where each system reaches 50% conversion. This doubles compute cost but is the only correct approach.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python | Runtime | Yes | 3.13.9 | -- |
| SciPy | ODE integration | Yes | 1.16.3 | -- |
| NumPy | Array operations | Yes | 2.3.5 | -- |
| PyTorch | ctFP tensor output | Yes | 2.10.0 | -- |
| Matplotlib | Validation plots | Yes | 3.10.6 | -- |
| pytest | Testing | Yes | 8.4.2 | -- |
| joblib | Parallel diagnostic generation | Yes | 1.5.2 | -- |

**Missing dependencies with no fallback:** None.
**Missing dependencies with fallback:** None.

**Note:** Python 3.13.9 is installed, which is newer than the recommended 3.11 in CLAUDE.md. This should work fine with all current packages. No downgrade needed.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.4.2 |
| Config file | none -- see Wave 0 |
| Quick run command | `pytest tests/ -x -q` |
| Full suite command | `pytest tests/ -v` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SIM-01 | ODE produces Mn and D vs conversion for given Ctr | unit | `pytest tests/test_raft_ode.py::test_forward_simulation -x` | Wave 0 |
| SIM-02 | Dithioester pre-equilibrium produces distinct curves from single-eq | integration | `pytest tests/test_raft_ode.py::test_preequilibrium_distinct -x` | Wave 0 |
| SIM-02 | All 4 RAFT agent types run without error | unit | `pytest tests/test_raft_ode.py::test_all_agent_types -x` | Wave 0 |
| SIM-03 | ODE succeeds across full Ctr range (0.01-10000) | integration | `pytest tests/test_raft_ode.py::test_parameter_range_coverage -x` | Wave 0 |
| SIM-03 | ODE succeeds across full [CTA]/[M] range (0.001-0.1) | integration | `pytest tests/test_raft_ode.py::test_cta_ratio_range -x` | Wave 0 |
| ENC-01 | transform() produces (2, 64, 64) tensor | unit | `pytest tests/test_ctfp_encoder.py::test_output_shape -x` | Wave 0 |
| ENC-01 | Mn in channel 0, D in channel 1, correct axis mapping | unit | `pytest tests/test_ctfp_encoder.py::test_channel_assignment -x` | Wave 0 |
| ENC-02 | Same input produces byte-identical output (determinism) | unit | `pytest tests/test_ctfp_encoder.py::test_deterministic_output -x` | Wave 0 |
| -- | Extreme-parameter limit check (high Ctr: D < 1.2 at 50% conv) | integration | `pytest tests/test_raft_ode.py::test_limit_behavior -x` | Wave 0 |
| -- | 1000-sample diagnostic succeeds with < 2% failure | smoke | `pytest tests/test_raft_ode.py::test_diagnostic_dataset -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/ -x -q`
- **Per wave merge:** `pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/conftest.py` -- shared fixtures (parameter sets, expected outputs)
- [ ] `tests/test_raft_ode.py` -- ODE tests covering SIM-01, SIM-02, SIM-03
- [ ] `tests/test_ctfp_encoder.py` -- encoder tests covering ENC-01, ENC-02
- [ ] `pytest.ini` or `pyproject.toml` with [tool.pytest.ini_options] -- test configuration
- [ ] `src/` directory creation

## Project Constraints (from CLAUDE.md)

- **Tech stack:** PyTorch + Streamlit, matching ViT-RR
- **Model architecture:** SimpViT (64x64 input, patch_size=16, hidden=64, 2-layer Transformer, 4-head attention), output_dim=3
- **Data format:** Dual-channel ctFP (Ch1=Mn, Ch2=D), axes: x=[CTA]/[M], y=conversion
- **Language:** Code comments in Chinese; paper/SI in English
- **GSD workflow:** All code changes through GSD commands
- **Hardware:** Local development (i5-10210U + MX350 2GB); large-scale training on Colab

## Sources

### Primary (HIGH confidence)
- ViT-RR `model_utils.py` -- direct inspection of SimpViT architecture and transform() function
- [Wilding et al. Macromolecules 2023 -- Dispersity Model for RAFT](https://pubs.acs.org/doi/10.1021/acs.macromol.2c01798) -- ODE model with experimental validation
- [Moad, Polymers 2022 -- Method of Partial Moments for RAFT](https://www.mdpi.com/2073-4360/14/22/5013) -- Moment equation methodology
- [SciPy solve_ivp documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html) -- Solver API and Radau configuration
- [Moad, Rizzardo, Thang -- 50th Anniversary RAFT User Guide, Macromolecules 2017](https://pubs.acs.org/doi/10.1021/acs.macromol.7b00767) -- Comprehensive RAFT review with Ctr tables

### Secondary (MEDIUM confidence)
- [Barner-Kowollik et al. -- RAFT Kinetics: Combination of Models, Macromolecules 2008](https://pubs.acs.org/doi/abs/10.1021/ma800388c) -- SF/IRT debate resolution
- [Arita, Buback, Vana -- CDB/Styrene at high T, Macromolecules 2005](https://pubs.acs.org/doi/10.1021/ma051012d) -- Dithioester validation data
- [Zapata-Gonzalez et al. -- RAFT Kinetic Modeling with Missing Step Theory, Chem. Eng. J. 2017](https://www.sciencedirect.com/science/article/abs/pii/S1385894717310033) -- Pre-equilibrium ODE with moment method
- [Rate Retardation Trends in RAFT, Polymer Chemistry RSC 2024](https://pubs.rsc.org/en/content/getauthorversionpdf/d3py01332d) -- Monomer classification by retardation behavior

### Tertiary (LOW confidence)
- Ray et al. xanthate/VAc data -- cited in multiple reviews but not directly accessed; validate against original paper during implementation

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all packages installed and verified; versions current
- Architecture: HIGH -- directly adapted from ViT-RR reference code with well-documented modifications
- ODE equations: MEDIUM -- moment equations well-established in literature but exact implementation details (pre-equilibrium species, moment closure) need verification against primary papers during coding
- Pitfalls: HIGH -- catalogued from both domain literature and PITFALLS.md research document
- Literature validation targets: MEDIUM -- papers identified but experimental data not yet digitized; exact rate constants need extraction from primary sources

**Research date:** 2026-03-25
**Valid until:** 2026-04-25 (stable domain; no fast-moving dependencies)
