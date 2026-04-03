# Phase 5: Streamlit Web Application - Research

**Researched:** 2026-04-03
**Domain:** Streamlit web application for ML inference (PyTorch + Bootstrap CI)
**Confidence:** HIGH

## Summary

Phase 5 builds a single-page Streamlit app that wraps the trained SimpViT model and Bootstrap uncertainty pipeline into a researcher-facing tool. The core technical challenge is bridging raw experimental input (Mn in g/mol, [CTA]/[M] as a ratio) to the normalized ctFP encoding the model expects (mn_norm = Mn/Mn_theory, cta_ratio_norm = [CTA]/[M] / 0.1). This requires the user to provide monomer molecular weight (M_monomer) so the app can compute Mn_theory internally.

All inference components already exist: `src/model.py` (SimpViT), `src/ctfp_encoder.py` (transform), and `src/bootstrap.py` (predict_with_uncertainty). The app is a thin UI layer. The ViT-RR deploy.py provides a complete reference, though ViT-Ctr's Bootstrap differs (head-swapping vs. input-perturbation).

**Primary recommendation:** Build a single `app.py` at project root. Keep all inference logic in existing `src/` modules.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
D-01 through D-14 — see CONTEXT.md for full text. Key constraints:
- st.data_editor with editable table (D-01), st.tabs for Manual/Upload (D-03)
- Excel/CSV upload populates same table (D-02), template download beside uploader (D-04)
- Validation on Predict click: conversion in (0,1), Mn>0, D>=1, >=3 points (D-05)
- Three cards via st.columns(3), Ctr visually prominent (D-06)
- Ctr as 10^x with log10 subtitle (D-07), CI as [lower, upper] (D-08)
- Dual ctFP heatmap below results (D-09/D-10)
- Single-page flow: title->input->predict->results->ctFP->citation (D-11/D-12)
- st.error() for validation (D-13), st.cache_resource for model (D-14)

### Claude's Discretion
- data_editor column config, Excel template format, colormap, page title wording, button style, spinner text, card CSS

### Deferred Ideas (OUT OF SCOPE)
None
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| APP-01 | 手动输入实验数据 | st.data_editor + num_rows="dynamic" + NumberColumn config |
| APP-02 | Excel/CSV上传+模板下载 | st.file_uploader + pd.read_excel/csv; st.download_button + BytesIO |
| APP-03 | 三参数预测+95% CI | predict_with_uncertainty(); 10^x back-transform for Ctr |
| APP-04 | 输入验证 | Pure Python validation; st.error() display |
| APP-05 | ctFP热力图 | matplotlib imshow + st.pyplot() |
| APP-06 | st.cache_resource缓存 | @st.cache_resource on model loader |
</phase_requirements>


## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Streamlit | 1.51.0 (installed; 1.56.0 latest) | Web app framework | Project constraint; matches ViT-RR |
| PyTorch | 2.10.0 (installed) | Model inference | SimpViT + Bootstrap heads |
| NumPy | 2.3.5 (installed) | Array ops, ctFP encoding | Required by ctfp_encoder.py, bootstrap.py |
| pandas | 2.3.3 (installed) | DataFrame for data_editor, Excel I/O | Required for st.data_editor and pd.read_excel |
| matplotlib | 3.10.6 (installed) | ctFP heatmap visualization | imshow for dual-channel fingerprint |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| openpyxl | 3.1.5 (installed) | Excel read/write backend | Template generation and upload parsing |
| SciPy | 1.16.3 (installed) | F-distribution for JCI | Used internally by bootstrap.py |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| st.data_editor | st.number_input per field | data_editor is tabular, natural for multi-row data; number_input needs expanders |
| matplotlib heatmap | plotly | plotly adds interactivity but overkill for 64x64 static image |
| openpyxl template | xlsxwriter | openpyxl already installed, handles read+write; xlsxwriter is write-only |

**Installation:** All dependencies already installed. No new packages needed.

## Architecture Patterns

### Recommended Project Structure
```
app.py                          # Streamlit entry point (UI only)
src/
├── model.py                    # SimpViT (existing)
├── ctfp_encoder.py             # transform() (existing)
├── bootstrap.py                # predict_with_uncertainty() (existing)
└── app_utils.py                # NEW: normalization bridge, validation, template gen
checkpoints/
├── best_model.pth              # Trained model weights
├── bootstrap_heads.pth         # 200 Bootstrap head state dicts
└── calibration.json            # Calibration factors [f1, f2, f3]
```

### Pattern 1: Normalization Bridge (CRITICAL)
**What:** The ctFP encoder expects normalized inputs, but users provide raw experimental values. A bridge function converts raw -> normalized.
**When to use:** Every prediction request.
**Why critical:** The training pipeline normalizes `cta_ratio / 0.1` and `Mn / Mn_theory`. If the app skips this, predictions are garbage.

```python
# Source: src/dataset_generator.py lines 244-248, src/literature_validation.py lines 175-181
def prepare_ctfp_input(df, m_monomer):
    """Bridge raw user input to ctFP encoder format.
    
    df columns: [CTA]/[M], conversion, Mn, D
    m_monomer: monomer molecular weight (g/mol)
    
    Returns: list of (cta_ratio_norm, conversion, mn_norm, dispersity) tuples
    """
    cta_ratio = df['[CTA]/[M]'].values
    cta_ratio_norm = cta_ratio / 0.1  # normalize to [0, 1]
    
    # Mn_theory = [M]/[CTA] * M_monomer = (1/cta_ratio) * M_monomer
    # But user provides [CTA]/[M], so Mn_theory = M_monomer / cta_ratio
    # HOWEVER: each row may have different [CTA]/[M], so Mn_theory varies per row
    mn_theory = m_monomer / cta_ratio  # per-row Mn_theory
    mn_norm = df['Mn'].values / mn_theory
    
    dispersity = df['D'].values
    conversion = df['conversion'].values
    
    return list(zip(cta_ratio_norm, conversion, mn_norm, dispersity))
```

**Key insight on Mn normalization:** In the training data (dataset_generator.py line 245), `Mn_theory = M0/CTA0 * M_monomer`. Since `[CTA]/[M] = CTA0/M0`, this simplifies to `Mn_theory = M_monomer / ([CTA]/[M])`. The user must provide M_monomer (or select a common monomer from a dropdown). This is NOT optional — without it, Mn normalization is impossible.


### Pattern 2: Model Loading with st.cache_resource
**What:** Load model + bootstrap heads + calibration once per Streamlit server process.
**When to use:** App startup.

```python
# Source: Streamlit docs st.cache_resource; ViT-RR deploy.py lines 103-118
import streamlit as st
import torch
import json
from src.model import SimpViT
from src.bootstrap import predict_with_uncertainty

@st.cache_resource
def load_model():
    """Load SimpViT + bootstrap heads + calibration. Cached across reruns."""
    model = SimpViT(num_outputs=3)
    ckpt = torch.load('checkpoints/best_model.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()
    
    # Bootstrap heads: weights_only=False required (list of dicts)
    bootstrap_ckpt = torch.load('checkpoints/bootstrap_heads.pth', map_location='cpu', weights_only=False)
    
    with open('checkpoints/calibration.json') as f:
        cal_factors = json.load(f)
    
    return model, bootstrap_ckpt, cal_factors
```

**Note:** `weights_only=False` for bootstrap_heads.pth is a known decision from Phase 3 (STATE.md). The base model uses `weights_only=True`.

### Pattern 3: st.data_editor with Column Config
**What:** Editable table with typed columns for experimental data input.

```python
import pandas as pd
import streamlit as st

default_df = pd.DataFrame({
    '[CTA]/[M]': pd.Series(dtype='float64'),
    'conversion': pd.Series(dtype='float64'),
    'Mn': pd.Series(dtype='float64'),
    'D': pd.Series(dtype='float64'),
})

edited_df = st.data_editor(
    default_df,
    num_rows="dynamic",  # allows adding/deleting rows
    column_config={
        '[CTA]/[M]': st.column_config.NumberColumn(
            '[CTA]/[M]', help='Molar ratio of CTA to monomer', min_value=0.0, format="%.4f",
        ),
        'conversion': st.column_config.NumberColumn(
            'Conversion', help='Monomer conversion (0-1)', min_value=0.0, max_value=1.0, format="%.3f",
        ),
        'Mn': st.column_config.NumberColumn(
            'Mn (g/mol)', help='Number-average molecular weight', min_value=0.0, format="%.0f",
        ),
        'D': st.column_config.NumberColumn(
            'Đ (Mw/Mn)', help='Dispersity', min_value=1.0, format="%.3f",
        ),
    },
    hide_index=True,
    key="data_input",
)
```

### Pattern 4: Excel Template Generation
**What:** Generate downloadable Excel template with correct column headers and example data.

```python
from io import BytesIO
import pandas as pd

def generate_template():
    """Generate Excel template with headers and 3 example rows."""
    template_df = pd.DataFrame({
        '[CTA]/[M]': [0.005, 0.005, 0.005],
        'conversion': [0.10, 0.30, 0.50],
        'Mn': [15000, 42000, 68000],
        'D': [1.15, 1.20, 1.25],
    })
    buf = BytesIO()
    template_df.to_excel(buf, index=False, engine='openpyxl')
    buf.seek(0)
    return buf

# In the File Upload tab:
st.download_button(
    label="Download Excel Template",
    data=generate_template(),
    file_name="ViT-Ctr_template.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
```

### Pattern 5: Ctr Back-Transform and CI Display
**What:** Model outputs log10(Ctr); display needs original scale with CI.

```python
import numpy as np

# mean_pred[0] = log10(Ctr), half_width[0] = half-width in log10 space
log10_ctr = mean_pred[0]
ctr = 10 ** log10_ctr

# CI in log10 space -> transform to original scale
ctr_lower = 10 ** (log10_ctr - half_width[0])
ctr_upper = 10 ** (log10_ctr + half_width[0])

# For inhibition_period and retardation_factor (indices 1, 2):
# These are already in natural scale, CI is symmetric
inh = mean_pred[1]
inh_lower = inh - half_width[1]
inh_upper = inh + half_width[1]
```

**Key insight:** The CI for Ctr is asymmetric in original scale because the log-transform is nonlinear. `[10^(log-hw), 10^(log+hw)]` is correct. Do NOT compute `10^log ± 10^hw`.

### Anti-Patterns to Avoid
- **Calling transform() with raw Mn values:** The encoder expects mn_norm (Mn/Mn_theory), not raw Mn in g/mol. This is the #1 silent failure mode.
- **Symmetric CI for Ctr in original scale:** `Ctr ± half_width` is wrong. Must transform CI bounds through 10^x individually.
- **Loading model inside prediction function:** Causes reload on every rerun. Must use @st.cache_resource.
- **Using st.cache_data for model:** st.cache_data serializes/deserializes the return value (pickle). Use st.cache_resource for ML models — it stores the object directly.
- **Forgetting weights_only=False for bootstrap_heads.pth:** Will crash with UnpicklingError.


## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Editable data table | Custom HTML/JS table | st.data_editor | Built-in, handles add/delete rows, type validation |
| Excel read/write | Manual openpyxl cell iteration | pd.read_excel / df.to_excel | pandas handles all edge cases (merged cells, dtypes) |
| Model caching | Manual global variable + flag | @st.cache_resource | Handles process lifecycle, invalidation, thread safety |
| Bootstrap inference | Re-implement head-swapping in app.py | src/bootstrap.predict_with_uncertainty() | Already tested, handles all edge cases (cov matrix, JCI) |
| ctFP encoding | Re-implement transform in app.py | src/ctfp_encoder.transform() | ENC-02 mandates shared implementation |
| Input validation | Inline if/else spaghetti | Dedicated validate_input() function | Testable, reusable, clear error messages |

**Key insight:** The app should contain ZERO inference logic. All ML code lives in `src/`. The app is purely UI + normalization bridge.

## Common Pitfalls

### Pitfall 1: Mn Normalization Mismatch
**What goes wrong:** User enters raw Mn (e.g., 50000 g/mol) but ctFP encoder expects mn_norm (Mn/Mn_theory, typically 0.5-2.0). Model sees values 1000x too large, predictions are nonsense.
**Why it happens:** The normalization step is implicit in the training pipeline (dataset_generator.py line 245-246) but not in the user-facing interface.
**How to avoid:** The app MUST collect M_monomer (or offer a monomer dropdown) and compute Mn_theory = M_monomer / [CTA]/[M] per row. Add a sidebar input or a field above the data table.
**Warning signs:** Predicted Ctr values are extreme (log10 > 10 or < -5), or all predictions cluster at the same value regardless of input.

### Pitfall 2: [CTA]/[M] Normalization
**What goes wrong:** User enters [CTA]/[M] = 0.005 but forgets the /0.1 normalization. ctFP column index maps to wrong position.
**Why it happens:** The training data normalizes cta_ratio by dividing by 0.1 (the max of the parameter range). This is hardcoded in dataset_generator.py line 248.
**How to avoid:** The normalization bridge function handles this automatically. User enters raw [CTA]/[M]; the app divides by 0.1 before calling transform().
**Warning signs:** All data points cluster in the leftmost columns of the ctFP image.

### Pitfall 3: predict_with_uncertainty Modifies Model State
**What goes wrong:** predict_with_uncertainty() calls model.load_state_dict() 200 times (once per bootstrap head). If the cached model object is shared, this mutates it.
**Why it happens:** The function swaps fc heads in-place on the model object.
**How to avoid:** This is actually fine because predict_with_uncertainty() restores the base state at the end (bootstrap.py line 132 loads base_state first, then iterates heads). But be aware: if the function crashes mid-way, the model is left in a corrupted state. Consider wrapping in try/finally. Also, st.cache_resource returns the SAME object to all sessions — concurrent requests could conflict. For a low-traffic research tool this is acceptable, but document the limitation.
**Warning signs:** Intermittent wrong predictions when multiple users access simultaneously.

### Pitfall 4: st.data_editor Returns NaN for Empty Rows
**What goes wrong:** User adds rows via num_rows="dynamic" but doesn't fill all cells. DataFrame contains NaN values that crash the encoder.
**Why it happens:** st.data_editor initializes new rows with None/NaN for numeric columns.
**How to avoid:** The validation function must drop rows where ANY required column is NaN before counting "at least 3 data points". Use `df.dropna()` before validation.
**Warning signs:** TypeError or ValueError in transform() when processing None values.

### Pitfall 5: Bootstrap Checkpoint Loading with weights_only
**What goes wrong:** `torch.load('bootstrap_heads.pth', weights_only=True)` raises UnpicklingError.
**Why it happens:** bootstrap_heads.pth contains a Python dict with a list of state_dicts — not a pure state_dict. PyTorch's safe loading rejects this.
**How to avoid:** Use `weights_only=False` for bootstrap_heads.pth specifically. This is a known Phase 3 decision (STATE.md).
**Warning signs:** App crashes on startup with pickle-related error.

### Pitfall 6: Dispersity Column Name Encoding
**What goes wrong:** Unicode character Đ (U+0110) in column name causes issues with some Excel parsers or CSV encodings.
**How to avoid:** Use ASCII-safe column name internally (e.g., 'D' or 'Dispersity') and display Đ only in the UI label via column_config. The Excel template should use 'D' as the column header.


## Code Examples

### Complete Validation Function
```python
def validate_input(df):
    """Validate user input DataFrame. Returns (clean_df, errors).
    
    clean_df: DataFrame with NaN rows dropped, or None if fatal errors.
    errors: list of error message strings (empty if valid).
    """
    errors = []
    
    # Drop completely empty rows
    df_clean = df.dropna(how='all')
    
    # Check minimum data points
    df_valid = df_clean.dropna()
    if len(df_valid) < 3:
        errors.append(f"At least 3 complete data points required (found {len(df_valid)})")
        return None, errors
    
    # Per-column validation
    for idx, row in df_valid.iterrows():
        row_num = idx + 1
        if not (0 < row['conversion'] < 1):
            errors.append(f"Row {row_num}: conversion must be in (0, 1), got {row['conversion']:.3f}")
        if row['Mn'] <= 0:
            errors.append(f"Row {row_num}: Mn must be > 0, got {row['Mn']:.1f}")
        if row['D'] < 1:
            errors.append(f"Row {row_num}: Dispersity must be >= 1.0, got {row['D']:.3f}")
        if row['[CTA]/[M]'] <= 0:
            errors.append(f"Row {row_num}: [CTA]/[M] must be > 0, got {row['[CTA]/[M]']:.6f}")
    
    return df_valid, errors
```

### ctFP Heatmap Visualization
```python
import matplotlib.pyplot as plt
import streamlit as st

def plot_ctfp_heatmap(ctfp_tensor):
    """Plot dual-channel ctFP as side-by-side heatmaps.
    
    ctfp_tensor: torch.Tensor shape (2, 64, 64)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    im1 = ax1.imshow(ctfp_tensor[0].numpy(), cmap='viridis', aspect='equal', origin='lower')
    ax1.set_title('Channel 0: Mn (normalized)')
    ax1.set_xlabel('[CTA]/[M]')
    ax1.set_ylabel('Conversion')
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    im2 = ax2.imshow(ctfp_tensor[1].numpy(), cmap='plasma', aspect='equal', origin='lower')
    ax2.set_title('Channel 1: Dispersity (Đ)')
    ax2.set_xlabel('[CTA]/[M]')
    ax2.set_ylabel('Conversion')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
```

### Result Card Display with st.metric and Custom HTML
```python
def display_results(mean_pred, half_width):
    """Display three prediction cards. D-06/D-07/D-08."""
    log10_ctr = mean_pred[0]
    ctr = 10 ** log10_ctr
    ctr_lower = 10 ** (log10_ctr - half_width[0])
    ctr_upper = 10 ** (log10_ctr + half_width[0])
    
    inh = mean_pred[1]
    ret = mean_pred[2]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style="text-align:center; padding:1em; border:2px solid #ff6b6b; border-radius:8px;">
            <h2 style="color:#ff6b6b;">C<sub>tr</sub> = {ctr:.1f}</h2>
            <p>log₁₀(C<sub>tr</sub>) = {log10_ctr:.2f}</p>
            <p>95% CI: [{ctr_lower:.1f}, {ctr_upper:.1f}]</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="text-align:center; padding:1em; border:1px solid #ddd; border-radius:8px;">
            <h3>Inhibition Period</h3>
            <h2>{inh:.3f}</h2>
            <p>95% CI: [{inh - half_width[1]:.3f}, {inh + half_width[1]:.3f}]</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="text-align:center; padding:1em; border:1px solid #ddd; border-radius:8px;">
            <h3>Retardation Factor</h3>
            <h2>{ret:.3f}</h2>
            <p>95% CI: [{ret - half_width[2]:.3f}, {ret + half_width[2]:.3f}]</p>
        </div>
        """, unsafe_allow_html=True)
```


## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| st.cache (deprecated) | st.cache_resource / st.cache_data | Streamlit 1.18 (Jan 2023) | Must use cache_resource for models, cache_data for data |
| st.experimental_data_editor | st.data_editor | Streamlit 1.23 (Jun 2023) | Stable API, no experimental prefix |
| Manual HTML tables | st.data_editor with column_config | Streamlit 1.24+ | Type-safe columns with NumberColumn, min/max validation |
| ViT-RR input-perturbation bootstrap | ViT-Ctr head-swapping bootstrap | Phase 3 design | predict_with_uncertainty() handles everything; app just calls it |

**Deprecated/outdated:**
- `st.cache`: Fully deprecated, use `st.cache_resource` or `st.cache_data`
- `st.experimental_data_editor`: Removed, use `st.data_editor`
- ViT-RR's `predict_model()` with input noise: ViT-Ctr uses `predict_with_uncertainty()` from bootstrap.py instead

## Open Questions

1. **M_monomer Input Strategy**
   - What we know: ctFP encoding requires Mn_theory = M_monomer / [CTA]/[M]. Training data uses fixed M_monomer per RAFT type (MMA=100.12, Styrene=104.15, VAc=86.09).
   - What's unclear: Should the user enter M_monomer manually, or select from a dropdown of common monomers? The CONTEXT.md specifics section flags this as a key design point.
   - Recommendation: Provide a dropdown of common monomers (MMA, Styrene, VAc, etc.) with their M_monomer values, plus a "Custom" option for manual entry. Place this above the data table as a required field. This matches the training data's monomer coverage and reduces user error.

2. **Concurrent Access Safety**
   - What we know: predict_with_uncertainty() mutates the model object (swaps fc heads). st.cache_resource returns the same object to all sessions.
   - What's unclear: Whether Streamlit serializes widget callbacks or allows true concurrent execution.
   - Recommendation: For a low-traffic research tool, this is acceptable. Document the limitation. If needed later, create a fresh model copy per prediction call (adds ~50ms overhead).

3. **Missing Checkpoint Files**
   - What we know: checkpoints/ currently has best_model.pth and training_log.json. bootstrap_heads.pth and calibration.json are not yet generated (Phase 3 bootstrap training pending).
   - What's unclear: Exact file format of calibration.json (list of 3 floats? dict with keys?).
   - Recommendation: The app should check for all 3 files on startup and show a clear error if any are missing. calibrate_coverage() returns a list of 3 floats, so calibration.json should be `[f1, f2, f3]`.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Streamlit | Web app | Yes | 1.51.0 | — |
| PyTorch | Model inference | Yes | 2.10.0 | — |
| NumPy | ctFP encoding | Yes | 2.3.5 | — |
| pandas | Data handling | Yes | 2.3.3 | — |
| matplotlib | Heatmap | Yes | 3.10.6 | — |
| openpyxl | Excel I/O | Yes | 3.1.5 | — |
| SciPy | Bootstrap JCI | Yes | 1.16.3 | — |
| best_model.pth | Model weights | Yes | — | — |
| bootstrap_heads.pth | Bootstrap CI | No (pending Phase 3) | — | Skip CI, show point estimate only |
| calibration.json | CI calibration | No (pending Phase 3) | — | Use uncalibrated CI (cal_factors=[1,1,1]) |

**Missing dependencies with fallback:**
- bootstrap_heads.pth and calibration.json: App should work in "point estimate only" mode when these are absent, with a warning banner. Full CI requires completing Phase 3 bootstrap training.


## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.4.2 |
| Config file | pyproject.toml [tool.pytest.ini_options] |
| Quick run command | `python -m pytest tests/ -x -q` |
| Full suite command | `python -m pytest tests/ -v` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| APP-01 | data_editor accepts manual input | unit (validate + normalize) | `python -m pytest tests/test_app.py::test_prepare_ctfp_input -x` | Wave 0 |
| APP-02 | Excel/CSV upload + template download | unit (template gen + parse) | `python -m pytest tests/test_app.py::test_template_generation -x` | Wave 0 |
| APP-03 | Three predictions with CI displayed | unit (back-transform logic) | `python -m pytest tests/test_app.py::test_ctr_back_transform -x` | Wave 0 |
| APP-04 | Input validation rules | unit (validation function) | `python -m pytest tests/test_app.py::test_validate_input -x` | Wave 0 |
| APP-05 | ctFP heatmap renders | manual-only | Manual: run app, verify heatmap appears | N/A |
| APP-06 | Model cached via st.cache_resource | manual-only | Manual: check no reload on rerun | N/A |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/test_app.py -x -q`
- **Per wave merge:** `python -m pytest tests/ -v`
- **Phase gate:** Full suite green + manual Streamlit smoke test

### Wave 0 Gaps
- [ ] `tests/test_app.py` — covers APP-01 through APP-04 (validation, normalization, template, back-transform)
- [ ] No Streamlit-specific test framework needed — test the pure Python functions (validate_input, prepare_ctfp_input, generate_template, back-transform logic) without Streamlit runtime

## Sources

### Primary (HIGH confidence)
- `src/ctfp_encoder.py` — transform() API, normalization expectations (direct inspection)
- `src/bootstrap.py` — predict_with_uncertainty() API, return format (direct inspection)
- `src/model.py` — SimpViT constructor, num_outputs=3 (direct inspection)
- `src/dataset_generator.py` lines 244-256 — Mn normalization and cta_ratio normalization (direct inspection)
- `src/literature_validation.py` lines 170-194 — ml_predict_single() as inference reference (direct inspection)
- `C:/CodingCraft/DL/ViT-RR/deploy.py` — ViT-RR Streamlit reference implementation (direct inspection)
- Streamlit st.data_editor docs — https://docs.streamlit.io/develop/api-reference/data/st.data_editor
- Streamlit st.column_config.NumberColumn — https://docs.streamlit.io/1.25.0/develop/api-reference/data/st.column_config/st.column_config.numbercolumn
- Streamlit st.cache_resource docs — https://docs.streamlit.io/1.25.0/develop/api-reference/caching-and-state/st.cache_resource

### Secondary (MEDIUM confidence)
- Streamlit community: download_button with Excel — https://discuss.streamlit.io/t/download-button-for-csv-or-xlsx-file/17385/2
- Streamlit community: data_editor num_rows dynamic — https://ryanandmattdatascience.com/streamlit-data-editior/

### Tertiary (LOW confidence)
- None — all findings verified against source code or official docs

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all packages installed and version-verified via pip show
- Architecture: HIGH — patterns derived from existing codebase (dataset_generator.py, literature_validation.py, ViT-RR deploy.py)
- Pitfalls: HIGH — Mn normalization issue identified from direct code inspection of training pipeline vs. inference requirements
- Validation: HIGH — test infrastructure exists (pytest + conftest.py), test patterns established in prior phases

**Research date:** 2026-04-03
**Valid until:** 2026-05-03 (stable — Streamlit API is mature, project dependencies are pinned)

