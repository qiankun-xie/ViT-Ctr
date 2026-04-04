"""
app.py - ViT-Ctr: RAFT Chain Transfer Constant Predictor

Single-page Streamlit web application. UI layer only -- all inference logic
lives in src/ modules. Normalization bridge provided by src/app_utils.py.

Usage: streamlit run app.py
"""
import json
import os

# Suppress OpenMP duplicate-library warning (common on Windows with conda PyTorch)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.app_utils import (
    format_results,
    generate_template,
    prepare_ctfp_input,
    validate_input,
)
from src.bootstrap import predict_with_uncertainty
from src.ctfp_encoder import transform
from src.model import SimpViT

# -- Page configuration (must be first Streamlit call) -------------------------
st.set_page_config(
    page_title="ViT-Ctr: RAFT Chain Transfer Constant Predictor",
    layout="wide",
)

# -- Checkpoint paths -----------------------------------------------------------
_CKPT_DIR = "checkpoints"
_MODEL_PATH = os.path.join(_CKPT_DIR, "best_model.pth")
_BOOTSTRAP_PATH = os.path.join(_CKPT_DIR, "bootstrap_heads.pth")
_CAL_PATH = os.path.join(_CKPT_DIR, "calibration.json")


# -- Model loading (cached for lifetime of Streamlit process) -------------------
@st.cache_resource
def load_model():
    """Load SimpViT + optional bootstrap heads + calibration factors."""
    model = SimpViT(num_outputs=3)
    # weights_only=False: checkpoint is a full training dict {epoch, model_state_dict, ...}
    ckpt = torch.load(_MODEL_PATH, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)  # handles both formats
    model.load_state_dict(state)
    model.eval()

    bootstrap_ckpt = None
    cal_factors = None
    if os.path.exists(_BOOTSTRAP_PATH) and os.path.exists(_CAL_PATH):
        bootstrap_ckpt = torch.load(
            _BOOTSTRAP_PATH, map_location="cpu", weights_only=False
        )
        with open(_CAL_PATH) as f:
            cal_factors = json.load(f)

    return model, bootstrap_ckpt, cal_factors


# -- Checkpoint detection -------------------------------------------------------
_model_ok = os.path.exists(_MODEL_PATH)

if not _model_ok:
    st.error(
        "Model checkpoint files not found. Ensure best_model.pth, "
        "bootstrap_heads.pth, and calibration.json are in the "
        "checkpoints/ directory."
    )
else:
    _model, _bootstrap_ckpt, _cal_factors = load_model()
    _has_bootstrap = _bootstrap_ckpt is not None
    if not _has_bootstrap:
        st.warning(
            "Bootstrap checkpoint not found. Showing point estimates only "
            "(no confidence intervals)."
        )


# -- Title ----------------------------------------------------------------------
st.title("ViT-Ctr: RAFT Chain Transfer Constant Predictor")
st.caption(
    "Simultaneously predict Ctr, inhibition period, and retardation factor "
    "from experimental kinetic data."
)


# -- Monomer selector -----------------------------------------------------------
_MONOMER_OPTIONS = {
    "MMA (100.12 g/mol)": 100.12,
    "Styrene (104.15 g/mol)": 104.15,
    "Vinyl Acetate (86.09 g/mol)": 86.09,
    "Custom": None,
}

_monomer_choice = st.selectbox(
    "Monomer molecular weight (g/mol)",
    options=list(_MONOMER_OPTIONS.keys()),
    index=0,
)

if _monomer_choice == "Custom":
    m_monomer = st.number_input(
        "Enter molecular weight (g/mol)",
        min_value=1.0,
        value=100.0,
        step=0.01,
        format="%.2f",
    )
else:
    m_monomer = _MONOMER_OPTIONS[_monomer_choice]


# -- Data editor column configuration ------------------------------------------
_COL_CONFIG = {
    "[CTA]/[M]": st.column_config.NumberColumn(
        "[CTA]/[M]",
        help="Molar ratio of CTA to monomer (e.g., 0.005)",
        min_value=0.0,
        format="%.4f",
    ),
    "conversion": st.column_config.NumberColumn(
        "Conversion",
        help="Monomer conversion (0 to 1)",
        min_value=0.0,
        max_value=1.0,
        format="%.3f",
    ),
    "Mn": st.column_config.NumberColumn(
        "Mn (g/mol)",
        help="Number-average molecular weight in g/mol",
        min_value=0.0,
        format="%.0f",
    ),
    "D": st.column_config.NumberColumn(
        "\u0110 (Mw/Mn)",
        help="Dispersity (Mw/Mn), must be >= 1.0",
        min_value=1.0,
        format="%.3f",
    ),
}

_DEFAULT_DF = pd.DataFrame(
    {
        "[CTA]/[M]": pd.Series(dtype="float64"),
        "conversion": pd.Series(dtype="float64"),
        "Mn": pd.Series(dtype="float64"),
        "D": pd.Series(dtype="float64"),
    }
)


# -- Input tabs -----------------------------------------------------------------
tab1, tab2 = st.tabs(["Manual Input", "File Upload"])

with tab1:
    manual_df = st.data_editor(
        _DEFAULT_DF,
        num_rows="dynamic",
        hide_index=True,
        column_config=_COL_CONFIG,
        key="data_input",
    )

with tab2:
    col_dl, col_up = st.columns([1, 2])
    with col_dl:
        st.download_button(
            label="Download Excel Template",
            data=generate_template(),
            file_name="ViT-Ctr_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with col_up:
        uploaded_file = st.file_uploader(
            "Upload Excel or CSV file",
            type=["xlsx", "csv"],
            key="file_upload",
        )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                _upload_df = pd.read_csv(uploaded_file)
            else:
                _upload_df = pd.read_excel(uploaded_file, engine="openpyxl")
            st.session_state["uploaded_df"] = _upload_df
            st.success(f"Loaded {len(_upload_df)} rows from {uploaded_file.name}")
            st.data_editor(
                _upload_df,
                num_rows="dynamic",
                hide_index=True,
                column_config=_COL_CONFIG,
                key="upload_review",
            )
        except Exception as e:
            st.error(f"Failed to read file: {e}")


# -- Active input DataFrame -----------------------------------------------------
# Use uploaded data if a file was loaded in Tab 2; otherwise use manual input.
_active_df: pd.DataFrame = (
    st.session_state["uploaded_df"]
    if (
        "uploaded_df" in st.session_state
        and st.session_state["uploaded_df"] is not None
    )
    else manual_df
)


# -- Predict button -------------------------------------------------------------
if _model_ok:
    _predict_clicked = st.button("Predict", type="primary", use_container_width=True)

    if _predict_clicked:
        df_valid, errors = validate_input(_active_df)

        if errors:
            st.error("\n\n".join(errors))
        elif df_valid is not None:
            data_tuples = prepare_ctfp_input(df_valid, m_monomer)
            fp = transform(data_tuples)
            fp_tensor = fp.unsqueeze(0)

            with st.spinner("Running prediction with 200 bootstrap samples..."):
                if _has_bootstrap:
                    mean_pred, half_width = predict_with_uncertainty(
                        _model, fp_tensor, _bootstrap_ckpt, _cal_factors, "cpu"
                    )
                else:
                    _model.eval()
                    with torch.no_grad():
                        mean_pred = _model(fp_tensor).cpu().numpy().squeeze()
                    half_width = None

            results = format_results(mean_pred, half_width)
            st.session_state["results"] = results
            st.session_state["fp"] = fp


# -- Results display ------------------------------------------------------------
if "results" in st.session_state:
    r = st.session_state["results"]
    fp_stored = st.session_state["fp"]

    col1, col2, col3 = st.columns(3)

    # -- Ctr card (accent border #ff4b4b, value 28px/700) ----------------------
    _ctr_ci_html = (
        f'<p style="font-size:14px;font-weight:400;color:#808495;margin:2px 0 0 0;">'
        f'95% CI: [{r["ctr_lower"]:.2f}, {r["ctr_upper"]:.2f}]</p>'
        if r["ctr_lower"] is not None
        else '<p style="font-size:14px;color:#808495;margin:2px 0 0 0;">Point estimate (no CI)</p>'
    )
    with col1:
        st.markdown(
            f"""
            <div style="text-align:center;padding:16px;border:2px solid #ff4b4b;
                        border-radius:8px;background:#ffffff;">
              <p style="font-size:14px;font-weight:400;color:#808495;margin-bottom:4px;">
                C<sub>tr</sub>
              </p>
              <p style="font-size:28px;font-weight:700;color:#262730;line-height:1.2;margin:0;">
                {r["ctr"]:.2f}
              </p>
              <p style="font-size:14px;font-weight:400;color:#808495;margin:4px 0 0 0;">
                log&#x2081;&#x2080;(C<sub>tr</sub>) = {r["log10_ctr"]:.3f}
              </p>
              {_ctr_ci_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -- Inhibition period card (muted border) ----------------------------------
    _inh_ci = (
        f'95% CI: [{r["inh_lower"]:.4f}, {r["inh_upper"]:.4f}]'
        if r["inh_lower"] is not None
        else "Point estimate (no CI)"
    )
    with col2:
        st.markdown(
            f"""
            <div style="text-align:center;padding:16px;border:1px solid #e0e0e0;
                        border-radius:8px;background:#ffffff;">
              <p style="font-size:14px;font-weight:400;color:#808495;margin-bottom:4px;">
                Inhibition Period
              </p>
              <p style="font-size:24px;font-weight:600;color:#262730;line-height:1.2;margin:0;">
                {r["inhibition_period"]:.4f}
              </p>
              <p style="font-size:14px;font-weight:400;color:#808495;margin:4px 0 0 0;">
                {_inh_ci}
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -- Retardation factor card (muted border) ---------------------------------
    _ret_ci = (
        f'95% CI: [{r["ret_lower"]:.4f}, {r["ret_upper"]:.4f}]'
        if r["ret_lower"] is not None
        else "Point estimate (no CI)"
    )
    with col3:
        st.markdown(
            f"""
            <div style="text-align:center;padding:16px;border:1px solid #e0e0e0;
                        border-radius:8px;background:#ffffff;">
              <p style="font-size:14px;font-weight:400;color:#808495;margin-bottom:4px;">
                Retardation Factor
              </p>
              <p style="font-size:24px;font-weight:600;color:#262730;line-height:1.2;margin:0;">
                {r["retardation_factor"]:.4f}
              </p>
              <p style="font-size:14px;font-weight:400;color:#808495;margin:4px 0 0 0;">
                {_ret_ci}
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -- ctFP heatmap ----------------------------------------------------------
    st.subheader("Model Input Visualization (ctFP)")

    import matplotlib.pyplot as plt  # lazy import -- not needed until results shown

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    im1 = ax1.imshow(
        fp_stored[0].numpy(), cmap="viridis", aspect="equal", origin="lower"
    )
    ax1.set_title("Channel 0: Mn (normalized)")
    ax1.set_xlabel("[CTA]/[M]")
    ax1.set_ylabel("Conversion")
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    im2 = ax2.imshow(
        fp_stored[1].numpy(), cmap="plasma", aspect="equal", origin="lower"
    )
    ax2.set_title("Channel 1: \u0110 (dispersity)")
    ax2.set_xlabel("[CTA]/[M]")
    ax2.set_ylabel("Conversion")
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

else:
    # Empty state -- shown before first prediction
    st.markdown(
        """
        <div style="text-align:center;padding:2em;color:#808495;">
          <p style="font-size:16px;font-weight:400;">Enter your experimental data above</p>
          <p style="font-size:14px;">Add at least 3 data points with [CTA]/[M], conversion,
          Mn, and \u0110 values, then click Predict.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -- Citation -------------------------------------------------------------------
st.divider()
st.caption(
    "If you use this tool, please cite: "
    "[citation placeholder \u2014 fill after Phase 6]"
)
