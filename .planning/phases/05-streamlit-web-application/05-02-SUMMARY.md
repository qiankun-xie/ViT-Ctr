---
phase: 05-streamlit-web-application
plan: 02
status: complete
completed: 2026-04-06
---

# Plan 05-02 Summary

## What was done

Built and verified the complete Streamlit web application (`app.py`, 378 lines) wiring Plan 01 utilities to the model inference pipeline.

## Key deliverables

- **app.py**: Single-page Streamlit app with all 6 APP requirements fulfilled
  - Manual data entry via `st.data_editor` (APP-01)
  - Excel/CSV upload + template download (APP-02)
  - Three prediction cards with 95% CI display (APP-03)
  - Input validation with error banners (APP-04)
  - Dual-channel ctFP heatmap visualization (APP-05)
  - Model cached via `@st.cache_resource` (APP-06)

## Bug fixed during verification

- `app.py:61`: `cal_factors = json.load(f)` → `json.load(f)["cal_factors"]`
  - Root cause: `calibration.json` contains a dict with multiple keys; code passed the full dict instead of extracting the `cal_factors` list
  - Effect: `predict_with_uncertainty()` crashed on `np.array(cal_factors)` with a dict input

## Human verification results (10/10 passed)

1. Page title: ViT-Ctr: RAFT Chain Transfer Constant Predictor ✓
2. Monomer selector: MMA, Styrene, Vinyl Acetate, Custom ✓
3. Manual Input tab: editable 4-column table ✓
4. File Upload tab: download button + file uploader ✓
5. Excel template download: correct .xlsx ✓
6. Normal prediction: 3 result cards + ctFP heatmap ✓
7. Ctr card red border, others gray ✓
8. 95% CI displayed on all cards ✓
9. Dual-channel ctFP heatmap (viridis + plasma) ✓
10. Invalid input error banner ✓

## Files modified

| File | Change |
|------|--------|
| app.py | Fix: `json.load(f)` → `json.load(f)["cal_factors"]` (line 61) |
