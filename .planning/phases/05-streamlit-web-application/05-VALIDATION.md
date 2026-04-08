---
phase: 5
slug: streamlit-web-application
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-03
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | none — Wave 0 installs if needed |
| **Quick run command** | `python -m pytest tests/test_app.py -x -q` |
| **Full suite command** | `python -m pytest tests/ -q` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_app.py -x -q`
- **After every plan wave:** Run `python -m pytest tests/ -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 05-01-01 | 01 | 1 | APP-01 | unit | `pytest tests/test_app.py::test_manual_input` | ❌ W0 | ⬜ pending |
| 05-01-02 | 01 | 1 | APP-02 | unit | `pytest tests/test_app.py::test_file_upload` | ❌ W0 | ⬜ pending |
| 05-01-03 | 01 | 1 | APP-03 | unit | `pytest tests/test_app.py::test_input_validation` | ❌ W0 | ⬜ pending |
| 05-01-04 | 01 | 1 | APP-04 | unit | `pytest tests/test_app.py::test_ctfp_heatmap` | ❌ W0 | ⬜ pending |
| 05-01-05 | 01 | 1 | APP-05 | unit | `pytest tests/test_app.py::test_prediction_display` | ❌ W0 | ⬜ pending |
| 05-01-06 | 01 | 1 | APP-06 | unit | `pytest tests/test_app.py::test_model_caching` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_app.py` — stubs for APP-01 through APP-06
- [ ] `tests/conftest.py` — shared fixtures (mock model, sample data)

*Existing test infrastructure from Phase 3 covers pytest setup.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Visual layout of 3 prediction cards | APP-05 | CSS/visual rendering | Open app, submit sample data, verify 3 cards display side-by-side with Ctr visually prominent |
| ctFP heatmap visual correctness | APP-04 | Image rendering quality | Open app, submit data, verify dual-channel heatmap renders with correct colorbars |
| Excel template download | APP-02 | Browser download behavior | Click download button, verify .xlsx file downloads with correct column headers |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending