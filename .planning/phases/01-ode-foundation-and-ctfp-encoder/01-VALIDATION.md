---
phase: 1
slug: ode-foundation-and-ctfp-encoder
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-25
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.4.2 |
| **Config file** | none — Wave 0 installs |
| **Quick run command** | `pytest tests/ -x -q` |
| **Full suite command** | `pytest tests/ -v` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/ -x -q`
- **After every plan wave:** Run `pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 01-01-01 | 01 | 1 | SIM-01 | unit | `pytest tests/test_raft_ode.py::test_forward_simulation -x` | ❌ W0 | ⬜ pending |
| 01-01-02 | 01 | 1 | SIM-02 | integration | `pytest tests/test_raft_ode.py::test_preequilibrium_distinct -x` | ❌ W0 | ⬜ pending |
| 01-01-03 | 01 | 1 | SIM-02 | unit | `pytest tests/test_raft_ode.py::test_all_agent_types -x` | ❌ W0 | ⬜ pending |
| 01-01-04 | 01 | 1 | SIM-03 | integration | `pytest tests/test_raft_ode.py::test_parameter_range_coverage -x` | ❌ W0 | ⬜ pending |
| 01-01-05 | 01 | 1 | SIM-03 | integration | `pytest tests/test_raft_ode.py::test_cta_ratio_range -x` | ❌ W0 | ⬜ pending |
| 01-02-01 | 02 | 1 | ENC-01 | unit | `pytest tests/test_ctfp_encoder.py::test_output_shape -x` | ❌ W0 | ⬜ pending |
| 01-02-02 | 02 | 1 | ENC-01 | unit | `pytest tests/test_ctfp_encoder.py::test_channel_assignment -x` | ❌ W0 | ⬜ pending |
| 01-02-03 | 02 | 1 | ENC-02 | unit | `pytest tests/test_ctfp_encoder.py::test_deterministic_output -x` | ❌ W0 | ⬜ pending |
| 01-03-01 | 03 | 2 | -- | integration | `pytest tests/test_raft_ode.py::test_limit_behavior -x` | ❌ W0 | ⬜ pending |
| 01-03-02 | 03 | 2 | -- | smoke | `pytest tests/test_raft_ode.py::test_diagnostic_dataset -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/conftest.py` — shared fixtures (parameter sets, expected outputs)
- [ ] `tests/test_raft_ode.py` — ODE tests covering SIM-01, SIM-02, SIM-03
- [ ] `tests/test_ctfp_encoder.py` — encoder tests covering ENC-01, ENC-02
- [ ] `pyproject.toml` with [tool.pytest.ini_options] — test configuration
- [ ] `src/` directory creation

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Literature curve comparison (Mn vs conversion) | D-11 | Requires visual inspection of ODE output against published plots | Run 01_ode_validation.ipynb, compare against Moad 2000/Arita 2005 plots |
| ctFP visual inspection | ENC-01 | Image quality is visual | Plot ctFP channels with matplotlib.imshow(), verify pixel activation patterns |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
