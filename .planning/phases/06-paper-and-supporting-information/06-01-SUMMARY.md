# Phase 06 Plan 01 Summary — Publication Figures

**Status:** Complete (Task 1 auto, Task 2 pending human visual review)
**Duration:** ~5min
**Commits:** 1

## What Was Done

Created `scripts/generate_figures.py` (385 lines) that generates four publication figures using matplotlib:

1. **Figure 1** (`fig1_concept.png`, 84KB) — Concept/workflow diagram: Experimental Data → ctFP Encoding → SimpViT → 3 parameters ± 95% CI
2. **Figure 2** (`fig2_ctfp_example.png`, 148KB) — Dual-channel ctFP heatmap from ODE-simulated dithioester data (Ctr ≈ 1000), showing Ch0 (Mn/Mn_theory) and Ch1 (Đ)
3. **Figure 3** (`fig3_parity_composite.png`, 1.6MB) — 1×3 composite of existing parity plots with (a)(b)(c) panel labels
4. **TOC graphic** (`toc_graphic.png`, 17KB) — Compact ~3.25×1.75 in graphic: ctFP grid → SimpViT → three outputs with CI error bars

## Key Details

- Figure 2 uses live ODE simulation via `src.raft_ode.simulate_raft` + `src.ctfp_encoder.transform` — not synthetic data
- Figure 3 composites existing `figures/parity_*.png` files via `plt.imread`
- All figures: 300 DPI, white background, `bbox_inches='tight'`
- All acceptance criteria passed (function names, file sizes, imports)

## Files Changed

| File | Action |
|------|--------|
| `scripts/generate_figures.py` | Created (385 lines) |
| `figures/fig1_concept.png` | Created |
| `figures/fig2_ctfp_example.png` | Created |
| `figures/fig3_parity_composite.png` | Created |
| `figures/toc_graphic.png` | Created |

## Pending

Task 2 is a human-verify checkpoint — user needs to visually review all four figures before this plan is fully complete.
