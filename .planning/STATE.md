---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: Ready to plan
stopped_at: Phase 5 context gathered
last_updated: "2026-04-03T02:51:00.096Z"
progress:
  total_phases: 6
  completed_phases: 4
  total_plans: 12
  completed_plans: 12
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-24)

**Core value:** 一次输入实验数据，同时提取Ctr、诱导期和减速因子三个参数——传统方法需要三组独立实验才能分别获得。
**Current focus:** Phase 03 — model-training-and-evaluation

## Current Position

Phase: 4
Plan: Not started

## Performance Metrics

**Velocity:**

- Total plans completed: 5
- Average duration: ~10min
- Total execution time: ~51min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 | 3 | ~28min | ~9min |
| 02 | 2 | ~23min | ~12min |

**Recent Trend:**

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 03 P03 | ~20min | 2 tasks | 9 files |
| Phase 03 P02 | ~8min | 2 tasks | 7 files |
| Phase 02 P02 | ~8min | 2 tasks | 3 files |
| Phase 02 P01 | ~15min | 2 tasks | 1 file |
| Phase 01 P03 | ~15min | 2 tasks | 3 files |
| Phase 01 P02 | 3min | 1 tasks | 5 files |
| Phase 01 P01 | 10min | 2 tasks | 6 files |
| Phase 03 P04 | 1 | 2 tasks | 3 files |
| Phase 03 P05 | 6 | 2 tasks | 3 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Init: ODE validation must precede all data generation — corruption at ODE level requires full dataset regeneration (research finding)
- Init: ctFP encoder extracted as shared module from day one — encoding divergence between training and inference silently invalidates predictions
- Init: Large-scale training targets Google Colab T4; local MX350 (2 GB) is insufficient for 1M sample training run
- [Phase 01]: Dispersity clipped at 4.0 in ctFP encoder to prevent outlier domination
- [Phase 01]: Dormant chain moments (nu) tracked separately from dead chains (lam) -- RAFT exchange is moment swap, not chain death
- [Phase 01]: State vector: 14 vars (single-eq) / 16 vars (pre-eq) with mu/nu/lam moment populations
- [Phase 01]: joblib prefer=threads for Windows diagnostic parallelism
- [Phase 02]: LHS采样7维连续参数空间，对数均匀采样覆盖多数量级范围
- [Phase 02]: 噪声sigma=0.03（3%相对误差），在D-05规定的0.02-0.05范围内
- [Phase 02]: HDF5存储(n,64,64,2)格式，chunk=1000，按RAFT类型分文件
- [Phase 02]: joblib prefer='processes' 用于大规模并行生成（避免GIL瓶颈）
- [Phase 02]: No automated Google Drive upload — manual upload with step-by-step instructions; OAuth overhead not justified
- [Phase 02]: Dataset info auto-generated from HDF5 files; upload_to_gdrive.py regenerates 02-DATASET-INFO.md
- [Phase 03]: Wave 0 gate in Colab notebook — raises RuntimeError if HDF5 files missing or < 100K samples
- [Phase 03]: Lightweight bootstrap: freeze backbone (fc layer only), 200 × 5-epoch head fine-tuning; NOT input-perturbation bootstrap
- [Phase 03]: F-distribution JCI uses p=3 (changed from ViT-RR p=2); f.ppf(0.95, dfn=3, dfd=197) ≈ 2.65
- [Phase 03]: Retardation factor ≈ 1.0 for TTC/xanthate/dithiocarbamate is EXPECTED, not a model failure
- [Phase 03]: weights_only=False required for torch.load of bootstrap_heads.pth (contains list of dicts)
- [Phase 03 P02]: CombinedHDF5Dataset uses self._handles=None + _get_handles() lazy pattern — critical for fork safety with num_workers>0
- [Phase 03 P02]: build_stratified_indices returns (file_idx, sample_idx, class_id) tuples; class_id == file_idx (RAFT type derived from file position)
- [Phase 03 P02]: evaluate.py compute_outlier_stats included now (Plan 04 dependency) but not tested in Wave 0
- [Phase 03 P03]: weights_only=True for load_checkpoint of base model (safe); weights_only=False still needed for bootstrap_heads.pth (Plan 05)
- [Phase 03 P03]: num_workers=0 default on Windows; Colab passes --num_workers 2 via CLI
- [Phase 03 P03]: JSON training_log.json appended and rewritten per epoch (simple, not perf-critical)
- [Phase 03 P01 fix]: SimpViT actual param count is 877,571 — planning doc's ~3.4M was a planning error (hidden_size=64 not 256); dim_feedforward defaults to 2048 in nn.TransformerEncoderLayer; test range corrected to [800K, 950K]
- [Phase 03]: Matplotlib lazy import in evaluate.py plotting functions for headless environment compatibility
- [Phase 03]: OUTPUT_NAMES constant ['log10_Ctr', 'inhibition_period', 'retardation_factor'] for consistent labeling across evaluation outputs
- [Phase 03]: Bootstrap uses training-time head fine-tuning (not inference-time input perturbation)
- [Phase 03]: F-distribution JCI with p=3, exact port from ViT-RR deploy.py

### Pending Todos

- Run actual large-scale generation: `python src/dataset_generator.py` (~hours of CPU time)
- Upload generated HDF5 files to Google Drive and fill in file IDs in 02-DATASET-INFO.md
- Execute Phase 3 plans in order: Wave 0 (plans 01+02 parallel) → Wave 1 (plan 03) → Wave 2 (plan 04) → Wave 3 (plan 05)
- Run colab/03-train-colab.ipynb on Colab T4 after full HDF5 dataset is on Drive
- Run colab/03-bootstrap-colab.ipynb after training completes
- Download best_model.pth, bootstrap_heads.pth, calibration.json to local checkpoints/

### Blockers/Concerns

- Phase 1 (resolved): RAFT two-stage pre-equilibrium ODE moment equations verified through literature validation
- Phase 1 (noted): Retardation factor ~1.0 for all tested systems — confirmed as expected behavior, documented in evaluate.py
- Phase 3 (pending): Full ~1M sample dataset not yet on Google Drive — Phase 3 local tests only work in debug mode (38 samples)
- Phase 4: Preliminary literature survey for 10+ Ctr validation points should begin early to confirm sufficient published data exists before training completes

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260403-gmb | 创建 colab/03-bootstrap-autodl.ipynb — AutoDL版bootstrap notebook | 2026-04-03 | d3e1f3b | [260403-gmb-colab-03-bootstrap-autodl-ipynb-autodl-b](./quick/260403-gmb-colab-03-bootstrap-autodl-ipynb-autodl-b/) |

## Session Continuity

Last activity: 2026-04-03 - Completed quick task 260403-gmb: 创建 colab/03-bootstrap-autodl.ipynb — AutoDL版bootstrap notebook
Last session: 2026-04-03T03:57:59.474Z
Stopped at: Quick task completed, Phase 5 ready for planning
Resume file: .planning/phases/05-streamlit-web-application/05-CONTEXT.md
