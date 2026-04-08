---
status: awaiting_human_verify
trigger: "AutoDL bootstrap workflow too tedious — need one-click scripting"
created: 2026-04-05T00:00:00Z
updated: 2026-04-05T00:00:00Z
---

## Current Focus

hypothesis: The problem is not code bugs but workflow friction — too many manual steps for upload/execute/download
test: Create consolidated scripts that reduce the workflow to 2-3 commands
expecting: User can sync code, run bootstrap, and download results with minimal manual steps
next_action: User tests the new workflow on actual AutoDL instance

## Symptoms

expected: One command or one-click to complete the full flow: upload code to AutoDL -> run bootstrap training -> download artifacts (bootstrap_heads.pth, calibration.json) back to local
actual: Requires many manual steps: JupyterLab web UI operations, web file upload/download, manual notebook cell execution, manual progress monitoring
errors: No runtime errors — this is a workflow engineering problem
reproduction: Every bootstrap run or re-run requires repeating these tedious steps
started: Since Phase 3 began

## Analysis

### Root Cause
The current workflow has 5 friction points:
1. **Upload**: User manually uploads code changes via AutoDL web file manager
2. **Execution**: User manually runs cells in JupyterLab or types commands in terminal
3. **Monitoring**: No structured progress reporting beyond print statements
4. **Download**: User manually downloads results via AutoDL web file manager
5. **Fragmentation**: Notebook and script are separate, with slightly different code paths

### Solution Plan
Create a production-grade 3-script workflow:

1. **`scripts/autodl-sync.bat`** (Windows) — SCP-based code sync to AutoDL
   - Uses SSH key or password (AutoDL provides SSH info)
   - Syncs only necessary files (src/, colab/autodl_bootstrap.py)
   - Idempotent — can re-run safely

2. **Enhanced `colab/autodl_bootstrap.py`** — Single definitive execution script
   - Already has bootstrap + calibration + resume
   - ADD: verification step (quick inference test after calibration)
   - ADD: results summary with clear success/fail indication
   - ADD: tar.gz packaging of all outputs for easy download
   - ADD: ETA estimation and progress percentage
   - ADD: `--run_all` default mode that does everything

3. **`scripts/autodl-download.bat`** (Windows) — SCP-based results download
   - Downloads the packaged tar.gz or individual files
   - Validates downloaded files exist and are non-empty

## Evidence

- timestamp: 2026-04-05
  checked: autodl_bootstrap.py current capabilities
  found: Already has resume support, preload optimization, batch_size=256, calibration. Missing: verification, packaging, ETA, summary.
  implication: Enhancement, not rewrite

- timestamp: 2026-04-05
  checked: AutoDL SSH access
  found: AutoDL provides SSH connection info (host, port, password). SCP is available.
  implication: Can script file transfers with SCP

- timestamp: 2026-04-05
  checked: User environment
  found: Windows (C:\CodingCraft\DL\ViT-Ctr), using JupyterLab web UI currently
  implication: Need .bat scripts for Windows; user may not have ssh/scp natively — need to handle this

## Resolution

root_cause: Workflow friction from too many manual steps, not code bugs
fix: |
  Created a 5-script automation toolkit + enhanced bootstrap script:

  1. `colab/autodl_bootstrap.py` — Enhanced to be a full pipeline:
     - Default paths (no --h5_dir/--ckpt_dir required)
     - Step-by-step progress with banners
     - ETA estimation per bootstrap iteration
     - Verification step (test inference after calibration)
     - tar.gz packaging of all outputs
     - Summary JSON report
     - Graceful error/interrupt handling

  2. `scripts/autodl-config.bat` — Shared SSH config (edit once, used by all scripts)
  3. `scripts/autodl-sync.bat` — SCP-based code sync to AutoDL
  4. `scripts/autodl-run.bat` — Remote tmux-based bootstrap launch (full/resume/calibrate)
  5. `scripts/autodl-status.bat` — Remote progress query
  6. `scripts/autodl-download.bat` — SCP-based results download with validation

  New workflow (3 commands):
    1. scripts\autodl-sync.bat       (sync code)
    2. scripts\autodl-run.bat        (start training)
    3. scripts\autodl-download.bat   (get results)

verification: User tests on actual AutoDL instance
files_changed:
  - colab/autodl_bootstrap.py (rewritten — full pipeline with ETA, verification, packaging)
  - scripts/autodl-config.bat (new — shared SSH config, edit HOST/PORT here)
  - scripts/autodl-sync.bat (new — sync code to AutoDL via SCP)
  - scripts/autodl-run.bat (new — start bootstrap in remote tmux)
  - scripts/autodl-status.bat (new — query progress remotely)
  - scripts/autodl-download.bat (new — download results via SCP)
