# Plan 03-03 Summary: Training Loop + Notebooks

**Plan:** 03-03
**Phase:** 03-model-training-and-evaluation
**Completed:** 2026-03-27
**Status:** DONE

## What Was Built

### Task 1: Training Infrastructure (src/train.py + tests)

**Files created:**
- `src/train.py` — full training loop with all locked hyperparameters
- `tests/test_train.py` — 5 tests (4 unit + 1 slow integration)

**Also created (dependencies from plans 03-01/03-02, needed locally):**
- `src/model.py` — SimpViT with num_outputs=3 (verbatim from ViT-RR)
- `src/dataset.py` — CombinedHDF5Dataset with lazy file handles
- `src/utils/__init__.py` — package marker
- `src/utils/split.py` — build_stratified_indices + RAFT_TYPES
- `src/utils/metrics.py` — r2_score_np, rmse_np, mae_np (no sklearn)

**Key functions in train.py:**
- `weighted_mse_loss(pred, target, weights=(2.0, 0.5, 0.5))` — device-safe weight tensor created inside function
- `EarlyStopper(patience=15)` — monitors val_loss, saves best_state via deepcopy
- `save_checkpoint / load_checkpoint` — .pth round-trip with weights_only=True
- `train_one_epoch` — NaN guard, per-output MSE tracking (ctr/inh/ret)
- `validate` — model.eval() + torch.no_grad()
- `main()` — argparse CLI: --h5_dir, --epochs, --batch_size, --lr, --checkpoint_dir, --seed, --num_workers, --debug

**Locked hyperparameters enforced:**
- D-02: batch_size=64
- D-03: lr=3e-4, ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
- D-04: EarlyStopper(patience=15)
- D-06/D-08: weights=[2.0, 0.5, 0.5]

**JSON logging:** `training_log.json` saved per-epoch with keys: epoch, train_loss, val_loss, loss_ctr, loss_inh, loss_ret, lr.

**Checkpoints:** best_model.pth on val_loss improvement + epoch_{NNNN}.pth every 5 epochs.

### Task 2: Colab Training Notebook + Local Debug Notebook

**Files created:**
- `colab/03-train-colab.ipynb` — 10-cell full T4 training notebook
- `notebooks/03-debug-local.ipynb` — 2-cell local debug (5 epochs, 38 samples)

**Colab notebook Wave 0 gate (Cell 3):**
- Checks all 4 HDF5 files exist in Google Drive
- Verifies each file has >= 100K samples (MIN_SAMPLES_PER_FILE)
- Raises `RuntimeError` if check fails — prevents training on debug data

**Colab workflow:**
1. Mount Drive → gate check → copy HDF5 to /content/data SSD
2. `!python train.py --h5_dir /content/data --epochs 200 --batch_size 64 --lr 3e-4 --num_workers 2`
3. Plot loss curves from training_log.json

## Test Results

```
tests/test_train.py::test_weighted_mse_loss_ctr_only   PASSED
tests/test_train.py::test_weighted_mse_loss_device_safe PASSED
tests/test_train.py::test_early_stopper_triggers        PASSED
tests/test_train.py::test_checkpoint_roundtrip          PASSED
tests/test_train.py::test_debug_training_loop           PASSED (slow)

5 passed, 4 warnings in 3.57s
```

## Success Criteria Verification

- [x] `pytest tests/test_train.py -m "not slow"` exits 0 (4 passed)
- [x] `grep "RuntimeError" colab/03-train-colab.ipynb` returns match (Wave 0 gate)
- [x] `grep "MIN_SAMPLES_PER_FILE" colab/03-train-colab.ipynb` returns match
- [x] `grep "weighted_mse_loss" src/train.py` returns match
- [x] `grep "training_log.json" src/train.py` returns match
- [x] `grep "ReduceLROnPlateau" src/train.py` returns match
- [x] `grep "EarlyStopper" src/train.py` with `patience=15` returns match
- [x] Colab notebook has 10 cells (>= 10 required)

## Decisions Made

- `weights_only=True` in `load_checkpoint` — secure default for base model checkpoint
- `num_workers=0` as default (Windows-safe; Colab uses `--num_workers 2`)
- Per-output MSE logged without weights (raw MSE for diagnostics, separate from weighted total)
- JSON log appended and rewritten each epoch (simple, not performance-critical)
- Checkpoint save: best only + every 5 epochs (per RESEARCH.md recommendation for Colab session survival)

## Commits

1. `feat(03-03): implement training infrastructure (src/train.py) with tests`
2. `feat(03-03): add Colab training notebook and local debug notebook`
