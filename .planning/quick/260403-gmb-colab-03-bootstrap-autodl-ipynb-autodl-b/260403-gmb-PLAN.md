# Plan: 创建 colab/03-bootstrap-autodl.ipynb

**Mode:** quick
**Created:** 2026-04-03
**Directory:** .planning/quick/260403-gmb-colab-03-bootstrap-autodl-ipynb-autodl-b

## Objective

将 `colab/03-bootstrap-colab.ipynb` 适配为 AutoDL 版本，保持完全相同的 cell 结构和逻辑，仅修改平台相关部分。

## Diff Summary（Colab → AutoDL）

| Cell | Colab | AutoDL |
|------|-------|--------|
| 0 | 标题（Colab注意事项） | 标题（AutoDL注意事项） |
| 1 | `!pip install h5py` | **删除**（AutoDL镜像已包含） |
| 2 | `drive.mount(...)` | **删除**（AutoDL本地存储） |
| 3 | `sys.path.insert(0, '/content/ViT-Ctr/src')` | `sys.path.insert(0, '/root/autodl-tmp/ViT-Ctr/src')` |
| 4 | `LOCAL_DATA_DIR = '/content/data'`，`CHECKPOINT_DIR = '/content/checkpoints'` | `/root/autodl-tmp/data`，`/root/autodl-tmp/checkpoints` |
| 5 | 同（无改动） | 同 |
| 6 | `num_workers=2` | `num_workers=4` |
| 7–11 | 同（无改动） | 同 |
| 12 | Colab下载说明 | AutoDL下载/导出说明 |

重新编号：删除 cell-1（pip）和 cell-2（drive.mount）后，原 cell-3 变为新 cell-1，以此类推，共 11 个 cell（0–10）。

## Tasks

### Task 1: 创建 colab/03-bootstrap-autodl.ipynb

**文件:** `colab/03-bootstrap-autodl.ipynb`
**操作:** Write（新建）
**内容:** 按以下 cell 顺序构建 notebook JSON：

**Cell 0（markdown）：** 标题 + AutoDL平台说明
```
# ViT-Ctr Phase 3: Bootstrap不确定性量化（AutoDL版）
在完成基础训练后运行。需要 `/root/autodl-tmp/checkpoints/best_model.pth`。

**策略 (D-12/D-13):** 冻结SimpViT backbone，200次有放回重采样训练集，每次微调fc输出头5个epoch。

**注意:** 每次迭代约6分钟（RTX 3090），共200次 ≈ 20小时。推荐在持久化会话（tmux/screen）中运行。
AutoDL数据已在本地 `/root/autodl-tmp/`，无需挂载Drive。
```

**Cell 1（code）：** sys.path setup
```python
import sys, os
CODE_DIR = '/root/autodl-tmp/ViT-Ctr'
sys.path.insert(0, os.path.join(CODE_DIR, 'src'))
print(f"Code directory: {CODE_DIR}")
```

**Cell 2（code）：** config（AutoDL路径，同 train-autodl 模式）
```python
import os, json, torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOCAL_DATA_DIR = '/root/autodl-tmp/data'
CHECKPOINT_DIR = '/root/autodl-tmp/checkpoints'
os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
H5_PATHS = [os.path.join(LOCAL_DATA_DIR, f) for f in ['dithioester.h5', 'trithiocarbonate.h5', 'xanthate.h5', 'dithiocarbamate.h5']]
BASE_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
BOOTSTRAP_HEADS_PATH = os.path.join(CHECKPOINT_DIR, 'bootstrap_heads.pth')
CALIBRATION_PATH = os.path.join(CHECKPOINT_DIR, 'calibration.json')
N_BOOTSTRAP = 200
N_EPOCHS_PER_HEAD = 5
BOOTSTRAP_LR = 1e-3
print(f'Device: {DEVICE}, N_BOOTSTRAP={N_BOOTSTRAP}')
```

**Cell 3（code）：** load base model（原 cell-5，无改动）

**Cell 4（code）：** build datasets/loaders（原 cell-6，num_workers=4）

**Cell 5（code）：** debug bootstrap 3次（原 cell-7，无改动）

**Cell 6（code）：** full bootstrap 200次（原 cell-8，无改动）

**Cell 7（code）：** calibration（原 cell-9，无改动）

**Cell 8（code）：** save calibration.json（原 cell-10，无改动）

**Cell 9（code）：** demo prediction（原 cell-11，无改动）

**Cell 10（markdown）：** 完成说明（AutoDL版）
```
## 完成后
1. 在 `/root/autodl-tmp/checkpoints/` 确认 `bootstrap_heads.pth` 和 `calibration.json` 已生成
2. 使用 AutoDL 文件管理器或 scp 下载到本地 `checkpoints/`
3. 验证 `calibration.json` 中 `empirical_coverage_after` 所有值 >= 0.95
```

**实现要点：**
- 输出标准 Jupyter notebook JSON 格式（`nbformat: 4, nbformat_minor: 5`）
- cell id 使用 `cell-0` 到 `cell-10` 格式（与 colab 版一致）
- 所有 code cell 的 `outputs: []`，`execution_count: null`
- markdown cell 的 `cell_type: "markdown"`

## Execution Notes

- **无依赖**：直接新建文件，不修改任何现有文件
- **验证方法**：打开 notebook 确认 11 个 cell 存在，搜索确认无 `drive.mount`、无 `pip install h5py`、无 `/content/` 路径
- **预计时间**：5 分钟
