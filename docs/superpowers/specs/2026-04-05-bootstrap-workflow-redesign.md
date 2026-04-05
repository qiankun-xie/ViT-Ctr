# Bootstrap Workflow Redesign

## Problem

AutoDL bootstrap工作流太复杂、容易出错：
- Import/路径问题：sys.path操作在本地和AutoDL不一致，经常import失败
- 上传部署繁琐：不确定该上传哪些文件、放到哪个目录
- 运行监控/resume不可靠：中断后resume不确定能否恢复
- 产物下载混乱：产物散落多个文件，不确定该下载哪些

## Solution

**方案A：自包含单文件 + 3步bat脚本**

将所有bootstrap逻辑合并成一个零依赖的自包含Python脚本，本地3步操作完成全流程。

## File Structure

### After Refactoring

```
colab/autodl_bootstrap.py    # 自包含脚本（唯一远程执行文件）
scripts/autodl-sync.bat      # Step 1: 上传
scripts/autodl-run.bat        # Step 2: 运行
scripts/autodl-download.bat   # Step 3: 下载
scripts/autodl-config.bat     # 共享SSH配置（保留）
scripts/autodl-status.bat     # 查询进度（保留）
src/bootstrap.py              # 本地推理接口（Streamlit/literature_validation用）
tests/test_bootstrap.py       # 测试本地接口
```

### Deleted

- `colab/03-bootstrap-colab.ipynb`
- `colab/03-bootstrap-autodl.ipynb`

## Component Design

### 1. colab/autodl_bootstrap.py — 自包含脚本（~300行）

零外部import（仅依赖torch/numpy/scipy/h5py）。包含运行bootstrap所需的一切。

**内部结构：**

```python
# ── 模型定义（~30行，从src/model.py复制） ──
class SimpViT(nn.Module): ...

# ── Bootstrap核心（~120行） ──
def freeze_backbone(model): ...
def run_bootstrap(model, train_fps, train_lbls, *, save_dir, resume=False, ...): ...
    # 每完成一个head立即追加保存到bootstrap_heads.pth
    # 实时更新bootstrap_progress.json

# ── 校准（~60行） ──
def collect_ensemble_predictions(model, fps, lbls, heads, ...): ...
def compute_jci(cov_matrix, n, p=3): ...
def calibrate_coverage(y_true, y_pred_mean, half_widths, target=0.95): ...

# ── 数据加载（~40行） ──
def load_data(h5_dir, base_model_path): ...
    # 读HDF5 → 分层split → preload到RAM → 返回(train/val/test)元组

# ── 打包（~20行） ──
def package_results(ckpt_dir): ...
    # → bootstrap_results.tar.gz

# ── CLI入口（~30行） ──
def main():
    parser: --h5_dir, --ckpt_dir, --resume, --calibrate_only
    # 流程：load_data → run_bootstrap → calibrate → verify → package
```

**关键设计点：**

- `run_bootstrap` 每完成一个head立即追加保存，resume时跳过已完成的
- `bootstrap_progress.json` 实时更新：`{"completed": 47, "total": 200, "eta_minutes": 12.3}`
- 最终产物打包为单个 `bootstrap_results.tar.gz`
- CLI参数：`--h5_dir`, `--ckpt_dir`, `--base_model`, `--resume`, `--calibrate_only`, `--batch_size`, `--skip_verify`, `--skip_package`

### 2. src/bootstrap.py — 本地推理接口（~80行）

只保留推理时需要的函数，供Streamlit和literature_validation调用：

```python
def predict_with_uncertainty(model, fp_tensor, bootstrap_ckpt, cal_factors, device='cpu'):
    """单样本或小批量推理，返回mean、half_widths"""

def compute_jci(cov_matrix, n, p=3):
    """F-distribution 95% JCI"""

def load_calibration(path):
    """读取calibration.json，返回cal_factors dict"""
```

`compute_jci` 在两个文件中重复（~10行），自包含方案的可接受代价。

**下游影响：**
- `src/literature_validation.py` — 继续 `from bootstrap import predict_with_uncertainty`，无变化
- Streamlit app — 继续用 `predict_with_uncertainty`，无变化

### 3. 3步工作流脚本

#### Step 1: `autodl-sync.bat`

```
autodl-sync.bat
```

- SCP上传 `colab/autodl_bootstrap.py` → 远程 `/root/autodl-tmp/bootstrap/autodl_bootstrap.py`
- SCP上传 `checkpoints/best_model.pth` → 远程 `/root/autodl-tmp/checkpoints/best_model.pth`（如果本地有）
- SSH检查远程HDF5数据是否存在，打印确认信息
- 失败时明确报错

只上传2个文件，不需要src/目录。

#### Step 2: `autodl-run.bat`

```
autodl-run.bat [full|resume|calibrate]
```

- SSH到AutoDL，在tmux session `bootstrap` 中执行 `python autodl_bootstrap.py [flags]`
- 打印"已启动，用 autodl-status.bat 查看进度"

#### Step 3: `autodl-download.bat`

```
autodl-download.bat
```

- SCP下载 `bootstrap_results.tar.gz` → 本地 `checkpoints/`
- 自动解压（得到 `bootstrap_heads.pth`, `calibration.json`, `bootstrap_summary.json`）
- 打印产物清单和文件大小
- 删除本地tar.gz

#### autodl-status.bat（保留）

```
autodl-status.bat
```

- SSH读取远程 `bootstrap_progress.json`，打印进度和ETA

### 4. 测试策略

`tests/test_bootstrap.py` 只测试 `src/bootstrap.py` 中保留的函数（~100行，5个测试）：

- `test_predict_with_uncertainty()` — 单样本推理，检查输出shape和值域
- `test_predict_with_uncertainty_restores_model()` — 推理后模型状态不被污染
- `test_compute_jci()` — F-distribution公式正确性
- `test_compute_jci_invalid_inputs()` — 边界情况（n ≤ p、非有限值）
- `test_load_calibration()` — JSON读取和格式校验

`autodl_bootstrap.py` 的训练逻辑不在本地测试——脚本末尾自带verification步骤（打印5个样本的预测结果），在AutoDL上端到端验证。

## Migration

1. 重写 `colab/autodl_bootstrap.py` 为自包含脚本
2. 精简 `src/bootstrap.py` 为本地推理接口
3. 重写 `tests/test_bootstrap.py`
4. 重写3个bat脚本（sync/run/download）
5. 删除2个notebook
6. 更新 `src/literature_validation.py` 的import（如需要）
