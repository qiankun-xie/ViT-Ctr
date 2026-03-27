---
phase: 03-model-training-and-evaluation
plan: 01
status: complete
completed: "2026-03-27"
---

# 03-01 SUMMARY: SimpViT模型 + 指标与可视化工具

## 构建内容

### Task 1: SimpViT模型 (`src/model.py`)

直接移植自ViT-RR的`model_utils.py`，唯一修改：`num_outputs=2 → num_outputs=3`。

**架构参数：**
- `img_size=64, patch_size=16` → `num_patches = (64//16)² = 16`
- `hidden_size=64, num_layers=2, num_heads=4`
- 输入通道：2（双通道ctFP：Ch1=Mn, Ch2=Đ）
- `position_embedding` 保持 `(1, 17, 64)` 的遗留形状（`num_patches+1=17`）
- 输出：`Linear(64, 3)` → `(B, 3)`

**前向传播流程：**
```
Input:  (B, 2, 64, 64)
Conv2d(in=2, out=64, kernel=16, stride=16) → (B, 64, 4, 4)
permute(0,2,3,1) → (B, 4, 4, 64)
view → (B, 16, 64)
+ position_embedding[:, :16]  # 切片 (1,17,64) 取前16
permute(1,0,2) → (16, B, 64)
TransformerEncoder (2层, batch_first=False) → (16, B, 64)
mean(dim=0) → (B, 64)
Linear(64, 3) → (B, 3)
```

### Task 2: 指标与可视化工具 (`src/utils/`)

- `src/utils/metrics.py`: `r2_score_np`, `rmse_np`, `mae_np`（纯NumPy，无sklearn）
- `src/utils/visualization.py`: `parity_plot`, `residual_hist`（`rasterized=True`防止百万点图文件过大）

## 参数量更正（规划文档 vs. 实际）

### 规划预估（错误）

`03-01-PLAN.md` 中写道：
> `test_simpvit_param_count`: assert count is between 3_000_000 and 4_000_000 (expected ~3.4M)

**该估计是规划错误**，可能源于将 `hidden_size=64` 误认为 `hidden_size=256`（256²×某系数 ≈ 3.4M）。

### 实际参数量：877,571

| 组件 | 参数量 |
|------|--------|
| `position_embedding` | 1,088 |
| `patch_embedding` (Conv2d 2→64, k=16) | 32,832 |
| `transformer_encoder_layer`（模板层，被PyTorch计入） | ~266,368 |
| `transformer_encoder.layers.0`（第1层） | ~266,368 |
| `transformer_encoder.layers.1`（第2层） | ~266,368 |
| `fc` (Linear 64→3) | 195 |
| **合计** | **877,571** |

**关键细节：** `nn.TransformerEncoderLayer(d_model=64)` 的 `dim_feedforward` 默认值为 **2048**，FFN层（`linear1: 64×2048`, `linear2: 2048×64`）贡献了绝大多数参数。同时，PyTorch会将模板层（`self.transformer_encoder_layer`）与`TransformerEncoder`内的复制层同时计入`named_parameters()`，因此参数被计数3次（模板层 + 2个encoder层），但这与ViT-RR的行为完全一致。

### 修复

将 `tests/test_model.py` 中的断言修改为：
```python
# 旧（错误）
assert 3_000_000 <= total_params <= 4_000_000

# 新（正确）
assert 800_000 <= total_params <= 950_000
```

## 测试结果

```
pytest tests/test_model.py -x -v

tests/test_model.py::test_simpvit_forward      PASSED
tests/test_model.py::test_simpvit_param_count  PASSED
tests/test_model.py::test_simpvit_eval_mode    PASSED

3 passed in 2.01s
```

## 符合成功标准

- [x] `pytest tests/test_model.py tests/test_metrics.py tests/test_visualization.py` 中 test_model.py 的3个测试全部通过
- [x] `src/model.py` 不含 `SimpViT_3D`
- [x] `src/model.py` 含 `num_outputs=3`
- [x] `src/utils/visualization.py` 含 `rasterized=True`（Task 2，由前一执行者完成）
- [x] `src/utils/` 无 sklearn 导入

## 提交记录

- `1cb5e7e` — `fix(03-01): correct param count range in test_simpvit_param_count`

## 后续影响

参数量877K与训练效率相关：
- Colab T4（16GB VRAM）可轻松容纳更大batch size
- 推理速度优于原始预估的3.4M参数模型
- 模型表达能力由FFN的`dim_feedforward=2048`保证（非hidden_size维度）
