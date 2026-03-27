# Plan 02-01 Summary: LHS Parameter Sampling and Parallel Dataset Generation

## Outcome: COMPLETED

## What Was Done

### Task 1: LHS参数采样器和噪声注入
- 创建 `src/dataset_generator.py`，实现三个核心函数：
  - `generate_lhs_parameters(n_samples, raft_type, seed)`: 使用 `scipy.stats.qmc.LatinHypercube(d=7)` 在7维参数空间中均匀采样，支持4种RAFT类型。参数包括 log10_Ctr, log10_kp, log10_kt, log10_kd (对数均匀), I0, f (线性), log10_cta_ratio (对数均匀)。Dithioester类型自动添加预平衡参数(kadd0, kfrag0)。
  - `inject_noise(Mn_array, D_array, sigma=0.03)`: 乘性高斯噪声注入，模拟GPC测量误差。确保分散度 >= 1.0。
  - `simulate_single_sample(params)`: 完整管线——ODE求解 -> 噪声注入 -> ctFP编码 -> 标签计算(log10_Ctr, inhibition, retardation)。返回结构化结果字典，失败时返回错误信息。

### Task 2: 并行生成和HDF5存储
- 在同一文件中实现：
  - `generate_dataset_parallel(raft_type, n_samples, output_path, seed)`: 使用 `joblib.Parallel(n_jobs=-1, prefer='processes')` 并行调用 `simulate_single_sample()`，集成tqdm进度条，计算失败率，超过5%阈值时记录失败参数日志。
  - `save_to_hdf5(samples, raft_type, output_path)`: 创建chunked HDF5文件(chunk=1000)，包含 fingerprints(n,64,64,2)、labels(n,3)、params(结构化数组) 三个数据集。
  - `main()`: 遍历4种RAFT类型，每种250K样本，打印总结报告。

## Verification Results

### Task 1 Verification
- LHS采样器生成100个dithioester参数: 包含所有必需key（log10_Ctr, raft_type, kadd0, kfrag0等）
- LHS采样器生成50个TTC参数: 无kadd0（正确）
- 噪声注入: shape保持不变，所有D >= 1.0
- 单样本模拟: TTC样本成功，指纹shape=(64,64,2)，标签=[log10_Ctr, inhibition, retardation]

### Task 2 Verification
- 20样本TTC集成测试: 20/20成功(0%失败率)
- HDF5文件结构正确: fingerprints(20,64,64,2), labels(20,3), params(20,)
- 20样本Dithioester集成测试: 20/20成功(0%失败率)
- 指纹值范围合理: [0.0, 2.6]
- 标签值合理: log10_Ctr~2.0, inhibition~0.008, retardation~1.0

## Decisions Made

- 噪声sigma固定为0.03（3%相对误差，在D-05规定的0.02-0.05范围内）
- Dithioester预平衡kadd0设为主平衡kadd的0.1倍，kfrag0=1.0（与diagnostic.py一致）
- HDF5存储格式: (n, 64, 64, 2) 即 (N, H, W, C)，便于后续训练管线的维度转换
- 所有RAFT类型共用固定kfrag=1e4（与Phase 1诊断脚本一致）

## Files Changed

| File | Change |
|------|--------|
| `src/dataset_generator.py` | 新建 — 完整的大规模数据集生成管线 |
| `tests/test_dataset_generator.py` | 新建 — 23单元测试 + 3集成测试 |

## Duration

~15 minutes

---
*Plan: 02-01-PLAN.md*
*Completed: 2026-03-27*
