---
phase: 02
phase_name: large-scale-dataset-generation
researched: 2026-03-26
status: complete
---

# Phase 2 Research: Large-Scale Dataset Generation

## Research Question

如何高效生成~1M样本的ctFP数据集，确保参数空间均匀覆盖、数值稳定性、存储可扩展性，并为Colab训练做好准备？

## Key Findings

### 1. Latin Hypercube Sampling (LHS) 必要性

**问题:** 8维参数空间（Ctr, kp, kt, kd, [I]₀, f, [CTA]₀/[M]₀, RAFT类型）的纯随机采样会产生聚集和空白区域。

**解决方案:** 使用`scipy.stats.qmc.LatinHypercube`进行分层采样：
- 每个维度划分为N个等概率区间
- 每个区间恰好采样一次
- 保证高维空间的均匀覆盖

**实现要点:**
```python
from scipy.stats import qmc
sampler = qmc.LatinHypercube(d=7)  # 7个连续维度
sample = sampler.random(n=250000)
# 对数空间参数需要逆变换: 10**(log_min + sample * (log_max - log_min))
```

### 2. 并行化策略

**约束:** 本地CPU并行（MX350 GPU不适合ODE求解），目标~1M样本。

**方案:** joblib + 批处理
- `joblib.Parallel(n_jobs=-1, prefer='processes')` 利用所有CPU核心
- 每批1000个参数组合，减少进程间通信开销
- `tqdm` 集成进度跟踪

**Phase 1经验:** `prefer='threads'` 用于Windows诊断，但大规模生成应使用 `prefer='processes'` 避免GIL瓶颈。

### 3. HDF5 分块存储设计

**需求:** 1M × 64×64×2 float32 ≈ 32 GB，超出单次加载内存限制。

**设计:**
- 按RAFT类型分4个文件（每个~250K样本，~8GB）
- HDF5 chunk size = 1000 samples
- 数据集结构:
  ```
  dithioester.h5
  ├── fingerprints: (250000, 64, 64, 2) float32, chunked
  ├── labels: (250000, 3) float32  # [log10_Ctr, inhibition, retardation]
  └── metadata: (250000,) structured array  # 参数记录
  ```

**h5py写入模式:**
```python
with h5py.File('dithioester.h5', 'w') as f:
    dset = f.create_dataset('fingerprints', shape=(250000,64,64,2),
                            dtype='float32', chunks=(1000,64,64,2))
    # 批量写入，避免一次性加载全部数据
```

### 4. 噪声注入时机

**决策 D-05/D-06:** 噪声在ODE输出后、ctFP编码前注入。

**原因:**
- 模拟GPC测量误差（Mn和Đ的独立误差）
- 编码前注入确保噪声体现在指纹图像中
- 乘性高斯噪声: `Mn_noisy = Mn * (1 + ε)`, ε ~ N(0, σ²), σ=0.02-0.05

### 5. 失败处理与质量控制

**Phase 1发现:** 极端参数组合可能导致ODE求解失败（刚性问题、数值溢出）。

**策略:**
- 每个ODE调用包裹在try-except中
- 失败样本跳过，记录失败参数到日志
- 可接受失败率: <5% (决策D-10)
- 失败率>5%触发报警，需检查参数范围

**验证指标:**
- Ctr分布直方图（log空间）应无明显空白
- 四种RAFT类型样本数接近均衡（±5%）
- 失败率<5%且无系统性偏斜

### 6. Google Drive 上传准备

**Phase 3依赖:** Colab需从Google Drive加载数据集。

**方案:**
- 使用`gdown`或Google Drive API上传4个HDF5文件
- 生成共享链接，记录在`.planning/phases/02-*/02-DATASET-INFO.md`
- 验证Colab可访问（测试脚本下载并加载前100样本）

## Implementation Checklist

- [ ] LHS采样器实现（7维连续参数 + 1维离散RAFT类型）
- [ ] 噪声注入函数（乘性高斯，σ可配置）
- [ ] joblib并行ODE生成循环，含失败处理
- [ ] HDF5分文件写入（4个RAFT类型）
- [ ] 数据集验证脚本（分布检查、失败率统计）
- [ ] Google Drive上传脚本 + Colab访问测试

## Dependencies

**Phase 1输出（必须完成）:**
- `src/raft_ode.py` — ODE模拟器
- `src/ctfp_encoder.py` — 指纹编码器
- Phase 1验证通过，确认ODE数值稳定性

**外部库:**
- `scipy.stats.qmc` (LHS采样)
- `joblib` (并行)
- `h5py` (HDF5存储)
- `tqdm` (进度条)

## Risks

1. **ODE失败率过高:** 参数范围可能需要收窄，影响模型泛化能力
2. **HDF5文件损坏:** 生成过程中断可能导致部分文件不可用，需支持断点续传
3. **Google Drive配额:** 32GB上传可能触发配额限制，需备用方案（如分批上传）

---

*Research completed: 2026-03-26*
*Ready for planning*
