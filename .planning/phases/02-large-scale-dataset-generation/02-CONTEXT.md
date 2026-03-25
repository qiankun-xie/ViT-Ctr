# Phase 2: Large-Scale Dataset Generation - Context

**Gathered:** 2026-03-25
**Status:** Ready for planning

<domain>
## Phase Boundary

基于Phase 1验证通过的RAFT动力学ODE和ctFP编码器，使用Latin Hypercube采样在完整参数空间上并行生成~1M合成ctFP样本（含噪声注入），按RAFT剂类型分文件存储为chunked HDF5格式，上传Google Drive供Colab训练使用。

</domain>

<decisions>
## Implementation Decisions

### 动力学参数采样策略
- **D-01:** kp（传播速率常数）和kt（终止速率常数）在合理范围内随机采样（如kp=100-10000 L/mol·s，kt=1e6-1e9 L/mol·s），不绑定特定单体。让模型学习更通用的特征。
- **D-02:** 引发剂参数（kd、[I]₀、f）同样在合理范围内随机采样（kd=1e-6–1e-4 s⁻¹，[I]₀=0.001–0.05 M，f=0.5–0.8）。
- **D-03:** 多参数联合采样使用Latin Hypercube Sampling (LHS)，确保高维参数空间的均匀覆盖，避免纯随机采样的聚集问题。
- **D-04:** 承继Phase 1决定：Ctr=0.01–10000 (log-uniform)，[CTA]₀/[M]₀=0.001–0.1 (log-uniform)，温度固定，四种RAFT剂均衡采样（~250K/类型）。

### 噪声注入与鲁棒性
- **D-05:** 在ODE模拟输出的"干净"Mn和Đ值上注入乘性高斯噪声，模拟GPC实验测量误差。每个数据点的Mn和Đ分别乘以(1+ε)，ε~N(0, σ²)，σ=0.02–0.05（2-5%相对误差）。
- **D-06:** 噪声注入在ctFP编码之前完成，即先加噪声再编码为64×64×2图像。

### HDF5存储组织方式
- **D-07:** 按RAFT剂类型分文件存储，共4个HDF5文件（dithioester、trithiocarbonate、xanthate、dithiocarbamate），每个~250K样本（~8GB）。便于分步生成、上传和按类型调试。
- **D-08:** HDF5 chunk大小为1000样本/chunk，兼顾I/O效率和内存占用。

### 模拟失败处理策略
- **D-09:** ODE求解失败的参数组合直接跳过，记录失败率和失败参数分布到日志文件。
- **D-10:** 可接受的最大失败率阈值为5%。超过5%则报警暂停，需检查ODE实现或参数范围是否有问题。失败率<5%且分布无明显偏斜即为合格。

### Claude's Discretion
- joblib并行度配置（n_jobs值、batch调度策略）
- ODE求解器具体容差设置和超时配置
- LHS采样的具体实现库选择（scipy.stats.qmc.LatinHypercube等）
- 噪声σ的具体值（在0.02-0.05范围内）
- HDF5压缩方案（gzip/lzf/不压缩）
- 进度报告频率和日志格式
- Google Drive上传脚本实现方式

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### 参考项目
- `C:/CodingCraft/DL/ViT-RR/model_utils.py` — transform()编码函数的参考实现，ctFP编码器直接参考
- `C:/CodingCraft/DL/ViT-RR/deploy.py` — Bootstrap推理模式参考

### 研究文档
- `.planning/research/STACK.md` — 技术栈推荐（joblib并行、HDF5存储、SciPy ODE）
- `.planning/research/ARCHITECTURE.md` — 系统架构和数据流
- `.planning/research/PITFALLS.md` — 领域陷阱，特别是C2（参数空间覆盖）和C4（数据存储策略）

### Phase 1输出
- `.planning/phases/01-ode-foundation-and-ctfp-encoder/01-CONTEXT.md` — ODE变量定义、三参数输出定义、参数空间设计等基础决策

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- Phase 1将产出 `raft_ode.py`（ODE模拟器）和 `ctfp_encoder.py`（指纹编码器），Phase 2直接调用
- ViT-RR的数据生成流程可参考其整体模式

### Established Patterns
- SciPy `solve_ivp` with `method='Radau'` 处理刚性ODE（Phase 1决定）
- joblib `Parallel(n_jobs=-1)(delayed(simulate)(params))` 并行模式（STACK.md推荐）
- tqdm进度条集成

### Integration Points
- Phase 1的ODE模拟器模块是Phase 2的核心依赖——必须先完成Phase 1
- Phase 1的ctFP编码器模块用于将模拟结果转换为训练数据
- 生成的HDF5文件是Phase 3训练的直接输入

</code_context>

<specifics>
## Specific Ideas

- 所有动力学参数使用对数均匀采样（log-uniform），因为这些参数跨越多个数量级
- 噪声注入模拟的是GPC（凝胶渗透色谱）仪器的测量误差特征
- 按RAFT类型分文件的设计便于在Phase 3中按类型评估模型性能（对应EVL-03要求）
- 失败率日志可用于回溯优化参数范围边界

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-large-scale-dataset-generation*
*Context gathered: 2026-03-25*
