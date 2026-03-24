# Phase 1: ODE Foundation and ctFP Encoder - Context

**Gathered:** 2026-03-24
**Status:** Ready for planning

<domain>
## Phase Boundary

构建并验证RAFT聚合动力学ODE模拟器（含完整矩量方程组），实现双通道ctFP指纹编码器，并通过文献曲线对比和极限行为校验验证ODE正确性。为Phase 2的百万级数据生成奠定基础。

</domain>

<decisions>
## Implementation Decisions

### ODE变量与矩量方程
- **D-01:** 使用完整矩量方程组，跟踪[M]、[I]、[CTA]、[P·]、[Int]（RAFT中间体）、λ0/λ1/λ2（活性链+死链）。约10-15个ODE变量。
- **D-02:** Mn和Đ通过矩量计算：Mn = λ1/λ0 × M_monomer，Đ = λ2·λ0/λ1²。不做简化假设（如稳态自由基浓度），以保留inhibition和retardation信号。
- **D-03:** 不同RAFT剂类型使用分类型ODE。Dithioester使用两阶段预平衡模型（slow fragmentation），其他类型（trithiocarbonate、xanthate、dithiocarbamate）使用简化平衡模型。通过kadd/kfrag比值区分行为差异。

### 三参数输出定义
- **D-04:** Ctr输出尺度：log10(Ctr)。与ViT-RR一致，适合跨多个数量级的参数。
- **D-05:** Inhibition period定义：t_inh / t_total（无量纲比值，范围0-1）。t_inh为反应开始到转化率达到1%的时间。无量纲化消除绝对速率常数的影响，使标签更稳健。
- **D-06:** Retardation factor定义：Rp(RAFT) / Rp(no CTA)（无量纲速率比，范围0-1）。在特定转化率点（如50%）计算有CTA和无CTA体系的聚合速率之比。值越小表示减速越严重。

### 参数空间设计
- **D-07:** Ctr采样范围：0.01-10000（log10: -2到4），对数均匀采样。覆盖xanthate（低Ctr）到dithioester（高Ctr）全范围。
- **D-08:** [CTA]₀/[M]₀采样范围：0.001-0.1，对数均匀采样。覆盖实验常用范围。
- **D-09:** 温度固定，不作为变量。Ctr = ktr/kp已隐含温度影响，简化模型输入。
- **D-10:** 四种RAFT剂类型均衡采样，每类约250K样本（总计~1M）。确保模型不偏向某类型。

### ODE验证策略
- **D-11:** 采用两者结合的验证方案：(1) 极限行为校验——在极端参数下检查ODE输出是否符合理论极限；(2) 文献曲线对比——找3-4篇发表文献中的Mn-conversion和Đ-conversion曲线，调参后对比模拟与实验是否匹配。至少覆盖dithioester、trithiocarbonate、xanthate三种类型。

### Claude's Discretion
- ODE求解器具体配置（Radau参数、容差设置）
- ctFP编码中的归一化策略（Mn和Đ的值域映射方式）
- 诊断数据集（1000样本）的具体参数网格分布

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### 参考项目
- `C:/CodingCraft/DL/ViT-RR/model_utils.py` — SimpViT架构定义和transform()编码函数，ctFP编码器的参考实现
- `C:/CodingCraft/DL/ViT-RR/deploy.py` — Bootstrap推理和Streamlit部署模式

### 研究文档
- `.planning/research/STACK.md` — 技术栈推荐（PyTorch + SciPy Radau + joblib）
- `.planning/research/ARCHITECTURE.md` — 系统架构和组件边界
- `.planning/research/PITFALLS.md` — 领域陷阱，特别是C1（预平衡机制）和C2（参数空间覆盖）
- `.planning/PROJECT.md` — 项目上下文和约束

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- ViT-RR `model_utils.py` 中的 `transform()` 函数：ctFP编码器的直接参考，需改造坐标轴含义（x=[CTA]/[M], y=conversion）和通道含义（Ch1=Mn, Ch2=Đ）
- ViT-RR `model_utils.py` 中的 `SimpViT` 类：模型架构可直接复用，仅需修改 `in_channels=2`（不变）和 `num_outputs=3`（从2改为3）

### Established Patterns
- 输出使用对数尺度（log10），与ViT-RR的log10(r)一致
- SciPy `solve_ivp` with `method='Radau'` 处理刚性ODE（research推荐）

### Integration Points
- ctFP编码函数必须作为独立模块（如 `ctfp_encoder.py`），同时被训练管线和Web应用导入
- ODE模拟器作为独立模块（如 `raft_ode.py`），被数据生成脚本调用

</code_context>

<specifics>
## Specific Ideas

- 参照ViT-RR的rFP概念命名为ctFP（chain transfer Fingerprint）
- ODE验证时需要的文献曲线由phase-researcher在研究阶段搜集具体论文引用
- 完整矩量方程组的选择是为了确保训练数据中包含inhibition和retardation信号——这是三参数预测的前提条件

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-ode-foundation-and-ctfp-encoder*
*Context gathered: 2026-03-24*
