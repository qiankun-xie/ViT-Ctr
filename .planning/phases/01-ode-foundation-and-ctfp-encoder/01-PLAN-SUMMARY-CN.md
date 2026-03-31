# Phase 1: ODE基础与ctFP编码器 — 计划摘要

**创建日期:** 2026-03-25
**状态:** 已规划，待执行
**计划数:** 3个计划，2个波次
**需求覆盖:** SIM-01, SIM-02, SIM-03, ENC-01, ENC-02 (5/5 全覆盖)

---

## 阶段目标

验证RAFT动力学ODE模拟器并冻结共享ctFP编码器——这两个根依赖是所有下游阶段的基础。

---

## 波次结构

| 波次 | 计划 | 并行性 | 内容概述 |
|------|------|--------|----------|
| 1 | 01-01, 01-02 | 可并行（零文件重叠） | ODE模拟器 + ctFP编码器 |
| 2 | 01-03 | 依赖波次1 | ODE验证 + 诊断数据集 + 人工检查点 |

---

## 计划 01-01: RAFT ODE模拟系统

**目标:** 基于矩量法构建RAFT聚合ODE模拟器，支持全部四种RAFT剂类型

**涉及需求:** SIM-01, SIM-02, SIM-03

**产出文件:**
- `src/raft_ode.py` — ODE系统核心实现
- `tests/test_raft_ode.py` — 单元/集成测试（10个测试）
- `tests/conftest.py` — 共享测试夹具
- `pyproject.toml` — 项目配置

**关键实现:**

1. **单平衡模型** (`raft_ode_single_eq`) — 11个ODE变量
   - 物种: [M], [I], [CTA], [P·], [Int]
   - 矩量: μ₀, μ₁, μ₂（活性链）, λ₀, λ₁, λ₂（死链）
   - 适用于三硫代碳酸酯、黄原酸酯、二硫代氨基甲酸酯

2. **预平衡模型** (`raft_ode_preequilibrium`) — 13个ODE变量
   - 额外跟踪: CTA₀（初始RAFT剂）, Int_pre（预平衡中间体）
   - 慢碎裂（kfrag₀ << kfrag）产生诱导期
   - 仅用于二硫代酯类型

3. **主接口函数** (`simulate_raft`)
   - 输入: 速率常数参数字典 + RAFT剂类型
   - ODE求解: `solve_ivp(method='Radau', rtol=1e-8, 逐分量atol)`
   - 输出: 均匀转化率网格上的Mn、分散度、归一化Mn

4. **三参数计算:**
   - `compute_retardation_factor` — 双ODE积分（有CTA vs 无CTA），D-06
   - `compute_inhibition_period` — 转化率达1%的时间比值，D-05

**测试覆盖（10个测试）:**
- `test_forward_simulation` — 基本模拟输出验证
- `test_preequilibrium_distinct` — 预平衡 vs 单平衡曲线差异
- `test_all_agent_types` — 四种RAFT剂类型均可运行
- `test_parameter_range_coverage` — Ctr范围 0.01-10000
- `test_cta_ratio_range` — [CTA]/[M]范围 0.001-0.1
- `test_limit_behavior_high_ctr` — 高Ctr极限（D < 1.3）
- `test_limit_behavior_low_ctr` — 低Ctr极限（D > 1.5）
- `test_retardation_factor_range` — 减速因子值域 (0, 1]
- `test_inhibition_period_range` — 诱导期值域 [0, 1]
- `test_mn_normalization` — Mn归一化验证

---

## 计划 01-02: ctFP指纹编码器

**目标:** 构建共享ctFP（链转移指纹）编码器，将RAFT动力学数据转换为64×64双通道图像张量

**涉及需求:** ENC-01, ENC-02

**产出文件:**
- `src/ctfp_encoder.py` — 编码器核心实现
- `tests/test_ctfp_encoder.py` — 编码器测试（10个测试）

**关键实现:**

`transform(data, img_size=64)` 函数：
- **输入:** (cta_ratio_norm, conversion, mn_norm, dispersity) 元组的可迭代对象
- **输出:** `torch.Tensor`，形状 (2, 64, 64)，dtype float32
- **坐标映射:**
  - x轴（列）→ [CTA]/[M]（归一化到[0,1]，训练时除以0.1）
  - y轴（行）→ 转化率 [0, 1]
- **通道定义:**
  - Ch0: Mn/Mn_theory（无量纲，约0-2）
  - Ch1: 分散度 Mw/Mn（≥1.0，截断于4.0）

**与ViT-RR的关键区别:**
- ViT-RR: row=进料比, col=总转化率, channels=各单体转化率
- ViT-Ctr: row=转化率, col=[CTA]/[M], channels=归一化Mn/分散度

**设计约束（ENC-02）:** 无Streamlit或训练框架依赖，可同时被训练管线和Web应用导入。

**测试覆盖（10个测试）:**
- `test_output_shape` — 输出形状 (2, 64, 64)
- `test_output_dtype` — float32类型
- `test_channel_assignment` — 通道分配正确性
- `test_axis_mapping` — 坐标轴映射验证
- `test_deterministic_output` — 确定性输出（字节一致）
- `test_empty_input` — 空输入处理
- `test_boundary_values` — 边界值映射
- `test_mn_not_raw` — Mn为归一化值
- `test_dispersity_range` — 分散度透传
- `test_no_framework_dependency` — 无框架依赖

---

## 计划 01-03: ODE验证与诊断数据集

**目标:** 对比文献曲线验证ODE，生成1000样本诊断数据集确认数值稳定性

**涉及需求:** SIM-01, SIM-02, SIM-03, ENC-01

**产出文件:**
- `src/diagnostic.py` — 诊断数据集生成器
- `notebooks/01_ode_validation.ipynb` — 交互式验证笔记本
- `tests/test_raft_ode.py` — 扩展诊断测试

**任务1: 诊断数据集与验证笔记本**

1. **诊断数据集生成器** (`generate_diagnostic_dataset`)
   - 每种RAFT剂类型250个样本 × 4类型 = 1000总计
   - 参数网格: 10个Ctr值 (log10 -2到4) × 25个[CTA]/[M]值 (0.001到0.1)
   - 使用 `joblib.Parallel` 并行生成
   - 成功标准: 失败率 < 2%

2. **文献验证笔记本** — 三个RAFT体系对比:
   - **体系1: CDB/苯乙烯（二硫代酯）** — 预期行为: 明显诱导期，D先降后升，Mn在诱导期后线性增长
   - **体系2: 十二烷基TTC/MMA（三硫代碳酸酯）** — 预期行为: 无诱导期，D < 1.4，Mn线性增长
   - **体系3: O-乙基黄原酸酯/VAc（黄原酸酯）** — 预期行为: D≈1.2至65%转化率，之后逐渐上升

**任务2: 人工验证检查点（阻塞性）**

用户需视觉确认:
1. ODE曲线与文献趋势定性匹配
2. 极限参数行为正确（高Ctr→低D，低Ctr→高D）
3. ctFP可视化显示合理的双通道模式
4. 全部测试通过
5. 诊断数据集失败率 < 2%

---

## 成功标准总结

| # | 标准 | 验证方法 |
|---|------|----------|
| 1 | ODE模拟器至少对三种RAFT剂类型（二硫代酯、三硫代碳酸酯、黄原酸酯）重现已发表的Mn-转化率和D-转化率曲线 | 笔记本可视化对比 |
| 2 | 二硫代酯的两阶段预平衡机制产生与单平衡模型明显不同的Mn/D曲线 | `test_preequilibrium_distinct` 测试 |
| 3 | ctFP编码器从原始数据产生64×64×2张量，在训练和Web应用上下文中输出字节一致 | `test_deterministic_output` 测试 |
| 4 | 1000样本诊断数据集覆盖全Ctr范围（log10 = -1到4），四种RAFT剂类型均无数值失败（<2%） | `test_diagnostic_dataset` 慢测试 |

---

## 执行命令

```
/gsd:execute-phase 1
```

> 建议先 `/clear` 清理上下文窗口

---

*Phase: 01-ode-foundation-and-ctfp-encoder*
*摘要生成: 2026-03-25*
