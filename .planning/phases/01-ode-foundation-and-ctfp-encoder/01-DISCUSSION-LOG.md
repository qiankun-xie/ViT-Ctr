# Phase 1: ODE Foundation and ctFP Encoder - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-03-24
**Phase:** 01-ODE Foundation and ctFP Encoder
**Areas discussed:** ODE变量与矩量方程, 三参数输出定义, 参数空间设计, ODE验证策略

---

## ODE变量与矩量方程

### Mn/Đ获取方法

| Option | Description | Selected |
|--------|-------------|----------|
| 矩量方程（推荐） | 跟踪链长分布的第0、1、2阶矩量，Mn=λ1/λ0×M_monomer，Đ=λ2λ0/λ1²。计算效率高 | ✓ |
| 分子数显式跟踪 | 跟踪每条链的长度分布，精确但计算量大 | |

**User's choice:** 矩量方程
**Notes:** 百万级数据生成需要高效计算

### RAFT剂类型ODE处理

| Option | Description | Selected |
|--------|-------------|----------|
| 统一ODE + 参数区分 | 所有类型用一套ODE，通过kadd/kfrag比值区分 | |
| 分类型ODE（推荐） | dithioester用两阶段预平衡，其他用简化模型 | ✓ |

**User's choice:** 分类型ODE
**Notes:** Dithioester的slow fragmentation需要特殊处理

### ODE变量组

| Option | Description | Selected |
|--------|-------------|----------|
| 完整矩量方程组 | 跟踪[M]、[I]、[CTA]、[P·]、[Int]、λ0/λ1/λ2（活性+死链）。约10-15个变量 | ✓ |
| 简化矩量方程 | 稳态假设减少变量数 | |

**User's choice:** 完整矩量方程组
**Notes:** Claude推荐完整版——简化版会消除inhibition和retardation信号，使2/3的预测目标无法学习

---

## 三参数输出定义

### Inhibition Period定义

| Option | Description | Selected |
|--------|-------------|----------|
| 时间刻度（t_inh, 秒） | 绝对时间，依赖于kd/kp等绝对速率常数 | |
| 无量纲比值（推荐） | t_inh/t_total，范围0-1，消除绝对时间尺度影响 | ✓ |

**User's choice:** 无量纲比值

### Retardation Factor定义

| Option | Description | Selected |
|--------|-------------|----------|
| 速率比值（推荐） | Rp(RAFT)/Rp(no CTA)，范围0-1 | ✓ |
| 转化率点延迟比 | 在特定转化率点计算时间延迟比 | |

**User's choice:** 速率比值

### Ctr输出尺度

| Option | Description | Selected |
|--------|-------------|----------|
| log10(Ctr)（推荐） | 与ViT-RR一致，适合跨数量级参数 | ✓ |
| 线性Ctr | 简单但小值被忽略 | |

**User's choice:** log10(Ctr)

---

## 参数空间设计

### Ctr范围

| Option | Description | Selected |
|--------|-------------|----------|
| 宽范围（推荐） | 0.01-10000 (log10: -2到4)，覆盖全部RAFT剂类型 | ✓ |
| 中等范围 | 0.1-1000，排除极端值 | |

**User's choice:** 宽范围

### [CTA]/[M]范围

| Option | Description | Selected |
|--------|-------------|----------|
| 0.001-0.1（推荐） | 对数均匀采样，覆盖实验常用范围 | ✓ |
| 0.0001-0.5 | 更宽，包含极端条件 | |

**User's choice:** 0.001-0.1

### 温度处理

| Option | Description | Selected |
|--------|-------------|----------|
| 含温度变量 | 通过Arrhenius方程调节速率常数 | |
| 固定温度（推荐） | Ctr已隐含温度影响 | ✓ |

**User's choice:** 固定温度

### 样本分配

| Option | Description | Selected |
|--------|-------------|----------|
| 均衡采样（推荐） | 四种类型各~250K | ✓ |
| 按使用频率采样 | trithiocarbonate最多 | |

**User's choice:** 均衡采样

---

## ODE验证策略

| Option | Description | Selected |
|--------|-------------|----------|
| 文献曲线对比（推荐） | 3-4篇文献的Mn/Đ-conversion曲线 | |
| 极限行为校验 | 极端参数下检查理论极限 | |
| 两者结合（最佳） | 先极限行为，再文献对比 | ✓ |

**User's choice:** 两者结合

---

## Claude's Discretion

- ODE求解器配置（Radau参数、容差）
- ctFP编码归一化策略
- 诊断数据集参数网格

## Deferred Ideas

None
