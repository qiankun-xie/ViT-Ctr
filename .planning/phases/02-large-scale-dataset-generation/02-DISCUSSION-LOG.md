# Phase 2: Large-Scale Dataset Generation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-03-25
**Phase:** 02-large-scale-dataset-generation
**Areas discussed:** 动力学参数采样策略, 噪声注入与鲁棒性, HDF5存储组织方式, 模拟失败处理策略

---

## 动力学参数采样策略

### Q1: kp/kt设定方式

| Option | Description | Selected |
|--------|-------------|----------|
| 随机范围采样（推荐） | kp和kt在合理范围内随机采样，让模型学到更通用的特征。数据集多样性高，但部分组合可能不物理。 | ✓ |
| 文献固定值（按单体类型） | 选几种代表性单体，用各自文献kp/kt值。物理上严格，但限制了模型泛化到未知单体的能力。 | |
| 文献值+扰动 | 以文献值为中心，在±50%范围内扰动。兼顾物理合理性和多样性。 | |

**User's choice:** 随机范围采样
**Notes:** 无额外说明

### Q2: 引发剂参数处理

| Option | Description | Selected |
|--------|-------------|----------|
| 同样随机采样（推荐） | kd和[I]₀也在合理范围内随机采样，与kp/kt策略一致。 | ✓ |
| 固定典型值 | kd固定为AIBN典型值，[I]₀固定为0.01M。减少变量维度。 | |
| kd固定，[I]₀随机 | kd固定（AIBN典型），但[I]₀在常用范围内随机采样。中间方案。 | |

**User's choice:** 同样随机采样
**Notes:** 无额外说明

### Q3: 多参数联合采样方式

| Option | Description | Selected |
|--------|-------------|----------|
| 独立随机采样 | 每个参数独立在其范围内对数均匀随机采样。实现简单，但可能产生不物理的参数组合。 | |
| Latin Hypercube（推荐） | 拉丁超立方体采样，确保参数空间均匀覆盖，避免随机采样的聚集问题。稍复杂但覆盖更好。 | ✓ |
| 网格采样 | 在每个参数维度上等距取点，生成完整网格。覆盖确定但维度诅咒严重（6+个参数无法生成合理网格）。 | |

**User's choice:** Latin Hypercube
**Notes:** 无额外说明

---

## 噪声注入与鲁棒性

### Q1: 是否注入噪声

| Option | Description | Selected |
|--------|-------------|----------|
| 注入噪声（推荐） | 每个样本的Mn和Đ值添加高斯噪声（如±5%相对误差），模拟实验测量不确定性。提升模型对真实数据的鲁棒性。 | ✓ |
| 不注入噪声 | 保持ODE输出的干净数据。简单直接，但模型可能对实验噪声敏感。 | |
| 两份数据对比 | 生成两份数据：一份干净、一份加噪。训练时可对比效果，但存储量翻倍。 | |

**User's choice:** 注入噪声
**Notes:** 无额外说明

### Q2: 噪声类型

| Option | Description | Selected |
|--------|-------------|----------|
| 乘性高斯噪声（推荐） | 每个数据点的Mn和Đ分别乘以(1+ε)，ε~N(0, σ²)，σ=0.02-0.05。符合GPC实验误差特征。 | ✓ |
| 加性高斯噪声 | 每个点加上固定标准差的高斯噪声。简单但不符合实际误差分布。 | |
| 随机噪声水平 | 噪声水平也作为随机变量，每个样本的σ在一定范围内随机采样。让模型适应不同精度的实验数据。 | |

**User's choice:** 乘性高斯噪声
**Notes:** 无额外说明

---

## HDF5存储组织方式

### Q1: 文件结构

| Option | Description | Selected |
|--------|-------------|----------|
| 单个大文件 | 所有样本存入一个HDF5文件，按chunk读取。管理简单，但单文件32GB上传Google Drive可能不稳定。 | |
| 按RAFT类型分文件（推荐） | 每种RAFT剂类型一个HDF5文件（~250K样本/文件，~8GB/文件）。便于分步生成和上传，也方便按类型调试。 | ✓ |
| 固定大小分片 | 分成10-20个小文件（各~50-100K样本），不按类型而按序号。灵活但管理复杂。 | |

**User's choice:** 按RAFT类型分文件
**Notes:** 无额外说明

### Q2: chunk大小

| Option | Description | Selected |
|--------|-------------|----------|
| 1000样本/chunk（推荐） | 读取粒度适中，兼顾I/O效率和内存占用。 | ✓ |
| 256样本/chunk | 与常见batch size匹配。chunk数多但读取更灵活。 | |
| 你决定 | 由Claude根据实际存储和读取性能测试决定。 | |

**User's choice:** 1000样本/chunk
**Notes:** 无额外说明

---

## 模拟失败处理策略

### Q1: 失败处理方式

| Option | Description | Selected |
|--------|-------------|----------|
| 跳过+记录（推荐） | 失败的参数组合直接跳过，记录失败率和参数分布。只要失败率<5%且分布无明显偏斜即可接受。 | ✓ |
| 扰动重试 | 失败后用微扰动的参数重试（最多3次）。提高成功率但增加计算时间。 | |
| 替换采样 | 失败后重新从LHS中采样一个替代参数组合。保证最终样本数精确达标，但LHS均匀性会稍有破坏。 | |

**User's choice:** 跳过+记录
**Notes:** 无额外说明

### Q2: 失败率阈值

| Option | Description | Selected |
|--------|-------------|----------|
| 5%阈值（推荐） | 失败率超过5%则报警暂停，需检查ODE或参数范围是否有问题。 | ✓ |
| 1%阈值 | 更严格，失败>1%就报警。确保数据集质量但可能需要更多参数调整。 | |
| 你决定 | 完全交给Claude根据实际运行情况判断。 | |

**User's choice:** 5%阈值
**Notes:** 无额外说明

---

## Claude's Discretion

- joblib并行度配置
- ODE求解器容差和超时
- LHS实现库选择
- 噪声σ具体值（0.02-0.05范围内）
- HDF5压缩方案
- 进度报告和日志格式
- Google Drive上传方式

## Deferred Ideas

None — discussion stayed within phase scope
