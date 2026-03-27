# Phase 3: Model Training and Evaluation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-03-27
**Phase:** 03-model-training-and-evaluation
**Areas discussed:** 训练策略, 损失函数设计, 数据划分策略, Bootstrap实现方案, 评估指标体系

---

## 训练策略

| Option | Description | Selected |
|--------|-------------|----------|
| 本地MX350训练 | 2GB显存,适合小规模调试 | |
| Colab T4训练(推荐) | 16GB显存,免费,适合完整1M样本训练 | ✓ |
| 混合方案 | 本地快速原型验证,Colab正式训练 | |

**User's choice:** Colab T4训练,本地调试
**Notes:**
- Batch size = 64
- 学习率3e-4,使用ReduceLROnPlateau调度器
- 早停patience=15 epochs
- 基础模型训练epochs由早停自动决定

---

## 损失函数设计

用户要求深入讨论四个方案后选择。

### 方案对比

| Option | Description | Selected |
|--------|-------------|----------|
| 方案A: 等权重MSE | 简单但尺度不平衡,Ctr会主导loss | |
| 方案B: 归一化后等权重 | 用std归一化,三个输出贡献相当 | |
| 方案C: 手动加权 | 灵活控制优先级,w_ctr=2.0, w_inh=0.5, w_ret=0.5 | ✓ |
| 方案D: 可学习权重 | 自动平衡但增加复杂度,理论基础强 | |

**User's choice:** 方案C手动加权
**Notes:**
- 权重: w_ctr=2.0, w_inh=0.5, w_ret=0.5
- 理由: 强调Ctr为核心预测目标,符合"Core Value"叙事
- 损失公式: `loss = 2.0*MSE(pred_ctr, true_ctr) + 0.5*MSE(pred_inh, true_inh) + 0.5*MSE(pred_ret, true_ret)`

**深入讨论要点:**
- 方案A的尺度不平衡问题: log10(Ctr)范围-2到4,而inhibition/retardation是0-1,会导致模型过度优化Ctr
- 方案B的隐含假设: 三个输出"同等重要",但实际Ctr是论文核心
- 方案C的优势: 明确表达优先级,符合论文定位
- 方案D的复杂度: 增加3个可训练参数,收敛可能变慢,可解释性下降

---

## 数据划分策略

| Option | Description | Selected |
|--------|-------------|----------|
| 方案A: 简单随机划分 | 80/10/10完全随机,实现简单 | |
| 方案B: 按Ctr范围分层 | log10(Ctr)每0.5一档,每档内80/10/10划分 | ✓ |
| 方案C: 按RAFT类型分层 | 四种类型各自独立划分 | |
| 方案D: Ctr+RAFT双重分层 | 同时保证Ctr和类型均衡 | |

**User's choice:** 方案B按Ctr范围分层
**Notes:**
- 划分比例: 80/10/10
- log10(Ctr)每0.5一档,共12档(-2.0到4.0)
- 保证测试集覆盖全Ctr范围

---

## Bootstrap实现方案

用户要求深入讨论Bootstrap原理和四个方案后选择。

### Bootstrap原理说明
- Bootstrap(自助法)是统计重采样方法,用于估计预测不确定性
- 核心思想: 有放回抽样模拟"多个训练集",训练多个模型,预测分布的分散程度反映不确定性
- 置信区间: 用协方差矩阵和F分布计算95%联合置信区间(JCI)

### 方案对比

| Option | Description | Selected |
|--------|-------------|----------|
| 方案A: 完整模型重训练 | 200次完整训练,理论最严格,成本~400小时 | |
| 方案B: MC Dropout | 训练一次,推理时Dropout开启200次前向传播,成本极低 | |
| 方案C: 轻量Bootstrap | 冻结backbone,200次输出头微调,成本~22小时 | ✓ |
| 方案D: 混合方案 | 5-10个完整模型+MC Dropout,复杂度高 | |

**User's choice:** 方案C轻量Bootstrap
**Notes:**
- 训练一个基础模型后冻结backbone
- 200次迭代,每次有放回抽样后微调输出头5 epochs(固定)
- F分布计算95% JCI
- 在验证集上事后校准,确保实际覆盖率达到95%

**深入讨论要点:**
- 方案A优点: 理论最严格,无偏,可发表性强; 缺点: 成本极高(400小时),Colab免费版需手动重启~34次
- 方案B优点: 成本极低,实现简单; 缺点: 理论基础较弱,可能低估不确定性,审稿人可能质疑
- 方案C优点: 成本中等(~22小时),保留数据随机性; 缺点: 不确定性可能被低估(冻结backbone)
- 方案D优点: 平衡成本和准确性; 缺点: 实现复杂,理论解释困难

---

## 评估指标体系

| Option | Description | Selected |
|--------|-------------|----------|
| 基础指标 | R²/RMSE/MAE per output | ✓ |
| 按Ctr范围分段 | 低/中/高三段评估 | ✓ |
| 按RAFT类型分类 | 四种类型各自指标+parity图 | ✓ |
| 残差分布分析 | 直方图,检查系统性偏差 | ✓ |
| 异常值分析 | 阈值2×std,统计比例和特征 | ✓ |

**User's choice:** 全部都要
**Notes:**
- 基础模型训练epochs由早停自动决定
- 输出头微调5 epochs固定
- 异常值阈值2×std

---

## Claude's Discretion

用户授权Claude决定的实现细节:
- DataLoader的num_workers配置
- ReduceLROnPlateau的具体参数
- 模型checkpoint保存策略
- HDF5数据加载的具体实现
- Parity图的可视化风格
- 残差直方图的bins数量
- 训练日志的记录频率和格式
