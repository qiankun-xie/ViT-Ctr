# Phase 4: Literature Validation and Mayo Baseline - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-03-27
**Phase:** 04-literature-validation-and-mayo-baseline
**Areas discussed:** 文献数据收集策略, Mayo方程基线实现, Fold-error计算与比较框架, 验证图设计

---

## A. 文献数据收集策略

| Option | Description | Selected |
|--------|-------------|----------|
| 有具体论文 | 已有几篇关键文章，可直接提供引用 | |
| 需要系统检索 | 没有具体论文，需要做文献搜索，覆盖各类RAFT剂的Ctr发表值 | ✓ |
| 两者结合 | 有几篇已知论文，但还需要补充搜索来凑够10+个点 | |

**User's choice:** 需要系统检索

---

| Option | Description | Selected |
|--------|-------------|----------|
| 四类全覆盖 | Dithioester、Trithiocarbonate、Xanthate、Dithiocarbamate 各至少2个数据点，论文说服力最强 | ✓ |
| Dithioester+TTC优先 | 这两类文献数据最丰富，Ctr分布也最广，凑够10+个点更容易 | |
| 不限类型，优先覆盖Ctr幅度 | 为论文的Ctr范围覆盖向其次元素，不强求每类都有 | |

**User's choice:** 四类全覆盖

---

| Option | Description | Selected |
|--------|-------------|----------|
| 全部纳入，按方法分组 | 三种方法的已发表Ctr均直接纳入验证集，最终按测量方法分组报告误差 | ✓ |
| 仅纳入Mayo法测定的Ctr | Mayo法被认为是最可靠的基准，其他方法测Ctr得到的值可能偏差大 | |
| 全部纳入，标注方法但不分层 | 每个数据点标注测量方法，论文里分组展示，读者自己判断方法有无影响模型精度 | |

**User's choice:** 全部纳入，按方法分组

---

## B. Mayo方程基线实现

| Option | Description | Selected |
|--------|-------------|----------|
| ODE曲线拟合（单次实验） | 对每个文献条件，用ODE模拟器生成模拟数据，然后对同一张ctFP用曲线拟合（优化Ctr使模拟曲线拟合实验数据）提取Ctr——这个才是传统分析学家的使用方式 | ✓ |
| 简化解析公式 | 第1个数据点：Mn ≈ [M]₀/[CTA]₀ × conversion × Mm，简单求解Ctr——计算鲁棒但论文中能说明传统方法的局限性 | |
| 真实多组实验Mayo图（模拟） | 每个文献条件用ODE模拟器生成不同[CTA]/[M]的多组数据，做传统Mayo图提取Ctr——更忠于传统方法，但语境是模拟 vs 模拟 | |

**User's choice:** ODE曲线拟合（单次实验）

---

| Option | Description | Selected |
|--------|-------------|----------|
| 仅Mn-conversion | 拟合目标：Mn-conversion曲线（不用Đ），这也是实验家最常用的方式 | ✓ |
| Mn+Đ同时拟合 | 同时拟合Mn-conversion和Đ-conversion，与ML模型输入完全一致，对比更公平 | |
| 仅Đ-conversion | 仅用Đ-conversion，通过分散度演变提取Ctr（分散度法的一种变体） | |

**User's choice:** 仅Mn-conversion

---

| Option | Description | Selected |
|--------|-------------|----------|
| 仅优化Ctr一个参数 | 其他动力学参数（kp、kt等）由文献提供或固定为典型局均值，仅对Ctr单一参数优化 | ✓ |
| Ctr + [CTA]₀/[M]₀ | 同时优化Ctr和[CTA]/[M]，这两个参数最影响曲线形状 | |
| 多参数优化 | Ctr、kadd/kfrag、引发剂效率等多参数同时优化，更灵活但可能欠拟合 | |

**User's choice:** 仅优化Ctr一个参数

---

| Option | Description | Selected |
|--------|-------------|----------|
| 相同模拟数据（公平对比） | 大部分文献点不会提供Mn-conversion原始数据，需要用ODE模拟器按文献条件生成"假实验"数据，最终两个方法都在模拟数据上跑，对比真Ctr | ✓ |
| 真实实验数据 | 文献读出原始实验数据，直接进行拟合（需要文献提供足够少的数据点） | |

**User's choice:** 相同模拟数据（公平对比）

---

## C. Fold-error计算与比较框架

| Option | Description | Selected |
|--------|-------------|----------|
| 10^\|对数差\|（对数空间） | fold_error = 10^|log10(Ctr_pred) - log10(Ctr_true)|，和Ctr的量级计算一致，也是药物化学领域标准定义 | |
| max(比值, 1/比值)（比值空间） | fold_error = max(Ctr_pred/Ctr_true, Ctr_true/Ctr_pred)，表达更直观但和上面公式等价 | |
| 两种均报告 | 两种定义均报告，一种用于指标表，一种用于图表可视化 | ✓ |

**User's choice:** 两种均报告

---

| Option | Description | Selected |
|--------|-------------|----------|
| 中位+占比 | 中位 fold-error + 在冒险范围内的展示比例（2倍阈内占%，10倍阈内占%），论文里常见且易解读 | |
| RMSE(log10) + R² | 平均对数误差（RMSE in log-space）+ R²，和Phase 3评估指标保持一致 | |
| 全部报告 | 两者全报告：中位fold-error、占比和 RMSE(log10) + R² | ✓ |

**User's choice:** 全部报告

---

| Option | Description | Selected |
|--------|-------------|----------|
| 表格对比 | 并排两列表格：一列ML fold-error，一列Mayo fold-error，最后一行报告中位和2倍阈内占比（数据点少时最清晰） | |
| 散点图（Ctr vs fold-error） | 第X轴=Ctr真实值，第Y轴=fold-error，ML和Mayo用不同颜色点对应，直观展示两种方法在各Ctr范围的误差 | |
| 表格+图 | 两者全展示：表格 + 散点图 | ✓ |

**User's choice:** 表格+图

---

## D. 验证图设计

| Option | Description | Selected |
|--------|-------------|----------|
| Log-log parity图 | 对数-对数尺度的parity图，点用RAFT剂类型上色，带误差棒（bootstrap 95% CI），画对角线和±2倍蓝色带——和Phase 3的模式一致 | ✓ |
| 残差图 | 残差图（log10(Ctr_pred/Ctr_true)），结合统计指标标注底部 | |
| Parity图+残差图 | 两张图同时展示 | |

**User's choice:** Log-log parity图

---

| Option | Description | Selected |
|--------|-------------|----------|
| ML的 Bootstrap CI | 误差棒 = ML模型Bootstrap 95% CI，文献发表值视为 ground truth（参照ViT-RR的展示方式） | ✓ |
| 无误差棒 | 不画误差棒，点直接展示，指标标注在图题/注释中 | |
| 仅Y轴画 CI | 两个方向都画：X轴无误差棒（ground truth），Y轴画 ML的 Bootstrap CI | |

**User's choice:** ML的 Bootstrap CI
**Notes:** 仅Y轴画CI（X轴为ground truth，不画误差棒）

---

| Option | Description | Selected |
|--------|-------------|----------|
| 不分小图 | 不需要。文献验证数据点数量少（每类可能只有3-5个点），小图意义不大，在主图用颜色区分RAFT类型即可 | ✓ |
| 需要分小图 | 即使每类只有3-5个点，展示每类验证覆盖也是一种责任感 | |
| 分类指标展示在表格中 | 指标表里按RAFT类型列出每类的fold-error平均就够，不画分小图 | |

**User's choice:** 不分小图

---

| Option | Description | Selected |
|--------|-------------|----------|
| 同图双方法 | 同一张parity图上，ML和Mayo分别用不同形状/边框的点，连线连接同一文献点的两个预测值展示比较 | ✓ |
| 并排两张独立parity图 | 左图=ML，右图=Mayo | |
| 只展示ML parity图 | 只展示ML的parity图，Mayo的比较放在表格里即可 | |

**User's choice:** 同图双方法

---

## Claude's Discretion

- scipy.optimize 的具体优化器选择和容差设置
- 模拟数据的噪声水平
- 文献条件中缺失参数的填充策略
- parity 图的配色方案
- 表格格式（HTML/Markdown/LaTeX）

## Deferred Ideas

None — discussion stayed within phase scope
