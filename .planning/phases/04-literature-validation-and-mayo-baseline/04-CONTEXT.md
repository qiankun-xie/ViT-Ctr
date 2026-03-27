# Phase 4: Literature Validation and Mayo Baseline - Context

**Gathered:** 2026-03-27
**Status:** Ready for planning

<domain>
## Phase Boundary

在训练好的 SimpViT 模型基础上，系统检索 10+ 篇文献的已发表 Ctr 实验值（覆盖四类 RAFT 剂），实现 ODE 曲线拟合形式的 Mayo 方程基线，在相同模拟数据上对两种方法进行公平比较，报告 fold-error 等汇总指标，并产出论文级验证图（Log-log parity 图，ML+Mayo 同图双标记）。

</domain>

<decisions>
## Implementation Decisions

### A. 文献数据收集策略
- **D-01:** 需系统检索，无预设论文——phase-researcher 负责搜集覆盖四类 RAFT 剂的已发表 Ctr 文献
- **D-02:** 四类 RAFT 剂均须覆盖（dithioester、trithiocarbonate、xanthate、dithiocarbamate），每类至少 2-3 个数据点，总计 ≥10 个文献 Ctr 值
- **D-03:** 三种测量方法（Mayo 法、CLD 法、分散度法）的 Ctr 值均纳入验证集；最终按测量方法分组报告 fold-error，让读者评判测量方法是否影响验证结果
- **D-04:** 每个文献 Ctr 值须标注：RAFT 剂类型、测量方法、单体、溶剂、温度（对应 VAL-02）

### B. Mayo 方程基线实现
- **D-05:** Mayo 基线 = ODE 曲线拟合（传统分析学家方式）：输入为单次实验的 Mn-conversion 曲线；用 scipy.optimize 对 Ctr 进行单参数优化，最小化模拟 Mn 与"实验" Mn 的 MSE
- **D-06:** ODE 拟合时，其他动力学参数（kp、kt、kd、[I]₀、f 等）固定为典型局平均值（与数据集生成时的采样中心值一致），仅 Ctr 作为自由参数——严格等价于传统分析学家"已知反应体系参数，只拟合 Ctr"的操作
- **D-07:** 对比数据来源：对每个文献条件（已知真实 Ctr、RAFT 类型、[CTA]/[M] 等），用 ODE 模拟器生成模拟 Mn-conversion 曲线（加噪声），ML 和 Mayo 均在此同一模拟数据上运行——消除真实实验数据异质性的干扰，确保公平对比

### C. Fold-error 计算与比较框架
- **D-08:** 同时报告两种 fold-error 定义：
  - 对数空间：`fold_log = 10^|log10(Ctr_pred) - log10(Ctr_true)|`
  - 比值空间：`fold_ratio = max(Ctr_pred/Ctr_true, Ctr_true/Ctr_pred)`（二者等价，均报告以满足不同读者习惯）
- **D-09:** 汇总指标（ML 和 Mayo 分别报告）：中位 fold-error、2× 阈内占比（%）、10× 阈内占比（%）、RMSE(log10(Ctr))、R²
- **D-10:** ML vs Mayo 对比展示形式：表格（每行一个文献点，ML/Mayo fold-error 并列）+ 散点图（x 轴 = log10(Ctr_true)，y 轴 = fold-error，ML 和 Mayo 用不同颜色/标记）

### D. 验证图设计
- **D-11:** 主图：Log-log parity 图（x = 发布 Ctr，y = 预测 Ctr）；点按 RAFT 剂类型上色（4 色）；误差棒 = ML Bootstrap 95% CI（仅 Y 轴，X 轴为 ground truth 无误差棒）；对角线 = 完美预测；±2 倍误差带为虚线
- **D-12:** ML 和 Mayo 同在一张 parity 图上——ML 用实心圆，Mayo 用空心菱形/正方形，同一文献点的两个预测值用细灰线连接，直观展示方法差异
- **D-13:** 不需要按 RAFT 类型分组的子图；类型差异通过颜色区分在主图展示，按类型的 fold-error 统计放在表格中（对应 VAL-03）

### Claude's Discretion
- scipy.optimize 的具体优化器选择（minimize / curve_fit）和容差设置
- 模拟数据的噪声水平（建议与数据集生成时一致，σ=0.03）
- 文献条件中缺失参数的填充策略（如文献未报告 kp，取典型范围均值）
- parity 图的具体配色方案（4 种 RAFT 类型的颜色选择）
- 表格格式（HTML/Markdown/LaTeX）——根据论文最终格式决定

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### 参考项目
- `C:/CodingCraft/DL/ViT-RR/model_utils.py` — SimpViT 架构和 transform() 编码函数；推理时直接调用
- `C:/CodingCraft/DL/ViT-RR/deploy.py` — Bootstrap 推理模式和 F 分布置信区间计算；Phase 4 推理时复用此模式

### Phase 1-3 上下文（必读基础决策）
- `.planning/phases/01-ode-foundation-and-ctfp-encoder/01-CONTEXT.md` — ODE 变量定义、三参数输出定义（D-04~D-06）、Ctr 的 log10 尺度
- `.planning/phases/02-large-scale-dataset-generation/02-CONTEXT.md` — 数据集组织方式、动力学参数典型值范围（Mayo 拟合时固定参数的参考来源）
- `.planning/phases/03-model-training-and-evaluation/03-CONTEXT.md` — Bootstrap CI 实现方式（D-12~D-16）

### 现有代码（Phase 4 直接依赖）
- `src/raft_ode.py` — ODE 模拟器，Phase 4 用于：(1) 按文献条件生成模拟验证数据；(2) Mayo ODE 拟合的正向模拟
- `src/ctfp_encoder.py` — ctFP 编码器，Phase 4 用于将模拟验证数据编码为 ML 输入
- `src/evaluate.py` — 现有评估函数（R²、RMSE、MAE），Phase 4 复用计算 fold-error 汇总指标
- `src/bootstrap.py` — Bootstrap 推理，Phase 4 用于获取验证数据的 95% CI

### 需求映射
- `.planning/REQUIREMENTS.md` §VAL-01, VAL-02, VAL-03 — 文献验证集要求（10+ 点、标注条件、fold-error 报告）
- `.planning/REQUIREMENTS.md` §EVL-04 — Mayo 方程基线实现要求

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/raft_ode.py` 的 `simulate()` 函数：Phase 4 需按文献条件调用此函数生成"虚拟实验数据"，无需修改 ODE 本身
- `src/evaluate.py` 的 `compute_test_metrics()`：R²/RMSE/MAE 计算逻辑可直接复用；fold-error 为新增计算
- `src/bootstrap.py`：Bootstrap 推理已实现，Phase 4 复用输出 Bootstrap 95% CI 用于 parity 图误差棒
- `src/ctfp_encoder.py` 的 `transform()`：将模拟验证数据编码为 ML 输入（与训练时编码完全一致）

### Established Patterns
- ODE 模拟 → ctFP 编码 → 模型推理 → Bootstrap CI：整套流程在 Phase 3 已建立，Phase 4 重复使用
- scipy.optimize 在 Phase 1 中已用于 ODE 参数验证；Mayo 拟合的优化器选择与此一致
- tqdm 进度条、JSON 结果记录已成为项目惯例

### Integration Points
- Phase 3 产出的 `checkpoints/best_model.pth` 和 `checkpoints/bootstrap_heads.pth` 是 Phase 4 的直接输入
- Phase 4 产出的验证图和指标表将直接用于 Phase 6 论文的 Results/Discussion 章节
- 文献验证集（Excel/CSV 格式）作为新增数据文件，建议存放在 `data/literature/` 目录

</code_context>

<specifics>
## Specific Ideas

- **论文叙事锚点**：ML 从单次实验的 ctFP 直接预测 Ctr，而传统 Mayo 分析同样可在单次实验数据上做 ODE 拟合——Phase 4 的比较正是在这个"公平竞技场"上展示 ML 的速度和准确度优势（无需手动调参，自动化推理）
- **公平对比设计的理由**：两种方法均在相同模拟数据上运行，而非真实实验数据，这消除了"真实数据质量不一"的混淆因素，使比较完全聚焦于方法本身的 Ctr 提取能力
- **Mayo 基线的局限性说明**：在论文 Discussion 中可指出，真实场景中 Mayo ODE 拟合需要研究者手动选择合适的动力学参数，而本工作将此固定为均值；这实际上对 Mayo 方法是"有利"的（使用了理想参数），ML 仍能表现更好则更有说服力
- **Ctr 验证范围**：应覆盖 log10(Ctr) ≈ -1 到 4 全范围，避免仅在某个 Ctr 段验证导致泛化性存疑

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 04-literature-validation-and-mayo-baseline*
*Context gathered: 2026-03-27*
