# Phase 6: Paper and Supporting Information - Context

**Gathered:** 2026-04-06
**Status:** Ready for planning

<domain>
## Phase Boundary

撰写完整英文学术论文（IMRAD结构）和Supporting Information，目标投稿Macromolecules。整合Phase 1-5的全部验证图表、ODE推导和模型结果，产出LaTeX投稿版 + Word英文版 + Word中文版三份文件。Route A（SMILES → Ctr）仅作为展望章节。

</domain>

<decisions>
## Implementation Decisions

### A. 目标期刊与格式
- **D-01:** 目标期刊为 **Macromolecules**（ACS），篇幅10-15页，正文6张图
- **D-02:** LaTeX使用achemso模板，生成可直接投稿的.tex文件
- **D-03:** 同时产出Word英文版(.docx)和Word中文版(.docx)，共三份输出文件

### B. 叙事角度与故事线
- **D-04:** 核心卖点："一次实验提取三参数"（实用价值驱动），而非"ML vs 传统方法"
- **D-05:** 叙事结构：Hook（传统需三组独立实验，我们一次搞定）→ 方法（ODE合成数据 + ViT + Bootstrap UQ）→ 验证（ML vs Mayo作为证据而非卖点）→ 落地（Web工具免费开放）
- **D-06:** ViT-RR作为背景引用，不作为叙事主线，避免被视为增量工作

### C. 图表规划
- **D-07:** 正文6张图：
  - Figure 1: 概念图/流程图（实验数据 → ctFP编码 → SimpViT → 三参数+CI）
  - Figure 2: ctFP示例图（双通道热力图，展示编码效果）
  - Figure 3: 三参数parity图（3合1复合图）
  - Figure 4: ML vs Mayo验证图（77个文献点）
  - Figure 5: 训练曲线（loss vs epoch）
  - Figure 6: 代表性的按RAFT类型分类结果
- **D-08:** SI图表：残差图（3张）、完整12张按RAFT类型分类parity图、其他补充图表
- **D-09:** 需制作TOC graphic（~3.25×1.75 in），纳入Phase 6范围

### D. 写作流程与分工
- **D-10:** Claude起草各章节初稿，用户逐章节审阅修改，迭代至满意
- **D-11:** 逐章节审核节奏：每完成一个章节停下来让用户审阅，确认后再写下一个

### E. SI结构与深度
- **D-12:** ODE推导写到**可复现级**：完整矩量方程组（~14个ODE变量），区分single-eq和pre-eq模型，方程与raft_ode.py代码一一对应
- **D-13:** SI包含以下全部内容：
  - 完整ODE推导
  - 数据集构建细节（规模、参数范围、LHS采样、失败率等）
  - 训练超参数表（batch size、学习率、损失权重、早停策略等）
  - Bootstrap UQ详细流程（200头微调、F分布JCI、事后校准因子）
  - 完整评估图表（12张分类parity图 + 3张残差图）

### F. 开源与数据共享
- **D-14:** 全部开源：代码（GitHub repo）、模型权重、文献验证集，配合Zenodo DOI

### G. 局限性讨论
- **D-15:** Discussion中坦诚讨论以下四个局限性：
  1. 模拟数据 vs 真实数据差距（ODE未捕获的副反应、非理想混合等）
  2. Retardation预测的适用范围（TTC/xanthate/dithiocarbamate中retardation≈1.0，预测价值有限）
  3. 参数范围边界（log10(Ctr)∈[-2,4]、[CTA]/[M]∈[0.001,0.1]，超出范围的外推风险）
  4. 未覆盖的实验条件（温度固定、未考虑溶剂效应、仅四种RAFT剂类型）

### Claude's Discretion
- 概念图/流程图的具体设计和排版
- TOC graphic的视觉设计
- ctFP示例图的具体样本选择
- Figure 6中选择哪种RAFT类型作为代表
- LaTeX排版细节（字体、间距、图表位置）
- Word版本的排版格式
- 各章节的具体段落结构和过渡语
- 参考文献的具体格式化

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### 参考项目
- `C:/CodingCraft/DL/ViT-RR/` — ViT-RR完整项目，叙事和方法论的直接参考（Angew. Chem. Int. Ed. 2025, DOI: 10.1002/anie.202513086）

### 现有代码（论文方法描述的依据）
- `src/raft_ode.py` — RAFT动力学ODE模拟器，SI中ODE推导必须与此代码匹配（PAP-02）
- `src/ctfp_encoder.py` — ctFP编码器，Methods中描述编码流程的依据
- `src/model.py` — SimpViT模型定义（num_outputs=3），Methods中模型架构描述的依据
- `src/bootstrap.py` — Bootstrap UQ实现，Methods/SI中不确定性量化描述的依据
- `src/literature_validation.py` — 文献验证和Mayo基线实现，Results中验证结果的依据
- `src/evaluate.py` — 评估指标计算，Results中性能报告的依据

### 现有图表（论文直接使用）
- `figures/parity_log10_Ctr.png` — Ctr parity图
- `figures/parity_inhibition_period.png` — Inhibition period parity图
- `figures/parity_retardation_factor.png` — Retardation factor parity图
- `figures/residuals_*.png` — 三参数残差图（SI用）
- `figures/parity_by_class/*.png` — 12张按RAFT类型分类parity图（SI用）
- `figures/validation/parity_ml_vs_mayo.png` — ML vs Mayo验证图
- `figures/validation/validation_results.csv` — 77点验证详细数据
- `figures/validation/validation_summary.json` — 验证汇总指标
- `checkpoints/loss_curves.png` — 训练曲线

### 现有数据（论文数据来源）
- `data/literature/literature_ctr.csv` — 77个文献Ctr验证点
- `checkpoints/training_log.json` — 训练日志
- `checkpoints/calibration.json` — Bootstrap校准因子
- `checkpoints/bootstrap_summary.json` — Bootstrap统计摘要

### Phase 1-5 上下文（论文各章节内容来源）
- `.planning/phases/01-ode-foundation-and-ctfp-encoder/01-CONTEXT.md` — ODE变量定义、三参数输出定义（D-04~D-06）
- `.planning/phases/02-large-scale-dataset-generation/02-CONTEXT.md` — 数据集构建策略、参数采样范围
- `.planning/phases/03-model-training-and-evaluation/03-CONTEXT.md` — 训练策略、损失函数、Bootstrap UQ设计
- `.planning/phases/04-literature-validation-and-mayo-baseline/04-CONTEXT.md` — 文献验证策略、Mayo基线设计、fold-error框架
- `.planning/phases/05-streamlit-web-application/05-CONTEXT.md` — Web应用设计（论文中提及工具可用性）

### 需求映射
- `.planning/REQUIREMENTS.md` §PAP-01 — 英文论文正文（IMRAD）
- `.planning/REQUIREMENTS.md` §PAP-02 — SI（ODE推导与代码匹配）
- `.planning/REQUIREMENTS.md` §PAP-03 — Route A作为展望章节

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/raft_ode.py` — SI中ODE推导的"ground truth"，方程必须与代码逐行对应
- `src/evaluate.py` — 性能指标（R²、RMSE、MAE）的计算逻辑，Results章节数据来源
- `src/literature_validation.py` — fold-error计算和Mayo基线实现，验证结果的数据来源
- `figures/` 目录下全部图表 — 大部分可直接用于论文，部分需要合成为复合图

### Established Patterns
- 所有评估指标已在Phase 3-4中计算完毕，论文只需引用结果
- 图表已生成为PNG格式，LaTeX中用\includegraphics引入
- 验证数据以CSV/JSON格式存储，可直接读取生成LaTeX表格

### Integration Points
- Phase 4产出的验证图和指标表直接用于Results/Discussion
- Phase 5的Web应用URL将在论文中引用
- GitHub repo和Zenodo DOI需要在投稿前创建

</code_context>

<specifics>
## Specific Ideas

- 叙事锚点："传统方法需要三组独立实验分别测定Ctr、诱导期和减速因子，我们展示一张ctFP指纹图已包含全部三个信号"
- Mayo对比的定位：作为验证证据（"ML在相同条件下优于传统ODE拟合"），而非论文卖点
- Mayo基线的局限性说明：论文Discussion中指出，真实场景中Mayo ODE拟合需要研究者手动选择动力学参数，本工作将此固定为均值——这对Mayo方法是"有利"的，ML仍能表现更好则更有说服力
- Retardation factor的诚实表述：在Discussion中明确说明retardation≈1.0的RAFT类型（TTC、xanthate、dithiocarbamate）中该参数预测价值有限，三参数同时预测的claim需限定在dithioester等有显著retardation的体系
- Web工具作为亮点：Macromolecules重视应用价值，论文中提及免费Web工具和开源代码是加分项

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 06-paper-and-supporting-information*
*Context gathered: 2026-04-06*
