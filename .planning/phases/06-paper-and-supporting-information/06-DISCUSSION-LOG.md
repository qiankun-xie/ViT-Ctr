# Phase 6: Paper and Supporting Information - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-06
**Phase:** 06-paper-and-supporting-information
**Areas discussed:** 目标期刊与格式, 叙事角度与故事线, 图表规划（正文vs SI）, 写作流程与分工, SI结构与深度, 开源与数据共享, 局限性讨论

---

## 目标期刊与格式

| Option | Description | Selected |
|--------|-------------|----------|
| Angew. Chem. Int. Ed. | 跟ViT-RR同刊，影响因子高，篇幅紧凑（Communication~4页 / Full Paper~8页），强调新颖性 | |
| Macromolecules | 高分子化学领域核心期刊，篇幅充裕（10-15页），可详细展开方法和结果 | ✓ |
| Polym. Chem. (RSC) | 高分子快通讯，篇幅适中（6-8页），审稿较快，方法类论文常见 | |

**User's choice:** Macromolecules
**Notes:** 用户询问Claude建议后采纳。理由：内容体量大（ODE推导+百万级数据集+模型训练+文献验证+Mayo对比+Web应用），10-15页篇幅能充分展开；受众精准（RAFT化学家最常读）；新颖性定位合适（对Macromolecules来说"首次用DL从单次RAFT实验同时提取三参数"是扎实的方法创新）。

---

## 叙事角度与故事线

| Option | Description | Selected |
|--------|-------------|----------|
| 一次实验提三参数（实用价值） | 开头强调"传统方法需要三组独立实验，我们一次搞定"——实用价值驱动，吸引实验化学家 | ✓ |
| ViT-RR范式拓展（方法传承） | 先讲ViT-RR成功预测反应性比，再说"我们将这个范式拓展到RAFT链转移常数"——学术传承线 | |
| ML vs 传统方法（技术对比） | 先说传统Mayo法的局限（需假设理想动力学、人工调参），再展示ML自动化+更准确——方法对比驱动 | |

**User's choice:** 一次实验提三参数（实用价值）
**Notes:** 用户询问Claude建议后采纳。ML vs Mayo作为验证证据而非卖点；ViT-RR作为背景引用不作为主线，避免被视为增量工作。

---

## 图表规划（正文 vs SI）

### 正文图表数量

| Option | Description | Selected |
|--------|-------------|----------|
| 精简（4张） | ~4张正文图，其余全部放SI | |
| 标准（6张） | ~6张正文图（概念图 + 3张parity + ML vs Mayo + 训练曲线），残差和分类放SI | ✓ |
| 充分（8张） | ~8张正文图，包含残差分析和部分分类图，SI只放ODE推导等补充材料 | |

**User's choice:** 标准（6张）

### 概念图

| Option | Description | Selected |
|--------|-------------|----------|
| 需要概念图/流程图 | 单张概念图展示全流程：实验数据 → ctFP编码 → SimpViT → 三参数+CI，作为Figure 1 | ✓ |
| 不需要，文字说明即可 | 直接从Methods文字描述，不单独画流程图 | |

**User's choice:** 需要概念图/流程图

### TOC graphic

| Option | Description | Selected |
|--------|-------------|----------|
| 需要，纳入计划 | Macromolecules要求提交TOC graphic（~3.25×1.75 in），纳入Phase 6范围 | ✓ |
| 后续再做 | 稿件提交前再做，不在Phase 6初稿范围内 | |

**User's choice:** 需要，纳入计划

**Notes:** 最终确认6张正文图方案：Figure 1概念图、Figure 2 ctFP示例、Figure 3三参数parity（复合图）、Figure 4 ML vs Mayo、Figure 5训练曲线、Figure 6代表性分类结果。

---

## 写作流程与分工

### 分工模式

| Option | Description | Selected |
|--------|-------------|----------|
| Claude起草 + 用户审核 | Claude起草各章节初稿，用户审阅修改，逐章节迭代。适合快速出稿。 | ✓ |
| 用户写初稿 + Claude辅助 | 用户写初稿，Claude负责润色、语法检查、格式调整。适合用户对内容有很强把控。 | |
| 分章节协作 | 用户写核心章节（Introduction/Discussion），Claude写技术性章节（Methods/Results），然后合并审阅。 | |

### 输出格式

| Option | Description | Selected |
|--------|-------------|----------|
| LaTeX（achemso模板） | 用Macromolecules的LaTeX模板，直接生成可提交的.tex文件 | ✓ |
| Markdown初稿 → 后转LaTeX | 先用Markdown写初稿，确认内容后再转LaTeX | |
| Word (.docx) | 用Word写，Macromolecules也接受.docx提交 | |

### 审核节奏

| Option | Description | Selected |
|--------|-------------|----------|
| 逐章节审核 | 每写完一个章节停下来让用户审阅，确认后再写下一个。节奏较慢但控制精确。 | ✓ |
| 全文完成后一次性审核 | 先写完全文初稿，然后用户一次性审阅全部内容。速度快但可能需要大幅调整。 | |

**User's choice:** Claude起草 + 逐章节审核 + LaTeX（achemso模板）
**Notes:** 用户额外要求：除LaTeX外，还需产出Word英文版和Word中文版，共三份输出文件。

---

## SI结构与深度

### ODE推导深度

| Option | Description | Selected |
|--------|-------------|----------|
| 完整推导（可复现级） | 完整写出矩量方程组（~14个ODE变量），区分single-eq和pre-eq模型，让读者能完全复现 | ✓ |
| 简要版（关键方程+代码引用） | 只列关键方程和变量定义，省略推导过程，引导读者看代码 | |

### SI包含内容

| Option | Description | Selected |
|--------|-------------|----------|
| 数据集构建细节 | 数据集规模、参数范围、LHS采样、噪声注入、失败率等 | ✓ |
| 训练超参数表 | batch size、学习率、损失函数权重、早停策略、模型参数量等 | ✓ |
| Bootstrap UQ详细流程 | 200头微调、F分布JCI、事后校准因子等完整流程 | ✓ |
| 完整评估图表 | 12张按RAFT类型分类parity图 + 3张残差图 | ✓ |

**User's choice:** 全部选中
**Notes:** SI内容非常充实，体现可复现性。

---

## 开源与数据共享

| Option | Description | Selected |
|--------|-------------|----------|
| 全部开源 | 代码（GitHub repo）、模型权重、文献验证集全部开源，配合Zenodo DOI | ✓ |
| 代码开源，模型不开放 | 代码开源，但模型权重和数据集不开放，仅通过Web应用提供推理服务 | |
| 不开源 | 什么都不开源，只提供Web应用访问 | |

**User's choice:** 全部开源
**Notes:** Macromolecules越来越重视可复现性，全部开源是加分项。

---

## 局限性讨论

| Option | Description | Selected |
|--------|-------------|----------|
| 模拟数据 vs 真实数据差距 | 模型在模拟数据上训练，真实实验数据可能有ODE未捕获的复杂性 | ✓ |
| Retardation预测的适用范围 | TTC/xanthate/dithiocarbamate中retardation≈1.0，预测价值有限 | ✓ |
| 参数范围边界 | log10(Ctr)∈[-2,4]、[CTA]/[M]∈[0.001,0.1]，超出范围的外推风险 | ✓ |
| 未覆盖的实验条件 | 温度固定、未考虑溶剂效应、仅覆盖四种RAFT剂类型 | ✓ |

**User's choice:** 全部选中
**Notes:** 四个局限性全部纳入Discussion，展示学术诚实性。

---

## Claude's Discretion

- 概念图/流程图的具体设计风格和工具选择
- TOC graphic的视觉设计
- ctFP示例图的具体样本选择
- Figure 6中选择哪种RAFT类型作为代表
- LaTeX排版细节
- Word版本的生成方式
- 各章节段落结构和过渡语
- 参考文献的具体选择和数量

## Deferred Ideas

None — discussion stayed within phase scope
