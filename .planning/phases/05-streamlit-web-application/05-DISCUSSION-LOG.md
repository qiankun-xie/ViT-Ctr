# Phase 5: Streamlit Web Application - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-03
**Phase:** 05-streamlit-web-application
**Areas discussed:** 数据输入流程, 结果展示, ctFP可视化, 应用结构与UX

---

## 数据输入流程

### Q1: 手动输入方式

| Option | Description | Selected |
|--------|-------------|----------|
| 可编辑表格（推荐） | 用 st.data_editor 显示可编辑表格，预填列名，用户直接在表格里输入数据，类似 Excel 体验 | ✓ |
| 逐行表单 | 每行用独立的 number_input 字段，点添加行逐行输入。更引导式但多行数据较慢 | |
| 仅文件上传 | 仅支持 Excel/CSV 上传，不提供手动输入。界面最简但灵活性低 | |

**User's choice:** 可编辑表格
**Notes:** 无

### Q2: 文件上传后处理

| Option | Description | Selected |
|--------|-------------|----------|
| 上传后可编辑（推荐） | 上传后数据填充到可编辑表格，用户可以检查/修改后再预测 | ✓ |
| 上传后直接预测 | 上传后直接进入预测流程，不显示中间编辑步骤 | |

**User's choice:** 上传后可编辑
**Notes:** 无

### Q3: 输入验证时机

| Option | Description | Selected |
|--------|-------------|----------|
| 提交时验证（推荐） | 点"预测"按钮时统一检查，不合格则显示错误信息并阻止推理 | ✓ |
| 实时 + 提交时 | 输入时实时标红不合格单元格，同时提交时再次检查 | |
| You decide | 由 Claude 决定具体验证时机和展示方式 | |

**User's choice:** 提交时验证
**Notes:** 无

### Q4: Excel 模板下载位置

| Option | Description | Selected |
|--------|-------------|----------|
| 上传区旁边的下载按钮（推荐） | 在文件上传区域旁边放一个"下载 Excel 模板"按钮 | ✓ |
| 侧边栏/底部链接 | 在侧边栏或页面底部放置模板下载链接 | |

**User's choice:** 上传区旁边的下载按钮
**Notes:** 无

---

## 结果展示

### Q5: 三参数展示布局

| Option | Description | Selected |
|--------|-------------|----------|
| 三卡片并排（推荐） | 三个参数并排显示（类似仪表盘），每个卡片包含参数名、预测值、±95% CI。Ctr 卡片突出显示 | ✓ |
| 统一表格 | 一个统一表格显示三行（参数名 / 预测值 / 95% CI） | |
| You decide | 由 Claude 决定具体布局方式 | |

**User's choice:** 三卡片并排
**Notes:** 无

### Q6: Ctr 数值格式

| Option | Description | Selected |
|--------|-------------|----------|
| 原始值为主 + log10副标题（推荐） | 显示原始 Ctr 值（如 Ctr = 125.3），同时在副标题显示 log10(Ctr) = 2.10 | ✓ |
| 仅 log10(Ctr) | 仅显示 log10(Ctr)，与模型输出一致 | |
| 仅原始 Ctr | 仅显示原始 Ctr 值，不显示对数 | |

**User's choice:** 原始值为主 + log10副标题
**Notes:** 无

### Q7: 95% CI 展示形式

| Option | Description | Selected |
|--------|-------------|----------|
| 区间形式 [lower, upper]（推荐） | 在预测值下方显示"95% CI: [lower, upper]" | ✓ |
| ± 形式 | 显示为"Ctr = 125.3 ± 18.7"，更紧凑但不对称时不够准确 | |

**User's choice:** 区间形式 [lower, upper]
**Notes:** 无

---

## ctFP 可视化

### Q8: 双通道热力图布局

| Option | Description | Selected |
|--------|-------------|----------|
| 并排双通道（推荐） | 左右并排两张热力图，左=Mn通道，右=Đ通道，各自有独立 colorbar | ✓ |
| 叠加显示 | 叠加显示，用不同颜色映射两个通道，更紧凑但可能难以区分 | |
| You decide | 由 Claude 决定具体可视化方式 | |

**User's choice:** 并排双通道
**Notes:** 无

### Q9: ctFP 热力图位置

| Option | Description | Selected |
|--------|-------------|----------|
| 结果下方（推荐） | 热力图放在预测结果下方，作为"模型输入可视化"补充信息 | ✓ |
| 与结果并排 | 热力图放在预测结果旁边，左右并排展示 | |
| Expander 折叠 | 放在可展开的 expander 里，默认收起，点击展开查看 | |

**User's choice:** 结果下方
**Notes:** 无

---

## 应用结构与 UX

### Q10: 页面结构

| Option | Description | Selected |
|--------|-------------|----------|
| 单页流式（推荐） | 所有功能在一个页面上从上到下流式展示：标题→输入区→预测按钮→结果→ctFP | ✓ |
| 侧边栏 + 主区域 | 侧边栏放输入控件，主区域显示结果。类似 ViT-RR 的 deploy.py 布局 | |
| 多页面 | 用 Streamlit 多页面，分为"手动输入"和"文件上传"两个页面 | |

**User's choice:** 单页流式
**Notes:** 无

### Q11: 输入方式切换

| Option | Description | Selected |
|--------|-------------|----------|
| Tabs 切换（推荐） | 用 st.tabs 分为"Manual Input"和"File Upload"两个标签页 | ✓ |
| 上下排列 | 两种输入方式上下排列，都显示在页面上 | |
| Radio 选择器 | 用 st.radio 或 selectbox 选择输入方式 | |

**User's choice:** Tabs 切换
**Notes:** 无

### Q12: 品牌/介绍信息

| Option | Description | Selected |
|--------|-------------|----------|
| 简洁标题 + 底部引用（推荐） | 页面顶部显示应用名称和一句话描述，底部显示论文引用信息和 GitHub 链接 | ✓ |
| 详细介绍区 | 顶部显示详细的项目介绍、使用说明、方法简介等 | |
| You decide | 由 Claude 决定品牌展示方式 | |

**User's choice:** 简洁标题 + 底部引用
**Notes:** 无

### Q13: 错误提示方式

| Option | Description | Selected |
|--------|-------------|----------|
| 页内错误框（推荐） | 用 st.error() 在页面内显示红色错误框，列出所有不合格的字段和原因 | ✓ |
| Toast 弹窗 | 用 st.toast() 弹出临时提示，几秒后自动消失 | |
| You decide | 由 Claude 决定错误展示方式 | |

**User's choice:** 页内错误框
**Notes:** 无

---

## Claude's Discretion

- st.data_editor 的具体列配置（数据类型、默认值、列宽）
- Excel 模板的具体格式（列名、示例数据行数）
- 热力图的 colormap 选择和图像尺寸
- 页面标题的具体措辞和描述文案
- Predict 按钮的样式和位置
- 加载状态的展示方式（spinner 文案等）
- 三卡片的具体 CSS/样式实现

## Deferred Ideas

None — discussion stayed within phase scope
