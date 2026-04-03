# Phase 5: Streamlit Web Application - Context

**Gathered:** 2026-04-03
**Status:** Ready for planning

<domain>
## Phase Boundary

部署一个 Streamlit Web 应用，研究者可通过可编辑表格手动输入或 Excel/CSV 上传 RAFT 动力学实验数据（[CTA]/[M]、conversion、Mn、Đ），一键获得 Ctr、诱导期和减速因子三个参数的预测值及 95% 置信区间，并可视化 ctFP 双通道热力图验证编码结果。

</domain>

<decisions>
## Implementation Decisions

### A. 数据输入流程
- **D-01:** 手动输入使用 st.data_editor 可编辑表格，预填列名（[CTA]/[M]、conversion、Mn、Đ），用户直接在表格中输入数据
- **D-02:** Excel/CSV 上传后数据填充到同一可编辑表格，用户可检查/修改后再提交预测
- **D-03:** 手动输入和文件上传通过 st.tabs 切换（"Manual Input" / "File Upload" 两个标签页）
- **D-04:** Excel 模板下载按钮放在文件上传区旁边
- **D-05:** 输入验证在点击"Predict"按钮时统一触发：conversion∈(0,1)、Mn>0、Đ≥1、至少3个数据点；不合格则用 st.error() 显示红色错误框，列出所有不合格字段和原因，阻止推理

### B. 结果展示
- **D-06:** 三个预测参数用三卡片并排展示（st.columns(3)），每个卡片包含参数名、预测值、95% CI；Ctr 卡片视觉突出（更大/加粗）
- **D-07:** Ctr 显示原始值为主（如 Ctr = 125.3），副标题显示 log10(Ctr) = 2.10；模型输出 log10(Ctr) 需反变换为 10^x
- **D-08:** 95% CI 以区间形式展示：95% CI: [lower, upper]

### C. ctFP 可视化
- **D-09:** 双通道 ctFP 热力图并排展示（左=Mn通道，右=Đ通道），各自有独立 colorbar
- **D-10:** ctFP 热力图放在预测结果下方，作为"模型输入可视化"补充信息

### D. 应用结构与 UX
- **D-11:** 单页流式布局：标题 → 输入区（tabs） → Predict 按钮 → 结果卡片 → ctFP 热力图 → 底部引用
- **D-12:** 页面顶部显示简洁标题和一句话描述，底部显示论文引用信息和 GitHub 链接（如有）
- **D-13:** 输入验证失败时用 st.error() 在页面内显示红色错误框

### E. 模型加载
- **D-14:** 使用 st.cache_resource 缓存模型权重和 Bootstrap 头，避免每次 rerun 重复加载（APP-06）

### Claude's Discretion
- st.data_editor 的具体列配置（数据类型、默认值、列宽）
- Excel 模板的具体格式（列名、示例数据行数）
- 热力图的 colormap 选择和图像尺寸
- 页面标题的具体措辞和描述文案
- Predict 按钮的样式和位置
- 加载状态的展示方式（spinner 文案等）
- 三卡片的具体 CSS/样式实现

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### 参考项目
- `C:/CodingCraft/DL/ViT-RR/deploy.py` — ViT-RR 的 Streamlit 部署参考：st.cache_resource 模型加载、Google Sheets 用户注册（v1 不需要）、Excel 上传处理、Bootstrap 推理流程

### 现有代码（Phase 5 直接依赖）
- `src/model.py` — SimpViT 模型定义（num_outputs=3），推理时加载
- `src/ctfp_encoder.py` — transform() 函数，Web 应用中编码用户输入数据为 ctFP（ENC-02 共享实现）
- `src/bootstrap.py` — predict_with_uncertainty() 函数，接收 ctFP 张量返回 (mean, calibrated_half_width)；compute_jci()、calibrate_coverage() 等辅助函数

### Phase 3 上下文（Bootstrap CI 实现细节）
- `.planning/phases/03-model-training-and-evaluation/03-CONTEXT.md` — D-12~D-16：Bootstrap 200 heads、F-dist JCI with p=3、事后校准因子

### 需求映射
- `.planning/REQUIREMENTS.md` §APP-01~APP-06 — Web 应用全部需求定义

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/ctfp_encoder.py` transform() — 直接在 Web 应用中调用，将用户输入编码为 (2, 64, 64) ctFP 张量
- `src/model.py` SimpViT — 加载训练好的权重进行推理
- `src/bootstrap.py` predict_with_uncertainty() — 完整的 Bootstrap 推理 + 校准 CI 流程，接收单个 ctFP 张量返回三参数预测值和校准后的半宽度
- ViT-RR deploy.py — Streamlit 应用的完整参考实现，包括 st.cache_resource 模式、数据输入处理、结果展示

### Established Patterns
- PyTorch 模型推理：model.eval() + torch.no_grad()
- Bootstrap 推理：加载 base_model + 200 heads，逐头预测取均值和协方差
- ctFP 编码：transform() 接收 (cta_ratio_norm, conversion, mn_norm, dispersity) 元组列表

### Integration Points
- Phase 3 产出的 `checkpoints/best_model.pth` 和 `checkpoints/bootstrap_heads.pth` 是推理的直接输入
- Phase 3 产出的 `checkpoints/calibration.json` 包含校准因子
- 用户输入数据需要预处理：cta_ratio 除以 0.1 归一化、Mn 除以 Mn_theory 归一化

</code_context>

<specifics>
## Specific Ideas

- 用户输入的 Mn 需要归一化为 Mn/Mn_theory，其中 Mn_theory = [M]₀/[CTA]₀ × M_monomer；Web 应用可能需要额外输入单体分子量和初始浓度比，或者让用户直接输入归一化后的值——这是 researcher 需要调研的关键点
- ViT-RR deploy.py 是最直接的参考，但 v1 不需要 Google Sheets 用户注册部分
- Ctr 卡片视觉突出的设计反映了项目的 Core Value：Ctr 是核心预测目标，inhibition 和 retardation 是附加输出

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 05-streamlit-web-application*
*Context gathered: 2026-04-03*
