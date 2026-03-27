# Phase 3: Model Training and Evaluation - Context

**Gathered:** 2026-03-27
**Status:** Ready for planning

<domain>
## Phase Boundary

在~1M ctFP数据集上训练SimpViT三参数输出模型(log10(Ctr)、诱导期、减速因子),用parity图和按RAFT类型分类的指标评估性能,实现轻量Bootstrap不确定性量化(冻结backbone,200次输出头微调)和事后校准置信区间。

</domain>

<decisions>
## Implementation Decisions

### 训练策略
- **D-01:** Colab T4用于正式训练(16GB显存),本地MX350仅用于代码调试和小规模验证
- **D-02:** Batch size = 64
- **D-03:** 学习率初始值3e-4,使用ReduceLROnPlateau调度器(factor=0.5, patience=5)
- **D-04:** 早停策略: 监控验证集loss,patience=15 epochs,保存验证集loss最低的checkpoint
- **D-05:** 基础模型训练epochs由早停自动决定(预计30-100 epochs)

### 损失函数设计
- **D-06:** 手动加权MSE损失函数,权重为 w_ctr=2.0, w_inh=0.5, w_ret=0.5
- **D-07:** 强调Ctr为核心预测目标,inhibition period和retardation factor为附加输出
- **D-08:** 损失计算公式: `loss = 2.0 * MSE(pred_ctr, true_ctr) + 0.5 * MSE(pred_inh, true_inh) + 0.5 * MSE(pred_ret, true_ret)`

### 数据划分策略
- **D-09:** 按Ctr范围分层划分,log10(Ctr)每0.5一档(共12档: -2.0到4.0)
- **D-10:** 每档内按80/10/10随机划分train/val/test
- **D-11:** 保证测试集覆盖全Ctr范围,避免外推风险

### Bootstrap不确定性量化
- **D-12:** 轻量Bootstrap方案: 训练一个基础模型后冻结backbone,仅微调输出头
- **D-13:** 200次Bootstrap迭代,每次有放回抽样训练集后微调输出头5 epochs(固定)
- **D-14:** 推理时用200个输出头的预测计算均值和协方差矩阵
- **D-15:** 使用F分布计算95%联合置信区间(JCI): `half_width = sqrt(F(0.95, p, n-p) * p * cov_diag / n)`, 其中p=3(三个输出)
- **D-16:** 在验证集上事后校准: 检查95% CI的实际覆盖率,如不足95%则用标量因子放大CI直到达标

### 评估指标体系
- **D-17:** 基础指标: 每个输出(log10(Ctr), inhibition, retardation)分别报告R²、RMSE、MAE
- **D-18:** 按Ctr范围分段评估: 低段(log10(Ctr) ∈ [-2, 0)), 中段([0, 2)), 高段([2, 4]),每段报告R²和MAE
- **D-19:** 按RAFT类型分类评估: 四种类型(dithioester, trithiocarbonate, xanthate, dithiocarbamate)各自报告R²/RMSE/MAE,生成四张分类别parity图(对应EVL-03)
- **D-20:** 残差分布分析: 绘制每个输出的残差直方图(预测值-真实值),检查系统性偏差(如总是高估或低估)
- **D-21:** 异常值分析: 定义异常值为|预测-真实| > 2×std,统计异常值比例,分析异常值样本的共同特征(如特定Ctr范围或RAFT类型)

### Claude's Discretion
- DataLoader的num_workers配置(建议2-4)
- ReduceLROnPlateau的具体参数(factor, patience, min_lr)
- 模型checkpoint保存策略(仅保存最佳,还是保存top-3)
- HDF5数据加载的具体实现(一次性加载vs按需加载)
- Parity图的具体可视化风格(颜色、标记、图例位置)
- 残差直方图的bins数量
- 训练日志的记录频率和格式

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### 参考项目
- `C:/CodingCraft/DL/ViT-RR/model_utils.py` — SimpViT架构定义,直接复用但修改num_outputs=3
- `C:/CodingCraft/DL/ViT-RR/deploy.py` — Bootstrap推理模式和F分布置信区间计算的参考实现

### 研究文档
- `.planning/research/STACK.md` — 技术栈推荐(PyTorch训练循环、HDF5数据加载)
- `.planning/research/ARCHITECTURE.md` — 系统架构和训练管线设计
- `.planning/research/PITFALLS.md` — 领域陷阱,特别是C3(Bootstrap实现)和C5(评估指标选择)

### Phase 1和Phase 2输出
- `.planning/phases/01-ode-foundation-and-ctfp-encoder/01-CONTEXT.md` — 三参数输出定义、尺度选择(log10(Ctr)、无量纲比值)
- `.planning/phases/02-large-scale-dataset-generation/02-CONTEXT.md` — 数据集组织方式(4个HDF5文件、chunk=1000)
- `src/ctfp_encoder.py` — ctFP编码器,训练时需导入用于数据预处理验证
- `src/raft_ode.py` — ODE模拟器,理解三参数的物理含义
- `.planning/phases/02-large-scale-dataset-generation/02-DATASET-INFO.md` — 数据集统计信息和Google Drive文件ID

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/ctfp_encoder.py` 的 `transform()` 函数 — 训练时可能需要用于数据验证或可视化
- ViT-RR的SimpViT类 — 直接复用,仅需修改`num_outputs=2`改为`num_outputs=3`
- ViT-RR的Bootstrap推理逻辑 — F分布置信区间计算可直接移植

### Established Patterns
- PyTorch标准训练循环: DataLoader + optimizer + loss.backward() + optimizer.step()
- HDF5数据集封装为PyTorch Dataset类,支持按需加载
- 使用tqdm显示训练进度
- Checkpoint保存为.pth文件,包含model.state_dict()和optimizer.state_dict()

### Integration Points
- Phase 2生成的HDF5文件是训练的直接输入
- 训练好的模型权重将在Phase 5的Streamlit应用中加载
- 评估结果(parity图、指标表)将用于Phase 6的论文撰写

</code_context>

<specifics>
## Specific Ideas

- 轻量Bootstrap方案的理由: 完整Bootstrap需要400小时(200次×2小时),轻量方案只需~22小时(基础模型2小时+200次×6分钟),在论文中说明"为了计算效率,我们冻结特征提取层,仅对输出头进行Bootstrap采样"
- 损失函数权重2.0/0.5/0.5的理由: Ctr是论文核心价值主张,inhibition和retardation是附加输出,权重分配反映了"Core Value"的优先级
- 按Ctr范围分层划分的必要性: 防止测试集只包含训练集未见过的Ctr范围,导致外推评估失真
- 异常值分析的价值: 识别模型失效的边界条件(如极低Ctr的xanthate或极高Ctr的dithioester),为Phase 6论文的Discussion提供素材
- F分布置信区间vs简单标准差: F分布考虑了三个输出的协方差,给出的是联合置信区间(椭球),比三个独立区间更严格

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-model-training-and-evaluation*
*Context gathered: 2026-03-27*

