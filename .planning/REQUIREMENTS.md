# Requirements: ViT-Ctr

**Defined:** 2026-03-24
**Core Value:** 一次输入实验数据，同时提取Ctr、诱导期和减速因子三个参数——传统方法需要三组独立实验才能分别获得。

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### 动力学模拟

- [x] **SIM-01**: 用户可通过RAFT动力学ODE模型正向模拟任意Ctr下的Mn-转化率和Đ-转化率曲线
- [x] **SIM-02**: ODE模型支持所有RAFT剂类型（dithioester、trithiocarbonate、xanthate、dithiocarbamate），含两阶段预平衡机制
- [x] **SIM-03**: 参数空间覆盖Ctr~0.01-10000、[CTA]/[M]~0.001-0.1、kadd/kfrag比、温度40-120°C
- [ ] **SIM-04**: 百万级数据集并行生成（joblib），存储为HDF5格式

### 指纹编码

- [x] **ENC-01**: 实验数据编码为64×64双通道ctFP图像（Ch1=Mn, Ch2=Đ），x轴=[CTA]/[M], y轴=conversion
- [x] **ENC-02**: ctFP编码函数在训练管线和Web应用中共享同一实现，防止编码不一致

### 模型训练

- [ ] **TRN-01**: SimpViT模型（2层Transformer, 4头注意力, hidden=64）训练，输出3个参数：log10(Ctr)、inhibition period、retardation factor
- [ ] **TRN-02**: 训练集/验证集/测试集按Ctr范围分层划分，防止数据泄漏
- [ ] **TRN-03**: 训练使用log-space MSE损失函数，记录收敛曲线（loss vs. epoch）

### 模型评估

- [x] **EVL-01**: 在测试集上报告每个输出参数的R²、RMSE、MAE
- [x] **EVL-02**: 绘制每个输出参数的parity图（预测值 vs. 真实值）
- [x] **EVL-03**: 按RAFT剂类型分别评估模型性能，生成分类别parity图
- [ ] **EVL-04**: 实现Mayo方程基线，在相同文献验证集上对比ML方法与传统方法的准确度

### 文献验证

- [ ] **VAL-01**: 收集10+篇文献的已发表Ctr实验值，覆盖多种RAFT剂类型
- [ ] **VAL-02**: 每个文献Ctr值标注测量方法（Mayo/CLD/分散度法）和实验条件
- [ ] **VAL-03**: 用训练好的模型在文献数据上预测Ctr，与发表值对比，报告fold-error

### 不确定性量化

- [ ] **UQ-01**: Bootstrap采样（200次迭代）+ F分布联合置信区间估计
- [ ] **UQ-02**: 在验证集上进行事后校准，确保95% CI覆盖率达标

### Web应用

- [ ] **APP-01**: Streamlit Web应用支持手动输入实验数据（[CTA]/[M], conversion, Mn, Đ）
- [ ] **APP-02**: 支持Excel/CSV文件上传，提供可下载的Excel模板
- [ ] **APP-03**: 同时显示三个预测参数（Ctr、诱导期、减速因子）及其95%置信区间
- [ ] **APP-04**: 输入数据验证：conversion∈(0,1), Mn>0, Đ≥1, 至少3个数据点
- [ ] **APP-05**: ctFP指纹双通道热力图可视化，展示模型输入的数据结构
- [ ] **APP-06**: 模型加载使用st.cache_resource缓存，避免重复加载

### 论文撰写

- [ ] **PAP-01**: 英文学术论文正文，包含Introduction、Methods、Results、Discussion、Conclusion
- [ ] **PAP-02**: Supporting Information，包含ODE推导、数据集构建方法、模型训练细节、完整验证结果
- [ ] **PAP-03**: 路线A探索性研究（分子结构→Ctr），作为论文亮点/展望章节

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### 扩展功能

- **EXT-01**: 用户注册追踪（Google Sheets轻量登录）
- **EXT-02**: 逆向设计（给定目标Ctr，推荐RAFT剂结构）
- **EXT-03**: 扩展到其他CRP体系（ATRP、NMP）
- **EXT-04**: 批量预测API（REST端点）
- **EXT-05**: 注意力图可视化（模型可解释性）

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| 完整kadd/kfrag解耦预测 | 从Mn/Đ数据无法可靠解耦这些参数，声称可预测将不可辩护 |
| 移动端适配 | 化学家使用实验室电脑，移动端优化无科学价值 |
| 用户密码认证系统 | 研究工具不需要完整认证，轻量邮箱注册已移至v2 |
| 实验数据存储/数据库 | 隐私、机构审批复杂性；工具设计为无状态 |
| 多语言界面 | Web应用面向国际研究者，英文界面即可 |
| 实时在线学习 | 破坏可复现性，超出论文工具范围 |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| SIM-01 | Phase 1 | Complete |
| SIM-02 | Phase 1 | Complete |
| SIM-03 | Phase 1 | Complete |
| SIM-04 | Phase 2 | Pending |
| ENC-01 | Phase 1 | Complete |
| ENC-02 | Phase 1 | Complete |
| TRN-01 | Phase 3 | Pending |
| TRN-02 | Phase 3 | Pending |
| TRN-03 | Phase 3 | Pending |
| EVL-01 | Phase 3 | Complete |
| EVL-02 | Phase 3 | Complete |
| EVL-03 | Phase 3 | Complete |
| EVL-04 | Phase 4 | Pending |
| VAL-01 | Phase 4 | Pending |
| VAL-02 | Phase 4 | Pending |
| VAL-03 | Phase 4 | Pending |
| UQ-01 | Phase 3 | Pending |
| UQ-02 | Phase 3 | Pending |
| APP-01 | Phase 5 | Pending |
| APP-02 | Phase 5 | Pending |
| APP-03 | Phase 5 | Pending |
| APP-04 | Phase 5 | Pending |
| APP-05 | Phase 5 | Pending |
| APP-06 | Phase 5 | Pending |
| PAP-01 | Phase 6 | Pending |
| PAP-02 | Phase 6 | Pending |
| PAP-03 | Phase 6 | Pending |

**Coverage:**
- v1 requirements: 27 total
- Mapped to phases: 27
- Unmapped: 0

---
*Requirements defined: 2026-03-24*
*Last updated: 2026-03-24 after roadmap creation — all 27 requirements mapped*
