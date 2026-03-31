# 阶段01 计划中文翻译

本文件包含阶段01（ODE基础与ctFP编码器）全部3个计划的中文翻译。

---

# 计划 01-01：RAFT ODE 模拟器

---
phase: 01-ode-foundation-and-ctfp-encoder
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - src/raft_ode.py
  - tests/test_raft_ode.py
  - tests/conftest.py
  - pyproject.toml
autonomous: true
requirements: [SIM-01, SIM-02, SIM-03]
must_haves:
  truths:
    - "单平衡ODE为TTC/黄原酸酯/二硫代氨基甲酸酯生成Mn和分散度随转化率变化曲线"
    - "二硫代酯预平衡ODE产生与单平衡模型明显不同的曲线（诱导期、减速）"
    - "ODE在完整Ctr范围0.01-10000和[CTA]/[M]范围0.001-0.1内无数值失败"
    - "减速因子通过双ODE积分正确计算为Rp(RAFT)/Rp(无CTA)"
  artifacts:
    - path: "src/raft_ode.py"
      provides: "含矩量法的RAFT ODE系统，包括单平衡和预平衡两种模型"
      exports: ["simulate_raft", "raft_ode_single_eq", "raft_ode_preequilibrium"]
    - path: "tests/test_raft_ode.py"
      provides: "ODE系统的单元测试和集成测试"
      min_lines: 100
    - path: "tests/conftest.py"
      provides: "共享测试夹具和参数集"
    - path: "pyproject.toml"
      provides: "包含pytest设置的项目配置"
  key_links:
    - from: "src/raft_ode.py"
      to: "scipy.integrate.solve_ivp"
      via: "使用逐分量atol的Radau方法"
      pattern: "solve_ivp.*method.*Radau"
    - from: "tests/test_raft_ode.py"
      to: "src/raft_ode.py"
      via: "import simulate_raft"
      pattern: "from src.raft_ode import"
---

<objective>
构建基于矩量法的RAFT聚合ODE模拟器，支持全部四种RAFT试剂类型，按类型分支处理（二硫代酯使用两阶段预平衡，其他使用单平衡）。

目的：这是整个项目的根依赖。所有训练数据都源自此ODE系统。此处的正确性可避免后期完整数据集的重新生成。
输出：`src/raft_ode.py`（经测试的ODE函数），`tests/test_raft_ode.py`（通过的测试）。
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/01-ode-foundation-and-ctfp-encoder/01-CONTEXT.md
@.planning/phases/01-ode-foundation-and-ctfp-encoder/01-RESEARCH.md

<interfaces>
<!-- ViT-RR参考：model_utils.py SimpViT使用in_channels=2（双通道指纹输入） -->
<!-- 本计划创建ODE系统，生成输入ctFP编码（计划02）的原始数据 -->
<!-- 无现有代码接口可参考——这是项目中的第一段代码 -->
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>任务1：项目脚手架和RAFT ODE系统</name>
  <files>pyproject.toml, src/__init__.py, src/raft_ode.py, tests/__init__.py, tests/conftest.py</files>
  <read_first>
    - C:/CodingCraft/DL/ViT-RR/model_utils.py （参考项目结构）
    - .planning/phases/01-ode-foundation-and-ctfp-encoder/01-RESEARCH.md （ODE方程、速率常数范围、求解器配置）
    - .planning/phases/01-ode-foundation-and-ctfp-encoder/01-CONTEXT.md （锁定决策D-01至D-11）
  </read_first>
  <action>
1. 创建 `pyproject.toml`，包含：
   - `[project]` 部分：name="vit-ctr", version="0.1.0", requires-python=">=3.11"
   - `[tool.pytest.ini_options]`：testpaths=["tests"], pythonpath=[".", "src"]

2. 创建 `src/__init__.py`（空文件）和 `tests/__init__.py`（空文件）。

3. 创建 `src/raft_ode.py`，按D-01至D-06实现完整RAFT ODE系统：

   **单平衡模型的状态向量（11个变量，按D-01）：**
   `y = [M, I, CTA, P_dot, Int, mu0, mu1, mu2, lam0, lam1, lam2]`
   其中 mu = 活性链矩，lam = 死链矩。

   **函数 `raft_ode_single_eq(t, y, kd, f, ki, kp, kt, kadd, kfrag)`：**
   - 物种守恒：dM/dt = -kp*P*M, dI/dt = -kd*I, dCTA/dt = -kadd*P*CTA + kfrag*Int, dP/dt = 2*f*kd*I + kfrag*Int - kadd*P*CTA - 2*kt*P^2, dInt/dt = kadd*P*CTA - kfrag*Int
   - 活性链矩：dmu0/dt = 2*f*kd*I/(kp*M+1e-30) ...（完整方程见研究文档示例1；注意：引发生成链长为1的链，因此贡献 mu0 += R_init, mu1 += R_init*1, mu2 += R_init*1）
   - 死链矩：dlam0/dt = kt*mu0*P + kadd*mu0*CTA（转移生成死链），dlam1/dt = kt*mu1*P + kadd*mu1*CTA, dlam2/dt = kt*mu2*P + kadd*mu2*CTA
   - 按D-02：Mn = (mu1+lam1)/(mu0+lam0) * M_monomer, D = (mu2+lam2)*(mu0+lam0)/(mu1+lam1)^2

   **函数 `raft_ode_preequilibrium(t, y, kd, f, ki, kp, kt, kadd, kfrag, kadd0, kfrag0)`：**
   - 扩展状态向量（13个变量）：按D-03添加 CTA_0（初始RAFT试剂）和 Int_pre（预平衡中间体）
   - 预平衡反应：P. + CTA_0 -> Int_pre (kadd0), Int_pre -> R. + 大分子CTA (kfrag0)
   - R. 作为新的增长自由基重新引发
   - 主平衡：P. + 大分子CTA -> Int -> P'. + 大分子CTA (kadd, kfrag)，与单平衡相同
   - 二硫代酯的慢速 kfrag0（kfrag0 << kfrag）产生诱导期

   **函数 `simulate_raft(params, raft_type='ttc', t_end=36000, n_conv_points=50)`：**
   - `params` 字典，键包括：kd, f, ki, kp, kt, kadd, kfrag, M0, I0, CTA0, M_monomer（二硫代酯还包括 kadd0, kfrag0）
   - `raft_type` 为以下之一：'dithioester', 'ttc', 'xanthate', 'dithiocarbamate'
   - 从 params 构建初始条件 y0（所有矩从0开始，P_dot=0, Int=0）
   - 对于二硫代酯：使用 raft_ode_preequilibrium，初始设置 CTA_0=CTA0, CTA=0（所有RAFT试剂均为初始形式）
   - 对于其他类型：使用 raft_ode_single_eq
   - 调用 solve_ivp，参数为 method='Radau', rtol=1e-8，逐分量 atol 数组（M:1e-6, I:1e-10, CTA:1e-8, P:1e-14, Int:1e-14, 矩：0阶/1阶/2阶分别为1e-14/1e-10/1e-6），dense_output=True, max_step=100.0
   - 若 sol.success 为 False：返回 None
   - 在 n_conv_points 个从0.02到0.95均匀分布的转化率处采样，使用 scipy.optimize.brentq 在 dense_output 上求解每个目标转化率对应的时间
   - 在每个转化率点按D-02计算 Mn 和 D
   - 归一化 Mn：mn_norm = Mn / (M0/CTA0 * M_monomer)，按研究文档的归一化策略（固定分母）

   **函数 `compute_retardation_factor(params, raft_type, conv_target=0.5)`：**
   - 按D-06：运行含CTA的 simulate_raft 和无CTA的FRP模拟（设置 CTA0=0, kadd=0）
   - 找到每个体系达到 min(conv_target, max_conv_raft) 时的时间
   - Rp = 该时刻的 d(转化率)/dt，通过 dense_output 计算为 -dM/dt / M0
   - 返回 Rp_raft / Rp_frp（无量纲，范围0-1）
   - 若RAFT体系转化率低于5%，返回0.01（接近零的哨兵值，按研究文档陷阱5）

   **函数 `compute_inhibition_period(sol_dense, M0, t_end)`：**
   - 按D-05：找到转化率首次达到1%的时间 t_inh
   - 返回 t_inh / t_end（无量纲，范围0-1）
   - 若转化率从未达到1%，返回1.0

4. 创建 `tests/conftest.py`，包含共享夹具：
   - `typical_ttc_params`：kd=1.5e-5, f=0.5, ki=1e4, kp=650, kt=1e8, kadd=1e6, kfrag=1e4, M0=1.0, I0=0.01, CTA0=0.01, M_monomer=100.12（MMA配TTC）
   - `typical_dithioester_params`：相同基础参数但 kadd0=1e6, kfrag0=1.0（慢速裂解），CTA0=0.005
   - `typical_xanthate_params`：kp=6700, kadd=1e4, kfrag=1e3, M_monomer=86.09（VAc）
   - `extreme_high_ctr_params`：kadd=1e7, kfrag=1e5（Ctr~10000）
   - `extreme_low_ctr_params`：kadd=1e3, kfrag=1e2（Ctr~0.01范围，取决于kp）
  </action>
  <verify>
    <automated>cd C:/CodingCraft/DL/ViT-Ctr && python -c "from src.raft_ode import simulate_raft, raft_ode_single_eq, raft_ode_preequilibrium, compute_retardation_factor, compute_inhibition_period; print('imports OK')"</automated>
  </verify>
  <acceptance_criteria>
    - src/raft_ode.py 包含 `def raft_ode_single_eq(t, y, kd, f, ki, kp, kt, kadd, kfrag)`
    - src/raft_ode.py 包含 `def raft_ode_preequilibrium(t, y, kd, f, ki, kp, kt, kadd, kfrag, kadd0, kfrag0)`
    - src/raft_ode.py 包含 `def simulate_raft(params, raft_type=`
    - src/raft_ode.py 包含 `def compute_retardation_factor(`
    - src/raft_ode.py 包含 `def compute_inhibition_period(`
    - src/raft_ode.py 包含 `solve_ivp` 和 `method='Radau'` 和 `rtol=1e-8`
    - src/raft_ode.py 包含 `brentq` 用于转化率到时间的映射
    - pyproject.toml 包含 `[tool.pytest.ini_options]` 和 `pythonpath`
    - tests/conftest.py 包含 `typical_ttc_params` 和 `typical_dithioester_params` 和 `typical_xanthate_params`
    - `python -c "from src.raft_ode import simulate_raft"` 退出码为0
  </acceptance_criteria>
  <done>
    含单平衡和预平衡模型的ODE系统可导入。全部四种RAFT试剂类型可进行模拟。减速因子和诱导期可计算。项目脚手架（pyproject.toml, src/, tests/）已就位。
  </done>
</task>

<task type="auto" tdd="true">
  <name>任务2：ODE单元测试和集成测试</name>
  <files>tests/test_raft_ode.py</files>
  <read_first>
    - src/raft_ode.py （待测试的实现）
    - tests/conftest.py （共享夹具）
    - .planning/phases/01-ode-foundation-and-ctfp-encoder/01-RESEARCH.md （"阶段需求到测试映射"部分的验证测试映射）
  </read_first>
  <behavior>
    - test_forward_simulation：使用 typical_ttc_params 的 simulate_raft 返回包含 'conversion'、'mn'、'dispersity' 数组的字典，长度为50；转化率值在 [0.02, 0.95] 范围内；Mn值 > 0；分散度值 >= 1.0
    - test_preequilibrium_distinct：二硫代酯模拟产生 inhibition_period > 0.01（可见延迟），而TTC模拟产生 inhibition_period < 0.005（可忽略延迟）——曲线有明显区别
    - test_all_agent_types：simulate_raft 对全部四种 raft_type 值（'dithioester', 'ttc', 'xanthate', 'dithiocarbamate'）均成功（返回非None）
    - test_parameter_range_coverage：simulate_raft 在 Ctr 的 log10 = -2, -1, 0, 1, 2, 3, 4（按D-07的7个点覆盖完整范围）时使用TTC类型均成功
    - test_cta_ratio_range：simulate_raft 在 [CTA]/[M] = 0.001, 0.005, 0.01, 0.05, 0.1（按D-08）时均成功
    - test_limit_behavior_high_ctr：在极高Ctr（log10=4）时，50%转化率处的分散度应 < 1.3（高度可控的RAFT）
    - test_limit_behavior_low_ctr：在极低Ctr（log10=-2）时，分散度应接近常规FRP值（> 1.5）
    - test_retardation_factor_range：compute_retardation_factor 对典型参数返回 (0, 1] 范围内的值
    - test_inhibition_period_range：compute_inhibition_period 对典型参数返回 [0, 1] 范围内的值
    - test_mn_normalization：mn_norm 值大约在 [0, 2.0] 范围内，用于良好控制的RAFT（不是原始 g/mol 值）
  </behavior>
  <action>
创建 `tests/test_raft_ode.py`，使用pytest实现上述所有行为的测试。每个测试从 `src.raft_ode` 导入并使用 conftest.py 中的夹具。

对于 `test_parameter_range_coverage`，通过改变 kadd 构建参数字典以达到目标Ctr值：Ctr ~ kadd / kp（简化），因此对于 kp=650：kadd = Ctr * kp。设置 kfrag=1e4（固定）。测试 Ctr 取值 [0.01, 0.1, 1, 10, 100, 1000, 10000]。

对于 `test_preequilibrium_distinct`，同时运行：
1. simulate_raft(dithioester_params, raft_type='dithioester') —— 应显示诱导期
2. simulate_raft(ttc_params, raft_type='ttc') —— 应无诱导期
比较诱导期；二硫代酯应至少大5倍。

用 `@pytest.mark.slow` 标记 `test_diagnostic_dataset`，用于1000样本冒烟测试（将在计划03中实现）。

运行：`pytest tests/test_raft_ode.py -x -v`
  </action>
  <verify>
    <automated>cd C:/CodingCraft/DL/ViT-Ctr && pytest tests/test_raft_ode.py -x -v</automated>
  </verify>
  <acceptance_criteria>
    - tests/test_raft_ode.py 包含 `def test_forward_simulation`
    - tests/test_raft_ode.py 包含 `def test_preequilibrium_distinct`
    - tests/test_raft_ode.py 包含 `def test_all_agent_types`
    - tests/test_raft_ode.py 包含 `def test_parameter_range_coverage`
    - tests/test_raft_ode.py 包含 `def test_cta_ratio_range`
    - tests/test_raft_ode.py 包含 `def test_limit_behavior_high_ctr`
    - tests/test_raft_ode.py 包含 `def test_limit_behavior_low_ctr`
    - tests/test_raft_ode.py 包含 `def test_retardation_factor_range`
    - tests/test_raft_ode.py 包含 `def test_inhibition_period_range`
    - tests/test_raft_ode.py 包含 `def test_mn_normalization`
    - `pytest tests/test_raft_ode.py -x -v` 退出码为0且所有测试通过
  </acceptance_criteria>
  <done>
    所有ODE测试通过。单平衡和预平衡模型产生物理上合理的输出。完整参数范围已覆盖。二硫代酯预平衡产生明显的诱导行为。
  </done>
</task>

</tasks>

<verification>
- `pytest tests/test_raft_ode.py -x -v` —— 所有ODE测试通过
- `python -c "from src.raft_ode import simulate_raft; r = simulate_raft({'kd':1.5e-5,'f':0.5,'ki':1e4,'kp':650,'kt':1e8,'kadd':1e6,'kfrag':1e4,'M0':1.0,'I0':0.01,'CTA0':0.01,'M_monomer':100.12}, raft_type='ttc'); print(len(r['conversion']), 'points')"` —— 输出 "50 points"
</verification>

<success_criteria>
- ODE系统模拟全部4种RAFT试剂类型无错误
- 预平衡模型对二硫代酯产生可见诱导期（inhibition_period > 0.01）
- ODE在Ctr范围0.01-10000和[CTA]/[M]范围0.001-0.1内均成功
- test_raft_ode.py 中全部10+项测试通过
</success_criteria>

<output>
完成后，创建 `.planning/phases/01-ode-foundation-and-ctfp-encoder/01-01-SUMMARY.md`
</output>

---

# 计划 01-02：ctFP编码器

---
phase: 01-ode-foundation-and-ctfp-encoder
plan: 02
type: execute
wave: 1
depends_on: []
files_modified:
  - src/ctfp_encoder.py
  - tests/test_ctfp_encoder.py
autonomous: true
requirements: [ENC-01, ENC-02]
must_haves:
  truths:
    - "ctFP编码器从原始动力学数据生成 (2, 64, 64) 的 torch.Tensor"
    - "通道0包含归一化Mn值，通道1包含分散度值"
    - "x轴映射到 [CTA]/[M]（列索引），y轴映射到转化率（行索引）"
    - "相同输入产生字节级相同的输出（确定性，无随机性）"
    - "编码器可作为独立模块导入，不依赖Streamlit或训练框架"
  artifacts:
    - path: "src/ctfp_encoder.py"
      provides: "共享ctFP编码函数"
      exports: ["transform"]
      min_lines: 30
    - path: "tests/test_ctfp_encoder.py"
      provides: "编码器单元测试"
      min_lines: 60
  key_links:
    - from: "src/ctfp_encoder.py"
      to: "torch"
      via: "torch.tensor 输出转换"
      pattern: "torch\\.tensor"
    - from: "tests/test_ctfp_encoder.py"
      to: "src/ctfp_encoder.py"
      via: "import transform"
      pattern: "from src.ctfp_encoder import"
---

<objective>
构建共享的ctFP（链转移指纹）编码器，将RAFT原始动力学数据转换为64x64双通道图像张量，作为SimpViT的输入。

目的：编码器是ODE模拟输出与模型输入之间的桥梁。它必须是在训练和Web应用上下文中完全相同使用的单一共享模块（按ENC-02），防止隐式编码偏差的陷阱。
输出：`src/ctfp_encoder.py`（经测试的transform函数），`tests/test_ctfp_encoder.py`（通过的测试）。
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/phases/01-ode-foundation-and-ctfp-encoder/01-CONTEXT.md
@.planning/phases/01-ode-foundation-and-ctfp-encoder/01-RESEARCH.md

<interfaces>
<!-- 参考：ViT-RR model_utils.py transform() 函数 -->
来自 C:/CodingCraft/DL/ViT-RR/model_utils.py：
```python
def transform(data, img_size=64):
    img_arr = np.zeros((2, img_size, img_size))
    for f1, total_conv, conv1, conv2 in data:
        row = min(math.floor(f1 * img_size), img_size - 1)
        col = min(math.floor(total_conv * img_size), img_size - 1)
        img_arr[0][row][col] = conv1
        img_arr[1][row][col] = conv2
    return torch.tensor(img_arr).float()
```
注意：ViT-RR 使用 (f1=进料比, total_conv, conv1=单体1转化率, conv2=单体2转化率)。
ViT-Ctr 调整轴映射：x=[CTA]/[M]（列），y=转化率（行），Ch0=Mn_norm，Ch1=分散度。
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>任务1：ctFP编码器实现与测试</name>
  <files>src/ctfp_encoder.py, tests/test_ctfp_encoder.py</files>
  <read_first>
    - C:/CodingCraft/DL/ViT-RR/model_utils.py （参考 transform() 函数——我们编码器的直接前身）
    - .planning/phases/01-ode-foundation-and-ctfp-encoder/01-RESEARCH.md （模式4：共享编码器模块，ctFP归一化策略，陷阱3和4）
    - .planning/phases/01-ode-foundation-and-ctfp-encoder/01-CONTEXT.md （决策D-01, D-02, ENC-01, ENC-02）
  </read_first>
  <behavior>
    - test_output_shape：使用10个数据点的 transform() 返回形状为 (2, 64, 64) 的张量
    - test_output_dtype：返回的张量 dtype 为 torch.float32
    - test_channel_assignment：单个数据点 (cta_norm=0.5, conv=0.3, mn_norm=1.2, disp=1.5) 将 mn_norm=1.2 放在通道0的 row=floor(0.3*64)=19, col=floor(0.5*64)=32 位置，disp=1.5 放在通道1的相同位置
    - test_axis_mapping：x轴为 [CTA]/[M]（映射到列索引），y轴为转化率（映射到行索引）——独立验证行/列分配
    - test_deterministic_output：使用相同输入调用 transform() 两次产生字节级相同的张量（torch.equal 返回 True）
    - test_empty_input：transform([]) 返回形状为 (2, 64, 64) 的全零张量
    - test_boundary_values：conversion=0.0 映射到第0行，conversion=0.99 映射到第63行（截断），cta_norm=0.0 映射到第0列，cta_norm=1.0 映射到第63列（截断）
    - test_mn_not_raw：mn_norm 值应为无量纲的（约0-2范围），不是原始 g/mol 值
    - test_dispersity_range：分散度值按原值传递（无归一化），典型范围1.0-3.0
    - test_no_framework_dependency：ctfp_encoder.py 不导入 streamlit，不导入 torch.nn
  </behavior>
  <action>
1. 创建 `src/ctfp_encoder.py`：

```python
"""
Chain Transfer Fingerprint (ctFP) encoder.
Shared module for encoding RAFT kinetic data into image tensors.
Used by both training pipeline and web application (ENC-02).

Axis mapping (per ENC-01):
  - x-axis (columns): [CTA]/[M] normalized to [0, 1]
  - y-axis (rows): monomer conversion in [0, 1]
  - Channel 0: Mn / Mn_theory (dimensionless, typically 0-2)
  - Channel 1: dispersity Mw/Mn (dimensionless, >= 1.0, clipped at 4.0)
"""
import numpy as np
import math
import torch


def transform(data, img_size=64):
    """
    Encode RAFT kinetic data into a chain-transfer fingerprint (ctFP).

    Args:
        data: iterable of (cta_ratio_norm, conversion, mn_norm, dispersity) tuples
              cta_ratio_norm: [CTA]/[M] normalized to [0, 1] (divide by 0.1 for training data)
              conversion: monomer conversion in [0, 1]
              mn_norm: Mn / Mn_theory (dimensionless); Mn_theory = M0/CTA0 * M_monomer
              dispersity: Mw/Mn >= 1.0 (clipped at 4.0 for safety)
        img_size: fingerprint resolution (default 64)

    Returns:
        torch.Tensor of shape (2, img_size, img_size), dtype float32
    """
    img = np.zeros((2, img_size, img_size), dtype=np.float32)
    for cta_norm, conv, mn_n, disp in data:
        col = min(int(math.floor(cta_norm * img_size)), img_size - 1)
        row = min(int(math.floor(conv * img_size)), img_size - 1)
        col = max(col, 0)
        row = max(row, 0)
        img[0, row, col] = mn_n
        img[1, row, col] = min(disp, 4.0)  # clip dispersity at 4.0
    return torch.tensor(img, dtype=torch.float32)
```

与ViT-RR transform()的关键区别：
- ViT-RR：row=f1(进料比), col=total_conv, channels=conv1/conv2
- ViT-Ctr：row=转化率, col=cta_ratio_norm, channels=mn_norm/分散度
- 新增：col/row 截断到 >= 0 以确保安全，分散度在4.0处截断
- 新增：在 np.zeros 中显式指定 dtype=np.float32 以确保一致的内存布局

2. 创建 `tests/test_ctfp_encoder.py`，实现上述所有行为的测试。每个测试直接断言特定像素值或张量属性。

运行：`pytest tests/test_ctfp_encoder.py -x -v`
  </action>
  <verify>
    <automated>cd C:/CodingCraft/DL/ViT-Ctr && pytest tests/test_ctfp_encoder.py -x -v</automated>
  </verify>
  <acceptance_criteria>
    - src/ctfp_encoder.py 包含 `def transform(data, img_size=64)`
    - src/ctfp_encoder.py 包含 `torch.tensor(img, dtype=torch.float32)`
    - src/ctfp_encoder.py 包含 `img = np.zeros((2, img_size, img_size), dtype=np.float32)`
    - src/ctfp_encoder.py 不包含 `import streamlit` 或 `import torch.nn`
    - src/ctfp_encoder.py 包含 `min(disp, 4.0)`（分散度截断）
    - tests/test_ctfp_encoder.py 包含 `def test_output_shape`
    - tests/test_ctfp_encoder.py 包含 `def test_channel_assignment`
    - tests/test_ctfp_encoder.py 包含 `def test_deterministic_output`
    - tests/test_ctfp_encoder.py 包含 `def test_boundary_values`
    - tests/test_ctfp_encoder.py 包含 `def test_no_framework_dependency`
    - `pytest tests/test_ctfp_encoder.py -x -v` 退出码为0且所有测试通过
  </acceptance_criteria>
  <done>
    ctFP编码器产生 (2, 64, 64) float32 张量，具有正确的轴映射（x=[CTA]/[M], y=转化率）和通道分配（Ch0=Mn_norm, Ch1=分散度）。确定性输出已确认。无框架依赖。所有编码器测试通过。
  </done>
</task>

</tasks>

<verification>
- `pytest tests/test_ctfp_encoder.py -x -v` —— 所有编码器测试通过
- `python -c "from src.ctfp_encoder import transform; t = transform([(0.5, 0.3, 1.2, 1.5)]); print(t.shape, t.dtype)"` —— 输出 "torch.Size([2, 64, 64]) torch.float32"
- `python -c "import ast; src = open('src/ctfp_encoder.py').read(); assert 'streamlit' not in src; assert 'torch.nn' not in src; print('No framework deps')"` —— 输出 "No framework deps"
</verification>

<success_criteria>
- transform() 产生 (2, 64, 64) float32 张量
- 通道0 = Mn_norm，通道1 = 分散度，位于正确的像素位置
- 相同输入始终产生相同输出
- 模块无Streamlit或训练框架导入
- 全部10项编码器测试通过
</success_criteria>

<output>
完成后，创建 `.planning/phases/01-ode-foundation-and-ctfp-encoder/01-02-SUMMARY.md`
</output>

---

# 计划 01-03：ODE验证与诊断数据集

---
phase: 01-ode-foundation-and-ctfp-encoder
plan: 03
type: execute
wave: 2
depends_on: [01-01, 01-02]
files_modified:
  - src/raft_ode.py
  - src/diagnostic.py
  - tests/test_raft_ode.py
  - notebooks/01_ode_validation.ipynb
autonomous: false
requirements: [SIM-01, SIM-02, SIM-03, ENC-01]
must_haves:
  truths:
    - "3种RAFT试剂类型的ODE曲线在定性上与已发表文献趋势一致"
    - "极端参数极限检查通过（高Ctr -> 低分散度，低Ctr -> 高分散度）"
    - "1000样本诊断数据集在全部4种RAFT类型中以低于2%的失败率生成"
    - "诊断数据集的ctFP图像显示物理上合理的双通道模式"
  artifacts:
    - path: "src/diagnostic.py"
      provides: "诊断数据集生成脚本"
      exports: ["generate_diagnostic_dataset"]
    - path: "notebooks/01_ode_validation.ipynb"
      provides: "将ODE输出与文献趋势对比的交互式验证图"
    - path: "tests/test_raft_ode.py"
      provides: "扩展了诊断数据集冒烟测试"
      contains: "test_diagnostic_dataset"
  key_links:
    - from: "src/diagnostic.py"
      to: "src/raft_ode.py"
      via: "import simulate_raft, compute_retardation_factor, compute_inhibition_period"
      pattern: "from src.raft_ode import"
    - from: "src/diagnostic.py"
      to: "src/ctfp_encoder.py"
      via: "导入 transform 用于ctFP生成"
      pattern: "from src.ctfp_encoder import"
    - from: "notebooks/01_ode_validation.ipynb"
      to: "src/raft_ode.py"
      via: "导入 simulate_raft 用于验证图"
      pattern: "from src.raft_ode import"
---

<objective>
针对文献曲线和极端参数极限验证ODE系统，然后生成1000样本的诊断数据集，在提交到阶段2的百万级样本生成之前确认全参数空间内的数值稳定性。

目的：这是阶段2之前的最终质量关口。此处发现的任何ODE错误都可以避免完整数据集的重新生成。
输出：包含文献对比的验证笔记本，诊断数据集脚本，扩展测试。
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/phases/01-ode-foundation-and-ctfp-encoder/01-CONTEXT.md
@.planning/phases/01-ode-foundation-and-ctfp-encoder/01-RESEARCH.md
@.planning/phases/01-ode-foundation-and-ctfp-encoder/01-01-SUMMARY.md
@.planning/phases/01-ode-foundation-and-ctfp-encoder/01-02-SUMMARY.md

<interfaces>
<!-- 来自计划01：src/raft_ode.py -->
计划01完成后的预期导出：
```python
def simulate_raft(params: dict, raft_type: str = 'ttc', t_end: float = 36000, n_conv_points: int = 50) -> dict | None:
    """Returns {'conversion': np.ndarray, 'mn': np.ndarray, 'dispersity': np.ndarray, 'mn_norm': np.ndarray} or None on failure"""

def compute_retardation_factor(params: dict, raft_type: str, conv_target: float = 0.5) -> float:
    """Returns Rp(RAFT)/Rp(no CTA) in (0, 1]"""

def compute_inhibition_period(sol_dense, M0: float, t_end: float) -> float:
    """Returns t_inh/t_end in [0, 1]"""
```

<!-- 来自计划02：src/ctfp_encoder.py -->
```python
def transform(data, img_size=64) -> torch.Tensor:
    """data: iterable of (cta_ratio_norm, conversion, mn_norm, dispersity). Returns (2, 64, 64) float32 tensor."""
```
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>任务1：诊断数据集生成器和ODE验证笔记本</name>
  <files>src/diagnostic.py, notebooks/01_ode_validation.ipynb, tests/test_raft_ode.py</files>
  <read_first>
    - src/raft_ode.py （计划01的当前实现）
    - src/ctfp_encoder.py （计划02的编码器）
    - tests/test_raft_ode.py （待扩展的现有测试）
    - .planning/phases/01-ode-foundation-and-ctfp-encoder/01-RESEARCH.md （文献验证目标部分：体系1 CDB/苯乙烯，体系2 TTC/MMA，体系3 黄原酸酯/VAc；诊断数据集设计部分；速率常数范围表）
    - .planning/phases/01-ode-foundation-and-ctfp-encoder/01-CONTEXT.md （D-07, D-08, D-10, D-11）
  </read_first>
  <action>
1. 创建 `src/diagnostic.py`：

   **函数 `generate_diagnostic_dataset(n_per_type=250, seed=42)`：**
   - 按D-10：每种RAFT类型生成250个样本（二硫代酯、三硫代碳酸酯、黄原酸酯、二硫代氨基甲酸酯）= 共1000个
   - 按研究文档"诊断数据集设计"：每种类型10个Ctr值 x 25个 [CTA]/[M] 值
   - Ctr网格：np.logspace(-2, 4, 10)，按D-07
   - CTA/M网格：np.logspace(-3, -1, 25)，按D-08
   - 每种类型使用固定动力学参数（来自研究文档速率常数范围表）：
     - TTC/MMA：kp=650, kt=1e8, kd=1.5e-5, f=0.5, ki=1e4, M_monomer=100.12
     - 二硫代酯/苯乙烯：kp=340, kt=1e8, kd=1.5e-5, f=0.5, ki=1e4, M_monomer=104.15, kadd0=1e6, kfrag0=1.0
     - 黄原酸酯/VAc：kp=6700, kt=1e8, kd=1.5e-5, f=0.5, ki=1e4, M_monomer=86.09
     - 二硫代氨基甲酸酯/VAc：kp=6700, kt=1e8, kd=1.5e-5, f=0.5, ki=1e4, M_monomer=86.09（与黄原酸酯相同，较低的kadd）
   - 对每个 (Ctr, CTA_M_ratio)：从 Ctr*kp 推导 kadd（简化），设置 kfrag=1e4
   - 对每个样本调用 simulate_raft + compute_retardation_factor + compute_inhibition_period
   - 使用 joblib.Parallel(n_jobs=-1) 配合 tqdm 进度条
   - 返回字典，包含：params（字典列表），results（包含 conversion/mn/dispersity/mn_norm 的字典列表），labels（包含 log10_ctr, inhibition_period, retardation_factor 的字典列表），ctfp_tensors（来自 transform() 的 torch.Tensor 列表），failures（失败参数索引列表）
   - 打印摘要：总样本数、成功数、失败数、失败率

   **函数 `save_diagnostic(dataset, path='data/diagnostic_1000.npz')`：**
   - 将ctFP张量保存为堆叠numpy数组，标签保存为结构化数组
   - 如需要则创建 data/ 目录

2. 创建 `notebooks/01_ode_validation.ipynb`（作为可转换为笔记本的Python脚本）：

   笔记本应包含以下单元格：
   - **单元格1：** 导入 src.raft_ode, matplotlib
   - **单元格2：** 文献体系1 —— CDB/苯乙烯（二硫代酯）验证：
     - 使用匹配CDB/苯乙烯60°C的参数模拟：kp=340, kt=1e8, Ctr~20（典型CDB），[CTA]/[M]=0.005
     - 绘制 Mn 随转化率变化和分散度随转化率变化
     - 标注预期行为："可见诱导期"、"分散度先降后升"、"诱导期后Mn线性增长"
   - **单元格3：** 文献体系2 —— TTC/MMA 验证：
     - 使用匹配十二烷基-TTC/MMA 60°C的参数模拟：kp=650, Ctr~50, [CTA]/[M]=0.01
     - 绘制 Mn 随转化率变化和分散度随转化率变化
     - 标注："无诱导期"、"分散度 < 1.4"、"Mn线性增长"
   - **单元格4：** 文献体系3 —— 黄原酸酯/VAc 验证：
     - 使用匹配O-乙基黄原酸酯/VAc 60°C的参数模拟：kp=6700, Ctr~1, [CTA]/[M]=0.01
     - 绘制 Mn 随转化率变化和分散度随转化率变化
     - 标注："分散度在65%转化率前约为1.2然后上升"、"准一级动力学"
   - **单元格5：** 极端极限检查：
     - 高Ctr（10000）：中等转化率下分散度应 < 1.1
     - 低Ctr（0.01）：分散度应 > 1.5，Mn应偏离理想值
   - **单元格6：** 使用 matplotlib imshow 可视化每个体系的一个ctFP样本

3. 扩展 `tests/test_raft_ode.py`：
   - `test_diagnostic_dataset` 标记 `@pytest.mark.slow`：运行 generate_diagnostic_dataset(n_per_type=250)，断言 failure_rate < 0.02（1000个中少于20个失败）
   - `test_diagnostic_labels_valid`：对成功样本，断言 log10_ctr 在 [-2, 4]，inhibition_period 在 [0, 1]，retardation_factor 在 (0, 1]
  </action>
  <verify>
    <automated>cd C:/CodingCraft/DL/ViT-Ctr && pytest tests/test_raft_ode.py -x -v -k "not slow" && python -c "from src.diagnostic import generate_diagnostic_dataset; ds = generate_diagnostic_dataset(n_per_type=5); print(f'Success: {len(ds[\"results\"])} samples, {len(ds[\"failures\"])} failures')"</automated>
  </verify>
  <acceptance_criteria>
    - src/diagnostic.py 包含 `def generate_diagnostic_dataset(n_per_type=250`
    - src/diagnostic.py 包含 `from src.raft_ode import simulate_raft`
    - src/diagnostic.py 包含 `from src.ctfp_encoder import transform`
    - src/diagnostic.py 包含 `joblib.Parallel`
    - notebooks/01_ode_validation.ipynb 存在（或 notebooks/01_ode_validation.py）
    - tests/test_raft_ode.py 包含 `def test_diagnostic_dataset`
    - tests/test_raft_ode.py 包含 `pytest.mark.slow`
    - `python -c "from src.diagnostic import generate_diagnostic_dataset"` 退出码为0
    - 快速冒烟测试（n_per_type=5）无崩溃完成
  </acceptance_criteria>
  <done>
    诊断数据集生成器在全部4种RAFT类型中生成样本。ODE验证笔记本显示3个文献体系的定性正确曲线。扩展测试套件包含诊断冒烟测试。
  </done>
</task>

<task type="checkpoint:human-verify" gate="blocking">
  <name>任务2：验证ODE验证图和诊断数据集质量</name>
  <files>notebooks/01_ode_validation.ipynb</files>
  <action>
    用户目视验证ODE输出曲线是否与全部3个RAFT体系的预期文献行为一致，以及诊断数据集是否通过数值稳定性阈值。
  </action>
  <verify>
    <automated>cd C:/CodingCraft/DL/ViT-Ctr && pytest tests/ -v</automated>
  </verify>
  <done>用户确认ODE曲线与文献趋势一致且诊断数据集失败率低于2%。</done>
  <what-built>
    完整的RAFT ODE系统，包含针对3个文献体系（CDB/苯乙烯二硫代酯、TTC/MMA三硫代碳酸酯、黄原酸酯/VAc）的验证。覆盖完整参数空间的诊断数据集生成器。产生双通道指纹的ctFP编码器。
  </what-built>
  <how-to-verify>
    1. 打开 `notebooks/01_ode_validation.ipynb`（或运行等效的 .py 文件）
    2. 检查 CDB/苯乙烯（二硫代酯）图：Mn随转化率变化应显示延迟起始（诱导期），分散度应先降后升
    3. 检查 TTC/MMA 图：Mn从一开始就应与转化率线性相关，分散度应保持在1.4以下
    4. 检查 黄原酸酯/VAc 图：分散度在65%转化率前应约为1.2，然后逐渐上升
    5. 检查极端极限图：高Ctr给出极低分散度，低Ctr给出高分散度
    6. 检查ctFP可视化：双通道热力图应显示稀疏但物理上合理的模式
    7. 运行 `cd C:/CodingCraft/DL/ViT-Ctr && pytest tests/test_raft_ode.py tests/test_ctfp_encoder.py -v` —— 所有测试应通过
    8. 运行 `cd C:/CodingCraft/DL/ViT-Ctr && pytest tests/test_raft_ode.py -v -k slow` —— 诊断数据集应以低于2%的失败率完成
  </how-to-verify>
  <resume-signal>如果ODE曲线与预期文献行为一致且所有测试通过，请输入"approved"。如有具体问题请描述。</resume-signal>
</task>

</tasks>

<verification>
- `pytest tests/ -v` —— 所有测试通过（ODE + 编码器）
- `pytest tests/test_raft_ode.py -v -k slow` —— 诊断数据集失败率 < 2%
- 笔记本显示3个RAFT体系的定性正确ODE曲线
- ctFP可视化显示合理的双通道模式
</verification>

<success_criteria>
- ODE再现二硫代酯（诱导期+减速）、TTC（干净控制）、黄原酸酯（中等控制）的预期Mn/分散度趋势
- 1000样本诊断数据集以低于2%的失败率生成
- 所有测试通过（ODE和编码器两个测试套件）
- 用户批准验证检查点
</success_criteria>

<output>
完成后，创建 `.planning/phases/01-ode-foundation-and-ctfp-encoder/01-03-SUMMARY.md`
</output>
