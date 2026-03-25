"""
RAFT聚合动力学ODE模拟器 — 基于矩量方法

支持四种RAFT剂类型：
- dithioester: 两阶段预平衡模型（slow fragmentation导致inhibition period）
- trithiocarbonate (TTC): 单平衡模型
- xanthate: 单平衡模型
- dithiocarbamate: 单平衡模型

核心参考文献：
- Wilding et al. Macromolecules 2023 — 分散度建模
- Moad, Polymers 2022 — 部分矩量方法
- Barner-Kowollik et al. 2006+ — 预平衡动力学

矩量方程说明：
  RAFT机理的核心是退化链转移(degenerative chain transfer)：
    P_n· + S=C(Z)S-P_m  →  P_n-S-C·(Z)-S-P_m  →  P_n-S-C(Z)=S + P_m·
  即活性链与休眠链交换身份。这不是真正的终止反应。

  状态向量需要跟踪三类链的矩量：
  - mu: 活性（增长）自由基链矩量
  - nu: 休眠（macro-CTA）链矩量
  - lam: 真正的死链矩量（仅来自终止反应）

  RAFT交换过程中，活性链的矩量与休眠链的矩量发生交换：
  - 活性链(长度n)加成到休眠链(长度m): mu失去n，nu失去m
  - 中间体断裂释放新自由基(长度m)和新休眠链(长度n): mu获得m，nu获得n
  - 假设RAFT交换在矩量层面的净效应为mu和nu之间的矩量交换
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


def raft_ode_single_eq(t, y, kd, f, ki, kp, kt, kadd, kfrag):
    """
    单平衡RAFT聚合ODE（矩量方法，含休眠链追踪）。

    适用于TTC、xanthate、dithiocarbamate等RAFT剂类型。

    RAFT主平衡: P_n· + CTA_m ⇌ Int → P_m· + CTA_n
    kadd: P· + CTA → Int 的速率常数
    kfrag: Int → P· + CTA 的速率常数

    状态向量 y (14个变量):
      [0]  M    — 单体浓度
      [1]  I    — 引发剂浓度
      [2]  P    — 增长自由基总浓度 (sum over all chain lengths)
      [3]  Int  — RAFT中间体自由基浓度
      [4]  mu0  — 活性链零阶矩
      [5]  mu1  — 活性链一阶矩
      [6]  mu2  — 活性链二阶矩
      [7]  nu0  — 休眠链(macro-CTA)零阶矩
      [8]  nu1  — 休眠链一阶矩
      [9]  nu2  — 休眠链二阶矩
      [10] lam0 — 死链零阶矩
      [11] lam1 — 死链一阶矩
      [12] lam2 — 死链二阶矩
      [13] CTA  — 小分子RAFT剂浓度（初始RAFT剂，尚未参与过链转移）

    注意: nu (休眠链) 追踪的是已经参与过至少一次RAFT交换的macro-CTA。
    CTA[13]是初始的小分子RAFT剂。当P·加成到CTA时，产生的新dormant chain
    进入nu矩量（因为它现在带有高分子链段）。
    """
    M, I, P, Int, mu0, mu1, mu2, nu0, nu1, nu2, lam0, lam1, lam2, CTA = y

    # 确保浓度非负
    M = max(M, 0.0)
    I = max(I, 0.0)
    P = max(P, 0.0)
    Int = max(Int, 0.0)
    CTA = max(CTA, 0.0)
    mu0 = max(mu0, 0.0)
    nu0 = max(nu0, 0.0)

    # 基元反应速率
    R_init = 2.0 * f * kd * I          # 引发速率
    R_prop = kp * P * M                 # 增长速率
    R_term = kt * P**2                  # 双基终止 (combination/disproportionation)

    # RAFT交换速率
    # P· + CTA(小分子) → Int_init → R· + macro-CTA(进入nu)
    R_add_cta = kadd * P * CTA          # 加成到初始CTA

    # P· + macro-CTA(nu) ⇌ Int → P'· + macro-CTA'(nu)
    # 前向: kadd * P * nu0 (P·可与任何休眠链反应，速率与nu0成正比)
    # 实际上，RAFT交换速率 = kadd * [P·] * [macro-CTA]
    # 其中[macro-CTA] = nu0（休眠链的数目浓度）
    R_add_macro = kadd * P * nu0        # 加成到macro-CTA
    R_frag = kfrag * Int                # 中间体断裂

    # ---- 小分子物种平衡 ----
    dM_dt = -R_prop
    dI_dt = -kd * I

    # P·的平衡: 引发产生 + 断裂释放 - 加成消耗 - 终止消耗
    dP_dt = R_init + R_frag - R_add_cta - R_add_macro - 2.0 * R_term

    # 中间体: 加成产生 - 断裂消耗
    dInt_dt = R_add_cta + R_add_macro - R_frag

    # 初始CTA: 被加成消耗
    dCTA_dt = -R_add_cta

    # ---- 活性链矩量 (mu) ----
    # 引发: +1 chain of length 1
    # 增长: chain length grows by 1 per propagation event
    # 终止: chains removed from active pool
    # RAFT加成到CTA: active chain → intermediate → dormant chain (removed from mu)
    # RAFT加成到macro-CTA: active chain → intermediate (removed from mu)
    # RAFT断裂释放: old dormant chain → new active chain (added to mu)

    # RAFT exchange的矩量效应:
    # 当P_n·(活性)加成到macro-CTA_m(休眠):
    #   mu失去: n的贡献 → -kadd*P*nu0对mu0, -kadd*mu1/mu0*P*nu0对mu1(按mu的平均链长)
    #   实际上对于矩量方程，R_add_macro移除mu的"当前分布"
    # 当Int断裂释放P_m·(原休眠链长度m):
    #   mu获得: m的贡献 → 按nu的平均链长

    # 简化处理（quasi-steady-state for intermediate composition）：
    # Int断裂时，释放的radical chain length来自nu分布
    # Int形成时，消耗的radical chain length来自mu分布
    # 在稳态中间体假设下(dInt/dt≈0 on moment level)：
    # 净效应 = mu获得nu的矩量 - mu失去自己的矩量

    # 加成到初始CTA: P_n· + CTA → dormant_n (mu减少, nu增加)
    # R·(来自CTA的R基团)重新引发，产生新链(长度1)
    R_reinit_cta = R_add_cta  # R·从初始CTA释放，重新引发

    # RAFT交换(与macro-CTA): 矩量交换
    # 需要追踪进出中间体的矩量流
    # 进入Int的mu矩量: kadd * [mu_k/mu0 * mu0] * nu0 = kadd * mu_k * nu0 / mu0 * mu0...
    # 不对，更仔细地推导:
    # 加成速率对mu的k阶矩贡献 = -kadd * mu_k * nu0 / mu0 * P
    # 但这不太对。正确的推导:
    # 活性链P_n·加成到任意macro-CTA的总速率 = kadd * sum_n([P_n·]) * nu0
    # 对mu_k的贡献 = -kadd * (sum_n n^k [P_n·]) * nu0 = -kadd * mu_k * nu0
    # 错：这里nu0是macro-CTA的浓度，不是每条链的速率
    # 正确: 每条活性链P_n的加成速率 = kadd * [P_n·] * (CTA + nu0)
    # 但[P·] = sum_n [P_n·] = mu0
    # 所以对mu_k的消耗项 = kadd * mu_k * nu0 (对应加成到macro-CTA)
    # 加上 kadd * mu_k * CTA (对应加成到初始CTA, 但CTA是小分子)

    # 不对，重新想。P是自由基总浓度(标量)，mu是追踪链长分布的矩量。
    # 如果所有链长的P_n·以相同速率(kadd)加成到CTA/macro-CTA，
    # 那么对mu_k的消耗 = kadd * mu_k * (CTA + nu0) (不是*P)
    # 不不不。kadd * [P_n·] * [macro-CTA] 是species P_n·的消耗速率。
    # mu_k = sum_n n^k [P_n·], 所以 d(mu_k)/dt from addition = -kadd * mu_k * (CTA + nu0)

    # 但P = mu0 (自由基总浓度等于零阶矩)
    # 所以 R_add_cta = kadd * P * CTA = kadd * mu0 * CTA
    # R_add_macro = kadd * P * nu0 = kadd * mu0 * nu0

    # 对mu_k的RAFT加成消耗项: -kadd * mu_k * (CTA + nu0)
    # 这意味着大链和小链以相同的速率常数加成(chain-length independent kadd)

    # 断裂释放时: Int断裂产生两种radical:
    # 1. 来自初始CTA的R·(小分子radical, length~1) — 对应R_add_cta形成的Int
    # 2. 来自macro-CTA的P_m·(高分子radical) — 对应R_add_macro形成的Int

    # 在中间体混合假设下(well-mixed intermediate pool):
    # Int断裂释放radical的矩量来源 =
    #   (fraction from CTA) * 1(链长1) + (fraction from macro-CTA) * nu的平均链长

    # 但这太复杂了。用更实用的简化：
    # RAFT交换是快速平衡，主要效果是均一化活性链和休眠链的分布。
    # 用"moment exchange"模型:

    # 对于加成到macro-CTA后的断裂:
    # mu失去自己的矩量，获得nu的矩量
    # Rate of exchange = kadd * mu0 * nu0 (交换频率)
    # 注意: 不是所有加成都导致交换——Int可能反向断裂回到原来的radical。
    # 但在对称RAFT agent (kadd=kadd_reverse)下, 50%正向50%反向。
    # 这里简化: 假设每次加成-断裂cycle导致一次完整交换。

    # 加成到macro-CTA的速率: kadd * mu0 * nu0
    # 断裂重新释放radical的速率: kfrag * Int

    # 用R_exchange = min(R_add_macro, R_frag)作为有效交换速率
    # 不，直接用矩量方程。

    # 最终采用的简化模型:
    # 1) 加成到初始CTA: mu减少(链变为dormant)，CTA减少，nu增加，R·产生新链
    # 2) 加成到macro-CTA + 断裂(净效应): mu和nu交换矩量

    # 对于(2)的处理: 假设中间体寿命短(kfrag很大)，
    # 使用有效交换速率 R_ex = kadd * mu0 * nu0 (rate of exchange events)
    # 每次交换: mu失去一条链(按mu分布采样)，获得一条链(按nu分布采样)
    # nu失去一条链(按nu分布采样)，获得一条链(按mu分布采样)

    # 对mu_k: d(mu_k)/dt_exchange = R_ex * (nu_k/nu0 - mu_k/mu0)
    #                              = kadd * mu0 * nu0 * (nu_k/nu0 - mu_k/mu0)
    #                              = kadd * (mu0 * nu_k - mu_k * nu0)
    # 对nu_k: d(nu_k)/dt_exchange = kadd * (mu_k * nu0 - mu0 * nu_k)
    # 这保证了mu_k + nu_k的RAFT交换项为零（守恒）

    # 最终矩量方程:
    R_ex_rate = kadd  # 交换速率常数

    # 活性链矩量
    # 引发: +1 chain of length 1
    # 增长: mu_k grows
    # 终止: -kt * mu_k * mu0 (简化: P≈mu0)
    # 加成到初始CTA: -kadd * mu_k * CTA, 同时R·重引发产生length=1的新链
    # RAFT交换(与macro-CTA): +kadd*(mu0*nu_k - mu_k*nu0)

    mu0_safe = max(mu0, 1e-30)
    nu0_safe = max(nu0, 1e-30)

    # 加成到初始CTA的矩量消耗
    loss_cta_mu0 = kadd * mu0 * CTA
    loss_cta_mu1 = kadd * mu1 * CTA
    loss_cta_mu2 = kadd * mu2 * CTA

    # RAFT交换项 (与macro-CTA)
    ex_mu0 = R_ex_rate * (mu0 * nu0 - mu0 * nu0)  # = 0 for k=0 (数目守恒)
    ex_mu1 = R_ex_rate * (mu0 * nu1 - mu1 * nu0)
    ex_mu2 = R_ex_rate * (mu0 * nu2 - mu2 * nu0)

    dmu0_dt = (R_init + R_reinit_cta  # 引发和CTA释放R·重引发
               - 2.0 * kt * mu0 * mu0  # 终止 (P≈mu0)
               - loss_cta_mu0           # 加成到初始CTA
               + ex_mu0)               # RAFT交换

    dmu1_dt = (kp * M * mu0            # 增长
               + R_init + R_reinit_cta  # 引发(length=1)
               - 2.0 * kt * mu1 * mu0  # 终止
               - loss_cta_mu1           # 加成到初始CTA
               + ex_mu1)               # RAFT交换

    dmu2_dt = (kp * M * (2.0 * mu1 + mu0)  # 增长
               + R_init + R_reinit_cta       # 引发(length=1, 1^2=1)
               - 2.0 * kt * mu2 * mu0       # 终止
               - loss_cta_mu2               # 加成到初始CTA
               + ex_mu2)                    # RAFT交换

    # ---- 休眠链矩量 (nu) ----
    # 加成到初始CTA产生新的dormant chain:
    #   活性链P_n·+ CTA → Int → dormant_n + R·
    #   nu获得一条按mu分布采样的链
    gain_from_cta_nu0 = loss_cta_mu0  # = kadd * mu0 * CTA
    gain_from_cta_nu1 = loss_cta_mu1  # = kadd * mu1 * CTA
    gain_from_cta_nu2 = loss_cta_mu2  # = kadd * mu2 * CTA

    # RAFT交换项 (与mu相反)
    ex_nu0 = -ex_mu0  # = 0
    ex_nu1 = -ex_mu1
    ex_nu2 = -ex_mu2

    dnu0_dt = gain_from_cta_nu0 + ex_nu0
    dnu1_dt = gain_from_cta_nu1 + ex_nu1
    dnu2_dt = gain_from_cta_nu2 + ex_nu2

    # ---- 死链矩量 (lam) ---- 仅来自终止!
    # 双基终止(combination): 两条活性链合并
    # 终止产生的死链长度 = n + m (for combination) or n, m (disproportionation)
    # 用combination模型: 每次终止产生1条死链(长度n+m)
    # lam0 += kt*mu0^2 (每次终止产生1条死链)
    # lam1 += kt*2*mu0*mu1 (E[n+m] = E[n]+E[m])
    # lam2 += kt*(2*mu0*mu2 + 2*mu1^2) (E[(n+m)^2] = E[n^2]+E[m^2]+2E[n]E[m])
    # 但这里简化用disproportionation为主(两条链都变死)
    dlam0_dt = 2.0 * kt * mu0 * mu0  # 终止产生2条死链/event (disproportionation)
    dlam1_dt = 2.0 * kt * mu1 * mu0
    dlam2_dt = 2.0 * kt * mu2 * mu0

    return [dM_dt, dI_dt, dP_dt, dInt_dt,
            dmu0_dt, dmu1_dt, dmu2_dt,
            dnu0_dt, dnu1_dt, dnu2_dt,
            dlam0_dt, dlam1_dt, dlam2_dt,
            dCTA_dt]


def raft_ode_preequilibrium(t, y, kd, f, ki, kp, kt, kadd, kfrag, kadd0, kfrag0):
    """
    两阶段预平衡RAFT聚合ODE（矩量方法，含休眠链追踪）。

    专用于dithioester类RAFT剂。在单平衡模型基础上增加：
    - CTA_0: 初始RAFT剂（未发生链转移的原始形式）
    - Int_pre: 预平衡中间体

    预平衡反应: P_n· + CTA_0 → Int_pre → R· + macro-CTA_n
      (R·是CTA_0上的离去基团，macro-CTA_n是带有n长度链段的新休眠链)
    主平衡反应: P_n· + macro-CTA_m ⇌ Int → P_m· + macro-CTA_n

    kfrag0 << kfrag 导致dithioester特有的诱导期(inhibition period)。

    状态向量 y (16个变量):
      [0]-[13]: 同single_eq
      [14] CTA_0   — 初始RAFT剂浓度(小分子)
      [15] Int_pre — 预平衡中间体浓度
    """
    (M, I, P, Int, mu0, mu1, mu2, nu0, nu1, nu2,
     lam0, lam1, lam2, CTA, CTA_0, Int_pre) = y

    # 确保浓度非负
    M = max(M, 0.0)
    I = max(I, 0.0)
    P = max(P, 0.0)
    Int = max(Int, 0.0)
    CTA = max(CTA, 0.0)
    CTA_0 = max(CTA_0, 0.0)
    Int_pre = max(Int_pre, 0.0)
    mu0 = max(mu0, 0.0)
    nu0 = max(nu0, 0.0)

    # 预平衡反应速率
    R_add0 = kadd0 * P * CTA_0         # P·加成到初始CTA_0
    R_frag0 = kfrag0 * Int_pre         # 预平衡中间体断裂 → R· + macro-CTA

    # 主平衡反应速率
    R_init = 2.0 * f * kd * I
    R_prop = kp * P * M
    R_term = kt * P**2
    R_add_macro = kadd * P * nu0       # P·加成到macro-CTA(nu)
    R_frag_main = kfrag * Int          # 主平衡中间体断裂

    # 注意：dithioester的初始CTA(CTA_0)和macro-CTA(CTA, nu)是不同物种
    # CTA[13]在这里不使用——dithioester的所有初始CTA在CTA_0中
    # macro-CTA由预平衡断裂产生，存储在nu矩量中

    # ---- 小分子物种 ----
    dM_dt = -R_prop
    dI_dt = -kd * I
    dP_dt = R_init + R_frag_main + R_frag0 - R_add0 - R_add_macro - 2.0 * R_term
    dInt_dt = R_add_macro - R_frag_main
    dCTA_dt = 0.0  # dithioester不使用CTA[13]
    dCTA_0_dt = -R_add0
    dInt_pre_dt = R_add0 - R_frag0

    # ---- 活性链矩量 ----
    R_reinit = R_frag0  # R·从预平衡断裂重新引发(length=1)

    # 预平衡加成到CTA_0的矩量消耗
    # P_n·加成到CTA_0, 消耗mu中一条按mu分布的链
    loss_pre_mu0 = kadd0 * mu0 * CTA_0
    loss_pre_mu1 = kadd0 * mu1 * CTA_0
    loss_pre_mu2 = kadd0 * mu2 * CTA_0

    # RAFT主交换项(与macro-CTA)
    ex_mu0 = kadd * (mu0 * nu0 - mu0 * nu0)  # = 0
    ex_mu1 = kadd * (mu0 * nu1 - mu1 * nu0)
    ex_mu2 = kadd * (mu0 * nu2 - mu2 * nu0)

    dmu0_dt = (R_init + R_reinit
               - 2.0 * kt * mu0 * mu0
               - loss_pre_mu0
               + ex_mu0)

    dmu1_dt = (kp * M * mu0
               + R_init + R_reinit
               - 2.0 * kt * mu1 * mu0
               - loss_pre_mu1
               + ex_mu1)

    dmu2_dt = (kp * M * (2.0 * mu1 + mu0)
               + R_init + R_reinit
               - 2.0 * kt * mu2 * mu0
               - loss_pre_mu2
               + ex_mu2)

    # ---- 休眠链矩量 (nu) ----
    # 预平衡产生新dormant chain: P_n·的链段变成dormant_n
    gain_pre_nu0 = loss_pre_mu0
    gain_pre_nu1 = loss_pre_mu1
    gain_pre_nu2 = loss_pre_mu2

    # RAFT主交换(反号)
    ex_nu0 = -ex_mu0
    ex_nu1 = -ex_mu1
    ex_nu2 = -ex_mu2

    dnu0_dt = gain_pre_nu0 + ex_nu0
    dnu1_dt = gain_pre_nu1 + ex_nu1
    dnu2_dt = gain_pre_nu2 + ex_nu2

    # ---- 死链矩量 ----
    dlam0_dt = 2.0 * kt * mu0 * mu0
    dlam1_dt = 2.0 * kt * mu1 * mu0
    dlam2_dt = 2.0 * kt * mu2 * mu0

    return [dM_dt, dI_dt, dP_dt, dInt_dt,
            dmu0_dt, dmu1_dt, dmu2_dt,
            dnu0_dt, dnu1_dt, dnu2_dt,
            dlam0_dt, dlam1_dt, dlam2_dt,
            dCTA_dt, dCTA_0_dt, dInt_pre_dt]


def simulate_raft(params, raft_type='ttc', t_end=36000, n_conv_points=50):
    """
    运行RAFT聚合ODE模拟，返回均匀转化率网格上的Mn和分散度。

    Parameters
    ----------
    params : dict
        动力学参数字典，包含:
        - kd: 引发剂分解速率常数 (1/s)
        - f: 引发剂效率 (0-1)
        - ki: 引发反应速率常数 (L/mol/s)
        - kp: 增长速率常数 (L/mol/s)
        - kt: 终止速率常数 (L/mol/s)
        - kadd: RAFT加成速率常数 (L/mol/s)
        - kfrag: RAFT断裂速率常数 (1/s)
        - M0: 初始单体浓度 (mol/L)
        - I0: 初始引发剂浓度 (mol/L)
        - CTA0: 初始RAFT剂浓度 (mol/L)
        - M_monomer: 单体摩尔质量 (g/mol)
        - kadd0, kfrag0: (仅dithioester) 预平衡速率常数
    raft_type : str
        RAFT剂类型: 'dithioester', 'ttc', 'xanthate', 'dithiocarbamate'
    t_end : float
        模拟终止时间 (s)
    n_conv_points : int
        均匀转化率采样点数

    Returns
    -------
    dict or None
        包含 'conversion', 'mn', 'dispersity', 'mn_norm', 'time' 数组。
        如果ODE求解失败则返回 None。
    """
    kd = params['kd']
    f_eff = params['f']
    kp = params['kp']
    kt = params['kt']
    kadd = params['kadd']
    kfrag = params['kfrag']
    M0 = params['M0']
    I0 = params['I0']
    CTA0 = params['CTA0']
    M_monomer = params['M_monomer']

    if raft_type == 'dithioester':
        kadd0 = params['kadd0']
        kfrag0 = params['kfrag0']
        # y = [M, I, P, Int, mu0-2, nu0-2, lam0-2, CTA, CTA_0, Int_pre]
        y0 = np.array([
            M0, I0, 0.0, 0.0,      # M, I, P, Int
            0.0, 0.0, 0.0,          # mu0, mu1, mu2
            0.0, 0.0, 0.0,          # nu0, nu1, nu2
            0.0, 0.0, 0.0,          # lam0, lam1, lam2
            0.0,                     # CTA (macro, not used for dithioester)
            CTA0, 0.0,              # CTA_0, Int_pre
        ])

        atol = np.array([
            1e-6,   # M
            1e-10,  # I
            1e-14,  # P
            1e-14,  # Int
            1e-14, 1e-10, 1e-6,   # mu0, mu1, mu2
            1e-10, 1e-6, 1e-2,    # nu0, nu1, nu2
            1e-10, 1e-6, 1e-2,    # lam0, lam1, lam2
            1e-8,                  # CTA
            1e-8, 1e-14,           # CTA_0, Int_pre
        ])

        ode_func = raft_ode_preequilibrium
        ode_args = (kd, f_eff, None, kp, kt, kadd, kfrag, kadd0, kfrag0)
    else:
        # 单平衡模型
        # y = [M, I, P, Int, mu0-2, nu0-2, lam0-2, CTA]
        y0 = np.array([
            M0, I0, 0.0, 0.0,      # M, I, P, Int
            0.0, 0.0, 0.0,          # mu0, mu1, mu2
            0.0, 0.0, 0.0,          # nu0, nu1, nu2
            0.0, 0.0, 0.0,          # lam0, lam1, lam2
            CTA0,                    # CTA (小分子RAFT剂)
        ])

        atol = np.array([
            1e-6,   # M
            1e-10,  # I
            1e-14,  # P
            1e-14,  # Int
            1e-14, 1e-10, 1e-6,   # mu0, mu1, mu2
            1e-10, 1e-6, 1e-2,    # nu0, nu1, nu2
            1e-10, 1e-6, 1e-2,    # lam0, lam1, lam2
            1e-8,                  # CTA
        ])

        ode_func = raft_ode_single_eq
        ode_args = (kd, f_eff, None, kp, kt, kadd, kfrag)

    sol = solve_ivp(
        ode_func,
        t_span=(0, t_end),
        y0=y0,
        method='Radau',
        rtol=1e-8,
        atol=atol,
        dense_output=True,
        max_step=100.0,
        args=ode_args,
    )

    if not sol.success:
        return None

    # 均匀转化率网格采样
    target_conv = np.linspace(0.02, 0.95, n_conv_points)

    y_end = sol.sol(sol.t[-1])
    max_conv = 1.0 - y_end[0] / M0
    if max_conv < 0.02:
        return None

    target_conv = target_conv[target_conv <= max_conv - 0.005]
    if len(target_conv) < 3:
        return None

    # brentq找每个目标转化率对应的时间
    times = np.zeros(len(target_conv))
    for i, conv_target in enumerate(target_conv):
        def conv_residual(t, _ct=conv_target):
            y_t = sol.sol(t)
            return (1.0 - y_t[0] / M0) - _ct

        try:
            t_found = brentq(conv_residual, 0, sol.t[-1], xtol=1e-6)
            times[i] = t_found
        except ValueError:
            target_conv = target_conv[:i]
            times = times[:i]
            break

    if len(target_conv) < 3:
        return None

    # 计算Mn和分散度
    conversions = np.zeros(len(target_conv))
    mn_values = np.zeros(len(target_conv))
    dispersity_values = np.zeros(len(target_conv))

    Mn_theory = M0 / CTA0 * M_monomer

    for i, t_val in enumerate(times):
        y_t = sol.sol(t_val)
        M_t = max(y_t[0], 0.0)

        # 提取矩量
        mu0_t = max(y_t[4], 0.0)
        mu1_t = max(y_t[5], 0.0)
        mu2_t = max(y_t[6], 0.0)
        nu0_t = max(y_t[7], 0.0)
        nu1_t = max(y_t[8], 0.0)
        nu2_t = max(y_t[9], 0.0)
        lam0_t = max(y_t[10], 0.0)
        lam1_t = max(y_t[11], 0.0)
        lam2_t = max(y_t[12], 0.0)

        conv = 1.0 - M_t / M0
        conversions[i] = conv

        # 总矩量 = 活性链 + 休眠链 + 死链
        total_0 = mu0_t + nu0_t + lam0_t
        total_1 = mu1_t + nu1_t + lam1_t
        total_2 = mu2_t + nu2_t + lam2_t

        if total_0 > 1e-30 and total_1 > 1e-30:
            mn = total_1 / total_0 * M_monomer
            disp = total_2 * total_0 / (total_1**2)
        else:
            mn = 0.0
            disp = 1.0

        mn_values[i] = max(mn, 0.0)
        dispersity_values[i] = max(disp, 1.0)

    mn_norm = mn_values / Mn_theory if Mn_theory > 0 else mn_values

    return {
        'conversion': conversions,
        'mn': mn_values,
        'dispersity': dispersity_values,
        'mn_norm': mn_norm,
        'time': times,
    }


def compute_retardation_factor(params, raft_type='ttc', conv_target=0.5):
    """
    计算减速因子: Rp(RAFT) / Rp(no CTA)。

    通过运行两次ODE（有CTA和无CTA）计算在特定转化率处的速率比。

    Parameters
    ----------
    params : dict
        RAFT体系的动力学参数
    raft_type : str
        RAFT剂类型
    conv_target : float
        计算减速因子的目标转化率（默认0.5）

    Returns
    -------
    float
        减速因子（无量纲，范围0-1）。值越小表示减速越严重。
    """
    t_end = 36000

    sol_raft = _run_ode_for_rate(params, raft_type, t_end)
    if sol_raft is None:
        return 0.01

    frp_params = params.copy()
    frp_params['CTA0'] = 0.0
    frp_params['kadd'] = 0.0
    frp_params['kfrag'] = 0.0
    if 'kadd0' in frp_params:
        frp_params['kadd0'] = 0.0
        frp_params['kfrag0'] = 0.0
    sol_frp = _run_ode_for_rate(frp_params, 'ttc', t_end)
    if sol_frp is None:
        return 0.01

    M0 = params['M0']

    y_raft_end = sol_raft.sol(sol_raft.t[-1])
    max_conv_raft = 1.0 - y_raft_end[0] / M0

    if max_conv_raft < 0.05:
        return 0.01

    actual_conv = min(conv_target, max_conv_raft - 0.01)
    if actual_conv < 0.05:
        return 0.01

    rp_raft = _compute_rate_at_conv(sol_raft, M0, actual_conv, t_end)
    rp_frp = _compute_rate_at_conv(sol_frp, M0, actual_conv, t_end)

    if rp_frp is None or rp_frp < 1e-20:
        return 0.01
    if rp_raft is None or rp_raft < 1e-20:
        return 0.01

    retardation = rp_raft / rp_frp
    return min(max(retardation, 0.01), 1.0)


def compute_inhibition_period(sol_dense, M0, t_end):
    """
    计算诱导期: t_inh / t_end（无量纲，范围0-1）。

    t_inh为转化率首次达到1%的时间。

    Parameters
    ----------
    sol_dense : OdeSolution
        solve_ivp的dense_output（sol.sol）
    M0 : float
        初始单体浓度
    t_end : float
        总模拟时间

    Returns
    -------
    float
        无量纲诱导期（0-1）。
    """
    conv_threshold = 0.01

    def conv_residual(t):
        y_t = sol_dense(t)
        return (1.0 - y_t[0] / M0) - conv_threshold

    y_end = sol_dense(t_end)
    conv_end = 1.0 - y_end[0] / M0
    if conv_end < conv_threshold:
        return 1.0

    y_start = sol_dense(0)
    conv_start = 1.0 - y_start[0] / M0
    if conv_start >= conv_threshold:
        return 0.0

    try:
        t_inh = brentq(conv_residual, 0, t_end, xtol=1.0)
        return t_inh / t_end
    except ValueError:
        return 1.0


def _run_ode_for_rate(params, raft_type, t_end):
    """
    运行ODE并返回solve_ivp的Solution对象。内部辅助函数。
    """
    kd = params['kd']
    f_eff = params['f']
    kp = params['kp']
    kt = params['kt']
    kadd = params['kadd']
    kfrag = params['kfrag']
    M0 = params['M0']
    I0 = params['I0']
    CTA0 = params['CTA0']

    if raft_type == 'dithioester' and 'kadd0' in params and params.get('kadd0', 0) > 0:
        kadd0 = params['kadd0']
        kfrag0 = params['kfrag0']
        y0 = np.array([
            M0, I0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, CTA0, 0.0,
        ])
        atol = np.array([
            1e-6, 1e-10, 1e-14, 1e-14,
            1e-14, 1e-10, 1e-6,
            1e-10, 1e-6, 1e-2,
            1e-10, 1e-6, 1e-2,
            1e-8, 1e-8, 1e-14,
        ])
        ode_func = raft_ode_preequilibrium
        ode_args = (kd, f_eff, None, kp, kt, kadd, kfrag, kadd0, kfrag0)
    else:
        y0 = np.array([
            M0, I0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            CTA0,
        ])
        atol = np.array([
            1e-6, 1e-10, 1e-14, 1e-14,
            1e-14, 1e-10, 1e-6,
            1e-10, 1e-6, 1e-2,
            1e-10, 1e-6, 1e-2,
            1e-8,
        ])
        ode_func = raft_ode_single_eq
        ode_args = (kd, f_eff, None, kp, kt, kadd, kfrag)

    sol = solve_ivp(
        ode_func,
        t_span=(0, t_end),
        y0=y0,
        method='Radau',
        rtol=1e-8,
        atol=atol,
        dense_output=True,
        max_step=100.0,
        args=ode_args,
    )

    if not sol.success:
        return None
    return sol


def _compute_rate_at_conv(sol, M0, conv_target, t_end):
    """
    计算给定转化率处的聚合速率 Rp = -dM/dt / M0。
    """
    def conv_residual(t):
        y_t = sol.sol(t)
        return (1.0 - y_t[0] / M0) - conv_target

    try:
        t_target = brentq(conv_residual, 0, sol.t[-1], xtol=1e-6)
    except ValueError:
        return None

    dt = min(1.0, t_end * 1e-6)
    t_lo = max(0, t_target - dt)
    t_hi = min(sol.t[-1], t_target + dt)

    y_lo = sol.sol(t_lo)
    y_hi = sol.sol(t_hi)

    dM_dt = (y_hi[0] - y_lo[0]) / (t_hi - t_lo)
    rp = -dM_dt / M0

    return max(rp, 0.0)
