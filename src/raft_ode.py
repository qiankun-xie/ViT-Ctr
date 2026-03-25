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
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


def raft_ode_single_eq(t, y, kd, f, ki, kp, kt, kadd, kfrag):
    """
    单平衡RAFT聚合ODE（矩量方法）。

    适用于TTC、xanthate、dithiocarbamate等RAFT剂类型。

    状态向量 y (11个变量):
      [0] M    — 单体浓度
      [1] I    — 引发剂浓度
      [2] CTA  — RAFT剂浓度（大分子RAFT剂）
      [3] P    — 增长自由基总浓度
      [4] Int  — RAFT中间体自由基浓度
      [5] mu0  — 活性链零阶矩
      [6] mu1  — 活性链一阶矩
      [7] mu2  — 活性链二阶矩
      [8] lam0 — 死链零阶矩
      [9] lam1 — 死链一阶矩
      [10] lam2 — 死链二阶矩
    """
    M, I, CTA, P, Int, mu0, mu1, mu2, lam0, lam1, lam2 = y

    # 确保浓度非负（数值误差可能导致微小负值）
    M = max(M, 0.0)
    I = max(I, 0.0)
    CTA = max(CTA, 0.0)
    P = max(P, 0.0)
    Int = max(Int, 0.0)

    # 基元反应速率
    R_init = 2.0 * f * kd * I          # 引发速率
    R_prop = kp * P * M                # 增长速率
    R_add = kadd * P * CTA             # 加成到RAFT剂
    R_frag = kfrag * Int               # RAFT中间体断裂
    R_term = kt * P**2                 # 双基终止

    # 小分子物种平衡
    dM_dt = -R_prop
    dI_dt = -kd * I
    dCTA_dt = -R_add + R_frag
    dP_dt = R_init + R_frag - R_add - 2.0 * R_term
    dInt_dt = R_add - R_frag

    # 活性链矩量（增长自由基）
    # 引发产生链长为1的新链
    dmu0_dt = R_init - 2.0 * kt * mu0 * P
    dmu1_dt = kp * M * mu0 + R_init - 2.0 * kt * mu1 * P
    dmu2_dt = kp * M * (2.0 * mu1 + mu0) + R_init - 2.0 * kt * mu2 * P

    # 死链矩量（终止产物 + 链转移产物）
    dlam0_dt = kt * mu0 * P + kadd * mu0 * CTA
    dlam1_dt = kt * mu1 * P + kadd * mu1 * CTA
    dlam2_dt = kt * mu2 * P + kadd * mu2 * CTA

    return [dM_dt, dI_dt, dCTA_dt, dP_dt, dInt_dt,
            dmu0_dt, dmu1_dt, dmu2_dt, dlam0_dt, dlam1_dt, dlam2_dt]


def raft_ode_preequilibrium(t, y, kd, f, ki, kp, kt, kadd, kfrag, kadd0, kfrag0):
    """
    两阶段预平衡RAFT聚合ODE（矩量方法）。

    专用于dithioester类RAFT剂。在单平衡模型基础上增加：
    - CTA_0: 初始RAFT剂（未发生链转移的原始形式）
    - Int_pre: 预平衡中间体

    预平衡反应: P· + CTA_0 → Int_pre → R· + macro-CTA
    主平衡反应: P· + macro-CTA → Int → P'· + macro-CTA（同单平衡）

    kfrag0 << kfrag 导致dithioester特有的诱导期(inhibition period)。

    状态向量 y (13个变量):
      [0]-[10]: 同single_eq
      [11] CTA_0   — 初始RAFT剂浓度
      [12] Int_pre — 预平衡中间体浓度
    """
    M, I, CTA, P, Int, mu0, mu1, mu2, lam0, lam1, lam2, CTA_0, Int_pre = y

    # 确保浓度非负
    M = max(M, 0.0)
    I = max(I, 0.0)
    CTA = max(CTA, 0.0)
    P = max(P, 0.0)
    Int = max(Int, 0.0)
    CTA_0 = max(CTA_0, 0.0)
    Int_pre = max(Int_pre, 0.0)

    # 预平衡反应速率
    R_add0 = kadd0 * P * CTA_0         # 加成到初始RAFT剂
    R_frag0 = kfrag0 * Int_pre         # 预平衡中间体断裂 → R· + macro-CTA

    # 主平衡反应速率
    R_init = 2.0 * f * kd * I
    R_prop = kp * P * M
    R_add = kadd * P * CTA
    R_frag = kfrag * Int
    R_term = kt * P**2

    # 小分子物种平衡
    dM_dt = -R_prop
    dI_dt = -kd * I
    # macro-CTA由预平衡断裂产生，由主平衡加成消耗，由主平衡断裂再生
    dCTA_dt = R_frag0 - R_add + R_frag
    dP_dt = R_init + R_frag + R_frag0 - R_add - R_add0 - 2.0 * R_term
    dInt_dt = R_add - R_frag

    # 初始RAFT剂和预平衡中间体
    dCTA_0_dt = -R_add0
    dInt_pre_dt = R_add0 - R_frag0

    # 活性链矩量
    # R·从预平衡断裂释放后重新引发聚合，贡献同引发
    R_reinit = R_frag0  # R·重新引发
    dmu0_dt = R_init + R_reinit - 2.0 * kt * mu0 * P
    dmu1_dt = kp * M * mu0 + R_init + R_reinit - 2.0 * kt * mu1 * P
    dmu2_dt = kp * M * (2.0 * mu1 + mu0) + R_init + R_reinit - 2.0 * kt * mu2 * P

    # 死链矩量
    dlam0_dt = kt * mu0 * P + kadd * mu0 * CTA + kadd0 * mu0 * CTA_0
    dlam1_dt = kt * mu1 * P + kadd * mu1 * CTA + kadd0 * mu1 * CTA_0
    dlam2_dt = kt * mu2 * P + kadd * mu2 * CTA + kadd0 * mu2 * CTA_0

    return [dM_dt, dI_dt, dCTA_dt, dP_dt, dInt_dt,
            dmu0_dt, dmu1_dt, dmu2_dt, dlam0_dt, dlam1_dt, dlam2_dt,
            dCTA_0_dt, dInt_pre_dt]


def simulate_raft(params, raft_type='ttc', t_end=36000, n_conv_points=50):
    """
    运行RAFT聚合ODE模拟，返回均匀转化率网格上的Mn和分散度。

    Parameters
    ----------
    params : dict
        动力学参数字典，包含:
        - kd: 引发剂分解速率常数 (1/s)
        - f: 引发剂效率 (0-1)
        - ki: 引发反应速率常数 (L/mol/s)，本模型中隐含在R_init中
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
    # 提取参数
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

    # 构建初始条件
    if raft_type == 'dithioester':
        # 预平衡模型: 所有RAFT剂以初始形式存在
        kadd0 = params['kadd0']
        kfrag0 = params['kfrag0']
        # y = [M, I, CTA(macro), P, Int, mu0-mu2, lam0-lam2, CTA_0, Int_pre]
        y0 = np.array([M0, I0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        CTA0, 0.0])

        # 每组分绝对容差
        atol = np.array([
            1e-6,   # M
            1e-10,  # I
            1e-8,   # CTA (macro)
            1e-14,  # P
            1e-14,  # Int
            1e-14, 1e-10, 1e-6,   # mu0, mu1, mu2
            1e-10, 1e-6, 1e-2,    # lam0, lam1, lam2
            1e-8,   # CTA_0
            1e-14,  # Int_pre
        ])

        ode_func = raft_ode_preequilibrium
        ode_args = (kd, f_eff, None, kp, kt, kadd, kfrag, kadd0, kfrag0)
    else:
        # 单平衡模型
        # y = [M, I, CTA, P, Int, mu0-mu2, lam0-lam2]
        y0 = np.array([M0, I0, CTA0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        atol = np.array([
            1e-6,   # M
            1e-10,  # I
            1e-8,   # CTA
            1e-14,  # P
            1e-14,  # Int
            1e-14, 1e-10, 1e-6,   # mu0, mu1, mu2
            1e-10, 1e-6, 1e-2,    # lam0, lam1, lam2
        ])

        ode_func = raft_ode_single_eq
        ode_args = (kd, f_eff, None, kp, kt, kadd, kfrag)

    # 求解ODE
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

    # 用dense_output在均匀转化率网格上采样
    target_conv = np.linspace(0.02, 0.95, n_conv_points)

    # 首先确定最大可达转化率
    y_end = sol.sol(sol.t[-1])
    max_conv = 1.0 - y_end[0] / M0
    if max_conv < 0.02:
        # 转化率太低，无法有意义地采样
        return None

    # 限制目标转化率不超过实际可达值
    target_conv = target_conv[target_conv <= max_conv - 0.005]
    if len(target_conv) < 3:
        return None

    # 用brentq对每个目标转化率找到对应时间
    times = np.zeros(len(target_conv))
    for i, conv_target in enumerate(target_conv):
        def conv_residual(t):
            y_t = sol.sol(t)
            return (1.0 - y_t[0] / M0) - conv_target

        try:
            t_found = brentq(conv_residual, 0, sol.t[-1], xtol=1e-6)
            times[i] = t_found
        except ValueError:
            # 如果brentq无法找到根（转化率非单调等边缘情况）
            # 截断到已找到的点
            target_conv = target_conv[:i]
            times = times[:i]
            break

    if len(target_conv) < 3:
        return None

    # 在找到的时间点上计算Mn和分散度
    conversions = np.zeros(len(target_conv))
    mn_values = np.zeros(len(target_conv))
    dispersity_values = np.zeros(len(target_conv))

    Mn_theory = M0 / CTA0 * M_monomer  # 理论Mn（用于归一化）

    for i, t in enumerate(times):
        y_t = sol.sol(t)
        M_t = max(y_t[0], 0.0)
        mu0_t = max(y_t[5], 1e-30)
        mu1_t = max(y_t[6], 1e-30)
        mu2_t = max(y_t[7], 0.0)
        lam0_t = max(y_t[8], 0.0)
        lam1_t = max(y_t[9], 0.0)
        lam2_t = max(y_t[10], 0.0)

        conv = 1.0 - M_t / M0
        conversions[i] = conv

        # 总矩量（活性链 + 死链）
        total_lam0 = mu0_t + lam0_t
        total_lam1 = mu1_t + lam1_t
        total_lam2 = mu2_t + lam2_t

        if total_lam0 > 1e-30 and total_lam1 > 1e-30:
            mn = total_lam1 / total_lam0 * M_monomer
            disp = total_lam2 * total_lam0 / (total_lam1**2)
        else:
            mn = 0.0
            disp = 1.0

        # 物理约束
        mn_values[i] = max(mn, 0.0)
        dispersity_values[i] = max(disp, 1.0)

    # 归一化Mn
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

    通过运行两次ODE（有CTA和无CTA）来计算在特定转化率处的速率比。

    Parameters
    ----------
    params : dict
        RAFT体系的动力学参数（同simulate_raft）
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

    # 1. 运行RAFT体系（有CTA）
    sol_raft = _run_ode_for_rate(params, raft_type, t_end)
    if sol_raft is None:
        return 0.01  # 近零哨兵值

    # 2. 运行FRP体系（无CTA）
    frp_params = params.copy()
    frp_params['CTA0'] = 0.0
    frp_params['kadd'] = 0.0
    frp_params['kfrag'] = 0.0
    if 'kadd0' in frp_params:
        frp_params['kadd0'] = 0.0
        frp_params['kfrag0'] = 0.0
    sol_frp = _run_ode_for_rate(frp_params, 'ttc', t_end)  # 无CTA时用单平衡即可
    if sol_frp is None:
        return 0.01

    M0 = params['M0']

    # 确定RAFT体系最大转化率
    y_raft_end = sol_raft.sol(sol_raft.t[-1])
    max_conv_raft = 1.0 - y_raft_end[0] / M0

    if max_conv_raft < 0.05:
        # RAFT体系几乎没有聚合，返回近零哨兵值
        return 0.01

    # 使用两个体系中较低的转化率
    actual_conv = min(conv_target, max_conv_raft - 0.01)
    if actual_conv < 0.05:
        return 0.01

    # 计算各体系在目标转化率处的速率
    rp_raft = _compute_rate_at_conv(sol_raft, M0, actual_conv, t_end)
    rp_frp = _compute_rate_at_conv(sol_frp, M0, actual_conv, t_end)

    if rp_frp is None or rp_frp < 1e-20:
        return 0.01
    if rp_raft is None or rp_raft < 1e-20:
        return 0.01

    retardation = rp_raft / rp_frp
    # 限制在合理范围
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

    # 检查终点是否达到1%转化率
    y_end = sol_dense(t_end)
    conv_end = 1.0 - y_end[0] / M0
    if conv_end < conv_threshold:
        return 1.0  # 从未达到1%

    # 检查起点
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
    运行ODE并返回solve_ivp的Solution对象（供速率计算使用）。

    内部辅助函数。
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
        y0 = np.array([M0, I0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        CTA0, 0.0])
        atol = np.array([1e-6, 1e-10, 1e-8, 1e-14, 1e-14,
                          1e-14, 1e-10, 1e-6, 1e-10, 1e-6, 1e-2,
                          1e-8, 1e-14])
        ode_func = raft_ode_preequilibrium
        ode_args = (kd, f_eff, None, kp, kt, kadd, kfrag, kadd0, kfrag0)
    else:
        y0 = np.array([M0, I0, CTA0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        atol = np.array([1e-6, 1e-10, 1e-8, 1e-14, 1e-14,
                          1e-14, 1e-10, 1e-6, 1e-10, 1e-6, 1e-2])
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

    使用dense_output和数值微分。
    """
    def conv_residual(t):
        y_t = sol.sol(t)
        return (1.0 - y_t[0] / M0) - conv_target

    try:
        t_target = brentq(conv_residual, 0, sol.t[-1], xtol=1e-6)
    except ValueError:
        return None

    # 数值微分计算dM/dt
    dt = min(1.0, t_end * 1e-6)
    t_lo = max(0, t_target - dt)
    t_hi = min(sol.t[-1], t_target + dt)

    y_lo = sol.sol(t_lo)
    y_hi = sol.sol(t_hi)

    dM_dt = (y_hi[0] - y_lo[0]) / (t_hi - t_lo)
    rp = -dM_dt / M0  # 正值

    return max(rp, 0.0)
