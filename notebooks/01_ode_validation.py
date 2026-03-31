# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # RAFT ODE Validation Notebook
#
# 验证ODE系统对3种文献RAFT体系的定性行为：
# 1. CDB/Styrene (dithioester) -- 应有诱导期和减速效应
# 2. Dodecyl TTC/MMA (trithiocarbonate) -- 线性Mn增长，低D
# 3. O-Ethyl Xanthate/VAc (xanthate) -- Ctr~1, D ~ 2.0 (弱控制)
#
# 以及极端参数极限检查和ctFP可视化。

# %%
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互后端，适合脚本模式
import matplotlib.pyplot as plt

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.raft_ode import simulate_raft, compute_retardation_factor, compute_inhibition_period, _run_ode_for_rate
from src.ctfp_encoder import transform

# %% [markdown]
# ## Cell 2: Literature System 1 -- CDB/Styrene (Dithioester)
#
# CDB (cumyl dithiobenzoate) / Styrene at 60C
# - 应有**明显的诱导期** (slow fragmentation of pre-equilibrium intermediate)
# - D先下降后上升 (dip then rise)
# - Mn在诱导期后线性增长
# - 典型 Ctr ~ 20

# %%
# System 1: CDB/Styrene (dithioester)
params_cdb = {
    'kd': 1.5e-5, 'f': 0.5, 'ki': 1e4,
    'kp': 340,      # Styrene at 60C
    'kt': 1e8,
    'kadd': 340 * 20,   # Ctr ~ 20 -> kadd = Ctr * kp
    'kfrag': 1e4,
    'kadd0': 1e6,    # 预平衡加成（快）
    'kfrag0': 1.0,   # 预平衡断裂（慢 -> inhibition）
    'M0': 1.0, 'I0': 0.01, 'CTA0': 0.005,
    'M_monomer': 104.15,
}

result_cdb = simulate_raft(params_cdb, raft_type='dithioester')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('System 1: CDB/Styrene (Dithioester) -- Ctr~20, [CTA]/[M]=0.005', fontsize=14)

if result_cdb is not None:
    conv = result_cdb['conversion']
    mn = result_cdb['mn']
    disp = result_cdb['dispersity']

    axes[0].plot(conv, mn, 'b-o', markersize=3)
    axes[0].set_xlabel('Conversion')
    axes[0].set_ylabel('Mn (g/mol)')
    axes[0].set_title('Mn vs Conversion')
    axes[0].annotate('Expect: Mn linear after inhibition period',
                     xy=(0.5, 0.95), xycoords='axes fraction', fontsize=9,
                     ha='center', va='top', color='red')

    axes[1].plot(conv, disp, 'r-o', markersize=3)
    axes[1].set_xlabel('Conversion')
    axes[1].set_ylabel('Dispersity (Mw/Mn)')
    axes[1].set_title('Dispersity vs Conversion')
    axes[1].annotate('Expect: D dip then rise (inhibition signature)',
                     xy=(0.5, 0.95), xycoords='axes fraction', fontsize=9,
                     ha='center', va='top', color='red')

    # 计算诱导期
    sol_cdb = _run_ode_for_rate(params_cdb, 'dithioester', 36000)
    if sol_cdb is not None:
        inh_cdb = compute_inhibition_period(sol_cdb.sol, params_cdb['M0'], 36000)
        axes[0].annotate(f'Inhibition period: {inh_cdb:.3f} (t_inh/t_total)',
                         xy=(0.5, 0.85), xycoords='axes fraction', fontsize=9, ha='center')
else:
    axes[0].text(0.5, 0.5, 'SIMULATION FAILED', ha='center', va='center',
                 transform=axes[0].transAxes, fontsize=16, color='red')
    axes[1].text(0.5, 0.5, 'SIMULATION FAILED', ha='center', va='center',
                 transform=axes[1].transAxes, fontsize=16, color='red')

plt.tight_layout()
plt.savefig('notebooks/system1_cdb_styrene.png', dpi=150)
plt.close()
print("System 1 plot saved: notebooks/system1_cdb_styrene.png")

# %% [markdown]
# ## Cell 3: Literature System 2 -- TTC/MMA (Trithiocarbonate)
#
# Dodecyl trithiocarbonate / MMA at 60C
# - **无诱导期** (fast fragmentation)
# - D < 1.4 throughout
# - Mn线性增长
# - 典型 Ctr ~ 50

# %%
# System 2: TTC/MMA
params_ttc = {
    'kd': 1.5e-5, 'f': 0.5, 'ki': 1e4,
    'kp': 650,      # MMA at 60C
    'kt': 1e8,
    'kadd': 650 * 50,   # Ctr ~ 50
    'kfrag': 1e4,
    'M0': 1.0, 'I0': 0.01, 'CTA0': 0.01,
    'M_monomer': 100.12,
}

result_ttc = simulate_raft(params_ttc, raft_type='ttc')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('System 2: Dodecyl TTC/MMA -- Ctr~50, [CTA]/[M]=0.01', fontsize=14)

if result_ttc is not None:
    conv = result_ttc['conversion']
    mn = result_ttc['mn']
    disp = result_ttc['dispersity']

    axes[0].plot(conv, mn, 'b-o', markersize=3)
    axes[0].set_xlabel('Conversion')
    axes[0].set_ylabel('Mn (g/mol)')
    axes[0].set_title('Mn vs Conversion')
    axes[0].annotate('Expect: linear Mn growth (no inhibition)',
                     xy=(0.5, 0.95), xycoords='axes fraction', fontsize=9,
                     ha='center', va='top', color='green')

    # Mn理论线
    Mn_theory = params_ttc['M0'] / params_ttc['CTA0'] * params_ttc['M_monomer']
    axes[0].plot(conv, conv * Mn_theory, 'k--', alpha=0.5, label='Mn_theory * conv')
    axes[0].legend()

    axes[1].plot(conv, disp, 'r-o', markersize=3)
    axes[1].set_xlabel('Conversion')
    axes[1].set_ylabel('Dispersity (Mw/Mn)')
    axes[1].set_title('Dispersity vs Conversion')
    axes[1].axhline(y=1.4, color='gray', linestyle='--', alpha=0.5, label='D=1.4')
    axes[1].annotate('Expect: D < 1.4 throughout',
                     xy=(0.5, 0.95), xycoords='axes fraction', fontsize=9,
                     ha='center', va='top', color='green')
    axes[1].legend()
else:
    axes[0].text(0.5, 0.5, 'SIMULATION FAILED', ha='center', va='center',
                 transform=axes[0].transAxes, fontsize=16, color='red')
    axes[1].text(0.5, 0.5, 'SIMULATION FAILED', ha='center', va='center',
                 transform=axes[1].transAxes, fontsize=16, color='red')

plt.tight_layout()
plt.savefig('notebooks/system2_ttc_mma.png', dpi=150)
plt.close()
print("System 2 plot saved: notebooks/system2_ttc_mma.png")

# %% [markdown]
# ## Cell 4: Literature System 3 -- Xanthate/VAc
#
# O-ethyl xanthate / Vinyl Acetate at 60C
# - Mn基本恒定（Ctr=1时CTA与单体同速消耗→链数与质量同比增长）
# - D ~ 2.0 (Ctr=1控制力弱，接近FRP的D=2)
# - 典型 Ctr ~ 1 (less active RAFT agent)

# %%
# System 3: Xanthate/VAc
params_xan = {
    'kd': 1.5e-5, 'f': 0.5, 'ki': 1e4,
    'kp': 6700,     # VAc at 60C
    'kt': 1e8,
    'kadd': 6700 * 1,   # Ctr ~ 1
    'kfrag': 1e4,
    'M0': 1.0, 'I0': 0.01, 'CTA0': 0.01,
    'M_monomer': 86.09,
}

result_xan = simulate_raft(params_xan, raft_type='xanthate')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('System 3: O-Ethyl Xanthate/VAc -- Ctr~1, [CTA]/[M]=0.01', fontsize=14)

if result_xan is not None:
    conv = result_xan['conversion']
    mn = result_xan['mn']
    disp = result_xan['dispersity']

    axes[0].plot(conv, mn, 'b-o', markersize=3)
    axes[0].set_xlabel('Conversion')
    axes[0].set_ylabel('Mn (g/mol)')
    axes[0].set_title('Mn vs Conversion')
    axes[0].annotate('Ctr=1: Mn roughly constant (CTA consumed with monomer)',
                     xy=(0.5, 0.95), xycoords='axes fraction', fontsize=9,
                     ha='center', va='top', color='orange')

    axes[1].plot(conv, disp, 'r-o', markersize=3)
    axes[1].set_xlabel('Conversion')
    axes[1].set_ylabel('Dispersity (Mw/Mn)')
    axes[1].set_title('Dispersity vs Conversion')
    axes[1].axhline(y=2.0, color='gray', linestyle='--', alpha=0.5, label='D=2.0')
    axes[1].annotate('Ctr=1: D ~ 2.0 (weak RAFT control, near FRP)',
                     xy=(0.5, 0.95), xycoords='axes fraction', fontsize=9,
                     ha='center', va='top', color='orange')
    axes[1].legend()
else:
    axes[0].text(0.5, 0.5, 'SIMULATION FAILED', ha='center', va='center',
                 transform=axes[0].transAxes, fontsize=16, color='red')
    axes[1].text(0.5, 0.5, 'SIMULATION FAILED', ha='center', va='center',
                 transform=axes[1].transAxes, fontsize=16, color='red')

plt.tight_layout()
plt.savefig('notebooks/system3_xanthate_vac.png', dpi=150)
plt.close()
print("System 3 plot saved: notebooks/system3_xanthate_vac.png")

# %% [markdown]
# ## Cell 5: Extreme Limit Checks
#
# - **High Ctr (10000):** D should be ~ 1.1-1.2 (dead chains add ~0.1 above Poisson)
# - **Low Ctr (0.01):** D ~ 2.0 (approaching FRP), Mn decreasing

# %%
# Extreme limit checks
params_high = {
    'kd': 1.5e-5, 'f': 0.5, 'ki': 1e4,
    'kp': 650, 'kt': 1e8,
    'kadd': 650 * 10000,  # Ctr ~ 10000
    'kfrag': 1e4,
    'M0': 1.0, 'I0': 0.01, 'CTA0': 0.01,
    'M_monomer': 100.12,
}

params_low = {
    'kd': 1.5e-5, 'f': 0.5, 'ki': 1e4,
    'kp': 650, 'kt': 1e8,
    'kadd': 650 * 0.01,  # Ctr ~ 0.01
    'kfrag': 1e4,
    'M0': 1.0, 'I0': 0.01, 'CTA0': 0.01,
    'M_monomer': 100.12,
}

result_high = simulate_raft(params_high, raft_type='ttc')
result_low = simulate_raft(params_low, raft_type='ttc')

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Extreme Parameter Limit Checks', fontsize=14)

# High Ctr
if result_high is not None:
    conv_h = result_high['conversion']
    mn_h = result_high['mn']
    disp_h = result_high['dispersity']

    axes[0, 0].plot(conv_h, mn_h, 'b-o', markersize=3)
    axes[0, 0].set_title('High Ctr (10000): Mn vs Conv')
    axes[0, 0].set_xlabel('Conversion')
    axes[0, 0].set_ylabel('Mn (g/mol)')

    axes[0, 1].plot(conv_h, disp_h, 'r-o', markersize=3)
    axes[0, 1].set_title('High Ctr (10000): D vs Conv')
    axes[0, 1].set_xlabel('Conversion')
    axes[0, 1].set_ylabel('Dispersity')
    axes[0, 1].axhline(y=1.2, color='gray', linestyle='--', alpha=0.5, label='D=1.2')
    axes[0, 1].annotate('Expect: D ~ 1.1-1.2 (dead chains add to Poisson limit)',
                         xy=(0.5, 0.95), xycoords='axes fraction', fontsize=9,
                         ha='center', va='top', color='green')
    axes[0, 1].legend()

    # 数值检查
    idx_50 = np.argmin(np.abs(conv_h - 0.5))
    if conv_h[idx_50] > 0.3:
        print(f"High Ctr: D at ~50% conv = {disp_h[idx_50]:.3f} (expect < 1.1)")
else:
    axes[0, 0].text(0.5, 0.5, 'FAILED', ha='center', va='center', transform=axes[0, 0].transAxes)
    axes[0, 1].text(0.5, 0.5, 'FAILED', ha='center', va='center', transform=axes[0, 1].transAxes)

# Low Ctr
if result_low is not None:
    conv_l = result_low['conversion']
    mn_l = result_low['mn']
    disp_l = result_low['dispersity']

    axes[1, 0].plot(conv_l, mn_l, 'b-o', markersize=3)
    axes[1, 0].set_title('Low Ctr (0.01): Mn vs Conv')
    axes[1, 0].set_xlabel('Conversion')
    axes[1, 0].set_ylabel('Mn (g/mol)')

    axes[1, 1].plot(conv_l, disp_l, 'r-o', markersize=3)
    axes[1, 1].set_title('Low Ctr (0.01): D vs Conv')
    axes[1, 1].set_xlabel('Conversion')
    axes[1, 1].set_ylabel('Dispersity')
    axes[1, 1].axhline(y=1.5, color='gray', linestyle='--', alpha=0.5, label='D=1.5')
    axes[1, 1].annotate('Expect: D ~ 2.0 (approaching FRP, poor RAFT control)',
                         xy=(0.5, 0.95), xycoords='axes fraction', fontsize=9,
                         ha='center', va='top', color='red')
    axes[1, 1].legend()

    print(f"Low Ctr: mean D = {np.mean(disp_l):.3f} (expect > 1.5)")
else:
    axes[1, 0].text(0.5, 0.5, 'FAILED', ha='center', va='center', transform=axes[1, 0].transAxes)
    axes[1, 1].text(0.5, 0.5, 'FAILED', ha='center', va='center', transform=axes[1, 1].transAxes)

plt.tight_layout()
plt.savefig('notebooks/extreme_limits.png', dpi=150)
plt.close()
print("Extreme limits plot saved: notebooks/extreme_limits.png")

# %% [markdown]
# ## Cell 6: ctFP Visualization
#
# 对每个体系生成ctFP (64x64双通道)，可视化Channel 0 (Mn) 和 Channel 1 (D)。

# %%
# ctFP visualization for each system
systems = [
    ('CDB/Styrene (Dithioester)', params_cdb, result_cdb, 'dithioester'),
    ('TTC/MMA', params_ttc, result_ttc, 'ttc'),
    ('Xanthate/VAc', params_xan, result_xan, 'xanthate'),
]

fig, axes = plt.subplots(len(systems), 2, figsize=(10, 4 * len(systems)))
fig.suptitle('ctFP Visualization (Channel 0: Mn, Channel 1: D)', fontsize=14)

for i, (name, params, result, rtype) in enumerate(systems):
    if result is not None:
        cta_m_ratio = params['CTA0'] / params['M0']
        cta_norm = cta_m_ratio / 0.1

        data_for_ctfp = list(zip(
            [cta_norm] * len(result['conversion']),
            result['conversion'],
            result['mn_norm'],
            result['dispersity'],
        ))
        ctfp = transform(data_for_ctfp)
        ctfp_np = ctfp.numpy()

        im0 = axes[i, 0].imshow(ctfp_np[0], aspect='auto', origin='lower',
                                 cmap='viridis', interpolation='nearest')
        axes[i, 0].set_title(f'{name} - Ch0 (Mn)')
        axes[i, 0].set_xlabel('[CTA]/[M] (normalized)')
        axes[i, 0].set_ylabel('Conversion')
        plt.colorbar(im0, ax=axes[i, 0])

        im1 = axes[i, 1].imshow(ctfp_np[1], aspect='auto', origin='lower',
                                 cmap='magma', interpolation='nearest')
        axes[i, 1].set_title(f'{name} - Ch1 (D)')
        axes[i, 1].set_xlabel('[CTA]/[M] (normalized)')
        axes[i, 1].set_ylabel('Conversion')
        plt.colorbar(im1, ax=axes[i, 1])
    else:
        axes[i, 0].text(0.5, 0.5, f'{name}: FAILED', ha='center', va='center',
                         transform=axes[i, 0].transAxes, fontsize=14, color='red')
        axes[i, 1].text(0.5, 0.5, f'{name}: FAILED', ha='center', va='center',
                         transform=axes[i, 1].transAxes, fontsize=14, color='red')

plt.tight_layout()
plt.savefig('notebooks/ctfp_visualization.png', dpi=150)
plt.close()
print("ctFP visualization saved: notebooks/ctfp_visualization.png")

# %% [markdown]
# ## Summary
#
# 如果所有模拟成功且满足以下条件，ODE系统验证通过：
# 1. CDB/Styrene: 有可见诱导期，D随转化率下降（低转化率区间内）
# 2. TTC/MMA: 无诱导期，D < 1.4（中后期），Mn线性增长
# 3. Xanthate/VAc: Ctr=1时 D ~ 2.0 (弱控制)，Mn基本恒定
# 4. High Ctr: D ~ 1.1-1.2 (含dead chain贡献)
# 5. Low Ctr: D ~ 2.0 (接近FRP)

# %%
# Print summary
print("\n" + "=" * 60)
print("ODE VALIDATION SUMMARY")
print("=" * 60)

for name, params, result, rtype in systems:
    if result is not None:
        conv = result['conversion']
        disp = result['dispersity']
        mn = result['mn']
        print(f"\n{name}:")
        print(f"  Conv range: {conv[0]:.3f} - {conv[-1]:.3f}")
        print(f"  Mn range: {mn[0]:.0f} - {mn[-1]:.0f}")
        print(f"  D range: {disp.min():.3f} - {disp.max():.3f}")
        print(f"  Mean D: {np.mean(disp):.3f}")

        # 诱导期
        sol = _run_ode_for_rate(params, rtype, 36000)
        if sol is not None:
            inh = compute_inhibition_period(sol.sol, params['M0'], 36000)
            ret = compute_retardation_factor(params, rtype)
            print(f"  Inhibition period: {inh:.4f}")
            print(f"  Retardation factor: {ret:.4f}")
    else:
        print(f"\n{name}: SIMULATION FAILED")

print("\nExtreme limits:")
if result_high is not None:
    idx_50 = np.argmin(np.abs(result_high['conversion'] - 0.5))
    print(f"  High Ctr (10000): D at ~50%% = {result_high['dispersity'][idx_50]:.3f}")
if result_low is not None:
    print(f"  Low Ctr (0.01): mean D = {np.mean(result_low['dispersity']):.3f}")
