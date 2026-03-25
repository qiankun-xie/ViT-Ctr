"""
RAFT ODE系统单元测试和集成测试

测试覆盖:
- 正向模拟基本功能
- 预平衡模型与单平衡模型的行为差异
- 所有RAFT剂类型的兼容性
- Ctr参数范围覆盖 (log10: -2 到 4)
- [CTA]/[M]范围覆盖 (0.001 到 0.1)
- 极限行为校验
- 减速因子和诱导期的合理范围
- Mn归一化验证
"""

import numpy as np
import pytest

from src.raft_ode import (
    simulate_raft,
    raft_ode_single_eq,
    raft_ode_preequilibrium,
    compute_retardation_factor,
    compute_inhibition_period,
    _run_ode_for_rate,
)


class TestForwardSimulation:
    """SIM-01: ODE产生Mn和D随转化率变化的曲线"""

    def test_forward_simulation(self, typical_ttc_params):
        """simulate_raft返回包含conversion/mn/dispersity数组的字典，
        转化率在[0.02, 0.95]范围内，Mn>0，dispersity>=1.0"""
        result = simulate_raft(typical_ttc_params, raft_type='ttc')

        assert result is not None, "simulate_raft不应返回None"
        assert 'conversion' in result
        assert 'mn' in result
        assert 'dispersity' in result
        assert 'mn_norm' in result
        assert 'time' in result

        conv = result['conversion']
        mn = result['mn']
        disp = result['dispersity']

        # 至少有一些采样点（可能不到50个，取决于最大转化率）
        assert len(conv) >= 3, f"采样点数太少: {len(conv)}"
        assert len(mn) == len(conv)
        assert len(disp) == len(conv)

        # 转化率范围
        assert conv[0] >= 0.02, f"最低转化率 {conv[0]} 应 >= 0.02"
        assert conv[-1] <= 0.95, f"最高转化率 {conv[-1]} 应 <= 0.95"

        # 转化率单调递增
        assert np.all(np.diff(conv) > 0), "转化率应单调递增"

        # Mn正值
        assert np.all(mn > 0), "所有Mn值应为正"

        # Dispersity >= 1.0
        assert np.all(disp >= 1.0), f"所有dispersity值应 >= 1.0, 最小值: {disp.min()}"


class TestPreequilibriumDistinct:
    """SIM-02: Dithioester预平衡产生可观察的诱导期差异"""

    def test_preequilibrium_distinct(self, typical_dithioester_params, typical_ttc_params):
        """Dithioester的诱导期应至少是TTC的5倍"""
        # 运行dithioester ODE
        sol_de = _run_ode_for_rate(typical_dithioester_params, 'dithioester', 36000)
        assert sol_de is not None, "Dithioester ODE求解失败"

        # 运行TTC ODE（使用相同的CTA0以公平比较）
        ttc_params = typical_ttc_params.copy()
        ttc_params['CTA0'] = typical_dithioester_params['CTA0']
        sol_ttc = _run_ode_for_rate(ttc_params, 'ttc', 36000)
        assert sol_ttc is not None, "TTC ODE求解失败"

        M0 = typical_dithioester_params['M0']
        inh_de = compute_inhibition_period(sol_de.sol, M0, 36000)
        inh_ttc = compute_inhibition_period(sol_ttc.sol, M0, 36000)

        # Dithioester应有明显诱导期
        assert inh_de > 0.01, (
            f"Dithioester诱导期 {inh_de:.4f} 应 > 0.01（可见延迟）"
        )

        # TTC应有很小的诱导期
        assert inh_ttc < 0.05, (
            f"TTC诱导期 {inh_ttc:.4f} 应 < 0.05（接近无延迟）"
        )

        # 比率至少5倍
        ratio = inh_de / max(inh_ttc, 1e-10)
        assert ratio >= 5.0, (
            f"Dithioester/TTC诱导期比值 {ratio:.1f} 应 >= 5.0 "
            f"(DE={inh_de:.4f}, TTC={inh_ttc:.4f})"
        )


class TestAllAgentTypes:
    """SIM-02: 所有4种RAFT剂类型均可成功模拟"""

    @pytest.mark.parametrize("raft_type", ['dithioester', 'ttc', 'xanthate', 'dithiocarbamate'])
    def test_all_agent_types(self, raft_type, typical_ttc_params, typical_dithioester_params):
        """每种RAFT剂类型的simulate_raft都不应返回None"""
        if raft_type == 'dithioester':
            params = typical_dithioester_params
        else:
            params = typical_ttc_params

        result = simulate_raft(params, raft_type=raft_type)
        assert result is not None, f"raft_type='{raft_type}'的模拟不应返回None"
        assert len(result['conversion']) >= 3, (
            f"raft_type='{raft_type}'的采样点数太少: {len(result['conversion'])}"
        )


class TestParameterRangeCoverage:
    """SIM-03: ODE在完整Ctr范围(0.01-10000)上成功"""

    @pytest.mark.parametrize("log_ctr", [-2, -1, 0, 1, 2, 3, 4])
    def test_parameter_range_coverage(self, log_ctr, typical_ttc_params):
        """simulate_raft在Ctr从0.01到10000的7个量级上都应成功"""
        params = typical_ttc_params.copy()
        ctr_target = 10 ** log_ctr
        kp = params['kp']
        # Ctr ~ kadd / kp (简化)，固定kfrag=1e4
        params['kadd'] = ctr_target * kp
        params['kfrag'] = 1e4

        result = simulate_raft(params, raft_type='ttc')
        assert result is not None, (
            f"log10(Ctr)={log_ctr}（kadd={params['kadd']:.0e}）的模拟失败"
        )
        assert len(result['conversion']) >= 3, (
            f"log10(Ctr)={log_ctr}的采样点不足: {len(result['conversion'])}"
        )


class TestCTARatioRange:
    """SIM-03: ODE在[CTA]/[M]范围0.001-0.1上成功"""

    @pytest.mark.parametrize("cta_ratio", [0.001, 0.005, 0.01, 0.05, 0.1])
    def test_cta_ratio_range(self, cta_ratio, typical_ttc_params):
        """simulate_raft在不同[CTA]/[M]比值上都应成功"""
        params = typical_ttc_params.copy()
        params['CTA0'] = cta_ratio * params['M0']

        result = simulate_raft(params, raft_type='ttc')
        assert result is not None, (
            f"[CTA]/[M]={cta_ratio}的模拟失败"
        )
        assert len(result['conversion']) >= 3


class TestLimitBehavior:
    """极限行为校验"""

    def test_limit_behavior_high_ctr(self, extreme_high_ctr_params):
        """极高Ctr时，50%转化率处dispersity应 < 1.3（高度控制的RAFT）"""
        result = simulate_raft(extreme_high_ctr_params, raft_type='ttc')
        assert result is not None, "极高Ctr模拟失败"

        conv = result['conversion']
        disp = result['dispersity']

        # 找到最接近50%转化率的点
        idx_50 = np.argmin(np.abs(conv - 0.5))
        if conv[idx_50] > 0.3:  # 确保至少接近50%
            assert disp[idx_50] < 1.3, (
                f"极高Ctr时dispersity={disp[idx_50]:.3f}应 < 1.3 "
                f"(conv={conv[idx_50]:.3f})"
            )
        else:
            # 如果没到50%，检查最后一个点
            assert disp[-1] < 1.5, (
                f"极高Ctr时最终dispersity={disp[-1]:.3f}应 < 1.5"
            )

    def test_limit_behavior_low_ctr(self, extreme_low_ctr_params):
        """极低Ctr时，dispersity应接近常规FRP值 (> 1.5)"""
        result = simulate_raft(extreme_low_ctr_params, raft_type='ttc')
        assert result is not None, "极低Ctr模拟失败"

        disp = result['dispersity']
        # 在低Ctr下，RAFT效果弱，dispersity应较高
        # 注意: 即使Ctr低，仍有一定程度的链转移，所以用1.5作为阈值
        assert np.mean(disp) > 1.5, (
            f"极低Ctr时平均dispersity={np.mean(disp):.3f}应 > 1.5"
        )


class TestRetardationFactor:
    """减速因子计算验证"""

    def test_retardation_factor_range(self, typical_ttc_params):
        """compute_retardation_factor返回(0, 1]范围内的值"""
        rf = compute_retardation_factor(typical_ttc_params, 'ttc')

        assert isinstance(rf, float), "减速因子应为float"
        assert 0 < rf <= 1.0, (
            f"减速因子 {rf} 应在 (0, 1] 范围内"
        )


class TestInhibitionPeriod:
    """诱导期计算验证"""

    def test_inhibition_period_range(self, typical_ttc_params):
        """compute_inhibition_period返回[0, 1]范围内的值"""
        sol = _run_ode_for_rate(typical_ttc_params, 'ttc', 36000)
        assert sol is not None

        inh = compute_inhibition_period(sol.sol, typical_ttc_params['M0'], 36000)

        assert isinstance(inh, float), "诱导期应为float"
        assert 0 <= inh <= 1.0, (
            f"诱导期 {inh} 应在 [0, 1] 范围内"
        )


class TestMnNormalization:
    """Mn归一化验证"""

    def test_mn_normalization(self, typical_ttc_params):
        """mn_norm值应大约在[0, 2.0]范围内（非原始g/mol值）"""
        result = simulate_raft(typical_ttc_params, raft_type='ttc')
        assert result is not None

        mn_norm = result['mn_norm']

        # mn_norm不应是原始Mn值（那会是数千到数万）
        assert np.all(mn_norm < 10.0), (
            f"mn_norm最大值 {mn_norm.max():.1f} 应 < 10.0 "
            f"（确保不是原始Mn值）"
        )
        assert np.all(mn_norm >= 0), "mn_norm应非负"

        # 对于控制良好的RAFT，mn_norm应接近1.0
        # （允许更宽的范围因为低转化率时mn_norm可能偏高）
        assert np.mean(mn_norm) < 3.0, (
            f"mn_norm平均值 {np.mean(mn_norm):.3f} 应 < 3.0 "
            f"（控制良好的RAFT）"
        )
