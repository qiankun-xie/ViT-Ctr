"""
RAFT ODE测试共享fixtures — 参数集定义

每个fixture提供一组典型的RAFT聚合动力学参数。
参数值基于文献数据和物理合理范围。
"""

import pytest


@pytest.fixture
def typical_ttc_params():
    """TTC/MMA体系典型参数 (60C, AIBN引发)"""
    return {
        'kd': 1.5e-5,       # AIBN at 60C (1/s)
        'f': 0.5,           # 引发剂效率
        'ki': 1e4,          # 引发反应速率常数 (L/mol/s)
        'kp': 650,          # MMA增长速率常数 (L/mol/s)
        'kt': 1e8,          # 终止速率常数 (L/mol/s)
        'kadd': 1e6,        # RAFT加成速率常数 (L/mol/s)
        'kfrag': 1e4,       # RAFT断裂速率常数 (1/s)
        'M0': 1.0,          # 初始单体浓度 (mol/L)
        'I0': 0.01,         # 初始引发剂浓度 (mol/L)
        'CTA0': 0.01,       # 初始RAFT剂浓度 (mol/L)
        'M_monomer': 100.12,  # MMA摩尔质量 (g/mol)
    }


@pytest.fixture
def typical_dithioester_params():
    """Dithioester/MMA体系典型参数 (60C, slow fragmentation)

    kfrag0=0.01使预平衡断裂非常缓慢，CTA0=0.02确保大量初始RAFT剂
    需要被消耗，产生明显的诱导期(inhibition period)。
    """
    return {
        'kd': 1.5e-5,
        'f': 0.5,
        'ki': 1e4,
        'kp': 650,
        'kt': 1e8,
        'kadd': 1e6,        # 主平衡加成
        'kfrag': 1e4,       # 主平衡断裂
        'kadd0': 1e6,       # 预平衡加成（与初始CTA）
        'kfrag0': 0.01,     # 预平衡断裂（极慢! 导致明显inhibition）
        'M0': 1.0,
        'I0': 0.01,
        'CTA0': 0.02,       # 较高CTA浓度以增强inhibition效果
        'M_monomer': 100.12,
    }


@pytest.fixture
def typical_xanthate_params():
    """Xanthate/VAc体系典型参数 (60C)"""
    return {
        'kd': 1.5e-5,
        'f': 0.5,
        'ki': 1e4,
        'kp': 6700,         # VAc增长速率常数较高
        'kt': 1e8,
        'kadd': 1e4,        # Xanthate加成速率较低
        'kfrag': 1e3,       # Xanthate断裂速率较低
        'M0': 1.0,
        'I0': 0.01,
        'CTA0': 0.01,
        'M_monomer': 86.09,  # VAc摩尔质量
    }


@pytest.fixture
def extreme_high_ctr_params():
    """极高Ctr参数 (Ctr ~ 10000)"""
    return {
        'kd': 1.5e-5,
        'f': 0.5,
        'ki': 1e4,
        'kp': 650,
        'kt': 1e8,
        'kadd': 1e7,        # 高kadd
        'kfrag': 1e5,       # 高kfrag
        'M0': 1.0,
        'I0': 0.01,
        'CTA0': 0.01,
        'M_monomer': 100.12,
    }


@pytest.fixture
def extreme_low_ctr_params():
    """极低Ctr参数 (Ctr ~ 0.01)"""
    return {
        'kd': 1.5e-5,
        'f': 0.5,
        'ki': 1e4,
        'kp': 650,
        'kt': 1e8,
        'kadd': 1e3,        # 低kadd
        'kfrag': 1e2,       # 低kfrag
        'M0': 1.0,
        'I0': 0.01,
        'CTA0': 0.01,
        'M_monomer': 100.12,
    }
