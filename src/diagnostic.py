"""
RAFT ODE诊断数据集生成器。

在提交百万级数据生成(Phase 2)之前，先生成1000样本诊断数据集，
验证ODE数值稳定性覆盖全参数空间。

每种RAFT剂类型250样本 = 1000总样本，
参数网格: 10 Ctr值 x 25 [CTA]/[M]值。
"""

import os
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

# 防止MKL/torch冲突（Windows环境）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.raft_ode import (
    simulate_raft,
    compute_retardation_factor,
    compute_inhibition_period,
    _run_ode_for_rate,
)
from src.ctfp_encoder import transform


# 每种RAFT剂类型的固定动力学参数（来自文献典型值）
TYPE_PARAMS = {
    'ttc': {
        'kp': 650,          # MMA增长速率常数 (L/mol/s)
        'kt': 1e8,          # 终止速率常数 (L/mol/s)
        'kd': 1.5e-5,       # AIBN at 60C (1/s)
        'f': 0.5,           # 引发剂效率
        'ki': 1e4,          # 引发反应速率常数 (L/mol/s)
        'M_monomer': 100.12,  # MMA摩尔质量 (g/mol)
        'M0': 1.0,
        'I0': 0.01,
    },
    'dithioester': {
        'kp': 340,          # Styrene增长速率常数 (L/mol/s)
        'kt': 1e8,
        'kd': 1.5e-5,
        'f': 0.5,
        'ki': 1e4,
        'M_monomer': 104.15,  # Styrene摩尔质量 (g/mol)
        'M0': 1.0,
        'I0': 0.01,
        # 预平衡参数
        'kadd0': 1e6,       # 预平衡加成速率常数
        'kfrag0': 1.0,      # 预平衡断裂速率常数（慢断裂 -> inhibition）
    },
    'xanthate': {
        'kp': 6700,         # VAc增长速率常数 (L/mol/s)
        'kt': 1e8,
        'kd': 1.5e-5,
        'f': 0.5,
        'ki': 1e4,
        'M_monomer': 86.09,  # VAc摩尔质量 (g/mol)
        'M0': 1.0,
        'I0': 0.01,
    },
    'dithiocarbamate': {
        'kp': 6700,         # VAc增长速率常数（与xanthate相同单体）
        'kt': 1e8,
        'kd': 1.5e-5,
        'f': 0.5,
        'ki': 1e4,
        'M_monomer': 86.09,  # VAc摩尔质量 (g/mol)
        'M0': 1.0,
        'I0': 0.01,
    },
}


def _simulate_one_sample(raft_type, ctr, cta_m_ratio, base_params):
    """
    模拟单个RAFT样本，返回结果或None（失败时）。

    Parameters
    ----------
    raft_type : str
        RAFT剂类型
    ctr : float
        目标链转移常数 Ctr
    cta_m_ratio : float
        [CTA]/[M] 比值
    base_params : dict
        该类型的基础动力学参数

    Returns
    -------
    dict or None
        成功时返回包含params, result, labels, ctfp的字典，失败返回None
    """
    params = base_params.copy()
    kp = params['kp']

    # 从Ctr推导kadd: Ctr ~ kadd/kp (简化), 固定kfrag=1e4
    params['kadd'] = ctr * kp
    params['kfrag'] = 1e4
    params['CTA0'] = cta_m_ratio * params['M0']

    # 运行ODE模拟
    result = simulate_raft(params, raft_type=raft_type)
    if result is None:
        return None

    # 计算减速因子
    retardation = compute_retardation_factor(params, raft_type=raft_type)

    # 计算诱导期
    t_end = 36000
    sol = _run_ode_for_rate(params, raft_type, t_end)
    if sol is None:
        inhibition = 0.0
    else:
        inhibition = compute_inhibition_period(sol.sol, params['M0'], t_end)

    # 构建ctFP数据
    # x轴: [CTA]/[M] 归一化到 [0, 1]（除以0.1，因为[CTA]/[M]范围是0.001-0.1）
    cta_norm = cta_m_ratio / 0.1
    data_for_ctfp = list(zip(
        [cta_norm] * len(result['conversion']),
        result['conversion'],
        result['mn_norm'],
        result['dispersity'],
    ))
    ctfp = transform(data_for_ctfp)

    # 标签
    labels = {
        'log10_ctr': np.log10(ctr),
        'inhibition_period': inhibition,
        'retardation_factor': retardation,
    }

    return {
        'params': {
            'raft_type': raft_type,
            'ctr': ctr,
            'cta_m_ratio': cta_m_ratio,
            **{k: v for k, v in params.items()
               if k not in ('raft_type', 'ctr', 'cta_m_ratio')},
        },
        'result': result,
        'labels': labels,
        'ctfp': ctfp,
    }


def generate_diagnostic_dataset(n_per_type=250, seed=42):
    """
    生成诊断数据集。

    Parameters
    ----------
    n_per_type : int
        每种RAFT剂类型的样本数（默认250，总计1000）
    seed : int
        随机种子（用于可重复性，虽然这里是确定性网格）

    Returns
    -------
    dict
        包含:
        - params: list of dict, 每个样本的参数
        - results: list of dict, 每个样本的ODE输出(conversion/mn/dispersity/mn_norm)
        - labels: list of dict, 每个样本的标签(log10_ctr, inhibition_period, retardation_factor)
        - ctfp_tensors: list of torch.Tensor, 每个样本的ctFP (2, 64, 64)
        - failures: list of int, 失败样本的索引
    """
    np.random.seed(seed)

    # 参数网格: 10 Ctr x 25 [CTA]/[M] = 250 per type
    # 如果n_per_type不是250，调整网格
    n_ctr = 10
    n_cta_m = n_per_type // n_ctr
    if n_cta_m < 1:
        n_cta_m = 1
        n_ctr = n_per_type

    ctr_grid = np.logspace(-2, 4, n_ctr)       # Ctr: 0.01 to 10000
    cta_m_grid = np.logspace(-3, -1, n_cta_m)  # [CTA]/[M]: 0.001 to 0.1

    raft_types = ['dithioester', 'ttc', 'xanthate', 'dithiocarbamate']

    # 构建所有任务
    tasks = []
    for raft_type in raft_types:
        base_params = TYPE_PARAMS[raft_type].copy()
        for ctr in ctr_grid:
            for cta_m in cta_m_grid:
                tasks.append((raft_type, ctr, cta_m, base_params))

    # 使用joblib并行执行
    print(f"生成诊断数据集: {len(tasks)}样本, 跨{len(raft_types)}种RAFT类型...")
    results_raw = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_simulate_one_sample)(rt, ctr, cm, bp)
        for rt, ctr, cm, bp in tqdm(tasks, desc="诊断样本生成")
    )

    # 整理结果
    params_list = []
    results_list = []
    labels_list = []
    ctfp_list = []
    failures = []

    for i, res in enumerate(results_raw):
        if res is None:
            failures.append(i)
        else:
            params_list.append(res['params'])
            results_list.append(res['result'])
            labels_list.append(res['labels'])
            ctfp_list.append(res['ctfp'])

    n_total = len(tasks)
    n_success = len(results_list)
    n_fail = len(failures)
    failure_rate = n_fail / n_total if n_total > 0 else 0.0

    print(f"\n--- 诊断数据集生成完成 ---")
    print(f"总样本数: {n_total}")
    print(f"成功: {n_success}")
    print(f"失败: {n_fail}")
    print(f"失败率: {failure_rate:.2%}")

    return {
        'params': params_list,
        'results': results_list,
        'labels': labels_list,
        'ctfp_tensors': ctfp_list,
        'failures': failures,
    }


def save_diagnostic(dataset, path='data/diagnostic_1000.npz'):
    """
    保存诊断数据集到npz文件。

    Parameters
    ----------
    dataset : dict
        generate_diagnostic_dataset()的返回值
    path : str
        保存路径
    """
    import torch

    # 创建目录
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

    # 堆叠ctFP张量为numpy数组
    if len(dataset['ctfp_tensors']) > 0:
        ctfp_stack = torch.stack(dataset['ctfp_tensors']).numpy()
    else:
        ctfp_stack = np.zeros((0, 2, 64, 64), dtype=np.float32)

    # 构建标签数组
    n = len(dataset['labels'])
    label_array = np.zeros((n, 3), dtype=np.float32)
    for i, lbl in enumerate(dataset['labels']):
        label_array[i, 0] = lbl['log10_ctr']
        label_array[i, 1] = lbl['inhibition_period']
        label_array[i, 2] = lbl['retardation_factor']

    np.savez_compressed(
        path,
        ctfp=ctfp_stack,
        labels=label_array,
        label_names=np.array(['log10_ctr', 'inhibition_period', 'retardation_factor']),
    )
    print(f"诊断数据集已保存到: {path}")
    print(f"  ctFP形状: {ctfp_stack.shape}")
    print(f"  标签形状: {label_array.shape}")
