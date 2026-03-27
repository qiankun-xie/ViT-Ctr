"""
大规模RAFT ctFP数据集生成器。

使用Latin Hypercube采样(LHS)在完整参数空间上采样，
通过joblib并行ODE模拟生成ctFP指纹样本，
存储为chunked HDF5格式（按RAFT剂类型分文件）。

参数空间（7维连续 + 1维离散RAFT类型）:
  - log10_Ctr: -2 到 4 (Ctr = 0.01 到 10000)
  - log10_kp: 2 到 4 (kp = 100 到 10000 L/mol/s)
  - log10_kt: 6 到 9 (kt = 1e6 到 1e9 L/mol/s)
  - log10_kd: -6 到 -4 (kd = 1e-6 到 1e-4 s^-1)
  - I0: 0.001 到 0.05 M (线性采样)
  - f: 0.5 到 0.8 (线性采样)
  - log10_cta_ratio: -3 到 -1 ([CTA]0/[M]0 = 0.001 到 0.1)
  - raft_type: 离散 (dithioester, trithiocarbonate, xanthate, dithiocarbamate)
"""

import os
import sys
import logging
import datetime

import numpy as np
from scipy.stats import qmc
import h5py
from joblib import Parallel, delayed
from tqdm import tqdm

# 防止MKL/torch冲突（Windows环境）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from raft_ode import (
    simulate_raft,
    compute_retardation_factor,
    compute_inhibition_period,
    _run_ode_for_rate,
)
from ctfp_encoder import transform


# ============================================================
# 参数范围定义
# ============================================================

# 7维LHS参数的范围 [lower, upper]
# 前4个在对数空间采样，后3个混合
PARAM_BOUNDS = {
    'log10_Ctr':      (-2.0,  4.0),    # Ctr = 0.01 到 10000
    'log10_kp':       ( 2.0,  4.0),    # kp = 100 到 10000
    'log10_kt':       ( 6.0,  9.0),    # kt = 1e6 到 1e9
    'log10_kd':       (-6.0, -4.0),    # kd = 1e-6 到 1e-4
    'I0':             ( 0.001, 0.05),   # 线性
    'f':              ( 0.5,   0.8),    # 线性
    'log10_cta_ratio':(-3.0, -1.0),    # [CTA]0/[M]0 = 0.001 到 0.1
}

# 4种RAFT剂类型
RAFT_TYPES = ['dithioester', 'trithiocarbonate', 'xanthate', 'dithiocarbamate']

# 各RAFT类型的固定参数（单体、摩尔质量等）
# 与diagnostic.py保持一致
TYPE_FIXED_PARAMS = {
    'trithiocarbonate': {
        'ki': 1e4,
        'M_monomer': 100.12,  # MMA
        'M0': 1.0,
    },
    'dithioester': {
        'ki': 1e4,
        'M_monomer': 104.15,  # Styrene
        'M0': 1.0,
        # 预平衡参数由Ctr和kp推导
    },
    'xanthate': {
        'ki': 1e4,
        'M_monomer': 86.09,   # VAc
        'M0': 1.0,
    },
    'dithiocarbamate': {
        'ki': 1e4,
        'M_monomer': 86.09,   # VAc
        'M0': 1.0,
    },
}


# ============================================================
# Task 1: LHS参数采样、噪声注入、单样本模拟
# ============================================================

def generate_lhs_parameters(n_samples, raft_type, seed=None):
    """
    使用Latin Hypercube采样生成7维连续参数组合。

    Parameters
    ----------
    n_samples : int
        采样数量
    raft_type : str
        RAFT剂类型
    seed : int or None
        随机种子（可复现）

    Returns
    -------
    list of dict
        每个dict包含完整的ODE模拟参数
    """
    sampler = qmc.LatinHypercube(d=7, seed=seed)
    sample = sampler.random(n=n_samples)  # shape: (n_samples, 7)

    # 参数名称（与PARAM_BOUNDS的key顺序一致）
    param_names = list(PARAM_BOUNDS.keys())
    l_bounds = np.array([PARAM_BOUNDS[k][0] for k in param_names])
    u_bounds = np.array([PARAM_BOUNDS[k][1] for k in param_names])

    # 将[0,1]映射到实际范围
    scaled = qmc.scale(sample, l_bounds, u_bounds)

    # 获取固定参数
    fixed = TYPE_FIXED_PARAMS[raft_type]

    params_list = []
    for i in range(n_samples):
        row = scaled[i]
        log10_Ctr = row[0]
        log10_kp = row[1]
        log10_kt = row[2]
        log10_kd = row[3]
        I0 = row[4]
        f_eff = row[5]
        log10_cta_ratio = row[6]

        kp = 10**log10_kp
        kt = 10**log10_kt
        kd = 10**log10_kd
        Ctr = 10**log10_Ctr
        cta_ratio = 10**log10_cta_ratio

        # 从Ctr推导kadd: Ctr ~ kadd/kp（简化模型）
        kadd = Ctr * kp
        kfrag = 1e4  # 固定断裂速率（与diagnostic一致）

        p = {
            'kp': kp,
            'kt': kt,
            'kd': kd,
            'f': f_eff,
            'I0': I0,
            'ki': fixed['ki'],
            'kadd': kadd,
            'kfrag': kfrag,
            'M0': fixed['M0'],
            'CTA0': cta_ratio * fixed['M0'],
            'M_monomer': fixed['M_monomer'],
            'raft_type': raft_type,
            'log10_Ctr': log10_Ctr,
            'cta_ratio': cta_ratio,
        }

        # dithioester特有的预平衡参数
        if raft_type == 'dithioester':
            p['kadd0'] = kadd * 0.1     # 预平衡加成速率（略低于主平衡）
            p['kfrag0'] = 1.0           # 慢断裂导致inhibition period

        params_list.append(p)

    return params_list


def inject_noise(Mn_array, D_array, sigma=0.03):
    """
    对Mn和分散度数组注入乘性高斯噪声，模拟GPC测量误差。

    Parameters
    ----------
    Mn_array : np.ndarray
        数均分子量数组
    D_array : np.ndarray
        分散度(Mw/Mn)数组
    sigma : float
        相对噪声标准差（默认3%）

    Returns
    -------
    tuple of np.ndarray
        (Mn_noisy, D_noisy)
    """
    Mn_noise = 1.0 + np.random.normal(0, sigma, size=Mn_array.shape)
    D_noise = 1.0 + np.random.normal(0, sigma, size=D_array.shape)

    Mn_noisy = Mn_array * Mn_noise
    D_noisy = D_array * D_noise

    # 确保Mn > 0 和 D >= 1.0
    Mn_noisy = np.maximum(Mn_noisy, 0.0)
    D_noisy = np.maximum(D_noisy, 1.0)

    return Mn_noisy, D_noisy


def simulate_single_sample(params):
    """
    模拟单个RAFT样本：ODE求解 -> 噪声注入 -> ctFP编码。

    Parameters
    ----------
    params : dict
        完整的ODE模拟参数（来自generate_lhs_parameters）

    Returns
    -------
    dict
        成功: {'fingerprint': np.ndarray(64,64,2), 'labels': [log10_Ctr, inhibition, retardation],
               'params': dict, 'success': True}
        失败: {'success': False, 'params': dict, 'error': str}
    """
    raft_type = params['raft_type']
    log10_Ctr = params['log10_Ctr']
    cta_ratio = params['cta_ratio']
    t_end = 36000

    try:
        # Step 1: ODE模拟
        result = simulate_raft(params, raft_type=raft_type, t_end=t_end)
        if result is None:
            return {'success': False, 'params': params, 'error': 'ODE solve failed'}

        # Step 2: 计算标签
        # 诱导期
        sol = _run_ode_for_rate(params, raft_type, t_end)
        if sol is None:
            inhibition = 0.0
        else:
            inhibition = compute_inhibition_period(sol.sol, params['M0'], t_end)

        # 减速因子
        retardation = compute_retardation_factor(params, raft_type=raft_type)

        # Step 3: 噪声注入
        mn_noisy, disp_noisy = inject_noise(result['mn'], result['dispersity'], sigma=0.03)

        # Step 4: ctFP编码
        mn_theory = params['M0'] / params['CTA0'] * params['M_monomer'] if params['CTA0'] > 0 else 1.0
        mn_norm = mn_noisy / mn_theory if mn_theory > 0 else mn_noisy

        cta_norm = cta_ratio / 0.1  # 归一化到[0,1]（[CTA]/[M]范围是0.001-0.1）
        data_for_ctfp = list(zip(
            [cta_norm] * len(result['conversion']),
            result['conversion'],
            mn_norm,
            disp_noisy,
        ))
        ctfp = transform(data_for_ctfp)
        # 转换为(64, 64, 2) HDF5格式: (H, W, C)
        fp_numpy = ctfp.numpy().transpose(1, 2, 0)  # (2, 64, 64) -> (64, 64, 2)

        labels = [log10_Ctr, inhibition, retardation]

        return {
            'fingerprint': fp_numpy,
            'labels': labels,
            'params': params,
            'success': True,
        }

    except Exception as e:
        return {'success': False, 'params': params, 'error': str(e)}


# ============================================================
# Task 2: 并行生成和HDF5存储
# ============================================================

def save_to_hdf5(samples, raft_type, output_path='data/'):
    """
    将成功的样本保存到HDF5文件。

    Parameters
    ----------
    samples : list of dict
        成功样本列表（每个包含fingerprint和labels）
    raft_type : str
        RAFT剂类型
    output_path : str
        输出目录

    Returns
    -------
    str
        HDF5文件路径
    """
    os.makedirs(output_path, exist_ok=True)
    h5_path = os.path.join(output_path, f'{raft_type}.h5')
    n = len(samples)

    if n == 0:
        logging.warning(f"没有成功样本可保存到 {h5_path}")
        return h5_path

    with h5py.File(h5_path, 'w') as f:
        # fingerprints: (n, 64, 64, 2), float32, chunked
        chunk_size = min(1000, n)
        fp_dset = f.create_dataset(
            'fingerprints',
            shape=(n, 64, 64, 2),
            dtype='float32',
            chunks=(chunk_size, 64, 64, 2),
        )

        # labels: (n, 3), float32 [log10_Ctr, inhibition, retardation]
        lbl_dset = f.create_dataset(
            'labels',
            shape=(n, 3),
            dtype='float32',
        )

        # params: 结构化数组存储关键参数（用于调试）
        param_dtype = np.dtype([
            ('log10_Ctr', 'f4'),
            ('kp', 'f4'),
            ('kt', 'f4'),
            ('kd', 'f4'),
            ('I0', 'f4'),
            ('f', 'f4'),
            ('cta_ratio', 'f4'),
        ])
        param_dset = f.create_dataset(
            'params',
            shape=(n,),
            dtype=param_dtype,
        )

        # 元数据
        f.attrs['raft_type'] = raft_type
        f.attrs['n_samples'] = n
        f.attrs['created'] = str(datetime.datetime.now())
        f.attrs['label_names'] = ['log10_Ctr', 'inhibition_period', 'retardation_factor']

        # 批量写入（每批1000样本）
        batch_size = 1000
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = samples[start:end]

            fp_batch = np.array([s['fingerprint'] for s in batch], dtype=np.float32)
            lbl_batch = np.array([s['labels'] for s in batch], dtype=np.float32)

            fp_dset[start:end] = fp_batch
            lbl_dset[start:end] = lbl_batch

            for j, s in enumerate(batch):
                p = s['params']
                param_dset[start + j] = (
                    p['log10_Ctr'],
                    p['kp'],
                    p['kt'],
                    p['kd'],
                    p['I0'],
                    p['f'],
                    p['cta_ratio'],
                )

    logging.info(f"HDF5保存完成: {h5_path} ({n}样本)")
    return h5_path


def generate_dataset_parallel(raft_type, n_samples=250000, output_path='data/', seed=None):
    """
    并行生成指定RAFT类型的数据集并保存到HDF5。

    Parameters
    ----------
    raft_type : str
        RAFT剂类型
    n_samples : int
        采样数量（默认250000）
    output_path : str
        输出目录
    seed : int or None
        随机种子

    Returns
    -------
    dict
        生成结果统计 {'n_total', 'n_success', 'n_failed', 'failure_rate', 'h5_path'}
    """
    print(f"\n{'='*60}")
    print(f"生成 {raft_type} 数据集: {n_samples} 样本")
    print(f"{'='*60}")

    # Step 1: LHS参数采样
    print("Step 1: LHS参数采样...")
    params_list = generate_lhs_parameters(n_samples, raft_type, seed=seed)
    print(f"  生成 {len(params_list)} 组参数")

    # Step 2: 并行ODE模拟
    print("Step 2: 并行ODE模拟 + ctFP编码...")
    results = Parallel(n_jobs=-1, prefer='processes')(
        delayed(simulate_single_sample)(p)
        for p in tqdm(params_list, desc=f"{raft_type}")
    )

    # Step 3: 收集结果
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    failure_rate = len(failed) / n_samples if n_samples > 0 else 0.0

    print(f"\n  成功: {len(successful)}/{n_samples}")
    print(f"  失败: {len(failed)}/{n_samples} ({failure_rate:.2%})")

    # 失败率检查
    if failure_rate > 0.05:
        warning_msg = (f"WARNING: {raft_type} 失败率 {failure_rate:.2%} 超过5%阈值! "
                      f"请检查ODE实现或参数范围。")
        print(f"  *** {warning_msg}")
        logging.warning(warning_msg)

        # 记录失败参数到日志
        log_path = os.path.join(output_path, f'{raft_type}_failures.log')
        os.makedirs(output_path, exist_ok=True)
        with open(log_path, 'w') as flog:
            flog.write(f"# {raft_type} 失败样本日志\n")
            flog.write(f"# 总样本: {n_samples}, 失败: {len(failed)}, 失败率: {failure_rate:.4f}\n")
            flog.write(f"# 时间: {datetime.datetime.now()}\n\n")
            for r in failed:
                p = r['params']
                err = r.get('error', 'unknown')
                flog.write(f"log10_Ctr={p.get('log10_Ctr', '?'):.3f}, "
                          f"kp={p.get('kp', '?'):.1f}, "
                          f"kt={p.get('kt', '?'):.2e}, "
                          f"kd={p.get('kd', '?'):.2e}, "
                          f"I0={p.get('I0', '?'):.4f}, "
                          f"f={p.get('f', '?'):.3f}, "
                          f"cta_ratio={p.get('cta_ratio', '?'):.4f}, "
                          f"error={err}\n")
        print(f"  失败参数日志: {log_path}")

    # Step 4: 保存HDF5
    print("Step 3: 保存HDF5...")
    h5_path = save_to_hdf5(successful, raft_type, output_path)
    print(f"  HDF5文件: {h5_path}")

    return {
        'n_total': n_samples,
        'n_success': len(successful),
        'n_failed': len(failed),
        'failure_rate': failure_rate,
        'h5_path': h5_path,
    }


def main():
    """
    主函数：遍历4种RAFT类型，生成完整数据集。
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    output_path = 'data/'
    n_per_type = 250000

    print("=" * 60)
    print("ViT-Ctr 大规模数据集生成器")
    print(f"目标: {len(RAFT_TYPES)} 种RAFT类型 x {n_per_type} 样本/类型 = {len(RAFT_TYPES)*n_per_type} 总样本")
    print(f"输出: {output_path}")
    print("=" * 60)

    all_stats = []
    for i, raft_type in enumerate(RAFT_TYPES):
        seed = 42 + i  # 每种类型不同的种子
        stats = generate_dataset_parallel(raft_type, n_per_type, output_path, seed=seed)
        all_stats.append(stats)

    # 总结
    print("\n" + "=" * 60)
    print("数据集生成总结")
    print("=" * 60)
    total_success = sum(s['n_success'] for s in all_stats)
    total_failed = sum(s['n_failed'] for s in all_stats)
    total_samples = sum(s['n_total'] for s in all_stats)
    overall_rate = total_failed / total_samples if total_samples > 0 else 0.0

    for s, rt in zip(all_stats, RAFT_TYPES):
        print(f"  {rt:25s}: {s['n_success']:>7d} 成功, "
              f"{s['n_failed']:>5d} 失败 ({s['failure_rate']:.2%}), "
              f"文件: {s['h5_path']}")

    print(f"\n  总计: {total_success}/{total_samples} 成功, "
          f"{total_failed} 失败 ({overall_rate:.2%})")
    print("=" * 60)


if __name__ == '__main__':
    main()
