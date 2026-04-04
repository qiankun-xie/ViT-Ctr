"""
文献验证与Mayo方程基线模块。
对比ML模型预测与传统Mayo ODE拟合在已发表Ctr值上的表现。
"""
import os
import json
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from raft_ode import simulate_raft
from ctfp_encoder import transform
from model import SimpViT
from bootstrap import predict_with_uncertainty

# 固定参数默认值（Mayo拟合时使用，与数据集生成参数中心值一致）
DEFAULT_FIXED_PARAMS = {
    'kp': 1000.0,
    'kt': 10**7.5,
    'kd': 1e-5,
    'f': 0.65,
    'ki': 1e4,
    'I0': 0.01,
    'M0': 1.0,
    'kfrag': 1e4,
}

MONOMER_MW = {
    'Styrene': 104.15,
    'MMA': 100.12,
    'VAc': 86.09,
    'NVP': 111.14,
    'MA': 86.09,
    'BA': 128.17,
}

RAFT_COLORS = {
    'dithioester': '#E64B35',
    'trithiocarbonate': '#4DBBD5',
    'xanthate': '#00A087',
    'dithiocarbamate': '#8491B4',
}


# 训练参数分布范围（与 dataset_generator.py PARAM_BOUNDS 一致）
_KINETIC_BOUNDS = {
    'log10_kp':  (2.0,  4.0),
    'log10_kt':  (6.0,  9.0),
    'log10_kd':  (-6.0, -4.0),
    'I0':        (0.001, 0.05),
    'f':         (0.5,   0.8),
}


def _map_raft_type(raft_type):
    """将CSV中的raft_type映射到ODE模拟器使用的类型名称。"""
    return 'ttc' if raft_type == 'trithiocarbonate' else raft_type


def sample_kinetic_params(rng):
    """从训练分布中均匀采样一组动力学参数（kp, kt, kd, I0, f）。"""
    return {
        'kp': 10 ** rng.uniform(*_KINETIC_BOUNDS['log10_kp']),
        'kt': 10 ** rng.uniform(*_KINETIC_BOUNDS['log10_kt']),
        'kd': 10 ** rng.uniform(*_KINETIC_BOUNDS['log10_kd']),
        'I0': rng.uniform(*_KINETIC_BOUNDS['I0']),
        'f':  rng.uniform(*_KINETIC_BOUNDS['f']),
    }


def build_ode_params(row, ctr_value, kinetic_override=None):
    """从文献CSV行和Ctr值构建ODE参数字典。
    kinetic_override: 可选dict，覆盖 kp/kt/kd/I0/f（用于集成预测）。
    """
    params = dict(DEFAULT_FIXED_PARAMS)
    if kinetic_override is not None:
        params.update(kinetic_override)
    params['kadd'] = ctr_value * params['kp']
    cta_ratio = 0.01
    params['CTA0'] = cta_ratio * params['M0']
    params['M_monomer'] = MONOMER_MW.get(row['monomer'], 100.0)
    if row['raft_type'] == 'dithioester':
        params['kadd0'] = params['kadd'] * 0.1
        params['kfrag0'] = 1.0
    return params


def generate_simulated_data(row, sigma=0.03, seed=None):
    """
    为一个文献数据点生成模拟实验数据（加噪声）。
    返回包含 'conversion', 'mn', 'dispersity', 'mn_norm', 'cta_ratio' 的字典，或None。
    """
    rng = np.random.default_rng(seed)
    params = build_ode_params(row, row['ctr'])
    raft_type = _map_raft_type(row['raft_type'])
    result = simulate_raft(params, raft_type=raft_type)
    if result is None:
        return None
    mn = np.array(result['mn'])
    mn_noisy = mn * (1 + rng.normal(0, sigma, size=mn.shape))
    dispersity = np.clip(np.array(result['dispersity']), 1.0, None)
    # mn_norm = mn / Mn_theory，与训练数据 dataset_generator.py 第244行一致
    mn_theory = params['M0'] / params['CTA0'] * params['M_monomer']
    mn_norm = mn_noisy / mn_theory
    return {
        'conversion': np.array(result['conversion']),
        'mn': mn_noisy,
        'dispersity': dispersity,
        'mn_norm': mn_norm,
        'cta_ratio': params['CTA0'] / params['M0'],
    }


def mayo_fit_ctr(target_conv, target_mn, row, bounds=(0.01, 20000)):
    """
    Mayo ODE单参数拟合：最小化模拟Mn与目标Mn的MSE，返回拟合的Ctr值。
    """
    def _loss(ctr_candidate):
        params = build_ode_params(row, ctr_candidate)
        raft_type = _map_raft_type(row['raft_type'])
        result = simulate_raft(params, raft_type=raft_type)
        if result is None:
            return 1e10
        mn_interp = np.interp(target_conv, result['conversion'], result['mn'])
        return np.mean((mn_interp - target_mn) ** 2)

    opt = minimize_scalar(_loss, bounds=bounds, method='bounded',
                          options={'xatol': 1e-3})
    return opt.x


def fold_error_log(ctr_pred, ctr_true):
    """对数空间fold-error: 10^|log10(pred) - log10(true)|"""
    return 10 ** np.abs(np.log10(ctr_pred) - np.log10(ctr_true))


def fold_error_ratio(ctr_pred, ctr_true):
    """比值空间fold-error: max(pred/true, true/pred)"""
    return np.maximum(ctr_pred / ctr_true, ctr_true / ctr_pred)


def compute_summary_stats(ctr_true, ctr_pred):
    """
    计算汇总统计指标。
    返回包含 median_fold_error, pct_within_2x, pct_within_10x, rmse_log10, r2_log10 的字典。
    """
    ctr_true = np.array(ctr_true, dtype=float)
    ctr_pred = np.array(ctr_pred, dtype=float)
    fe = fold_error_log(ctr_pred, ctr_true)
    log_true = np.log10(ctr_true)
    log_pred = np.log10(ctr_pred)
    ss_res = np.sum((log_true - log_pred) ** 2)
    ss_tot = np.sum((log_true - np.mean(log_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
    rmse = float(np.sqrt(np.mean((log_true - log_pred) ** 2)))
    return {
        'median_fold_error': float(np.median(fe)),
        'pct_within_2x': float(np.mean(fe <= 2.0) * 100),
        'pct_within_10x': float(np.mean(fe <= 10.0) * 100),
        'rmse_log10': rmse,
        'r2_log10': r2,
    }


def ml_predict_single(model, sim_data, device='cpu', bootstrap_ckpt=None, cal_factors=None):
    """
    对单个文献数据点进行ML推理。
    返回 (pred: np.ndarray(3,), half_width: np.ndarray(3,) or None)
    """
    cta_ratio_norm = sim_data['cta_ratio'] / 0.1
    data_tuples = list(zip(
        [cta_ratio_norm] * len(sim_data['conversion']),
        sim_data['conversion'],
        sim_data['mn_norm'],
        sim_data['dispersity'],
    ))
    fp = transform(data_tuples)
    fp_tensor = fp.unsqueeze(0)

    if bootstrap_ckpt is not None and cal_factors is not None:
        mean_pred, half_width = predict_with_uncertainty(
            model, fp_tensor, bootstrap_ckpt, cal_factors, device
        )
        return mean_pred, half_width
    else:
        model.eval()
        with torch.no_grad():
            pred = model(fp_tensor.to(device)).cpu().numpy().squeeze()
        return pred, None


def ml_predict_ensemble(model, row, n_samples=50, sigma=0.03, seed=42, device='cpu'):
    """
    集成预测：对每个文献点随机采样 n_samples 组动力学参数，
    每组生成 ctFP 并获得 ML 预测，取中位数作为最终结果。

    这样 ML 看到的 ctFP 来自训练分布，而非固定参数生成的异类输入。
    返回 (median_pred: np.ndarray(3,), std_pred: np.ndarray(3,))
    """
    rng = np.random.default_rng(seed)
    preds = []
    attempts = 0
    while len(preds) < n_samples and attempts < n_samples * 3:
        attempts += 1
        kinetic = sample_kinetic_params(rng)
        params = build_ode_params(row, row['ctr'], kinetic_override=kinetic)
        raft_type = _map_raft_type(row['raft_type'])
        result = simulate_raft(params, raft_type=raft_type)
        if result is None:
            continue
        mn = np.array(result['mn'])
        mn_noisy = mn * (1 + rng.normal(0, sigma, size=mn.shape))
        dispersity = np.clip(np.array(result['dispersity']), 1.0, None)
        mn_theory = params['M0'] / params['CTA0'] * params['M_monomer']
        mn_norm = mn_noisy / mn_theory
        sim_data = {
            'conversion': np.array(result['conversion']),
            'mn_norm': mn_norm,
            'dispersity': dispersity,
            'cta_ratio': params['CTA0'] / params['M0'],
        }
        pred, _ = ml_predict_single(model, sim_data, device=device)
        preds.append(pred)

    if len(preds) == 0:
        return np.array([float('nan')] * 3), np.array([float('nan')] * 3)

    preds = np.array(preds)
    return np.median(preds, axis=0), np.std(preds, axis=0)


def run_validation_pipeline(csv_path, model_path, bootstrap_path=None, calibration_path=None,
                             output_dir='figures/validation', sigma=0.03, device='cpu', seed=42,
                             n_ensemble=50):
    """
    端到端验证流程：对所有文献数据点运行ML集成预测和Mayo预测，生成结果文件和图表。
    n_ensemble: ML集成预测的采样次数（方向A，默认50）。
    返回 (results_df: pd.DataFrame, summary_dict: dict)
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    model = SimpViT()
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    bootstrap_ckpt = None
    cal_factors = None
    if bootstrap_path is not None and os.path.exists(bootstrap_path):
        bootstrap_ckpt = torch.load(bootstrap_path, map_location=device, weights_only=False)
    if calibration_path is not None and os.path.exists(calibration_path):
        with open(calibration_path) as f_cal:
            cal_factors = json.load(f_cal)

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Validating'):
        # ML集成预测（方向A：随机采样动力学参数）
        ml_median, ml_std = ml_predict_ensemble(
            model, row, n_samples=n_ensemble, sigma=sigma,
            seed=seed + int(row['id']), device=device
        )

        # Mayo基线（固定参数，单次拟合）
        sim_data = generate_simulated_data(row, sigma=sigma, seed=seed + int(row['id']))
        if sim_data is None:
            print(f"Warning: ODE simulation failed for id={row['id']} ({row['raft_agent']})")
            continue
        mayo_ctr = mayo_fit_ctr(sim_data['conversion'], sim_data['mn'], row)

        ml_log10_ctr = float(ml_median[0])
        mayo_log10_ctr = float(np.log10(mayo_ctr))
        ml_fold_err = float(fold_error_log(10 ** ml_log10_ctr, row['ctr']))
        mayo_fold_err = float(fold_error_log(mayo_ctr, row['ctr']))

        # 用集成std作为不确定性（±1σ）
        ci_low = float(ml_log10_ctr - ml_std[0])
        ci_high = float(ml_log10_ctr + ml_std[0])

        results.append({
            'id': row['id'],
            'raft_type': row['raft_type'],
            'method': row['method'],
            'ctr_true': row['ctr'],
            'log10_ctr_true': row['log10_ctr'],
            'ml_log10_ctr': ml_log10_ctr,
            'mayo_log10_ctr': mayo_log10_ctr,
            'ml_fold_error': ml_fold_err,
            'mayo_fold_error': mayo_fold_err,
            'ml_ci_low': ci_low,
            'ml_ci_high': ci_high,
            'ml_std': float(ml_std[0]),
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'validation_results.csv'), index=False)

    ml_summary = compute_summary_stats(results_df['ctr_true'], 10 ** results_df['ml_log10_ctr'])
    mayo_summary = compute_summary_stats(results_df['ctr_true'], 10 ** results_df['mayo_log10_ctr'])
    summary_dict = {'ml': ml_summary, 'mayo': mayo_summary, 'n_ensemble': n_ensemble}

    with open(os.path.join(output_dir, 'validation_summary.json'), 'w') as f_out:
        json.dump(summary_dict, f_out, indent=2)

    plot_parity_ml_vs_mayo(results_df, output_dir)

    print(f"\nValidated {len(results_df)} / {len(df)} literature points (ensemble n={n_ensemble})")
    return results_df, summary_dict


def plot_parity_ml_vs_mayo(results_df, output_dir='figures/validation'):
    """生成ML vs Mayo的log-log parity图，保存为PNG。"""
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))

    lims = [-1, 5]
    ax.plot(lims, lims, 'k-', lw=1.5, zorder=1, label='Identity')
    ax.plot(lims, [l + np.log10(2) for l in lims], 'k--', lw=0.8, alpha=0.5)
    ax.plot(lims, [l - np.log10(2) for l in lims], 'k--', lw=0.8, alpha=0.5)

    for raft_type, color in RAFT_COLORS.items():
        sub = results_df[results_df['raft_type'] == raft_type]
        if sub.empty:
            continue
        x = sub['log10_ctr_true'].values
        y_ml = sub['ml_log10_ctr'].values
        y_mayo = sub['mayo_log10_ctr'].values

        for xi, ymi, yma in zip(x, y_ml, y_mayo):
            ax.plot([xi, xi], [ymi, yma], color='gray', lw=0.6, alpha=0.5, zorder=2)

        has_ci = not sub['ml_ci_low'].isna().all()
        if has_ci:
            yerr_low = y_ml - sub['ml_ci_low'].values
            yerr_high = sub['ml_ci_high'].values - y_ml
            ax.errorbar(x, y_ml, yerr=[yerr_low, yerr_high],
                        fmt='o', color=color, ms=8, capsize=3, zorder=4,
                        label=f'{raft_type} (ML)')
        else:
            ax.scatter(x, y_ml, color=color, s=64, marker='o', zorder=4,
                       label=f'{raft_type} (ML)')

        ax.scatter(x, y_mayo, color=color, s=64, marker='D',
                   facecolors='none', linewidths=1.5, zorder=3,
                   label=f'{raft_type} (Mayo)')

    ax.set_xlabel('Published log\u2081\u2080(Ctr)', fontsize=13)
    ax.set_ylabel('Predicted log\u2081\u2080(Ctr)', fontsize=13)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    ax.legend(fontsize=8, loc='upper left', ncol=2)
    ax.set_title('ML vs Mayo Baseline: Literature Ctr Validation', fontsize=12)

    fig.savefig(os.path.join(output_dir, 'parity_ml_vs_mayo.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Literature validation: ML vs Mayo')
    parser.add_argument('--csv', default='data/literature/literature_ctr.csv')
    parser.add_argument('--model', default='checkpoints/best_model.pth')
    parser.add_argument('--bootstrap', default=None)
    parser.add_argument('--calibration', default=None)
    parser.add_argument('--output-dir', default='figures/validation')
    parser.add_argument('--sigma', type=float, default=0.03)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-ensemble', type=int, default=50)
    args = parser.parse_args()
    results_df, summary = run_validation_pipeline(
        args.csv, args.model, args.bootstrap, args.calibration,
        args.output_dir, args.sigma, args.device, args.seed,
        n_ensemble=args.n_ensemble
    )
    print("\n=== Validation Summary ===")
    print(json.dumps(summary, indent=2))
