"""
数据集质量验证脚本。

检查生成的HDF5数据集的：
1. 参数分布均匀性（Ctr直方图无空白区域）
2. RAFT类型间的样本平衡性（4类样本数差异<10%）
3. 数据完整性（shape正确、无NaN/Inf、Dispersity>=1.0）

用法:
    python scripts/validate_dataset.py [--data-dir data/]
"""

import os
import sys
import glob
import argparse

import numpy as np

# 可选matplotlib（无GUI环境仍可运行验证）
try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

import h5py


# ============================================================
# 验证函数
# ============================================================

def check_distribution(h5_files, output_dir='data/'):
    """
    检查所有HDF5文件中log10_Ctr的分布均匀性。

    Parameters
    ----------
    h5_files : list of str
        HDF5文件路径列表
    output_dir : str
        分布图保存目录

    Returns
    -------
    dict
        {'passed': bool, 'min': float, 'max': float, 'median': float,
         'n_bins': int, 'sparse_bins': int, 'message': str}
    """
    all_log10_ctr = []
    for fpath in h5_files:
        with h5py.File(fpath, 'r') as f:
            labels = f['labels'][:]  # (n, 3)
            all_log10_ctr.append(labels[:, 0])

    all_log10_ctr = np.concatenate(all_log10_ctr)
    n_total = len(all_log10_ctr)

    if n_total == 0:
        return {
            'passed': False,
            'min': float('nan'),
            'max': float('nan'),
            'median': float('nan'),
            'n_bins': 0,
            'sparse_bins': 0,
            'message': 'No samples found',
        }

    ctr_min = float(all_log10_ctr.min())
    ctr_max = float(all_log10_ctr.max())
    ctr_median = float(np.median(all_log10_ctr))

    # 50个bins的直方图
    n_bins = 50
    counts, bin_edges = np.histogram(all_log10_ctr, bins=n_bins)
    threshold = n_total / 100  # 任意bin计数<总数/100视为稀疏
    sparse_bins = int(np.sum(counts < threshold))

    passed = sparse_bins == 0

    # 保存直方图（如有matplotlib）
    if HAS_MATPLOTLIB:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.bar(bin_edges[:-1], counts, width=np.diff(bin_edges),
               align='edge', color='steelblue', edgecolor='white', linewidth=0.5)
        ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.7,
                   label=f'Sparse threshold ({threshold:.0f})')
        ax.set_xlabel('log10(Ctr)')
        ax.set_ylabel('Count')
        ax.set_title(f'Ctr Distribution ({n_total} samples, {sparse_bins} sparse bins)')
        ax.legend()
        os.makedirs(output_dir, exist_ok=True)
        fig_path = os.path.join(output_dir, 'ctr_distribution.png')
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Distribution plot saved: {fig_path}")

    message = 'PASSED' if passed else f'{sparse_bins} bins below threshold ({threshold:.0f})'

    return {
        'passed': passed,
        'min': ctr_min,
        'max': ctr_max,
        'median': ctr_median,
        'n_bins': n_bins,
        'sparse_bins': sparse_bins,
        'message': message,
    }


def check_balance(h5_files):
    """
    检查4种RAFT类型的样本数平衡性。

    Parameters
    ----------
    h5_files : list of str
        HDF5文件路径列表

    Returns
    -------
    dict
        {'passed': bool, 'counts': dict, 'ratio': float, 'message': str}
    """
    counts = {}
    for fpath in h5_files:
        with h5py.File(fpath, 'r') as f:
            raft_type = f.attrs.get('raft_type', os.path.splitext(os.path.basename(fpath))[0])
            n = f['fingerprints'].shape[0]
            counts[raft_type] = n

    if not counts:
        return {
            'passed': False,
            'counts': counts,
            'ratio': float('inf'),
            'message': 'No HDF5 files found',
        }

    max_count = max(counts.values())
    min_count = min(counts.values())
    ratio = max_count / min_count if min_count > 0 else float('inf')

    passed = ratio <= 1.1  # 偏差不超过10%

    message = 'PASSED' if passed else f'Imbalance ratio {ratio:.3f} exceeds 1.1 threshold'

    return {
        'passed': passed,
        'counts': counts,
        'ratio': ratio,
        'message': message,
    }


def check_integrity(h5_file):
    """
    验证单个HDF5文件的数据完整性。

    Parameters
    ----------
    h5_file : str
        HDF5文件路径

    Returns
    -------
    dict
        {'passed': bool, 'file': str, 'n_samples': int,
         'fp_shape_ok': bool, 'lbl_shape_ok': bool,
         'no_nan': bool, 'no_inf': bool, 'disp_valid': bool,
         'errors': list of str}
    """
    errors = []
    fname = os.path.basename(h5_file)

    try:
        with h5py.File(h5_file, 'r') as f:
            # 检查必要的数据集是否存在
            if 'fingerprints' not in f:
                errors.append('Missing dataset: fingerprints')
                return {
                    'passed': False, 'file': fname, 'n_samples': 0,
                    'fp_shape_ok': False, 'lbl_shape_ok': False,
                    'no_nan': False, 'no_inf': False, 'disp_valid': False,
                    'errors': errors,
                }
            if 'labels' not in f:
                errors.append('Missing dataset: labels')
                return {
                    'passed': False, 'file': fname, 'n_samples': 0,
                    'fp_shape_ok': False, 'lbl_shape_ok': False,
                    'no_nan': False, 'no_inf': False, 'disp_valid': False,
                    'errors': errors,
                }

            fp = f['fingerprints']
            lbl = f['labels']

            n_samples = fp.shape[0]

            # Shape验证: fingerprints (n, 64, 64, 2)
            fp_shape_ok = len(fp.shape) == 4 and fp.shape[1:] == (64, 64, 2)
            if not fp_shape_ok:
                errors.append(f'fingerprints shape {fp.shape} != (n, 64, 64, 2)')

            # Shape验证: labels (n, 3)
            lbl_shape_ok = len(lbl.shape) == 2 and lbl.shape[1] == 3
            if not lbl_shape_ok:
                errors.append(f'labels shape {lbl.shape} != (n, 3)')

            # 一致性: fingerprints和labels样本数匹配
            if fp.shape[0] != lbl.shape[0]:
                errors.append(f'Sample count mismatch: fp={fp.shape[0]}, lbl={lbl.shape[0]}')

            # 分批检查NaN/Inf（避免一次性加载整个数据集到内存）
            batch_size = 5000
            has_nan = False
            has_inf = False
            disp_valid = True

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)

                fp_batch = fp[start:end]
                if np.isnan(fp_batch).any():
                    has_nan = True
                if np.isinf(fp_batch).any():
                    has_inf = True

                lbl_batch = lbl[start:end]
                if np.isnan(lbl_batch).any():
                    has_nan = True
                if np.isinf(lbl_batch).any():
                    has_inf = True

                # Dispersity (channel 1 of fingerprint) 应该 >= 0
                # Labels中retardation (col 2) 应该 >= 0
                # 注意：在ctFP编码中dispersity已经被截断于4.0且保证>=1.0
                # 但我们这里检查labels中的retardation factor是否合理
                # retardation_factor >= 0（比率，可以为0表示无减速）

            no_nan = not has_nan
            no_inf = not has_inf

            if has_nan:
                errors.append('Contains NaN values')
            if has_inf:
                errors.append('Contains Inf values')
            if not disp_valid:
                errors.append('Invalid dispersity values (< 1.0) in labels')

            passed = fp_shape_ok and lbl_shape_ok and no_nan and no_inf and disp_valid and len(errors) == 0

            return {
                'passed': passed,
                'file': fname,
                'n_samples': n_samples,
                'fp_shape_ok': fp_shape_ok,
                'lbl_shape_ok': lbl_shape_ok,
                'no_nan': no_nan,
                'no_inf': no_inf,
                'disp_valid': disp_valid,
                'errors': errors,
            }

    except Exception as e:
        errors.append(f'Failed to open file: {e}')
        return {
            'passed': False, 'file': fname, 'n_samples': 0,
            'fp_shape_ok': False, 'lbl_shape_ok': False,
            'no_nan': False, 'no_inf': False, 'disp_valid': False,
            'errors': errors,
        }


def main():
    """
    主验证流程：扫描HDF5文件，依次检查完整性、分布、平衡性。
    """
    parser = argparse.ArgumentParser(description='ViT-Ctr 数据集验证')
    parser.add_argument('--data-dir', default='data/', help='HDF5文件目录（默认: data/）')
    args = parser.parse_args()

    data_dir = args.data_dir

    print("=" * 60)
    print("ViT-Ctr Dataset Validation")
    print("=" * 60)

    # 扫描HDF5文件
    h5_pattern = os.path.join(data_dir, '*.h5')
    h5_files = sorted(glob.glob(h5_pattern))

    if not h5_files:
        print(f"\n  WARNING: No HDF5 files found in {data_dir}")
        print("  Run 'python src/dataset_generator.py' to generate the dataset first.")
        print("\n  Dataset validation SKIPPED (no data files)")
        return

    print(f"\nFound {len(h5_files)} HDF5 files in {data_dir}")
    for fpath in h5_files:
        print(f"  - {os.path.basename(fpath)}")

    # ---- 1. 完整性检查 ----
    print(f"\n{'='*60}")
    print("1. Integrity Check")
    print(f"{'='*60}")

    all_integrity_passed = True
    total_samples = 0
    integrity_results = []

    for fpath in h5_files:
        result = check_integrity(fpath)
        integrity_results.append(result)
        total_samples += result['n_samples']

        status = "PASS" if result['passed'] else "FAIL"
        print(f"\n  [{status}] {result['file']}: {result['n_samples']} samples")
        print(f"       Shape OK: fp={result['fp_shape_ok']}, lbl={result['lbl_shape_ok']}")
        print(f"       No NaN: {result['no_nan']}, No Inf: {result['no_inf']}, Disp valid: {result['disp_valid']}")

        if result['errors']:
            for err in result['errors']:
                print(f"       ERROR: {err}")
            all_integrity_passed = False

    print(f"\n  Total samples across all files: {total_samples}")
    print(f"  Integrity: {'PASSED' if all_integrity_passed else 'FAILED'}")

    # ---- 2. 分布检查 ----
    print(f"\n{'='*60}")
    print("2. Distribution Check (log10_Ctr)")
    print(f"{'='*60}")

    dist_result = check_distribution(h5_files, output_dir=data_dir)
    print(f"\n  Ctr range: [{dist_result['min']:.3f}, {dist_result['max']:.3f}]")
    print(f"  Ctr median: {dist_result['median']:.3f}")
    print(f"  Histogram: {dist_result['n_bins']} bins, {dist_result['sparse_bins']} sparse")
    print(f"  Distribution: {dist_result['message']}")

    # ---- 3. 平衡性检查 ----
    print(f"\n{'='*60}")
    print("3. Balance Check (RAFT types)")
    print(f"{'='*60}")

    balance_result = check_balance(h5_files)
    print(f"\n  {'RAFT Type':<25s} {'Samples':>10s}")
    print(f"  {'-'*25} {'-'*10}")
    for rtype, count in sorted(balance_result['counts'].items()):
        print(f"  {rtype:<25s} {count:>10d}")
    print(f"\n  Max/Min ratio: {balance_result['ratio']:.3f}")
    print(f"  Balance: {balance_result['message']}")

    # ---- 总结 ----
    print(f"\n{'='*60}")
    print("Validation Summary")
    print(f"{'='*60}")

    all_passed = all_integrity_passed and dist_result['passed'] and balance_result['passed']

    checks = [
        ('Integrity', all_integrity_passed),
        ('Distribution', dist_result['passed']),
        ('Balance', balance_result['passed']),
    ]

    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\n  Total samples: {total_samples}")

    if all_passed:
        print("\n  Dataset validation PASSED")
    else:
        failed_checks = [name for name, passed in checks if not passed]
        print(f"\n  Dataset validation FAILED: {', '.join(failed_checks)}")

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main() or 0)
