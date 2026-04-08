"""文献验证模块测试 — 验证pipeline输出包含inhibition/retardation列。"""
import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from literature_validation import (
    fold_error_log,
    fold_error_ratio,
    compute_summary_stats,
)


def test_fold_error_log_identity():
    """pred == true时fold-error应为1.0。"""
    assert fold_error_log(100, 100) == pytest.approx(1.0)


def test_fold_error_log_symmetric():
    """fold-error对称: fe(a,b) == fe(b,a)。"""
    assert fold_error_log(10, 100) == pytest.approx(fold_error_log(100, 10))


def test_fold_error_ratio_identity():
    assert fold_error_ratio(100, 100) == pytest.approx(1.0)


def test_compute_summary_stats_perfect():
    """完美预测时R²=1, RMSE=0, fold-error=1。"""
    true = np.array([1, 10, 100, 1000])
    stats = compute_summary_stats(true, true)
    assert stats['r2_log10'] == pytest.approx(1.0)
    assert stats['rmse_log10'] == pytest.approx(0.0, abs=1e-10)
    assert stats['median_fold_error'] == pytest.approx(1.0)
    assert stats['pct_within_2x'] == pytest.approx(100.0)


def test_pipeline_results_have_inhibition_retardation_columns():
    """验证pipeline输出DataFrame包含inhibition和retardation列。"""
    expected_cols = [
        'ml_inhibition', 'ml_retardation',
        'ml_inhibition_std', 'ml_retardation_std',
    ]
    from literature_validation import EXPECTED_RESULT_COLUMNS
    for col in expected_cols:
        assert col in EXPECTED_RESULT_COLUMNS, f"Missing column: {col}"


def test_plot_inhibition_retardation_by_class(tmp_path):
    """验证inhibition/retardation分类图生成无报错。"""
    from literature_validation import plot_inhibition_retardation_by_class
    rng = np.random.default_rng(42)
    rows = []
    for raft_type in ['dithioester', 'trithiocarbonate', 'xanthate', 'dithiocarbamate']:
        for i in range(5):
            inh = rng.uniform(0.05, 0.3) if raft_type == 'dithioester' else rng.uniform(0, 0.02)
            ret = rng.uniform(0.3, 0.8) if raft_type == 'dithioester' else rng.uniform(0.9, 1.0)
            rows.append({
                'raft_type': raft_type,
                'ml_inhibition': inh,
                'ml_retardation': ret,
                'ml_inhibition_std': 0.01,
                'ml_retardation_std': 0.02,
            })
    df = pd.DataFrame(rows)
    plot_inhibition_retardation_by_class(df, str(tmp_path))
    assert (tmp_path / 'inhibition_retardation_by_class.png').exists()


def test_plot_inhibition_retardation_by_class(tmp_path):
    """验证inhibition/retardation分类图生成无报错。"""
    from literature_validation import plot_inhibition_retardation_by_class
    rng = np.random.default_rng(42)
    rows = []
    for raft_type in ['dithioester', 'trithiocarbonate', 'xanthate', 'dithiocarbamate']:
        for i in range(5):
            inh = rng.uniform(0.05, 0.3) if raft_type == 'dithioester' else rng.uniform(0, 0.02)
            ret = rng.uniform(0.3, 0.8) if raft_type == 'dithioester' else rng.uniform(0.9, 1.0)
            rows.append({
                'raft_type': raft_type,
                'ml_inhibition': inh,
                'ml_retardation': ret,
                'ml_inhibition_std': 0.01,
                'ml_retardation_std': 0.02,
            })
    df = pd.DataFrame(rows)
    plot_inhibition_retardation_by_class(df, str(tmp_path))
    assert (tmp_path / 'inhibition_retardation_by_class.png').exists()
