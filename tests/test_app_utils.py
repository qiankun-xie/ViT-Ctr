"""
tests/test_app_utils.py — app_utils 模块的单元测试

覆盖 validate_input, prepare_ctfp_input, generate_template, format_results 四个函数。
"""
import numpy as np
import pandas as pd
import pytest

from src.app_utils import (
    format_results,
    generate_template,
    prepare_ctfp_input,
    validate_input,
)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _make_valid_df(n=5):
    """生成 n 行有效输入 DataFrame。"""
    base = {
        '[CTA]/[M]': [0.005, 0.005, 0.005, 0.005, 0.005],
        'conversion': [0.10, 0.20, 0.30, 0.40, 0.50],
        'Mn':         [15000, 25000, 42000, 55000, 68000],
        'D':          [1.10, 1.15, 1.20, 1.22, 1.25],
    }
    return pd.DataFrame({k: v[:n] for k, v in base.items()})


# ---------------------------------------------------------------------------
# validate_input
# ---------------------------------------------------------------------------

def test_validate_input_valid_5_rows():
    """有效5行DataFrame返回 (clean_df with 5 rows, [])。"""
    df = _make_valid_df(5)
    clean_df, errors = validate_input(df)
    assert clean_df is not None
    assert len(clean_df) == 5
    assert errors == []


def test_validate_input_too_few_rows():
    """仅2个有效行返回 (None, [错误消息])。"""
    df = _make_valid_df(2)
    clean_df, errors = validate_input(df)
    assert clean_df is None
    assert errors == ["At least 3 complete data points required (found 2)."]


def test_validate_input_conversion_out_of_range():
    """conversion=1.5 返回对应行的错误消息（含1.500格式）。"""
    df = pd.DataFrame({
        '[CTA]/[M]': [0.005, 0.005, 0.005],
        'conversion': [1.5, 0.3, 0.5],
        'Mn':         [50000.0, 50000.0, 50000.0],
        'D':          [1.2, 1.2, 1.2],
    })
    clean_df, errors = validate_input(df)
    assert clean_df is not None  # 仍返回 df_valid（非 fatal）
    assert len(errors) == 1
    assert errors[0] == "Row 1: conversion must be in (0, 1), got 1.500."


def test_validate_input_mn_nonpositive():
    """Mn=-100 返回对应行的错误消息（含-100.0格式）。"""
    df = pd.DataFrame({
        '[CTA]/[M]': [0.005, 0.005, 0.005],
        'conversion': [0.1, 0.3, 0.5],
        'Mn':         [-100.0, 50000.0, 50000.0],
        'D':          [1.2, 1.2, 1.2],
    })
    clean_df, errors = validate_input(df)
    assert clean_df is not None
    assert len(errors) == 1
    assert errors[0] == "Row 1: Mn must be > 0, got -100.0."


def test_validate_input_dispersity_low():
    """D=0.8 返回含 Đ 字符的错误消息。"""
    df = pd.DataFrame({
        '[CTA]/[M]': [0.005, 0.005, 0.005],
        'conversion': [0.1, 0.3, 0.5],
        'Mn':         [50000.0, 50000.0, 50000.0],
        'D':          [0.8, 1.2, 1.2],
    })
    clean_df, errors = validate_input(df)
    assert clean_df is not None
    assert len(errors) == 1
    assert errors[0] == "Row 1: Dispersity (\u0110) must be >= 1.0, got 0.800."


def test_validate_input_cta_ratio_nonpositive():
    """[CTA]/[M]=-0.01 返回含6位小数的错误消息。"""
    df = pd.DataFrame({
        '[CTA]/[M]': [-0.01, 0.005, 0.005],
        'conversion': [0.1, 0.3, 0.5],
        'Mn':         [50000.0, 50000.0, 50000.0],
        'D':          [1.2, 1.2, 1.2],
    })
    clean_df, errors = validate_input(df)
    assert clean_df is not None
    assert len(errors) == 1
    assert errors[0] == "Row 1: [CTA]/[M] must be > 0, got -0.010000."


def test_validate_input_drops_nan_rows():
    """5行中3行含NaN，有效行仅2行，返回 (None, [错误消息])。"""
    df = pd.DataFrame({
        '[CTA]/[M]': [0.005, 0.005, 0.005, 0.005, 0.005],
        'conversion': [0.10, np.nan, np.nan, np.nan, 0.50],
        'Mn':         [15000.0, np.nan, np.nan, np.nan, 68000.0],
        'D':          [1.15, np.nan, np.nan, np.nan, 1.25],
    })
    clean_df, errors = validate_input(df)
    assert clean_df is None
    assert errors == ["At least 3 complete data points required (found 2)."]


def test_validate_input_all_nan_row_dropped():
    """全NaN行被删除后不计入有效数据点数。"""
    df = pd.DataFrame({
        '[CTA]/[M]': [0.005, np.nan, 0.005, 0.005],
        'conversion': [0.1, np.nan, 0.3, 0.5],
        'Mn':         [50000.0, np.nan, 50000.0, 50000.0],
        'D':          [1.2, np.nan, 1.2, 1.2],
    })
    clean_df, errors = validate_input(df)
    assert clean_df is not None
    assert len(clean_df) == 3
    assert errors == []


# ---------------------------------------------------------------------------
# prepare_ctfp_input
# ---------------------------------------------------------------------------

def test_prepare_ctfp_input_normalization():
    """cta_ratio_norm = [CTA]/[M] / 0.1, mn_norm = Mn / (M_monomer / [CTA]/[M])。"""
    df = pd.DataFrame({
        '[CTA]/[M]': [0.005],
        'conversion': [0.3],
        'Mn':         [50000.0],
        'D':          [1.2],
    })
    m_monomer = 100.12
    result = prepare_ctfp_input(df, m_monomer)

    assert len(result) == 1
    cta_norm, conv, mn_norm, disp = result[0]

    # cta_ratio_norm = 0.005 / 0.1 = 0.05
    assert abs(cta_norm - 0.05) < 1e-9
    # mn_theory = 100.12 / 0.005 = 20024; mn_norm = 50000 / 20024
    mn_theory = m_monomer / 0.005
    assert abs(mn_norm - 50000.0 / mn_theory) < 1e-6
    assert conv == pytest.approx(0.3)
    assert disp == pytest.approx(1.2)


def test_prepare_ctfp_input_compatible_with_transform():
    """输出元组可直接传入 ctfp_encoder.transform()，张量形状正确。"""
    from src.ctfp_encoder import transform

    df = pd.DataFrame({
        '[CTA]/[M]': [0.005, 0.005, 0.005],
        'conversion': [0.1, 0.3, 0.5],
        'Mn':         [15000.0, 42000.0, 68000.0],
        'D':          [1.15, 1.20, 1.25],
    })
    tuples = prepare_ctfp_input(df, m_monomer=100.12)
    fp = transform(tuples)
    assert fp.shape == (2, 64, 64)


# ---------------------------------------------------------------------------
# generate_template
# ---------------------------------------------------------------------------

def test_generate_template_columns_and_rows():
    """模板包含正确列名和3行示例数据。"""
    buf = generate_template()
    df = pd.read_excel(buf, engine='openpyxl')
    assert list(df.columns) == ['[CTA]/[M]', 'conversion', 'Mn', 'D']
    assert len(df) == 3


def test_generate_template_returns_bytesio():
    """generate_template 返回 BytesIO，可被 pandas 直接读取。"""
    from io import BytesIO
    buf = generate_template()
    assert isinstance(buf, BytesIO)
    # 验证可多次读取（seek to 0已完成）
    buf.seek(0)
    content = buf.read()
    assert len(content) > 0


# ---------------------------------------------------------------------------
# format_results
# ---------------------------------------------------------------------------

def test_format_results_with_ci():
    """log10(Ctr)=2.0, hw=0.3 → Ctr=100, CI=[10^1.7, 10^2.3]（非对称）。"""
    mean = np.array([2.0, 0.05, 1.1])
    hw = np.array([0.3, 0.01, 0.05])
    r = format_results(mean, hw)

    assert r['log10_ctr'] == pytest.approx(2.0)
    assert r['ctr'] == pytest.approx(100.0)
    assert r['ctr_lower'] == pytest.approx(10 ** 1.7)
    assert r['ctr_upper'] == pytest.approx(10 ** 2.3)

    assert r['inhibition_period'] == pytest.approx(0.05)
    assert r['inh_lower'] == pytest.approx(0.04)
    assert r['inh_upper'] == pytest.approx(0.06)

    assert r['retardation_factor'] == pytest.approx(1.1)
    assert r['ret_lower'] == pytest.approx(1.05)
    assert r['ret_upper'] == pytest.approx(1.15)


def test_format_results_no_ci():
    """half_width=None 时，所有CI字段为None（点估计模式）。"""
    mean = np.array([2.0, 0.05, 1.1])
    r = format_results(mean, None)

    assert r['ctr'] == pytest.approx(100.0)
    assert r['log10_ctr'] == pytest.approx(2.0)
    assert r['ctr_lower'] is None
    assert r['ctr_upper'] is None
    assert r['inhibition_period'] == pytest.approx(0.05)
    assert r['inh_lower'] is None
    assert r['inh_upper'] is None
    assert r['retardation_factor'] == pytest.approx(1.1)
    assert r['ret_lower'] is None
    assert r['ret_upper'] is None
