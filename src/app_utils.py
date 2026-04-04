"""
src/app_utils.py — Web应用工具函数

将用户原始输入转换为模型推理所需格式的桥接模块。
所有可测试的业务逻辑（验证、归一化、模板生成、结果格式化）集中于此，
与 Streamlit UI 层解耦，支持自动化测试。

导出函数:
    validate_input       — 验证输入 DataFrame
    prepare_ctfp_input   — 归一化桥接（原始输入 → ctFP 编码格式）
    generate_template    — 生成 Excel 模板
    format_results       — 格式化预测结果（含 CI 反变换）
"""
from io import BytesIO

import numpy as np
import pandas as pd


def validate_input(df: pd.DataFrame) -> tuple:
    """验证用户输入的 DataFrame。

    处理流程：
    1. 删除所有列均为 NaN 的行（data_editor 的空行）
    2. 删除任意列含 NaN 的行（不完整行）
    3. 检查有效数据点数量 >= 3
    4. 逐行验证各参数范围

    Args:
        df: 包含 '[CTA]/[M]', 'conversion', 'Mn', 'D' 列的 DataFrame

    Returns:
        (df_valid, errors) 元组：
          - df_valid: 清洗后的 DataFrame；若不足 3 个有效点则为 None
          - errors:   错误消息列表；无错误则为空列表
          调用方可在 errors 非空时决定是否继续（除 < 3 点的 fatal 情况外）。
    """
    errors = []

    # 删除所有列均为 NaN 的空行（data_editor 动态添加的空行）
    df_clean = df.dropna(how='all')

    # 删除任意列含 NaN 的不完整行
    df_valid = df_clean.dropna()

    if len(df_valid) < 3:
        errors.append(
            f"At least 3 complete data points required (found {len(df_valid)})."
        )
        return None, errors

    # 逐行验证，行号使用 1-based（原始索引 + 1，与用户视角一致）
    for idx, row in df_valid.iterrows():
        row_num = idx + 1

        if not (0 < row['conversion'] < 1):
            errors.append(
                f"Row {row_num}: conversion must be in (0, 1), got {row['conversion']:.3f}."
            )
        if row['Mn'] <= 0:
            errors.append(
                f"Row {row_num}: Mn must be > 0, got {row['Mn']:.1f}."
            )
        if row['D'] < 1.0:
            errors.append(
                f"Row {row_num}: Dispersity (\u0110) must be >= 1.0, got {row['D']:.3f}."
            )
        if row['[CTA]/[M]'] <= 0:
            errors.append(
                f"Row {row_num}: [CTA]/[M] must be > 0, got {row['[CTA]/[M]']:.6f}."
            )

    return df_valid, errors


def prepare_ctfp_input(df: pd.DataFrame, m_monomer: float) -> list:
    """将原始用户输入归一化为 ctFP 编码器所需的格式。

    归一化规则（与训练数据保持一致）：
    - cta_ratio_norm = [CTA]/[M] / 0.1          （归一化至 [0, 1] 范围）
    - mn_theory      = m_monomer / [CTA]/[M]    （逐行计算，各行 [CTA]/[M] 可能不同）
    - mn_norm        = Mn / mn_theory            （无量纲）

    Args:
        df:        包含 '[CTA]/[M]', 'conversion', 'Mn', 'D' 列的 DataFrame
        m_monomer: 单体分子量（g/mol），用于计算理论分子量 Mn_theory

    Returns:
        (cta_ratio_norm, conversion, mn_norm, dispersity) 元组列表，
        可直接传入 ctfp_encoder.transform()
    """
    cta_ratio = df['[CTA]/[M]'].values
    cta_ratio_norm = cta_ratio / 0.1

    # Mn_theory = M_monomer / [CTA]/[M]（逐行，因各行 [CTA]/[M] 可能不同）
    mn_theory = m_monomer / cta_ratio
    mn_norm = df['Mn'].values / mn_theory

    conversion = df['conversion'].values
    dispersity = df['D'].values

    return list(zip(cta_ratio_norm, conversion, mn_norm, dispersity))


def generate_template() -> BytesIO:
    """生成包含示例数据的 Excel 模板文件。

    模板列名与内部 DataFrame 列名完全一致（ASCII），
    避免 Unicode 编码问题（RESEARCH.md Pitfall 6）。

    Returns:
        BytesIO: 包含 xlsx 内容的内存缓冲区（已 seek(0)）
    """
    template_df = pd.DataFrame({
        '[CTA]/[M]': [0.005, 0.005, 0.005],
        'conversion': [0.10, 0.30, 0.50],
        'Mn':         [15000, 42000, 68000],
        'D':          [1.15, 1.20, 1.25],
    })
    buf = BytesIO()
    template_df.to_excel(buf, index=False, engine='openpyxl')
    buf.seek(0)
    return buf


def format_results(mean_pred: np.ndarray, half_width) -> dict:
    """格式化预测结果，将 log10(Ctr) 反变换至原始尺度并计算置信区间。

    Ctr 的置信区间在原始尺度下是非对称的（因 log 变换的非线性）：
        ctr_lower = 10^(log10_ctr - hw[0])
        ctr_upper = 10^(log10_ctr + hw[0])

    诱导期和减速因子的置信区间在原始尺度下是对称的：
        value_lower = value - hw
        value_upper = value + hw

    Args:
        mean_pred:  np.ndarray(3,) — [log10_Ctr, inhibition_period, retardation_factor]
        half_width: np.ndarray(3,) 或 None（点估计模式，CI 字段返回 None）

    Returns:
        dict，包含键：
          'ctr', 'log10_ctr', 'ctr_lower', 'ctr_upper',
          'inhibition_period', 'inh_lower', 'inh_upper',
          'retardation_factor', 'ret_lower', 'ret_upper'
    """
    log10_ctr = mean_pred[0]
    inh = mean_pred[1]
    ret = mean_pred[2]

    ctr = 10 ** log10_ctr

    if half_width is not None:
        ctr_lower = 10 ** (log10_ctr - half_width[0])
        ctr_upper = 10 ** (log10_ctr + half_width[0])
        inh_lower = inh - half_width[1]
        inh_upper = inh + half_width[1]
        ret_lower = ret - half_width[2]
        ret_upper = ret + half_width[2]
    else:
        ctr_lower = ctr_upper = None
        inh_lower = inh_upper = None
        ret_lower = ret_upper = None

    return {
        'ctr':               ctr,
        'log10_ctr':         log10_ctr,
        'ctr_lower':         ctr_lower,
        'ctr_upper':         ctr_upper,
        'inhibition_period': inh,
        'inh_lower':         inh_lower,
        'inh_upper':         inh_upper,
        'retardation_factor': ret,
        'ret_lower':         ret_lower,
        'ret_upper':         ret_upper,
    }
