"""
ctFP编码器单元测试。
验证chain transfer fingerprint编码的正确性：形状、通道分配、轴映射、确定性。
"""
import math
import inspect

import torch
import pytest

from src.ctfp_encoder import transform


class TestOutputShape:
    """transform()输出形状和类型测试。"""

    def test_output_shape(self):
        """transform() with 10 data points returns tensor of shape (2, 64, 64)."""
        data = [(i / 10, i / 10, 1.0, 1.5) for i in range(10)]
        result = transform(data)
        assert result.shape == (2, 64, 64)

    def test_output_dtype(self):
        """返回的tensor dtype是torch.float32。"""
        data = [(0.5, 0.3, 1.2, 1.5)]
        result = transform(data)
        assert result.dtype == torch.float32

    def test_custom_img_size(self):
        """支持自定义img_size参数。"""
        data = [(0.5, 0.3, 1.2, 1.5)]
        result = transform(data, img_size=32)
        assert result.shape == (2, 32, 32)


class TestChannelAssignment:
    """通道分配和像素值测试。"""

    def test_channel_assignment(self):
        """单数据点(cta_norm=0.5, conv=0.3, mn_norm=1.2, disp=1.5)
        在channel 0的row=floor(0.3*64)=19, col=floor(0.5*64)=32处放置mn_norm=1.2，
        在channel 1的相同位置放置disp=1.5。"""
        data = [(0.5, 0.3, 1.2, 1.5)]
        result = transform(data)

        row = math.floor(0.3 * 64)  # 19
        col = math.floor(0.5 * 64)  # 32

        assert result[0, row, col].item() == pytest.approx(1.2)
        assert result[1, row, col].item() == pytest.approx(1.5)

    def test_mn_in_channel_0(self):
        """Mn归一化值放在channel 0。"""
        data = [(0.1, 0.2, 0.8, 2.0)]
        result = transform(data)
        row = math.floor(0.2 * 64)
        col = math.floor(0.1 * 64)
        assert result[0, row, col].item() == pytest.approx(0.8)

    def test_dispersity_in_channel_1(self):
        """Dispersity值放在channel 1。"""
        data = [(0.1, 0.2, 0.8, 2.0)]
        result = transform(data)
        row = math.floor(0.2 * 64)
        col = math.floor(0.1 * 64)
        assert result[1, row, col].item() == pytest.approx(2.0)


class TestAxisMapping:
    """坐标轴映射测试：x=[CTA]/[M] (列索引), y=conversion (行索引)。"""

    def test_axis_mapping(self):
        """x-axis is [CTA]/[M] (maps to column), y-axis is conversion (maps to row)。
        独立验证行列分配。"""
        # 两个数据点，不同的cta_norm和conv
        data_a = [(0.8, 0.1, 1.0, 1.0)]  # 高cta_norm -> 高列号
        data_b = [(0.1, 0.8, 1.0, 1.0)]  # 高conv -> 高行号

        result_a = transform(data_a)
        result_b = transform(data_b)

        # data_a: col=floor(0.8*64)=51, row=floor(0.1*64)=6
        assert result_a[0, 6, 51].item() == pytest.approx(1.0)
        # data_b: col=floor(0.1*64)=6, row=floor(0.8*64)=51
        assert result_b[0, 51, 6].item() == pytest.approx(1.0)

    def test_cta_ratio_maps_to_column(self):
        """[CTA]/[M]值映射到列索引。"""
        data = [(0.0, 0.5, 1.0, 1.0)]
        result = transform(data)
        row = math.floor(0.5 * 64)
        assert result[0, row, 0].item() == pytest.approx(1.0)

    def test_conversion_maps_to_row(self):
        """Conversion值映射到行索引。"""
        data = [(0.5, 0.0, 1.0, 1.0)]
        result = transform(data)
        col = math.floor(0.5 * 64)
        assert result[0, 0, col].item() == pytest.approx(1.0)


class TestDeterministicOutput:
    """确定性输出测试。"""

    def test_deterministic_output(self):
        """相同输入两次调用产生完全相同的tensor（torch.equal返回True）。"""
        data = [
            (0.1, 0.2, 0.8, 1.3),
            (0.5, 0.7, 1.1, 1.8),
            (0.9, 0.4, 0.6, 2.5),
        ]
        result1 = transform(data)
        result2 = transform(data)
        assert torch.equal(result1, result2)


class TestEdgeCases:
    """边界值和空输入测试。"""

    def test_empty_input(self):
        """transform([])返回全零tensor，形状(2, 64, 64)。"""
        result = transform([])
        assert result.shape == (2, 64, 64)
        assert torch.all(result == 0)

    def test_boundary_values(self):
        """边界值映射测试：
        conversion=0.0 -> row 0,
        conversion=0.99 -> row 63 (clamped),
        cta_norm=0.0 -> col 0,
        cta_norm=1.0 -> col 63 (clamped)。"""
        data = [
            (0.0, 0.0, 1.0, 1.0),    # 左上角
            (1.0, 0.99, 2.0, 2.0),   # 右下角（clamped）
        ]
        result = transform(data)

        # (0.0, 0.0) -> row=0, col=0
        assert result[0, 0, 0].item() == pytest.approx(1.0)
        # (1.0, 0.99) -> row=63(clamped), col=63(clamped)
        assert result[0, 63, 63].item() == pytest.approx(2.0)

    def test_negative_clamping(self):
        """负值输入被clamp到0。"""
        data = [(-0.1, -0.1, 1.0, 1.0)]
        result = transform(data)
        # 应该clamp到row=0, col=0
        assert result[0, 0, 0].item() == pytest.approx(1.0)


class TestValueRange:
    """值域验证测试。"""

    def test_mn_not_raw(self):
        """mn_norm值应该是无量纲的（约0-2范围），不是原始g/mol值。
        编码器传递原始值不做变换，但测试确认正常范围的值被正确存储。"""
        # 正常范围mn_norm值
        data = [(0.5, 0.5, 1.05, 1.2)]
        result = transform(data)
        row = math.floor(0.5 * 64)
        col = math.floor(0.5 * 64)
        val = result[0, row, col].item()
        # mn_norm应该在合理范围（0-2）
        assert 0 <= val <= 2.0

    def test_dispersity_range(self):
        """Dispersity值直接传递（无归一化），典型范围1.0-3.0。"""
        data = [(0.5, 0.5, 1.0, 2.5)]
        result = transform(data)
        row = math.floor(0.5 * 64)
        col = math.floor(0.5 * 64)
        val = result[1, row, col].item()
        assert val == pytest.approx(2.5)

    def test_dispersity_clipping(self):
        """Dispersity > 4.0被clip到4.0。"""
        data = [(0.5, 0.5, 1.0, 6.0)]
        result = transform(data)
        row = math.floor(0.5 * 64)
        col = math.floor(0.5 * 64)
        val = result[1, row, col].item()
        assert val == pytest.approx(4.0)


class TestNoFrameworkDependency:
    """确保ctfp_encoder.py没有框架依赖。"""

    def test_no_framework_dependency(self):
        """ctfp_encoder.py不导入streamlit，不导入torch.nn。"""
        import src.ctfp_encoder as mod
        source = inspect.getsource(mod)
        assert "import streamlit" not in source
        assert "import torch.nn" not in source
        assert "from streamlit" not in source
        assert "from torch.nn" not in source
