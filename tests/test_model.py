# tests/test_model.py
# TRN-01 测试覆盖：SimpViT模型结构与前向传播

import torch
import pytest
from model import SimpViT


def test_simpvit_forward():
    """SimpViT(num_outputs=3) 对 (B=4, 2, 64, 64) 输入产生 (4, 3) 输出张量。"""
    model = SimpViT(num_outputs=3)
    x = torch.randn(4, 2, 64, 64)
    output = model(x)
    assert output.shape == (4, 3), f"期望输出形状 (4, 3), 实际得到 {output.shape}"


def test_simpvit_param_count():
    """SimpViT(num_outputs=3) 总参数量应在 800_000 到 950_000 之间（实测约877K）。

    规划文档中的 ~3.4M 预估值是规划错误（混淆了 hidden_size=64 和 hidden_size=256）。
    实际架构（hidden_size=64, dim_feedforward=2048 默认值, num_layers=2）的正确参数量约为 877,571。
    """
    model = SimpViT(num_outputs=3)
    total_params = sum(p.numel() for p in model.parameters())
    assert 800_000 <= total_params <= 950_000, (
        f"参数量 {total_params:,} 不在预期范围 [800_000, 950_000] 内"
    )


def test_simpvit_eval_mode():
    """eval模式下 SimpViT(num_outputs=3) 对 (1, 2, 64, 64) 输入产生有限值的 (1, 3) 输出。"""
    model = SimpViT(num_outputs=3)
    model.eval()
    x = torch.randn(1, 2, 64, 64)
    output = model(x)
    assert output.shape == (1, 3), f"期望输出形状 (1, 3), 实际得到 {output.shape}"
    assert torch.isfinite(output).all(), "eval模式下输出包含非有限值（NaN或Inf）"
