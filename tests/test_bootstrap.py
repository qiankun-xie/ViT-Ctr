# Bootstrap UQ测试 — UQ-01和UQ-02覆盖
import numpy as np
import torch
import pytest
from model import SimpViT
from bootstrap import (
    freeze_backbone,
    run_bootstrap,
    compute_jci,
    compute_coverage,
    calibrate_coverage,
    predict_with_uncertainty,
)


def test_bootstrap_produces_heads():
    """测试run_bootstrap生成正确数量的头并保存backbone。"""
    model = SimpViT(num_outputs=3)

    # 创建dummy DataLoader: 20个随机样本
    dummy_fps = torch.randn(20, 2, 64, 64)
    dummy_labels = torch.randn(20, 3)
    from torch.utils.data import TensorDataset, DataLoader
    dummy_ds = TensorDataset(dummy_fps, dummy_labels)
    dummy_loader = DataLoader(dummy_ds, batch_size=4, shuffle=True)

    result = run_bootstrap(
        model, train_loader=dummy_loader,
        n_bootstrap=3, n_epochs=2, lr=1e-3,
        device='cpu', seed=0
    )

    assert len(result['heads']) == 3
    for head in result['heads']:
        assert 'fc.weight' in head
        assert 'fc.bias' in head
    assert 'base_model_state_dict' in result
    assert isinstance(result['base_model_state_dict'], dict)
    assert result['n_bootstrap'] == 3
    assert result['debug_mode'] is True  # n_bootstrap < 50


def test_f_dist_jci():
    """测试F分布JCI公式的正确性。"""
    cov_matrix = np.diag([4.0, 1.0, 0.25])
    half_width = compute_jci(cov_matrix, n=200, p=3)

    from scipy.stats import f as fdist
    f_val = fdist.ppf(0.95, dfn=3, dfd=197)
    expected = np.sqrt(np.array([4.0, 1.0, 0.25]) * 3 * f_val / 197)

    np.testing.assert_allclose(half_width, expected, rtol=1e-5)


def test_calibration_factors():
    """测试校准因子计算：窄CI应产生>1的因子。"""
    val_true = np.zeros((100, 3))
    val_pred_mean = np.zeros((100, 3))
    val_pred_ci = np.full((100, 3), 0.001)  # 极窄CI → 低覆盖率

    cal_factors = calibrate_coverage(val_true, val_pred_mean, val_pred_ci, target=0.95)

    assert len(cal_factors) == 3
    assert all(f >= 1.0 for f in cal_factors)


def test_predict_with_uncertainty():
    """测试单样本不确定性预测。"""
    model = SimpViT(num_outputs=3)

    # 创建5个dummy头（n=5, p=3, dfd=2 > 0）
    heads = []
    for i in range(5):
        # 添加小扰动使预测有差异
        fc_weight = model.fc.weight.data.clone() + torch.randn_like(model.fc.weight.data) * 0.01
        fc_bias = model.fc.bias.data.clone() + torch.randn(3) * 0.01
        heads.append({
            'fc.weight': fc_weight,
            'fc.bias': fc_bias,
        })

    bootstrap_ckpt = {
        'heads': heads,
        'base_model_state_dict': model.state_dict(),
        'n_bootstrap': 5,
        'debug_mode': True,
    }

    fp_tensor = torch.randn(1, 2, 64, 64)
    mean, half_width = predict_with_uncertainty(
        model, fp_tensor, bootstrap_ckpt,
        cal_factors=[1.0, 1.0, 1.0], device='cpu'
    )

    assert mean.shape == (3,)
    assert half_width.shape == (3,)
    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(half_width))
