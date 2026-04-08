# Bootstrap UQ测试 — 只测 src/bootstrap.py 保留的推理接口
import json
import numpy as np
import torch
import pytest
from model import SimpViT
from bootstrap import compute_jci, predict_with_uncertainty, load_calibration, P_OUTPUTS


def test_compute_jci():
    """测试F分布JCI公式的正确性。"""
    cov_matrix = np.diag([4.0, 1.0, 0.25])
    half_width = compute_jci(cov_matrix, n=200, p=3)

    from scipy.stats import f as fdist
    f_val = fdist.ppf(0.95, dfn=3, dfd=197)
    expected = np.sqrt(np.array([4.0, 1.0, 0.25]) * 3 * f_val / 197)
    np.testing.assert_allclose(half_width, expected, rtol=1e-5)


def test_compute_jci_invalid_inputs():
    """测试compute_jci对无效输入的错误处理。"""
    cov = np.diag([1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="dfd"):
        compute_jci(cov, n=3, p=3)
    with pytest.raises(ValueError, match="dfd"):
        compute_jci(cov, n=2, p=3)
    bad_cov = np.array([[1, 0, 0], [0, np.inf, 0], [0, 0, 1.0]])
    with pytest.raises(ValueError, match="non-finite"):
        compute_jci(bad_cov, n=200, p=3)


def test_predict_with_uncertainty():
    """测试单样本不确定性预测的输出shape和值域。"""
    model = SimpViT(num_outputs=3)
    heads = []
    for _ in range(5):
        heads.append({
            'fc.weight': model.fc.weight.data.clone() + torch.randn_like(model.fc.weight.data) * 0.01,
            'fc.bias': model.fc.bias.data.clone() + torch.randn(3) * 0.01,
        })
    bootstrap_ckpt = {
        'heads': heads,
        'base_model_state_dict': model.state_dict(),
        'n_bootstrap': 5,
    }
    fp_tensor = torch.randn(1, 2, 64, 64)
    mean, half_width = predict_with_uncertainty(
        model, fp_tensor, bootstrap_ckpt, cal_factors=[1.0, 1.0, 1.0])
    assert mean.shape == (3,)
    assert half_width.shape == (3,)
    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(half_width))
    assert np.all(half_width > 0)


def test_predict_with_uncertainty_restores_model():
    """测试推理后模型恢复到base state。"""
    model = SimpViT(num_outputs=3)
    original_state = {k: v.clone() for k, v in model.state_dict().items()}
    heads = []
    for _ in range(5):
        heads.append({
            'fc.weight': model.fc.weight.data.clone() + torch.randn_like(model.fc.weight.data) * 0.1,
            'fc.bias': model.fc.bias.data.clone() + torch.randn(3) * 0.1,
        })
    bootstrap_ckpt = {
        'heads': heads,
        'base_model_state_dict': original_state,
        'n_bootstrap': 5,
    }
    fp_tensor = torch.randn(1, 2, 64, 64)
    predict_with_uncertainty(model, fp_tensor, bootstrap_ckpt,
                             cal_factors=[1.0, 1.0, 1.0])
    for key in original_state:
        torch.testing.assert_close(model.state_dict()[key], original_state[key])


def test_load_calibration(tmp_path):
    """测试calibration.json读取和格式校验。"""
    # 正常情况
    cal_data = {'cal_factors': [1.2, 1.0, 1.5], 'other_key': 'ignored'}
    cal_path = tmp_path / 'calibration.json'
    cal_path.write_text(json.dumps(cal_data))
    factors = load_calibration(str(cal_path))
    assert factors == [1.2, 1.0, 1.5]

    # 缺少key
    bad_path = tmp_path / 'bad.json'
    bad_path.write_text(json.dumps({'wrong_key': [1, 2, 3]}))
    with pytest.raises(ValueError, match="cal_factors"):
        load_calibration(str(bad_path))

    # 长度不对
    bad_path2 = tmp_path / 'bad2.json'
    bad_path2.write_text(json.dumps({'cal_factors': [1.0, 2.0]}))
    with pytest.raises(ValueError, match="3"):
        load_calibration(str(bad_path2))
