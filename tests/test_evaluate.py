# tests/test_evaluate.py — EVL-03测试覆盖：按RAFT类型的分类评估
"""
验证:
- per_class_metrics 返回4键字典（按RAFT类型名称），每个值含 r2/rmse/mae
- compute_test_metrics 对完美预测返回正确结果（r2=1.0或ss_tot=0时0.0）
"""

import numpy as np
import pytest

from evaluate import per_class_metrics, compute_test_metrics


# ============================================================
# EVL-03: 按RAFT类型分类评估
# ============================================================

def test_per_class_eval():
    """per_class_metrics返回4键字典，每键含 r2/rmse/mae。"""
    rng = np.random.default_rng(0)

    # 40样本，每类10个（class_id: 0,1,2,3）
    n_per_class = 10
    n_classes = 4
    n_total = n_per_class * n_classes

    y_true = rng.random((n_total, 3), dtype=np.float32)
    y_pred = y_true + rng.normal(0, 0.1, (n_total, 3)).astype(np.float32)
    class_ids = np.repeat(np.arange(n_classes), n_per_class)

    result = per_class_metrics(y_true, y_pred, class_ids)

    # 返回值为字典
    assert isinstance(result, dict), f"期望dict，实际 {type(result)}"

    # 必须含4个RAFT类型键
    expected_keys = ['dithioester', 'trithiocarbonate', 'xanthate', 'dithiocarbamate']
    for key in expected_keys:
        assert key in result, f"缺少键: {key}"

    # 每个值含 r2/rmse/mae，各为长度3的序列
    for key in expected_keys:
        metrics = result[key]
        assert 'r2' in metrics, f"{key} 缺少 r2"
        assert 'rmse' in metrics, f"{key} 缺少 rmse"
        assert 'mae' in metrics, f"{key} 缺少 mae"
        assert len(metrics['r2']) == 3, f"{key}.r2 长度应为3，实际 {len(metrics['r2'])}"
        assert len(metrics['rmse']) == 3, f"{key}.rmse 长度应为3，实际 {len(metrics['rmse'])}"
        assert len(metrics['mae']) == 3, f"{key}.mae 长度应为3，实际 {len(metrics['mae'])}"


# ============================================================
# EVL-01: 全输出计算指标
# ============================================================

def test_compute_metrics_all_outputs():
    """compute_test_metrics对完美预测返回r2=1.0（或ss_tot=0时0.0），长度为3的列表。"""
    y_true = np.zeros((20, 3), dtype=np.float32)
    y_pred = np.zeros((20, 3), dtype=np.float32)

    result = compute_test_metrics(y_true, y_pred)

    # 返回值为字典
    assert isinstance(result, dict), f"期望dict，实际 {type(result)}"
    assert 'r2' in result, "缺少 r2"
    assert 'rmse' in result, "缺少 rmse"
    assert 'mae' in result, "缺少 mae"

    # 每个指标均有3个输出
    assert len(result['r2']) == 3, f"r2长度应为3，实际 {len(result['r2'])}"
    assert len(result['rmse']) == 3, f"rmse长度应为3，实际 {len(result['rmse'])}"
    assert len(result['mae']) == 3, f"mae长度应为3，实际 {len(result['mae'])}"

    # 完美预测：ss_tot=0（全为0），r2应返回0.0（per r2_score_np规范）
    for r2 in result['r2']:
        assert r2 == 1.0 or r2 == 0.0, f"r2={r2} 应为1.0（完美）或0.0（ss_tot=0时）"

    # RMSE和MAE均为0（完美预测）
    for rmse in result['rmse']:
        assert rmse == pytest.approx(0.0, abs=1e-6), f"完美预测RMSE应为0，实际 {rmse}"
    for mae in result['mae']:
        assert mae == pytest.approx(0.0, abs=1e-6), f"完美预测MAE应为0，实际 {mae}"
