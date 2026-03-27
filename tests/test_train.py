# 训练模块测试 — 覆盖TRN-03: 加权MSE损失、EarlyStopper、训练循环
import os
import pytest
import torch
import sys

# 确保src在路径中（pyproject.toml已配置pythonpath，此处保留以便直接运行）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ──────────────────────────────────────────────
# Test: weighted_mse_loss 基本行为
# ──────────────────────────────────────────────

def test_weighted_mse_loss_ctr_only():
    """只有ctr预测误差为1.0时，加权loss = 2.0 * 1.0 = 2.0"""
    from train import weighted_mse_loss
    pred   = torch.tensor([[1.0, 2.0, 3.0]])   # (1, 3)
    target = torch.tensor([[2.0, 2.0, 3.0]])   # 只有ctr差1.0
    loss = weighted_mse_loss(pred, target)
    # 期望: 2.0*1.0 + 0.5*0.0 + 0.5*0.0 = 2.0
    assert abs(loss.item() - 2.0) < 1e-5, f"Expected 2.0, got {loss.item()}"


def test_weighted_mse_loss_device_safe():
    """所有输出均有1.0误差时: 2.0*1.0 + 0.5*1.0 + 0.5*1.0 = 3.0"""
    from train import weighted_mse_loss
    pred   = torch.tensor([[1.0, 1.0, 1.0]])
    target = torch.tensor([[0.0, 0.0, 0.0]])
    loss = weighted_mse_loss(pred, target)
    # 期望: 2.0*1.0 + 0.5*1.0 + 0.5*1.0 = 3.0
    assert abs(loss.item() - 3.0) < 1e-5, f"Expected 3.0, got {loss.item()}"


# ──────────────────────────────────────────────
# Test: EarlyStopper 行为
# ──────────────────────────────────────────────

def test_early_stopper_triggers():
    """EarlyStopper在patience次无改善后返回True"""
    from train import EarlyStopper
    from model import SimpViT

    stopper = EarlyStopper(patience=3)
    model = SimpViT(num_outputs=3)
    model.eval()

    # 第1步: val_loss=1.0 → 改善，should_stop=False, counter=0
    should_stop = stopper.step(1.0, model)
    assert not should_stop, "第1步不应停止"
    assert stopper.best_loss == 1.0
    assert stopper.counter == 0
    assert stopper.best_state is not None, "改善时应保存best_state"

    # 第2步: val_loss=1.1 → 无改善, counter=1
    should_stop = stopper.step(1.1, model)
    assert not should_stop, "第2步(counter=1)不应停止"
    assert stopper.counter == 1

    # 第3步: val_loss=1.2 → 无改善, counter=2
    should_stop = stopper.step(1.2, model)
    assert not should_stop, "第3步(counter=2)不应停止"
    assert stopper.counter == 2

    # 第4步: val_loss=1.3 → 无改善, counter=3 >= patience=3 → 停止
    should_stop = stopper.step(1.3, model)
    assert should_stop, "第4步(counter=3 >= patience=3)应停止"


# ──────────────────────────────────────────────
# Test: Checkpoint save/load round-trip
# ──────────────────────────────────────────────

def test_checkpoint_roundtrip(tmp_path):
    """save_checkpoint + load_checkpoint保留模型权重"""
    from train import save_checkpoint, load_checkpoint
    from model import SimpViT

    model = SimpViT(num_outputs=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    ckpt_path = str(tmp_path / "test_checkpoint.pth")
    save_checkpoint(model, optimizer, epoch=5, val_loss=0.123, path=ckpt_path)

    # 修改模型权重
    for p in model.parameters():
        p.data.fill_(0.0)

    # 加载并验证权重恢复
    model2 = SimpViT(num_outputs=3)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=3e-4)
    start_epoch, loaded_val_loss = load_checkpoint(model2, optimizer2, ckpt_path, device=torch.device('cpu'))

    assert start_epoch == 6, f"Expected epoch 6 (5+1), got {start_epoch}"
    assert abs(loaded_val_loss - 0.123) < 1e-6, f"val_loss mismatch: {loaded_val_loss}"


# ──────────────────────────────────────────────
# Test: Debug training loop (slow, uses actual data)
# ──────────────────────────────────────────────

@pytest.mark.slow
def test_debug_training_loop():
    """在38个真实样本上运行2个epoch, 验证管道端对端无误"""
    import os
    from train import train_one_epoch
    from model import SimpViT
    from dataset import CombinedHDF5Dataset
    from utils.split import build_stratified_indices
    from torch.utils.data import DataLoader

    # 使用主仓库的data目录（绝对路径，Windows兼容）
    data_dir = r'C:\CodingCraft\DL\ViT-Ctr\data'
    h5_paths = [
        os.path.join(data_dir, 'dithioester.h5'),
        os.path.join(data_dir, 'trithiocarbonate.h5'),
        os.path.join(data_dir, 'xanthate.h5'),
        os.path.join(data_dir, 'dithiocarbamate.h5'),
    ]

    # 验证数据文件存在
    for p in h5_paths:
        if not os.path.exists(p):
            pytest.skip(f"数据文件不存在: {p}")

    device = torch.device('cpu')
    train_idx, val_idx, test_idx = build_stratified_indices(h5_paths, seed=42)
    assert len(train_idx) > 0, "train_idx不能为空"

    train_ds = CombinedHDF5Dataset(h5_paths, train_idx)
    loader = DataLoader(train_ds, batch_size=4, num_workers=0, shuffle=True)

    model = SimpViT(num_outputs=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # 运行2个epoch
    for epoch in range(2):
        avg_loss, per_output = train_one_epoch(model, loader, optimizer, device)
        assert isinstance(avg_loss, float), f"avg_loss应为float, 得到 {type(avg_loss)}"
        assert torch.isfinite(torch.tensor(avg_loss)), f"Epoch {epoch}: 损失为NaN/Inf: {avg_loss}"
        assert 'ctr' in per_output, "per_output应包含'ctr'键"
        assert 'inh' in per_output, "per_output应包含'inh'键"
        assert 'ret' in per_output, "per_output应包含'ret'键"
