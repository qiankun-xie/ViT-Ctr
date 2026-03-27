# 训练主脚本 — SimpViT三参数输出，加权MSE损失，EarlyStopper，周期性checkpoint
import copy
import os
import argparse
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import SimpViT
from dataset import CombinedHDF5Dataset
from utils.split import build_stratified_indices, RAFT_TYPES


# ──────────────────────────────────────────────
# 损失函数 (D-06/D-08 锁定)
# ──────────────────────────────────────────────

def weighted_mse_loss(pred, target, weights=(2.0, 0.5, 0.5)):
    """
    加权MSE损失函数。
    pred, target: (B, 3) — [log10_Ctr, inhibition, retardation]
    weights: (w_ctr, w_inh, w_ret) = (2.0, 0.5, 0.5) (D-06/D-08锁定)

    注意: weight tensor必须在函数内部创建，保证device安全性（Pitfall 4）。
    """
    w = torch.tensor(weights, dtype=pred.dtype, device=pred.device)
    sq_err = (pred - target) ** 2        # (B, 3)
    per_output_mse = sq_err.mean(dim=0)  # (3,) — 批次内平均
    return (w * per_output_mse).sum()    # 标量


# ──────────────────────────────────────────────
# EarlyStopper (D-04 锁定: patience=15)
# ──────────────────────────────────────────────

class EarlyStopper:
    """
    监控验证集loss，patience次无改善时触发早停。
    同时保存当前最佳模型权重（deep copy）。
    """
    def __init__(self, patience=15, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_state = None

    def step(self, val_loss, model):
        """
        更新状态。
        返回 True 表示应停止训练（counter >= patience）。
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
        return self.counter >= self.patience  # True = 停止训练


# ──────────────────────────────────────────────
# Checkpoint 工具函数
# ──────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch, val_loss, path):
    """保存训练checkpoint（模型权重 + 优化器状态 + epoch + val_loss）。"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, path)


def load_checkpoint(model, optimizer, path, device):
    """
    恢复训练checkpoint。
    返回 (next_epoch, val_loss)，next_epoch = saved_epoch + 1。
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return ckpt['epoch'] + 1, ckpt['val_loss']


# ──────────────────────────────────────────────
# 单epoch训练
# ──────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device):
    """
    执行一个epoch的训练，返回 (avg_total_loss, per_output_dict)。
    per_output_dict = {'ctr': float, 'inh': float, 'ret': float}

    NaN guard: 检测到NaN loss时打印警告并跳过该batch（Pitfall 5）。
    """
    model.train()
    total_loss = 0.0
    sum_ctr = 0.0
    sum_inh = 0.0
    sum_ret = 0.0
    n_steps = 0

    for step, (fp, lbl) in enumerate(loader):
        fp  = fp.to(device)
        lbl = lbl.to(device)

        optimizer.zero_grad()
        pred = model(fp)                         # (B, 3)
        loss = weighted_mse_loss(pred, lbl)

        # NaN guard (Pitfall 5)
        if torch.isnan(loss):
            print(f"[WARN] NaN loss at step {step}, skipping")
            continue

        loss.backward()
        optimizer.step()

        # 记录各输出分量（不带权重，仅MSE，用于诊断Pitfall 3）
        with torch.no_grad():
            sq_err = (pred - lbl) ** 2   # (B, 3)
            sum_ctr += sq_err[:, 0].mean().item()
            sum_inh += sq_err[:, 1].mean().item()
            sum_ret += sq_err[:, 2].mean().item()

        total_loss += loss.item()
        n_steps += 1

    if n_steps == 0:
        return float('nan'), {'ctr': float('nan'), 'inh': float('nan'), 'ret': float('nan')}

    avg_total_loss = total_loss / n_steps
    per_output = {
        'ctr': sum_ctr / n_steps,
        'inh': sum_inh / n_steps,
        'ret': sum_ret / n_steps,
    }
    return avg_total_loss, per_output


# ──────────────────────────────────────────────
# 验证集评估
# ──────────────────────────────────────────────

def validate(model, loader, device):
    """
    在验证集上计算加权MSE loss，返回平均值。
    model.eval() + torch.no_grad() 确保无梯度计算。
    """
    model.eval()
    total_loss = 0.0
    n_steps = 0

    with torch.no_grad():
        for fp, lbl in loader:
            fp  = fp.to(device)
            lbl = lbl.to(device)
            pred = model(fp)
            loss = weighted_mse_loss(pred, lbl)
            if not torch.isnan(loss):
                total_loss += loss.item()
                n_steps += 1

    return total_loss / n_steps if n_steps > 0 else float('nan')


# ──────────────────────────────────────────────
# 主训练入口
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ViT-Ctr SimpViT训练脚本")
    parser.add_argument('--h5_dir', type=str, default='data', help='HDF5数据目录路径')
    parser.add_argument('--epochs', type=int, default=200, help='最大训练epoch数 (D-05)')
    parser.add_argument('--batch_size', type=int, default=64, help='批大小 (D-02: 锁定64)')
    parser.add_argument('--lr', type=float, default=3e-4, help='初始学习率 (D-03: 锁定3e-4)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='checkpoint保存目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader工作进程数(Windows本地建议0)')
    parser.add_argument('--debug', action='store_true', help='调试模式（38样本）')
    args = parser.parse_args()

    # ── 设备 ──
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[train] Device: {device}")

    # ── 数据路径 ──
    h5_dir = args.h5_dir
    H5_PATHS = [
        os.path.join(h5_dir, 'dithioester.h5'),
        os.path.join(h5_dir, 'trithiocarbonate.h5'),
        os.path.join(h5_dir, 'xanthate.h5'),
        os.path.join(h5_dir, 'dithiocarbamate.h5'),
    ]
    for p in H5_PATHS:
        if not os.path.exists(p):
            raise FileNotFoundError(f"HDF5文件不存在: {p}")

    # ── 数据划分 (D-09/D-10/D-11) ──
    torch.manual_seed(args.seed)
    train_idx, val_idx, test_idx = build_stratified_indices(H5_PATHS, seed=args.seed)
    print(f"[train] Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    if args.debug:
        print("[train] 调试模式: 使用全部38个样本（训练结果无科学意义）")

    # ── Dataset & DataLoader ──
    train_ds = CombinedHDF5Dataset(H5_PATHS, train_idx)
    val_ds   = CombinedHDF5Dataset(H5_PATHS, val_idx)

    pin_memory = (device.type == 'cuda')
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory
    )

    # ── 模型 ──
    model = SimpViT(num_outputs=3).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[train] SimpViT参数量: {total_params:,}")

    # ── 优化器 + 调度器 (D-03/D-04 锁定) ──
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    stopper = EarlyStopper(patience=15)

    # ── Checkpoint目录 ──
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(args.checkpoint_dir, 'best_model.pth')

    # ── 训练日志 ──
    training_log = []
    log_path = os.path.join(args.checkpoint_dir, 'training_log.json')

    # ── 训练循环 ──
    best_val_loss = float('inf')
    print(f"[train] 开始训练, max_epochs={args.epochs}")

    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        # 训练
        train_loss, per_output = train_one_epoch(model, train_loader, optimizer, device)

        # 验证
        val_loss = validate(model, val_loader, device)

        # 学习率调度 (ReduceLROnPlateau, D-03)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # 日志记录（每epoch）
        log_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'loss_ctr': per_output['ctr'],
            'loss_inh': per_output['inh'],
            'loss_ret': per_output['ret'],
            'lr': current_lr,
        }
        training_log.append(log_entry)

        # 保存training_log.json（每epoch追写）
        with open(log_path, 'w') as f:
            json.dump(training_log, f, indent=2)

        # 打印进度
        tqdm.write(
            f"Epoch {epoch:4d} | train={train_loss:.4f} val={val_loss:.4f} | "
            f"ctr={per_output['ctr']:.4f} inh={per_output['inh']:.6f} ret={per_output['ret']:.6f} | "
            f"lr={current_lr:.2e}"
        )

        # 最佳checkpoint (val loss改善时保存)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, best_ckpt_path)
            tqdm.write(f"  → 新的最佳val_loss={val_loss:.6f}, checkpoint已保存")

        # 周期性checkpoint (每5个epoch)
        if (epoch + 1) % 5 == 0:
            periodic_path = os.path.join(args.checkpoint_dir, f'epoch_{epoch:04d}.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, periodic_path)

        # EarlyStopper (D-04: patience=15)
        should_stop = stopper.step(val_loss, model)
        if should_stop:
            print(f"\n[train] 早停触发 (patience={stopper.patience}), epoch={epoch}")
            break

    # ── 训练结束: 加载最佳权重 ──
    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)
        print(f"[train] 已恢复最佳模型权重 (best_val_loss={stopper.best_loss:.6f})")
    elif os.path.exists(best_ckpt_path):
        # fallback: 从文件加载
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"[train] 从文件恢复最佳模型 (epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f})")

    print(f"[train] 训练完成. 最终最佳val_loss={stopper.best_loss:.6f}")
    print(f"[train] 训练日志已保存: {log_path}")


if __name__ == '__main__':
    main()
