"""
Chain Transfer Fingerprint (ctFP) 编码器。
将RAFT聚合动力学数据编码为图像张量的共享模块。
训练管线和Web应用共用同一编码函数（ENC-02）。

坐标轴映射（ENC-01）:
  - x轴（列）: [CTA]/[M] 归一化到 [0, 1]
  - y轴（行）: 单体转化率 conversion in [0, 1]
  - Channel 0: Mn / Mn_theory（无量纲，典型值 0-2）
  - Channel 1: 分散度 Mw/Mn（无量纲，>= 1.0，截断于 4.0）
"""
import numpy as np
import math
import torch


def transform(data, img_size=64):
    """
    将RAFT动力学数据编码为链转移指纹（ctFP）。

    Args:
        data: (cta_ratio_norm, conversion, mn_norm, dispersity) 元组的可迭代对象
              cta_ratio_norm: [CTA]/[M] 归一化到 [0, 1]（训练数据除以0.1）
              conversion: 单体转化率 in [0, 1]
              mn_norm: Mn / Mn_theory（无量纲）; Mn_theory = M0/CTA0 * M_monomer
              dispersity: Mw/Mn >= 1.0（截断于 4.0 防止异常值）
        img_size: 指纹分辨率（默认 64）

    Returns:
        torch.Tensor 形状 (2, img_size, img_size), dtype float32
    """
    img = np.zeros((2, img_size, img_size), dtype=np.float32)
    for cta_norm, conv, mn_n, disp in data:
        # [CTA]/[M] 映射到列索引
        col = min(int(math.floor(cta_norm * img_size)), img_size - 1)
        # conversion 映射到行索引
        row = min(int(math.floor(conv * img_size)), img_size - 1)
        # 负值安全截断
        col = max(col, 0)
        row = max(row, 0)
        # Channel 0: Mn归一化值, Channel 1: 分散度（截断于4.0）
        img[0, row, col] = mn_n
        img[1, row, col] = min(disp, 4.0)
    return torch.tensor(img, dtype=torch.float32)
