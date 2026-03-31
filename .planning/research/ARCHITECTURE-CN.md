# 架构模式

**项目：** ViT-Ctr — RAFT链转移常数预测系统
**调研时间：** 2026-03-24
**整体置信度：** 高（基于ViT-RR参考代码库的直接检查 + 已验证的ODE文献）

---

## 推荐架构

系统遵循与ViT-RR相同的三层架构，针对RAFT化学进行了调整。三层足够独立，可以分别构建和测试，使分阶段开发变得简单。

```
┌─────────────────────────────────────────────────────────────────┐
│  层1：数据生成                                                    │
│                                                                 │
│  raft_ode.py         param_sampler.py      dataset_builder.py  │
│  ┌─────────────┐     ┌──────────────┐      ┌─────────────────┐ │
│  │ ODE系统     │────▶│ 参数空间     │─────▶│ ctFP编码器      │ │
│  │ (Mn, Đ vs   │     │ (Ctr, kadd,  │      │ 64×64×2 图像    │ │
│  │  conversion)│     │  kfrag, CTA) │      │ + 标签 (3)      │ │
│  └─────────────┘     └──────────────┘      └─────────────────┘ │
│                                                    │            │
└────────────────────────────────────────────────────┼────────────┘
                                                     │
                                                     ▼ .pt数据集
┌─────────────────────────────────────────────────────────────────┐
│  层2：模型训练与评估                                              │
│                                                                 │
│  model.py            train.py              evaluate.py          │
│  ┌─────────────┐     ┌──────────────┐      ┌─────────────────┐ │
│  │ SimpViT     │◀────│ DataLoader   │      │ Bootstrap UQ    │ │
│  │ 2通道输入   │     │ train/val/   │      │ (200次迭代)     │ │
│  │ 3个输出     │────▶│ test划分     │─────▶│ F分布CI         │ │
│  └─────────────┘     └──────────────┘      └─────────────────┘ │
│                                                    │            │
└────────────────────────────────────────────────────┼────────────┘
                                                     │
                                                     ▼ .pth权重
┌─────────────────────────────────────────────────────────────────┐
│  层3：Web应用                                                     │
│                                                                 │
│  deploy.py (Streamlit)                                          │
│  ┌─────────────┐     ┌──────────────┐      ┌─────────────────┐ │
│  │ 数据输入    │────▶│ ctFP编码器   │─────▶│ 模型推理        │ │
│  │ (手动 /     │     │ (与层1相同)  │      │ + Bootstrap UQ  │ │
│  │  Excel)     │     │              │      │ → Ctr ± CI      │ │
│  └─────────────┘     └──────────────┘      └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

**关键原则：**
1. **层1和层3共享相同的ctFP编码器** — 训练-推理不一致会静默破坏预测
2. **层2独立于层1** — 可以用调试数据集测试训练管线，无需完整生成
3. **层3独立于层2** — 可以用预训练权重测试Web应用，无需重新训练

---

## 组件边界

| 组件 | 文件 | 职责 | 通信对象 |
|------|------|------|---------|
| ODE求解器 | `raft_ode.py` | 给定动力学参数，积分RAFT矩方程（Mn, Đ vs. conversion） | param_sampler, dataset_builder |
| 参数采样器 | `param_sampler.py` | 采样参数空间：log10(Ctr)、kadd/kfrag、[CTA]/[M]、引发剂负载、RAFT类型 | raft_ode, dataset_builder |
| ctFP编码器 | `ctfp_encoder.py` | 转换(conversion, Mn, Đ)轨迹 → 64×64×2图像张量；dataset_builder和deploy.py共享 | dataset_builder, deploy.py |
| 数据集构建器 | `dataset_builder.py` | 编排并行ODE运行，收集ctFP + 标签，保存到.pt文件 | raft_ode, param_sampler, ctfp_encoder |
| SimpViT模型 | `model.py` | 2通道64×64基于patch的Transformer；输出维度=3（log10(Ctr), inhibition, retardation） | train.py, deploy.py |
| 训练脚本 | `train.py` | DataLoader、损失、优化器、检查点保存 | model.py, dataset |
| 评估模块 | `evaluate.py` | 测试集指标、parity图、文献验证、bootstrap CI计算 | model.py, dataset |
| Web应用 | `deploy.py` | Streamlit UI：数据输入、ctFP编码、模型推理、bootstrap UQ显示 | ctfp_encoder.py, model.py |
| 论文/SI | `paper/`, `si/` | LaTeX/Word文档；图表由evaluate.py生成 | evaluate.py输出 |

### 关键边界规则
`ctfp_encoder.py`必须是独立的可导入模块，无Streamlit或训练依赖。数据集构建器和Web应用以相同方式调用它。这是关键的共享接口 — 编码逻辑的任何更改必须传播到两条路径。

---

## 数据流

```
RAFT动力学参数
(Ctr, kadd, kfrag, [CTA]/[M], kp, kt, f, kd)
         │
         ▼
   scipy.integrate.solve_ivp (Radau, 刚性)
   矩方程 → Mn(α), Đ(α) 轨迹
         │
         ▼
   ctfp_encoder.transform()
   x轴: [CTA]/[M]  (0→1 → 像素 0→63)
   y轴: conversion  (0→1 → 像素 0→63)
   Ch0: 每个(x, y)点的Mn值
   Ch1: 每个(x, y)点的Đ值
   输出: torch.Tensor shape (2, 64, 64)
         │
         ▼
   SimpViT.forward(x)
   patch_embedding (Conv2d 2→64, kernel=16)
   → 16个patches → 位置编码
   → 2× TransformerEncoderLayer (d=64, heads=4)
   → patches上的均值池化 (无CLS token)
   → fc Linear(64, 3)
   输出: [log10(Ctr), inhibition_period, retardation_factor]
         │
         ▼
   Bootstrap (200次输入噪声迭代)
   → 协方差矩阵 → F分布95% JCI
   输出: 3个参数各自的均值 ± 半宽
```

### 逆变换（训练标签）
标签存储为物理值的log10（匹配ViT-RR约定）。推理期间，输出被指数化：`10^pred → 物理值`。这种压缩使回归目标保持在良好条件的范围内。

---

## 遵循的模式

### 模式1：共享编码器（从ViT-RR代码库验证）
**内容：** 将原始实验数据转换为图像张量的`transform()`函数定义一次，由数据集构建器和Streamlit应用导入。
**时机：** 始终 — 训练数据和推理之间的编码一致性不可协商。
**示例（ctfp_encoder.py）：**
```python
import numpy as np, math, torch

def transform(data, img_size=64):
    """
    data: (cta_ratio, conversion, Mn, D)行的列表
    返回: torch.Tensor shape (2, img_size, img_size)
    Ch0 = Mn, Ch1 = D (分散度)
    x轴 = [CTA]/[M]分数, y轴 = conversion
    """
    img = np.zeros((2, img_size, img_size))
    for cta_ratio, conv, mn, dispersity in data:
        col = min(math.floor(cta_ratio * img_size), img_size - 1)
        row = min(math.floor(conv * img_size), img_size - 1)
        img[0][row][col] = mn        # 通道0: Mn
        img[1][row][col] = dispersity  # 通道1: D
    return torch.tensor(img, dtype=torch.float32)
```

### 模式2：Mn和Đ的ODE矩方程
**内容：** 使用scipy `solve_ivp`配合`method='Radau'`（隐式，处理刚性RAFT系统）和矩方法计算链长分布的零阶/一阶/二阶矩。RAFT动力学由于快速RAFT平衡vs.慢速增长产生刚性ODE。
**时机：** 始终在数据生成层。训练集永远不要通过其他方式重新推导Mn/Đ。
**跟踪的关键参数：** [M]、[CTA]、[I]，以及矩λ0、λ1、λ2（死链）+ μ0、μ1、μ2（活链）。在等转化率间隔（非时间间隔）采样点，以获得一致的指纹密度。

### 模式3：通过输入扰动的Bootstrap不确定性
**内容：** 推理时，在输入的噪声扰动版本上运行模型200次（每个值3%的高斯噪声）。收集预测，计算协方差，应用95%的F分布CI。完全匹配ViT-RR — 从`deploy.py`的`predict_model()`验证。
**时机：** 始终在Web应用中。开发迭代期间可选跳过（复选框切换）。

### 模式4：Log10标签空间
**内容：** 在训练标签中将Ctr存储为log10(Ctr)。模型预测log10空间；转换为线性以供显示。
**时机：** 始终 — Ctr在RAFT剂类型间跨越几个数量级。对原始Ctr值的线性回归会产生差的梯度。

### 模式5：按RAFT剂类型的参数空间覆盖
**内容：** 按RAFT剂类别（dithioester、trithiocarbonate、xanthate、dithiocarbamate）分层合成数据生成。每个类别有特征性的Ctr范围。在每个类别的物理范围内均匀采样以避免不平衡训练。
**时机：** 在dataset_builder设计期间。未能分层会导致在最常见类别之外的预测不佳。

---

## 要避免的反模式

### 反模式1：应用与训练中的独立编码逻辑
**内容：** 在`deploy.py`中独立于训练时编码器重新实现ctFP变换。
**为何不好：** 训练和推理编码之间的任何细微差异（轴顺序、归一化、边缘处理）都会使预测失效。这是静默失败 — 模型会产生看似合理但错误的数字。
**替代方案：** 在所有地方从单一共享的`ctfp_encoder.py`模块导入。

### 反模式2：在等时间间隔采样
**内容：** 将ODE运行到固定时间点并将其用作指纹条目。
**为何不好：** 早期时间点聚集在低转化率；指纹图像在高转化率时几乎为空。ctFP通过(CTA比率, conversion)位置编码数据密度 — 均匀时间步长浪费像素分辨率。
**替代方案：** 使用`solve_ivp`的`dense_output=True`将ODE解插值到固定的转化率网格（例如0.02, 0.04, ..., 0.98）。

### 反模式3：直接在物理Mn值上训练
**内容：** 使用原始Mn（单位g/mol，典型范围1,000–200,000）作为通道值。
**为何不好：** 极端值范围主导像素强度。模型学习大Mn系统并忽略小系统。
**替代方案：** 通过完全转化时的理论Mn（Mn_theory = [M]0 / [CTA]0 × MW_monomer）归一化Mn，得到对于良好控制的RAFT接近1的无量纲比率。

### 反模式4：单体deploy.py
**内容：** 所有逻辑（编码、模型加载、bootstrap、UI）在一个文件中，如ViT-RR的deploy.py。
**为何在ViT-RR中可接受（2个模型，2种化学类型）但在此处有问题：** ViT-Ctr添加了RAFT类型特定的参数范围和潜在的路线A（结构→Ctr）路径。单体文件变得难以测试。
**替代方案：** 从一开始就提取`ctfp_encoder.py`和`model.py`作为可导入模块，即使应用仍是单个Streamlit脚本。

### 反模式5：在同一模型中混合路线A和路线B
**内容：** 训练单个SimpViT，同时接受ctFP（路线B输入）和分子结构描述符（路线A输入）。
**为何不好：** 两条路线有完全不同的数据需求、架构和验证策略。耦合它们会延迟两者。
**替代方案：** 首先构建路线B作为主要交付物。路线A是在最终探索阶段添加的独立、单独的模型。

---

## 组件构建顺序（阶段依赖）

依赖图决定构建顺序。每个组件仅依赖于其上方的组件。

```
阶段1：ODE基础
  └── raft_ode.py（独立，无ML依赖）
  └── param_sampler.py（无ML依赖）
  └── ctfp_encoder.py（无ML，无ODE — 纯图像编码）

阶段2：数据集生成
  └── dataset_builder.py → 依赖raft_ode + param_sampler + ctfp_encoder
  └── 产出：dataset.pt（ctFP + 标签）

阶段3：模型训练
  └── model.py（SimpViT，output_dim=3）→ 独立
  └── train.py → 依赖model + dataset.pt

阶段4：评估与验证
  └── evaluate.py → 依赖model.py + 训练的.pth + 文献数据
  └── 产出：图表、指标、parity图（论文输入）

阶段5：Web部署
  └── deploy.py（Streamlit）→ 依赖ctfp_encoder + model.py + .pth
  └── 不依赖dataset.pt或train.py

阶段6：论文与SI撰写
  └── 依赖所有evaluate.py输出
  └── 路线A探索与论文撰写并行运行
```

**关键路径：** ODE正确性（阶段1）直接决定数据集质量（阶段2），进而决定模型质量（阶段3）。阶段4发现的错误可能需要重新生成数据集。在大规模数据生成前为一次ODE验证迭代预留时间。

---

## 可扩展性考虑

| 关注点 | 本地（i5-10210U + MX350） | Colab免费GPU |
|--------|--------------------------|--------------|
| ODE模拟速度 | 单核约1,000参数集/分钟；使用multiprocessing.Pool | ODE不需要 |
| 数据集大小 | 1M样本 × 64×64×2 float32 = 约32 GB未压缩；使用分块.pt文件或HDF5 | 通过Google Drive挂载加载 |
| 模型训练 | SimpViT约3.4MB；1M样本在CPU上慢。训练使用Colab | 每次训练约1-2小时 |
| 推理（部署） | CPU推理，每次预测<100ms + bootstrap增加约5秒（200次迭代） | N/A — Streamlit Community Cloud（CPU） |
| 训练期间内存 | MX350 2GB显存上批大小256：对64×64×2 float32图像可行 | Colab T4上批大小512+ |

**推荐的数据集存储策略：** 以10万样本为单位生成，存储为`dataset_chunk_000.pt`到`dataset_chunk_009.pt`。训练期间，使用`ConcatDataset`或流式处理避免一次性将全部32GB加载到RAM。

---

## 来源

- ViT-RR参考实现：`C:/CodingCraft/DL/ViT-RR/model_utils.py`、`deploy.py`（直接检查 — 高置信度）
- RAFT ODE矩方程和刚性：[Macromolecules 2023, Dispersity model for in silico RAFT](https://pubs.acs.org/doi/10.1021/acs.macromol.2c01798)、[Polymers 2022, Method of Partial Moments](https://www.mdpi.com/2073-4360/14/22/5013) — 高置信度
- 神经网络的Bootstrap不确定性：[npj Computational Materials — Calibration after bootstrap](https://www.nature.com/articles/s41524-022-00794-8)、[arXiv Bootstrapped Deep Ensembles](https://arxiv.org/pdf/2202.10903) — 高置信度
- ViT-RR论文（Angew. Chem. 2025）：[Copolymer Sequence Regulation Enabled by Reactivity Ratio Fingerprints](https://onlinelibrary.wiley.com/doi/abs/10.1002/anie.202513086) — 中等置信度（仅摘要；从代码验证完整架构）
- 刚性化学动力学的SciPy solve_ivp：[SciPy文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html) — 高置信度
