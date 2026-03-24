# ViT-Ctr: 基于Vision Transformer的RAFT链转移常数预测系统

## What This Is

基于简化Vision Transformer（SimpViT）架构，从RAFT聚合实验动力学数据中同时提取链转移常数（Ctr）、诱导期（inhibition period）和减速因子（retardation factor）三个关键参数的深度学习系统。模仿ViT-RR（Angew. Chem. Int. Ed. 2025, DOI: 10.1002/anie.202513086）的范式，将实验数据编码为链转移指纹（ctFP），覆盖所有类型的RAFT剂体系（dithioester、trithiocarbonate、xanthate、dithiocarbamate等），并提供Web应用供研究者在线使用。

## Core Value

一次输入实验数据，同时提取Ctr、诱导期和减速因子三个参数——传统方法需要三组独立实验才能分别获得。

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] RAFT动力学ODE模型，支持所有RAFT剂类型的正向模拟
- [ ] ctFP指纹编码：64×64双通道图像（Mn通道 + Đ通道），x轴=[CTA]/[M]，y轴=conversion
- [ ] 百万级合成训练数据集生成（遍历Ctr、kadd/kfrag、[CTA]/[M]等参数空间）
- [ ] SimpViT模型训练，输出三参数：log10(Ctr)、inhibition period、retardation factor
- [ ] Bootstrap不确定性估计（200次迭代 + F分布置信区间）
- [ ] 文献实验数据验证（与已发表的Ctr值对比）
- [ ] Streamlit Web应用部署（支持手动输入和Excel上传）
- [ ] 英文学术论文撰写
- [ ] Supporting Information撰写（动力学推导、数据集构建、模型细节、验证结果）
- [ ] 路线A探索性研究：分子结构→Ctr预测（作为论文亮点/展望）

### Out of Scope

- 实时在线学习/模型更新 — 超出当前项目范围，属于工程优化
- 移动端应用 — Web应用足够，移动端无必要
- RAFT以外的CRP体系（ATRP、NMP等） — 聚焦RAFT，其他体系留待后续工作
- 完整的kadd/kfrag解耦预测 — 这些参数无法从常规聚合数据中可靠解耦

## Context

**参考项目：** ViT-RR（C:/CodingCraft/DL/ViT-RR），发表于Angew. Chem. Int. Ed. 2025，使用SimpViT从共聚实验数据中预测反应性比。本项目复用其核心范式（指纹编码 + SimpViT + Bootstrap），但针对RAFT聚合这一不同且更复杂的化学体系。

**关键差异：**
- ViT-RR预测反应性比（r1, r2），本项目预测Ctr + 诱导期 + 减速因子
- ViT-RR的数据生成用kMC模拟，本项目用RAFT动力学ODE数值积分
- RAFT涉及加成-断裂平衡（addition-fragmentation equilibrium），动力学比自由基共聚更复杂
- 本项目覆盖所有RAFT剂类型，需要处理不同RAFT剂的动力学差异

**硬件环境：** Intel i5-10210U + MX350 (2GB) + 16GB RAM（笔记本）。模型规模小（SimpViT ~3.4MB），本地可训练。如需加速可使用Google Colab免费GPU。

**论文叙事核心：** 传统方法测Ctr用Mayo方程、测诱导期用conversion-time曲线、测retardation用速率对比——三个独立实验流程。本模型一次输入三个输出，这是真正的效率提升。

## Constraints

- **技术栈**: PyTorch + Streamlit，与ViT-RR保持一致
- **模型架构**: SimpViT（64×64输入, patch_size=16, hidden=64, 2层Transformer, 4头注意力），输出维度改为3
- **数据格式**: 双通道ctFP（Ch1=Mn, Ch2=Đ），坐标轴（x=[CTA]/[M], y=conversion）
- **硬件**: 本地开发调试，大规模训练可选Colab
- **语言**: 项目文档和代码注释用中文，论文和SI用英文

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| 路线B为主（实验数据→参数）+ 路线A为辅（结构→Ctr） | B落地确定性高（模拟数据可生成），A创新性强但数据量不足以支撑深度学习 | — Pending |
| 三参数同时预测（Ctr + inhibition + retardation） | 单预测Ctr叙事太薄，三参数对应三组传统独立实验，差异化明确 | — Pending |
| 双通道ctFP编码（Mn + Đ） | 包含提取三个参数的充分信息，与ViT-RR的双通道rFP直接对应 | — Pending |
| 覆盖所有RAFT剂类型 | 扩大适用范围，增强论文说服力 | — Pending |
| ODE数值积分生成数据（非kMC） | RAFT动力学需要追踪Mn和Đ，ODE比kMC更直接且高效 | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd:transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-03-24 after initialization*
