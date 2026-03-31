# 技术栈

**项目：** ViT-Ctr — 基于Vision Transformer的RAFT链转移常数预测系统
**调研时间：** 2026-03-24
**整体置信度：** 高（技术栈基于参考项目ViT-RR + 已验证的最新版本）

---

## 推荐技术栈

### 核心深度学习

| 技术 | 版本 | 用途 | 理由 |
|------|------|------|------|
| Python | 3.11 | 运行时环境 | 稳定，所有依赖支持良好；3.12也可用但3.11在Windows上PyTorch预编译包覆盖更广 |
| PyTorch | 2.6.x–2.10.x | 模型定义、训练、推理 | 参考项目ViT-RR已验证；`nn.TransformerEncoderLayer`是SimpViT的骨干；最新稳定版2.10.0（2026年1月） |
| torchvision | 匹配PyTorch | 图像变换、张量工具 | PyTorch的必需依赖；虽不需要单独的CNN但提供一致的变换API |

**置信度：高** — PyTorch 2.6+是经PyPI验证的当前稳定版。参考项目已在此技术栈上运行。

**硬件说明：** 项目运行在Intel i5-10210U + MX350（2 GB显存）上。SimpViT约3.4 MB — 本地CPU训练可用于测试。百万样本数据集生成使用Google Colab免费GPU（T4）。不要添加PyTorch Lightning或其他封装；SimpViT太小无法受益，且参考项目使用原生PyTorch。

---

### 数据生成（ODE模拟）

| 技术 | 版本 | 用途 | 理由 |
|------|------|------|------|
| SciPy | 1.15.x | RAFT ODE积分的`solve_ivp` | Python标准ODE求解器；`method='Radau'`或`method='BDF'`处理刚性RAFT动力学（自由基聚合中的宽时间尺度范围）；当前稳定版1.15.0（2025年1月） |
| NumPy | 2.2.x | 数组操作、指纹构建 | SciPy和PyTorch都需要；NumPy 2.x是当前版本；最新为2.2.2（2025年1月） |
| joblib | 1.4.x | 跨参数网格的CPU并行ODE生成 | `Parallel(n_jobs=-1)(delayed(simulate)(params) for params in grid)`模式；在多核CPU上生成百万样本必不可少，无需自定义多进程样板代码 |
| tqdm | 4.x | 生成和训练期间的进度跟踪 | `tqdm.contrib.concurrent.process_map`集成tqdm与`ProcessPoolExecutor`；开销极小（60 ns/迭代） |

**置信度：高**（SciPy/NumPy/joblib）。中等（具体版本固定） — 安装时通过`pip install --upgrade`验证。

**为什么用`solve_ivp`而非`odeint`：** RAFT动力学是刚性的（快速自由基反应耦合慢速聚合物生长）。`solve_ivp`配合`method='Radau'`是现代API，暴露刚性求解器并支持密集输出和事件检测。`odeint`是遗留API，仅封装LSODA。使用`solve_ivp(fun, t_span, y0, method='Radau', dense_output=True)`。

**为什么不用polykin：** PolyKin（HugoMVale/polykin）是针对聚合工程计算的早期库。它没有暴露符合ViT-Ctr参数化需求的专用RAFT ODE系统（扫描Ctr、kadd/kfrag、[CTA]/[M]、引发剂浓度）。从第一性原理用`solve_ivp`编写RAFT ODE提供完全控制，是文献中的标准方法（包括ViT-RR参考使用kMC）。

---

### 指纹编码和数据管线

| 技术 | 版本 | 用途 | 理由 |
|------|------|------|------|
| NumPy | 2.2.x | ctFP构建（64x64双通道数组） | 直接移植ViT-RR model_utils.py的`transform()`；散点图到图像编码约10行NumPy |
| pandas | 2.2.x | 参数网格构建、结果DataFrame、Excel I/O | `pd.read_excel(..., engine='openpyxl')`是标准模式；当前稳定版3.0.x但2.2.x广泛部署且稳定 |
| openpyxl | 3.1.x | pandas的xlsx读写后端 | pandas 2.x+中`.xlsx`的默认引擎；Streamlit Excel上传功能必需 |
| h5py 或 numpy .npz | — | 序列化百万样本指纹数据集 | HDF5（h5py）用于训练期间的分块访问；如果数据集适合RAM则`.npz`可用（~64x64x2xfloat32 x 1M = ~32 GB — 需要HDF5）；生成时选择 |

**置信度：高**（pandas+openpyxl模式，已在ViT-RR deploy.py中验证）。中等（h5py推荐） — 取决于最终数据集大小。

**pandas版本说明：** pandas 3.0.x是当前版本但引入了写时复制变更，可能破坏为2.x编写的代码。在requirements.txt中使用`pandas>=2.2,<3.0`避免开发期间意外破坏，测试后再升级。

---

### 训练基础设施

| 技术 | 版本 | 用途 | 理由 |
|------|------|------|------|
| PyTorch | 2.6+ | `DataLoader`、`Dataset`、优化器、损失函数 | 一体化；无需单独的数据集库 |
| torch.utils.data | （PyTorch的一部分） | 封装ctFP数组的自定义Dataset | 如果数据集适合RAM使用`TensorDataset`；否则自定义`Dataset` + HDF5 |
| `torch.optim.Adam` | （PyTorch的一部分） | 优化器 | Adam配合lr=3e-4是小型Transformer的标准；ViT-RR使用此模式 |
| MSELoss | （PyTorch的一部分） | log10(Ctr)、inhibition、retardation的回归损失 | 三输出回归；训练前对Ctr进行log10变换防止尺度不平衡 |
| `torch.compile` | （PyTorch 2.x的一部分） | 可选：CPU上10–30%加速 | 作为最终优化步骤添加；初始开发不需要 |

**置信度：高** — 直接来自ViT-RR架构。

---

### 不确定性估计

| 技术 | 版本 | 用途 | 理由 |
|------|------|------|------|
| NumPy | 2.2.x | Bootstrap循环、协方差矩阵 | 按ViT-RR deploy.py的`np.cov(predictions, rowvar=False)` |
| `scipy.stats.f` | （SciPy 1.15.x的一部分） | F分布95% JCI | ViT-RR的精确模式：`f.ppf(0.95, dfn=p, dfd=n-p)` → 联合置信区间半宽 |

**置信度：高** — 直接复制自经同行评审的ViT-RR。

---

### Web部署

| 技术 | 版本 | 用途 | 理由 |
|------|------|------|------|
| Streamlit | 1.41+（当前：1.52.x） | Web应用框架 | PROJECT.md的直接约束；匹配ViT-RR deploy.py；此用例无需JS/HTML |
| gspread | 6.x | Google Sheets用户注册 | ViT-RR中用于跟踪机构用户；研究工具的轻量级数据库替代方案 |
| oauth2client | 4.x | Google Sheets认证 | ViT-RR中使用；注意：`oauth2client`技术上已弃用，推荐`google-auth`，但仍可用且ViT-RR模式稳定 |

**置信度：高**（Streamlit）。中等（gspread/oauth2client） — 如果放弃用户跟踪（可选功能），可完全省略这些。

**Streamlit版本说明：** 当前稳定版1.52.x（2025年12月）。1.41.x系列仍可用；没有影响ViT-RR模式的破坏性变更。固定到`>=1.40`以保持与现有deploy.py模式兼容同时允许升级。

**部署目标：** Streamlit Community Cloud（免费层）用于研究。需要仓库根目录的`requirements.txt`和Google Sheets凭据的`secrets.toml`。模型权重提交到仓库（每个模型约3.4 MB，Git可接受）。

---

### 开发工具

| 技术 | 版本 | 用途 | 理由 |
|------|------|------|------|
| Jupyter Notebook / JupyterLab | 4.x | 探索性ODE测试、指纹可视化、模型原型 | 科学ML的标准；在提交到脚本前允许内联绘制ctFP图像和ODE曲线 |
| Matplotlib | 3.10.x | 指纹可视化、训练曲线、parity图 | 标准科学绘图；`imshow`用于ctFP检查；当前稳定版3.10.0（2024年12月） |
| Google Colab | — | 免费T4 GPU上的大规模训练 | 硬件约束：MX350不足以进行百万样本训练；Colab T4提供约16 GB显存；需要将数据集导出到Google Drive |

**置信度：高**（Matplotlib）。中等（Colab，免费层可用性不保证，但对此工作负载规模历史上可靠）。

---

## 考虑的替代方案

| 类别 | 推荐 | 替代方案 | 为何不用 |
|------|------|---------|---------|
| Web框架 | Streamlit | Gradio | Gradio同样有效但项目约束要求Streamlit；ViT-RR是Streamlit |
| Web框架 | Streamlit | Flask + React | 对研究工具过度工程化；额外数周工作无益处 |
| ODE求解器 | `scipy.integrate.solve_ivp` | Julia / DifferentialEquations.jl | Julia更快但破坏纯Python栈；此规模无必要 |
| ODE求解器 | `scipy.integrate.solve_ivp` | polykin库 | 太早期；未暴露所需的参数化RAFT ODE系统 |
| ODE并行 | joblib | multiprocessing标准库 | joblib更高级，更好处理序列化边缘情况，与tqdm集成 |
| ODE并行 | joblib | Dask | Dask增加集群开销，单机生成任务不合理 |
| 深度学习 | PyTorch | TensorFlow/Keras | 项目约束要求PyTorch；ViT-RR是PyTorch；无切换理由 |
| 深度学习 | PyTorch（原生） | PyTorch Lightning | Lightning对大型分布式训练有用；SimpViT训练循环约30行 — 开销不合理 |
| 数据集存储 | HDF5（h5py） | Zarr | Zarr现代但h5py教程更多且普遍支持；两者都可用 |
| 用户认证 | gspread（Google Sheets） | SQLite | SQLite更清晰但需要服务器端DB；Google Sheets适用于低流量研究应用 |

---

## 安装

```bash
# 核心运行时
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# (如果在CUDA GPU上训练，使用cu121代替cpu)

# 科学计算栈
pip install numpy scipy pandas matplotlib

# 数据管线
pip install joblib tqdm openpyxl h5py

# Web部署
pip install streamlit gspread oauth2client

# 开发工具
pip install jupyterlab
```

**Streamlit Community Cloud的requirements.txt：**
```
torch
torchvision
numpy
scipy
pandas
matplotlib
openpyxl
gspread
oauth2client
tqdm
```

**注意：** 不要在Streamlit Cloud的requirements.txt中将PyTorch固定到仅CPU的wheel — 它会尝试安装CUDA变体并静默失败；`torch`（未固定）在该环境中解析为正确的CPU wheel。

---

## 来源

- PyTorch 2.10.0发布：[PyPI torch](https://pypi.org/project/torch/) — 高置信度
- PyTorch 2.6发布博客：[pytorch.org/blog/pytorch2-6/](https://pytorch.org/blog/pytorch2-6/) — 高置信度
- SciPy 1.15.0：[scipy.org](https://scipy.org/) — 高置信度
- NumPy 2.2.2：[numpy.org/news/](https://numpy.org/news/) — 高置信度
- Matplotlib 3.10.0：Scientific Python SPEC 0 — 高置信度
- Streamlit 1.52.x：[discuss.streamlit.io version-1-52-0](https://discuss.streamlit.io/t/version-1-52-0/120253) — 高置信度
- Streamlit 2025发布说明：[docs.streamlit.io/develop/quick-reference/release-notes/2025](https://docs.streamlit.io/develop/quick-reference/release-notes/2025) — 高置信度
- `solve_ivp`刚性求解器推荐：[scipy Python ODE Solvers](https://pythonnumericalmethods.berkeley.edu/notebooks/chapter22.06-Python-ODE-Solvers.html) — 中等置信度
- joblib并行：[joblib.readthedocs.io](https://joblib.readthedocs.io/) — 高置信度
- tqdm process_map：[tqdm GitHub](https://github.com/tqdm/tqdm) — 高置信度
- pandas + openpyxl：[pandas.read_excel文档](https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html) — 高置信度
- ViT-RR参考项目（model_utils.py, deploy.py, requirements.txt）：直接检查 — 高置信度
