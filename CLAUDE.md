<!-- GSD:project-start source:PROJECT.md -->
## Project

**ViT-Ctr: 基于Vision Transformer的RAFT链转移常数预测系统**

基于简化Vision Transformer（SimpViT）架构，从RAFT聚合实验动力学数据中同时提取链转移常数（Ctr）、诱导期（inhibition period）和减速因子（retardation factor）三个关键参数的深度学习系统。模仿ViT-RR（Angew. Chem. Int. Ed. 2025, DOI: 10.1002/anie.202513086）的范式，将实验数据编码为链转移指纹（ctFP），覆盖所有类型的RAFT剂体系（dithioester、trithiocarbonate、xanthate、dithiocarbamate等），并提供Web应用供研究者在线使用。

**Core Value:** 一次输入实验数据，同时提取Ctr、诱导期和减速因子三个参数——传统方法需要三组独立实验才能分别获得。

### Constraints

- **技术栈**: PyTorch + Streamlit，与ViT-RR保持一致
- **模型架构**: SimpViT（64×64输入, patch_size=16, hidden=64, 2层Transformer, 4头注意力），输出维度改为3
- **数据格式**: 双通道ctFP（Ch1=Mn, Ch2=Đ），坐标轴（x=[CTA]/[M], y=conversion）
- **硬件**: 本地开发调试，大规模训练可选Colab
- **语言**: 项目文档和代码注释用中文，论文和SI用英文
<!-- GSD:project-end -->

<!-- GSD:stack-start source:research/STACK.md -->
## Technology Stack

## Recommended Stack
### Core Deep Learning
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Python | 3.11 | Runtime | Stable, well-supported by all deps; 3.12 works but 3.11 has wider pre-built wheel coverage for PyTorch on Windows |
| PyTorch | 2.6.x–2.10.x | Model definition, training, inference | Proven in reference project ViT-RR; `nn.TransformerEncoderLayer` is the SimpViT backbone; latest stable is 2.10.0 (Jan 2026) |
| torchvision | match PyTorch | Image transforms, tensor utilities | Required peer of PyTorch; no separate CNN needed but provides consistent transform API |
### Data Generation (ODE Simulation)
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| SciPy | 1.15.x | `solve_ivp` for RAFT ODE integration | The canonical Python ODE solver; `method='Radau'` or `method='BDF'` handles stiff RAFT kinetics (wide timescale range in radical polymerization); current stable is 1.15.0 (Jan 2025) |
| NumPy | 2.2.x | Array operations, fingerprint construction | Required by both SciPy and PyTorch; NumPy 2.x is the current generation; current is 2.2.2 (Jan 2025) |
| joblib | 1.4.x | CPU-parallel ODE generation across parameter grid | `Parallel(n_jobs=-1)(delayed(simulate)(params) for params in grid)` pattern; essential for generating millions of samples on a multi-core CPU without custom multiprocessing boilerplate |
| tqdm | 4.x | Progress tracking during generation and training | `tqdm.contrib.concurrent.process_map` integrates tqdm with `ProcessPoolExecutor`; minimal overhead (60 ns/iteration) |
### Fingerprint Encoding and Data Pipeline
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| NumPy | 2.2.x | ctFP construction (64x64 dual-channel arrays) | Direct port of `transform()` from ViT-RR model_utils.py; scatter-plot-to-image encoding is ~10 lines of NumPy |
| pandas | 2.2.x | Parameter grid construction, results DataFrame, Excel I/O | `pd.read_excel(..., engine='openpyxl')` is the standard pattern; current stable is 3.0.x but 2.2.x is widely deployed and stable |
| openpyxl | 3.1.x | xlsx read/write backend for pandas | Default engine for `.xlsx` in pandas 2.x+; required for Streamlit Excel upload feature |
| h5py or numpy .npz | — | Serializing the million-sample fingerprint dataset | HDF5 (h5py) for chunked access during training; `.npz` is fine if dataset fits in RAM (~64x64x2xfloat32 x 1M = ~32 GB — HDF5 required); choose at generation time |
### Training Infrastructure
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| PyTorch | 2.6+ | `DataLoader`, `Dataset`, optimizer, loss | All-in-one; no need for separate dataset library |
| torch.utils.data | (part of PyTorch) | Custom Dataset wrapping the ctFP array | Use `TensorDataset` if dataset fits in RAM; custom `Dataset` + HDF5 if not |
| `torch.optim.Adam` | (part of PyTorch) | Optimizer | Adam with lr=3e-4 is the standard for small Transformers; ViT-RR uses this pattern |
| MSELoss | (part of PyTorch) | Loss for regression of log10(Ctr), inhibition, retardation | Three-output regression; log10-transforming Ctr before training prevents scale imbalance |
| `torch.compile` | (part of PyTorch 2.x) | Optional: 10–30% speedup on CPU | Add as a final optimization step; not needed for initial development |
### Uncertainty Estimation
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| NumPy | 2.2.x | Bootstrap loop, covariance matrix | `np.cov(predictions, rowvar=False)` per ViT-RR deploy.py |
| `scipy.stats.f` | (part of SciPy 1.15.x) | F-distribution 95% JCI | Exact pattern from ViT-RR: `f.ppf(0.95, dfn=p, dfd=n-p)` → joint confidence interval half-width |
### Web Deployment
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Streamlit | 1.41+ (current: 1.52.x) | Web app framework | Direct constraint from PROJECT.md; matches ViT-RR deploy.py; zero JS/HTML needed for this use case |
| gspread | 6.x | Google Sheets user registry | Used in ViT-RR for tracking institutional users; lightweight alternative to a database for a research tool |
| oauth2client | 4.x | Google Sheets authentication | Used in ViT-RR; note: `oauth2client` is technically deprecated in favor of `google-auth`, but it still works and the ViT-RR pattern is stable |
### Development Tools
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Jupyter Notebook / JupyterLab | 4.x | Exploratory ODE testing, fingerprint visualization, model prototyping | Standard for scientific ML; allows inline plots of ctFP images and ODE curves before committing to scripts |
| Matplotlib | 3.10.x | Fingerprint visualization, training curves, parity plots | Standard scientific plotting; `imshow` for ctFP inspection; current stable is 3.10.0 (Dec 2024) |
| Google Colab | — | Large-scale training on free T4 GPU | Hardware constraint: MX350 is insufficient for million-sample training; Colab T4 provides ~16 GB VRAM; requires exporting dataset to Google Drive |
## Alternatives Considered
| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Web framework | Streamlit | Gradio | Gradio is equally valid but project constraint mandates Streamlit; ViT-RR is Streamlit |
| Web framework | Streamlit | Flask + React | Massively overengineered for a research tool; weeks of extra work with no benefit |
| ODE solver | `scipy.integrate.solve_ivp` | Julia / DifferentialEquations.jl | Julia would be faster but breaks the pure-Python stack; unnecessary for this scale |
| ODE solver | `scipy.integrate.solve_ivp` | polykin library | Too early-stage; doesn't expose the parametric RAFT ODE system needed |
| ODE parallelism | joblib | multiprocessing stdlib | joblib is higher-level, handles serialization edge cases better, integrates with tqdm |
| ODE parallelism | joblib | Dask | Dask adds cluster overhead that is not justified for a single-machine generation job |
| Deep learning | PyTorch | TensorFlow/Keras | Project constraint mandates PyTorch; ViT-RR is PyTorch; no reason to switch |
| Deep learning | PyTorch (raw) | PyTorch Lightning | Lightning is useful for large distributed training; SimpViT training loop is ~30 lines — overhead not justified |
| Dataset storage | HDF5 (h5py) | Zarr | Zarr is modern but h5py has more tutorials and is universally supported; either works |
| User auth | gspread (Google Sheets) | SQLite | SQLite would be cleaner but requires server-side DB; Google Sheets works for a low-traffic research app |
## Installation
# Core runtime
# (Use cu121 instead of cpu if training on a CUDA GPU)
# Scientific stack
# Data pipeline
# Web deployment
# Development
## Sources
- PyTorch 2.10.0 release: [PyPI torch](https://pypi.org/project/torch/) — HIGH confidence
- PyTorch 2.6 release blog: [pytorch.org/blog/pytorch2-6/](https://pytorch.org/blog/pytorch2-6/) — HIGH confidence
- SciPy 1.15.0: [scipy.org](https://scipy.org/) — HIGH confidence
- NumPy 2.2.2: [numpy.org/news/](https://numpy.org/news/) — HIGH confidence
- Matplotlib 3.10.0: Scientific Python SPEC 0 — HIGH confidence
- Streamlit 1.52.x: [discuss.streamlit.io version-1-52-0](https://discuss.streamlit.io/t/version-1-52-0/120253) — HIGH confidence
- Streamlit 2025 release notes: [docs.streamlit.io/develop/quick-reference/release-notes/2025](https://docs.streamlit.io/develop/quick-reference/release-notes/2025) — HIGH confidence
- `solve_ivp` stiff solver recommendation: [scipy Python ODE Solvers](https://pythonnumericalmethods.berkeley.edu/notebooks/chapter22.06-Python-ODE-Solvers.html) — MEDIUM confidence
- joblib parallel: [joblib.readthedocs.io](https://joblib.readthedocs.io/) — HIGH confidence
- tqdm process_map: [tqdm GitHub](https://github.com/tqdm/tqdm) — HIGH confidence
- pandas + openpyxl: [pandas.read_excel docs](https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html) — HIGH confidence
- ViT-RR reference project (model_utils.py, deploy.py, requirements.txt): direct inspection — HIGH confidence
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

Conventions not yet established. Will populate as patterns emerge during development.
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

Architecture not yet mapped. Follow existing patterns found in the codebase.
<!-- GSD:architecture-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd:quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd:debug` for investigation and bug fixing
- `/gsd:execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd:profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
