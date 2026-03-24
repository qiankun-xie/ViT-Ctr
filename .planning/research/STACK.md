# Technology Stack

**Project:** ViT-Ctr — RAFT Chain Transfer Constant Prediction via Vision Transformer
**Researched:** 2026-03-24
**Overall confidence:** HIGH (stack is established from reference project ViT-RR + verified current versions)

---

## Recommended Stack

### Core Deep Learning

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Python | 3.11 | Runtime | Stable, well-supported by all deps; 3.12 works but 3.11 has wider pre-built wheel coverage for PyTorch on Windows |
| PyTorch | 2.6.x–2.10.x | Model definition, training, inference | Proven in reference project ViT-RR; `nn.TransformerEncoderLayer` is the SimpViT backbone; latest stable is 2.10.0 (Jan 2026) |
| torchvision | match PyTorch | Image transforms, tensor utilities | Required peer of PyTorch; no separate CNN needed but provides consistent transform API |

**Confidence: HIGH** — PyTorch 2.6+ is current stable verified via PyPI. Reference project already runs on this stack.

**Note on hardware:** The project runs on Intel i5-10210U + MX350 (2 GB VRAM). SimpViT is ~3.4 MB — local CPU training is feasible for test runs. For million-sample dataset generation, use Google Colab free GPU (T4). Do NOT add PyTorch Lightning or other wrappers; SimpViT is too small to benefit and the reference project uses raw PyTorch.

---

### Data Generation (ODE Simulation)

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| SciPy | 1.15.x | `solve_ivp` for RAFT ODE integration | The canonical Python ODE solver; `method='Radau'` or `method='BDF'` handles stiff RAFT kinetics (wide timescale range in radical polymerization); current stable is 1.15.0 (Jan 2025) |
| NumPy | 2.2.x | Array operations, fingerprint construction | Required by both SciPy and PyTorch; NumPy 2.x is the current generation; current is 2.2.2 (Jan 2025) |
| joblib | 1.4.x | CPU-parallel ODE generation across parameter grid | `Parallel(n_jobs=-1)(delayed(simulate)(params) for params in grid)` pattern; essential for generating millions of samples on a multi-core CPU without custom multiprocessing boilerplate |
| tqdm | 4.x | Progress tracking during generation and training | `tqdm.contrib.concurrent.process_map` integrates tqdm with `ProcessPoolExecutor`; minimal overhead (60 ns/iteration) |

**Confidence: HIGH** for SciPy/NumPy/joblib. MEDIUM for specific version pins — verify at install time via `pip install --upgrade`.

**Why `solve_ivp` not `odeint`:** RAFT kinetics are stiff (fast radical reactions coupled with slow polymer growth). `solve_ivp` with `method='Radau'` is the modern API that exposes stiff solvers with dense output and event detection. `odeint` is legacy and only wraps LSODA. Use `solve_ivp(fun, t_span, y0, method='Radau', dense_output=True)`.

**Why not polykin:** PolyKin (HugoMVale/polykin) is an early-stage library targeting polymerization engineering calculations. It does not expose a purpose-built RAFT ODE system that matches ViT-Ctr's parametric requirements (sweeping Ctr, kadd/kfrag, [CTA]/[M], initiator concentration). Writing the RAFT ODE from first principles with `solve_ivp` gives full control and is the standard approach in the literature (including the ViT-RR reference using kMC).

---

### Fingerprint Encoding and Data Pipeline

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| NumPy | 2.2.x | ctFP construction (64x64 dual-channel arrays) | Direct port of `transform()` from ViT-RR model_utils.py; scatter-plot-to-image encoding is ~10 lines of NumPy |
| pandas | 2.2.x | Parameter grid construction, results DataFrame, Excel I/O | `pd.read_excel(..., engine='openpyxl')` is the standard pattern; current stable is 3.0.x but 2.2.x is widely deployed and stable |
| openpyxl | 3.1.x | xlsx read/write backend for pandas | Default engine for `.xlsx` in pandas 2.x+; required for Streamlit Excel upload feature |
| h5py or numpy .npz | — | Serializing the million-sample fingerprint dataset | HDF5 (h5py) for chunked access during training; `.npz` is fine if dataset fits in RAM (~64x64x2xfloat32 x 1M = ~32 GB — HDF5 required); choose at generation time |

**Confidence: HIGH** for pandas+openpyxl pattern (verified in ViT-RR deploy.py). MEDIUM for h5py recommendation — depends on final dataset size.

**pandas version note:** pandas 3.0.x is current but introduces copy-on-write changes that may break code written for 2.x. Use `pandas>=2.2,<3.0` in requirements.txt to avoid unexpected breakage during development, then upgrade after testing.

---

### Training Infrastructure

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| PyTorch | 2.6+ | `DataLoader`, `Dataset`, optimizer, loss | All-in-one; no need for separate dataset library |
| torch.utils.data | (part of PyTorch) | Custom Dataset wrapping the ctFP array | Use `TensorDataset` if dataset fits in RAM; custom `Dataset` + HDF5 if not |
| `torch.optim.Adam` | (part of PyTorch) | Optimizer | Adam with lr=3e-4 is the standard for small Transformers; ViT-RR uses this pattern |
| MSELoss | (part of PyTorch) | Loss for regression of log10(Ctr), inhibition, retardation | Three-output regression; log10-transforming Ctr before training prevents scale imbalance |
| `torch.compile` | (part of PyTorch 2.x) | Optional: 10–30% speedup on CPU | Add as a final optimization step; not needed for initial development |

**Confidence: HIGH** — directly from ViT-RR architecture.

---

### Uncertainty Estimation

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| NumPy | 2.2.x | Bootstrap loop, covariance matrix | `np.cov(predictions, rowvar=False)` per ViT-RR deploy.py |
| `scipy.stats.f` | (part of SciPy 1.15.x) | F-distribution 95% JCI | Exact pattern from ViT-RR: `f.ppf(0.95, dfn=p, dfd=n-p)` → joint confidence interval half-width |

**Confidence: HIGH** — copied directly from ViT-RR, which is peer-reviewed.

---

### Web Deployment

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Streamlit | 1.41+ (current: 1.52.x) | Web app framework | Direct constraint from PROJECT.md; matches ViT-RR deploy.py; zero JS/HTML needed for this use case |
| gspread | 6.x | Google Sheets user registry | Used in ViT-RR for tracking institutional users; lightweight alternative to a database for a research tool |
| oauth2client | 4.x | Google Sheets authentication | Used in ViT-RR; note: `oauth2client` is technically deprecated in favor of `google-auth`, but it still works and the ViT-RR pattern is stable |

**Confidence: HIGH** for Streamlit. MEDIUM for gspread/oauth2client — if user tracking is dropped (optional feature), these can be omitted entirely.

**Streamlit version note:** Current stable is 1.52.x (Dec 2025). The 1.41.x series is still functional; no breaking changes that affect ViT-RR's pattern. Pin to `>=1.40` to stay compatible with existing deploy.py patterns while allowing upgrades.

**Deployment target:** Streamlit Community Cloud (free tier) for research use. Requires `requirements.txt` at repo root and `secrets.toml` for Google Sheets credentials. Model weights are committed to the repo (at ~3.4 MB per model, this is fine for Git).

---

### Development Tools

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Jupyter Notebook / JupyterLab | 4.x | Exploratory ODE testing, fingerprint visualization, model prototyping | Standard for scientific ML; allows inline plots of ctFP images and ODE curves before committing to scripts |
| Matplotlib | 3.10.x | Fingerprint visualization, training curves, parity plots | Standard scientific plotting; `imshow` for ctFP inspection; current stable is 3.10.0 (Dec 2024) |
| Google Colab | — | Large-scale training on free T4 GPU | Hardware constraint: MX350 is insufficient for million-sample training; Colab T4 provides ~16 GB VRAM; requires exporting dataset to Google Drive |

**Confidence: HIGH** for Matplotlib. MEDIUM for Colab (free tier availability is not guaranteed, but historically reliable for this workload size).

---

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

---

## Installation

```bash
# Core runtime
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# (Use cu121 instead of cpu if training on a CUDA GPU)

# Scientific stack
pip install numpy scipy pandas matplotlib

# Data pipeline
pip install joblib tqdm openpyxl h5py

# Web deployment
pip install streamlit gspread oauth2client

# Development
pip install jupyterlab
```

**requirements.txt for Streamlit Community Cloud:**
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

**Note:** Do not pin PyTorch to a CPU-only wheel in requirements.txt for Streamlit Cloud — it will attempt to install the CUDA variant and fail silently; `torch` (unpinned) resolves to the correct CPU wheel in that environment.

---

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
