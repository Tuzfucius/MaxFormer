# Repository Environment Baseline

## Git baseline

- Repository: `/home/tzufucius/Project/MaxFormer`
- Branch: `feat/qkta-contrast-history-probe`
- Baseline HEAD: `6ddd4ccde2cc9c4b965059403868e57803b9b6b0`
- Remote: `origin` → `https://github.com/Tuzfucius/MaxFormer` (fetch and push)
- Preserved untracked prompt: `MaxFormer_QKTA_初步开发与实验验证_智能体提示词.md` (not edited)
- Baseline `git status --short` contained the prompt, `reports/`, and `research_qkta/` as untracked. The `.gitignore` now suppresses `data/`, checkpoints, and `outputs/qkta/`; it does not hide the prompt or reports. Final `git status --short --untracked-files=no` was empty; final `git status --short` shows no tracked-file modifications, preserves the prompt as untracked, and does not list data, checkpoint, or `outputs/qkta/` files. `git check-ignore -v` confirmed matches for the CIFAR archive, official checkpoint, and GPU ledger.

## GPU and CUDA

- GPU: NVIDIA GeForce RTX 4080, Ada Lovelace, 32,760 MiB, PCI bus `00000000:01:00.0`, VBIOS `95.03.33.40.5E`.
- Driver: `595.99.02`; `nvidia-smi` reports maximum supported CUDA `13.2`.
- No system `nvcc` was found. The isolated environment uses the PyTorch CUDA 12.4 wheel and reports `torch.cuda.is_available() == True`.

## Conda environment and compatibility

- Environment: `maxformer-qkta`, Python `3.10.22`.
- PyTorch: `2.5.1+cu124`; torchvision: `0.20.1+cu124`; `torch.version.cuda == 12.4`.
- SpikingJelly: `0.0.0.0.12`; timm: `0.6.12`; CuPy: `13.6.0` (`cupy.cuda.runtime.runtimeGetVersion() == 12090`).
- Imports of `timm`, `MultiStepLIFNode`, and all three model source files succeeded with `PYTHONPATH` unset and `PYTHONNOUSERSITE=1`.
- SpikingJelly initially logged `No module named 'tensorboard'` while loading its optional CuPy kernels, causing its internal CuPy reference to be set to `None`; the constructor then raised `AssertionError: CuPy is not installed! You can install it from "https://github.com/cupy/cupy".` Installing `tensorboard==2.14.0` (keeping `protobuf==4.25.9`, compatible with the repository's `wandb==0.16.0`) resolved the import.
- The first CuPy CUDA execution then failed exactly with `CuPy failed to load libnvrtc.so.12: OSError: libnvrtc.so.12: cannot open shared object file: No such file or directory`. The CUDA 12.4 NVRTC and runtime libraries are installed under this Conda environment's `site-packages/nvidia/`; setting `LD_LIBRARY_PATH` to the environment's `nvidia/cuda_nvrtc/lib` and `nvidia/cuda_runtime/lib` allowed both a direct `MultiStepLIFNode(backend='cupy')` probe and the model CuPy smoke tests to pass. No site-packages were patched. Conda `activate.d`/`deactivate.d` hooks were added inside `maxformer-qkta` so an activated environment exposes those paths; a fresh `conda run -n maxformer-qkta` CuPy LIF probe passed.
- PyTorch 2.5.1 + timm 0.6.12 + SpikingJelly 0.0.0.0.12 imported and ran the `backend='torch'` LIF node on CPU and CUDA. No `torch._six` compatibility error occurred. `pip check` reported `No broken requirements found.`

### Environment package inventory

`pip freeze --all` from the Conda environment with inherited `PYTHONPATH` removed:

```text
absl-py==2.5.1
appdirs==1.4.4
certifi==2026.10.23
cffi==2.1.1
charset-normalizer==3.5.2
click==8.5.1
contourpy==1.3.2
cryptography==50.0.2
cupy-cuda12x==13.6.0
cycler==0.12.1
docker-pycreds==0.4.0
einops==0.7.0
exceptiongroup==1.3.2
fastrlock==0.8.4
filelock==4.1.0
fonttools==4.65.0
fsspec==2026.9.0
gitdb==4.0.12
GitPython==3.2.1
google-auth==2.62.2
google-auth-oauthlib==1.0.0
grpcio==1.85.3
huggingface-hub==0.24.6
idna==3.20
iniconfig==2.3.1
Jinja2==3.1.6
kiwisolver==1.5.1
Markdown==3.10.3
MarkupSafe==3.0.3
matplotlib==3.10.9
mpmath==1.3.0
networkx==3.4.2
numpy==1.26.4
nvidia-cublas-cu12==12.4.5.8
nvidia-cuda-cupti-cu12==12.4.127
nvidia-cuda-nvrtc-cu12==12.4.127
nvidia-cuda-runtime-cu12==12.4.127
nvidia-cudnn-cu12==9.1.0.70
nvidia-cufft-cu12==11.2.1.3
nvidia-curand-cu12==10.3.5.147
nvidia-cusolver-cu12==11.6.1.9
nvidia-cusparse-cu12==12.3.1.170
nvidia-nccl-cu12==2.21.5
nvidia-nvjitlink-cu12==12.4.127
nvidia-nvtx-cu12==12.4.127
oauthlib==4.0.1
opencv-python==4.8.1.78
packaging==26.3
pandas==2.3.3
Pillow==9.2.0
pip==26.3
pluggy==1.7.0
protobuf==4.25.9
psutil==7.2.2
pyasn1==0.6.5
pyasn1_modules==0.4.2
pycparser==3.11
Pygments==2.21.0
pyparsing==3.3.3
pytest==9.2.0
python-dateutil==3.0.0
pytz==2026.6
PyYAML==6.0
requests==2.34.2
requests-oauthlib==2.0.0
scipy==1.15.3
sentry-sdk==2.75.0
setproctitle==1.3.8
setuptools==59.8.0
six==1.17.0
smmap==5.0.3
spikingjelly==0.0.0.0.12
sympy==1.13.1
tensorboard==2.14.0
tensorboard-data-server==0.7.2
timm==0.6.12
tomli==2.5.0
torch==2.5.1+cu124
torchvision==0.20.1+cu124
tqdm==4.70.1
triton==3.1.0
typing_extensions==4.16.0
tzdata==2026.6
urllib3==2.8.0
wandb==0.16.0
Werkzeug==3.1.9
wheel==0.48.0
```

## Checkpoint and datasets

- Official checkpoint: `imagenet/checkpoints/ms_qk_10-384_T4.pth.tar`; **66,288,470 bytes**; SHA256 `792a7c95f7ce47492c6cb9603be366c70eac2d233981c8e8daadf76913b33707`.
- Imagenette2-320: extracted at `data/imagenette2-320/`; `val/` exists with **10** class directories and **3,925** image files. The downloaded `.tgz` archive was deleted after extraction.
- CIFAR-10: downloaded and extracted through `torchvision.datasets.CIFAR10` to `data/cifar10` with both `train=True` and `train=False`, `download=True`; **50,000 train** and **10,000 test** examples across 10 classes. The verified archive `data/cifar10/cifar-10-python.tar.gz` is 170,498,071 bytes.
- Exact `du -sh data/ imagenet/checkpoints/` output: `701M	data/` and `64M	imagenet/checkpoints/` (about 765 MiB combined, well under 5 GB).

## Model import and forward smoke tests

All model imports succeeded. Models were set to `eval()` and run under `torch.no_grad()` with T=4 dummy input, first on CPU and CUDA using `backend='torch'`, then on CUDA using `backend='cupy'` where applicable. CuPy was enabled with the Conda NVRTC/runtime paths in `LD_LIBRARY_PATH`.

| Variant | Input | backend=torch CPU | backend=torch CUDA | backend=cupy CUDA |
| --- | --- | --- | --- | --- |
| ImageNet `MS_QKformer`, 384 dims, 1,000 classes | `(1,3,64,64)` | `[1,1000]`, 0.034923 s | `[1,1000]`, 0.134126 s | `[1,1000]`, 0.037656 s |
| CIFAR `MS_QKformer`, 384 dims, 10 classes | `(1,3,32,32)` | `[1,10]`, 0.047327 s | `[1,10]`, 0.019323 s | `[1,10]`, 0.005900 s |
| Event `MS_QKformer`, 512 dims, 11 classes | `(1,2,128,128)` | TypeError | TypeError | TypeError |

The requested ImageNet factory call `ms_qkformer_10_384(T=4, num_classes=1000, in_channels=3)` raises `TypeError: qkta_smoke_0.MS_QKformer() got multiple values for keyword argument 'num_classes'` because the wrapper supplies those values and also forwards the explicit kwargs. The equivalent underlying `MS_QKformer(in_channels=3, num_classes=1000, embed_dims=384, mlp_ratios=4, T=4, depths=10)` was instantiated and passed all three backend/device smoke tests.

The event constructor was instantiated with the requested dimensions and `mlp_ratios=4`; its forward pass fails in all tested modes with the exact error `Embed.forward() got an unexpected keyword argument 'dual'`. No upstream source file was modified.

The GPU ledger at `outputs/qkta/gpu_budget_log.json` contains the P0 smoke entry and direct CuPy LIF probe: **0.224849 GPU seconds** total (**0.000062458 GPU hours**) against 7-hour soft and 8-hour hard caps.

## P0 Code & Risk Audit Notes

### 1. QKTA (`Token_QK_Attention`) Implementation & 5D Tensor Contract

Across all three task directories, `Token_QK_Attention` and `Block_QKA` are defined in `mixer_hub.py` with identical tensor contracts and `MultiStepLIFNode` hyperparameters (`tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy'`):

| File | `Block_QKA` Lines | `Token_QK_Attention` Lines | `attn_lif` Definition | `forward` Key Lines |
| --- | --- | --- | --- | --- |
| `imagenet/mixer_hub.py` | L46–L67 | L69–L121 | L85: `MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')` | L92–L121 |
| `event/mixer_hub.py` | L39–L60 | L62–L113 | L78: `MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')` | L85–L113 |
| `cifar10-100/mixer_hub.py` | L41–L62 | L64–L115 | L80: `MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')` | L87–L115 |

#### Exact 5D Tensor Contract (`Token_QK_Attention.forward`)
Using `imagenet/mixer_hub.py` L93–L118 as reference:
1. Input `x`: `[T, B, C, H, W]` (L93), `N = H * W` after `x.flatten(3)` (L97–L98).
2. `q` and `k`: after `q_lif` / `k_lif` and `.unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)` (L104, L109), both have shape `[T, B, heads, D_head, N]` with binary `{0, 1}` spikes, where `D_head = C // self.num_heads`.
3. `q_sum`: `q = torch.sum(q, dim=3, keepdim=True)` (L111) produces shape **`[T, B, heads, 1, N]`** (integer channel spike counts in `{0, 1, ..., D_head}`).
4. Gate output `attn = self.attn_lif(q)` (L112) must preserve shape **`[T, B, heads, 1, N]`**, `dtype`, and `device`.
5. Key gating `x = torch.mul(attn, k)` (L113) broadcasts `attn` across `D_head` to produce `[T, B, heads, D_head, N]`.
6. Spatial grid `(H, W)` must be captured directly from `x.shape` at `Token_QK_Attention.forward` entry (`T, B, C, H, W = x.shape`), **never** inferred via `sqrt(N)`, so non-square feature maps (`H != W`) are supported.

---

### 2. `MS_QKformer` Architecture & `Token_QK_Attention` Placement

| Variant | File | Stage 1 (`Block_QKA`) | Stage 2 | Stage 3 | Total `Token_QK_Attention` Modules & Exact Module Paths |
| --- | --- | --- | --- | --- | --- |
| ImageNet | `imagenet/ms_qkformer.py` | L120–L125: 1× `Block_QKA` | L132–L137: 2× `Block_QKA` | L144–L149: 7× `Block_SSA` | **3**: `stage1.0.attn`, `stage2.0.attn`, `stage2.1.attn` |
| Event (CIFAR10-DVS) | `event/ms_qkformer.py` | L28–L32: 1× `Block_QKA` | L38–L40: 1× `Block_SSA` | *(None)* | **1**: `stage1.0.attn` (Stage 2 is `Block_SSA`, NOT QKTA) |
| CIFAR10 / CIFAR100 | `cifar10-100/ms_qkformer.py` | L26–L28: 1× `Block_QKA` | L35–L37: 1× `Block_QKA` | L42–L45: 2× `Block_SSA` | **2**: `stage1.0.attn`, `stage2.0.attn` |

> [!WARNING]
> `event/ms_qkformer.py` has only **1** `Token_QK_Attention` module (`stage1.0.attn`); Stage 2 uses `Block_SSA` (L38). Any experiment on `event` must target `stage1.0.attn` only and must not claim "Stage 1/2 are both QKTA".

> [!NOTE]
> All `MultiStepLIFNode` instances in `imagenet/`, `event/`, and `cifar10-100/` hardcode `backend='cupy'` (except `head_lif` in `event/ms_qkformer.py` L47 and `cifar10-100/ms_qkformer.py` L54 which omit `backend`). For CPU unit tests / CPU inference without CuPy, `backend` must be switched in-memory to `'torch'` on `MultiStepLIFNode` modules.

---

### 3. Default YAML `model` Field Pitfalls

All default YAML configs specify `max_former` / `maxformer_10_384` rather than `ms_qkformer` / `ms_qkformer_10_384`:
- `imagenet/conf/10_384_t4.yml` L12: `model: maxformer_10_384` (and `time_step: 4` at L15; note L16 also has a typo `warmup:epochs: 5`). For MS-QKFormer ImageNet T=4, experiments must explicitly use `ms_qkformer_10_384` with `T=4`.
- `event/cifar10dvs.yaml` L33: `model: max_former` (`T: 16` at L1). Must explicitly switch to `ms_qkformer` for QKTA experiments.
- `cifar10-100/cifar10.yaml` L35: `model: max_former` (`time_step: 4` at L2).
- `cifar10-100/cifar100.yaml` L34: `model: max_former` (`time_step: 4` at L2).

Applying QKTA gates to `max_former` is invalid because `max_former` does not use the same `Token_QK_Attention` structure as `ms_qkformer`.

---

### 4. Checkpoint Loading Risk: `imagenet/misc.py::load_old_ckpt`

In `imagenet/misc.py` L332–L383 (`load_old_ckpt`):
- L350–L351 converts `ckpt.items()` and `model.state_dict().items()` to ordered lists.
- L362–L373 iterates by index `i` and copies `new_state_dict[mk] = cv` **solely if `mv.shape == cv.shape`**, ignoring parameter key names (`ck` vs `mk`).
- L376 loads with `model.load_state_dict(new_state_dict, strict=False)`.

**Risk**: If a checkpoint comes from a different architecture variant (`maxformer_10_384` vs `ms_qkformer_10_384`), has different layer ordering, or has transposed/permuted tensors of matching shape, `load_old_ckpt` will **silently mis-map weights** without raising an error. Research evaluation must **never** rely on `load_old_ckpt` blindly; it must verify exact key-name correspondence, shape matching, and `strict=True` (or explicit audited key mapping).

---

### 5. ImageNet Single-GPU Evaluation Risks: `imagenet/test.py`, `test.sh`, `utils.py`

1. **Missing default config**: `imagenet/test.py` L35 sets `default="imagenet.yml"`, which does not exist in `imagenet/` or repo root. Running without `-c <existing_yaml>` raises `FileNotFoundError`.
2. **Single-GPU `args.rank` and `global_rank` UnboundLocalError / AttributeError**:
   - In `imagenet/misc.py` L234–L238, when distributed mode is not initialized (`Not using distributed mode`), `init_distributed_mode(args)` returns **without setting `args.rank`**.
   - `imagenet/test.py` L198 immediately calls `random_seed(args.seed, int(args.rank))`, which raises `AttributeError: 'Namespace' object has no attribute 'rank'` on single-GPU/CPU runs.
   - `imagenet/test.py` L204–L218 only defines `global_rank = misc.get_rank()` inside `if args.distributed:` (L206). When `args.distributed` is `False`, L221 (`if global_rank == 0 and ...`) raises `UnboundLocalError: local variable 'global_rank' referenced before assignment`.
3. **8-GPU `test.sh` configuration**: `imagenet/test.sh` L9 hardcodes `torchrun --nnodes=1 --nproc_per_node=8 ... -c ./conf/10_512_t1.yml --resume ./checkpoints/10-512-T1.pth.tar`, which is unsuitable for single-GPU / CPU environments.
4. **`imagenet/utils.py::evaluate` reset & AMP**:
   - L256 uses `with torch.cuda.amp.autocast():`, and L261 calls `functional.reset_net(model)` after each batch (essential to prevent membrane potential leakage across batches). On CPU-only runs, `torch.cuda.amp.autocast()` should be disabled or guarded.
