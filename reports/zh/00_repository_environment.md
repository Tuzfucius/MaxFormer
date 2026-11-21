# 仓库环境基线

## Git 基线

- 仓库：`/home/tzufucius/Project/MaxFormer`；分支：`feat/qkta-contrast-history-probe`；基线 HEAD：`6ddd4ccde2cc9c4b965059403868e57803b9b6b0`。
- 远程仓库：`origin` → `https://github.com/Tuzfucius/MaxFormer`（fetch/push）。
- 保留的未跟踪提示文件 `MaxFormer_QKTA_初步开发与实验验证_智能体提示词.md` 未编辑。基线 `git status --short` 将该提示文件、`reports/` 和 `research_qkta/` 列为未跟踪内容。`.gitignore` 已忽略 `data/`、检查点和 `outputs/qkta/`，不会忽略提示文件或报告。最终 `git status --short --untracked-files=no` 为空；最终 `git status --short` 没有已跟踪文件改动，提示文件仍未跟踪，且不列出数据、检查点或 `outputs/qkta/` 文件。`git check-ignore -v` 确认 CIFAR 压缩包、官方检查点和 GPU 记录均被忽略。

## GPU 与 CUDA

- GPU：NVIDIA GeForce RTX 4080，Ada Lovelace，32,760 MiB，PCI 总线 `00000000:01:00.0`，VBIOS `95.03.33.40.5E`。
- 驱动 `595.99.02`；`nvidia-smi` 报告支持的最高 CUDA 版本为 `13.2`。
- 系统中没有 `nvcc`。隔离环境使用 PyTorch CUDA 12.4 wheel，`torch.cuda.is_available() == True`。

## Conda 环境与兼容性

- 环境 `maxformer-qkta`，Python `3.10.22`；PyTorch `2.5.1+cu124`；torchvision `0.20.1+cu124`；`torch.version.cuda == 12.4`。
- SpikingJelly `0.0.0.0.12`；timm `0.6.12`；CuPy `13.6.0`（`cupy.cuda.runtime.runtimeGetVersion() == 12090`）。清除 `PYTHONPATH` 并设置 `PYTHONNOUSERSITE=1` 后，`timm`、`MultiStepLIFNode` 和全部三个模型源文件均成功导入。
- SpikingJelly 首次加载可选 CuPy 内核时记录 `No module named 'tensorboard'`，内部 CuPy 引用因此被设为 `None`，之后构造器报 `AssertionError: CuPy is not installed!`。安装 `tensorboard==2.14.0`（保留与仓库 `wandb==0.16.0` 兼容的 `protobuf==4.25.9`）解决了导入问题。
- 首次 CuPy CUDA 执行时报 `CuPy failed to load libnvrtc.so.12`。CUDA 12.4 的 NVRTC/runtime 库位于此 Conda 环境的 `site-packages/nvidia/` 下；将环境中的 `nvidia/cuda_nvrtc/lib` 和 `nvidia/cuda_runtime/lib` 加入 `LD_LIBRARY_PATH` 后，直接的 `MultiStepLIFNode(backend='cupy')` 探测和模型 CuPy 冒烟测试均通过。未修改 site-packages。已在 `maxformer-qkta` 中添加 Conda `activate.d`/`deactivate.d` hooks，使激活环境时能找到这些路径；新运行的 `conda run -n maxformer-qkta` CuPy LIF 探测通过。
- PyTorch 2.5.1、timm 0.6.12 和 SpikingJelly 0.0.0.0.12 均成功导入，并在 CPU 与 CUDA 上运行 `backend='torch'` 的 LIF 节点。未出现 `torch._six` 兼容错误；`pip check` 报告 `No broken requirements found.`

### 环境软件包清单

清单由在移除继承的 `PYTHONPATH` 后，于 Conda 环境中运行 `pip freeze --all` 得到。软件包名和版本号是机器可读标识，保持原样：

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

## 检查点与数据集

- 官方检查点 `imagenet/checkpoints/ms_qk_10-384_T4.pth.tar`：**66,288,470 字节**；SHA256 `792a7c95f7ce47492c6cb9603be366c70eac2d233981c8e8daadf76913b33707`。
- Imagenette2-320 已解压到 `data/imagenette2-320/`；`val/` 中有 **10** 个类别目录、**3,925** 张图片。下载的 `.tgz` 在解压后已删除。
- CIFAR-10 通过 `torchvision.datasets.CIFAR10` 下载并解压到 `data/cifar10`，训练集和测试集都设置 `download=True`；共 10 类，**50,000** 个训练样本和 **10,000** 个测试样本。已验证的 `data/cifar10/cifar-10-python.tar.gz` 大小为 170,498,071 字节。
- `du -sh data/ imagenet/checkpoints/` 的原始输出为 `701M data/`、`64M imagenet/checkpoints/`，合计约 765 MiB，远低于 5 GB。

## 模型导入与前向冒烟测试

所有模型均成功导入。模型设置为 `eval()`，在 `torch.no_grad()` 下用 T=4 虚拟输入运行：先在 CPU/CUDA 上使用 `backend='torch'`，再在适用时于 CUDA 上使用 `backend='cupy'`。CuPy 通过 Conda 的 NVRTC/runtime 路径加入 `LD_LIBRARY_PATH` 启用。

| 变体 | 输入 | `backend=torch` CPU | `backend=torch` CUDA | `backend=cupy` CUDA |
| --- | --- | --- | --- | --- |
| ImageNet `MS_QKformer`，384 维，1,000 类 | `(1,3,64,64)` | `[1,1000]`，0.034923 s | `[1,1000]`，0.134126 s | `[1,1000]`，0.037656 s |
| CIFAR `MS_QKformer`，384 维，10 类 | `(1,3,32,32)` | `[1,10]`，0.047327 s | `[1,10]`，0.019323 s | `[1,10]`，0.005900 s |
| Event `MS_QKformer`，512 维，11 类 | `(1,2,128,128)` | TypeError | TypeError | TypeError |

按要求调用 ImageNet 工厂函数 `ms_qkformer_10_384(T=4, num_classes=1000, in_channels=3)` 会报 `TypeError: qkta_smoke_0.MS_QKformer() got multiple values for keyword argument 'num_classes'`，因为包装器提供这些参数后又转发了显式 kwargs。之后直接实例化等效底层模型 `MS_QKformer(in_channels=3, num_classes=1000, embed_dims=384, mlp_ratios=4, T=4, depths=10)`，三个后端/设备冒烟测试均通过。

Event 构造器按要求尺寸和 `mlp_ratios=4` 成功实例化，但所有模式的前向运行均报 `Embed.forward() got an unexpected keyword argument 'dual'`。未修改上游源文件。`outputs/qkta/gpu_budget_log.json` 的 P0 冒烟测试和直接 CuPy LIF 探测合计耗时 **0.224849 GPU 秒**（**0.000062458 GPU 小时**），低于 7 小时软上限和 8 小时硬上限。

## P0 代码与风险审计记录

### 1. QKTA（`Token_QK_Attention`）实现与 5D 张量约定

三个任务目录的 `mixer_hub.py` 都定义 `Token_QK_Attention` 和 `Block_QKA`，张量契约及 `MultiStepLIFNode` 超参数一致（`tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy'`）：

| 文件 | `Block_QKA` 行号 | `Token_QK_Attention` 行号 | `attn_lif` 定义 | `forward` 关键行 |
| --- | --- | --- | --- | --- |
| `imagenet/mixer_hub.py` | L46–L67 | L69–L121 | L85 | L92–L121 |
| `event/mixer_hub.py` | L39–L60 | L62–L113 | L78 | L85–L113 |
| `cifar10-100/mixer_hub.py` | L41–L62 | L64–L115 | L80 | L87–L115 |

#### 精确的 5D 张量契约（`Token_QK_Attention.forward`）

以下以 `imagenet/mixer_hub.py` L93–L118 为例：

1. 输入 `x` 为 `[T, B, C, H, W]`；执行 `x.flatten(3)` 后 `N = H * W`。
2. `q`、`k` 经 `q_lif` / `k_lif` 及 `.unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)` 后形状均为 `[T, B, heads, D_head, N]`，元素为 `{0,1}`，其中 `D_head = C // self.num_heads`。
3. `q = torch.sum(q, dim=3, keepdim=True)` 得到形状 **`[T, B, heads, 1, N]`**，整数通道脉冲计数范围为 `{0, 1, ..., D_head}`。
4. `attn = self.attn_lif(q)` 必须保持 **`[T, B, heads, 1, N]`** 形状、`dtype` 和 `device`。
5. `x = torch.mul(attn, k)` 沿 `D_head` 广播 `attn`，产生 `[T, B, heads, D_head, N]`。
6. 必须在 `Token_QK_Attention.forward` 入口直接从 `x.shape` 捕获 `(H,W)`；不得通过 `sqrt(N)` 推断，从而支持非正方形特征图（`H != W`）。

### 2. `MS_QKformer` 架构与 `Token_QK_Attention` 位置

| 变体 | Stage 1 | Stage 2 | Stage 3 | QKTA 模块数及路径 |
| --- | --- | --- | --- | --- |
| ImageNet | 1 个 `Block_QKA`（L120–L125） | 2 个 `Block_QKA`（L132–L137） | 7 个 `Block_SSA`（L144–L149） | **3**：`stage1.0.attn`、`stage2.0.attn`、`stage2.1.attn` |
| Event（CIFAR10-DVS） | 1 个 `Block_QKA`（L28–L32） | 1 个 `Block_SSA`（L38–L40） | 无 | **1**：`stage1.0.attn` |
| CIFAR10 / CIFAR100 | 1 个 `Block_QKA`（L26–L28） | 1 个 `Block_QKA`（L35–L37） | 2 个 `Block_SSA`（L42–L45） | **2**：`stage1.0.attn`、`stage2.0.attn` |

> [!WARNING]
> `event/ms_qkformer.py` 只有 **1 个** QKTA 模块（`stage1.0.attn`）；Stage 2 是 `Block_SSA`。Event 实验只能针对 `stage1.0.attn`，不得声称 Stage 1/2 都是 QKTA。

> [!NOTE]
> `imagenet/`、`event/`、`cifar10-100/` 中的 `MultiStepLIFNode` 都将 `backend='cupy'` 写死；例外是 `event/ms_qkformer.py` L47 和 `cifar10-100/ms_qkformer.py` L54 的 `head_lif`，未指定 backend。无 CuPy 的 CPU 单测/推理需在内存中将这些模块的 backend 改为 `'torch'`。

### 3. 默认 YAML 的 `model` 字段注意事项

默认 YAML 使用 `max_former` / `maxformer_10_384`，不是 `ms_qkformer` / `ms_qkformer_10_384`：

- `imagenet/conf/10_384_t4.yml` L12：`model: maxformer_10_384`；L15 为 `time_step: 4`；L16 有拼写错误 `warmup:epochs: 5`。ImageNet T=4 的 MS-QKFormer 实验须显式指定 `ms_qkformer_10_384` 与 `T=4`。
- `event/cifar10dvs.yaml` L33：`model: max_former`（L1 为 `T: 16`）；QKTA 实验须显式改用 `ms_qkformer`。
- `cifar10-100/cifar10.yaml` L35 和 `cifar10-100/cifar100.yaml` L34：均为 `model: max_former`（`time_step: 4`）。

`max_former` 没有 `ms_qkformer` 的 `Token_QK_Attention` 结构，因此不能将 QKTA 门控应用于前者。

### 4. 检查点加载风险：`imagenet/misc.py::load_old_ckpt`

在 L332–L383 的 `load_old_ckpt` 中，L350–L351 将 `ckpt.items()` 和 `model.state_dict().items()` 转为有序列表；L362–L373 按索引遍历，**仅在 `mv.shape == cv.shape` 时**执行 `new_state_dict[mk] = cv`，不核对键名；L376 以 `strict=False` 加载。

**风险：** 检查点若来自不同架构（`maxformer_10_384` 与 `ms_qkformer_10_384`）、层顺序不同，或张量虽形状相同但已转置/置换，权重可能被静默错配且不报错。研究评估不得盲目依赖此函数；须核验精确键名对应、形状匹配，并使用 `strict=True` 或经过审计的显式映射。

### 5. ImageNet 单 GPU 评估风险（`imagenet/test.py`、`test.sh`、`utils.py`）

1. **默认配置缺失：** `imagenet/test.py` L35 默认设为 `imagenet.yml`，但该文件在 `imagenet/` 和仓库根目录都不存在。不传 `-c <现有 yaml>` 会报 `FileNotFoundError`。
2. **单 GPU 下 `args.rank` / `global_rank` 未定义：** `imagenet/misc.py` L234–L238 在未初始化分布式模式时直接返回且不设置 `args.rank`；`imagenet/test.py` L198 随即访问它，会报 `AttributeError`。此外 `global_rank` 只在 `args.distributed` 为真时定义；非分布式运行到 L221 会报 `UnboundLocalError`。
3. **`test.sh` 固定使用 8 GPU：** L9 写死 `torchrun --nnodes=1 --nproc_per_node=8 ... -c ./conf/10_512_t1.yml --resume ./checkpoints/10-512-T1.pth.tar`，不适用于单 GPU/CPU 环境。
4. **`evaluate` 的状态重置与 AMP：** `imagenet/utils.py` L256 使用 `torch.cuda.amp.autocast()`；L261 每个 batch 后调用 `functional.reset_net(model)`，这可避免 batch 间膜电位泄漏。仅 CPU 运行时应禁用或保护 AMP 调用。
