# MaxFormer / MS-QKFormer 二次开发：历史抑制与局部对比保护门控

> **可直接交给 Codex / Claude Code / 其他开发智能体执行的主提示词与分阶段验收协议**  
> 研究性质：SNN / Spiking Transformer 机制验证；当前阶段 **不重训练、不修改主干结构、不虚构实验结果**。  
> 目标仓库：https://github.com/Tuzfucius/MaxFormer  
> 上游仓库：https://github.com/bic-L/MaxFormer  
> 核查基线（2026-11-20）：远端 `main` 与上游 `main` 均位于 `6ddd4ccde2cc9c4b965059403868e57803b9b6b0`。智能体必须在本地重新核实，因为用户的本地克隆状态可能不同。

---

# 0. 开发智能体的角色与总任务

你是一名具有 PyTorch、SpikingJelly、SNN 神经元动力学、计算机视觉模型复现和科研实验设计经验的高级研发工程师。请在用户**已经克隆的 `Tuzfucius/MaxFormer` 本地仓库**中完成本协议所规定的初步开发与验证。

核心研究问题：**在 Q-K Token Attention 的特定默认实现中，门控 LIF 等价于当前时间步的通道 OR 门。能否通过“历史活动抑制 + 空间局部对比保护”，减少无效重复脉冲，同时尽量保持对分类有用的特征？**

请严格遵守以下规则：

1. **先审查后修改**。读取本地 Git 状态、模型文件、运行配置、权重与数据条件；不得凭本提示词中的旧路径猜测文件当前结构。
2. **保留原版可复现性**。所有新功能通过独立模块与配置开关接入，默认行为必须与原版一致；保留原模型、原权重、数据预处理及推理接口。
3. **分阶段实施并验收**。按 P0→P1→P2→P3→P4 顺序推进。一个阶段的阻断条件未满足时，停止依赖该条件的后续阶段并记录原因；其他独立阶段可以继续。
4. **当前不训练模型**。使用冻结推理 `model.eval()` + `torch.no_grad()`；不调用反向传播、优化器或训练循环。没有兼容 checkpoint / ImageNet 数据时，交付 CPU 单元测试、合成探针和真实网络的随机权重前向兼容性测试，并明确“尚未得到预训练网络性能结果”。
5. **实验不得挑结果**。预先规定参数扫描、层位、数据索引、随机种子及失败记录；合成图不能仅包含容易使新方案获胜的样本。
6. **Git 纪律**。新建独立功能分支、阶段性提交（commit）；远端 `origin` 是用户仓库且具有授权时推送该分支。不强推、不覆盖 main、不提交数据集、预训练权重、数 GB 结果或敏感凭据。每次提交前检查 diff 与测试结果。
7. **不夸大结论**。区分“数学等价证明”“合成探针行为”“冻结模型性能”“实际全网能耗”；不能把局部门后脉冲率下降直接写成芯片级节能。
8. **任务结束提供完整报告**：代码改动、关键 diff、运行命令、实验配置、图表路径、成功/失败项、可复现的 Git SHA、下一步建议。

---

# 1. 已核查的仓库事实与开发落点

## 1.1 本地项目检查（必须重新执行）

从本地仓库根目录运行：

```bash
pwd
git status --short
git branch --show-current
git rev-parse HEAD
git remote -v
find . -maxdepth 2 -type f | sort | head -120
python --version
python -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available(), 'torch_cuda', torch.version.cuda)" 2>&1 || true
nvidia-smi || true
```

核查 `origin` 是否指向 `Tuzfucius/MaxFormer`，`upstream`（如有）是否指向 `bic-L/MaxFormer`。如仓库有未提交用户改动，**不得清理或覆盖**；先报告并选择不冲突分支。若可以安全创建新分支，命名：

```bash
git switch -c feat/qkta-contrast-history-probe
```

保留本地 HEAD SHA、依赖版本、CUDA/驱动信息、可用显存，并写入 `reports/00_repository_environment.md`。

## 1.2 仓库文件映射（本次公开仓库核查结果）

| 任务/环境 | 已检查路径 | 注意事项 |
| --- | --- | --- |
| QKTA 核心实现 | `imagenet/mixer_hub.py`、`event/mixer_hub.py`、`cifar10-100/mixer_hub.py` | 均定义 `Token_QK_Attention` 和 `Block_QKA` |
| ImageNet MS-QKFormer | `imagenet/ms_qkformer.py` | Stage 1：1 个 QKTA；Stage 2：2 个 QKTA；Stage 3：7 个 SSA |
| 事件视觉 MS-QKFormer | `event/ms_qkformer.py` | **仅 Stage 1：1 个 QKTA**；Stage 2 是 SSA；不能称“Stage 1/2 全部 QKTA” |
| CIFAR10/100 MS-QKFormer | `cifar10-100/ms_qkformer.py` | Stage 1、Stage 2 各 1 个 QKTA；Stage 3 是 SSA |
| 事件默认配置 | `event/cifar10dvs.yaml` | `model: max_former`，T=16；做 QKTA 研究必须显式切换 `ms_qkformer` |
| 静态默认配置 | `cifar10-100/cifar10.yaml`、`cifar100.yaml` | 也默认使用 `max_former`；不是 MS-QKFormer |
| ImageNet 配置 | `imagenet/conf/10_384_t4.yml` | `model: maxformer_10_384`；使用 MS-QKFormer 需单独创建覆盖配置 |
| ImageNet 测试 | `imagenet/test.py`、`imagenet/test.sh` | `test.sh` 使用 8 进程分布式配置；**不适合作为单 GPU 直接执行指令** |
| 权重读取 | `imagenet/misc.py::load_old_ckpt` | 存在**按参数遍历顺序+形状拷贝**的旧权重加载逻辑，需要额外核验 |

已核对 `Token_QK_Attention.forward` 结构（以 ImageNet / Event 的已检查版本为例）：

```python
# q: [T, B, num_heads, D_head, N]，已是 0/1 脉冲
q = torch.sum(q, dim=3, keepdim=True)  # [T, B, num_heads, 1, N]
attn = self.attn_lif(q)
x = torch.mul(attn, k)           # k: [T, B, num_heads, D_head, N]
```

门控神经元构造为：

```python
MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')
```

**警告**：上游项目的 `max_former` 主模型与 `ms_qkformer` 变体不是同一个模型；不能给 `max_former` 的卷积/SSA 模块强行替换 QKTA 门控。

## 1.3 已确认的预训练权重与资源门槛

公开发布页：https://github.com/bic-L/MaxFormer/releases/tag/checkpoints  
本项目首选权重：`ms_qk_10-384_T4.pth.tar`（约 63 MiB），对应 **ImageNet、MS-QKFormer、T=4**。

权重入口：https://github.com/bic-L/MaxFormer/releases/download/checkpoints/ms_qk_10-384_T4.pth.tar

必须核实 checkpoint 内部格式、参数键、尺寸、网络 `ms_qkformer_10_384` 的结构与时间步；并对比原版预测结果。**禁止**因为 `strict=False` 或按顺序拷贝没有报错，就假定权重加载成功。若存在名称差异，应创建显式键映射和对应关系报告；拒绝不明来源权重及未经验证的同形状误匹配。

截至本次检查，尚未确认发布了能直接加载到 `event/ms_qkformer.py` 的 CIFAR10-DVS 预训练权重。**ImageNet 权重不得直接当作事件模型权重。**

---

# 2. 数学动机、三种核心门控与对照条件

以下索引 `t,b,h,i` 分别表示时间、批次、注意力头和空间 Token。为突出机制，以下只写 `t,i`。

## 2.1 B0：原始 QKTA 门控及严格退化条件

令二值 Q 特征 `Q[t,i,d] ∈ {0,1}`，通道计数：

```text
q[t,i] = Σ_{d=1}^{D_head} Q[t,i,d]  ∈ {0,1,...,D_head}
```

设零初值 `u[-1]=0`，LIF 时间常数 `tau=2`、阈值 `theta0=0.5`、硬重置到 0：

```text
h[t] = 0.5*u[t-1] + 0.5*q[t]
A[t] = 1[h[t] >= 0.5]
u[t] = h[t]*(1-A[t])
```

由于 `q[t]` 是非负整数，若起始电位为 0，则每一步 `u[t]=0`，故：

```text
A_B0[t,i] = 1[q[t,i] >= 1] = OR_d Q[t,i,d]
```

**这是通道 OR 门，不是时间 OR 门。** 必须用符号推导、穷举测试及官方 `MultiStepLIFNode` 独立对照确认。限定条件之外（非整数输入、非零初始状态、改动阈值/重置、不同 LIF 实现）不得泛化结论。

## 2.2 B1：固定阈值对照（保留两种版本）

**B1-LIF**：保持标准 LIF 动力学，将阈值改为 `theta ∈ {0.5,1.0,1.5,2.0}`，可能因未发放电位积累引入真实时间依赖。  
**B1-Count**：纯无记忆门 `A[t]=1[q[t]>=k]`，`k∈{1,2,3,4}`，直接排除“仅提高通道脉冲计数阈值”这一平凡解释。

不得把 B1-LIF 称为“无记忆固定门”。

## 2.3 H：活动依赖的历史抑制

```text
h[t]       = 0.5*u[t-1] + 0.5*q[t]
theta_H[t] = theta0 + b*R[t]
A_H[t]     = 1[h[t] >= theta_H[t]]
u[t]       = h[t]*(1-A_H[t])
R[t+1]     = rho*R[t] + (1-rho)*A_H[t]
```

初值 `R[0]=0`；首轮 `rho=0.5`；抑制参数 `b>=0`。 `R[t]` 是当前时间步之前的历史门控活动，**要在计算完 A[t] 后更新**，避免偷看当前输出。

机制假设：近期频繁开启的 Token 需要更强的证据才能持续放行，可能减少重复脉冲，但也可能伤害持续有效目标。

## 2.4 C：空间对比条件化历史抑制

设二维 Token 网格为 `(H,W)`，每个位置有 8 邻域 `N8(i)`（不包含中心）。邻域计数按边界实际像素数归一：

```text
q_bar[t,i] = sum_{j in N8(i)} q[t,j] / |N8(i)|
c[t,i]     = 1[q[t,i] - q_bar[t,i] >= kappa]
theta_C[t,i] = theta0 + b*R[t,i]*(1-eta*c[t,i])
A_C[t,i]   = 1[h[t,i] >= theta_C[t,i]]
```

推荐约束 `eta ∈ [0,1]`、`kappa>=0`；首次实验固定 `theta0=0.5, rho=0.5, eta=0.5, kappa=1.0`，再做受限扫描。`c=0` 时与 H 完全相同；`c=1` 时减小历史惩罚。

**明确边界**：高对比噪声也可能受到保护；低对比目标可能被压制；静止物体也可能被历史门误抑制。本方法不自带目标/背景语义标签。

## 2.5 P：随机空间置换负对照

使用 C 的参数与历史状态更新，但将每个 `(t,b,head)` 的 `c[t]` 空间位置随机置换，再构造阈值。要求：

- 只置换 **c 掩码**，不同时置换 Q/K、真实图像、标签或模型特征。
- 每次置换应有显式随机种子；重复至少 3 个置换种子，并汇总均值和范围。
- 保持掩码激活比例不变，但破坏局部对比与空间位置的对应关系。
- 只对**事先在校准集选定**的 C 配置开展正式 P 对照；不允许在测试集上挑选最有利的随机置换。

P 的性能下降只能支持“空间对应关系对这一实验有贡献”，不能单独证明“已经学到真正目标语义”。

---

# 3. 工程设计与模块接口（必须遵循）

## 3.1 修改范围与建议文件

请优先创建独立目录，不大范围重写上游项目：

```text
<repo>/
  research_qkta/
    __init__.py
    gates.py                 # B0 / B1-count / H / C / P 的统一纯 PyTorch 实现
    adapter.py               # QKTA 小范围接入器，保留原网络张量契约
    telemetry.py             # 逐层门控、K 脉冲和分布统计
    synthetic_probe.py       # 合成输入和标注生成
    checkpoint_audit.py      # 权重格式与参数映射审计
    frozen_eval.py           # 冻结推理，数据采样，确定性运行
    sweep.py                 # 固定实验网格及结果输出
    plots.py                 # 图像生成，所有图标注明确指标
  tests/
    test_qkta_or_equivalence.py
    test_gate_shape_and_boundary.py
    test_gate_temporal_state.py
    test_gate_integration.py
    test_checkpoint_safety.py
  configs/research_qkta/
    synthetic.yaml
    imagenet_ms_qk_t4.yaml
    event_ms_qk_t16.yaml
  reports/
    00_repository_environment.md
    01_math_and_tests.md
    02_checkpoint_and_baseline.md
    03_synthetic_results.md
    04_frozen_results.md
    05_conclusion_limits.md
  outputs/qkta/          # 生成文件，不纳入 Git（可保留轻量总结）
```

目录名称可微调，但应保留功能解耦。**不要求 50 行完成科研级验证**；优先完整正确、可复现、可检查。

## 3.2 QKTA 精确输入输出契约

上游 QKTA 在 `mixer_hub.py` 中经过 Q 分支和求和后：

```text
q_sum : [T, B, heads, 1, N]
K     : [T, B, heads, D_head, N]
H*W   : N
A     : [T, B, heads, 1, N]
A*K   : [T, B, heads, D_head, N]
```

新增门控实现推荐接受 `q_sum`、`H`、`W` 与可选 `k`，输出 **与原 `attn_lif(q_sum)` 完全一致的张量形状、数值类型与设备**。从输入的原始 `x.shape` 读取 H/W，不允许由 `sqrt(N)` 猜测；虽然许多阶段是方形特征图，接口必须支持非方形 H/W。

当前代码示例中把五维 `q_sum=[T,B,heads,1,N]` 的最后两个维度误认成 H/W 的写法是错误的。正确的逻辑应当显式去掉单元素通道维：

```python
q_count = q_sum.squeeze(3)              # [T,B,heads,N]
A = gate(q_count, H=H, W=W)              # [T,B,heads,N]
attn = A.unsqueeze(3)                   # [T,B,heads,1,N]
# k: [T,B,heads,D_head,N]
post_k = attn * k
```

所有额外统计采用明确形状断言，不得依赖广播碰运气。若创建 wrapper，严禁把 `attn_lif` 的模块对象直接替换成只能接收 `(q,k,H,W)` 的不兼容接口；应在 `Token_QK_Attention.forward` 的局部调用点显式传参，或编写保持旧接口的上下文适配器。

## 3.3 邻域计算和时间状态

- 使用 `torch.nn.functional.conv2d` 实现八邻域计数，可固定卷积核，不引入新的可学习卷积权重。
- 对角点/边缘/内部按 `3 / 5 / 8` 个真实邻居归一；对于 `H*W=1` 的退化场景，明确约定 `c=0`，避免除零。
- 按每个 `t` 递推 `u` 和 `R`，但卷积等无状态计算可沿 `[B*heads,H,W]` 批量执行。
- 两次独立样本或独立实验配置之间，状态必须清零。若所有状态在 gate.forward 内局部创建，说明这一设计并确保统计计数器不泄露上一轮数据。
- 图像推理前后调用适当的 `functional.reset_net(model)`；不能跨 Batch 累积膜电位。
- B0 对照优先使用**上游原始 `attn_lif` 路径**；自定义 B0 仅用于单元测试和辅助检查。
- 不修改 Q/K 投影层、SSA、Patch Embedding、残差、MLP 和分类头；首阶段也不做自动化模型搜索。

## 3.4 状态、统计与额外开销

收集每层、每配置下：

```text
q_count_histogram
q_nonzero_fraction
A_open_count, A_total_count, A_open_rate
K_pre_spikes  = sum(K)
K_post_spikes = sum(A*K)
K_retention_vs_own_input = K_post_spikes / max(K_pre_spikes, 1)
K_post_vs_B0 = K_post_variant / max(K_post_B0, 1)
per_step_open_rate, per_step_post_k
per_head_open_rate
```

结果需同时提供原始分子、分母。对于多层替换，早期门控会影响后续层的 K 分布，因此 `K_post / K_pre` 与 `K_post / K_post_B0` 不能互换解释。另记录模型端到端延迟（含新增邻域计算），如有条件记录 GPU 峰值显存。

`K_post_spikes` 只代表局部门后脉冲活动预算，不得作为完整模型功耗或硬件 mJ。

---

# 4. 分阶段开发计划与阻断验收

## P0｜仓库、环境、数据与权重可行性审计

**目标：确认能够正确运行哪个模型、在哪种数据上运行、权重能否安全加载。**

执行顺序：

1. 读取本地三个 `mixer_hub.py` 中 `Token_QK_Attention` 的真实实现与差异；标出 Q、K 和 A 的张量维度，检查正确的层路径。
2. 检查 `event/ms_qkformer.py`、`imagenet/ms_qkformer.py` 和对应 YAML；确认 `model` 类型与 T。
3. 使用独立环境安装依赖；不要为了兼容过时版本直接覆盖用户已有全局 Torch/CUDA 环境。仓库 root `requirements.txt` 含 `timm==0.6.12`、`spikingjelly==0.0.0.0.12`、`numpy<2` 等约束。验证 CuPy 与 CUDA 的兼容性，再决定具体安装组合。
4. 审计 `/path/to/checkpoint`：文件存在性、SHA256、内部字段、参数数量、键名与形状；禁止直接使用危险的按顺序沉默拷贝。
5. 审查 ImageNet 测试路径：`imagenet/test.py` 的 `global_rank` 在单进程路径可能未初始化，默认配置路径 `imagenet.yml` 在仓库根下并不存在；官方 `test.sh` 采用 8 卡设置。若有实际错误，仅在研究专用推理入口作最小修补，不重构训练脚本。
6. 检查 ImageNet 验证集是否存在、是否为完整 1000 类 `ImageFolder` 类目映射；不可用随意挑选的少数类别目录重新编号再声称 ImageNet Top-1。

**P0 输出：** `reports/00_repository_environment.md`、checkpoint 审计报告（若有权重）、可执行环境说明。

**P0 验收：** 代码能导入且 `MS_QKFormer` 构造成功；checkpoint 键、T、类别数、预处理对应关系有明确结论。若无数据或权重，记录阻断并进入 P1/P2，不伪造 P3/P4 性能。

## P1｜OR 退化的数学证明与 CPU 单元测试

1. 提供一页归纳证明，说明 `tau=2, v_reset=0, theta0=0.5, q∈N_0, u0=0` 下 `A=1[q>=1]`。
2. 对 `D_head ∈ {4,8,16,32}`、`T ∈ {1,2,4,16}` 及随机多批次整数输入做穷举/随机测试。
3. 与 SpikingJelly `MultiStepLIFNode` 实际输出做逐元素对照（可用 CPU backend；若版本只允许 CuPy，保留独立纯 Python LIF 真值机并在可用 CUDA 时补实装对照）。
4. 检查零输入、全一输入、边界阈值、不同 Batch、跨样本 reset。记录特殊情况下为何不能声称 OR 等价。

**P1 验收：** 合法条件下 OR 等价率 100%；代码与数学证明一致；`pytest` 无失败。

## P2｜合成动态视觉探针（无需预训练权重）

构造可复现、带 ground-truth 区域标注的 `16×16`、`T=10` 或可配置样例。以 **Q 通道脉冲计数图**（或生成合法二值 Q 后再求和）为输入，而非直接把未经处理的像素当成 QKTA 的 q。

至少生成下列六种模式，固定种子，并展示每个位置的 q、邻域平均、c、R、阈值和门控 A 的变化：

1. 低对比且长期反复出现的背景纹理（期望历史抑制有效）。
2. 高对比的移动轮廓或目标边缘（观察空间保护）。
3. **高对比背景闪烁噪声**（C 可能误保护，应如实展示）。
4. **低对比真实目标**（C 可能过度抑制）。
5. 高对比但长期静止的目标（检验历史 H 误杀和 C 的保护条件）。
6. 稀疏事件/短暂遮挡后再出现的目标（检验 R 的衰减和恢复）。

分别运行 B0、B1-Count、B1-LIF、H、C、P。可视化每个时间步的输入、目标掩码、门控掩码及抑制区域；并给出：

- 目标区域门控保留率 `TPR_gate = sum(A*GT)/sum(GT)`；
- 背景放行率 `FPR_gate = sum(A*(1-GT))/sum(1-GT)`；
- 每步门控开启率、历史迹 R 和动态阈值；
- C 对 H 的改进/退化及失败案例。

**P2 验收：** 所有图均来自可复现数据，附随机种子、配置、分子分母。不得把对机制有利的手工构造条件当作现实视觉去噪结论。

## P3｜真实模型兼容性与原版冻结基线（有权重后执行）

**优先路线：ImageNet MS-QKFormer + 官方 `ms_qk_10-384_T4.pth.tar`。**

1. 在研究入口实例化 `ms_qkformer_10_384`，并显式设置 `T=4`、`num_classes=1000`、`in_channels=3`。不能实例化 `maxformer_10_384` 后强行加载 MS-QKFormer 权重。
2. 安全加载 checkpoint 并打印匹配键数量、缺失键、意外键、形状不一致项，严格评估权重是否完整可信；遇到同名不匹配或大规模漏载则立即停止正式准确率实验。
3. 固定数据预处理（ImageNet 224 中心裁剪与标准化以仓库实际实现为准）；先跑小批量 smoke test，再执行适当数量的校准/评估子集。
4. `model.eval()`、`torch.no_grad()`；每个样本 Batch 之后严格 reset。验证相同样本两次输出一致、不同评估顺序不会改变结果。
5. 原版 QKTA 的输出/掩码/分类分数保存作基线。**不改动门控，先得到可用的真实模型预测。**

**关键注意事项：**

- ImageNet 验证集不是仓库自带资源；若不可用，说明真实准确率实验暂时受阻。
- 500 条样本的 Top-1 仅用于探索；不能写成官方完整 ImageNet Top-1。
- 验证集子集需固定样本 ID / 索引、类别映射与随机种子。调参子集与最终报告子集必须不同。
- `imagenet/misc.py::load_old_ckpt` 按顺序匹配参数，存在同形状错误映射风险；优先通过严格键映射加载或经过检查的显式转换，不要把“加载成功”当作“权重正确”。

**P3 验收：** 完整的权重审计、确定性原版推理、可复现 baseline 输出与数据样本列表。任何条件失败时，仅保留“随机权重端到端 shape smoke test”，不得报告分类改进。

## P4｜冻结推理门控消融与帕累托图（有 P3 后执行）

**单层预注册范围：** ImageNet `stage1[0].attn` 的 QKTA；通过运行时模块类型打印确认实际路径。  
**多层预注册范围：** ImageNet `stage1[0].attn + stage2[0].attn + stage2[1].attn`（需实际确认）。  
事件模型仅能修改 `stage1[0].attn`，其 Stage 2 的 SSA 不属于当前 q 计数门控。

固定原模型、数据子集、权重、T、批大小及采样顺序，执行：

| 组别 | 参数/定义 | 目的 |
| --- | --- | --- |
| B0-original | 官方 `attn_lif` 不替换 | 真正的冻结基线 |
| B0-custom | `b=0,theta0=0.5` | 验证自定义 gate 对 B0 逐元素等价 |
| B1-LIF | `theta={0.5,1.0,1.5,2.0}` | 不加空间条件的阈值控制 |
| B1-Count | `k={1,2,3,4}` | 无状态计数阈值控制 |
| H | `eta=0,b={0.5,1,2,3,4}` | 单纯历史抑制 |
| C | `eta=0.5,kappa=1.0,b={0.5,1,2,3,4}` | 历史抑制 + 空间对比保护 |
| P | 固定 C 校准得到的参数，空间置换 c | 破坏空间对应关系负对照 |

可以根据 P2 的活动范围预先调整参数网格，但**必须在查看最终测试子集结果之前记录调整理由**。为减少模型运行成本，先在小型校准集扫描，选定有限候选，再在独立评估集统一计算；无须在所有候选上反复跑完整测试集。

**必须输出：**

1. `accuracy_vs_postK_ratio.png`（横轴明确为门后 K 脉冲保留率，绝不标成能耗比）；
2. `per_step_gate_rate.png`（各时间步 B0/H/C 的活动变化）；
3. `mask_examples.png`（预先固定样本，不只展示成功样本）；
4. `controls_and_failure_cases.png`（固定阈值、空间置换与失败例）；
5. `frozen_results.csv`、`frozen_results.json`（含 run_id、SHA、checkpoint hash、数据索引、参数、指标）；
6. `reports/04_frozen_results.md`（方法、结果、失败/阻断、限制）。

**主要判据：** 在近似相同 `K_post / K_post_B0` 下，C 是否比 H 与 B1 保持更好的分类表现。若只有一层门控变化且分类变化很小，也如实记录；不通过事后挑选层位、样本或 b 值“制造显著差异”。

**P4 验收：** 每条曲线有实际 CSV 来源；所有配置可重跑；P 重复多种子；单层和多层比较独立；明确陈述是否看到性能收益、只看到稀疏性变化，还是没有效果。

---

# 5. 必须完成的测试清单

- [ ] B0 退化证明与官方 LIF 实际值一致；`B0-original` 与 `B0-custom` 的 `A` 逐元素完全相同。
- [ ] B0 在模型上的 logits 一致或在明确数值精度容忍范围内一致，记录最大差值与设置。
- [ ] `q_sum` 五维张量不被错误解析为 H×W；能够支持非方形特征图。
- [ ] `A`、`K` 乘法的广播尺寸正确，且门控输出仍是二值。
- [ ] 角点邻域计数 3、普通边缘 5、内部 8；对于 1×1 网格没有 NaN。
- [ ] `R[t]` 在 A[t] 后更新，初始时间步 `R=0`；样本/批次间状态不串扰。
- [ ] 令 `eta=0`，C 的输出与同参数 H 完全一致；`b=0,theta0=0.5` 与原 OR 一致。
- [ ] 按不同 T 运行、重复同一输入、交换 Batch 顺序；输出满足预期的确定性和重置条件。
- [ ] P 置换后 `sum(c)` 不变、位置关系打乱且固定种子可复现。
- [ ] 无 checkpoint 时程序禁止展示“真实准确率”；权重加载失败时返回明确错误。
- [ ] 统计 `A_open`、`K_pre`、`K_post` 与汇总指标一致，无错误广播或层重复累加。
- [ ] 推理模式禁用训练梯度；日志保留当前 HEAD、所有随机种子、T、设备、预处理、权重哈希。
- [ ] `.gitignore` 排除大型 checkpoint、数据集、原始中间 Tensor 和图像/CSV 大量输出。

**不必强行要求**对所有 b 的 A 开启率逐步单调下降：有膜电位积累和 R 的状态递推，存在非单调的可能，需要用数据说明。

---

# 6. 研究证据的呈现规范

## 6.1 三类证据不能混合

| 证据 | 可得出的结论 | 不可得出的结论 |
| --- | --- | --- |
| 数学/单元测试 | 原门在特定条件下等价 OR，新机制递推实现正确 | 新机制一定提高视觉精度 |
| 合成探针 | 新机制对可控动态模式具有某些保护/抑制行为 | 在真实视觉场景已经具有泛化能力 |
| 预训练冻结推理 | 在既有表征和权重下的精度—稀疏性初步权衡 | 训练后最终性能、新 SOTA、实际芯片能耗 |

## 6.2 结果图必须注明

- 数据集、样本数、模型变体、T、替换层、冻结权重文件及 SHA；
- 使用的是校准集、探索子集还是独立报告子集；
- 横轴为 **post-gate K spikes / B0 post-gate K spikes** 还是 **post-gate K / pre-gate K**；
- 纵轴为 Top-1 accuracy（真实有标签数据）还是 synthetic target retention（合成区域指标）；
- 如有准确率差异，报告绝对百分点（pp），不用相对百分比混淆；
- 推理延迟应包括对比卷积和状态更新开销，并与统计 hooks 带来的测量扰动区分。

## 6.3 负结果仍然有效

若 C 减少 K 脉冲却明显降精度，应检查：冻结权重分布漂移、局部对比与真实语义不一致、b/kappa 尺度、层位与残差掩盖。记录这些发现后，可将“低成本冻结机制筛查完成，端到端微调是后续工作”作为诚实结论；不能调出一张漂亮图就宣称验证成功。

---

# 7. Git、环境与交付标准

## 7.1 提交纪律

建立 `feat/qkta-contrast-history-probe` 分支，建议以下提交节点（名称可依实际改动调整）：

```text
chore(research): audit repo and baseline environment
feat(qkta): add history and contrast-conditioned gates
test(qkta): verify OR equivalence, state reset and boundary shapes
feat(experiments): add synthetic probes and metrics
feat(eval): add checkpoint audit and frozen inference
 docs(research): add ablation protocol, results and limitations
```

实际提交前运行 `git diff --check`、静态检查、相关 `pytest`，再核查 `git status`；确认没有额外修改预训练权重、数据集、第三方依赖文件。远程存在且获授权时：

```bash
git push -u origin feat/qkta-contrast-history-probe
```

不要在未经确认时推到 `main`；不要使用 `--force`；如果远端权限不通，保留本地提交并明确说明未推送。

## 7.2 环境策略

推荐创建独立 Conda/venv。优先复用仓库列出的依赖约束，但必须匹配当前 CUDA 驱动与 GPU；不能机械安装不兼容的 CuPy。仅为 CPU 合成测试时，不要求安装 CUDA/CuPy。实际环境版本与安装错误写入报告，拒绝“临时降级系统环境”破坏其他工程。

## 7.3 输出交付

最终提供：

1. 改动总览和代码关键 diff；
2. 具体运行与重现命令（包括工作目录、配置路径和权重路径占位）；
3. P0–P4 每阶段 PASS/FAIL/BLOCKED 状态与原因；
4. 所有测试命令及摘要、报告、图表与 CSV 路径；
5. 未解决问题、科研限制与后续轻量微调建议；
6. 当前 Git 分支和 HEAD、已创建的 commits、推送到的远端分支地址（若成功）。

---

# 8. 分阶段给开发智能体的简短指令（建议逐条发送）

## Prompt A — 首次执行：仅 P0 + P1

> 请在当前已克隆的 `Tuzfucius/MaxFormer` 仓库中，严格遵照《MaxFormer / MS-QKFormer 二次开发：历史抑制与局部对比保护门控》协议执行 **P0 仓库审计 + P1 原 OR 退化验证**。先检查 Git、依赖、QKTA 的实际路径及形状；建立功能分支，不改训练脚本；实现可在 CPU 运行的 OR/IF 真值验证和 pytest；生成报告并完成阶段 commit，远端授权可用时推送。暂不开发复杂可视化或进行模型训练。汇报所有代码差异、运行结果和阻断条件。

## Prompt B — 第二次执行：P2 合成探针 + 门控实现

> 在 P0/P1 通过的前提下，按协议实现独立、可复用的 B0、B1、H、C、P 门控及六种合成时序模式，特别检查五维张量接口、空间邻居归一化、历史状态更新次序和 `eta=0` 等价性。输出输入/目标/门控/历史迹可视化和可复算指标；加入反例，不允许预设 C 必然获胜。保留原版运行路径，完成相关测试、commit 和获授权后的 push。

## Prompt C — 第三次执行：P3 原模型冻结基线

> 在不训练的前提下核验 ImageNet MS-QKFormer `ms_qkformer_10_384` 与 `ms_qk_10-384_T4.pth.tar` 的匹配关系，检查 `imagenet/misc.py` 的顺序拷贝风险和测试脚本的单卡问题。若 ImageNet val 与可信权重均可用，做原版冻结推理、输出一致性与模型状态重置测试；否则仅完成随机权重前向 smoke test，清楚报告 BLOCKED。禁止“权重部分加载后”计算并宣称真实模型准确率。提交报告与必要最小修复，并推送功能分支（若授权）。

## Prompt D — 第四次执行：P4 消融

> 仅在 P3 基线验收通过后，冻结相同权重、固定相同样本与 T，先测试单个 QKTA，再测试 ImageNet Stage 1 + Stage 2 全部 QKTA。预注册运行 B0-original/B0-custom/B1-LIF/B1-Count/H/C/P，不做反向传播。用校准子集选参数，在未参与选参的子集上报告 Top-1、门后 K 脉冲预算、时间步发放率、局部/端到端延迟以及失败样本。制作数据可追溯的帕累托曲线和门控图；说明结果边界，运行测试、提交并推送。

---

# 9. 关键参考资料与阅读顺序

1. **项目 Fork（本次实际核查）**：https://github.com/Tuzfucius/MaxFormer
2. **上游仓库与权重**：https://github.com/bic-L/MaxFormer ；https://github.com/bic-L/MaxFormer/releases/tag/checkpoints
3. **MaxFormer 论文**：*Spiking Neural Networks Need High-Frequency Information*, NeurIPS 2025，https://proceedings.neurips.cc/paper_files/paper/2025/hash/956834836f36dd07df7064ff42ca69f2-Abstract-Conference.html
4. **QKFormer 论文**：*QKFormer: Hierarchical Spiking Transformer using Q-K Attention*, NeurIPS 2024，https://proceedings.neurips.cc/paper_files/paper/2024/hash/179f5dcdeedc149443ebd3ba70811dbd-Abstract-Conference.html
5. **神经适应依据**：*Advancing spatio-temporal processing through adaptation in spiking neural networks*, Nature Communications 2025，https://www.nature.com/articles/s41467-025-60878-z
6. **侧抑制视觉注意力**：*SpiLiFormer*, ICCV 2025，https://openaccess.thecvf.com/content/ICCV2025/html/Zheng_SpiLiFormer_Enhancing_Spiking_Transformers_with_Lateral_Inhibition_ICCV_2025_paper.html
7. **时空注意力参照**：*Spiking Transformer with Spatial-Temporal Attention*, CVPR 2025，https://openaccess.thecvf.com/content/CVPR2025/html/Lee_Spiking_Transformer_with_Spatial-Temporal_Attention_CVPR_2025_paper.html

**引用规范**：论文给出设计启发，不代表论文已经实现本项目提出的条件历史门控；自己的公式、实验指标和局限分析必须明确标注“本研究设计/假设”。

---

# 10. 最终科研决策标准

核心判据：在同等门后 K 脉冲预算下，新门控 C 是否比 H 和固定阈值保持更高准确率？

- **如果是**：进入小规模可学习参数/有限 epoch 微调、多个数据集及模型验证。
- **如果仅减少脉冲但精度下降**：分析是否为冻结权重分布漂移，进行有限微调前不要放弃方案，也不能掩盖负结果。
- **如果 C 与 H 无显著差异**：检查局部对比标志 `c` 实际激活率、邻域统计和 `eta` 的有效范围；必要时放弃空间条件化部分。
- **如果只能完成合成实验**：明确作为“机制可行性初筛”，不得宣称得到网络分类精度收益。

**本阶段不追求超越 SOTA；优先得到可复核的数学证据、运行正确的模块、诚实的冻结消融与清晰的下一步研究方向。**
