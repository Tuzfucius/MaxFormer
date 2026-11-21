# P1 数学证明：默认 QKTA `attn_lif` 的 Channel-OR 退化

## 1. 设置与符号

固定任意 batch 元素 $b$、注意力头 $h$ 和空间 token $i \in \{1, \dots, N\}$。令 $Q[t,d] \in \{0,1\}$ 表示时间步 $t \in \mathbb{N}_0$、头内通道 $d \in \{1, \dots, D_{\text{head}}\}$ 上的二值 query 脉冲。输入 `Token_QK_Attention.attn_lif` 的通道求和结果为：

$$q[t] = \sum_{d=1}^{D_{\text{head}}} Q[t,d] \in \{0,1,\dots,D_{\text{head}}\} \subset \mathbb{N}_0.$$

在上游 `Token_QK_Attention`（`mixer_hub.py`）中，`attn_lif` 是 `MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True)`；默认硬重置电位为 $v_{\text{reset}}=0$，初始膜电位为 $u[-1]=0$。对 $t\ge0$，离散时间动力学为：

$$h[t] = \left(1 - \frac{1}{\tau}\right) u[t-1] + \frac{1}{\tau} q[t] = \frac{1}{2}u[t-1] + \frac{1}{2}q[t],$$
$$A[t] = \mathbb{1}\!\left[h[t] \ge \theta_0\right] = \mathbb{1}\!\left[h[t] \ge \frac{1}{2}\right],$$
$$u[t] = h[t](1-A[t]) + v_{\text{reset}}A[t] = h[t](1-A[t]).$$

---

## 2. 定理与数学归纳证明

**定理（Channel-OR 退化）。** *当 $\tau=2.0$、$\theta_0=0.5$、硬重置 $v_{\text{reset}}=0$、初始状态 $u[-1]=0$，且对所有 $t\ge0$ 均有非负整数输入 $q[t]\in\mathbb{N}_0$ 时，对每个 $t\ge0$ 都成立：*

1. $u[t]=0$；
2. $A[t]=\mathbb{1}[q[t]\ge1]=\bigvee_{d=1}^{D_{\text{head}}} Q[t,d]$。

**证明。** 对 $t$ 作归纳，证明所有 $t\ge0$ 都有 $u[t]=0$ 且 $A[t]=\mathbb{1}[q[t]\ge1]$。

- **基础情形（$t=0$）：** 因为 $u[-1]=0$，所以 $h[0]=q[0]/2$。由于 $q[0]\in\mathbb{N}_0$，分两种情况：
  - 若 $q[0]=0$，则 $h[0]=0<0.5$，因此 $A[0]=0$、$u[0]=0$，且 $\mathbb{1}[q[0]\ge1]=0$。
  - 若 $q[0]\ge1$，则 $h[0]=q[0]/2\ge0.5=\theta_0$，因此 $A[0]=1$、硬重置后 $u[0]=0$，且 $\mathbb{1}[q[0]\ge1]=1$。
  两个结论在 $t=0$ 均成立。
- **归纳步骤：** 假设某个 $t\ge1$ 有 $u[t-1]=0$。则 $h[t]=q[t]/2$。对 $q[t]$ 作同样分类：若 $q[t]=0$，则 $A[t]=0$ 且 $u[t]=0$；若 $q[t]\ge1$，则 $A[t]=1$，硬重置后 $u[t]=0$。两种情况下都满足 $A[t]=\mathbb{1}[q[t]\ge1]$。

由归纳法，对所有 $t\ge0$，$u[t]=0$ 且 $A[t]=\mathbb{1}[q[t]\ge1]$。又因 $Q[t,d]\in\{0,1\}$，$\sum_d Q[t,d]\ge1$ 等价于存在某个 $d$ 使 $Q[t,d]=1$，也等价于 $\bigvee_d Q[t,d]=1$。证毕。$\blacksquare$

---

## 3. 通道 OR 门与时间 OR 门

必须区分**通道 OR**与**时间 OR**：

- **通道 OR（`B0` 实际计算的内容）：** $A[t,i]=\bigvee_{d=1}^{D_{\text{head}}} Q[t,i,d]$，只取决于**当前时间步 $t$**。由于每步都有 $u[t]\equiv0$，膜电位不会从 $t-1$ 延续至 $t$。若当前 $q[t,i]=0$，则无论上一时刻 $q[t-1,i]$ 是否大于零，当前 $A[t,i]=0$。
- **时间 OR（`B0` 并未计算的内容）：** 时间 OR 会输出 $A[t,i]=\bigvee_{s=0}^{t}\bigvee_d Q[s,i,d]$，过去任意时刻出现脉冲后便持续开启。`B0` 在时间维度上是**无记忆的**，不是时间 OR。

---

## 4. 等价性不成立的情形（边界反例）

定理的任一前提被破坏时，Channel-OR 等价性都可能不成立：

| 被违反的前提 | 失效机制 | 具体反例 |
| --- | --- | --- |
| **$q[t]$ 非整数**（如 $q[t]\in(0,1)$） | $0<q[t]<1$ 时 $0<h[t]<0.5$，故 $A[t]=0$ 且 $u[t]=h[t]>0$，会发生阈下累积。 | $q=[0.6,0.6]$：$h[0]=0.3$、$A[0]=0$、$u[0]=0.3$；$h[1]=0.45<0.5$、$A[1]=0$。若 $q=[0.8,0.8]$，则 $h[0]=0.4$，而 $h[1]=0.6\ge0.5$，第二步触发。 |
| **初态 $u[-1]>0$** | 残留膜电位可在 $q[0]=0$ 时直接触发 $A[0]=1$，或与输入累积。 | $u[-1]=1.0,q[0]=0$ 时，$h[0]=0.5\ge0.5$，故 $A[0]=1\ne\mathbb{1}[0\ge1]$。 |
| **阈值 $\theta_0\ne0.5$** | 若 $\theta_0>0.5$（如 `B1-LIF` 的 $\theta_0=1.0$），$q[0]=1$ 得到 $h[0]=0.5<1.0$，不发放并产生阈下累积；若 $\theta_0\le0$，即使 $q[0]=0$ 也会发放。 | $\theta_0=1.0,q=[1,1]$：$h[0]=0.5,A[0]=0,u[0]=0.5$；$h[1]=0.75<1.0,A[1]=0$。对 $q=[1,2]$，第二步 $h[1]=1.25\ge1.0$，故 $A[1]=1$。 |
| **时间常数 $\tau\ne2.0$** | $h[0]=q[0]/\tau$。若 $\tau>2$，$q[0]=1$ 时电位低于阈值并累积；若 $\tau<2$，在当前阈值下整数输入仍可保持 OR 等价性。 | $\tau=4.0,\theta_0=0.5,q=[1,1]$：$h[0]=0.25,A[0]=0,u[0]=0.25$；$h[1]=0.4375<0.5,A[1]=0$。 |
| **软重置（$v_{\text{reset}}=\text{None}$）** | 发放后电位改为 $u[t]=h[t]-\theta_0$ 而非归零。若 $q[t]\ge2$，残留电位可能让下一个零输入时间步自发放电。 | 对 $q=[3,0]$，$h[0]=1.5$、$A[0]=1$、$u[0]=1.0$；于是 $h[1]=0.5$、$A[1]=1\ne0$，尽管 $q[1]=0$。（$q=[2,0]$ 时第二步为 $0.25$，不会触发。） |

---

## 单元测试实测结果

### 完整测试套件

- 成功命令：`env -u PYTHONPATH PYTHONNOUSERSITE=1 conda run -n maxformer-qkta pytest -v -s tests/`。
- Pytest 收集 45 个测试；**45 通过、0 跳过、0 失败**，耗时 **2.54 秒**。包含 Conda 启动在内的实测总耗时为 **4.26 秒**。
- CuPy Conda 激活 hooks 已生效；`tests/test_qkta_or_equivalence.py::test_optional_cupy_backend_matches_custom_gate` 在 CUDA 上执行并通过，因此没有跳过该测试。
- `tests/test_gate_shape_and_boundary.py` 曾需修正一个测试索引：`c_trace` 形状为 `[T, B, heads, N]`，边界断言索引维度过多，已改为 `[0, 0, 0, 0]`；随后完整套件通过。未改动 `research_qkta/` 或 `configs/` 中的文件。
- 共 **5 条警告**：torchvision 经 Pillow 发出的 4 条弃用警告（`BILINEAR`、两个 `NEAREST`、`BICUBIC`），以及 Weights & Biases 的 Sentry 集成发出的 1 条 `SentryHubDeprecationWarning`。没有其他警告、跳过或失败。
- 在 `maxformer-qkta` 中导入 QKFormer 成功，输出 `MS_QKFormer import OK`。

第一次未隔离的 `conda run -n maxformer-qkta pytest -v -s tests/` 在测试收集前中止：继承的 ROS `PYTHONPATH` 使 pytest 自动加载主机的 `launch_testing` 插件，该插件因缺少 `lark` 而报 `ModuleNotFoundError`。清除继承的 `PYTHONPATH` 并禁用用户 site-packages 后，Conda 环境隔离成功；通过的命令仍使用 `conda run`，因此保留了 CuPy 激活 hooks。

### 实测 B0 等价性

`tests/test_gate_integration.py` 输出的 `B0_original` 与 `B0_custom` 最大绝对 logit 差异：

| 轨道 | 目标层 | 最大绝对差异 |
| --- | --- | ---: |
| ImageNet | `stage1` | `0` |
| ImageNet | `all_qkta` | `0` |
| CIFAR | `stage1` | `0` |
| CIFAR | `all_qkta` | `0` |

以上四项比较的最大差异为 **0**。

### CPU 与 CUDA/CuPy 等价性

上述全模型集成比较在 CPU 上使用 torch 后端运行，B0 结果完全相同。可选 CUDA CuPy LIF 门控测试也通过了与自定义门控的逐位比较（`torch.equal`），没有跳过 CuPy 测试。该测试套件没有比较全模型在 CPU 与 CUDA/CuPy 上的 logits，因此不据此声称全模型跨设备等价。
