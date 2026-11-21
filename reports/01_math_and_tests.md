# P1 Mathematical Proof: Channel-OR Degeneracy of Default QKTA `attn_lif`

## 1. Setup & Notation

For any fixed batch element $b$, attention head $h$, and spatial token $i \in \{1, \dots, N\}$, let $Q[t, d] \in \{0, 1\}$ denote the binary query spike at time step $t \in \mathbb{N}_0 = \{0, 1, 2, \dots\}$ and head-channel $d \in \{1, \dots, D_{\text{head}}\}$. The channel-summed input to `Token_QK_Attention.attn_lif` is:
$$q[t] = \sum_{d=1}^{D_{\text{head}}} Q[t, d] \in \{0, 1, \dots, D_{\text{head}}\} \subset \mathbb{N}_0.$$

In upstream `Token_QK_Attention` (`mixer_hub.py`), `attn_lif` is a `MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True)` with default hard reset $v_{\text{reset}} = 0.0$ and zero initial membrane potential $u[-1] = 0.0$. Its discrete-time dynamics for $t \ge 0$ are:
$$h[t] = \left(1 - \frac{1}{\tau}\right) u[t-1] + \frac{1}{\tau} q[t] = \frac{1}{2} u[t-1] + \frac{1}{2} q[t],$$
$$A[t] = \mathbb{1}\!\left[h[t] \ge \theta_0\right] = \mathbb{1}\!\left[h[t] \ge \frac{1}{2}\right],$$
$$u[t] = h[t]\,(1 - A[t]) + v_{\text{reset}}\,A[t] = h[t]\,(1 - A[t]).$$

---

## 2. Theorem & Proof by Mathematical Induction

**Theorem (Channel-OR Degeneracy).** *Under $\tau = 2.0$, $\theta_0 = 0.5$, hard reset $v_{\text{reset}} = 0.0$, initial state $u[-1] = 0.0$, and non-negative integer inputs $q[t] \in \mathbb{N}_0$ for all $t \ge 0$, the following holds for every $t \ge 0$:*
1. $u[t] = 0.0$, and
2. $A[t] = \mathbb{1}[q[t] \ge 1] = \bigvee_{d=1}^{D_{\text{head}}} Q[t, d]$.

**Proof.** We prove $u[t] = 0$ and $A[t] = \mathbb{1}[q[t] \ge 1]$ for all $t \ge 0$ by strong/standard induction on $t$.

- **Base Case ($t = 0$):** Since $u[-1] = 0$, we have:
  $$h[0] = \frac{1}{2}(0) + \frac{1}{2} q[0] = \frac{q[0]}{2}.$$
  Because $q[0] \in \mathbb{N}_0$, there are two mutually exclusive cases:
  - *Case 1 ($q[0] = 0$):* $h[0] = 0 < 0.5$, so $A[0] = 0$ and $u[0] = h[0](1 - 0) = 0$. Also $\mathbb{1}[q[0] \ge 1] = 0$.
  - *Case 2 ($q[0] \ge 1$):* $h[0] = \frac{q[0]}{2} \ge \frac{1}{2} = \theta_0$, so $A[0] = 1$ and $u[0] = h[0](1 - 1) = 0$. Also $\mathbb{1}[q[0] \ge 1] = 1$.
  Thus both claims hold at $t = 0$.

- **Inductive Step:** Assume $u[t-1] = 0$ for some $t \ge 1$. Then at step $t$:
  $$h[t] = \frac{1}{2} u[t-1] + \frac{1}{2} q[t] = \frac{q[t]}{2}.$$
  By the exact same case split on $q[t] \in \mathbb{N}_0$:
  - If $q[t] = 0$, then $h[t] = 0 < 0.5 \implies A[t] = 0$ and $u[t] = 0 \cdot 1 = 0 = \mathbb{1}[q[t] \ge 1]$.
  - If $q[t] \ge 1$, then $h[t] \ge 0.5 \implies A[t] = 1$ and $u[t] = h[t] \cdot 0 = 0$, matching $\mathbb{1}[q[t] \ge 1] = 1$.

By induction, $u[t] = 0$ and $A[t] = \mathbb{1}[q[t] \ge 1]$ for all $t \ge 0$. Finally, since $Q[t, d] \in \{0, 1\}$, $\sum_{d=1}^{D_{\text{head}}} Q[t, d] \ge 1 \iff \exists\, d \text{ s.t. } Q[t, d] = 1 \iff \bigvee_{d=1}^{D_{\text{head}}} Q[t, d] = 1$. $\blacksquare$

---

## 3. Channel OR Gate vs. Temporal OR Gate

It is critical to distinguish **channel OR** from **temporal OR**:
- **Channel OR (what `B0` actually computes):** $A[t, i] = \bigvee_{d=1}^{D_{\text{head}}} Q[t, i, d]$ depends **only on the current time step $t$**. Because $u[t] \equiv 0$ at every step, no membrane potential is ever carried from $t-1$ to $t$. If $q[t, i] = 0$ at step $t$, then $A[t, i] = 0$ regardless of whether $q[t-1, i] > 0$.
- **Temporal OR (what `B0` does NOT compute):** A temporal OR gate would output $A[t, i] = \bigvee_{s=0}^{t} \bigvee_{d=1}^{D_{\text{head}}} Q[s, i, d]$, remaining open forever after any past spike. `B0` is **memoryless** across time, not a temporal OR.

---

## 4. Non-Equivalence Regimes (Boundary Counterexamples)

The Channel-OR equivalence **breaks** whenever any of the theorem's hypotheses is violated:

| Violated Hypothesis | Mechanism of Breakdown | Concrete Counterexample |
| --- | --- | --- |
| **Non-integer $q[t]$** (e.g., $q[t] \in (0, 1)$) | $0 < q[t] < 1 \implies 0 < h[t] < 0.5$, so $A[t]=0$ and $u[t] = h[t] > 0$ (subthreshold accumulation). | $q = [0.6, 0.6]$: $h[0]=0.3 \implies A[0]=0, u[0]=0.3$; $h[1]=0.15+0.3=0.45 < 0.5 \implies A[1]=0$ (or $q=[0.8, 0.8] \implies h[0]=0.4, A[0]=0; h[1]=0.2+0.4=0.6 \ge 0.5 \implies A[1]=1$). |
| **Non-zero initial state $u[-1] > 0$** | Residual membrane potential can trigger $A[0]=1$ even when $q[0] = 0$ (or accumulate with $q$). | $u[-1] = 1.0, q[0] = 0 \implies h[0] = 0.5 \ge 0.5 \implies A[0] = 1 \ne \mathbb{1}[0 \ge 1]$. |
| **Threshold $\theta_0 \ne 0.5$** | If $\theta_0 > 0.5$ (e.g. `B1-LIF` with $\theta_0 = 1.0$), $q[0] = 1$ gives $h[0] = 0.5 < 1.0 \implies A[0]=0, u[0]=0.5 > 0$, introducing temporal accumulation. If $\theta_0 \le 0$, $q[0]=0 \implies A[0]=1$. | $\theta_0 = 1.0, q = [1, 1]$: $h[0]=0.5 \implies A[0]=0, u[0]=0.5$; $h[1]=0.25+0.5=0.75 < 1.0 \implies A[1]=0$. For $q=[1, 2]$: $A[0]=0, u[0]=0.5$; $h[1]=0.25+1.0=1.25 \ge 1.0 \implies A[1]=1$. |
| **Time constant $\tau \ne 2.0$** | $h[0] = q[0]/\tau$. If $\tau > 2$ (e.g. $\tau = 4$), $q[0]=1 \implies h[0]=0.25 < 0.5 \implies A[0]=0, u[0]=0.25 > 0$. If $\tau < 2$ (e.g. $\tau = 1.5$), $q[0]=1 \implies h[0]=2/3 \ge 0.5 \implies A[0]=1, u[0]=0$, still OR for $q \in \mathbb{N}_0$ unless $\theta_0$ changes, but for $\tau > 2$ it breaks. | $\tau = 4.0, \theta_0 = 0.5, q = [1, 1]$: $h[0]=0.25 \implies A[0]=0, u[0]=0.25$; $h[1]=0.75(0.25)+0.25 = 0.4375 < 0.5 \implies A[1]=0$. |
| **Soft reset ($v_{\text{reset}} = \text{None}$)** | Post-spike potential becomes $u[t] = h[t] - \theta_0$ instead of $0$. When $q[t] \ge 2$, $u[t] = q[t]/2 - 0.5 \ge 0.5$, causing a spontaneous spike at $t+1$ even if $q[t+1] = 0$! | $v_{\text{reset}} = \text{None}, q = [2, 0]$: $h[0]=1.0 \ge 0.5 \implies A[0]=1, u[0]=1.0-0.5=0.5$; $h[1]=0.5(0.5)+0=0.25$ (for $q=[3,0]$: $u[0]=1.5-0.5=1.0 \implies h[1]=0.5 \ge 0.5 \implies A[1]=1 \ne 0$). |

---

## Measured Unit Test Results

### Full suite

- Successful command: `env -u PYTHONPATH PYTHONNOUSERSITE=1 conda run -n maxformer-qkta pytest -v -s tests/`.
- Pytest collected 45 tests; **45 passed, 0 skipped, 0 failed** in **2.54 s**. The timed wall clock, including Conda startup, was **4.26 s**.
- The CuPy Conda activation hooks were active; `tests/test_qkta_or_equivalence.py::test_optional_cupy_backend_matches_custom_gate` ran on CUDA and passed, so this test was not skipped.
- One test-side correction was needed in `tests/test_gate_shape_and_boundary.py`: `c_trace` has shape `[T, B, heads, N]`, so the corner assertion used one too many indices. It now indexes `[0, 0, 0, 0]`; the full suite then passed. No files in `research_qkta/` or `configs/` were changed.
- Warnings: **5 total**—four Pillow deprecations emitted through torchvision for `BILINEAR`, two `NEAREST`, and `BICUBIC`; one `SentryHubDeprecationWarning` from Weights & Biases' Sentry integration. No other warnings, skips, or failures were reported.
- QKFormer import check in `maxformer-qkta` succeeded with output `MS_QKFormer import OK`.

The first unisolated attempt, `conda run -n maxformer-qkta pytest -v -s tests/`, aborted before collection because the inherited ROS `PYTHONPATH` caused pytest to auto-load the host `launch_testing` plugin, which failed with `ModuleNotFoundError: No module named 'lark'`. Unsetting inherited `PYTHONPATH` and disabling user-site packages isolated the Conda environment; the successful command above still used `conda run`, so its CuPy activation hooks remained enabled.

### Measured B0 equivalence

`tests/test_gate_integration.py` printed the following `B0_original` versus `B0_custom` maximum absolute logit differences:

| Track | Target | Maximum absolute difference |
| --- | --- | ---: |
| ImageNet | `stage1` | `0` |
| ImageNet | `all_qkta` | `0` |
| CIFAR | `stage1` | `0` |
| CIFAR | `all_qkta` | `0` |

The maximum across all four comparisons was **0**.

### CPU and CUDA/CuPy equivalence

The full-model integration comparisons above ran on CPU with the torch backend and were exactly equal for B0. The optional CUDA CuPy LIF gate test also passed bit-for-bit against the custom gate (`torch.equal`); no CuPy skips occurred. The suite does not compare full-model CPU logits against full-model CUDA/CuPy logits, so no cross-device full-model equivalence claim is made.
