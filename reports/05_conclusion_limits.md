# P5 Conclusions, Evidence Boundaries, Limitations & Next Steps

## 1. Strict 3-Tier Evidence Boundary

| Evidence Tier | Source Reports & Artifacts | What It Proves | What It Does NOT Prove |
| --- | --- | --- | --- |
| **P1: Math & Unit Tests** | [`01_math_and_tests.md`](file:///home/tzufucius/Project/MaxFormer/reports/01_math_and_tests.md), `tests/test_*.py` (`45/45` passed) | Under `tau=2, theta0=0.5, v_reset=0, u[-1]=0, q in N_0`, `B0` `attn_lif` is mathematically and bit-for-bit identical to memoryless Channel-OR (`1[q >= 1]`); `H`/`C`/`P` recurrences and `3/5/8` boundary norms are implemented correctly. | Does NOT prove that any modified gate improves vision accuracy or sparsity on real models. |
| **P2: Synthetic Probes** | [`03_synthetic_results.md`](file:///home/tzufucius/Project/MaxFormer/reports/03_synthetic_results.md), [`synthetic_results.csv`](file:///home/tzufucius/Project/MaxFormer/outputs/qkta/synthetic/synthetic_results.csv) | `C` protects compact high-contrast static/occluded targets (`+0.2750` TPR over `H` at `b=3,4`), `R` decays during occlusion, and `C` fails predictably on `1x1` flickering noise and large low-contrast targets. | Does NOT prove generalization on real natural images or learned feature maps. |
| **P3–P4: Dual-Track Frozen Vision** | [`02_checkpoint_and_baseline.md`](file:///home/tzufucius/Project/MaxFormer/reports/02_checkpoint_and_baseline.md), [`04_frozen_results.md`](file:///home/tzufucius/Project/MaxFormer/reports/04_frozen_results.md) | `B0_original == B0_custom` (`0.0` logit diff) on real networks; raising thresholds (`B1_lif`, `B1_count`) cuts `K_post` to `0.22–0.48x` (Track 1) and `0.66–0.76x` (Track 2) with zero accuracy loss. | Does NOT prove `H + C` superiority, post-training SOTA, or hardware mJ energy savings. |

---

## 2. Scientific Verdict (Protocol §10)

1. **Channel-OR Degeneracy & Redundancy — CONFIRMED:** Both mathematical proof (`P1`) and real-network telemetry (`P3`: `q_nonzero_fraction == A_open_rate` to 9 decimal places) confirm that upstream `Token_QK_Attention.attn_lif` operates strictly as a memoryless Channel-OR gate (`1[q >= 1]`). Raising the threshold (`B1_count` / `B1_lif`) reveals substantial post-gate `K` spike redundancy (`63–78%` reduction on Track 1 `stage1` / `all_qkta` and `24–34%` on Track 2 with `>= 0` pp change).
2. **Spatial-Contrast History Protection (`H + C`) — NEGATIVE / NOT VALIDATED in Frozen `T=4` Static Inference:** Across both Track 1 (`n=2,925`) and Track 2 (`n=8,000`), **`C` does NOT consistently outperform `H`, 3-seed `P`, or simple fixed thresholds (`B1_count` / `B1_lif`)** at matched `K_post_vs_B0`. Instead, memoryless `B1_count` (`k=3, 4`) and `B1_lif` (`theta0=1.5, 2.0`) dominate the accuracy–sparsity Pareto frontier on both tracks.

---

## 3. Mechanistic Explanation of the Negative Result

1. **Short Horizon (`T = 4`) with Untouched `t = 0`:** Because `R[0] = 0`, `t = 0` is identical across `B0`, `H`, `C`, and `P`. Only `3` timesteps (`t = 1, 2, 3`) experience history inhibition, capping maximum `K_post` reduction to `~0.69–0.95x` while `B1` acts on all `4` timesteps (`0.22–0.76x`).
2. **Repeated Static Frames:** Both Track 1 and Track 2 repeat the same static image across `T = 4` steps (`x.unsqueeze(0).repeat(T, 1, 1, 1, 1)`). Unlike dynamic moving/occluded stimuli, static images do not produce transient motion contours where `H` and `C` diverge cleanly.
3. **Frozen OR-Trained Weights & Residual Bypasses:** Pretrained weights were optimized under `B0` (`1[q >= 1]`) and every `Token_QK_Attention` block has an identity residual (`out = out + identity`), buffering small `0.03–0.05` `A_open_rate` differences between `H`, `C`, and `P`.
4. **Weak Contrast Modulation (`eta = 0.5`) & Unaligned 8-Neighbor Feature Contrast:** With `eta = 0.5`, `theta_C` only halves the `b * R` penalty when `c = 1`, and 8-neighbor `q_count` contrast on downsampled feature maps does not cleanly separate semantic foreground from background clutter.

---

## 4. Limitations & Explicit Disclaimers

- **Local Spike Budget != Hardware Energy:** `K_post_spikes` / `K_post_vs_B0` measures local post-gate `K` spike counts only; it must **not** be reported as chip-level energy or mJ.
- **Track 1 Scope:** Imagenette2-320 (`3,925` images, `10` WNIDs mapped to `0..999` logits) is a 10-synset subset, **not** full 1000-class ImageNet-1K Top-1.
- **Track 2 Training Cap:** `cifar10_ms_qk_t4_baseline.pth` stopped cleanly at Epoch 74 (`91.68%` val acc) under the pre-registered `2.2h` cap (`3.3308h` total GPU across `164` runs in [`gpu_budget_log.json`](file:///home/tzufucius/Project/MaxFormer/outputs/qkta/gpu_budget_log.json)).
- **Untouched Upstream Event Bug:** `event/ms_qkformer.py` was not modified; upstream `Embed.forward(dual=...)` compatibility issues remain out of scope.
- **Synthetic vs Real:** `P2` synthetic results characterize gate mechanics only and do not imply real-world visual denoising.

---

## 5. Actionable Next Steps

1. **Gate-Aware Fine-Tuning / QAT of Fixed Thresholds (`B1_count` / `B1_lif`):** Since `B1_count (k=3,4)` and `B1_lif (theta0=1.5)` dominated frozen Pareto frontiers, evaluate lightweight QAT/fine-tuning with `B1` gates on full ImageNet-1K (`50,000` val images).
2. **Long-Horizon Dynamic Event / Video Evaluation (`T >= 8..16`):** Test `H` vs `C` vs `P` on genuinely dynamic temporal streams (e.g., CIFAR10-DVS / DVS128-Gesture after fixing upstream `event/ms_qkformer.py` and auditing `c_active_rate` distributions).
3. **Diagnostic `c`-Mask Saliency Audit:** If spatial contrast protection is revisited, measure whether `c` correlates with gradient/attention saliency or mostly fires on high-frequency feature-map noise.
