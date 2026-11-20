# P4 Frozen Inference Ablation & Pareto Analysis

## 1. Protocol, Metric Definitions & Disclaimers

- **Pre-registered Protocol:** Fixed frozen weights (`model.eval()`, `torch.no_grad()`, `functional.reset_net(model)` before and after every batch), `T = 4`, `seed = 42`.
- **Blind Calibration Lock:** Candidate `C` hyperparameters (`tau=2.0, theta0=0.5, rho=0.5, eta=0.5, kappa=1.0`) were locked **strictly from `cal_results.csv`** before any `eval` run:
  - **Track 1** ([`locked_candidates.json`](file:///home/tzufucius/Project/MaxFormer/outputs/qkta/track1_imagenet/locked_candidates.json), `cal_size = 1,000`): `b in {2.0, 3.0}` for both `stage1` and `all_qkta` (`B0_original` cal acc = `84.00%`; `C_b3.0` peaked at `84.10%` on `all_qkta`). See [`cal_results.csv`](file:///home/tzufucius/Project/MaxFormer/outputs/qkta/track1_imagenet/cal_results.csv).
  - **Track 2** ([`locked_candidates.json`](file:///home/tzufucius/Project/MaxFormer/outputs/qkta/track2_cifar10/locked_candidates.json), `cal_size = 2,000`): `b in {1.0, 3.0}` for both `stage1` and `all_qkta` (`B0_original` cal acc = `91.90%`; `C_b3.0` peaked at `92.45%` on `stage1` and `92.30%` on `all_qkta`). See [`cal_results.csv`](file:///home/tzufucius/Project/MaxFormer/outputs/qkta/track2_cifar10/cal_results.csv).
- **Metric Definitions & Measurement Separation:**
  - `K_post_vs_B0 = K_post_variant / max(K_post_B0_original, 1)`: post-gate K spike ratio relative to `B0_original`. **This is a local spike-count ratio, NOT chip-level energy or mJ.**
  - `K_retention_vs_own_input = K_post_spikes / max(K_pre_spikes, 1)`.
  - Telemetry pass (`QKTATelemetry(enabled=True)`) is strictly separated from the hook-free, CUDA-synchronized timing & peak-VRAM pass (`enabled=False` + `torch.cuda.synchronize`).
- **Dataset Scope Disclaimer:** Track 1 evaluates a **10-synset ImageNet-1K subset (`Imagenette2-320/val`, `eval_size = 2,925`)** with `0..999` WNID mapping (1000-way logits), **NOT** full 1000-class ImageNet-1K Top-1 accuracy.
- **GPU Budget Ledger:** [`gpu_budget_log.json`](file:///home/tzufucius/Project/MaxFormer/outputs/qkta/gpu_budget_log.json) — **`3.3308` GPU hours** (`11,990.87 s`) across **`164` logged runs** (well below the `7.0 h` soft cap / `8.0 h` hard cap).

---

## 2. Track 1 Independent Eval (`ms_qkformer_10_384`, Imagenette2-320, `n = 2,925`)

- **Sources:** [`frozen_results.csv`](file:///home/tzufucius/Project/MaxFormer/outputs/qkta/track1_imagenet/frozen_results.csv), [`frozen_results.json`](file:///home/tzufucius/Project/MaxFormer/outputs/qkta/track1_imagenet/frozen_results.json), [`frozen_metadata.json`](file:///home/tzufucius/Project/MaxFormer/outputs/qkta/track1_imagenet/frozen_metadata.json)
- **Plots:** [`accuracy_vs_postK_ratio.png`](file:///home/tzufucius/Project/MaxFormer/outputs/qkta/track1_imagenet/plots/accuracy_vs_postK_ratio.png) | [`per_step_gate_rate.png`](file:///home/tzufucius/Project/MaxFormer/outputs/qkta/track1_imagenet/plots/per_step_gate_rate.png) | [`mask_examples.png`](file:///home/tzufucius/Project/MaxFormer/outputs/qkta/track1_imagenet/plots/mask_examples.png) | [`controls_and_failure_cases.png`](file:///home/tzufucius/Project/MaxFormer/outputs/qkta/track1_imagenet/plots/controls_and_failure_cases.png)

### 2.1 Single-Layer (`stage1`: `stage1.0.attn`)

| Mode / Params | Top-1 (%) | `delta_pp` | `K_pre_spikes` | `K_post_spikes` | `K_post_vs_B0` | `K_ret_own` | `A_open_rate` | Latency (ms/img) | Peak VRAM (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `B0_original` | 82.09 | +0.00 | 269,671,994 | 90,685,002 | 1.0000 | 0.3363 | 0.1823 | 7.81 | 18,811 |
| `B0_custom` | 82.09 | +0.00 | 269,671,994 | 90,685,002 | 1.0000 | 0.3363 | 0.1823 | 7.83 | 18,785 |
| `B1_lif` (`theta0=0.5`) | 82.09 | +0.00 | 269,671,994 | 90,685,002 | 1.0000 | 0.3363 | 0.1823 | 7.83 | 18,785 |
| `B1_lif` (`theta0=1.0`) | 82.60 | +0.51 | 269,671,994 | 51,250,484 | 0.5651 | 0.1900 | 0.0836 | 7.83 | 18,785 |
| `B1_lif` (`theta0=1.5`) | 83.11 | +1.03 | 269,671,994 | 33,321,547 | 0.3674 | 0.1236 | 0.0492 | 7.84 | 18,785 |
| `B1_lif` (`theta0=2.0`) | 82.94 | +0.85 | 269,671,994 | 22,013,717 | 0.2427 | 0.0816 | 0.0303 | 7.83 | 18,785 |
| `B1_count` (`k=1`) | 82.09 | +0.00 | 269,671,994 | 90,685,002 | 1.0000 | 0.3363 | 0.1823 | 7.83 | 18,785 |
| `B1_count` (`k=2`) | 82.60 | +0.51 | 269,671,994 | 51,250,484 | 0.5651 | 0.1900 | 0.0836 | 7.83 | 18,785 |
| `B1_count` (`k=3`) | 83.08 | +0.99 | 269,671,994 | 31,430,469 | 0.3466 | 0.1166 | 0.0457 | 7.83 | 18,785 |
| `B1_count` (`k=4`) | 82.80 | +0.72 | 269,671,994 | 19,689,648 | 0.2171 | 0.0730 | 0.0266 | 7.84 | 18,785 |
| `H` (`b=2.0`) | 83.04 | +0.96 | 269,671,994 | 67,868,666 | 0.7484 | 0.2517 | 0.1352 | 7.84 | 18,785 |
| `C` (`b=2.0`) | 82.53 | +0.44 | 269,671,994 | 70,185,323 | 0.7739 | 0.2603 | 0.1391 | 7.83 | 18,785 |
| `P` (`b=2.0`, 3 seeds) | 82.94 [82.56, 83.18] | +0.85 | 269,671,994 | 68,701,735 [68,700,858..68,702,988] | 0.7576 | 0.2548 | 0.1366 | 8.59 | 18,785 |
| `H` (`b=3.0`) | 82.56 | +0.48 | 269,671,994 | 62,492,694 | 0.6891 | 0.2317 | 0.1261 | 7.83 | 18,785 |
| `C` (`b=3.0`) | 83.32 | +1.23 | 269,671,994 | 65,732,435 | 0.7248 | 0.2437 | 0.1311 | 7.84 | 18,785 |
| `P` (`b=3.0`, 3 seeds) | 82.42 [82.05, 82.70] | +0.33 | 269,671,994 | 63,286,493 [63,285,242..63,288,401] | 0.6979 | 0.2347 | 0.1274 | 8.59–8.61 | 18,785 |

### 2.2 Multi-Layer (`all_qkta`: `stage1.0.attn`, `stage2.0.attn`, `stage2.1.attn`)

| Mode / Params | Top-1 (%) | `delta_pp` | `K_pre_spikes` | `K_post_spikes` | `K_post_vs_B0` | `K_ret_own` | `A_open_rate` | Latency (ms/img) | Peak VRAM (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `B0_original` | 82.09 | +0.00 | 501,220,929 | 194,585,286 | 1.0000 | 0.3882 | 0.2342 | 7.82 | 18,811 |
| `B0_custom` | 82.09 | +0.00 | 501,220,929 | 194,585,286 | 1.0000 | 0.3882 | 0.2342 | 7.85 | 18,769 |
| `B1_lif` (`theta0=0.5`) | 82.09 | +0.00 | 501,220,929 | 194,585,286 | 1.0000 | 0.3882 | 0.2342 | 7.84 | 18,769 |
| `B1_lif` (`theta0=1.0`) | 82.84 | +0.75 | 497,579,783 | 124,488,791 | 0.6398 | 0.2502 | 0.1369 | 7.84 | 18,769 |
| `B1_lif` (`theta0=1.5`) | 82.94 | +0.85 | 495,943,217 | 93,534,978 | 0.4807 | 0.1886 | 0.1023 | 7.84 | 18,769 |
| `B1_lif` (`theta0=2.0`) | 82.53 | +0.44 | 494,792,937 | 74,212,531 | 0.3814 | 0.1500 | 0.0831 | 7.85 | 18,769 |
| `B1_count` (`k=1`) | 82.09 | +0.00 | 501,220,929 | 194,585,286 | 1.0000 | 0.3882 | 0.2342 | 7.84 | 18,769 |
| `B1_count` (`k=2`) | 82.84 | +0.75 | 497,579,783 | 124,488,791 | 0.6398 | 0.2502 | 0.1369 | 7.85 | 18,769 |
| `B1_count` (`k=3`) | 82.53 | +0.44 | 495,795,663 | 90,606,273 | 0.4656 | 0.1827 | 0.0995 | 7.84 | 18,769 |
| `B1_count` (`k=4`) | 83.04 | +0.96 | 494,586,953 | 70,590,990 | 0.3628 | 0.1427 | 0.0801 | 7.84 | 18,769 |
| `H` (`b=2.0`) | 82.70 | +0.62 | 499,294,248 | 154,027,966 | 0.7916 | 0.3085 | 0.1874 | 7.84 | 18,769 |
| `C` (`b=2.0`) | 82.70 | +0.62 | 499,455,646 | 157,651,345 | 0.8102 | 0.3156 | 0.1907 | 7.85 | 18,769 |
| `P` (`b=2.0`, 3 seeds) | 83.01 [82.77, 83.21] | +0.92 | 499,359,911 | 155,787,574 [155,782,881..155,790,385] | 0.8006 | 0.3120 | 0.1893 | 9.73–9.74 | 18,769 |
| `H` (`b=3.0`) | 83.21 | +1.13 | 498,919,784 | 143,955,771 | 0.7398 | 0.2885 | 0.1772 | 7.85 | 18,769 |
| `C` (`b=3.0`) | 82.97 | +0.89 | 499,059,499 | 149,120,679 | 0.7664 | 0.2988 | 0.1816 | 7.84 | 18,769 |
| `P` (`b=3.0`, 3 seeds) | 82.80 [82.70, 82.87] | +0.71 | 498,945,974 | 145,678,979 [145,673,538..145,683,872] | 0.7487 | 0.2920 | 0.1791 | 9.72–9.73 | 18,769 |

- **Per-step `A_open_rate` (`t=0..3` at `b=3.0`):**
  - `stage1`: `B0_original = [0.0871, 0.2204, 0.2220, 0.1996]`; `H = [0.0871, 0.1861, 0.1283, 0.1027]`; `C = [0.0871, 0.1907, 0.1374, 0.1093]`; `P = [0.0871, 0.1872, 0.1305, 0.1046]`.
  - `all_qkta`: `B0_original = [0.1172, 0.2554, 0.2787, 0.2854]`; `H = [0.1172, 0.2212, 0.1890, 0.1814]`; `C = [0.1172, 0.2250, 0.1964, 0.1878]`; `P = [0.1172, 0.2224, 0.1919, 0.1848]`.

---

## 3. Track 2 Independent Eval (`cifar10-100/ms_qkformer.py`, CIFAR-10, `n = 8,000`)

- **Sources:** [`frozen_results.csv`](file:///home/tzufucius/Project/MaxFormer/outputs/qkta/track2_cifar10/frozen_results.csv), [`frozen_results.json`](file:///home/tzufucius/Project/MaxFormer/outputs/qkta/track2_cifar10/frozen_results.json), [`frozen_metadata.json`](file:///home/tzufucius/Project/MaxFormer/outputs/qkta/track2_cifar10/frozen_metadata.json)
- **Plots:** [`accuracy_vs_postK_ratio.png`](file:///home/tzufucius/Project/MaxFormer/outputs/qkta/track2_cifar10/plots/accuracy_vs_postK_ratio.png) | [`per_step_gate_rate.png`](file:///home/tzufucius/Project/MaxFormer/outputs/qkta/track2_cifar10/plots/per_step_gate_rate.png) | [`mask_examples.png`](file:///home/tzufucius/Project/MaxFormer/outputs/qkta/track2_cifar10/plots/mask_examples.png) | [`controls_and_failure_cases.png`](file:///home/tzufucius/Project/MaxFormer/outputs/qkta/track2_cifar10/plots/controls_and_failure_cases.png)

### 3.1 Single-Layer (`stage1`: `stage1.0.attn`)

| Mode / Params | Top-1 (%) | `delta_pp` | `K_pre_spikes` | `K_post_spikes` | `K_post_vs_B0` | `K_ret_own` | `A_open_rate` | Latency (ms/img) | Peak VRAM (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `B0_original` / `B0_custom` | 91.47 | +0.00 | 41,274,731 | 30,760,789 | 1.0000 | 0.7453 | 0.0803 | 1.47 | 7,465 / 7,441 |
| `B1_lif` (`theta0=1.0`) / `B1_count` (`k=2`) | 91.62 | +0.15 | 41,274,731 | 26,470,967 | 0.8605 | 0.6413 | 0.0588 | 1.47 | 7,441 |
| `B1_lif` (`theta0=1.5`) | 91.59 | +0.11 | 41,274,731 | 23,181,659 | 0.7536 | 0.5616 | 0.0466 | 1.47 | 7,441 |
| `B1_lif` (`theta0=2.0`) | 91.54 | +0.06 | 41,274,731 | 20,389,013 | 0.6628 | 0.4940 | 0.0379 | 1.47 | 7,441 |
| `B1_count` (`k=3`) | 91.54 | +0.06 | 41,274,731 | 23,037,536 | 0.7489 | 0.5582 | 0.0461 | 1.47 | 7,441 |
| `B1_count` (`k=4`) | 91.47 | +0.00 | 41,274,731 | 20,162,629 | 0.6555 | 0.4885 | 0.0372 | 1.47 | 7,441 |
| `H` (`b=1.0`) | 91.53 | +0.05 | 41,274,731 | 29,224,870 | 0.9501 | 0.7081 | 0.0741 | 1.47 | 7,441 |
| `C` (`b=1.0`) | 91.44 | -0.04 | 41,274,731 | 29,302,767 | 0.9526 | 0.7099 | 0.0743 | 1.47 | 7,441 |
| `P` (`b=1.0`, 3 seeds) | 91.56 [91.51, 91.59] | +0.08 | 41,274,731 | 29,255,303 [29,255,267..29,255,348] | 0.9511 | 0.7088 | 0.0742 | 2.24 | 7,441 |
| `H` (`b=3.0`) | 91.60 | +0.13 | 41,274,731 | 27,327,297 | 0.8884 | 0.6621 | 0.0684 | 1.47 | 7,441 |
| `C` (`b=3.0`) | 91.53 | +0.05 | 41,274,731 | 28,086,514 | 0.9131 | 0.6805 | 0.0705 | 1.47 | 7,441 |
| `P` (`b=3.0`, 3 seeds) | 91.47 [91.40, 91.59] | -0.00 | 41,274,731 | 27,453,416 [27,452,675..27,454,019] | 0.8925 | 0.6651 | 0.0688 | 2.05–2.24 | 7,441 |

### 3.2 Multi-Layer (`all_qkta`: `stage1.0.attn`, `stage2.0.attn`)

| Mode / Params | Top-1 (%) | `delta_pp` | `K_pre_spikes` | `K_post_spikes` | `K_post_vs_B0` | `K_ret_own` | `A_open_rate` | Latency (ms/img) | Peak VRAM (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `B0_original` / `B0_custom` | 91.47 | +0.00 | 43,736,348 | 32,430,906 | 1.0000 | 0.7415 | 0.0501 | 1.36 | 7,465 / 7,435 |
| `B1_lif` (`theta0=1.0`) / `B1_count` (`k=2`) | 91.29 | -0.19 | 43,855,091 | 28,072,808 | 0.8656 | 0.6401 | 0.0378 | 1.47 | 7,435 |
| `B1_lif` (`theta0=1.5`) | 91.66 | +0.19 | 43,941,284 | 24,726,708 | 0.7624 | 0.5627 | 0.0308 | 1.47 | 7,435 |
| `B1_lif` (`theta0=2.0`) | 91.39 | -0.09 | 44,005,774 | 21,898,865 | 0.6752 | 0.4976 | 0.0259 | 1.47 | 7,435 |
| `B1_count` (`k=3`) | 91.69 | +0.21 | 43,943,718 | 24,582,620 | 0.7580 | 0.5594 | 0.0305 | 1.47 | 7,435 |
| `B1_count` (`k=4`) | 91.46 | -0.01 | 44,009,652 | 21,672,850 | 0.6683 | 0.4925 | 0.0255 | 1.47 | 7,435 |
| `H` (`b=1.0`) | 91.46 | -0.01 | 43,752,609 | 30,882,717 | 0.9523 | 0.7058 | 0.0469 | 1.47 | 7,435 |
| `C` (`b=1.0`) | 91.50 | +0.03 | 43,751,748 | 30,960,650 | 0.9547 | 0.7076 | 0.0470 | 1.47 | 7,435 |
| `P` (`b=1.0`, 3 seeds) | 91.51 [91.49, 91.55] | +0.04 | 43,752,314 | 30,913,053 [30,912,974..30,913,146] | 0.9532 | 0.7065 | 0.0469 | 2.87–2.89 | 7,435 |
| `H` (`b=3.0`) | 91.50 | +0.03 | 43,777,604 | 28,969,723 | 0.8933 | 0.6617 | 0.0439 | 1.47 | 7,435 |
| `C` (`b=3.0`) | 91.54 | +0.06 | 43,767,194 | 29,731,517 | 0.9168 | 0.6793 | 0.0449 | 1.47 | 7,435 |
| `P` (`b=3.0`, 3 seeds) | 91.41 [91.30, 91.50] | -0.06 | 43,776,017 | 29,095,676 [29,094,946..29,096,179] | 0.8972 | 0.6647 | 0.0441 | 2.62–2.64 | 7,435 |

- **Per-step `A_open_rate` (`t=0..3` at `b=3.0`):**
  - `stage1`: `B0_original = [0.0023, 0.0862, 0.0748, 0.1578]`; `H = [0.0023, 0.0860, 0.0565, 0.1289]`; `C = [0.0023, 0.0860, 0.0600, 0.1335]`; `P = [0.0023, 0.0860, 0.0568, 0.1300]`.
  - `all_qkta`: `B0_original = [0.0089, 0.0714, 0.0402, 0.0801]`; `H = [0.0089, 0.0703, 0.0306, 0.0658]`; `C = [0.0089, 0.0704, 0.0324, 0.0681]`; `P = [0.0089, 0.0703, 0.0307, 0.0664]`.

---

## 4. Matched-Sparsity Pareto Analysis & Mask Snapshots

1. **Simple Fixed Thresholds (`B1_count` / `B1_lif`) Dominate Both Pareto Frontiers:**
   - On **Track 1**, `B1_count (k=4)` reaches `K_post_vs_B0 = 0.2171` (`82.80%`, `+0.72 pp`) on `stage1` and `0.3628` (`83.04%`, `+0.96 pp`) on `all_qkta`, while `B1_lif (theta0=1.5)` reaches `0.3674` (`83.11%`, `+1.03 pp`) on `stage1` and `0.4807` (`82.94%`, `+0.85 pp`) on `all_qkta`. By contrast, `H` and `C` stay in `0.689–0.810×` `K_post_vs_B0` with similar accuracy (`82.53–83.32%`).
   - On **Track 2**, `B1_count (k=4)` reaches `K_post_vs_B0 = 0.6555` (`91.47%`, `0.00 pp`) on `stage1` and `0.6683` (`91.46%`, `-0.01 pp`) on `all_qkta`, and `B1_count (k=3)` achieves `91.69%` (`+0.21 pp`) at `0.7580×` on `all_qkta`, whereas `H` and `C` remain in `0.888–0.955×`.
2. **`C` Does Not Consistently Outperform `H` or `P`:**
   - While `C (b=3.0)` on Track 1 `stage1` hits `83.32%` (`+1.23 pp`) vs `H (b=3.0)` `82.56%` (`+0.48 pp`) and `P` `82.42%`, on Track 1 `all_qkta` `H (b=3.0)` (`83.21%`, `0.7398×`) outperforms `C (b=3.0)` (`82.97%`, `0.7664×`), and at `b=2.0` `P` (`82.94%` `stage1`, `83.01%` `all_qkta`) equals or exceeds `C` (`82.53%`, `82.70%`). On Track 2, all differences among `B0`, `H`, `C`, and `P` lie within `[-0.17, +0.13] pp`.
3. **Mask Snapshots & Failure Cases ([`track1 mask_snapshots.npz`](file:///home/tzufucius/Project/MaxFormer/outputs/qkta/track1_imagenet/mask_snapshots.npz), [`track2 mask_snapshots.npz`](file:///home/tzufucius/Project/MaxFormer/outputs/qkta/track2_cifar10/mask_snapshots.npz)):**
   - Track 1 snapshots (`64` fixed samples, `12` misclassified, e.g., `sample_00008`, `sample_00009`): on `sample_00000` (`pred=571, target=571`), `B0_A = 0.1625`, `H_A = 0.1204`, `C_A = 0.1249`, `P_A = 0.1216`.
   - Track 2 snapshots (`128` fixed samples, `14` misclassified, e.g., `sample_00002` `pred=4, target=5`, `sample_00003` `pred=8, target=9`): `H_A`, `C_A`, and `P_A` differ by `< 0.0005` (`0.0996` vs `0.0998` vs `0.0998` on `sample_00002`).
4. **Latency Overhead:** `H` and `C` add negligible latency (`7.83–7.85 ms/img` vs `7.81–7.82 ms/img` `B0` on Track 1; `1.47 ms/img` vs `1.36 ms/img` on Track 2), whereas `P` incurs `~0.8–1.5 ms/img` overhead (`8.59–9.74 ms/img` Track 1; `2.05–2.89 ms/img` Track 2) from per-`(t, b, head)` CPU `torch.randperm` generation.

---

## 5. Optional Lightweight Adaptation Check — SKIPPED

Per the pre-registered trigger rule, Track 2 lightweight adaptation (`adapt_cifar`) is triggered **only if** locked `C` candidates achieve `K_post_vs_B0 <= 0.85` **AND** suffer `delta_top1_pp <= -1.0 pp` (with `total_gpu_hours < 5.5`). Neither condition was met on Track 2 Eval (`delta_top1_pp` in `[-0.04, +0.06] pp`, `K_post_vs_B0` in `[0.9131, 0.9547]`), so adaptation was cleanly **skipped**.
