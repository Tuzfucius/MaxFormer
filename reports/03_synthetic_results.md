# P2 Synthetic Dynamic Visual Probes — Results & Mechanism Analysis

> **Evidence Boundary:** Synthetic probes test controllable dynamic patterns on binary query spikes $Q \in \{0,1\}^{T \times B \times \text{heads} \times D_{\text{head}} \times (HW)}$ to characterize gate mechanics (`B0`, `B1`, `H`, `C`, `P`). They **do not** establish generalization on real visual datasets or downstream classification accuracy.

## 1. Experimental Configuration

- **Generator & Runner:** [`synthetic_probe.py`](file:///home/tzufucius/Project/MaxFormer/research_qkta/synthetic_probe.py) (`run_synthetic_suite`), [`synthetic.yaml`](file:///home/tzufucius/Project/MaxFormer/configs/research_qkta/synthetic.yaml)
- **Dimensions:** `H = 16`, `W = 16` (`N = 256` tokens), `T = 10` timesteps, `B = 1`, `heads = 1`, `D_head = 8`
- **Base Seed & Pattern Seeds:** `base_seed = 42`; per-pattern seeds `42..47` (`42 + pattern_idx`)
- **Gate Parameters:**
  - `B0_original`, `B0_custom`: `tau = 2.0, theta0 = 0.5, b = 0.0`
  - `B1_count`: `k_thresh in {1, 2, 3, 4}` (memoryless integer threshold)
  - `B1_lif`: `tau = 2.0, theta0 in {0.5, 1.0, 1.5, 2.0}` (standard LIF)
  - `H`: `tau = 2.0, theta0 = 0.5, rho = 0.5, eta = 0.0, b in {0.5, 1.0, 2.0, 3.0, 4.0}`
  - `C`: `tau = 2.0, theta0 = 0.5, rho = 0.5, eta = 0.5, kappa = 1.0, b in {0.5, 1.0, 2.0, 3.0, 4.0}`
  - `P`: Same as `C` at `b in {1.0, 2.0, 3.0}`, with spatial permutation of `c` per `(t, b, head)` across `perm_seeds = (101, 202, 303)`
- **Output Artifacts:**
  - CSV: [`synthetic_results.csv`](file:///home/tzufucius/Project/MaxFormer/outputs/qkta/synthetic/synthetic_results.csv)
  - JSON: [`synthetic_results.json`](file:///home/tzufucius/Project/MaxFormer/outputs/qkta/synthetic/synthetic_results.json)
  - Traces: [`synthetic_traces.npz`](file:///home/tzufucius/Project/MaxFormer/outputs/qkta/synthetic/synthetic_traces.npz)
  - Summary Figure: [`synthetic_summary.png`](file:///home/tzufucius/Project/MaxFormer/outputs/qkta/synthetic/plots/synthetic_summary.png)

---

## 2. Pattern-by-Pattern Quantitative Summary (with Raw Numerators & Denominators)

Metric definitions:
- `TPR_gate = tpr_num / tpr_den = sum(A * GT) / sum(GT)` (`N/A (0/0)` when `GT = 0` everywhere)
- `FPR_gate = fpr_num / fpr_den = sum(A * (1 - GT)) / sum(1 - GT)`
- `A_open_rate = a_open_count / a_total_count = sum(A) / (T * B * heads * H * W)` (`a_total_count = 2560`)

### 2.1 `low_contrast_persistent_bg` (`seed = 42`, `GT = 0`, `q_sum_mean = 1.7695`, `q_nonzero_fraction = 0.8590`)
- Plot: [`synthetic_low_contrast_persistent_bg.png`](file:///home/tzufucius/Project/MaxFormer/outputs/qkta/synthetic/plots/synthetic_low_contrast_persistent_bg.png)

| Config | `TPR_gate` (`num/den`) | `FPR_gate` (`num/den`) | `A_open_rate` (`num/den`) | `c_active_rate` |
| --- | --- | --- | --- | --- |
| `B0_original` / `B0_custom` | `N/A (0/0)` | `0.8590 (2199/2560)` | `0.8590 (2199/2560)` | `0.0000` |
| `B1_count_k2` / `B1_lif_theta1.0` | `N/A (0/0)` | `0.5582 (1429/2560)` | `0.5582 (1429/2560)` | `0.0000` |
| `B1_count_k3` | `N/A (0/0)` | `0.2578 (660/2560)` | `0.2578 (660/2560)` | `0.0000` |
| `B1_lif_theta1.5` | `N/A (0/0)` | `0.3395 (869/2560)` | `0.3395 (869/2560)` | `0.0000` |
| `H_b1.0` / `H_b2.0` / `H_b3.0` / `H_b4.0` | `N/A (0/0)` | `0.5410 (1385/2560)` / `0.4516 (1156/2560)` / `0.4082 (1045/2560)` / `0.3770 (965/2560)` | same as FPR | `0.0000` |
| `C_b1.0` / `C_b2.0` / `C_b3.0` / `C_b4.0` | `N/A (0/0)` | `0.5445 (1394/2560)` / `0.4758 (1218/2560)` / `0.4410 (1129/2560)` / `0.3977 (1018/2560)` | same as FPR | `0.2340` |
| `P_b2.0` (3 seeds mean [min, max]) | `N/A (0/0)` | `0.4702 [0.4684, 0.4719]` | `0.4702` | `0.2340` |

**Observation:** History inhibition (`H`) cuts persistent background opening from `0.8590` (`2199/2560`) down to `0.3770` (`965/2560`) at `b=4.0`. Because random integer channel-count fluctuations occasionally exceed `kappa = 1.0` (`c_active_rate = 0.2340`), `C` admits slightly more background (`+0.0207` to `+0.0328` FPR for `b=2..4`) than `H`.

### 2.2 `high_contrast_moving_target` (`seed = 43`, `gt_pos = 0.0156`, `q_sum_mean = 0.5078`)
- Plot: [`synthetic_high_contrast_moving_target.png`](file:///home/tzufucius/Project/MaxFormer/outputs/qkta/synthetic/plots/synthetic_high_contrast_moving_target.png)

| Config | `TPR_gate` (`num/den`) | `FPR_gate` (`num/den`) | `A_open_rate` (`num/den`) | `c_active_rate` |
| --- | --- | --- | --- | --- |
| `B0_original` / `B0_custom` | `1.0000 (40/40)` | `0.3556 (896/2520)` | `0.3656 (936/2560)` | `0.0000` |
| `B1_count_k3` | `1.0000 (40/40)` | `0.0052 (13/2520)` | `0.0207 (53/2560)` | `0.0000` |
| `B1_lif_theta1.5` | `1.0000 (40/40)` | `0.0075 (19/2520)` | `0.0230 (59/2560)` | `0.0000` |
| `H_b2.0` / `H_b3.0` / `H_b4.0` | `1.0000 (40/40)` / `0.9750 (39/40)` / `0.9000 (36/40)` | `0.1810 (456/2520)` / `0.1702 (429/2520)` / `0.1639 (413/2520)` | `0.1938` / `0.1828` / `0.1754` | `0.0000` |
| `C_b2.0` / `C_b3.0` / `C_b4.0` | `1.0000 (40/40)` / `0.9750 (39/40)` / `0.9500 (38/40)` | `0.1869 (471/2520)` / `0.1730 (436/2520)` / `0.1655 (417/2520)` | `0.1996` / `0.1855` / `0.1777` | `0.0828` |
| `P_b3.0` (3 seeds mean [min, max]) | `0.9917 [0.9750, 1.0000]` | `0.1712 [0.1702, 0.1718]` | `0.1840` | `0.0828` |

### 2.3 `high_contrast_static_target` (`seed = 46`, `gt_pos = 0.0156`, `q_sum_mean = 0.4781`)
- Plot: [`synthetic_high_contrast_static_target.png`](file:///home/tzufucius/Project/MaxFormer/outputs/qkta/synthetic/plots/synthetic_high_contrast_static_target.png)

| Config | `TPR_gate` (`num/den`) | `FPR_gate` (`num/den`) | `A_open_rate` (`num/den`) | `c_active_rate` |
| --- | --- | --- | --- | --- |
| `B0_original` / `B0_custom` | `1.0000 (40/40)` | `0.3302 (832/2520)` | `0.3406 (872/2560)` | `0.0000` |
| `B1_count_k3` | `1.0000 (40/40)` | `0.0063 (16/2520)` | `0.0219 (56/2560)` | `0.0000` |
| `B1_lif_theta2.0` | `0.9000 (36/40)` | `0.0016 (4/2520)` | `0.0156 (40/2560)` | `0.0000` |
| `H_b1.0` / `H_b2.0` / `H_b3.0` / `H_b4.0` | `1.0000 (40/40)` / `0.9000 (36/40)` / `0.6500 (26/40)` / `0.6000 (24/40)` | `0.1925 (485/2520)` / `0.1786 (450/2520)` / `0.1698 (428/2520)` / `0.1647 (415/2520)` | `0.2051` / `0.1898` / `0.1773` / `0.1715` | `0.0000` |
| `C_b1.0` / `C_b2.0` / `C_b3.0` / `C_b4.0` | `1.0000 (40/40)` / `0.9750 (39/40)` / `0.9250 (37/40)` / `0.8750 (35/40)` | `0.1956 (493/2520)` / `0.1810 (456/2520)` / `0.1722 (434/2520)` / `0.1679 (423/2520)` | `0.2082` / `0.1934` / `0.1840` / `0.1789` | `0.0934` |
| `P_b2.0` (3 seeds mean [min, max]) | `0.9083 [0.9000, 0.9250]` | `0.1798 [0.1794, 0.1802]` | `0.1911` | `0.0934` |
| `P_b3.0` (3 seeds mean [min, max]) | `0.6750 [0.6500, 0.7000]` | `0.1706 [0.1702, 0.1710]` | `0.1785` | `0.0934` |

**Observation:** This pattern isolates the core mechanism hypothesis: `H` falsely suppresses the stationary target as `R` builds (`TPR` falls to `0.6500` at `b=3.0` and `0.6000` at `b=4.0`), whereas `C` preserves `0.9250` (`+0.2750` vs `H`) and `0.8750` (`+0.2750` vs `H`) with only `+0.0024` to `+0.0032` FPR overhead. Spatial permutation (`P` at `b=3.0`) collapses `TPR_mean` back to `0.6750`, confirming that spatial alignment of `c` drives the protection.

### 2.4 `sparse_occluded_reappearing_target` (`seed = 47`, `gt_pos = 0.0211`, `q_sum_mean = 0.5242`)
- Plot: [`synthetic_sparse_occluded_reappearing_target.png`](file:///home/tzufucius/Project/MaxFormer/outputs/qkta/synthetic/plots/synthetic_sparse_occluded_reappearing_target.png)

| Config | `TPR_gate` (`num/den`) | `FPR_gate` (`num/den`) | `A_open_rate` (`num/den`) | `c_active_rate` |
| --- | --- | --- | --- | --- |
| `B0_original` / `B0_custom` | `1.0000 (54/54)` | `0.3468 (869/2506)` | `0.3605 (923/2560)` | `0.0000` |
| `H_b2.0` / `H_b3.0` / `H_b4.0` | `0.9630 (52/54)` / `0.8889 (48/54)` / `0.7037 (38/54)` | `0.1868 (468/2506)` / `0.1728 (433/2506)` / `0.1700 (426/2506)` | `0.2031` / `0.1879` / `0.1812` | `0.0000` |
| `C_b2.0` / `C_b3.0` / `C_b4.0` | `0.9630 (52/54)` / `0.9444 (51/54)` / `0.9074 (49/54)` | `0.1891 (474/2506)` / `0.1756 (440/2506)` / `0.1716 (430/2506)` | `0.2055` / `0.1918` / `0.1871` | `0.0828` |
| `P_b3.0` (3 seeds mean [min, max]) | `0.9074 [0.8889, 0.9259]` | `0.1735 [0.1732, 0.1740]` | `0.1889` | `0.0828` |

### 2.5 Counterexample 1: `high_contrast_flickering_noise` (`seed = 44`, `gt_pos = 0.0625`, `q_sum_mean = 1.0895`)
- Plot: [`synthetic_high_contrast_flickering_noise.png`](file:///home/tzufucius/Project/MaxFormer/outputs/qkta/synthetic/plots/synthetic_high_contrast_flickering_noise.png)

| Config | `TPR_gate` (`num/den`) | `FPR_gate` (`num/den`) | `A_open_rate` (`num/den`) | `c_active_rate` |
| --- | --- | --- | --- | --- |
| `B0_original` / `B0_custom` | `0.9938 (159/160)` | `0.3246 (779/2400)` | `0.3664 (938/2560)` | `0.0000` |
| `B1_count_k3` | `0.9000 (144/160)` | `0.0875 (210/2400)` | `0.1383 (354/2560)` | `0.0000` |
| `B1_lif_theta1.5` | `0.9000 (144/160)` | `0.0879 (211/2400)` | `0.1387 (355/2560)` | `0.0000` |
| `H_b2.0` / `H_b3.0` / `H_b4.0` | `0.7312 (117/160)` / `0.6062 (97/160)` / `0.5500 (88/160)` | `0.2004 (481/2400)` / `0.1942 (466/2400)` / `0.1900 (456/2400)` | `0.2336` / `0.2199` / `0.2125` | `0.0000` |
| `C_b2.0` / `C_b3.0` / `C_b4.0` | `0.7812 (125/160)` / `0.7250 (116/160)` / `0.6625 (106/160)` | `0.2046 (491/2400)` / `0.1963 (471/2400)` / `0.1925 (462/2400)` | `0.2406` / `0.2293` / `0.2219` | `0.1492` |
| `P_b2.0` (3 seeds mean [min, max]) | `0.7521 [0.7438, 0.7562]` | `0.2022 [0.2021, 0.2025]` | `0.2366` | `0.1492` |

**Failure Analysis:** Isolated `1x1` flickering background spikes have low 8-neighbor averages `q_bar` and high `q`, so `q - q_bar >= kappa` sets `c = 1` (`c_active_rate = 0.1492`). Consequently, `C` **falsely protects** background noise (`FPR_C > FPR_H` for all `b >= 1.0`, e.g., `491/2400` vs `481/2400` at `b=2.0`). Moreover, simple fixed-threshold controls `B1_count_k3` (`TPR = 0.9000, FPR = 0.0875`) and `B1_lif_theta1.5` (`TPR = 0.9000, FPR = 0.0879`) achieve both higher target retention and lower background leakage than `C` or `H` on this pattern.

### 2.6 Counterexample 2: `low_contrast_true_target` (`seed = 45`, `gt_pos = 0.1406`, `q_sum_mean = 1.8766`)
- Plot: [`synthetic_low_contrast_true_target.png`](file:///home/tzufucius/Project/MaxFormer/outputs/qkta/synthetic/plots/synthetic_low_contrast_true_target.png)

| Config | `TPR_gate` (`num/den`) | `FPR_gate` (`num/den`) | `A_open_rate` (`num/den`) | `c_active_rate` |
| --- | --- | --- | --- | --- |
| `B0_original` / `B0_custom` | `0.9472 (341/360)` | `0.8577 (1887/2200)` | `0.8703 (2228/2560)` | `0.0000` |
| `B1_count_k2` / `B1_lif_theta1.0` | `0.7583 (273/360)` | `0.5482 (1206/2200)` | `0.5777 (1479/2560)` | `0.0000` |
| `H_b1.0` / `H_b2.0` / `H_b3.0` / `H_b4.0` | `0.6417 (231/360)` / `0.5472 (197/360)` / `0.4806 (173/360)` / `0.4361 (157/360)` | `0.5382 (1184/2200)` / `0.4559 (1003/2200)` / `0.4114 (905/2200)` / `0.3755 (826/2200)` | `0.5527` / `0.4688` / `0.4211` / `0.3840` | `0.0000` |
| `C_b1.0` / `C_b2.0` / `C_b3.0` / `C_b4.0` | `0.6417 (231/360)` / `0.5611 (202/360)` / `0.5167 (186/360)` / `0.4722 (170/360)` | `0.5409 (1190/2200)` / `0.4786 (1053/2200)` / `0.4418 (972/2200)` / `0.4009 (882/2200)` | `0.5551` / `0.4902` / `0.4523` / `0.4109` | `0.2410` |
| `P_b2.0` (3 seeds mean [min, max]) | `0.5630 [0.5583, 0.5694]` | `0.4711 [0.4682, 0.4750]` | `0.4840` | `0.2410` |

**Failure Analysis:** Because the `6x6` target has low contrast against the background (`p_fg = 0.30` vs `p_bg = 0.22`), interior target tokens have `q - q_bar ≈ 0 < kappa = 1.0` (`c = 0`). Thus `C` **over-suppresses** the true target almost identically to `H` (`TPR_C = 0.5611` vs `TPR_H = 0.5472` at `b=2.0`, far below `B0`'s `0.9472`), while `P` (`0.5630`) performs indistinguishably from `C`.

---

## 3. Temporal Dynamics (`per_step_open_rate`, `R_trace_mean`, `theta_trace_mean`)

- **`high_contrast_static_target` (`b = 3.0`):**
  - `B0_custom`: `per_step_tpr = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]`, `theta_trace_mean = 0.5` constant.
  - `H_b3.0`: `per_step_tpr = [1.0, 0.75, 0.5, 0.75, 0.75, 0.25, 0.75, 0.75, 0.5, 0.5]`, `R_trace_mean` rises from `0.0` at `t=0` to `0.1777` at `t=1` and `theta_trace_mean` rises from `0.5` to `1.0947` at `t=2`.
  - `C_b3.0`: `per_step_tpr = [1.0, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 0.75, 1.0, 0.75]` (`theta_trace_mean` stays lower at `[0.5, 1.0098, 1.0581, ...]`).
- **`sparse_occluded_reappearing_target` (`b = 3.0`):**
  - During occlusion (`t = 3..6`), `R_trace_mean` decays from `0.2070` (`t=3`) to `0.1338` (`t=6`) in `H_b3.0` (and `0.2158` to `0.1378` in `C_b3.0`), lowering `theta_trace_mean` from `1.1211` (`t=3`) to `0.9014` (`t=6`) in `H_b3.0` (`1.1306` to `0.8929` in `C_b3.0`).
  - Upon target reappearance at `t = 7`, both `H_b3.0` and `C_b3.0` recover immediately to `per_step_tpr[7] = 1.0`.

---

## 4. Key Takeaways & Calibration Guidance

1. `B0_original`, `B0_custom`, `B1_count_k1`, and `B1_lif_theta0.5` are bit-for-bit identical across all 6 patterns, confirming clean baseline equivalence.
2. When targets are compact and high-contrast (`high_contrast_static_target`, `sparse_occluded_reappearing_target`), `C` (`eta=0.5, kappa=1.0`) substantially mitigates `H`'s false target suppression (`+0.2750` TPR at `b=3,4` on static target) and outperforms 3-seed `P` (`0.9250` vs `0.6750`), showing spatial correspondence is causal in those regimes.
3. Both failure modes behave as theoretically predicted: local contrast `c` has no semantic awareness, so it falsely protects isolated `1x1` flickering noise (`high_contrast_flickering_noise`) and fails to protect large, low-contrast targets (`low_contrast_true_target`).
4. For P4 frozen-model calibration, sweeping `b in {0.5, 1.0, 2.0, 3.0, 4.0}` with `eta=0.5, kappa=1.0, rho=0.5` spans the full regime from near-`B0` (`b=0.5`) to strong history suppression (`b=3.0, 4.0`) where `C` vs `H` diverges clearly.
