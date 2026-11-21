# P3 Checkpoint Audit & Frozen Baseline Verification

## 1. Track 1 — Official ImageNet Model (`ms_qkformer_10_384` + Imagenette2-320)

> **Dataset Scope Disclaimer:** Track 1 evaluates a **10-synset ImageNet-1K subset (`Imagenette2-320/val`, 3,925 images)** using 1000-way ImageNet-1K logits (`0..999` WNID mapping, never renumbered `0..9`). These numbers are **NOT** full 1000-class ImageNet-1K Top-1 accuracy.

- **Model & QKTA Scope:** [`imagenet/ms_qkformer.py`](imagenet/ms_qkformer.py) (`ms_qkformer_10_384`, `T = 4`, `in_channels = 3`, `num_classes = 1000`, `embed_dims = 384`), containing **3 QKTA layers**: `stage1.0.attn`, `stage2.0.attn`, and `stage2.1.attn` (`stage3` uses 7 `Block_SSA` layers).
- **Checkpoint & Strict Audit:**
  - File: [`ms_qk_10-384_T4.pth.tar`](imagenet/checkpoints/ms_qk_10-384_T4.pth.tar) (`66,288,470` bytes)
  - SHA-256: `792a7c95f7ce47492c6cb9603be366c70eac2d233981c8e8daadf76913b33707`
  - Audited via [`checkpoint_audit.py`](research_qkta/checkpoint_audit.py): **`434/434` exact key + shape match** (`0` missing keys, `0` unexpected keys, `0` shape mismatches, `is_exact_match = true`).
  - Unsafe sequential copying such as [`imagenet/misc.py::load_old_ckpt`](imagenet/misc.py#L332-L383) is explicitly rejected.
- **Preprocessing & WNID Mapping:**
  - Validation transforms match [`imagenet/utils.py::build_transform`](imagenet/utils.py#L141-L156): `Resize(256, BICUBIC)` -> `CenterCrop(224)` -> `ToTensor()` -> `Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)`.
  - Verified by `verify_imagenette_wnid_mapping`: all 10 Imagenette2-320 folders map strictly to their true `0..999` ImageNet-1K indices (`n01440764: 0`, `n02102040: 217`, `n02979186: 482`, `n03000684: 491`, `n03028079: 497`, `n03394916: 566`, `n03417042: 569`, `n03425413: 571`, `n03445777: 574`, `n03888257: 701`).
- **Stratified Split & Sanity Checks ([`sanity.json`](outputs/qkta/track1_imagenet/sanity.json)):**
  - Stratified split (`seed = 42`): [`cal_indices.json`](outputs/qkta/track1_imagenet/cal_indices.json) (`n = 1,000`) and [`eval_indices.json`](outputs/qkta/track1_imagenet/eval_indices.json) (`n = 2,925`), `0` overlap.
  - Determinism & Reset (`model.eval()` + `torch.no_grad()` + `functional.reset_net(model)`): `repeat_max_abs_diff = 0.0`, `reverse_order_max_abs_diff = 0.0`.
  - `B0_original` vs `B0_custom` logits parity: `B0_original_vs_custom_logits_max_abs_diff = 0.0`.
  - Calibration `B0_original` (`stage1`): `Top-1 = 84.00%` (`0.84`), latency `7.8298 ms/img`, `q_nonzero_fraction == A_open_rate == 0.182471726` (confirming exact Channel-OR behavior on real activations).

---

## 2. Track 2 — Lightweight CIFAR-10 AMP Baseline (`MS_QKformer`)

- **Model & QKTA Scope:** Unmodified upstream [`cifar10-100/ms_qkformer.py`](cifar10-100/ms_qkformer.py) (`MS_QKformer`, `T = 4`, `in_channels = 3`, `num_classes = 10`, `embed_dims = 384`), containing **2 QKTA layers**: `stage1.0.attn` and `stage2.0.attn` (`stage3` uses 2 `Block_SSA` layers).
- **AMP Baseline Training ([`frozen_eval.py::train_cifar10_baseline`](research_qkta/frozen_eval.py)):**
  - Config ([`cifar10_ms_qk_t4.yaml`](configs/research_qkta/cifar10_ms_qk_t4.yaml)): `batch_size = 128`, `lr = 0.0015`, `weight_decay = 0.06`, `min_lr = 1e-5`, `seed = 42`, Mixup/CutMix, AMP (`torch.cuda.amp`).
  - Stopped cleanly at the pre-registered `2.2h` nominal training cap (`7920.1648s`, peak VRAM `13,612.85 MB`), reaching **Epoch 74** best validation checkpoint (`best_val_acc = 0.9167668269230769`, `91.68%`).
- **Checkpoint & Strict Audit:**
  - File: [`cifar10_ms_qk_t4_baseline.pth`](outputs/qkta/track2_cifar10/cifar10_ms_qk_t4_baseline.pth) (`26,990,010` bytes)
  - SHA-256: `0a12794e4320916025c3cc7c902e781ac651a7f89252e193e045b93d5e59acd3`
  - Strict audit via `audit_and_load_checkpoint`: **`200/200` exact key + shape match** (`0` missing, `0` unexpected, `0` shape mismatch, `is_exact_match = true`).
- **Stratified Split & Sanity Checks ([`sanity.json`](outputs/qkta/track2_cifar10/sanity.json)):**
  - Stratified split on CIFAR-10 test set (`n = 10,000`, `seed = 42`): [`cal_indices.json`](outputs/qkta/track2_cifar10/cal_indices.json) (`n = 2,000`) and [`eval_indices.json`](outputs/qkta/track2_cifar10/eval_indices.json) (`n = 8,000`), `0` overlap.
  - Determinism & Reset: `repeat_max_abs_diff = 0.0`, `reverse_order_max_abs_diff = 0.0`.
  - `B0_original` vs `B0_custom` logits parity: `B0_original_vs_custom_logits_max_abs_diff = 0.0`.
  - Calibration `B0_original` (`stage1`): `Top-1 = 91.90%` (`0.919`), latency `1.3588 ms/img`, `q_nonzero_fraction == A_open_rate == 0.080433701`.
