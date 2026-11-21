# P3 检查点审计与冻结基线验证

## 1. Track 1 — 官方 ImageNet 模型（`ms_qkformer_10_384` + Imagenette2-320）

> **数据集范围说明：** Track 1 使用 ImageNet-1K 的 10 类子集（`Imagenette2-320/val`，共 3,925 张图像）进行评估，模型输出为 1000 类 ImageNet-1K logits，并使用原始 `0..999` WNID 类别映射，不会重新编号为 `0..9`。这些结果**不代表**完整 1000 类 ImageNet-1K Top-1 准确率。

- **模型与 QKTA 范围：** [`imagenet/ms_qkformer.py`](../../imagenet/ms_qkformer.py)（`ms_qkformer_10_384`，`T = 4`，`in_channels = 3`，`num_classes = 1000`，`embed_dims = 384`），包含 **3 个 QKTA 层**：`stage1.0.attn`、`stage2.0.attn` 和 `stage2.1.attn`（`stage3` 使用 7 个 `Block_SSA` 层）。
- **检查点与严格审计：**
  - 文件：`imagenet/checkpoints/ms_qk_10-384_T4.pth.tar`（`66,288,470` 字节）
  - SHA-256：`792a7c95f7ce47492c6cb9603be366c70eac2d233981c8e8daadf76913b33707`
  - 由 `research_qkta/checkpoint_audit.py` 审计：**`434/434` 个键名和张量形状完全匹配**（缺失键、意外键、形状不匹配均为 `0`，`is_exact_match = true`）。
  - 明确拒绝 `imagenet/misc.py::load_old_ckpt` 这类不安全的顺序复制方式。
- **预处理与 WNID 映射：**
  - 验证集变换与 `imagenet/utils.py::build_transform` 一致：`Resize(256, BICUBIC)` → `CenterCrop(224)` → `ToTensor()` → `Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)`。
  - `verify_imagenette_wnid_mapping` 检查确认，Imagenette2-320 的 10 个目录均映射到对应的 ImageNet-1K 原始类别索引：`n01440764: 0`、`n02102040: 217`、`n02979186: 482`、`n03000684: 491`、`n03028079: 497`、`n03394916: 566`、`n03417042: 569`、`n03425413: 571`、`n03445777: 574`、`n03888257: 701`。
- **分层划分与合理性检查（`sanity.json`）：**
  - 分层划分（`seed = 42`）：`cal_indices.json`（`n = 1,000`）和 `eval_indices.json`（`n = 2,925`），两者无重叠。
  - 确定性与状态重置（`model.eval()`、`torch.no_grad()`、`functional.reset_net(model)`）：`repeat_max_abs_diff = 0.0`，`reverse_order_max_abs_diff = 0.0`。
  - `B0_original` 与 `B0_custom` 的 logits 一致：`B0_original_vs_custom_logits_max_abs_diff = 0.0`。
  - 校准集上的 `B0_original`（`stage1`）：Top-1 为 `84.00%`（`0.84`），延迟 `7.8298 ms/img`，且 `q_nonzero_fraction == A_open_rate == 0.182471726`，证实真实激活上精确符合 Channel-OR 行为。

---

## 2. Track 2 — CIFAR-10 轻量级 AMP 基线（`MS_QKformer`）

- **模型与 QKTA 范围：** 未修改的上游 `cifar10-100/ms_qkformer.py`（`MS_QKformer`，`T = 4`，`in_channels = 3`，`num_classes = 10`，`embed_dims = 384`），包含 **2 个 QKTA 层**：`stage1.0.attn` 和 `stage2.0.attn`（`stage3` 使用 2 个 `Block_SSA` 层）。
- **AMP 基线训练（`frozen_eval.py::train_cifar10_baseline`）：**
  - 配置（`configs/research_qkta/cifar10_ms_qk_t4.yaml`）：`batch_size = 128`、`lr = 0.0015`、`weight_decay = 0.06`、`min_lr = 1e-5`、`seed = 42`，使用 Mixup/CutMix 和 AMP（`torch.cuda.amp`）。
  - 在预先登记的 `2.2h` 名义训练时限（`7920.1648s`，峰值显存 `13,612.85 MB`）内正常结束；最佳验证检查点为 **Epoch 74**（`best_val_acc = 0.9167668269230769`，即 `91.68%`）。
- **检查点与严格审计：**
  - 文件：`outputs/qkta/track2_cifar10/cifar10_ms_qk_t4_baseline.pth`（`26,990,010` 字节）
  - SHA-256：`0a12794e4320916025c3cc7c902e781ac651a7f89252e193e045b93d5e59acd3`
  - `audit_and_load_checkpoint` 严格审计结果：**`200/200` 个键名和张量形状完全匹配**（缺失、意外和形状不匹配均为 `0`，`is_exact_match = true`）。
- **分层划分与合理性检查（`sanity.json`）：**
  - CIFAR-10 测试集（`n = 10,000`，`seed = 42`）分层划分：`cal_indices.json`（`n = 2,000`）和 `eval_indices.json`（`n = 8,000`），两者无重叠。
  - 确定性与状态重置：`repeat_max_abs_diff = 0.0`，`reverse_order_max_abs_diff = 0.0`。
  - `B0_original` 与 `B0_custom` 的 logits 一致：`B0_original_vs_custom_logits_max_abs_diff = 0.0`。
  - 校准集上的 `B0_original`（`stage1`）：Top-1 为 `91.90%`（`0.919`），延迟 `1.3588 ms/img`，且 `q_nonzero_fraction == A_open_rate == 0.080433701`。
