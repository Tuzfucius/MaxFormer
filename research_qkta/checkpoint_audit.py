"""Strict checkpoint auditing, safe loading, and Imagenette/ImageNet validation helpers.

This module explicitly rejects silent sequential parameter copying such as
``imagenet/misc.py::load_old_ckpt`` and enforces exact key-name and tensor-shape
verification before loading weights into a model.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn as nn


class CheckpointAuditError(RuntimeError):
    """Raised when a checkpoint or dataset mapping fails strict audit checks."""


@dataclass
class CheckpointAuditReport:
    """Structured diagnostic report produced by :func:`audit_and_load_checkpoint`."""

    checkpoint_path: str
    sha256: str
    file_size_bytes: int
    top_level_keys: list[str]
    state_dict_source_key: str
    stripped_module_prefix: bool
    model_param_count: int
    ckpt_param_count: int
    matched_keys: list[str]
    missing_keys: list[str]
    unexpected_keys: list[str]
    shape_mismatches: list[dict[str, Any]]
    is_exact_match: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation of the report."""
        return asdict(self)

    def format_summary(self, max_items: int = 10) -> str:
        """Format a human-readable diagnostic summary."""
        lines = [
            f"Checkpoint: {self.checkpoint_path}",
            f"SHA256: {self.sha256}",
            f"Size (bytes): {self.file_size_bytes}",
            f"Top-level keys: {self.top_level_keys}",
            f"State dict source key: {self.state_dict_source_key}",
            f"Stripped 'module.' prefix: {self.stripped_module_prefix}",
            f"Model state_dict entries: {self.model_param_count}",
            f"Checkpoint state_dict entries: {self.ckpt_param_count}",
            f"Matched keys: {len(self.matched_keys)}",
            f"Missing keys: {len(self.missing_keys)}",
            f"Unexpected keys: {len(self.unexpected_keys)}",
            f"Shape mismatches: {len(self.shape_mismatches)}",
            f"Exact match: {self.is_exact_match}",
        ]
        if self.missing_keys:
            preview = self.missing_keys[:max_items]
            lines.append(f"  Sample missing keys: {preview}")
        if self.unexpected_keys:
            preview = self.unexpected_keys[:max_items]
            lines.append(f"  Sample unexpected keys: {preview}")
        if self.shape_mismatches:
            preview = self.shape_mismatches[:max_items]
            lines.append(f"  Sample shape mismatches: {preview}")
        return "\n".join(lines)


# Official Imagenette2-320 WNID -> ImageNet-1K class index (0..999).
# Never renumber these to 0..9 when evaluating a 1000-class ImageNet model.
IMAGENETTE_WNID_TO_IMAGENET1K_IDX: dict[str, int] = {
    "n01440764": 0,    # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888257": 701,  # parachute
}

IMAGENETTE_WNID_TO_LABEL: dict[str, str] = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}


def compute_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute the SHA-256 hex digest of a file by streaming in chunks."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found for SHA-256 computation: {file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_torch_checkpoint_cpu(ckpt_path: Path) -> Any:
    """Load a PyTorch checkpoint on CPU with weights_only=False and older-torch fallback."""
    try:
        return torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(ckpt_path), map_location="cpu")


def _is_tensor_mapping(obj: Any) -> bool:
    return isinstance(obj, Mapping) and len(obj) > 0 and all(
        isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in obj.items()
    )


def _extract_state_dict(raw_ckpt: Any) -> tuple[list[str], str, dict[str, torch.Tensor]]:
    """Extract a {param_name: Tensor} mapping cleanly from a loaded checkpoint object."""
    if isinstance(raw_ckpt, nn.Module):
        sd = {k: v for k, v in raw_ckpt.state_dict().items()}
        return [type(raw_ckpt).__name__], "<nn.Module.state_dict()>", sd

    if not isinstance(raw_ckpt, Mapping):
        raise CheckpointAuditError(
            f"Unsupported checkpoint payload type: {type(raw_ckpt).__name__}; expected Mapping or nn.Module."
        )

    top_level_keys = [str(k) for k in raw_ckpt.keys()]

    for candidate_key in ("state_dict", "model", "model_state", "net", "ema_state_dict"):
        if candidate_key in raw_ckpt:
            sub = raw_ckpt[candidate_key]
            if isinstance(sub, nn.Module):
                sub = sub.state_dict()
            if _is_tensor_mapping(sub):
                return top_level_keys, candidate_key, dict(sub)

    if _is_tensor_mapping(raw_ckpt):
        return top_level_keys, "<raw>", dict(raw_ckpt)

    raise CheckpointAuditError(
        f"Could not locate a valid tensor state_dict in checkpoint. Top-level keys: {top_level_keys}"
    )


def _maybe_strip_module_prefix(
    ckpt_sd: dict[str, torch.Tensor],
    allow_module_prefix_strip: bool,
) -> tuple[dict[str, torch.Tensor], bool]:
    """Strip 'module.' prefix only when allowed and most/all keys carry it."""
    if not allow_module_prefix_strip or not ckpt_sd:
        return ckpt_sd, False

    prefixed_count = sum(1 for k in ckpt_sd if k.startswith("module."))
    if prefixed_count == 0 or prefixed_count <= len(ckpt_sd) / 2:
        return ckpt_sd, False

    stripped: dict[str, torch.Tensor] = {}
    for k, v in ckpt_sd.items():
        new_k = k[len("module."):] if k.startswith("module.") else k
        if new_k in stripped:
            raise CheckpointAuditError(
                f"Key collision when stripping 'module.' prefix: '{new_k}' already exists."
            )
        stripped[new_k] = v
    return stripped, True


def audit_and_load_checkpoint(
    model: nn.Module,
    ckpt_path: str | Path,
    allow_module_prefix_strip: bool = True,
    strict: bool = True,
) -> CheckpointAuditReport:
    """Strictly audit a checkpoint against ``model.state_dict()`` and load if valid.

    Unlike ``imagenet/misc.py::load_old_ckpt``, this function NEVER copies
    parameters by sequential index. Every key name and tensor shape in
    ``model.state_dict()`` (both parameters and persistent buffers) must match.
    If ``strict=True`` and any mismatch exists, no weights are loaded and
    :class:`CheckpointAuditError` is raised with a full diagnostic summary.
    """
    path = Path(ckpt_path)
    if not path.is_file():
        raise CheckpointAuditError(f"Checkpoint file does not exist: {path}")

    sha256 = compute_sha256(path)
    file_size_bytes = path.stat().st_size

    try:
        raw_ckpt = _load_torch_checkpoint_cpu(path)
    except Exception as exc:
        raise CheckpointAuditError(f"Failed to load checkpoint '{path}': {exc}") from exc

    top_level_keys, source_key, ckpt_sd = _extract_state_dict(raw_ckpt)
    ckpt_sd, stripped_module_prefix = _maybe_strip_module_prefix(
        ckpt_sd, allow_module_prefix_strip=allow_module_prefix_strip
    )

    model_sd = model.state_dict()
    missing_keys = sorted(k for k in model_sd if k not in ckpt_sd)
    unexpected_keys = sorted(k for k in ckpt_sd if k not in model_sd)

    matched_keys: list[str] = []
    shape_mismatches: list[dict[str, Any]] = []
    for k in sorted(model_sd):
        if k not in ckpt_sd:
            continue
        model_shape = tuple(model_sd[k].shape)
        ckpt_shape = tuple(ckpt_sd[k].shape)
        if model_shape != ckpt_shape:
            shape_mismatches.append(
                {
                    "key": k,
                    "model_shape": list(model_shape),
                    "ckpt_shape": list(ckpt_shape),
                }
            )
        else:
            matched_keys.append(k)

    is_exact_match = (
        len(missing_keys) == 0
        and len(unexpected_keys) == 0
        and len(shape_mismatches) == 0
        and len(matched_keys) == len(model_sd)
    )

    report = CheckpointAuditReport(
        checkpoint_path=str(path),
        sha256=sha256,
        file_size_bytes=file_size_bytes,
        top_level_keys=top_level_keys,
        state_dict_source_key=source_key,
        stripped_module_prefix=stripped_module_prefix,
        model_param_count=len(model_sd),
        ckpt_param_count=len(ckpt_sd),
        matched_keys=matched_keys,
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
        shape_mismatches=shape_mismatches,
        is_exact_match=is_exact_match,
    )

    if strict and not is_exact_match:
        raise CheckpointAuditError(
            "Strict checkpoint audit failed; refusing to load partial weights.\n"
            + report.format_summary()
        )

    if is_exact_match:
        model.load_state_dict(ckpt_sd, strict=True)
    else:
        safe_partial = {k: ckpt_sd[k] for k in matched_keys}
        model.load_state_dict(safe_partial, strict=False)

    return report


def verify_imagenette_wnid_mapping(
    val_dir: str | Path,
    imagenet_categories: Optional[Sequence[str] | Mapping[int, str]] = None,
) -> dict[str, Any]:
    """Verify that Imagenette2-320 validation folders map strictly to ImageNet-1K 0..999 indices.

    Raises :class:`CheckpointAuditError` if ``val_dir`` is missing, empty, or
    contains unknown folder names (including 0..9 renumbered folders).
    """
    root = Path(val_dir)
    if not root.is_dir():
        raise CheckpointAuditError(f"Imagenette val_dir does not exist or is not a directory: {root}")

    subdirs = sorted(p.name for p in root.iterdir() if p.is_dir())
    if not subdirs:
        raise CheckpointAuditError(f"No class subdirectories found in Imagenette val_dir: {root}")

    unknown = [d for d in subdirs if d not in IMAGENETTE_WNID_TO_IMAGENET1K_IDX]
    if unknown:
        raise CheckpointAuditError(
            f"Unexpected folder names in Imagenette val_dir (must be official WNIDs, never 0..9): {unknown}"
        )

    wnid_to_idx: dict[str, int] = {}
    wnid_to_label: dict[str, str] = {}
    category_checks: dict[str, Optional[str]] = {}

    for wnid in subdirs:
        idx = IMAGENETTE_WNID_TO_IMAGENET1K_IDX[wnid]
        if not (0 <= idx < 1000):
            raise CheckpointAuditError(f"Invalid ImageNet-1K index {idx} for WNID {wnid}")
        wnid_to_idx[wnid] = idx
        wnid_to_label[wnid] = IMAGENETTE_WNID_TO_LABEL[wnid]

        if imagenet_categories is not None:
            if isinstance(imagenet_categories, Mapping):
                if idx not in imagenet_categories:
                    raise CheckpointAuditError(
                        f"ImageNet-1K index {idx} (WNID {wnid}) missing from imagenet_categories mapping."
                    )
                category_checks[wnid] = str(imagenet_categories[idx])
            else:
                if idx >= len(imagenet_categories):
                    raise CheckpointAuditError(
                        f"ImageNet-1K index {idx} (WNID {wnid}) out of range for imagenet_categories "
                        f"(len={len(imagenet_categories)})."
                    )
                category_checks[wnid] = str(imagenet_categories[idx])
        else:
            category_checks[wnid] = None

    return {
        "val_dir": str(root),
        "wnids": subdirs,
        "num_classes_found": len(subdirs),
        "is_complete_imagenette": len(subdirs) == len(IMAGENETTE_WNID_TO_IMAGENET1K_IDX),
        "wnid_to_imagenet1k_idx": wnid_to_idx,
        "wnid_to_label": wnid_to_label,
        "imagenet_category_names": category_checks,
    }


def get_imagenet_val_transform_spec(input_size: int = 224) -> dict[str, Any]:
    """Return the exact upstream ImageNet validation preprocessing specification.

    Derived directly from ``imagenet/utils.py::build_transform(is_train=False, args)``.
    """
    crop_pct = 224 / 256 if input_size <= 224 else 1.0
    resize_size = int(input_size / crop_pct)
    mean = (0.485, 0.456, 0.406)  # timm.data.constants.IMAGENET_DEFAULT_MEAN
    std = (0.229, 0.224, 0.225)   # timm.data.constants.IMAGENET_DEFAULT_STD

    return {
        "source": "imagenet/utils.py::build_transform(is_train=False)",
        "input_size": input_size,
        "crop_pct": crop_pct,
        "resize_size": resize_size,
        "interpolation": "PIL.Image.BICUBIC",
        "center_crop": input_size,
        "mean": mean,
        "std": std,
        "steps": [
            f"transforms.Resize({resize_size}, interpolation=PIL.Image.BICUBIC)",
            f"transforms.CenterCrop({input_size})",
            "transforms.ToTensor()",
            f"transforms.Normalize(mean={mean}, std={std})",
        ],
    }
