from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .gates import QKTAConfig, QKTAMaskGate
from .telemetry import QKTATelemetry


def discover_qkta_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Return `(name, module)` pairs for all `Token_QK_Attention` modules in `model`."""
    return [
        (name, module)
        for name, module in model.named_modules()
        if type(module).__name__ == "Token_QK_Attention"
    ]


def set_lif_backend(model: nn.Module, backend: str = "torch") -> Dict[str, str]:
    """In-memory switch of `MultiStepLIFNode.backend` (returns `{name: previous_backend}`)."""
    prev_backends: Dict[str, str] = {}
    for name, module in model.named_modules():
        if type(module).__name__ == "MultiStepLIFNode" and hasattr(module, "backend"):
            prev_backends[name] = module.backend
            module.backend = backend
    return prev_backends


def _resolve_target_layers(
    discovered: List[Tuple[str, nn.Module]],
    target_layers: Union[str, Sequence[str]],
) -> List[Tuple[str, nn.Module]]:
    if not discovered:
        raise ValueError("No Token_QK_Attention modules discovered in model.")

    if isinstance(target_layers, str):
        if target_layers == "all_qkta":
            return discovered
        if target_layers == "stage1":
            selected = [(n, m) for n, m in discovered if n.startswith("stage1.")]
            if not selected:
                raise ValueError(f"No stage1 Token_QK_Attention layers found among {[n for n, _ in discovered]}")
            return selected
        # Single explicit layer name
        target_set = {target_layers}
    else:
        target_set = set(target_layers)

    by_name = dict(discovered)
    missing = sorted(target_set - set(by_name.keys()))
    if missing:
        raise ValueError(
            f"Requested QKTA layers not found: {missing}. Available: {list(by_name.keys())}"
        )
    return [(n, m) for n, m in discovered if n in target_set]


def _make_patched_forward(
    module: nn.Module,
    layer_name: str,
    config: QKTAConfig,
    gate: QKTAMaskGate,
    telemetry: Optional[QKTATelemetry],
) -> Callable[[torch.Tensor], torch.Tensor]:
    def patched_forward(x: torch.Tensor) -> torch.Tensor:
        T, B, C, H, W = x.shape
        identity = x
        x_proj = module.proj_lif(x)
        x_flat = x_proj.flatten(3)
        T, B, C, N = x_flat.shape
        x_for_qkv = x_flat.flatten(0, 1)

        q_conv_out = module.q_conv(x_for_qkv)
        q_conv_out = module.q_bn(q_conv_out).reshape(T, B, C, N)
        q_conv_out = module.q_lif(q_conv_out)
        q = q_conv_out.unsqueeze(2).reshape(T, B, module.num_heads, C // module.num_heads, N)

        k_conv_out = module.k_conv(x_for_qkv)
        k_conv_out = module.k_bn(k_conv_out).reshape(T, B, C, N)
        k_conv_out = module.k_lif(k_conv_out)
        k = k_conv_out.unsqueeze(2).reshape(T, B, module.num_heads, C // module.num_heads, N)

        q_sum = torch.sum(q, dim=3, keepdim=True)
        if config.mode == "B0_original":
            attn = module.attn_lif(q_sum)
        else:
            attn = gate(q_sum, H=H, W=W)

        post_k = torch.mul(attn, k)
        if telemetry is not None and telemetry.enabled:
            telemetry.record(layer_name=layer_name, q_sum=q_sum, attn=attn, k=k, post_k=post_k)

        out = post_k.flatten(2, 3)
        out = module.proj_bn(module.proj_conv(out.flatten(0, 1))).reshape(T, B, C, H, W)
        out = out + identity
        return out

    return patched_forward


def apply_qkta_gates(
    model: nn.Module,
    config: QKTAConfig,
    target_layers: Union[str, Sequence[str]] = "all_qkta",
    telemetry: Optional[QKTATelemetry] = None,
) -> Dict[str, Callable[..., torch.Tensor]]:
    """Non-invasively patch selected `Token_QK_Attention.forward` methods and return originals."""
    discovered = discover_qkta_layers(model)
    selected = _resolve_target_layers(discovered, target_layers)
    originals: Dict[str, Callable[..., torch.Tensor]] = {}
    for layer_name, module in selected:
        originals[layer_name] = module.forward
        gate = QKTAMaskGate(config=config)
        module.forward = _make_patched_forward(
            module=module,
            layer_name=layer_name,
            config=config,
            gate=gate,
            telemetry=telemetry,
        )
    return originals


def restore_qkta_gates(
    model: nn.Module,
    originals: Dict[str, Callable[..., torch.Tensor]],
) -> None:
    """Restore previously patched `Token_QK_Attention.forward` methods."""
    by_name = dict(discover_qkta_layers(model))
    for layer_name, orig_forward in originals.items():
        if layer_name in by_name:
            by_name[layer_name].forward = orig_forward


class QKTAExperimentContext:
    """Reversible context manager that patches `Token_QK_Attention.forward` for an experiment."""

    def __init__(
        self,
        model: nn.Module,
        config: QKTAConfig,
        target_layers: Union[str, Sequence[str]] = "all_qkta",
        telemetry: Optional[QKTATelemetry] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.target_layers = target_layers
        self.telemetry = telemetry
        self._originals: Dict[str, Callable[..., torch.Tensor]] = {}

    def __enter__(self) -> "QKTAExperimentContext":
        self._originals = apply_qkta_gates(
            model=self.model,
            config=self.config,
            target_layers=self.target_layers,
            telemetry=self.telemetry,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        restore_qkta_gates(self.model, self._originals)
        self._originals.clear()
