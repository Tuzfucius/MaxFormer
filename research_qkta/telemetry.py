from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

import torch


class QKTATelemetry:
    """Opt-in per-layer telemetry collector for QKTA gating and K-spike statistics."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self._layers: Dict[str, Dict[str, Any]] = {}

    def reset(self) -> None:
        self._layers.clear()

    def _ensure_layer(self, layer_name: str, T: int, heads: int) -> Dict[str, Any]:
        if layer_name not in self._layers:
            self._layers[layer_name] = {
                "q_hist": defaultdict(int),
                "q_nonzero_count": 0,
                "q_total_count": 0,
                "A_open_count": 0,
                "A_total_count": 0,
                "K_pre_spikes": 0,
                "K_post_spikes": 0,
                "per_step_open_count": [0] * T,
                "per_step_total_count": [0] * T,
                "per_step_post_k": [0] * T,
                "per_head_open_count": [0] * heads,
                "per_head_total_count": [0] * heads,
            }
        return self._layers[layer_name]

    @torch.no_grad()
    def record(
        self,
        layer_name: str,
        q_sum: torch.Tensor,
        attn: torch.Tensor,
        k: torch.Tensor,
        post_k: torch.Tensor,
    ) -> None:
        if not self.enabled:
            return

        q_det = q_sum.detach()
        a_det = attn.detach()
        k_det = k.detach()
        pk_det = post_k.detach()

        T, B, heads, _, N = q_det.shape
        entry = self._ensure_layer(layer_name, T=T, heads=heads)

        # q_sum histogram & nonzero fraction
        q_int = q_det.to(torch.int64).reshape(-1)
        uniq, counts = torch.unique(q_int, return_counts=True)
        for val, cnt in zip(uniq.tolist(), counts.tolist()):
            entry["q_hist"][int(val)] += int(cnt)
        entry["q_nonzero_count"] += int((q_det > 0).sum().item())
        entry["q_total_count"] += int(q_det.numel())

        # A open counts
        a_bool = a_det > 0.5
        entry["A_open_count"] += int(a_bool.sum().item())
        entry["A_total_count"] += int(a_bool.numel())

        # K pre/post spikes
        entry["K_pre_spikes"] += int(round(k_det.sum().item()))
        entry["K_post_spikes"] += int(round(pk_det.sum().item()))

        # Per-step stats (dim 0 = T)
        step_open = a_bool.sum(dim=(1, 2, 3, 4)).tolist()
        step_total = B * heads * N
        step_pk = pk_det.sum(dim=(1, 2, 3, 4)).tolist()
        for t in range(T):
            entry["per_step_open_count"][t] += int(step_open[t])
            entry["per_step_total_count"][t] += int(step_total)
            entry["per_step_post_k"][t] += int(round(step_pk[t]))

        # Per-head stats (dim 2 = heads)
        head_open = a_bool.sum(dim=(0, 1, 3, 4)).tolist()
        head_total = T * B * N
        for h in range(heads):
            entry["per_head_open_count"][h] += int(head_open[h])
            entry["per_head_total_count"][h] += int(head_total)

    def summary(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for name, e in self._layers.items():
            q_tot = max(e["q_total_count"], 1)
            a_tot = max(e["A_total_count"], 1)
            k_pre = max(e["K_pre_spikes"], 1)
            out[name] = {
                "q_count_histogram": dict(sorted(e["q_hist"].items())),
                "q_nonzero_count": e["q_nonzero_count"],
                "q_total_count": e["q_total_count"],
                "q_nonzero_fraction": e["q_nonzero_count"] / q_tot,
                "A_open_count": e["A_open_count"],
                "A_total_count": e["A_total_count"],
                "A_open_rate": e["A_open_count"] / a_tot,
                "K_pre_spikes": e["K_pre_spikes"],
                "K_post_spikes": e["K_post_spikes"],
                "K_retention_vs_own_input": e["K_post_spikes"] / k_pre,
                "per_step_open_count": list(e["per_step_open_count"]),
                "per_step_total_count": list(e["per_step_total_count"]),
                "per_step_open_rate": [
                    oc / max(tc, 1)
                    for oc, tc in zip(e["per_step_open_count"], e["per_step_total_count"])
                ],
                "per_step_post_k": list(e["per_step_post_k"]),
                "per_head_open_count": list(e["per_head_open_count"]),
                "per_head_total_count": list(e["per_head_total_count"]),
                "per_head_open_rate": [
                    oc / max(tc, 1)
                    for oc, tc in zip(e["per_head_open_count"], e["per_head_total_count"])
                ],
            }
        return out


def compute_k_post_vs_b0(
    variant_summary: Dict[str, Dict[str, Any]],
    b0_summary: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Compute per-layer and total K_post_variant / max(K_post_B0, 1) with raw numerators/denominators."""
    result: Dict[str, Dict[str, Any]] = {}
    total_var = 0
    total_b0 = 0
    for layer_name, v_stats in variant_summary.items():
        v_post = int(v_stats["K_post_spikes"])
        b0_post = int(b0_summary.get(layer_name, {}).get("K_post_spikes", 0))
        total_var += v_post
        total_b0 += b0_post
        result[layer_name] = {
            "K_post_variant": v_post,
            "K_post_B0": b0_post,
            "K_post_vs_B0": v_post / max(b0_post, 1),
        }
    result["__total__"] = {
        "K_post_variant": total_var,
        "K_post_B0": total_b0,
        "K_post_vs_B0": total_var / max(total_b0, 1),
    }
    return result
