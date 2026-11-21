"""Synthetic dynamic visual probes for QKTA gating mechanisms (Protocol §4 P2).

Generates reproducible 5D binary Q spike tensors ``[T, B, heads, D_head, H*W]``
and sums across ``D_head`` to produce ``q_sum: [T, B, heads, 1, H*W]`` alongside
ground-truth spatial target masks ``GT: [T, B, 1, 1, H*W]``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

from research_qkta.gates import QKTAConfig, QKTAMaskGate


PATTERN_NAMES: tuple[str, ...] = (
    "low_contrast_persistent_bg",
    "high_contrast_moving_target",
    "high_contrast_flickering_noise",
    "low_contrast_true_target",
    "high_contrast_static_target",
    "sparse_occluded_reappearing_target",
)


@dataclass
class SyntheticProbeSample:
    """Container for a single synthetic dynamic visual probe."""

    pattern: str
    seed: int
    T: int
    B: int
    heads: int
    D_head: int
    H: int
    W: int
    q_binary: torch.Tensor   # [T, B, heads, D_head, H*W], values in {0, 1}
    q_sum: torch.Tensor      # [T, B, heads, 1, H*W], integer counts in [0, D_head]
    gt_mask: torch.Tensor    # [T, B, 1, 1, H*W], values in {0, 1}
    description: str


def _sample_binary_q_from_probs(
    probs_tbhw: torch.Tensor,
    D_head: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample binary Q in {0, 1} of shape [T, B, heads, D_head, H*W] and sum over D_head."""
    T, B, heads, H, W = probs_tbhw.shape
    N = H * W
    probs_expanded = probs_tbhw.unsqueeze(3).expand(T, B, heads, D_head, H, W)
    rand_u = torch.rand(
        (T, B, heads, D_head, H, W),
        generator=generator,
        dtype=torch.float32,
    )
    q_binary = (rand_u < probs_expanded).to(torch.float32).reshape(T, B, heads, D_head, N)
    q_sum = q_binary.sum(dim=3, keepdim=True)
    return q_binary, q_sum


def generate_synthetic_probe(
    pattern: str,
    seed: int = 42,
    T: int = 10,
    B: int = 1,
    heads: int = 1,
    D_head: int = 8,
    H: int = 16,
    W: int = 16,
) -> SyntheticProbeSample:
    """Generate one of the 6 required synthetic dynamic probes from binary Q spikes."""
    if pattern not in PATTERN_NAMES:
        raise ValueError(f"Unknown synthetic pattern '{pattern}'. Expected one of {PATTERN_NAMES}.")

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    probs = torch.zeros((T, B, heads, H, W), dtype=torch.float32)
    gt_spatial = torch.zeros((T, B, 1, H, W), dtype=torch.float32)

    cy, cx = H // 2, W // 2

    if pattern == "low_contrast_persistent_bg":
        # Uniform low-contrast persistent background across all timesteps; no foreground target.
        probs.fill_(0.22)
        description = (
            "Uniform low-contrast persistent background (p=0.22); "
            "no target (GT=0); tests history suppression on static/low-contrast texture."
        )

    elif pattern == "high_contrast_moving_target":
        # Compact 2x2 high-contrast target moving diagonally/horizontally across low-activity background.
        probs.fill_(0.05)
        box_h, box_w = min(2, H), min(2, W)
        for t in range(T):
            y0 = min(max(2 + (t // 2), 0), H - box_h)
            x0 = min(max(2 + t, 0), W - box_w)
            probs[t, :, :, y0:y0 + box_h, x0:x0 + box_w] = 0.80
            gt_spatial[t, :, :, y0:y0 + box_h, x0:x0 + box_w] = 1.0
        description = (
            "High-contrast 2x2 moving target (p_fg=0.80) over quiet background (p_bg=0.05); "
            "tests spatial contrast protection on moving contours."
        )

    elif pattern == "high_contrast_flickering_noise":
        # Explicit counterexample: isolated high-contrast 1x1 flickering noise pixels on quiet background,
        # alongside a static 4x4 target. Isolated spikes have high local contrast so C can falsely protect them.
        probs.fill_(0.04)
        ty0, tx0 = max(cy - 2, 0), max(cx - 2, 0)
        ty1, tx1 = min(ty0 + 4, H), min(tx0 + 4, W)
        probs[:, :, :, ty0:ty1, tx0:tx1] = 0.55
        gt_spatial[:, :, :, ty0:ty1, tx0:tx1] = 1.0

        noise_mask = torch.rand((T, B, heads, H, W), generator=gen) < 0.08
        noise_mask[:, :, :, ty0:ty1, tx0:tx1] = False
        probs[noise_mask] = 0.88
        description = (
            "Counterexample: isolated 1x1 high-contrast flickering noise (p=0.88, density=8%) "
            "plus 4x4 target (p=0.55); C can falsely protect isolated noise spikes."
        )

    elif pattern == "low_contrast_true_target":
        # Explicit counterexample: large low-contrast target (6x6, p=0.30) over moderate background (p=0.22).
        # Interior target pixels have q - q_bar ~ 0 < kappa, so C fails to protect and over-suppresses.
        probs.fill_(0.22)
        ty0, tx0 = max(cy - 3, 0), max(cx - 3, 0)
        ty1, tx1 = min(ty0 + 6, H), min(tx0 + 6, W)
        probs[:, :, :, ty0:ty1, tx0:tx1] = 0.30
        gt_spatial[:, :, :, ty0:ty1, tx0:tx1] = 1.0
        description = (
            "Counterexample: large 6x6 low-contrast target (p_fg=0.30) on moderate background (p_bg=0.22); "
            "C can over-suppress when local contrast falls below kappa."
        )

    elif pattern == "high_contrast_static_target":
        # Compact 2x2 high-contrast static target for all T steps; tests H false suppression vs C protection.
        probs.fill_(0.05)
        ty0, tx0 = max(cy - 1, 0), max(cx - 1, 0)
        ty1, tx1 = min(ty0 + 2, H), min(tx0 + 2, W)
        probs[:, :, :, ty0:ty1, tx0:tx1] = 0.65
        gt_spatial[:, :, :, ty0:ty1, tx0:tx1] = 1.0
        description = (
            "Static 2x2 high-contrast target (p_fg=0.65) over quiet background (p_bg=0.05); "
            "tests H false suppression of static targets vs C spatial protection."
        )

    elif pattern == "sparse_occluded_reappearing_target":
        # Target visible in early steps (t=0..2), occluded in middle steps (t=3..6), reappears in late steps (t=7..9).
        probs.fill_(0.05)
        ty0, tx0 = max(cy - 1, 0), max(cx - 1, 0)
        ty1, tx1 = min(ty0 + 3, H), min(tx0 + 3, W)
        occ_start = max(1, T // 3)
        occ_end = min(T - 1, (2 * T) // 3 + 1)
        for t in range(T):
            if t < occ_start or t >= occ_end:
                probs[t, :, :, ty0:ty1, tx0:tx1] = 0.70
                gt_spatial[t, :, :, ty0:ty1, tx0:tx1] = 1.0
        description = (
            f"3x3 target visible for t<{occ_start} and t>={occ_end}, occluded in [{occ_start},{occ_end}); "
            "tests R decay during occlusion and gate recovery upon reappearance."
        )

    q_binary, q_sum = _sample_binary_q_from_probs(probs, D_head=D_head, generator=gen)
    gt_mask = gt_spatial.reshape(T, B, 1, 1, H * W)

    return SyntheticProbeSample(
        pattern=pattern,
        seed=seed,
        T=T,
        B=B,
        heads=heads,
        D_head=D_head,
        H=H,
        W=W,
        q_binary=q_binary,
        q_sum=q_sum,
        gt_mask=gt_mask,
        description=description,
    )


def build_default_gate_configs(
    perm_seeds: Sequence[int] = (101, 202, 303),
) -> list[tuple[str, QKTAConfig]]:
    """Build the pre-registered suite of QKTA gate configurations."""
    configs: list[tuple[str, QKTAConfig]] = [
        ("B0_original", QKTAConfig(mode="B0_original", tau=2.0, theta0=0.5)),
        ("B0_custom", QKTAConfig(mode="B0_custom", tau=2.0, theta0=0.5, b=0.0)),
    ]
    for k in (1, 2, 3, 4):
        configs.append((f"B1_count_k{k}", QKTAConfig(mode="B1_count", k_thresh=k)))
    for theta in (0.5, 1.0, 1.5, 2.0):
        configs.append((f"B1_lif_theta{theta}", QKTAConfig(mode="B1_lif", tau=2.0, theta0=theta)))
    for b in (0.5, 1.0, 2.0, 3.0, 4.0):
        configs.append(
            (f"H_b{b}", QKTAConfig(mode="H", tau=2.0, theta0=0.5, rho=0.5, b=b, eta=0.0))
        )
    for b in (0.5, 1.0, 2.0, 3.0, 4.0):
        configs.append(
            (f"C_b{b}", QKTAConfig(mode="C", tau=2.0, theta0=0.5, rho=0.5, b=b, eta=0.5, kappa=1.0))
        )
    for b in (1.0, 2.0, 3.0):
        for p_seed in perm_seeds:
            configs.append(
                (
                    f"P_b{b}_seed{p_seed}",
                    QKTAConfig(
                        mode="P",
                        tau=2.0,
                        theta0=0.5,
                        rho=0.5,
                        b=b,
                        eta=0.5,
                        kappa=1.0,
                        perm_seed=p_seed,
                    ),
                )
            )
    return configs


def _extract_trace_tensor(
    trace: Optional[dict[str, Any]],
    candidate_keys: Sequence[str],
) -> Optional[torch.Tensor]:
    if not isinstance(trace, dict):
        return None
    for key in candidate_keys:
        if key in trace and isinstance(trace[key], torch.Tensor):
            return trace[key]
    return None


def _safe_div(num: float, den: float) -> Optional[float]:
    if den <= 0:
        return None
    return float(num / den)


def evaluate_gate_on_probe(
    sample: SyntheticProbeSample,
    config_id: str,
    config: QKTAConfig,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run a single gate configuration on a synthetic probe and compute exact metrics + traces."""
    gate = QKTAMaskGate(config)
    with torch.no_grad():
        out = gate(sample.q_sum, sample.H, sample.W, return_trace=True)

    if isinstance(out, tuple):
        A, trace = out[0], out[1]
    else:
        A, trace = out, None

    A = A.to(torch.float32)
    GT = sample.gt_mask.to(torch.float32).expand_as(A)
    not_GT = 1.0 - GT

    tpr_num = float((A * GT).sum().item())
    tpr_den = float(GT.sum().item())
    fpr_num = float((A * not_GT).sum().item())
    fpr_den = float(not_GT.sum().item())

    a_open_count = float(A.sum().item())
    a_total_count = int(A.numel())

    per_step_open_rate = [float(A[t].mean().item()) for t in range(sample.T)]
    per_step_tpr: list[Optional[float]] = []
    per_step_fpr: list[Optional[float]] = []
    for t in range(sample.T):
        t_tpr_den = float(GT[t].sum().item())
        t_fpr_den = float(not_GT[t].sum().item())
        per_step_tpr.append(_safe_div(float((A[t] * GT[t]).sum().item()), t_tpr_den))
        per_step_fpr.append(_safe_div(float((A[t] * not_GT[t]).sum().item()), t_fpr_den))

    r_tensor = _extract_trace_tensor(trace, ("R", "R_trace", "r"))
    theta_tensor = _extract_trace_tensor(trace, ("theta", "theta_trace", "threshold"))
    c_tensor = _extract_trace_tensor(trace, ("c", "c_trace", "c_mask", "contrast_mask"))

    r_trace_mean = (
        [float(r_tensor[t].mean().item()) for t in range(sample.T)]
        if r_tensor is not None
        else None
    )
    theta_trace_mean = (
        [float(theta_tensor[t].mean().item()) for t in range(sample.T)]
        if theta_tensor is not None
        else None
    )
    c_active_rate = float(c_tensor.to(torch.float32).mean().item()) if c_tensor is not None else None

    record: dict[str, Any] = {
        "pattern": sample.pattern,
        "seed": sample.seed,
        "config_id": config_id,
        "mode": config.mode,
        "tau": config.tau,
        "theta0": config.theta0,
        "k_thresh": config.k_thresh,
        "rho": config.rho,
        "b": config.b,
        "eta": config.eta,
        "kappa": config.kappa,
        "perm_seed": config.perm_seed,
        "tpr_num": tpr_num,
        "tpr_den": tpr_den,
        "TPR_gate": _safe_div(tpr_num, tpr_den),
        "fpr_num": fpr_num,
        "fpr_den": fpr_den,
        "FPR_gate": _safe_div(fpr_num, fpr_den),
        "a_open_count": a_open_count,
        "a_total_count": a_total_count,
        "A_open_rate": _safe_div(a_open_count, float(a_total_count)),
        "c_active_rate": c_active_rate,
        "per_step_open_rate": per_step_open_rate,
        "per_step_tpr": per_step_tpr,
        "per_step_fpr": per_step_fpr,
        "R_trace_mean": r_trace_mean,
        "theta_trace_mean": theta_trace_mean,
    }

    arrays: dict[str, np.ndarray] = {
        "A": A.cpu().numpy(),
    }
    if r_tensor is not None:
        arrays["R"] = r_tensor.cpu().numpy()
    if theta_tensor is not None:
        arrays["theta"] = theta_tensor.cpu().numpy()
    if c_tensor is not None:
        arrays["c"] = c_tensor.cpu().numpy()

    return record, arrays


def _compute_c_vs_h_deltas(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute paired C vs H deltas for matching (pattern, b) pairs."""
    by_key = {(r["pattern"], r["mode"], r["b"]): r for r in records}
    deltas: list[dict[str, Any]] = []
    for (pattern, mode, b), c_rec in sorted(by_key.items()):
        if mode != "C":
            continue
        h_rec = by_key.get((pattern, "H", b))
        if h_rec is None:
            continue
        tpr_c, tpr_h = c_rec["TPR_gate"], h_rec["TPR_gate"]
        fpr_c, fpr_h = c_rec["FPR_gate"], h_rec["FPR_gate"]
        open_c, open_h = c_rec["A_open_rate"], h_rec["A_open_rate"]
        deltas.append(
            {
                "pattern": pattern,
                "b": b,
                "TPR_C": tpr_c,
                "TPR_H": tpr_h,
                "delta_TPR_C_minus_H": (
                    (tpr_c - tpr_h) if (tpr_c is not None and tpr_h is not None) else None
                ),
                "FPR_C": fpr_c,
                "FPR_H": fpr_h,
                "delta_FPR_C_minus_H": (
                    (fpr_c - fpr_h) if (fpr_c is not None and fpr_h is not None) else None
                ),
                "A_open_rate_C": open_c,
                "A_open_rate_H": open_h,
                "delta_open_rate_C_minus_H": (
                    (open_c - open_h) if (open_c is not None and open_h is not None) else None
                ),
            }
        )
    return deltas


def _summarize_perm_controls(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate permutation negative controls (P) across perm_seeds."""
    groups: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for r in records:
        if r["mode"] != "P":
            continue
        groups.setdefault((r["pattern"], float(r["b"])), []).append(r)

    summaries: list[dict[str, Any]] = []
    for (pattern, b), items in sorted(groups.items()):
        tprs = [x["TPR_gate"] for x in items if x["TPR_gate"] is not None]
        fprs = [x["FPR_gate"] for x in items if x["FPR_gate"] is not None]
        opens = [x["A_open_rate"] for x in items if x["A_open_rate"] is not None]
        summaries.append(
            {
                "pattern": pattern,
                "b": b,
                "num_seeds": len(items),
                "perm_seeds": [x["perm_seed"] for x in items],
                "TPR_mean": float(np.mean(tprs)) if tprs else None,
                "TPR_min": float(np.min(tprs)) if tprs else None,
                "TPR_max": float(np.max(tprs)) if tprs else None,
                "FPR_mean": float(np.mean(fprs)) if fprs else None,
                "FPR_min": float(np.min(fprs)) if fprs else None,
                "FPR_max": float(np.max(fprs)) if fprs else None,
                "A_open_rate_mean": float(np.mean(opens)) if opens else None,
            }
        )
    return summaries


def run_synthetic_suite(
    output_dir: str | Path,
    seed: int = 42,
    T: int = 10,
    B: int = 1,
    heads: int = 1,
    D_head: int = 8,
    H: int = 16,
    W: int = 16,
    perm_seeds: Sequence[int] = (101, 202, 303),
) -> dict[str, Any]:
    """Run all 6 synthetic probes across all gate configurations and save CSV, JSON, and NPZ traces."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    gate_configs = build_default_gate_configs(perm_seeds=perm_seeds)
    records: list[dict[str, Any]] = []
    npz_payload: dict[str, np.ndarray] = {}
    pattern_meta: dict[str, Any] = {}

    for idx, pattern in enumerate(PATTERN_NAMES):
        pattern_seed = seed + idx
        sample = generate_synthetic_probe(
            pattern=pattern,
            seed=pattern_seed,
            T=T,
            B=B,
            heads=heads,
            D_head=D_head,
            H=H,
            W=W,
        )
        pattern_meta[pattern] = {
            "seed": pattern_seed,
            "description": sample.description,
            "q_sum_mean": float(sample.q_sum.mean().item()),
            "q_nonzero_fraction": float((sample.q_sum > 0).to(torch.float32).mean().item()),
            "gt_positive_fraction": float(sample.gt_mask.mean().item()),
        }
        npz_payload[f"{pattern}__q_sum"] = sample.q_sum.cpu().numpy()
        npz_payload[f"{pattern}__gt_mask"] = sample.gt_mask.cpu().numpy()

        for config_id, cfg in gate_configs:
            rec, arrays = evaluate_gate_on_probe(sample, config_id=config_id, config=cfg)
            records.append(rec)
            for arr_name, arr_val in arrays.items():
                npz_payload[f"{pattern}__{config_id}__{arr_name}"] = arr_val

    c_vs_h_deltas = _compute_c_vs_h_deltas(records)
    perm_summaries = _summarize_perm_controls(records)

    csv_path = out_path / "synthetic_results.csv"
    scalar_fields = [
        "pattern",
        "seed",
        "config_id",
        "mode",
        "tau",
        "theta0",
        "k_thresh",
        "rho",
        "b",
        "eta",
        "kappa",
        "perm_seed",
        "tpr_num",
        "tpr_den",
        "TPR_gate",
        "fpr_num",
        "fpr_den",
        "FPR_gate",
        "a_open_count",
        "a_total_count",
        "A_open_rate",
        "c_active_rate",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=scalar_fields)
        writer.writeheader()
        for rec in records:
            row = {k: rec[k] for k in scalar_fields}
            writer.writerow(row)

    json_path = out_path / "synthetic_results.json"
    summary_payload = {
        "base_seed": seed,
        "dimensions": {"T": T, "B": B, "heads": heads, "D_head": D_head, "H": H, "W": W},
        "perm_seeds": list(perm_seeds),
        "patterns": pattern_meta,
        "results": records,
        "c_vs_h_deltas": c_vs_h_deltas,
        "perm_control_summaries": perm_summaries,
        "artifacts": {
            "csv": str(csv_path),
            "json": str(json_path),
            "traces_npz": str(out_path / "synthetic_traces.npz"),
        },
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    npz_path = out_path / "synthetic_traces.npz"
    np.savez_compressed(npz_path, **npz_payload)

    return summary_payload


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QKTA synthetic dynamic visual probes.")
    parser.add_argument("--output-dir", type=str, default="outputs/qkta/synthetic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument("--D-head", type=int, default=8)
    parser.add_argument("--H", type=int, default=16)
    parser.add_argument("--W", type=int, default=16)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    result = run_synthetic_suite(
        output_dir=args.output_dir,
        seed=args.seed,
        T=args.T,
        D_head=args.D_head,
        H=args.H,
        W=args.W,
    )
    print(
        f"Synthetic probe suite complete: {len(result['results'])} runs saved to {args.output_dir}"
    )
