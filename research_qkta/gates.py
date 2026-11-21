from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

VALID_MODES = frozenset(
    {
        "B0_original",
        "B0_custom",
        "B1_lif",
        "B1_count",
        "H",
        "C",
        "P",
    }
)


def reference_lif_step(
    q_count: torch.Tensor,
    tau: float = 2.0,
    v_threshold: float = 0.5,
    v_reset: Optional[float] = 0.0,
    v_init: Union[float, torch.Tensor] = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch reference LIF truth machine over dim-0 time steps.

    Supports both hard reset (`v_reset=0.0`) and soft reset (`v_reset=None`).
    Returns `(spikes, u_post_reset_trace)` with the same shape as `q_count`.
    """
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    if q_count.ndim < 1:
        raise ValueError(f"q_count must have at least 1 dimension (time), got shape {q_count.shape}")

    work_dtype = q_count.dtype if q_count.is_floating_point() else torch.float32
    q_work = q_count.to(dtype=work_dtype)
    T = q_work.shape[0]
    step_shape = q_work.shape[1:]

    if isinstance(v_init, torch.Tensor):
        u = v_init.to(device=q_work.device, dtype=work_dtype).expand(step_shape).clone()
    else:
        u = torch.full(step_shape, float(v_init), device=q_work.device, dtype=work_dtype)

    decay = 1.0 - 1.0 / float(tau)
    drive = 1.0 / float(tau)

    spikes = []
    u_trace = []
    for t in range(T):
        h = decay * u + drive * q_work[t]
        spk = (h >= v_threshold).to(dtype=work_dtype)
        if v_reset is None:
            u = h - float(v_threshold) * spk
        else:
            u = h * (1.0 - spk) + float(v_reset) * spk
        spikes.append(spk)
        u_trace.append(u)

    return torch.stack(spikes, dim=0), torch.stack(u_trace, dim=0)


def channel_or_gate(q_count: torch.Tensor) -> torch.Tensor:
    """Memoryless channel OR gate: 1[q_count >= 1]."""
    out_dtype = q_count.dtype if q_count.is_floating_point() else torch.float32
    return (q_count >= 1).to(dtype=out_dtype)


@dataclass
class QKTAConfig:
    mode: str = "B0_custom"
    tau: float = 2.0
    theta0: float = 0.5
    k_thresh: int = 1
    rho: float = 0.5
    b: float = 0.0
    eta: float = 0.5
    kappa: float = 1.0
    perm_seed: int = 0

    def __post_init__(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(f"Unsupported mode {self.mode!r}; expected one of {sorted(VALID_MODES)}")
        if self.tau <= 0:
            raise ValueError(f"tau must be > 0, got {self.tau}")
        if not (0.0 <= self.rho <= 1.0):
            raise ValueError(f"rho must be in [0, 1], got {self.rho}")


class QKTAMaskGate(nn.Module):
    """Unified pure-PyTorch QKTA gate implementing B0_custom, B1_lif, B1_count, H, C, and P."""

    def __init__(self, config: Optional[QKTAConfig] = None) -> None:
        super().__init__()
        self.config = config if config is not None else QKTAConfig()
        kernel = torch.tensor(
            [[[[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]]],
            dtype=torch.float32,
        )
        self.register_buffer("_neighbor_kernel", kernel, persistent=False)

    def _compute_neighbor_mean(
        self,
        q_count: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """Compute 8-neighborhood mean q_bar with exact boundary normalization (3/5/8)."""
        T, B, heads, N = q_count.shape
        if N == 1:
            return torch.zeros_like(q_count)

        kernel = self._neighbor_kernel.to(device=q_count.device, dtype=q_count.dtype)
        q_grid = q_count.reshape(T * B * heads, 1, H, W)
        neighbor_sum = F.conv2d(q_grid, kernel, padding=1)

        ones = torch.ones((1, 1, H, W), device=q_count.device, dtype=q_count.dtype)
        neighbor_count = F.conv2d(ones, kernel, padding=1).clamp_min(1.0)
        q_bar = (neighbor_sum / neighbor_count).reshape(T, B, heads, N)
        return q_bar

    def _permute_contrast(
        self,
        c_t: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        """Independently permute spatial tokens [0..N-1] per (t, b, head) on CPU generator."""
        B, heads, N = c_t.shape
        if N <= 1:
            return c_t
        out = torch.empty_like(c_t)
        gen = torch.Generator(device="cpu")
        base_seed = int(self.config.perm_seed)
        for b_idx in range(B):
            for h_idx in range(heads):
                seed = (
                    (base_seed * 1_000_003)
                    + (t * 10_007)
                    + (b_idx * 101)
                    + h_idx
                ) & 0x7FFF_FFFF
                gen.manual_seed(seed)
                perm = torch.randperm(N, generator=gen).to(device=c_t.device)
                out[b_idx, h_idx] = c_t[b_idx, h_idx, perm]
        return out

    def forward(
        self,
        q_sum: torch.Tensor,
        H: int,
        W: int,
        return_trace: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if q_sum.ndim != 5:
            raise AssertionError(f"Expected 5D q_sum [T, B, heads, 1, N], got shape {tuple(q_sum.shape)}")
        if q_sum.shape[3] != 1:
            raise AssertionError(f"Expected q_sum.shape[3] == 1, got shape {tuple(q_sum.shape)}")
        if H <= 0 or W <= 0:
            raise AssertionError(f"H and W must be positive, got H={H}, W={W}")
        if q_sum.shape[4] != H * W:
            raise AssertionError(
                f"Expected q_sum.shape[4] == H * W ({H} * {W} = {H * W}), got {q_sum.shape[4]}"
            )

        out_dtype = q_sum.dtype
        work_dtype = out_dtype if out_dtype in (torch.float32, torch.float64) else torch.float32
        q_count = q_sum.squeeze(3).to(dtype=work_dtype)  # [T, B, heads, N]
        T, B, heads, N = q_count.shape
        device = q_count.device
        cfg = self.config

        if cfg.mode == "B0_custom":
            tau = 2.0
            theta0 = 0.5
            b_val = 0.0
        else:
            tau = float(cfg.tau)
            theta0 = float(cfg.theta0)
            b_val = float(cfg.b)

        decay = 1.0 - 1.0 / tau
        drive = 1.0 / tau
        rho = float(cfg.rho)
        eta = float(cfg.eta)

        q_bar = self._compute_neighbor_mean(q_count, H=H, W=W)
        if N == 1:
            c_raw = torch.zeros_like(q_count)
        else:
            c_raw = (q_count - q_bar >= float(cfg.kappa)).to(dtype=work_dtype)

        # Fresh local state per forward call (zero cross-call leakage).
        u = torch.zeros((B, heads, N), device=device, dtype=work_dtype)
        R = torch.zeros((B, heads, N), device=device, dtype=work_dtype)

        A_list = []
        R_trace = []
        theta_trace = []
        c_trace = []
        u_trace = []

        for t in range(T):
            h_t = decay * u + drive * q_count[t]
            c_used = torch.zeros((B, heads, N), device=device, dtype=work_dtype)

            if cfg.mode == "B1_count":
                theta_t = torch.full((B, heads, N), float(cfg.k_thresh), device=device, dtype=work_dtype)
                A_t = (q_count[t] >= int(cfg.k_thresh)).to(dtype=work_dtype)
                u = torch.zeros_like(u)
            else:
                if cfg.mode in ("B0_custom", "B1_lif", "B0_original"):
                    theta_t = torch.full((B, heads, N), theta0, device=device, dtype=work_dtype)
                elif cfg.mode == "H":
                    theta_t = theta0 + b_val * R
                elif cfg.mode == "C":
                    c_used = c_raw[t]
                    theta_t = theta0 + b_val * R * (1.0 - eta * c_used)
                elif cfg.mode == "P":
                    c_used = self._permute_contrast(c_raw[t], t=t)
                    theta_t = theta0 + b_val * R * (1.0 - eta * c_used)
                else:
                    raise ValueError(f"Unsupported mode: {cfg.mode}")

                A_t = (h_t >= theta_t).to(dtype=work_dtype)
                u = h_t * (1.0 - A_t)

            # Record pre-update R_t (used for A_t), then update R_{t+1} strictly after A_t.
            if return_trace:
                R_trace.append(R.clone())
                theta_trace.append(theta_t.clone())
                c_trace.append(c_used.clone())
                u_trace.append(u.clone())

            R = rho * R + (1.0 - rho) * A_t
            A_list.append(A_t)

        attn = torch.stack(A_list, dim=0).unsqueeze(3).to(dtype=out_dtype)
        if not return_trace:
            return attn

        trace = {
            "R_trace": torch.stack(R_trace, dim=0),
            "theta_trace": torch.stack(theta_trace, dim=0),
            "c_trace": torch.stack(c_trace, dim=0),
            "u_trace": torch.stack(u_trace, dim=0),
            "q_bar_trace": q_bar,
        }
        return attn, trace
