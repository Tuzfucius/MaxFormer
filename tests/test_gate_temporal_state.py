from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_qkta.gates import QKTAConfig, QKTAMaskGate, channel_or_gate


def test_history_starts_at_zero_and_updates_after_current_spike() -> None:
    rho, b, theta0 = 0.5, 2.0, 0.5
    q = torch.tensor([2.0, 0.0, 2.0]).reshape(3, 1, 1, 1, 1)
    gate = QKTAMaskGate(QKTAConfig(mode="H", rho=rho, b=b, theta0=theta0))
    A, trace = gate(q, H=1, W=1, return_trace=True)
    R, theta = trace["R_trace"], trace["theta_trace"]

    assert torch.equal(R[0], torch.zeros_like(R[0]))
    assert torch.equal(theta[0], torch.full_like(theta[0], theta0))
    for t in range(q.shape[0] - 1):
        assert torch.equal(R[t + 1], rho * R[t] + (1.0 - rho) * A[t, :, :, 0, :])
    for t in range(q.shape[0]):
        assert torch.equal(theta[t], theta0 + b * R[t])
    # This step distinguishes using pre-update R[1]=0.5 (threshold 1.5)
    # from accidentally using R[2]=0.25 (threshold 1.0).
    assert theta[1].item() == 1.5
    assert A[1].item() == 0


def test_gate_calls_have_no_cross_call_state_leakage() -> None:
    q_high = torch.full((4, 2, 2, 1, 9), 3.0)
    q_low = torch.zeros_like(q_high)
    cfg = QKTAConfig(mode="H", rho=0.8, b=2.0, theta0=0.5)
    shared = QKTAMaskGate(cfg)
    shared(q_high, H=3, W=3)
    after_high = shared(q_low, H=3, W=3)
    fresh = QKTAMaskGate(cfg)(q_low, H=3, W=3)
    assert torch.equal(after_high, fresh)


def test_eta_zero_makes_c_identical_to_h_bit_for_bit() -> None:
    generator = torch.Generator().manual_seed(440)
    inputs = [
        torch.randint(0, 5, (5, 2, 3, 1, 15), generator=generator).float(),
        torch.zeros((3, 1, 2, 1, 8)),
        torch.ones((4, 2, 1, 1, 6)),
    ]
    for b in (0.5, 1.0, 2.0, 4.0):
        h_gate = QKTAMaskGate(QKTAConfig(mode="H", b=b, eta=0.0, rho=0.7))
        c_gate = QKTAMaskGate(QKTAConfig(mode="C", b=b, eta=0.0, rho=0.7))
        for q in inputs:
            height, width = (3, 5) if q.shape[-1] == 15 else ((1, 8) if q.shape[-1] == 8 else (2, 3))
            assert torch.equal(h_gate(q, H=height, W=width), c_gate(q, H=height, W=width))


def test_zero_history_strength_and_default_threshold_reduce_to_channel_or() -> None:
    generator = torch.Generator().manual_seed(83)
    q = torch.randint(0, 5, (6, 2, 3, 1, 15), generator=generator).float()
    q = torch.cat([q, torch.zeros_like(q), torch.ones_like(q)], dim=0)
    for mode in ("H", "C"):
        gate = QKTAMaskGate(QKTAConfig(mode=mode, b=0.0, theta0=0.5, eta=0.8))
        assert torch.equal(gate(q, H=3, W=5), channel_or_gate(q))


def test_permutation_control_preserves_counts_and_is_seeded() -> None:
    generator = torch.Generator().manual_seed(2025)
    q = torch.randint(0, 5, (4, 2, 3, 1, 24), generator=generator).float()
    configs = [QKTAConfig(mode="P", b=2.0, rho=0.5, eta=0.9, perm_seed=seed) for seed in (17, 17, 29)]
    traces = [QKTAMaskGate(config)(q, H=4, W=6, return_trace=True) for config in configs]
    _, control_trace = QKTAMaskGate(
        QKTAConfig(mode="C", b=2.0, rho=0.5, eta=0.9, perm_seed=17)
    )(q, H=4, W=6, return_trace=True)
    A0, trace0 = traces[0]
    A1, trace1 = traces[1]
    A2, trace2 = traces[2]

    assert torch.equal(A0, A1)
    assert torch.equal(trace0["c_trace"], trace1["c_trace"])
    assert torch.equal(trace0["c_trace"].sum(dim=-1), trace2["c_trace"].sum(dim=-1))
    assert torch.equal(trace0["c_trace"].sum(dim=-1), trace1["c_trace"].sum(dim=-1))
    assert torch.equal(trace0["c_trace"].sum(dim=-1), control_trace["c_trace"].sum(dim=-1))
    # The deterministic samples contain mixed contrast positions; a different
    # seed should move at least one spatial position.
    assert not torch.equal(trace0["c_trace"], trace2["c_trace"])
    assert trace0["c_trace"].sum().item() > 0
