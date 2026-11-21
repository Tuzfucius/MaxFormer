from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_qkta.gates import QKTAConfig, QKTAMaskGate, channel_or_gate, reference_lif_step


def _official_lif(q: torch.Tensor, **kwargs) -> torch.Tensor:
    from spikingjelly.clock_driven import functional
    from spikingjelly.clock_driven.neuron import MultiStepLIFNode

    node = MultiStepLIFNode(
        tau=kwargs.get("tau", 2.0),
        v_threshold=kwargs.get("v_threshold", 0.5),
        v_reset=kwargs.get("v_reset", 0.0),
        detach_reset=True,
        backend="torch",
    ).to(q.device)
    functional.reset_net(node)
    try:
        with torch.no_grad():
            return node(q)
    finally:
        functional.reset_net(node)


def _assert_equal_all(*values: torch.Tensor) -> None:
    for value in values[1:]:
        assert torch.equal(values[0], value)


@pytest.mark.parametrize("T", (1, 2, 4))
def test_exhaustive_small_grid_matches_reference_custom_and_official_lif(T: int) -> None:
    # Each batch member is one of the 5**T possible integer sequences. Spatial
    # positions and heads carry the same sequence to keep this exhaustive test compact.
    rows = list(itertools.product(range(5), repeat=T))
    q = torch.tensor(rows, dtype=torch.float32).T[:, :, None, None, None]
    q = q.expand(T, len(rows), 2, 1, 3).contiguous()

    expected, u_trace = reference_lif_step(q, tau=2.0, v_threshold=0.5, v_reset=0.0, v_init=0.0)
    or_result = channel_or_gate(q)
    gate_result = QKTAMaskGate(QKTAConfig(mode="B0_custom"))(q, H=1, W=3)
    official = _official_lif(q)

    _assert_equal_all(expected, or_result, gate_result, official)
    assert torch.count_nonzero(u_trace) == 0


@pytest.mark.parametrize("D_head", (4, 8, 16, 32))
@pytest.mark.parametrize("T", (1, 2, 4, 16))
def test_random_binary_multiscale_matches_all_implementations(T: int, D_head: int) -> None:
    generator = torch.Generator().manual_seed(1000 + T * 31 + D_head)
    B, heads, H, W = 3, 2, 3, 5
    q_binary = torch.randint(0, 2, (T, B, heads, D_head, H * W), generator=generator).float()
    q_sum = q_binary.sum(dim=3, keepdim=True)

    reference, u_trace = reference_lif_step(q_sum, tau=2.0, v_threshold=0.5, v_reset=0.0, v_init=0.0)
    result = channel_or_gate(q_sum)
    custom = QKTAMaskGate(QKTAConfig(mode="B0_custom"))(q_sum, H=H, W=W)
    official = _official_lif(q_sum)

    _assert_equal_all(reference, result, custom, official)
    assert torch.count_nonzero(u_trace) == 0


def test_zero_ones_batch_independence_order_and_repeated_reset() -> None:
    from spikingjelly.clock_driven import functional
    from spikingjelly.clock_driven.neuron import MultiStepLIFNode

    gate = QKTAMaskGate(QKTAConfig(mode="B0_custom"))
    q_zero = torch.zeros((4, 3, 2, 1, 6))
    q_one = torch.ones_like(q_zero)
    assert torch.count_nonzero(gate(q_zero, H=2, W=3)) == 0
    assert torch.equal(gate(q_one, H=2, W=3), torch.ones_like(q_one))

    q = torch.randint(0, 5, (4, 3, 2, 1, 6)).float()
    full = gate(q, H=2, W=3)
    individual = torch.cat([gate(q[:, i : i + 1], H=2, W=3) for i in range(q.shape[1])], dim=1)
    permutation = torch.tensor([2, 0, 1])
    reordered = gate(q[:, permutation], H=2, W=3)
    assert torch.equal(full, individual)
    assert torch.equal(reordered, full[:, permutation])

    node = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend="torch")
    outputs = []
    for _ in range(2):
        functional.reset_net(node)
        with torch.no_grad():
            outputs.append(node(q.squeeze(3)))
        functional.reset_net(node)
    assert torch.equal(outputs[0], outputs[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_optional_cupy_backend_matches_custom_gate() -> None:
    from spikingjelly.clock_driven import functional
    from spikingjelly.clock_driven.neuron import MultiStepLIFNode

    q = torch.randint(0, 2, (4, 2, 3, 1, 15), device="cuda").float()
    expected = QKTAMaskGate(QKTAConfig(mode="B0_custom"))(q, H=3, W=5)
    try:
        node = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy").cuda()
        functional.reset_net(node)
        with torch.no_grad():
            actual = node(q)
        torch.cuda.synchronize()
    except Exception as exc:
        pytest.skip(f"SpikingJelly CuPy backend is unavailable or incompatible: {exc}")
    finally:
        if "node" in locals():
            functional.reset_net(node)
    assert torch.equal(actual, expected)


def test_or_equivalence_counterexamples_when_assumptions_are_violated() -> None:
    q_fractional = torch.tensor([[0.6]])
    spikes, _ = reference_lif_step(q_fractional)
    assert spikes.item() == 0 and channel_or_gate(q_fractional).item() == 0
    q_accumulate = torch.tensor([[0.8], [0.8]])
    spikes, _ = reference_lif_step(q_accumulate)
    assert spikes[1].item() == 1 and channel_or_gate(q_accumulate)[1].item() == 0

    q_zero = torch.tensor([[0.0]])
    spikes, _ = reference_lif_step(q_zero, v_init=1.0)
    assert spikes[0].item() == 1 and channel_or_gate(q_zero)[0].item() == 0

    q_threshold = torch.tensor([[1.0], [2.0]])
    spikes, _ = reference_lif_step(q_threshold, v_threshold=1.0)
    assert spikes[0].item() == 0 and channel_or_gate(q_threshold)[0].item() == 1

    q_tau = torch.tensor([[1.0], [1.0]])
    spikes, _ = reference_lif_step(q_tau, tau=4.0)
    assert torch.equal(spikes, torch.zeros_like(spikes))
    assert torch.equal(channel_or_gate(q_tau), torch.ones_like(q_tau))

    q_soft = torch.tensor([[3.0], [0.0]])
    spikes, _ = reference_lif_step(q_soft, v_reset=None)
    assert spikes[1].item() == 1 and channel_or_gate(q_soft)[1].item() == 0
