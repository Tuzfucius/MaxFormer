from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_qkta.gates import QKTAConfig, QKTAMaskGate


@pytest.mark.parametrize(
    "q_sum,H,W",
    [
        (torch.zeros(2, 3, 4, 5), 1, 20),
        (torch.zeros(2, 3, 4, 2, 6), 2, 6),
        (torch.zeros(2, 3, 4, 1, 7), 2, 4),
    ],
)
def test_invalid_q_sum_shapes_raise_assertion(q_sum: torch.Tensor, H: int, W: int) -> None:
    with pytest.raises(AssertionError):
        QKTAMaskGate()(q_sum, H=H, W=W)


@pytest.mark.parametrize("H,W", ((3, 5), (1, 8), (7, 2)))
def test_non_square_grids_preserve_shape_dtype_device(H: int, W: int) -> None:
    q_sum = torch.randint(0, 4, (2, 3, 2, 1, H * W), dtype=torch.int32)
    result = QKTAMaskGate()(q_sum, H=H, W=W)
    assert result.shape == q_sum.shape
    assert result.dtype == q_sum.dtype
    assert result.device == q_sum.device


def test_attention_broadcast_matches_key_shape_and_is_binary() -> None:
    T, B, heads, D_head, H, W = 4, 2, 3, 8, 3, 5
    q_sum = torch.randint(0, 5, (T, B, heads, 1, H * W)).float()
    k = torch.randint(0, 2, (T, B, heads, D_head, H * W)).float()
    A = QKTAMaskGate()(q_sum, H=H, W=W)
    post_k = A * k
    assert post_k.shape == (T, B, heads, D_head, H * W)
    assert set(torch.unique(A).tolist()) <= {0.0, 1.0}


def test_neighbor_means_use_exact_corner_edge_and_interior_counts() -> None:
    gate = QKTAMaskGate(QKTAConfig(mode="C", kappa=0.5))
    grid = torch.zeros((1, 1, 1, 3, 3))
    # Set the 3 neighbors of corner (0,0), 5 neighbors of edge (0,1), and
    # all 8 neighbors of interior (1,1) to one.
    grid[0, 0, 0, 0, 1] = 1
    grid[0, 0, 0, 1, 0] = 1
    grid[0, 0, 0, 1, 1] = 1
    grid[0, 0, 0, 0, 0] = 1
    grid[0, 0, 0, 0, 2] = 1
    grid[0, 0, 0, 1, 2] = 1
    grid[0, 0, 0, 2, 0] = 1
    grid[0, 0, 0, 2, 1] = 1
    grid[0, 0, 0, 2, 2] = 1
    _, trace = gate(grid.flatten(3).unsqueeze(3), H=3, W=3, return_trace=True)
    q_bar = trace["q_bar_trace"].reshape(3, 3)
    assert q_bar[0, 0].item() == pytest.approx(3 / 3)
    assert q_bar[0, 1].item() == pytest.approx(5 / 5)
    assert q_bar[1, 1].item() == pytest.approx(8 / 8)
    assert trace["c_trace"][0, 0, 0, 0].item() == 0  # traces are [T, B, heads, N]


def test_single_pixel_grid_has_zero_contrast_and_finite_traces() -> None:
    q_sum = torch.tensor([[[[[0.0]]]], [[[[1.0]]]]])
    output, trace = QKTAMaskGate(QKTAConfig(mode="C"))(q_sum, H=1, W=1, return_trace=True)
    assert torch.equal(trace["c_trace"], torch.zeros_like(trace["c_trace"]))
    for tensor in (output, *trace.values()):
        assert torch.isfinite(tensor).all()
