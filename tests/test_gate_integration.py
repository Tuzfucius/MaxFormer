from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_qkta.adapter import QKTAExperimentContext, discover_qkta_layers, set_lif_backend
from research_qkta.frozen_eval import _import_track_model
from research_qkta.gates import QKTAConfig
from research_qkta.telemetry import QKTATelemetry


def _reset(model: torch.nn.Module) -> None:
    from spikingjelly.clock_driven import functional

    functional.reset_net(model)


def _make_model(track: str) -> torch.nn.Module:
    if track == "imagenet":
        model = _import_track_model("imagenet", "ms_qkformer", "ms_qkformer_10_384", T=4)
        expected = ["stage1.0.attn", "stage2.0.attn", "stage2.1.attn"]
    else:
        model = _import_track_model(
            "cifar10-100", "ms_qkformer", "MS_QKformer",
            in_channels=3, num_classes=10, embed_dims=384, T=4,
        )
        expected = ["stage1.0.attn", "stage2.0.attn"]
    set_lif_backend(model, "torch")
    model.eval()
    assert [name for name, _ in discover_qkta_layers(model)] == expected
    return model


@pytest.mark.parametrize("track", ("imagenet", "cifar"))
def test_upstream_model_discovery_frozen_logits_context_and_temporal_lengths(track: str) -> None:
    from spikingjelly.clock_driven import functional

    torch.manual_seed(51)
    model = _make_model(track).cpu()
    layers = discover_qkta_layers(model)
    original_forwards = {name: module.forward for name, module in layers}
    batch = torch.randn(2, 3, 32, 32)
    model.eval()
    assert model.training is False

    with torch.no_grad():
        _reset(model)
        original_logits = model(batch)
        _reset(model)
        for target in ("stage1", "all_qkta"):
            telemetry = QKTATelemetry(enabled=True)
            try:
                with QKTAExperimentContext(
                    model,
                    QKTAConfig(mode="B0_custom"),
                    target_layers=target,
                    telemetry=telemetry,
                ):
                    _reset(model)
                    patched_logits = model(batch)
                diff = float((original_logits - patched_logits).abs().max().item())
                print(f"{track} {target} B0_original-vs-B0_custom max_abs_diff={diff:.9g}")
                assert torch.allclose(original_logits, patched_logits, atol=1e-5, rtol=1e-5)
            finally:
                _reset(model)
            assert telemetry.summary()
            for stats in telemetry.summary().values():
                assert stats["A_open_count"] >= 0
                assert stats["K_pre_spikes"] >= 0
                assert 0 <= stats["K_post_spikes"] <= stats["K_pre_spikes"]
            telemetry.reset()
            assert telemetry.summary() == {}
            assert all(module.forward == original_forwards[name] for name, module in layers)

        disabled = QKTATelemetry(enabled=False)
        with QKTAExperimentContext(model, QKTAConfig(mode="B0_custom"), telemetry=disabled):
            _reset(model)
            model(batch)
        _reset(model)
        assert disabled.summary() == {}

        # Repeated calls and batch-order permutations are stable after reset.
        _reset(model)
        first = model(batch)
        _reset(model)
        second = model(batch)
        _reset(model)
        reordered = model(batch.flip(0)).flip(0)
        _reset(model)
        assert torch.allclose(first, second, atol=1e-5, rtol=1e-5)
        assert torch.allclose(first, reordered, atol=1e-5, rtol=1e-5)

        for T in (2, 4):
            model.T = T
            _reset(model)
            logits = model(batch)
            assert logits.shape[0] == batch.shape[0]
        _reset(model)

    assert model.training is False
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(module.forward == original_forwards[name] for name, module in layers)


def test_telemetry_counts_match_manual_tensors_and_reset() -> None:
    telemetry = QKTATelemetry(enabled=True)
    q_sum = torch.tensor([[[[[0.0, 1.0]]]], [[[[2.0, 1.0]]]]])
    attn = (q_sum >= 1).float()
    k = torch.tensor([[[[[1.0, 0.0], [1.0, 1.0]]]], [[[[0.0, 1.0], [1.0, 1.0]]]]])
    post_k = attn * k
    telemetry.record("manual", q_sum, attn, k, post_k)
    stats = telemetry.summary()["manual"]
    assert stats["A_open_count"] == int(attn.sum().item())
    assert stats["K_pre_spikes"] == int(k.sum().item())
    assert stats["K_post_spikes"] == int(post_k.sum().item())
    assert 0 <= stats["K_post_spikes"] <= stats["K_pre_spikes"]
    telemetry.reset()
    assert telemetry.summary() == {}

    disabled = QKTATelemetry(enabled=False)
    disabled.record("manual", q_sum, attn, k, post_k)
    assert disabled.summary() == {}
