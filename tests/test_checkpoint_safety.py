from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_qkta.checkpoint_audit import (
    CheckpointAuditError,
    IMAGENETTE_WNID_TO_IMAGENET1K_IDX,
    audit_and_load_checkpoint,
    verify_imagenette_wnid_mapping,
)


def _model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 3), nn.BatchNorm1d(3))


def _snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def _assert_unchanged(model: nn.Module, before: dict[str, torch.Tensor]) -> None:
    after = model.state_dict()
    assert after.keys() == before.keys()
    assert all(torch.equal(after[name], before[name]) for name in before)


def test_missing_corrupt_and_non_mapping_checkpoints_raise(tmp_path: Path) -> None:
    model = _model()
    with pytest.raises(CheckpointAuditError):
        audit_and_load_checkpoint(model, tmp_path / "missing.pth", strict=True)

    corrupt = tmp_path / "corrupt.pth"
    corrupt.write_bytes(b"this is not a torch checkpoint")
    with pytest.raises(CheckpointAuditError):
        audit_and_load_checkpoint(model, corrupt, strict=True)

    non_mapping = tmp_path / "non_mapping.pth"
    torch.save(["not", "a", "state_dict"], non_mapping)
    with pytest.raises(CheckpointAuditError):
        audit_and_load_checkpoint(model, non_mapping, strict=True)


@pytest.mark.parametrize("kind", ("missing", "unexpected", "shape"))
def test_strict_mismatch_raises_without_mutating_target(kind: str, tmp_path: Path) -> None:
    model = _model()
    before = _snapshot(model)
    state = {key: value.detach().clone() for key, value in before.items()}
    if kind == "missing":
        state.pop(next(iter(state)))
    elif kind == "unexpected":
        state["unexpected.weight"] = torch.ones(2)
    else:
        key = next(name for name in state if name.endswith("weight") and state[name].ndim == 2)
        state[key] = torch.ones((state[key].shape[0] + 1, state[key].shape[1]))
    path = tmp_path / f"{kind}.pth"
    torch.save({"state_dict": state}, path)

    with pytest.raises(CheckpointAuditError):
        audit_and_load_checkpoint(model, path, strict=True)
    _assert_unchanged(model, before)


def test_exact_state_dict_and_module_prefixed_state_dict_load_with_sha256(tmp_path: Path) -> None:
    source = _model()
    target = _model()
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.add_(torch.randn_like(parameter))
    state = {key: value.detach().clone() for key, value in source.state_dict().items()}
    plain_path = tmp_path / "plain.pth"
    torch.save({"state_dict": state}, plain_path)
    report = audit_and_load_checkpoint(target, plain_path, strict=True)
    expected_sha = hashlib.sha256(plain_path.read_bytes()).hexdigest()
    assert report.is_exact_match is True
    assert report.sha256 == expected_sha
    assert all(torch.equal(target.state_dict()[key], state[key]) for key in state)

    prefixed_target = _model()
    prefixed_path = tmp_path / "module_prefixed.pth"
    torch.save({"state_dict": {f"module.{key}": value for key, value in state.items()}}, prefixed_path)
    prefixed_report = audit_and_load_checkpoint(
        prefixed_target,
        prefixed_path,
        allow_module_prefix_strip=True,
        strict=True,
    )
    assert prefixed_report.is_exact_match is True
    assert prefixed_report.stripped_module_prefix is True
    assert prefixed_report.sha256 == hashlib.sha256(prefixed_path.read_bytes()).hexdigest()
    assert all(torch.equal(prefixed_target.state_dict()[key], state[key]) for key in state)


def test_imagenette_wnids_keep_official_imagenet_indices(tmp_path: Path) -> None:
    valid = tmp_path / "valid"
    valid.mkdir()
    for wnid in IMAGENETTE_WNID_TO_IMAGENET1K_IDX:
        (valid / wnid).mkdir()
    mapping = verify_imagenette_wnid_mapping(valid)
    assert mapping["num_classes_found"] == 10
    assert mapping["wnid_to_imagenet1k_idx"] == {
        "n01440764": 0,
        "n02102040": 217,
        "n02979186": 482,
        "n03000684": 491,
        "n03028079": 497,
        "n03394916": 566,
        "n03417042": 569,
        "n03425413": 571,
        "n03445777": 574,
        "n03888257": 701,
    }
    assert sorted(mapping["wnid_to_imagenet1k_idx"].values()) == [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]

    renumbered = tmp_path / "renumbered"
    renumbered.mkdir()
    for index in range(10):
        (renumbered / str(index)).mkdir()
    with pytest.raises(CheckpointAuditError):
        verify_imagenette_wnid_mapping(renumbered)
