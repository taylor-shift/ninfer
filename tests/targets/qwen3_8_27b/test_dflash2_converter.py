"""DFlash 2 converter-pipeline integration tests (CPU, synthetic data only).

Covers the additive wiring of tools/convert/qwen3_8_27b/dflash2.py into the
27B NVFP4 conversion entry point (convert_nvfp4 / inventory_nvfp4): the
66-object ground-truth table (transcribed from the bind_artifact dflash
section of src/targets/qwen3_6_27b/impl/load/bindings.cpp, kDFlashLayers =
5), config validation, the exact 81-source preflight gates, the no-flag /
flagged NVFP4 object plans, small encode/decode round-trips, and the report
source pin.  The full-size (2.1 GiB synthetic) accept/materialize path is
exercised by tools/convert/qwen3_8_27b/scratch_dflash2_selftest.py, which is
deliberately not part of the unit suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from tools.artifact.layouts import decode_direct, dequantize_row_split
from tools.convert.qwen3_6.common import conversion as family_conversion
from tools.convert.qwen3_6.common import recipe as family_recipe
from tools.convert.qwen3_8_27b import convert_nvfp4 as convert
from tools.convert.qwen3_8_27b import dflash2
from tools.convert.qwen3_8_27b import inventory_nvfp4 as inventory


W8 = "W8G32_F16S"
BF16 = "BF16"
REVISION = "50307d4c4cde6860d4eee73e2547cd786fe8e8a4"
PAYLOAD_BYTES = 2_226_792_960

# --- Ground truth transcribed from src/targets/qwen3_6_27b/impl/load/
# --- bindings.cpp bind_artifact dflash section (kDFlashLayers = 5).
def _bindings_order() -> list[tuple[str, tuple[int, ...], str]]:
    order: list[tuple[str, tuple[int, ...], str]] = [
        ("dflash/feature_projection", (5120, 25600), W8),
        ("dflash/context_norm", (5120,), BF16),
    ]
    for layer in range(5):
        prefix = f"dflash/layers/{layer}/"
        order += [
            (prefix + "input_norm", (5120,), BF16),
            (prefix + "attention/query_key_value", (6144, 5120), W8),
            (prefix + "attention/query_norm", (128,), BF16),
            (prefix + "attention/key_norm", (128,), BF16),
            (prefix + "attention/output", (5120, 4096), W8),
            (prefix + "post_attention_norm", (5120,), BF16),
            (prefix + "mlp/gate_up", (34816, 5120), W8),
            (prefix + "mlp/down", (5120, 17408), W8),
            (prefix + "attention_conv/base_kernel", (2, 2, 5120), BF16),
            (prefix + "attention_conv/kernel_projection", (1280, 5120), BF16),
            (prefix + "mlp_conv/base_kernel", (2, 2, 5120), BF16),
            (prefix + "mlp_conv/kernel_projection", (1280, 5120), BF16),
        ]
    order += [
        ("dflash/final_norm", (5120,), BF16),
        ("dflash/selector/predecessor_codebook", (248320, 256), BF16),
        ("dflash/selector/successor_codebook", (248320, 256), BF16),
        ("dflash/selector/hidden_projection", (256, 5120), BF16),
    ]
    return order


# --- Ground truth transcribed from spec 06 section 3 (81 BF16 tensors).
def _source_order() -> list[tuple[str, tuple[int, ...]]]:
    sources: list[tuple[str, tuple[int, ...]]] = [
        ("fc.weight", (5120, 25600)),
        ("hidden_norm.weight", (5120,)),
        ("norm.weight", (5120,)),
    ]
    for layer in range(5):
        prefix = f"layers.{layer}."
        sources += [
            (prefix + "input_layernorm.weight", (5120,)),
            (prefix + "self_attn.q_proj.weight", (4096, 5120)),
            (prefix + "self_attn.k_proj.weight", (1024, 5120)),
            (prefix + "self_attn.v_proj.weight", (1024, 5120)),
            (prefix + "self_attn.o_proj.weight", (5120, 4096)),
            (prefix + "self_attn.q_norm.weight", (128,)),
            (prefix + "self_attn.k_norm.weight", (128,)),
            (prefix + "post_attention_layernorm.weight", (5120,)),
            (prefix + "mlp.gate_proj.weight", (17408, 5120)),
            (prefix + "mlp.up_proj.weight", (17408, 5120)),
            (prefix + "mlp.down_proj.weight", (5120, 17408)),
            (prefix + "attention_conv.base_kernel", (2, 2, 5120)),
            (prefix + "attention_conv.kernel_projection.weight", (1280, 5120)),
            (prefix + "mlp_conv.base_kernel", (2, 2, 5120)),
            (prefix + "mlp_conv.kernel_projection.weight", (1280, 5120)),
        ]
    sources += [
        ("candidate_selector.predecessor_codebook", (248320, 256)),
        ("candidate_selector.successor_codebook", (248320, 256)),
        ("candidate_selector.hidden_projection.weight", (256, 5120)),
    ]
    return sources


def _spec_by_name(name: str) -> inventory.TensorSpec:
    return next(spec for spec in inventory.DFLASH2_TENSOR_SPECS if spec.name == name)


def _resources() -> dict[str, bytes]:
    return {spec.name: b"\x00" * 16 for spec in inventory.RESOURCE_SPECS}


# (a) Object-by-object table: names, shapes, formats, order.
def test_dflash2_object_table_matches_bindings_ground_truth() -> None:
    expected = _bindings_order()
    got = [
        (spec.name, spec.shape, spec.format) for spec in inventory.DFLASH2_TENSOR_SPECS
    ]
    assert len(got) == 66, f"object count {len(got)} != 66"
    first = next(
        ((a, b) for a, b in zip(got, expected) if a != b), (len(got), len(expected))
    )
    assert got == expected, f"table differs from bindings.cpp, first at {first}"

    # The module constants (single home) pin the spec 06 section 5/6 contract.
    assert dflash2.EXPECTED_OBJECT_COUNT == 66
    assert dflash2.EXPECTED_SOURCE_COUNT == 81
    assert dflash2.EXPECTED_PAYLOAD_BYTES == PAYLOAD_BYTES
    assert dflash2.DFLASH2_REVISION == REVISION
    assert dflash2.DFLASH2_REPOSITORY == "z-lab/Qwen3.8-27B-DFlash2"

    # The NVFP4 converter's expected-payload cross-check constants agree
    # with the module and with the recomputed byte total.
    assert convert.DFLASH2_SOURCE_REVISION == REVISION
    assert convert.DFLASH2_SOURCE_REPOSITORY == "z-lab/Qwen3.8-27B-DFlash2"
    assert convert.EXPECTED_DFLASH2_OBJECT_COUNT == 66
    assert convert.EXPECTED_DFLASH2_SOURCE_COUNT == 81
    assert convert.EXPECTED_DFLASH2_PAYLOAD_BYTES == PAYLOAD_BYTES
    assert (
        family_conversion.tensor_payload_bytes(inventory.DFLASH2_TENSOR_SPECS)
        == PAYLOAD_BYTES
    )

    # The 66 specs have exactly one home: inventory re-exports the module
    # table and the 5-layer schedule.
    assert inventory.DFLASH2_TENSOR_SPECS is dflash2.DFLASH2_TENSOR_SPECS
    assert inventory.DFLASH2_LAYERS == tuple(range(5))
    assert dflash2.DFLASH2_LAYERS == tuple(range(5))


def test_dflash2_source_requirements_match_spec() -> None:
    requirements = dflash2.dflash2_source_requirements()
    got = sorted(
        (name, requirement.shape) for name, requirement in requirements.items()
    )
    assert got == sorted(_source_order()), "source set differs from spec section 3"
    assert len(requirements) == 81
    assert all(item.dtype == "BF16" for item in requirements.values())


# (b) Config validation: accept the pinned facts, reject all 9 drifts.
GOOD_CONFIG = {
    "architectures": ["DFlash2DraftModel"],
    "dflash_config": {
        "block_size": 8,
        "mask_token_id": 248070,
        "target_layer_ids": [5, 19, 33, 47, 61],
        "selector_rank": 256,
        "selector_top_k": 16,
        "conv_kernel_size": 2,
        "conv_group_size": 16,
    },
}


def test_validate_dflash2_config_accepts_pinned_facts() -> None:
    summary = dflash2.validate_dflash2_config(GOOD_CONFIG)
    assert summary == {
        "block_size": 8,
        "mask_token_id": 248070,
        "target_layer_ids": [5, 19, 33, 47, 61],
        "selector_rank": 256,
        "selector_top_k": 16,
        "conv_kernel_size": 2,
        "conv_group_size": 16,
    }
    # Top-level placement must also be accepted (reference _draft_value).
    dflash2.validate_dflash2_config(dict(GOOD_CONFIG["dflash_config"]))
    # input_embedding_scale / output_multiplier are absent-or-1.0.
    relaxed = {
        "dflash_config": {
            **GOOD_CONFIG["dflash_config"],
            "input_embedding_scale": 1.0,
            "output_multiplier": 1.0,
        }
    }
    dflash2.validate_dflash2_config(relaxed)


@pytest.mark.parametrize(
    ("field", "bad"),
    (
        ("block_size", 16),
        ("mask_token_id", 248077),
        ("target_layer_ids", [1, 6, 11, 16, 22]),
        ("selector_rank", 128),
        ("selector_top_k", 8),
        ("conv_kernel_size", 3),
        ("conv_group_size", 8),
        ("input_embedding_scale", 0.5),
        ("output_multiplier", 2.0),
    ),
)
def test_validate_dflash2_config_rejects_drift(field: str, bad: object) -> None:
    config = {"dflash_config": {**GOOD_CONFIG["dflash_config"], field: bad}}
    with pytest.raises(ValueError, match=field):
        dflash2.validate_dflash2_config(config)


# (c) Preflight gates against a tiny synthetic tmp-dir checkpoint.  The
# name-set gate runs before shape validation, so sub-1 KiB files suffice.
def _write_tiny_checkpoint(
    path: Path, drop: str | None = None, extra: str | None = None
) -> None:
    names = list(dflash2.dflash2_source_requirements())
    tensors: dict[str, torch.Tensor] = {}
    for name in names:
        if name == drop:
            continue
        tensors[name] = torch.zeros(1, dtype=torch.bfloat16)
    if extra is not None:
        tensors[extra] = torch.zeros(1, dtype=torch.bfloat16)
    save_file(tensors, str(path))


def test_preflight_dflash2_sources_rejects_missing_tensor(tmp_path) -> None:
    _write_tiny_checkpoint(
        tmp_path / dflash2.DFLASH2_TENSOR_FILENAME,
        drop="candidate_selector.hidden_projection.weight",
    )
    with pytest.raises(ValueError, match="hidden_projection.weight"):
        dflash2.preflight_dflash2_sources(tmp_path)
    # The error must name the missing side, not just the contract.
    with pytest.raises(ValueError, match="missing="):
        dflash2.preflight_dflash2_sources(tmp_path)


def test_preflight_dflash2_sources_rejects_extra_tensor(tmp_path) -> None:
    _write_tiny_checkpoint(
        tmp_path / dflash2.DFLASH2_TENSOR_FILENAME,
        extra="layers.0.stray.weight",
    )
    with pytest.raises(ValueError, match="stray"):
        dflash2.preflight_dflash2_sources(tmp_path)
    with pytest.raises(ValueError, match="extra="):
        dflash2.preflight_dflash2_sources(tmp_path)


def test_preflight_dflash2_sources_passes_name_gate_for_exact_set(tmp_path) -> None:
    # Exactly the 81 required names but synthetic shapes: the name-set gate
    # must pass and validation must proceed to the shape check.
    _write_tiny_checkpoint(tmp_path / dflash2.DFLASH2_TENSOR_FILENAME)
    with pytest.raises(ValueError, match=r"fc\.weight: source shape"):
        dflash2.preflight_dflash2_sources(tmp_path)
    try:
        dflash2.preflight_dflash2_sources(tmp_path)
    except ValueError as error:
        assert "source inventory differs" not in str(error)
    else:
        raise AssertionError("tiny-shape checkpoint must not pass preflight")


# (d) Inventory mechanism at PLAN level on the NVFP4 pipeline: the no-flag
# plan is the registered 1124-object inventory; the flagged plan appends
# exactly the 66 dflash objects (no full conversion run: the real source
# checkpoints are 20+ GiB and unavailable locally).
def test_no_flag_object_plan_has_no_dflash_objects() -> None:
    plan = convert.build_object_plan(_resources())
    names = [object.name for object in plan.objects]
    assert len(names) == 1124
    assert not any(name.startswith("dflash/") for name in names)
    # The additive dflash group does not change the registered totals.
    assert len(inventory.TENSOR_SPECS) == 1118
    assert len(inventory.OBJECT_SPECS) == 1124
    assert not any(
        spec.name.startswith("dflash/") for spec in inventory.OBJECT_SPECS
    )


def test_dflash_object_plan_appends_exactly_66_objects() -> None:
    base = convert.build_object_plan(_resources())
    extended = convert.build_object_plan(_resources(), include_dflash=True)
    assert len(base.objects) == 1124
    assert len(extended.objects) == 1124 + 66
    assert extended.objects[:1124] == base.objects
    added = extended.objects[1124:]
    specs = inventory.DFLASH2_TENSOR_SPECS
    assert [object.name for object in added] == [spec.name for spec in specs]
    for artifact_object, spec in zip(added, specs):
        assert artifact_object.shape == spec.shape
        assert artifact_object.format == spec.format
    assert sum(object.bytes for object in added) == PAYLOAD_BYTES


# (e) Small real encode/decode round-trips through the module's encoder.
def test_bf16_passthrough_round_trip_is_bit_exact() -> None:
    spec = _spec_by_name("dflash/layers/0/attention_conv/base_kernel")
    tensor = (torch.arange(64, dtype=torch.float32) / 7.0).reshape(
        2, 2, 16
    ).to(torch.bfloat16)
    payload = dflash2.encode_dflash2_payload(tensor, spec, "cpu")
    decoded = decode_direct(payload, BF16, tuple(tensor.shape))
    assert torch.equal(decoded, tensor)


def test_w8g32_object_round_trip_dequantizes_close_to_source() -> None:
    spec = _spec_by_name("dflash/feature_projection")
    generator = torch.Generator().manual_seed(20260824)
    tensor = (torch.randn((64, 128), generator=generator) / 4.0).to(
        torch.bfloat16
    )
    payload = dflash2.encode_dflash2_payload(tensor, spec, "cpu")
    rebuilt = dequantize_row_split(payload, W8, tuple(tensor.shape), "cpu")
    error = (rebuilt.float() - tensor.float()).abs().max().item()
    assert error < 0.02, f"W8 dequantization error too large: {error}"


def test_encode_dflash2_payload_requires_bf16_source() -> None:
    spec = _spec_by_name("dflash/context_norm")
    with pytest.raises(TypeError, match="must be BF16"):
        dflash2.encode_dflash2_payload(torch.randn(8), spec, "cpu")


class _TensorReader:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self.tensors = tensors

    def get(self, name: str) -> torch.Tensor:
        return self.tensors[name]


def test_qkv_and_gate_up_merges_follow_35b_row_order() -> None:
    prefix = "layers.0."
    reader = _TensorReader(
        {
            prefix + "self_attn.q_proj.weight": torch.full(
                (4096, 1), 1, dtype=torch.uint8
            ).expand(-1, 5120),
            prefix + "self_attn.k_proj.weight": torch.full(
                (1024, 1), 2, dtype=torch.uint8
            ).expand(-1, 5120),
            prefix + "self_attn.v_proj.weight": torch.full(
                (1024, 1), 3, dtype=torch.uint8
            ).expand(-1, 5120),
            prefix + "mlp.gate_proj.weight": torch.full(
                (17408, 1), 4, dtype=torch.uint8
            ).expand(-1, 5120),
            prefix + "mlp.up_proj.weight": torch.full(
                (17408, 1), 5, dtype=torch.uint8
            ).expand(-1, 5120),
        }
    )
    qkv = dflash2.materialize_dflash2_tensor(
        _spec_by_name("dflash/layers/0/attention/query_key_value"), reader
    )
    assert qkv.shape == (6144, 5120)
    assert torch.all(qkv[:4096] == 1)
    assert torch.all(qkv[4096:5120] == 2)
    assert torch.all(qkv[5120:] == 3)

    gate_up = dflash2.materialize_dflash2_tensor(
        _spec_by_name("dflash/layers/0/mlp/gate_up"), reader
    )
    assert gate_up.shape == (34816, 5120)
    assert torch.all(gate_up[:17408] == 4)
    assert torch.all(gate_up[17408:] == 5)


# (f) Report metadata: the dflash source pin appears only when flagged.
def test_report_records_dflash_source_pin_when_flagged(tmp_path) -> None:
    resources = _resources()
    base_source = family_recipe.SourcePreflight(1118, 1118, 1, {"BF16": 1118})
    dflash_source = dflash2.SourcePreflight(66, 81, 1, {"BF16": 81})
    plan = convert.build_object_plan(resources, include_dflash=True)
    common = dict(
        model_dir=tmp_path / "model",
        out_path=tmp_path / "out.ninfer",
        arguments={},
        config_summary={},
        source_preflight=base_source,
        objects=plan.objects,
        elapsed_seconds=1.0,
        final_bytes=123,
        device=torch.device("cpu"),
        ranking_path=tmp_path / "ranking.json",
    )

    report_no = convert.build_conversion_report(**common)
    assert "dflash" not in report_no

    report = convert.build_conversion_report(
        **common,
        dflash_model_dir=tmp_path / "dflash",
        dflash_config_summary={"block_size": 8},
        dflash_source=dflash_source,
        dflash_sha256="ab" * 32,
    )
    assert report["dflash"] == {
        "repository": "z-lab/Qwen3.8-27B-DFlash2",
        "revision": REVISION,
        "model_path": str((tmp_path / "dflash").resolve()),
        "sha256": "ab" * 32,
        "objects": 66,
        "payload_bytes": PAYLOAD_BYTES,
        "config_summary": {"block_size": 8},
        "source_preflight": {
            "recipes": 66,
            "tensors": 81,
            "shards": 1,
            "dtypes": {"BF16": 81},
        },
    }
    # The flagged run opens the writer with the extended 1190-object plan.
    assert len(plan.objects) == 1190