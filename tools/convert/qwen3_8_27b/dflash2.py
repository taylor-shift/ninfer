"""DFlash 2 drafter section for the Qwen3.8-27B artifact.

Maps the 81 BF16 source tensors of the
``z-lab/Qwen3.8-27B-DFlash2`` checkpoint (revision
``50307d4c4cde6860d4eee73e2547cd786fe8e8a4``) to the 66 ``dflash/*``
artifact objects bound by the 27B target
(``src/targets/qwen3_6_27b/impl/load/bindings.cpp``, ``kDFlashLayers = 5``):
the 21 big GEMM weights (feature projection + four per layer) are
quantized with the shared W8G32 group-quantization primitive, and the
remaining 45 conv, norm, and selector tensors pass through as BF16.

The object order, names, formats, and shapes in :data:`DFLASH2_TENSOR_SPECS`
mirror the ``bind_artifact`` dflash section of the bindings verbatim; the
q/k/v and gate/up merges follow the 35B DFlash precedent
(``tools/convert/qwen3_6_35b_a3b/recipe.py`` ``_build_dflash_recipes``) so
that the engine's ``context_key``/``context_value`` row views (rows 4096..
6144 / 5120..6144 of ``attention/query_key_value``) land on the k/v
projections.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Mapping

import torch

from tools.convert.common.quantize import pick_device
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import conversion as family_conversion
from tools.convert.qwen3_6.common import inventory as family_inventory
from tools.convert.qwen3_6.common import recipe as family_recipe


DFLASH2_REPOSITORY = "z-lab/Qwen3.8-27B-DFlash2"
DFLASH2_REVISION = "50307d4c4cde6860d4eee73e2547cd786fe8e8a4"
DFLASH2_TENSOR_FILENAME = "model.safetensors"

# Target geometry (spec 06 section 3; 27B bindings ground truth).
HIDDEN = 5120
HEAD_DIM = 128
QUERY_SIZE = 32 * HEAD_DIM  # 4096
KV_SIZE = 8 * HEAD_DIM  # 1024
QKV_SIZE = QUERY_SIZE + 2 * KV_SIZE  # 6144
INTERMEDIATE = 17408
GATE_UP_SIZE = 2 * INTERMEDIATE  # 34816
FEATURE_ROWS = 5 * HIDDEN  # 25600 (5 target feature layers x hidden)
CONV_USE = 2  # one base-kernel slice per conv attachment point
CONV_KERNEL_SIZE = 2
CONV_GROUP_SIZE = 16
CONV_GROUPS = HIDDEN // CONV_GROUP_SIZE  # 320
CONV_PROJECTION_ROWS = CONV_USE * CONV_KERNEL_SIZE * CONV_GROUPS  # 1280
SELECTOR_RANK = 256
SELECTOR_TOP_K = 16
VOCAB_SIZE = 248320
MASK_TOKEN_ID = 248070
BLOCK_SIZE = 8
TARGET_LAYER_IDS = (5, 19, 33, 47, 61)

DFLASH2_LAYERS = tuple(range(5))

BF16 = family_inventory.BF16
W8 = family_inventory.W8

TensorSpec = family_inventory.TensorSpec
TensorRecipe = family_recipe.TensorRecipe
SourcePreflight = family_recipe.SourcePreflight

# Payload byte totals (W8G32_F16S = 1.0625 B/param at group 32, BF16 = 2 B).
EXPECTED_W8_OBJECT_COUNT = 1 + len(DFLASH2_LAYERS) * 4  # 21
EXPECTED_BF16_OBJECT_COUNT = 45
EXPECTED_OBJECT_COUNT = 66  # 2 + 5*12 + 4, bindings.cpp object-by-object
EXPECTED_SOURCE_COUNT = 81  # 3 + 5*15 + 3, spec 06 section 3
EXPECTED_PAYLOAD_BYTES = 2_226_792_960


def _build_dflash2_specs() -> tuple[TensorSpec, ...]:
    """The 66 dflash object specs in exact bindings.cpp order."""

    specs: list[TensorSpec] = [
        family_inventory.tensor_spec(
            "dflash/feature_projection", (HIDDEN, FEATURE_ROWS), W8
        ),
        family_inventory.tensor_spec("dflash/context_norm", (HIDDEN,), BF16),
    ]
    for layer in DFLASH2_LAYERS:
        prefix = f"dflash/layers/{layer}/"
        specs.extend(
            (
                family_inventory.tensor_spec(prefix + "input_norm", (HIDDEN,), BF16),
                family_inventory.tensor_spec(
                    prefix + "attention/query_key_value", (QKV_SIZE, HIDDEN), W8
                ),
                family_inventory.tensor_spec(
                    prefix + "attention/query_norm", (HEAD_DIM,), BF16
                ),
                family_inventory.tensor_spec(
                    prefix + "attention/key_norm", (HEAD_DIM,), BF16
                ),
                family_inventory.tensor_spec(
                    prefix + "attention/output", (HIDDEN, QUERY_SIZE), W8
                ),
                family_inventory.tensor_spec(
                    prefix + "post_attention_norm", (HIDDEN,), BF16
                ),
                family_inventory.tensor_spec(
                    prefix + "mlp/gate_up", (GATE_UP_SIZE, HIDDEN), W8
                ),
                family_inventory.tensor_spec(
                    prefix + "mlp/down", (HIDDEN, INTERMEDIATE), W8
                ),
                family_inventory.tensor_spec(
                    prefix + "attention_conv/base_kernel",
                    (CONV_USE, CONV_KERNEL_SIZE, HIDDEN),
                    BF16,
                ),
                family_inventory.tensor_spec(
                    prefix + "attention_conv/kernel_projection",
                    (CONV_PROJECTION_ROWS, HIDDEN),
                    BF16,
                ),
                family_inventory.tensor_spec(
                    prefix + "mlp_conv/base_kernel",
                    (CONV_USE, CONV_KERNEL_SIZE, HIDDEN),
                    BF16,
                ),
                family_inventory.tensor_spec(
                    prefix + "mlp_conv/kernel_projection",
                    (CONV_PROJECTION_ROWS, HIDDEN),
                    BF16,
                ),
            )
        )
    specs.extend(
        (
            family_inventory.tensor_spec("dflash/final_norm", (HIDDEN,), BF16),
            family_inventory.tensor_spec(
                "dflash/selector/predecessor_codebook",
                (VOCAB_SIZE, SELECTOR_RANK),
                BF16,
            ),
            family_inventory.tensor_spec(
                "dflash/selector/successor_codebook",
                (VOCAB_SIZE, SELECTOR_RANK),
                BF16,
            ),
            family_inventory.tensor_spec(
                "dflash/selector/hidden_projection",
                (SELECTOR_RANK, HIDDEN),
                BF16,
            ),
        )
    )
    return tuple(specs)


def _build_dflash2_recipes() -> tuple[TensorRecipe, ...]:
    """The 81-to-66 source transform, mirroring the 35B dflash recipe.

    q/k/v projections merge into ``attention/query_key_value`` rows 0..4095
    (q), 4096..5119 (k), 5120..6143 (v); gate/up merge into ``mlp/gate_up``
    rows 0..17407 / 17408..34815.  Conv base kernels keep their
    ``(use, tap, hidden)`` rank-3 source shape; the codebooks are stored in
    the checkpoint without a ``.weight`` suffix (reference
    ``DFlash2DraftModel.key_mapping``).
    """

    recipes: list[TensorRecipe] = [
        TensorRecipe(
            "dflash/feature_projection",
            family_recipe.source("fc.weight", (HIDDEN, FEATURE_ROWS)),
        ),
        TensorRecipe(
            "dflash/context_norm",
            family_recipe.source("hidden_norm.weight", (HIDDEN,)),
        ),
    ]
    for layer in DFLASH2_LAYERS:
        source_prefix = f"layers.{layer}."
        object_prefix = f"dflash/layers/{layer}/"
        recipes.extend(
            (
                TensorRecipe(
                    object_prefix + "input_norm",
                    family_recipe.source(
                        source_prefix + "input_layernorm.weight", (HIDDEN,)
                    ),
                ),
                TensorRecipe(
                    object_prefix + "attention/query_key_value",
                    family_recipe.Concat(
                        (
                            family_recipe.source(
                                source_prefix + "self_attn.q_proj.weight",
                                (QUERY_SIZE, HIDDEN),
                            ),
                            family_recipe.source(
                                source_prefix + "self_attn.k_proj.weight",
                                (KV_SIZE, HIDDEN),
                            ),
                            family_recipe.source(
                                source_prefix + "self_attn.v_proj.weight",
                                (KV_SIZE, HIDDEN),
                            ),
                        ),
                        0,
                    ),
                ),
                TensorRecipe(
                    object_prefix + "attention/query_norm",
                    family_recipe.source(
                        source_prefix + "self_attn.q_norm.weight", (HEAD_DIM,)
                    ),
                ),
                TensorRecipe(
                    object_prefix + "attention/key_norm",
                    family_recipe.source(
                        source_prefix + "self_attn.k_norm.weight", (HEAD_DIM,)
                    ),
                ),
                TensorRecipe(
                    object_prefix + "attention/output",
                    family_recipe.source(
                        source_prefix + "self_attn.o_proj.weight",
                        (HIDDEN, QUERY_SIZE),
                    ),
                ),
                TensorRecipe(
                    object_prefix + "post_attention_norm",
                    family_recipe.source(
                        source_prefix + "post_attention_layernorm.weight",
                        (HIDDEN,),
                    ),
                ),
                TensorRecipe(
                    object_prefix + "mlp/gate_up",
                    family_recipe.Concat(
                        (
                            family_recipe.source(
                                source_prefix + "mlp.gate_proj.weight",
                                (INTERMEDIATE, HIDDEN),
                            ),
                            family_recipe.source(
                                source_prefix + "mlp.up_proj.weight",
                                (INTERMEDIATE, HIDDEN),
                            ),
                        ),
                        0,
                    ),
                ),
                TensorRecipe(
                    object_prefix + "mlp/down",
                    family_recipe.source(
                        source_prefix + "mlp.down_proj.weight",
                        (HIDDEN, INTERMEDIATE),
                    ),
                ),
                TensorRecipe(
                    object_prefix + "attention_conv/base_kernel",
                    family_recipe.source(
                        source_prefix + "attention_conv.base_kernel",
                        (CONV_USE, CONV_KERNEL_SIZE, HIDDEN),
                    ),
                ),
                TensorRecipe(
                    object_prefix + "attention_conv/kernel_projection",
                    family_recipe.source(
                        source_prefix + "attention_conv.kernel_projection.weight",
                        (CONV_PROJECTION_ROWS, HIDDEN),
                    ),
                ),
                TensorRecipe(
                    object_prefix + "mlp_conv/base_kernel",
                    family_recipe.source(
                        source_prefix + "mlp_conv.base_kernel",
                        (CONV_USE, CONV_KERNEL_SIZE, HIDDEN),
                    ),
                ),
                TensorRecipe(
                    object_prefix + "mlp_conv/kernel_projection",
                    family_recipe.source(
                        source_prefix + "mlp_conv.kernel_projection.weight",
                        (CONV_PROJECTION_ROWS, HIDDEN),
                    ),
                ),
            )
        )
    recipes.extend(
        (
            TensorRecipe(
                "dflash/final_norm",
                family_recipe.source("norm.weight", (HIDDEN,)),
            ),
            TensorRecipe(
                "dflash/selector/predecessor_codebook",
                family_recipe.source(
                    "candidate_selector.predecessor_codebook",
                    (VOCAB_SIZE, SELECTOR_RANK),
                ),
            ),
            TensorRecipe(
                "dflash/selector/successor_codebook",
                family_recipe.source(
                    "candidate_selector.successor_codebook",
                    (VOCAB_SIZE, SELECTOR_RANK),
                ),
            ),
            TensorRecipe(
                "dflash/selector/hidden_projection",
                family_recipe.source(
                    "candidate_selector.hidden_projection.weight",
                    (SELECTOR_RANK, HIDDEN),
                ),
            ),
        )
    )
    return tuple(recipes)


DFLASH2_TENSOR_SPECS = _build_dflash2_specs()
DFLASH2_RECIPE_SPECS = _build_dflash2_recipes()
DFLASH2_RECIPES_BY_NAME = {item.object_name: item for item in DFLASH2_RECIPE_SPECS}


def dflash2_source_requirements() -> dict[str, family_recipe.SourceTensor]:
    return family_recipe.source_requirements(DFLASH2_RECIPE_SPECS)


def validate_dflash2_recipes() -> None:
    """Prove the 66-object / 81-source DFlash 2 contract before any payload."""

    family_recipe.validate_recipe_coverage(
        DFLASH2_RECIPE_SPECS, DFLASH2_TENSOR_SPECS
    )
    if len(DFLASH2_TENSOR_SPECS) != EXPECTED_OBJECT_COUNT:
        raise ValueError(
            f"DFlash 2 section holds {len(DFLASH2_TENSOR_SPECS)} objects, "
            f"expected {EXPECTED_OBJECT_COUNT}"
        )
    if len(DFLASH2_RECIPE_SPECS) != EXPECTED_OBJECT_COUNT or len(
        DFLASH2_RECIPES_BY_NAME
    ) != EXPECTED_OBJECT_COUNT:
        raise ValueError("DFlash 2 recipe set does not pair one-to-one with objects")
    requirements = dflash2_source_requirements()
    if len(requirements) != EXPECTED_SOURCE_COUNT:
        raise ValueError(
            f"DFlash 2 recipe covers {len(requirements)} unique sources, "
            f"expected {EXPECTED_SOURCE_COUNT}"
        )
    if {item.dtype for item in requirements.values()} != {BF16}:
        raise ValueError("DFlash 2 source recipes must contain only BF16 tensors")
    for spec in DFLASH2_TENSOR_SPECS:
        if spec.format == W8:
            k = spec.shape[1]
            if k % 32:
                raise ValueError(
                    f"{spec.name}: W8G32 K dimension {k} is not a multiple of "
                    "group size 32"
                )
    w8_count = sum(spec.format == W8 for spec in DFLASH2_TENSOR_SPECS)
    if (w8_count, len(DFLASH2_TENSOR_SPECS) - w8_count) != (
        EXPECTED_W8_OBJECT_COUNT,
        EXPECTED_BF16_OBJECT_COUNT,
    ):
        raise ValueError(
            f"DFlash 2 format split drifted: {w8_count} W8G32_F16S, "
            f"{len(DFLASH2_TENSOR_SPECS) - w8_count} BF16"
        )
    payload_bytes = family_conversion.tensor_payload_bytes(DFLASH2_TENSOR_SPECS)
    if payload_bytes != EXPECTED_PAYLOAD_BYTES:
        raise ValueError(
            f"DFlash 2 payload byte total drifted: {payload_bytes}, "
            f"expected {EXPECTED_PAYLOAD_BYTES}"
        )


validate_dflash2_recipes()


def _draft_value(config: Mapping[str, object], name: str) -> object:
    """Look up a draft-model fact: ``dflash_config`` first, then top level.

    Mirrors the reference implementation's ``_draft_value``
    (``/code/dflash`` ``model.py``).
    """

    draft = config.get("dflash_config")
    if isinstance(draft, Mapping) and name in draft:
        return draft[name]
    return config.get(name)


def validate_dflash2_config(config: Mapping[str, object]) -> dict[str, object]:
    """Validate the DFlash 2 config facts that fix storage and execution.

    Spec 06 section 5 preflight contract: ``block_size == 8``,
    ``mask_token_id == 248070``,
    ``target_layer_ids == [5, 19, 33, 47, 61]``,
    ``selector_rank/top_k == 256/16``, ``conv_* == 2/16``, and
    ``input_embedding_scale``/``output_multiplier`` absent-or-1.0.
    """

    expected_facts = {
        "block_size": BLOCK_SIZE,
        "mask_token_id": MASK_TOKEN_ID,
        "target_layer_ids": list(TARGET_LAYER_IDS),
        "selector_rank": SELECTOR_RANK,
        "selector_top_k": SELECTOR_TOP_K,
        "conv_kernel_size": CONV_KERNEL_SIZE,
        "conv_group_size": CONV_GROUP_SIZE,
    }
    mismatches = [
        f"dflash config.{name}: expected {expected!r}, "
        f"got {_draft_value(config, name)!r}"
        for name, expected in expected_facts.items()
        if _draft_value(config, name) != expected
    ]
    if mismatches:
        raise ValueError(
            "DFlash 2 checkpoint config mismatch:\n  " + "\n  ".join(mismatches)
        )
    for name in ("input_embedding_scale", "output_multiplier"):
        value = _draft_value(config, name)
        if value is not None and value != 1.0:
            raise ValueError(
                f"dflash config.{name}: must be absent or 1.0, got {value!r}"
            )
    return {name: _draft_value(config, name) for name in expected_facts}


def _preflight_exact_source(
    reader: ShardReader,
    label: str,
) -> SourcePreflight:
    with reader:
        actual_names = set(reader.names)
    requirements = dflash2_source_requirements()
    required_names = set(requirements)
    if actual_names != required_names:
        missing = sorted(required_names - actual_names)
        extra = sorted(actual_names - required_names)
        details = []
        if missing:
            details.append(f"missing={missing[:8]!r}")
        if extra:
            details.append(f"extra={extra[:8]!r}")
        raise ValueError(
            f"{label} source inventory differs from its exact 81-tensor contract"
            + (": " + ", ".join(details) if details else "")
        )
    with reader:
        return family_recipe.preflight_source_reader(reader, DFLASH2_RECIPE_SPECS)


def preflight_dflash2_sources(model_dir: str | Path) -> SourcePreflight:
    """Assert the exact dflash tensor set, shapes, and BF16 dtypes."""

    model = Path(model_dir)
    return _preflight_exact_source(
        ShardReader.from_file(model / DFLASH2_TENSOR_FILENAME),
        "DFlash 2 checkpoint",
    )


def dflash2_source_sha256(model_dir: str | Path) -> str:
    """SHA-256 of the single-file dflash checkpoint (artifact metadata)."""

    path = Path(model_dir) / DFLASH2_TENSOR_FILENAME
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_dflash2_tensor(
    spec: TensorSpec,
    reader: ShardReader,
) -> torch.Tensor:
    """Materialize one dflash object from the source checkpoint."""

    tensor = family_recipe.materialize_recipe(DFLASH2_RECIPES_BY_NAME[spec.name], reader)
    if tuple(tensor.shape) != spec.shape:
        raise ValueError(
            f"{spec.name}: materialized shape {tuple(tensor.shape)} != {spec.shape}"
        )
    return tensor


def encode_dflash2_payload(
    tensor: torch.Tensor,
    spec: TensorSpec,
    device: str | torch.device,
) -> bytes:
    """Encode one materialized object: BF16 pass-through or W8G32 quantization.

    The named dtype guard runs before the shared primitives; the W8G32 range
    checks (rank, floating point, finite max-abs, finite positive binary16
    scales, code clamping) stay in :func:`tools.convert.common.quantize`.
    """

    if tensor.dtype != torch.bfloat16:
        raise TypeError(
            f"{spec.name}: source tensor must be BF16 "
            f"({'W8G32_F16S quantized' if spec.format == W8 else 'pass-through'}), "
            f"got {tensor.dtype}"
        )
    return family_conversion.encode_tensor_payload(tensor, spec, device)


def convert_dflash2(
    reader: ShardReader,
    emit: Callable[[TensorSpec, bytes], None],
    device: str | torch.device = "cuda",
) -> None:
    """Materialize and emit all 66 dflash objects, in inventory order.

    Mirrors the 35B converter's dflash emission loop
    (``tools/convert/qwen3_6_35b_a3b/convert.py``): one spec at a time,
    materialize -> encode -> emit, so the peak tensor footprint is one
    object.  ``emit(spec, payload)`` is the caller's artifact emitter, e.g.
    the ``write_payload(spec, payload)`` closure around ``ArtifactWriter``.
    """

    resolved_device = pick_device(device)
    for spec in DFLASH2_TENSOR_SPECS:
        tensor = materialize_dflash2_tensor(spec, reader)
        payload = encode_dflash2_payload(tensor, spec, resolved_device)
        del tensor
        emit(spec, payload)
        del payload


__all__ = [
    "BLOCK_SIZE",
    "BF16",
    "CONV_GROUP_SIZE",
    "CONV_KERNEL_SIZE",
    "DFLASH2_LAYERS",
    "DFLASH2_RECIPE_SPECS",
    "DFLASH2_RECIPES_BY_NAME",
    "DFLASH2_REPOSITORY",
    "DFLASH2_REVISION",
    "DFLASH2_TENSOR_FILENAME",
    "DFLASH2_TENSOR_SPECS",
    "EXPECTED_BF16_OBJECT_COUNT",
    "EXPECTED_OBJECT_COUNT",
    "EXPECTED_PAYLOAD_BYTES",
    "EXPECTED_SOURCE_COUNT",
    "EXPECTED_W8_OBJECT_COUNT",
    "MASK_TOKEN_ID",
    "SourcePreflight",
    "TARGET_LAYER_IDS",
    "TensorRecipe",
    "TensorSpec",
    "W8",
    "convert_dflash2",
    "dflash2_source_requirements",
    "dflash2_source_sha256",
    "encode_dflash2_payload",
    "materialize_dflash2_tensor",
    "preflight_dflash2_sources",
    "validate_dflash2_config",
    "validate_dflash2_recipes",
]