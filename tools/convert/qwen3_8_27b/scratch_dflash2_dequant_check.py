"""Compare the artifact's W8-quantized dflash weights against the BF16 originals.

The engine drafts at ~1.1 accepted tokens/step against a published ~4.8 while
every op passes its unit tests and the context append is verified correct. The
one structural difference from BOTH reference implementations (vLLM PR 52816 and
the llama.cpp study) is that we store the drafter's 21 big GEMMs as W8G32_F16S;
the references keep them BF16. A bad scale, group stride, or row mapping there
degrades every draft while leaving the target (separate NVFP4 weights) correct —
exactly the observed signature.

This reads the built artifact, dequantizes each dflash W8 object, and reports the
error against the corresponding tensor in the raw checkpoint. No GPU required.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import torch


def read_artifact_index(path: Path) -> tuple[dict, int]:
    with path.open("rb") as handle:
        handle.read(8)  # magic
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len).decode())
        payload_offset = 16 + header_len
    return header, payload_offset


def object_entries(header: dict) -> list[dict]:
    objects = header.get("objects") or header.get("tensors") or []
    if isinstance(objects, dict):
        return [{"name": k, **v} for k, v in objects.items()]
    return objects


def read_bytes(path: Path, offset: int, length: int) -> bytes:
    with path.open("rb") as handle:
        handle.seek(offset)
        return handle.read(length)


def dequantize_w8g32(blob: bytes, rows: int, cols: int) -> torch.Tensor:
    """W8G32_F16S: int8 codes with one fp16 scale per 32-element group, row-major."""
    groups = cols // 32
    code_bytes = rows * cols
    codes = torch.frombuffer(bytearray(blob[:code_bytes]), dtype=torch.int8)
    codes = codes.view(rows, cols).to(torch.float32)
    scale_bytes = rows * groups * 2
    scales = torch.frombuffer(
        bytearray(blob[code_bytes : code_bytes + scale_bytes]), dtype=torch.float16
    )
    scales = scales.view(rows, groups).to(torch.float32)
    return (codes.view(rows, groups, 32) * scales.unsqueeze(-1)).view(rows, cols)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", default="/mnt/f/models/src/out/qwen3_8_27b_nvfp4.ninfer")
    parser.add_argument("--checkpoint", default="/mnt/f/models/src/Qwen3.8-27B-DFlash2")
    parser.add_argument("--limit", type=int, default=6)
    args = parser.parse_args()

    from safetensors.torch import load_file

    artifact = Path(args.artifact)
    header, payload_offset = read_artifact_index(artifact)
    entries = {e["name"]: e for e in object_entries(header)}
    source = load_file(str(Path(args.checkpoint) / "model.safetensors"))

    # artifact object -> source tensor (and row slice within a merged parent)
    pairs: list[tuple[str, str, slice | None]] = []
    for layer in range(5):
        qkv = f"dflash/layers/{layer}/attention/query_key_value"
        pairs.append((qkv, f"layers.{layer}.self_attn.q_proj.weight", slice(0, 4096)))
        pairs.append((qkv, f"layers.{layer}.self_attn.k_proj.weight", slice(4096, 5120)))
        pairs.append((qkv, f"layers.{layer}.self_attn.v_proj.weight", slice(5120, 6144)))
        pairs.append(
            (f"dflash/layers/{layer}/attention/output", f"layers.{layer}.self_attn.o_proj.weight", None)
        )
        gate_up = f"dflash/layers/{layer}/mlp/gate_up"
        pairs.append((gate_up, f"layers.{layer}.mlp.gate_proj.weight", slice(0, 17408)))
        pairs.append((gate_up, f"layers.{layer}.mlp.up_proj.weight", slice(17408, 34816)))
        pairs.append((f"dflash/layers/{layer}/mlp/down", f"layers.{layer}.mlp.down_proj.weight", None))
    pairs.append(("dflash/feature_projection", "fc.weight", None))

    print(f"{'artifact object':52s} {'rows':>6s} {'rel_err':>10s} {'max_abs':>10s}")
    cache: dict[str, torch.Tensor] = {}
    shown = 0
    for obj_name, src_name, rows in pairs:
        entry = entries.get(obj_name)
        if entry is None or src_name not in source:
            print(f"  MISSING {obj_name} / {src_name}")
            continue
        if obj_name not in cache:
            shape = entry.get("shape") or entry.get("padded_shape")
            n, k = int(shape[0]), int(shape[1])
            blob = read_bytes(artifact, payload_offset + int(entry["offset"]), int(entry["length"]))
            cache[obj_name] = dequantize_w8g32(blob, n, k)
        got = cache[obj_name]
        want = source[src_name].to(torch.float32)
        if rows is not None:
            got = got[rows]
        if got.shape != want.shape:
            print(f"  SHAPE MISMATCH {obj_name}: artifact {tuple(got.shape)} vs src {tuple(want.shape)}")
            continue
        err = (got - want).norm() / want.norm().clamp_min(1e-9)
        print(f"{src_name:52s} {want.shape[0]:6d} {err.item():10.5f} {(got-want).abs().max().item():10.5f}")
        shown += 1
        if shown >= args.limit:
            break


if __name__ == "__main__":
    main()
