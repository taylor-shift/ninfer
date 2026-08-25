"""Reference DFlash 2 drafter forward, in PyTorch, for cross-checking the engine.

Ports `vllm/model_executor/models/qwen3_dflash2.py` (PR 52816) against the raw
`z-lab/Qwen3.8-27B-DFlash2` checkpoint so a single propose block can be compared
stage by stage with the ninfer engine's `[dflash.trace]` output.

The engine currently accepts ~1.1 drafts per step against a published ~4.8, and
every op passes its own unit test, so the question is whether the drafter's
forward produces the reference's tokens at all. This script answers that without
the engine: give it the committed context hidden states (the five target feature
layers) and the anchor token, and it prints the drafts the reference would
propose.

Not part of the build; a debugging instrument.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import torch
import torch.nn.functional as F


HIDDEN = 5120
HEAD_DIM = 128
Q_HEADS = 32
KV_HEADS = 8
LAYERS = 5
CONV_TAPS = 2
CONV_GROUP = 16
SELECTOR_RANK = 256
SELECTOR_TOP_K = 16
MASK_TOKEN_ID = 248070
ROPE_THETA = 1.0e7
RMS_EPS = 1.0e-6
SLIDING_WINDOW = 2048


def load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(str(path))


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = RMS_EPS) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (x * weight.float()).to(dtype)


def rope(x: torch.Tensor, positions: torch.Tensor, theta: float = ROPE_THETA) -> torch.Tensor:
    """Split-half NeoX rotation over the last dim (head_dim)."""
    half = x.shape[-1] // 2
    index = torch.arange(half, dtype=torch.float32, device=x.device)
    freq = theta ** (-2.0 * index / x.shape[-1])
    angle = positions.float().unsqueeze(-1) * freq
    cos, sin = angle.cos(), angle.sin()
    while cos.dim() < x.dim():
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    left, right = x[..., :half], x[..., half:]
    return torch.cat((left * cos - right * sin, right * cos + left * sin), dim=-1)


def grouped_conv(
    hidden: torch.Tensor, delta: torch.Tensor, base: torch.Tensor, block_size: int
) -> torch.Tensor:
    """Two-tap causal grouped convolution (vLLM `_grouped_conv`)."""
    tokens = hidden.shape[0]
    groups = HIDDEN // CONV_GROUP
    blocks = hidden.unflatten(-1, (groups, CONV_GROUP))
    coefficients = base.view(1, CONV_TAPS, groups, CONV_GROUP) + delta.unsqueeze(-1)
    out = coefficients[:, 0] * blocks
    position = torch.arange(tokens, device=hidden.device)
    position = position % block_size
    for tap in range(1, CONV_TAPS):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        out = out + coefficients[:, tap] * shifted * (position >= tap).view(-1, 1, 1)
    return out.flatten(-2)


class Conv:
    def __init__(self, weights: dict[str, torch.Tensor], prefix: str, block_size: int):
        self.base = weights[f"{prefix}.base_kernel"]
        self.projection = weights[f"{prefix}.kernel_projection.weight"]
        self.block_size = block_size

    def prepare(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        groups = HIDDEN // CONV_GROUP
        coefficients = F.linear(hidden, self.projection).reshape(
            hidden.shape[0], 2, CONV_TAPS, groups
        )
        return (
            grouped_conv(hidden, coefficients[:, 0], self.base[0], self.block_size),
            coefficients[:, 1],
        )

    def finish(self, hidden: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
        return grouped_conv(hidden, coefficients, self.base[1], self.block_size)


def drafter_forward(
    weights: dict[str, torch.Tensor],
    block_embeds: torch.Tensor,
    block_positions: torch.Tensor,
    context_states: torch.Tensor,
    context_positions: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Run the 5 drafter layers over the query block against precomputed context K/V.

    context_states are the target's combined feature hiddens AFTER fc+hidden_norm,
    matching `precompute_and_store_context_kv`.
    """
    hidden = block_embeds
    residual = None
    width = block_embeds.shape[0]

    for layer in range(LAYERS):
        p = f"layers.{layer}"
        if residual is None:
            residual = hidden
            normed = rms_norm(hidden, weights[f"{p}.input_layernorm.weight"])
        else:
            residual = residual + hidden
            normed = rms_norm(residual, weights[f"{p}.input_layernorm.weight"])

        attn_conv = Conv(weights, f"{p}.attention_conv", block_size)
        normed, coefficients = attn_conv.prepare(normed)

        q = F.linear(normed, weights[f"{p}.self_attn.q_proj.weight"])
        k = F.linear(normed, weights[f"{p}.self_attn.k_proj.weight"])
        v = F.linear(normed, weights[f"{p}.self_attn.v_proj.weight"])
        q = q.view(width, Q_HEADS, HEAD_DIM)
        k = k.view(width, KV_HEADS, HEAD_DIM)
        v = v.view(width, KV_HEADS, HEAD_DIM)
        q = rms_norm(q, weights[f"{p}.self_attn.q_norm.weight"])
        k = rms_norm(k, weights[f"{p}.self_attn.k_norm.weight"])
        q = rope(q.transpose(0, 1), block_positions).transpose(0, 1)
        k = rope(k.transpose(0, 1), block_positions).transpose(0, 1)

        # Context K/V for this layer, from the target feature states.
        ck = F.linear(context_states, weights[f"{p}.self_attn.k_proj.weight"])
        cv = F.linear(context_states, weights[f"{p}.self_attn.v_proj.weight"])
        ctx = context_states.shape[0]
        ck = ck.view(ctx, KV_HEADS, HEAD_DIM)
        cv = cv.view(ctx, KV_HEADS, HEAD_DIM)
        ck = rms_norm(ck, weights[f"{p}.self_attn.k_norm.weight"])
        ck = rope(ck.transpose(0, 1), context_positions).transpose(0, 1)

        all_k = torch.cat((ck, k), dim=0)
        all_v = torch.cat((cv, v), dim=0)
        all_pos = torch.cat((context_positions, block_positions), dim=0)

        group = Q_HEADS // KV_HEADS
        all_k = all_k.repeat_interleave(group, dim=1)
        all_v = all_v.repeat_interleave(group, dim=1)

        scores = torch.einsum("qhd,khd->hqk", q.float(), all_k.float())
        scores = scores / (HEAD_DIM**0.5)
        # Non-causal inside the block (is_causal=false), sliding window by distance.
        distance = (block_positions.view(-1, 1) - all_pos.view(1, -1)).abs()
        scores = scores.masked_fill((distance >= SLIDING_WINDOW).unsqueeze(0), float("-inf"))
        probs = scores.softmax(dim=-1)
        attn = torch.einsum("hqk,khd->qhd", probs, all_v.float()).to(hidden.dtype)
        attn = attn.reshape(width, Q_HEADS * HEAD_DIM)
        attn = attn_conv.finish(
            F.linear(attn, weights[f"{p}.self_attn.o_proj.weight"]), coefficients
        )
        hidden = attn

        residual = residual + hidden
        normed = rms_norm(residual, weights[f"{p}.post_attention_layernorm.weight"])
        mlp_conv = Conv(weights, f"{p}.mlp_conv", block_size)
        normed, coefficients = mlp_conv.prepare(normed)
        gate = F.linear(normed, weights[f"{p}.mlp.gate_proj.weight"])
        up = F.linear(normed, weights[f"{p}.mlp.up_proj.weight"])
        mlp_out = F.linear(F.silu(gate) * up, weights[f"{p}.mlp.down_proj.weight"])
        hidden = mlp_conv.finish(mlp_out, coefficients)

    return rms_norm(residual + hidden, weights["norm.weight"])


def select_path(
    weights: dict[str, torch.Tensor],
    proposal_hidden: torch.Tensor,
    lm_head: torch.Tensor,
    anchor_token: int,
) -> list[int]:
    """Top-k candidates, edge scores, and the greedy walk (vLLM CandidateSelector)."""
    steps = proposal_hidden.shape[0]
    logits = F.linear(proposal_hidden.float(), lm_head.float())
    unary, candidates = torch.topk(logits, SELECTOR_TOP_K, dim=-1)

    projected = F.linear(proposal_hidden, weights["candidate_selector.hidden_projection.weight"])
    pred_book = weights["candidate_selector.predecessor_codebook"]
    succ_book = weights["candidate_selector.successor_codebook"]

    drafts: list[int] = []
    previous_index = 0
    for step in range(steps):
        if step == 0:
            predecessor_ids = torch.full((SELECTOR_TOP_K,), anchor_token, dtype=torch.long)
        else:
            predecessor_ids = candidates[step - 1]
        pred = pred_book[predecessor_ids].float()
        succ = succ_book[candidates[step]].float()
        hidden = projected[step].float()
        edge = unary[step].float().unsqueeze(0) + torch.einsum(
            "pr,cr->pc", pred * hidden.unsqueeze(0), succ
        )
        row = edge[previous_index]
        previous_index = int(torch.argmax(row).item())
        drafts.append(int(candidates[step][previous_index].item()))
    return drafts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/mnt/f/models/src/Qwen3.8-27B-DFlash2")
    parser.add_argument("--features", required=True, help=".pt with target feature states")
    parser.add_argument("--anchor", type=int, required=True)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    weights = load_checkpoint(Path(args.checkpoint) / "model.safetensors")
    weights = {k: v.to(args.device) for k, v in weights.items()}

    payload = torch.load(args.features, map_location=args.device)
    features = payload["features"]  # [ctx, 25600] raw concatenated target hiddens
    context_positions = payload["context_positions"]
    frontier = int(payload["frontier"])
    lm_head = payload["lm_head"].to(args.device)

    projected = F.linear(features, weights["fc.weight"])
    context_states = rms_norm(projected, weights["hidden_norm.weight"])

    width = args.block_size
    embed = payload["embed_tokens"].to(args.device)
    ids = torch.full((width,), MASK_TOKEN_ID, dtype=torch.long, device=args.device)
    ids[0] = args.anchor
    block_embeds = embed[ids]
    block_positions = torch.arange(frontier, frontier + width, device=args.device)

    hidden = drafter_forward(
        weights,
        block_embeds,
        block_positions,
        context_states,
        context_positions,
        width,
    )
    drafts = select_path(weights, hidden[1:], lm_head, args.anchor)
    print("reference drafts:", drafts)


if __name__ == "__main__":
    main()
