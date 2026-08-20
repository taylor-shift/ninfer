#!/usr/bin/env bash
#
# RunPod Serverless (load balancing) entrypoint for NInfer.
#
# ninfer-serve is already a conforming load balancing worker: it serves the
# OpenAI and Anthropic HTTP surfaces directly and exposes an unauthenticated
# /health route for the load balancer to poll. No handler process sits in the
# request path; this script only prepares arguments and then execs the engine,
# so the container's PID 1 is ninfer-serve itself and signals reach it directly.
#
# Responsibilities:
#   1. resolve the cached .ninfer artifact inside the Hugging Face cache layout;
#   2. size --max-context for whichever sm_120a GPU the worker landed on;
#   3. exec ninfer-serve.

set -euo pipefail

log() { printf '[ninfer-entrypoint] %s\n' "$*" >&2; }
fail() { log "ERROR: $*"; exit 1; }

HF_CACHE_ROOT="${HF_CACHE_ROOT:-/runpod-volume/huggingface-cache/hub}"

# --- 1. Locate the artifact -------------------------------------------------

# NINFER_ARTIFACT_PATH bypasses discovery entirely, for an artifact baked into the
# image or placed on a network volume at a known path.
artifact="${NINFER_ARTIFACT_PATH:-}"

if [[ -z "$artifact" ]]; then
    model="${NINFER_MODEL:?NINFER_MODEL must be set to the cached Hugging Face repo id}"
    [[ "$model" == */* ]] || fail "NINFER_MODEL must be 'org/name', got '$model'"

    # The cache directory replaces the '/' in a repo id with '--'.
    model_root="${HF_CACHE_ROOT}/models--${model//\//--}"
    [[ -d "$model_root" ]] || fail \
        "cached model '$model' not found at $model_root. Set it as the endpoint's cached model, or set NINFER_ARTIFACT_PATH."

    # refs/main names the materialized snapshot; fall back to the newest snapshot
    # directory when that file is absent.
    snapshot=""
    if [[ -f "${model_root}/refs/main" ]]; then
        candidate="${model_root}/snapshots/$(<"${model_root}/refs/main")"
        [[ -d "$candidate" ]] && snapshot="$candidate"
    fi
    if [[ -z "$snapshot" ]]; then
        snapshot=$(find "${model_root}/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null |
            sort | tail -1)
    fi
    [[ -n "$snapshot" && -d "$snapshot" ]] || fail "no snapshot directory under $model_root"

    if [[ -n "${NINFER_ARTIFACT:-}" ]]; then
        artifact="${snapshot}/${NINFER_ARTIFACT}"
    else
        # Repos publish exactly one .ninfer artifact; more than one is ambiguous.
        mapfile -t found < <(find "$snapshot" -maxdepth 1 -type f -name '*.ninfer' -o \
            -maxdepth 1 -type l -name '*.ninfer' | sort)
        case "${#found[@]}" in
            0) fail "no .ninfer artifact in $snapshot (still downloading, or not an NInfer repo?)" ;;
            1) artifact="${found[0]}" ;;
            *) fail "multiple .ninfer artifacts in $snapshot; set NINFER_ARTIFACT to choose one" ;;
        esac
    fi
fi

# The cache stores blobs out of tree and links them into the snapshot.
[[ -e "$artifact" ]] || fail "artifact not found: $artifact"
artifact=$(readlink -f "$artifact")
[[ -f "$artifact" ]] || fail "artifact is not a regular file: $artifact"

artifact_bytes=$(stat -Lc %s "$artifact")
log "artifact $artifact ($((artifact_bytes / 1024 / 1024 / 1024)) GiB)"

# --- 2. Select a profile and size the context --------------------------------

kv_dtype="${NINFER_KV_DTYPE:-int8}"

gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
[[ -n "$gpu_name" ]] && log "device: ${gpu_name}"

# The engine is built for sm_120a and refuses anything else, so fail here with a
# clear message rather than deep inside engine startup.
if [[ -n "$gpu_name" ]]; then
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
    if [[ -n "$cc" && "$cc" != "12.0" ]]; then
        fail "NInfer is built for sm_120a (compute capability 12.0); this worker has ${gpu_name} at compute capability ${cc}. Restrict the endpoint's GPU types to RTX 5090 or RTX PRO 6000 Blackwell."
    fi
fi

total_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null |
    head -1 | tr -d ' ' || true)
[[ "$total_mib" =~ ^[0-9]+$ ]] || total_mib=0

# NINFER_PROFILE selects a deployment shape. Two are defined:
#
#   standard  one 32 GiB RTX 5090, C=4. Matches the measured local configuration
#             and is the cheaper tier; intended for burst offload alongside a
#             local engine.
#   highconc  one 96 GiB RTX PRO 6000, C=8. The larger KV pool lets every
#             concurrent request hold a full-length private context, which does
#             not fit on a 32 GiB part.
#
# "auto" picks by VRAM. Either field can still be overridden explicitly by
# setting NINFER_MAX_CONCURRENCY or NINFER_MAX_CONTEXT.
profile="${NINFER_PROFILE:-auto}"
if [[ "$profile" == "auto" ]]; then
    if (( total_mib > 65536 )); then profile=highconc; else profile=standard; fi
fi

case "$profile" in
    standard) profile_concurrency=4 ;;
    highconc) profile_concurrency=8 ;;
    *) fail "unknown NINFER_PROFILE '$profile' (expected 'standard', 'highconc', or 'auto')" ;;
esac
max_concurrency="${NINFER_MAX_CONCURRENCY:-$profile_concurrency}"
log "profile ${profile}: --max-concurrency ${max_concurrency}"

max_context="${NINFER_MAX_CONTEXT:-}"
if [[ -z "$max_context" ]]; then
    if (( total_mib > 0 )); then
        # Shared KV cost per token for the 27B hybrid text stack: 16 full-attention
        # layers hold paged K and V over 4 KV heads of head_dim 256. INT8 group-64
        # adds one fp16 scale per 64 elements; BF16 stores two bytes per element.
        # GDN layers keep fixed-size state and do not grow per token.
        if [[ "$kv_dtype" == "bf16" ]]; then kv_kib_per_token=64; else kv_kib_per_token=33; fi

        # Headroom above weights and the KV pool: workspace arenas, allocator slack,
        # and the per-batch CUDA Graph families, which are instantiated once for
        # every batch size in [1,C] and so grow with concurrency. Fitted to the two
        # measured RTX 5090 configurations (C=2 at 255k, C=4 at 232k), which this
        # reproduces to within 0.03%.
        reserve_mib="${NINFER_RESERVE_MIB:-$(( 3151 + 371 * max_concurrency ))}"
        artifact_mib=$(( artifact_bytes / 1024 / 1024 ))
        budget_mib=$(( total_mib - artifact_mib - reserve_mib ))

        if (( budget_mib <= 0 )); then
            fail "device has ${total_mib} MiB but the artifact needs ${artifact_mib} MiB plus ${reserve_mib} MiB reserve; use a larger GPU or a smaller profile"
        fi

        # tokens = budget / per-token cost, floored to the 64-token KV page size and
        # capped at the 27B native context.
        max_context=$(( budget_mib * 1024 / kv_kib_per_token ))
        (( max_context > 262144 )) && max_context=262144
        max_context=$(( max_context - max_context % 64 ))

        log "autotune: ${total_mib} MiB device, ${artifact_mib} MiB weights, ${reserve_mib} MiB reserve, ${kv_dtype} KV -> --max-context ${max_context}"
    else
        max_context=32768
        log "GPU not detected; defaulting --max-context ${max_context}"
    fi
fi

# --- 3. Run the engine ------------------------------------------------------

# The load balancer routes traffic to PORT and polls HEALTH_CHECK_PATH on
# PORT_HEALTH. ninfer-serve answers /health on its single listener, so set
# HEALTH_CHECK_PATH=/health on the endpoint and leave PORT_HEALTH equal to PORT.
port="${PORT:-80}"

args=(
    ninfer-serve "$artifact"
    --host 0.0.0.0
    --port "$port"
    --max-context "$max_context"
    --kv-capacity "${NINFER_KV_CAPACITY:-auto}"
    --kv-dtype "$kv_dtype"
    --max-concurrency "$max_concurrency"
    --pending-timeout-ms "${NINFER_PENDING_TIMEOUT_MS:-120000}"
)

spec="${NINFER_SPEC:-mtp}"
if [[ "$spec" == "mtp" || "$spec" == "dflash" ]]; then
    args+=(--spec "$spec" --draft-tokens "${NINFER_DRAFT_TOKENS:-3}")
    [[ "${NINFER_LM_HEAD_DRAFT:-1}" == "1" ]] && args+=(--lm-head-draft)
fi

[[ "${NINFER_PRESERVE_THINKING:-1}" == "1" ]] && args+=(--preserve-thinking)
[[ -n "${NINFER_API_KEY:-}" ]] && args+=(--api-key "$NINFER_API_KEY")
[[ "${NINFER_VISION:-0}" == "1" ]] && args+=(--vision)
[[ "${NINFER_CORS:-0}" == "1" ]] && args+=(--cors)

# Word splitting is intended here: NINFER_EXTRA_ARGS carries additional flags.
# shellcheck disable=SC2206
[[ -n "${NINFER_EXTRA_ARGS:-}" ]] && args+=(${NINFER_EXTRA_ARGS})

log "exec: ${args[*]}"
exec "${args[@]}"
