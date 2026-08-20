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

# --- 0. Answer health probes immediately ------------------------------------

# The load balancer starts polling as soon as the container exists and reads a
# refused connection as unhealthy, so the port must answer before any slow work:
# waiting for the cached model, loading 20+ GiB of weights, and capturing CUDA
# Graphs all happen after this point and together take minutes.
port="${PORT:-80}"
health_port="${PORT_HEALTH:-$port}"
if [[ "$health_port" == "$port" ]]; then
    engine_port="${NINFER_INTERNAL_PORT:-18080}"
else
    engine_port="$port"
fi

# Readiness gate. Bind the health port now so the balancer sees 204 (initializing)
# instead of a refused connection, and keep answering until the engine reports 200.
# Perl is already present in the runtime image, so this needs no extra package.
readiness_gate() {
    perl -e '
        use IO::Socket::INET;
        my ($hport, $eport, $path) = @ARGV;
        my $srv = IO::Socket::INET->new(
            LocalAddr => "0.0.0.0", LocalPort => $hport, Listen => 128,
            Proto => "tcp", ReuseAddr => 1) or die "bind $hport: $!\n";
        # Stop as soon as the engine answers its own health route.
        my $ready = 0;
        while (!$ready) {
            my $c = $srv->accept or next;
            my $probe = IO::Socket::INET->new(PeerAddr => "127.0.0.1", PeerPort => $eport,
                                              Proto => "tcp", Timeout => 2);
            if ($probe) {
                print $probe "GET $path HTTP/1.0\r\nHost: localhost\r\n\r\n";
                my $resp = <$probe>; close $probe;
                $ready = 1 if defined $resp && $resp =~ m{ 200 };
            }
            # 204 keeps the worker in the pool while it loads; 200 hands over.
            my $code = $ready ? "200 OK" : "204 No Content";
            print $c "HTTP/1.1 $code\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
            close $c;
        }
        exit 0;
    ' "$1" "$2" "$3"
}

log "readiness gate on :${health_port} -> engine :${engine_port}"
readiness_gate "$health_port" "$engine_port" "${NINFER_HEALTH_PATH:-/health}" &
gate_pid=$!

# Stop the gate with the container, so a failure here cannot leave a port bound
# and a worker looking healthy while nothing serves it.
trap 'kill "$gate_pid" 2>/dev/null || true' EXIT


# --- 1. Locate the artifact -------------------------------------------------

# NINFER_ARTIFACT_PATH bypasses discovery entirely, for an artifact baked into the
# image or placed on a network volume at a known path.
artifact="${NINFER_ARTIFACT_PATH:-}"

if [[ -z "$artifact" ]]; then
    model="${NINFER_MODEL:?NINFER_MODEL must be set to the cached Hugging Face repo id}"
    [[ "$model" == */* ]] || fail "NINFER_MODEL must be 'org/name', got '$model'"

    # The cache directory replaces the '/' in a repo id with '--'.
    model_root="${HF_CACHE_ROOT}/models--${model//\//--}"

    # A worker can start before its cached model is on the host: the volume may not be
    # mounted yet, or the model may still be downloading. Exiting here would kill the
    # container, and the platform would immediately rent another worker that fails the
    # same way, so wait for the cache instead of failing fast.
    #
    # The wait is bounded by RUNPOD_INIT_TIMEOUT, which is the same budget the platform
    # allows for cold start, minus a margin for the engine's own load.
    cache_wait="${NINFER_CACHE_WAIT_S:-$(( ${RUNPOD_INIT_TIMEOUT:-800} / 2 ))}"
    if [[ ! -d "$model_root" ]]; then
        log "waiting up to ${cache_wait}s for cached model '$model' to appear at $model_root"
        waited=0
        while [[ ! -d "$model_root" ]] && (( waited < cache_wait )); do
            sleep 5
            waited=$(( waited + 5 ))
            (( waited % 60 == 0 )) && log "  still waiting for the cache (${waited}s)"
        done
    fi
    [[ -d "$model_root" ]] || fail \
        "cached model '$model' did not appear at $model_root within ${cache_wait}s. Confirm the endpoint's Model field is set to '$model', or set NINFER_ARTIFACT_PATH."

    # The directory can exist while the download is still in flight, so resolve the
    # snapshot and the artifact inside the same bounded wait.
    resolve_artifact() {
        local snap="" cand
        if [[ -f "${model_root}/refs/main" ]]; then
            cand="${model_root}/snapshots/$(<"${model_root}/refs/main")"
            [[ -d "$cand" ]] && snap="$cand"
        fi
        if [[ -z "$snap" ]]; then
            snap=$(find "${model_root}/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null |
                sort | tail -1)
        fi
        [[ -n "$snap" && -d "$snap" ]] || return 1
        snapshot="$snap"

        if [[ -n "${NINFER_ARTIFACT:-}" ]]; then
            [[ -e "${snap}/${NINFER_ARTIFACT}" ]] || return 1
            artifact="${snap}/${NINFER_ARTIFACT}"
            return 0
        fi
        # Repos publish exactly one .ninfer artifact; more than one is ambiguous.
        mapfile -t found < <(find "$snap" -maxdepth 1 \( -type f -o -type l \) \
            -name '*.ninfer' 2>/dev/null | sort)
        case "${#found[@]}" in
            0) return 1 ;;
            1) artifact="${found[0]}"; return 0 ;;
            *) fail "multiple .ninfer artifacts in $snap; set NINFER_ARTIFACT to choose one" ;;
        esac
    }

    if ! resolve_artifact; then
        log "cache directory present but no artifact yet; waiting up to ${cache_wait}s"
        waited=0
        while ! resolve_artifact && (( waited < cache_wait )); do
            sleep 5
            waited=$(( waited + 5 ))
            (( waited % 60 == 0 )) && log "  still waiting for the artifact (${waited}s)"
        done
    fi
    [[ -n "$artifact" ]] || fail \
        "no .ninfer artifact under $model_root after ${cache_wait}s; the download may be incomplete"
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

# Ports were resolved in section 0, before the gate was started. ninfer-serve
# defaults to 8080, so --port is always passed explicitly below.
args=(
    ninfer-serve "$artifact"
    --host 0.0.0.0
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
[[ "${NINFER_VISION:-0}" == "1" ]] && args+=(--vision)
[[ "${NINFER_CORS:-0}" == "1" ]] && args+=(--cors)

# A load balancing endpoint is reachable at a public URL, so the engine is given
# its own key rather than relying on the platform in front of it. Refuse to start
# without one: an unauthenticated engine here is open to anyone with the URL, and
# it bills GPU time while serving them.
#
# Present the key as x-api-key, not Authorization. Runpod's proxy uses the
# Authorization header for account-level auth, so a bearer token aimed at the
# engine can be consumed before it arrives; ninfer-serve accepts either form.
if [[ -n "${NINFER_API_KEY:-}" ]]; then
    args+=(--api-key "$NINFER_API_KEY")
elif [[ "${NINFER_ALLOW_ANONYMOUS:-0}" == "1" ]]; then
    log "WARNING: NINFER_ALLOW_ANONYMOUS=1 — serving without an API key"
else
    fail "NINFER_API_KEY is not set. Set it on the endpoint so requests must carry
       x-api-key, or set NINFER_ALLOW_ANONYMOUS=1 to deliberately serve without auth."
fi

# Word splitting is intended here: NINFER_EXTRA_ARGS carries additional flags.
# shellcheck disable=SC2206
[[ -n "${NINFER_EXTRA_ARGS:-}" ]] && args+=(${NINFER_EXTRA_ARGS})

args+=(--port "$engine_port")

if [[ "$health_port" == "$port" ]]; then
    # The gate from section 0 already holds this port and is answering 204. Start
    # the engine on its internal port; the gate releases the port as soon as the
    # engine's own /health returns 200.
    log "exec: ${args[*]}"
    "${args[@]}" &
    engine_pid=$!

    # If the engine dies during load, stop rather than proxying to nothing.
    wait "$gate_pid" 2>/dev/null || true
    if ! kill -0 "$engine_pid" 2>/dev/null; then
        wait "$engine_pid" 2>/dev/null
        fail "ninfer-serve exited during startup; see the engine output above"
    fi
    trap - EXIT
    log "engine is ready; forwarding :${port} to :${engine_port}"
    exec perl -e '
        use IO::Socket::INET; use POSIX ":sys_wait_h";
        my ($lp, $ep) = @ARGV;
        $SIG{CHLD} = sub { 1 while waitpid(-1, WNOHANG) > 0 };
        my $srv = IO::Socket::INET->new(LocalAddr => "0.0.0.0", LocalPort => $lp,
            Listen => 512, Proto => "tcp", ReuseAddr => 1) or die "bind $lp: $!\n";
        while (my $c = $srv->accept) {
            my $pid = fork; next if $pid;
            my $u = IO::Socket::INET->new(PeerAddr => "127.0.0.1", PeerPort => $ep,
                                          Proto => "tcp") or exit 1;
            # Relay both directions until either side closes.
            my $r = fork;
            if ($r) { while (sysread($c, my $b, 65536)) { syswrite($u, $b) } }
            else    { while (sysread($u, my $b, 65536)) { syswrite($c, $b) } exit 0 }
            close $c; close $u; exit 0;
        }
    ' "$port" "$engine_port"
fi

log "exec: ${args[*]}"
exec "${args[@]}"
