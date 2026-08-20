#!/usr/bin/env bash
#
# Multi-GPU pod entrypoint for NInfer.
#
# Runs one ninfer-serve per visible GPU and fronts them with a least-connections
# balancer on $PORT, so the pod exposes a single OpenAI/Anthropic endpoint.
#
# ninfer has no tensor parallelism (no NCCL, no all-reduce), so N cards are used
# by replication: each engine holds a full copy of the artifact and its own KV
# pool. Per-stream speed equals one card; aggregate throughput and concurrency
# scale with N.
#
# Relationship to deploy/runpod/entrypoint.sh (serverless, single GPU):
# artifact discovery, the sm_120a guard, and the context autosizer are the same
# logic; this variant drops the serverless readiness gate (a pod is not polled by
# a load balancer) and adds per-device fan-out plus the balancer.
#
# Env (in addition to the single-GPU entrypoint's):
#   GPUS                 engines to start (default: all visible GPUs)
#   ENGINE_BASE_PORT     first internal engine port (default 18080)
#   NINFER_SINGLE_GPU=1  force one engine on device 0 (bypasses the balancer)
#   READY_TIMEOUT_S      per-engine readiness budget (default 1200)

set -euo pipefail

log() { printf '[ninfer-multi] %s\n' "$*" >&2; }
fail() { log "ERROR: $*"; exit 1; }

HF_CACHE_ROOT="${HF_CACHE_ROOT:-/runpod-volume/huggingface-cache/hub}"
PORT="${PORT:-8000}"
ENGINE_BASE_PORT="${ENGINE_BASE_PORT:-18080}"
# Soft budget only (never fatal). Seven engines loading 20 GiB each from one
# volume genuinely takes a while on a cold page cache; 1200s was too short and
# turned a slow-but-healthy start into a restart loop.
READY_TIMEOUT_S="${READY_TIMEOUT_S:-2400}"

# --- 0. sshd ----------------------------------------------------------------
#
# The base image has no ssh daemon, so a pod that wedges during a multi-hour
# download cannot be inspected at all — the only signal is a log stream that
# drops lines. RunPod already maps 22/tcp and injects PUBLIC_KEY, so starting
# sshd here costs nothing and makes that mapping real.

if [[ "${NINFER_ENABLE_SSHD:-1}" == "1" ]] && command -v /usr/sbin/sshd >/dev/null 2>&1; then
    if [[ -n "${PUBLIC_KEY:-}" ]]; then
        mkdir -p /root/.ssh && chmod 700 /root/.ssh
        # PUBLIC_KEY may hold several keys; RunPod passes them newline separated.
        printf '%s\n' "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
        chmod 600 /root/.ssh/authorized_keys
        /usr/sbin/sshd -e 2>/dev/null && log "sshd started on :22"
    else
        log "sshd available but PUBLIC_KEY is unset; skipping"
    fi
fi

# --- 1. Locate the artifact -------------------------------------------------
# Same resolution order as the serverless entrypoint: explicit path wins, then
# the HF cache layout for NINFER_MODEL. On a pod the model is usually already on
# local disk or a mounted volume, so the wait budget is short by default.

artifact="${NINFER_ARTIFACT_PATH:-}"

if [[ -z "$artifact" ]]; then
    model="${NINFER_MODEL:?NINFER_MODEL or NINFER_ARTIFACT_PATH must be set}"
    [[ "$model" == */* ]] || fail "NINFER_MODEL must be 'org/name', got '$model'"

    # Search the volume cache, the image-local HF cache, and common pod paths.
    search_roots=(
        "${HF_CACHE_ROOT}"
        "${HF_HOME:-/root/.cache/huggingface}/hub"
        "/workspace/huggingface-cache/hub"
        "/workspace"
        "/models"
    )
    model_dir="models--${model//\//--}"

    resolve_artifact() {
        local root snap cand
        for root in "${search_roots[@]}"; do
            [[ -d "$root" ]] || continue
            # HF cache layout
            if [[ -d "${root}/${model_dir}" ]]; then
                local mr="${root}/${model_dir}"
                snap=""
                if [[ -f "${mr}/refs/main" ]]; then
                    cand="${mr}/snapshots/$(<"${mr}/refs/main")"
                    [[ -d "$cand" ]] && snap="$cand"
                fi
                [[ -z "$snap" ]] && snap=$(find "${mr}/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -1)
                if [[ -n "$snap" && -d "$snap" ]]; then
                    cand=$(find "$snap" -maxdepth 1 \( -type f -o -type l \) -name '*.ninfer' 2>/dev/null | sort | head -1)
                    [[ -n "$cand" ]] && { artifact="$cand"; return 0; }
                fi
            fi
            # Loose artifact directly under the root
            cand=$(find "$root" -maxdepth 2 \( -type f -o -type l \) -name '*.ninfer' 2>/dev/null | sort | head -1)
            [[ -n "$cand" ]] && { artifact="$cand"; return 0; }
        done
        return 1
    }

    if ! resolve_artifact; then
        wait_s="${NINFER_CACHE_WAIT_S:-60}"
        log "no .ninfer artifact on disk; waiting up to ${wait_s}s for a mount/cache to appear"
        waited=0
        while ! resolve_artifact && (( waited < wait_s )); do
            sleep 5; waited=$(( waited + 5 ))
            (( waited % 30 == 0 )) && log "  still waiting (${waited}s)"
        done
    fi

    # Still nothing: download it. The base image did this with one curl stream
    # (~1 MB/s measured from RunPod = 40+ hours for 21.5 GB); the fetch helper
    # uses hf_transfer or aria2c over many connections instead.
    if [[ -z "$artifact" ]]; then
        target_dir="${NINFER_VOLUME_PATH:-/models}"
        target="${target_dir}/${NINFER_ARTIFACT:-$(basename "$model").ninfer}"
        # NINFER_NO_DOWNLOAD=1 makes a missing artifact a hard stop instead of a
        # download. Retrying a fetch in a supervised container fights any manual
        # repair: the supervisor respawns the downloader, which overwrites the
        # file being inspected or replaced. With this set the container stays up
        # (sshd alive) and waits, so the artifact can be staged deliberately.
        if [[ "${NINFER_NO_DOWNLOAD:-0}" == "1" ]]; then
            log "artifact missing at ${target} and NINFER_NO_DOWNLOAD=1."
            log "Stage the file there, then restart the pod. Container stays up for ssh."
            while [[ ! -f "$target" ]]; do sleep 30; done
            log "artifact appeared at ${target}; continuing"
            artifact="$target"
        else
        log "artifact absent; downloading ${model}/${NINFER_ARTIFACT} -> ${target}"
        free_mib=$(df -Pm "$target_dir" 2>/dev/null | awk 'NR==2 {print $4}')
        [[ "$free_mib" =~ ^[0-9]+$ ]] && log "free space at ${target_dir}: $(( free_mib / 1024 )) GiB"
        ninfer-fetch-artifact "$model" "${NINFER_ARTIFACT:?NINFER_ARTIFACT must name the file}" "$target" \
            || fail "artifact download failed"
        artifact="$target"
        fi
    fi

    [[ -n "$artifact" ]] || fail "no .ninfer artifact found for '$model'. Set NINFER_ARTIFACT_PATH."
fi

[[ -e "$artifact" ]] || fail "artifact not found: $artifact"
artifact=$(readlink -f "$artifact")
[[ -f "$artifact" ]] || fail "artifact is not a regular file: $artifact"
artifact_bytes=$(stat -Lc %s "$artifact")
artifact_mib=$(( artifact_bytes / 1024 / 1024 ))
log "artifact ${artifact} ($(( artifact_mib / 1024 )) GiB)"

# --- 2. GPU inventory and guard ---------------------------------------------

detected=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 0)
(( detected > 0 )) || fail "no GPUs visible to the container"
GPUS="${GPUS:-$detected}"
[[ "${NINFER_SINGLE_GPU:-0}" == "1" ]] && GPUS=1
(( GPUS <= detected )) || fail "GPUS=${GPUS} but only ${detected} GPUs are visible"

log "GPUs visible: ${detected}; starting ${GPUS} engine(s)"
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader 2>/dev/null |
    while IFS= read -r line; do log "  gpu: ${line}"; done

# The engine is built for sm_120a only. Fail here with a clear message rather
# than deep inside engine startup.
mapfile -t caps < <(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d ' ')
for i in $(seq 0 $(( GPUS - 1 ))); do
    cc="${caps[$i]:-}"
    [[ -z "$cc" || "$cc" == "12.0" ]] || fail \
        "GPU ${i} has compute capability ${cc}; NInfer is built for sm_120a (12.0). Use RTX 5090 or RTX PRO 6000 Blackwell."
done

# CUDA 13.1 userland needs a native 13.1+ driver: forward compatibility is not
# available on GeForce parts, so a mismatch here fails at first CUDA call with a
# far less obvious error than this message.
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
log "driver: ${driver:-unknown}"
if [[ -n "$driver" ]]; then
    driver_major="${driver%%.*}"
    if [[ "$driver_major" =~ ^[0-9]+$ ]] && (( driver_major < 580 )); then
        log "WARNING: driver ${driver} may predate the CUDA 13.1 runtime in this image."
        log "         If the engine fails at CUDA init, the pod needs a CUDA 13.1+ host."
    fi
fi

# --- 3a. Authentication (resolved before anything starts) --------------------
#
# A pod port is reachable at a public URL, so the engine carries its own key
# rather than trusting anything in front of it. An unauthenticated engine here
# is open to anyone who finds the URL and bills GPU time while serving them.
#
# This check is deliberately at top level: build_args() runs inside a command
# substitution, so a failure there cannot stop the parent shell.
#
# NINFER_API_KEY_FILE is honoured first so a key can arrive as a mounted secret
# instead of an environment variable visible in the pod's config.

if [[ -z "${NINFER_API_KEY:-}" && -n "${NINFER_API_KEY_FILE:-}" ]]; then
    [[ -r "$NINFER_API_KEY_FILE" ]] || fail "NINFER_API_KEY_FILE is not readable: ${NINFER_API_KEY_FILE}"
    NINFER_API_KEY="$(tr -d '\r\n' < "$NINFER_API_KEY_FILE")"
    export NINFER_API_KEY
fi

if [[ -n "${NINFER_API_KEY:-}" ]]; then
    # Never log the key itself; a length and fingerprint are enough to confirm
    # which key the engines came up with.
    key_len=${#NINFER_API_KEY}
    (( key_len >= 16 )) || log "WARNING: NINFER_API_KEY is only ${key_len} chars; use a long random key"
    if command -v sha256sum >/dev/null 2>&1; then
        key_fp=$(printf '%s' "$NINFER_API_KEY" | sha256sum | cut -c1-8)
        log "auth: enabled (key ${key_len} chars, sha256:${key_fp})"
    else
        log "auth: enabled (key ${key_len} chars)"
    fi
    log "auth: send 'Authorization: Bearer <key>' or 'x-api-key: <key>'; /health stays open"
elif [[ "${NINFER_ALLOW_ANONYMOUS:-0}" == "1" ]]; then
    log "WARNING: NINFER_ALLOW_ANONYMOUS=1 — every engine serves WITHOUT authentication."
    log "WARNING: anyone who reaches this pod's URL can spend its GPU time."
else
    fail "NINFER_API_KEY is not set.
       Set NINFER_API_KEY (or NINFER_API_KEY_FILE) so requests must carry a key,
       or set NINFER_ALLOW_ANONYMOUS=1 to deliberately serve without auth."
fi

# --- 3. Per-engine settings --------------------------------------------------

# Engine role: indices [0, NINFER_TEXT_ENGINES) are text-only; the rest run
# --vision. NINFER_TEXT_ENGINES defaults to GPUS-2 (two vision cards), and
# NINFER_VISION=0 forces an all-text fleet.
engine_is_vision() {
    local idx="$1"
    [[ "${NINFER_VISION:-1}" == "0" ]] && return 1
    local ntext="${NINFER_TEXT_ENGINES:-}"
    if [[ -z "$ntext" ]]; then
        ntext=$(( GPUS > 2 ? GPUS - 2 : GPUS ))
    fi
    (( idx >= ntext ))
}

kv_dtype="${NINFER_KV_DTYPE:-int8}"
spec="${NINFER_SPEC:-mtp}"
draft_tokens="${NINFER_DRAFT_TOKENS:-3}"

# Profile picks per-engine concurrency; every field stays overridable.
profile="${NINFER_PROFILE:-auto}"
first_total=$(nvidia-smi --id=0 --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
[[ "$first_total" =~ ^[0-9]+$ ]] || first_total=0
if [[ "$profile" == "auto" ]]; then
    if (( first_total > 65536 )); then profile=highconc; else profile=standard; fi
fi
case "$profile" in
    standard) profile_concurrency=4 ;;
    highconc) profile_concurrency=8 ;;
    *) fail "unknown NINFER_PROFILE '$profile' (expected standard|highconc|auto)" ;;
esac
max_concurrency="${NINFER_MAX_CONCURRENCY:-$profile_concurrency}"

autosize_context() {  # device index -> token budget for that card
    local idx="$1" total_mib kv_kib reserve_mib budget ctx
    total_mib=$(nvidia-smi --id="$idx" --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    [[ "$total_mib" =~ ^[0-9]+$ ]] || { echo 32768; return; }
    # 16 full-attention layers, 4 KV heads, head_dim 256. int8 group-64 adds one
    # fp16 scale per 64 elements; bf16 stores two bytes per element. GDN layers
    # hold fixed-size state and do not grow per token.
    if [[ "$kv_dtype" == "bf16" ]]; then kv_kib=64; else kv_kib=33; fi
    # Workspace arenas, allocator slack, and per-batch CUDA Graph families
    # (instantiated for every batch size in [1,C], so this grows with C).
    reserve_mib="${NINFER_RESERVE_MIB:-$(( 3151 + 371 * max_concurrency ))}"
    budget=$(( total_mib - artifact_mib - reserve_mib ))
    (( budget > 0 )) || fail "GPU ${idx}: ${total_mib} MiB cannot hold ${artifact_mib} MiB weights + ${reserve_mib} MiB reserve"
    ctx=$(( budget * 1024 / kv_kib ))
    (( ctx > 262144 )) && ctx=262144
    # Vision loads fixed GPU allocations (media cache/live buffers, Vision
    # runtime) and the engine's minimum runtime reservation scales with context.
    # Measured on 32 GB RTX 5090, C=2, int8 KV, MTP3 (2026-08-20):
    #   231936 ctx  -> FAILS  (needs 11.70 GiB after weights, 11.59 available)
    #   225280 ctx  -> FAILS  (11.46 GiB needed; reservation tracks context)
    #   196608 ctx  -> FITS   (9.79 GiB runtime + media buffers + 1 GiB headroom)
    # Cap vision engines at the validated ceiling; media input is further
    # bounded by the Vision runtime to min(ctx, 32768) merged tokens.
    # Split fleet: engines below NINFER_TEXT_ENGINES are text-only and keep the
    # full context; the rest get --vision and the reduced ceiling. Most work is
    # text, so paying vision's ~65k context cost on every card is wasteful.
    if engine_is_vision "$idx" && [[ -z "${NINFER_MAX_CONTEXT:-}" ]]; then
        local vision_cap="${NINFER_VISION_CONTEXT:-196608}"
        (( ctx > vision_cap )) && ctx=$vision_cap
    fi
    echo $(( ctx - ctx % 64 ))
}

build_args() {  # device index, port -> engine argv on stdout (NUL separated)
    local idx="$1" port="$2" ctx
    ctx="${NINFER_MAX_CONTEXT:-$(autosize_context "$idx")}"
    # Device isolation via CUDA_VISIBLE_DEVICES + --device 0, NOT --device N:
    # on multi-GPU hosts --device N crashes warmup with cudaErrorInvalidValue at
    # gqa_attention_prefill.cu:58 (some warmup path touches device 0 context).
    # Isolation also guarantees an engine can never allocate on a sibling card.
    local -a a=(
        env CUDA_VISIBLE_DEVICES="$idx"
        ninfer-serve "$artifact"
        --host 127.0.0.1
        --port "$port"
        --device 0
        --max-context "$ctx"
        --kv-capacity "${NINFER_KV_CAPACITY:-auto}"
        --kv-dtype "$kv_dtype"
        --max-concurrency "$max_concurrency"
        --pending-timeout-ms "${NINFER_PENDING_TIMEOUT_MS:-120000}"
    )
    # --draft-tokens and --lm-head-draft are only legal with --spec mtp|dflash;
    # the engine rejects them outright when speculation is off.
    if [[ "$spec" == "mtp" || "$spec" == "dflash" ]]; then
        a+=(--spec "$spec" --draft-tokens "$draft_tokens")
        [[ "${NINFER_LM_HEAD_DRAFT:-1}" == "1" ]] && a+=(--lm-head-draft)
    fi
    [[ "${NINFER_PRESERVE_THINKING:-1}" == "1" ]] && a+=(--preserve-thinking)
    engine_is_vision "$idx" && a+=(--vision)
    [[ "${NINFER_CORS:-0}" == "1" ]] && a+=(--cors)
    # Auth was resolved at top level (section 3a); this only applies it. A `fail`
    # here would run inside build_args' command substitution subshell and could
    # not stop the parent, which is exactly how an engine ends up public and
    # unauthenticated.
    [[ -n "${NINFER_API_KEY:-}" ]] && a+=(--api-key "$NINFER_API_KEY")
    # shellcheck disable=SC2206
    [[ -n "${NINFER_EXTRA_ARGS:-}" ]] && a+=(${NINFER_EXTRA_ARGS})
    printf '%s\0' "${a[@]}"
}

log "profile ${profile}: --max-concurrency ${max_concurrency} per engine, kv ${kv_dtype}, spec ${spec}"

# --- 4. Single-GPU fast path -------------------------------------------------
# One card needs no balancer: bind the engine straight to the public port so the
# container's PID 1 is ninfer-serve and signals reach it directly.

if (( GPUS == 1 )); then
    ctx="${NINFER_MAX_CONTEXT:-$(autosize_context 0)}"
    mapfile -d '' -t args < <(build_args 0 "$PORT")
    # Rewrite the loopback bind to the public interface for the direct path.
    for i in "${!args[@]}"; do
        [[ "${args[$i]}" == "127.0.0.1" ]] && args[$i]="0.0.0.0"
    done
    log "single-GPU: exec ninfer-serve on :${PORT} (ctx ${ctx})"
    exec "${args[@]}"
fi

# --- 5. Fan out --------------------------------------------------------------

logdir="${NINFER_LOG_DIR:-/var/log/ninfer}"
mkdir -p "$logdir"

declare -a engine_pids=() engine_ports=()
declare -A reported_dead=()   # engine idx -> 1 once its death has been logged

cleanup() {
    log "shutting down"
    [[ -n "${balancer_pid:-}" ]] && kill "$balancer_pid" 2>/dev/null || true
    for pid in "${engine_pids[@]:-}"; do kill "$pid" 2>/dev/null || true; done
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Engines are started with a stagger. Seven engines each reading a 20 GiB
# artifact at once thrashes a single volume: the reads interleave, none of them
# gets sequential throughput, and every engine loads slowly. Starting them a few
# seconds apart lets the first engine pull the file into page cache, so later
# engines read mostly from RAM. NINFER_START_STAGGER_S=0 restores the old
# all-at-once behaviour.
stagger="${NINFER_START_STAGGER_S:-20}"

for i in $(seq 0 $(( GPUS - 1 ))); do
    port=$(( ENGINE_BASE_PORT + i ))
    ctx="${NINFER_MAX_CONTEXT:-$(autosize_context "$i")}"
    mapfile -d '' -t args < <(build_args "$i" "$port")
    log "engine ${i}: device=${i} port=${port} ctx=${ctx}"
    "${args[@]}" > "${logdir}/engine-${i}.log" 2>&1 &
    engine_pids+=("$!")
    engine_ports+=("$port")
    if (( i < GPUS - 1 && stagger > 0 )); then
        sleep "$stagger"
    fi
done

# --- 6. Readiness ------------------------------------------------------------

# The whole function is wrapped so a refused connection stays silent. Bash
# prints "connect: Connection refused" from the /dev/tcp redirect itself, and
# with 7 ports probed every 5 seconds during a multi-minute load that noise
# buries every real log line.
probe_health() {
    {
        exec 3<>"/dev/tcp/127.0.0.1/$1" || return 1
        printf 'GET /health HTTP/1.0\r\nHost: localhost\r\n\r\n' >&3
        local line
        read -r line <&3 || { exec 3<&- 3>&-; return 1; }
        exec 3<&- 3>&-
        [[ "$line" == *" 200"* ]]
    } 2>/dev/null
}

# Readiness is REPORTING, never a kill switch.
#
# This process is PID 1. If it exits, the container dies, RunPod restarts it,
# every engine reloads 20 GiB from scratch, and sshd (started in section 0) dies
# with it — which is exactly the restart loop that made the pod impossible to
# debug. A slow load is not a failure: engines are served as they become ready
# and stragglers keep loading behind the balancer.
log "waiting for ${GPUS} engines (soft budget ${READY_TIMEOUT_S}s; never fatal)"
waited=0
ready=0
announced_partial=0

while true; do
    ready=0
    for p in "${engine_ports[@]}"; do probe_health "$p" && ready=$(( ready + 1 )); done
    (( ready == GPUS )) && { log "all ${GPUS} engines ready (${waited}s)"; break; }

    # A dead engine is logged and dropped, not fatal: six good replicas still
    # serve traffic. Only losing every engine is unrecoverable.
    alive=0
    for idx in "${!engine_pids[@]}"; do
        if kill -0 "${engine_pids[$idx]}" 2>/dev/null; then
            alive=$(( alive + 1 ))
        elif [[ "${reported_dead[$idx]:-0}" != "1" ]]; then
            reported_dead[$idx]=1
            log "WARNING: engine ${idx} exited; tail of ${logdir}/engine-${idx}.log:"
            tail -20 "${logdir}/engine-${idx}.log" >&2 || true
        fi
    done
    if (( alive == 0 )); then
        log "ERROR: every engine exited. Container will stay up for inspection over ssh."
        log "       logs: ${logdir}/engine-*.log"
        # Sleep instead of exiting so the pod remains debuggable.
        while true; do sleep 3600; done
    fi

    sleep 5; waited=$(( waited + 5 ))
    (( waited % 60 == 0 )) && log "  ${ready}/${GPUS} ready, ${alive} loading (${waited}s)"

    # Past the soft budget, start serving with whatever is up rather than
    # waiting for the slowest engine.
    if (( waited >= READY_TIMEOUT_S && ready > 0 && announced_partial == 0 )); then
        log "soft budget reached: starting balancer with ${ready}/${GPUS} ready; the rest join as they finish"
        announced_partial=1
        break
    fi
done

# Engines that were still loading are added once they answer /health, so the
# balancer never routes to a port that is not serving yet.
if (( ready < GPUS )); then
    live_ports=()
    for p in "${engine_ports[@]}"; do probe_health "$p" && live_ports+=("$p"); done
    log "balancer starts with ${#live_ports[@]} engine(s): ${live_ports[*]}"
    engine_ports=("${live_ports[@]}")
fi

# --- 7. Balancer -------------------------------------------------------------

# Clear the EXIT trap: cleanup() would otherwise kill every engine at the exact
# moment the balancer takes over.
#
# This shell then SUPERVISES rather than exec'ing the balancer. With exec, a
# container stop delivers SIGTERM to the balancer while the engines sit
# orphaned in the process group: they still die (the kernel reclaims VRAM on
# process death) but in-flight streams are severed instead of drained.
# Supervising lets shutdown() TERM each engine — measured ~2s to release a
# card's 30 GiB — before the container goes away.
trap - EXIT

shutdown() {
    log "received shutdown signal; draining engines"
    [[ -n "${vision_balancer_pid:-}" ]] && kill -TERM "$vision_balancer_pid" 2>/dev/null || true
    [[ -n "${text_balancer_pid:-}" ]] && kill -TERM "$text_balancer_pid" 2>/dev/null || true
    for pid in "${engine_pids[@]:-}"; do kill -TERM "$pid" 2>/dev/null || true; done
    local deadline=$(( SECONDS + ${NINFER_SHUTDOWN_GRACE_S:-30} ))
    while (( SECONDS < deadline )); do
        pgrep -f ninfer-serve >/dev/null 2>&1 || break
        sleep 1
    done
    pkill -9 -f ninfer-serve 2>/dev/null || true
    log "shutdown complete"
    exit 0
}
trap shutdown INT TERM

# Two balancers so a text request never lands on a vision engine (smaller
# context) and vice versa:
#   :$PORT      -> text engines      (full context)
#   :$((PORT+1)) -> vision engines   (media-capable, reduced context)
text_ports=(); vision_ports=()
for idx in "${!engine_ports[@]}"; do
    if engine_is_vision "$idx"; then vision_ports+=("${engine_ports[$idx]}")
    else text_ports+=("${engine_ports[$idx]}"); fi
done

if (( ${#vision_ports[@]} > 0 )); then
    vision_port=$(( PORT + 1 ))
    log "vision balancer on :${vision_port} -> ${vision_ports[*]}"
    NINFER_ENGINE_PORTS="${vision_ports[*]}" NINFER_LISTEN_PORT="$vision_port" \
        setsid perl /usr/local/bin/ninfer-balancer < /dev/null \
        > "${logdir}/balancer-vision.log" 2>&1 &
    vision_balancer_pid=$!
fi

# Text balancer becomes PID 1 via exec. If every engine is a vision engine,
# serve them on the main port instead of leaving it unbound.
main_ports=("${text_ports[@]}")
(( ${#main_ports[@]} == 0 )) && main_ports=("${vision_ports[@]}")

log "text balancer on :${PORT} -> ${main_ports[*]}"
NINFER_ENGINE_PORTS="${main_ports[*]}" NINFER_LISTEN_PORT="$PORT" \
    perl /usr/local/bin/ninfer-balancer &
text_balancer_pid=$!

# Remain PID 1 so signals reach shutdown() and engines drain on container stop.
wait "$text_balancer_pid"
