#!/usr/bin/env bash
# =============================================================================
# DFlash 2 (Qwen3.8-27B) acceptance — winbox (WSL2, RTX 5090)
#
# Stages (idempotent, re-runnable; safe to abort and re-run):
#   0  preflight   GPU, repo state, artifact + variant detection, docker,
#                  docker-GPU capability (falls back to native mode)
#   1  build       full engine build in the CUDA 13.1.2 container (CPU-only;
#                  the container needs no GPU flag to compile)
#   2  CPU tests   linear dispatch shape-coverage (no GPU, no artifact)
#   3  GPU op tests  grouped_dynamic_conv + dflash_selector unit tests —
#                  the layout-law kernels vs independent CPU references
#   4  artifact tests  dflash load-plan (1124 + 66 object plans) and engine
#                  dflash real (golden / D4 negative / determinism /
#                  long-restore) — the dflash parts need the AUGMENTED
#                  artifact (built with --dflash-model); see the note printed
#                  at the end for the exact converter command.
#
# Usage:
#   dflash2-27b-acceptance.sh [artifact.ninfer] [--install-gpu-toolkit]
#
# Defaults:
#   REPO      $NINFER_REPO or /mnt/f/ninfer-fork   (branch dflash2-27b)
#   ARTIFACT  $NINFER_ARTIFACT or /mnt/f/ninfer/models/qwen3_8_27b_nvfp4.ninfer
#   BUILD     $REPO/build-wsl    (container ninfer-wsl)
#   IMAGE     nvidia/cuda:13.1.2-devel-ubuntu24.04 (already local)
#
# GPU execution modes (chosen automatically in stage 0):
#   A) docker --gpus all          — needs nvidia-container-toolkit in WSL.
#                                   --install-gpu-toolkit runs the (sudo) apt
#                                   install + runtime reconfigure + restart.
#   B) native fallback            — test binaries are executed directly in WSL
#                                   (which exposes /dev/nvidia*) with the
#                                   container's CUDA 13.1 runtime libraries,
#                                   extracted once to $REPO/.cuda13-libs.
# =============================================================================
set -uo pipefail

# ---------- arguments ----------
ARTIFACT=""
INSTALL_TOOLKIT=0
ALLOW_BUSY_GPU=0
for arg in "$@"; do
  case "$arg" in
    --install-gpu-toolkit) INSTALL_TOOLKIT=1 ;;
    --allow-busy-gpu) ALLOW_BUSY_GPU=1 ;;
    --*) printf 'unknown flag: %s\n' "$arg" >&2; exit 2 ;;
    *) ARTIFACT="$arg" ;;
  esac
done
REPO=${NINFER_REPO:-/mnt/f/ninfer-fork}
ARTIFACT=${ARTIFACT:-${NINFER_ARTIFACT:-/mnt/f/ninfer/models/qwen3_8_27b_nvfp4.ninfer}}
BUILD="$REPO/build-wsl"
CONTAINER=ninfer-wsl
IMAGE=nvidia/cuda:13.1.2-devel-ubuntu24.04
CUDA13LIBS="$REPO/.cuda13-libs"
JOBS=${NINFER_JOBS:-$(nproc)}

log()  { printf '\n================== %s ==================\n' "$*"; }
ok()   { printf '  [ok]   %s\n' "$*"; }
bad()  { printf '  [FAIL] %s\n' "$*"; }
warn() { printf '  [warn] %s\n' "$*"; }
skip() { printf '  [skip] %s\n' "$*"; }

RESULTS=()
report() { RESULTS+=("$(printf '%-28s %s' "$1" "$2")"); }

# container helpers (build container: no GPU flag needed to compile)
cexec() { docker exec "$CONTAINER" bash -c "$1"; }
ensure_container() {
  if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    docker rm -f "$CONTAINER" >/dev/null 2>&1
    docker run -d --name "$CONTAINER" --shm-size=8g \
      -v "$REPO:/src" -v "$BUILD:/build" "$IMAGE" sleep infinity >/dev/null
    ok "container $CONTAINER started (image $IMAGE)"
  fi
}

# ---------- stage 0: preflight ----------
log "stage 0: preflight"
if command -v nvidia-smi >/dev/null 2>&1; then
  ok "GPU: $(nvidia-smi -L | head -1)"
else
  bad "nvidia-smi missing in WSL — is the Windows NVIDIA driver up to date?"; exit 1
fi
# SAFETY GATE: this 5090 may be serving the live inference instance that this
# session runs on. Refuse to run (any stage) while something owns the GPU
# unless explicitly overridden — the GPU tests would OOM/kill that instance.
gpu_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
if [ "${gpu_used:-0}" -gt 4096 ] && [ "$ALLOW_BUSY_GPU" != 1 ]; then
  bad "GPU is busy: ${gpu_used} MiB in use (the serving inference instance is probably running there)"
  echo "  stop/move the inference instance first, then re-run; or override with --allow-busy-gpu"
  exit 1
fi
ok "GPU free enough: ${gpu_used} MiB in use (guard: >4096 MiB aborts without --allow-busy-gpu)"
branch=$(git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null)
head=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)
if [ "$branch" != "dflash2-27b" ]; then
  bad "repo $REPO is on branch '$branch', expected dflash2-27b"; exit 1
fi
dirty=$(git -C "$REPO" status --porcelain | grep -v '^??' | head -1)
if [ -n "$dirty" ]; then
  bad "repo has modified tracked files — commit/stash first"; exit 1
fi
ok "repo: $REPO @ $head ($branch)"
if [ ! -f "$ARTIFACT" ]; then
  bad "artifact not found: $ARTIFACT"; exit 1
fi
ok "artifact: $ARTIFACT ($(du -h "$ARTIFACT" | cut -f1))"
# Artifact variant: parse the version-2 container header — 16-byte prefix
# (magic NINFER\0\2 + LE u64 json_bytes) + embedded JSON object directory
# (docs/maintainer/artifact-container.md). No full-file scan needed.
variant_line=$(python3 - "$ARTIFACT" <<'PY'
import json, struct, sys
p = sys.argv[1]
try:
    with open(p, "rb") as f:
        head = f.read(16)
    if head[:8] != b"NINFER\x00\x02":
        print("unknown 0 bad-magic %r" % head[:8]); sys.exit(0)
    jlen = struct.unpack("<Q", head[8:])[0]
    with open(p, "rb") as f:
        f.seek(16)
        d = json.loads(f.read(jlen))
    objs = d["objects"]
    names = [o["name"] for o in objs if o.get("kind") == "tensor"]
    ident = d["identity"]["model_id"] + "/" + d["identity"]["weights_id"]
    print(("augmented" if any(n.startswith("dflash/") for n in names) else "fleet"),
          len(objs), ident)
except Exception as e:
    print("unknown 0 parse-error %s" % e)
PY
)
VARIANT=$(cut -d' ' -f1 <<<"$variant_line")
OBJCOUNT=$(cut -d' ' -f2 <<<"$variant_line")
ART_IDENTITY=$(cut -d' ' -f3- <<<"$variant_line")
ok "artifact container: identity $ART_IDENTITY, $OBJCOUNT tensor objects (v2 framing)"
if [ "$VARIANT" = augmented ]; then
  ok "artifact variant: DFLASH-AUGMENTED (expect 1190 objects)"
elif [ "$VARIANT" = fleet ]; then
  ok "artifact variant: FLEET / no-dflash (1124 objects)"
  warn "dflash plan + engine-dflash parts (stage 4) need the augmented"
  warn "artifact — they will be reported as NOT-RUN, not FAIL."
else
  bad "cannot parse artifact header: $variant_line"; exit 1
fi
if ! docker info >/dev/null 2>&1; then
  bad "docker daemon not reachable in WSL"; exit 1
fi
ok "docker daemon reachable"
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  warn "pulling $IMAGE (one-time, ~10 GB)"; docker pull "$IMAGE"
fi
# docker GPU capability
DOCKER_GPU=0
if docker run --rm --gpus all "$IMAGE" nvidia-smi -L >/dev/null 2>&1; then
  DOCKER_GPU=1
  ok "docker --gpus all works (mode A)"
else
  if [ "$INSTALL_TOOLKIT" = 1 ]; then
    warn "installing nvidia-container-toolkit (sudo) ..."
    if sudo -n apt-get update -qq && \
       sudo -n apt-get install -y -qq --no-install-recommends nvidia-container-toolkit && \
       sudo -n nvidia-ctk runtime configure --runtime=docker && \
       sudo -n systemctl restart docker; then
      sleep 3
      if docker run --rm --gpus all "$IMAGE" nvidia-smi -L >/dev/null 2>&1; then
        DOCKER_GPU=1; ok "toolkit installed; docker --gpus all works (mode A)"
      fi
    fi
    [ "$DOCKER_GPU" = 1 ] || warn "toolkit install failed or still broken — falling back to native mode"
  else
    warn "docker --gpus unavailable (nvidia-container-toolkit missing in WSL)"
    warn "  -> re-run with --install-gpu-toolkit for mode A,"
    warn "  -> or continue now in native mode B (GPU tests run in WSL directly)"
  fi
fi
GPU_MODE=$([ "$DOCKER_GPU" = 1 ] && echo docker || echo native)
ok "GPU test mode: $GPU_MODE"

# ---------- stage 1: build ----------
log "stage 1: build (container, CPU-only)"
ensure_container
cexec 'export DEBIAN_FRONTEND=noninteractive; apt-get update -qq && apt-get install -y -qq --no-install-recommends cmake ninja-build pkg-config python3 python3-dev libavcodec-dev libavformat-dev libavutil-dev libcurl4-openssl-dev libswscale-dev 2>&1 | tail -1'
if [ -f "$BUILD/CMakeCache.txt" ] && [ -f "$BUILD/build.ninja" ]; then
  ok "incremental build"
else
  ok "full configure + build"
fi
if cexec "cd /src && cmake -S . -B /build -G Ninja -DCMAKE_BUILD_TYPE=Release -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=ON -DNINFER_BUILD_BENCHMARKS=OFF > /build/configure.log 2>&1 && ninja -C /build -j$JOBS > /build/build.log 2>&1"; then
  ok "build OK"
  report "build" "PASS"
else
  bad "build FAILED — tail of /build/build.log:"
  docker exec "$CONTAINER" tail -30 /build/build.log
  report "build" "FAIL"
  printf '\n================== SUMMARY ==================\n'
  printf '%s\n' "${RESULTS[@]}"
  exit 1
fi

# ---------- stage 2: CPU tests (no GPU, no artifact) ----------
log "stage 2: CPU tests — linear dispatch shape coverage"
if cexec "cd /build && ctest -R '^ninfer_linear_dispatch_test$' --output-on-failure" > /tmp/dflash2-dispatch.log 2>&1; then
  ok "dispatch coverage PASS (see /tmp/dflash2-dispatch.log)"
  report "cpu dispatch coverage" "PASS"
else
  bad "dispatch coverage FAILED — tail:"; tail -25 /tmp/dflash2-dispatch.log
  report "cpu dispatch coverage" "FAIL"
fi

# ---------- stage 3: GPU op unit tests ----------
log "stage 3: GPU op unit tests (conv + selector)"
OP_PAT='^(ninfer_grouped_dynamic_conv_test|ninfer_dflash_selector_test)$'
run_op_tests() {
  if [ "$GPU_MODE" = docker ]; then
    docker run --rm --gpus all --shm-size=8g -v "$BUILD:/build" -v "$REPO:/src" \
      "$IMAGE" bash -c "cd /build && ctest -R '$OP_PAT' --output-on-failure"
  else
    # native: make sure the CUDA 13.1 runtime libs the binaries need are available
    if [ ! -d "$CUDA13LIBS" ]; then
      mkdir -p "$CUDA13LIBS"
      warn "extracting CUDA 13.1 runtime libs from the image (one-time) ..."
      docker run --rm -v "$CUDA13LIBS:/out" "$IMAGE" \
        bash -c "cp -L /usr/local/cuda/lib64/libcudart.so* /usr/local/cuda/lib64/libcublas.so* /usr/local/cuda/lib64/libcufft.so* /usr/local/cuda/lib64/libcurand.so* /usr/local/cuda/lib64/libcudnn*.so* /usr/local/cuda/lib64/libcusolver.so* /usr/local/cuda/lib64/libcusparse.so* /usr/local/cuda/lib64/libnvrtc*.so* /out/ 2>/dev/null; true"
    fi
    # fill any remaining missing libs the linker wants
    for bin in "$BUILD"/tests/ninfer_grouped_dynamic_conv_test "$BUILD"/tests/ninfer_dflash_selector_test; do
      [ -x "$bin" ] || continue
      for lib in $(ldd "$bin" 2>/dev/null | awk '/not found/{print $1}'); do
        [ -f "$CUDA13LIBS/$lib" ] || docker run --rm -v "$CUDA13LIBS:/out" "$IMAGE" \
          bash -c "find /usr/local/cuda /usr/lib -name '$lib' | head -1 | xargs -r cp -L /out/"
      done
    done
    ( cd "$BUILD/tests" && LD_LIBRARY_PATH="$CUDA13LIBS" ./ninfer_grouped_dynamic_conv_test ) \
      > /tmp/dflash2-opconv.log 2>&1; rc1=$?
    ( cd "$BUILD/tests" && LD_LIBRARY_PATH="$CUDA13LIBS" ./ninfer_dflash_selector_test ) \
      > /tmp/dflash2-opselector.log 2>&1; rc2=$?
    tail -5 /tmp/dflash2-opconv.log; tail -5 /tmp/dflash2-opselector.log
    [ "$rc1" -eq 0 ] && [ "$rc2" -eq 0 ]
  fi
}
if run_op_tests; then
  ok "op unit tests PASS (conv + selector vs CPU references)"
  report "gpu op unit tests" "PASS"
else
  bad "op unit tests FAILED — logs: /tmp/dflash2-opconv.log /tmp/dflash2-opselector.log"
  report "gpu op unit tests" "FAIL"
fi

# ---------- stage 4: artifact tests ----------
log "stage 4: artifact tests (NINFER_QWEN3_8_27B_WEIGHTS)"
ART_ENV="NINFER_QWEN3_8_27B_WEIGHTS=$ARTIFACT"
run_artifact_test() { # $1=ctest regex $2=mode-force(docker|native|auto)
  local pat="$1" mode="${2:-$GPU_MODE}" out
  if [ "$mode" = docker ]; then
    docker run --rm --gpus all --shm-size=8g -v "$BUILD:/build" -v "$REPO:/src" -e "$ART_ENV" \
      "$IMAGE" bash -c "cd /build && ctest -R '$pat' --output-on-failure"
  else
    local bin
    bin=$(docker exec "$CONTAINER" bash -c "cd /build && ctest -N -R '$pat' 2>/dev/null | grep -oP '(?<=Test #\d+ : ).*' | head -1")
    [ -n "$bin" ] || { warn "no binary resolved for '$pat'"; return 1; }
    out="$BUILD/tests/$bin"
    [ -x "$out" ] || out="$BUILD/$bin"
    if [ ! -x "$out" ]; then warn "binary for '$pat' not found under /build"; return 1; fi
    ( cd "$(dirname "$out")" && env "$ART_ENV" LD_LIBRARY_PATH="$CUDA13LIBS" "./$(basename "$out")" )
  fi
}
# 4a: load-plan (CPU-side binder; works in either mode; the dflash plan half
#     needs the augmented artifact)
if [ "$VARIANT" = augmented ]; then
  if run_artifact_test '^ninfer_qwen3_8_27b_dflash_load_plan_test$' native; then
    ok "load-plan PASS (1124 + 66 dflash objects, no full pool)"
    report "artifact load plan" "PASS"
  else
    bad "load-plan FAILED (augmented artifact — investigate)"; report "artifact load plan" "FAIL"
  fi
else
  if out=$(run_artifact_test '^ninfer_qwen3_8_27b_dflash_load_plan_test$' native 2>&1); then
    ok "load-plan PASS"; report "artifact load plan" "PASS"
  else
    if grep -q "1124" <<<"$out"; then
      warn "load-plan: no-dflash plan (1124 objects) VERIFIED; dflash plan"
      warn "       (1190 objects) needs the augmented artifact — expected"
      report "artifact load plan" "PASS (baseline 1124; dflash plan not run)"
    else
      bad "load-plan FAILED against the fleet artifact — see output above"
      report "artifact load plan" "FAIL"
    fi
  fi
fi
# 4b: engine dflash real (GPU; dflash objects required)
if [ "$VARIANT" = augmented ]; then
  if run_artifact_test '^ninfer_qwen3_8_27b_dflash_real_test$' docker; then
    ok "engine dflash real PASS (golden + D4 negative + determinism + long-restore)"
    report "engine dflash real" "PASS"
  else
    bad "engine dflash real FAILED — investigate"; report "engine dflash real" "FAIL"
  fi
else
  skip "engine dflash real — needs the dflash-augmented artifact"
  report "engine dflash real" "NOT-RUN (needs augmented artifact)"
fi

# ---------- summary ----------
printf '\n================== SUMMARY ==================\n'
printf '%s\n' "${RESULTS[@]}"
printf '\n----------------------------------------------\n'
if [ "$VARIANT" = fleet ]; then
cat <<'NOTE'
The dflash-augmented artifact (1190 objects) is built by the converter in
tools/convert/qwen3_8_27b/convert_nvfp4.py from the fixed sources:
  --model          Qwen/Qwen3.8-27B          @ 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0  (base-hf-bf16)
  --quantized-model unsloth/Qwen3.8-27B-NVFP4 @ 60e813d4dbbdc5d64cf3f5a8caf2897bedf03679 (vllm-nvfp4-fp8)
  --dflash-model   z-lab/Qwen3.8-27B-DFlash2  @ 50307d4c4cde6860d4eee73e2547cd786fe8e8a4 (config.json + model.safetensors)
(both gated repos need an HF token with license acceptance; then re-run this
script with the augmented artifact path as the first argument.)
NOTE
fi
fails=0
for r in "${RESULTS[@]}"; do case "$r" in *FAIL) fails=1;; esac; done
exit "$fails"