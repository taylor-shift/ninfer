#!/usr/bin/env bash
# =============================================================================
# DFlash 2 (Qwen3.8-27B) acceptance — winbox (WSL2, RTX 5090)
#
# Stages (idempotent, re-runnable; safe to abort and re-run):
#   0  preflight   GPU, repo state, artifact + variant detection, docker,
#                  docker-GPU capability (falls back to native mode)
#   1  build       full engine build in the CUDA 13.1.2 container (CPU-only;
#                  the container needs no GPU flag to compile) — verified by
#                  the actual test binaries, not the exit code alone
#   2  CPU tests   linear dispatch shape-coverage (no GPU, no artifact)
#   3  GPU op tests  grouped_dynamic_conv + dflash_selector unit tests —
#                  the layout-law kernels vs independent CPU references
#   4  artifact tests  dflash load-plan (1124 + 66 object plans) and engine
#                  dflash real (golden / D4 negative / determinism /
#                  long-restore) — the dflash parts need the AUGMENTED
#                  artifact (built with --dflash-model)
#
# Usage:
#   dflash2-27b-acceptance.sh [artifact.ninfer] [--install-gpu-toolkit]
#                             [--allow-busy-gpu]
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
#                                   extracted once (only the missing ones) to
#                                   $REPO/.cuda13-libs.
#
# NOTE: the build runs inside the docker container. Ctrl-C at the terminal
# kills the docker-exec client but NOT the in-container build. The gate judges
# freshness by an idempotent ninja re-run (rc=0 only when fully up-to-date) —
# "test binaries exist" is NOT proof of a fresh build: a failed ninja leaves
# stale binaries on disk, and running those silently tests old code.
# =============================================================================
set -uo pipefail
trap 'printf "\n[aborted — in-container build state may be partial; re-run to resume]\n"; exit 130' INT TERM

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
# The build container is committed after stage 1 (apt toolchain + libav* etc.
# installed): the GPU-mode docker runs use that image, because the base CUDA
# image lacks the apt libraries the test binaries link against.
FULL_IMAGE=ninfer-wsl-full
test_image() { docker image inspect "$FULL_IMAGE" >/dev/null 2>&1 && printf '%s' "$FULL_IMAGE" || printf '%s' "$IMAGE"; }
CUDA13LIBS="$REPO/.cuda13-libs"
JOBS=${NINFER_JOBS:-$(nproc)}

log()  { printf '\n================== %s ==================\n' "$*"; }
ok()   { printf '  [ok]   %s\n' "$*"; }
bad()  { printf '  [FAIL] %s\n' "$*"; }
warn() { printf '  [warn] %s\n' "$*"; }
info() { printf '  [..]   %s\n' "$*" >&2; }
skip() { printf '  [skip] %s\n' "$*"; }

RESULTS=()
report() { RESULTS+=("$(printf '%-30s %s' "$1" "$2")"); }

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

# Count test binaries that ctest expects but the build tree does not have.
# (Binaries land in /build/tests/<name>; a few in /build/<name>.)
# Exact ctest test name for a unique substring (machine-derived;
# the full target names are never typed by hand).
ctest_name() {
  cexec "cd /build && ctest -N 2>/dev/null" | sed -n 's/^ *Test *#[0-9]*: *//p' | grep -F "$1" | head -1
}
# ctest helpers taking the regex as an argument: keeps \"...2>/dev/null...\"
# out of the same double-quoted string inside $( ) (a bash 5.2 parser bug
# rejects that combination), and ctest -R exits 0 even when no test matches
# — so ctest_count doubles as the no-match guard.
ctest_count() { docker exec "$CONTAINER" bash -c "cd /build && ctest -N -R '$1' 2>/dev/null" | grep -c 'Test *#'; }
# The CPU dispatch test loads libcuda.so.1 at startup even though it never calls a driver
# API; on the no-GPU build container the devel image's compat shim satisfies the loader.
ctest_run() { docker exec "$CONTAINER" bash -c "cd /build && LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat ctest -R '$1' --output-on-failure"; }

missing_test_binaries() {
  cexec "cd /build && out=\$(ctest -N 2>/dev/null | sed -n 's/^ *Test *#[0-9]*: *//p') && m=0 && n=0 && for t in \$out; do n=\$((n+1)); if [ ! -f tests/\$t ] && [ ! -f \$t ]; then m=\$((m+1)); fi; done; echo \"\$m/\$n\""
}

# Extract only the shared libraries the WSL host is missing (per ldd).
# Preferred source: the build container itself (its apt install provides libav*
# & co.) — copied through the /src mount ($CUDA13LIBS = /src/.cuda13-libs inside
# the container). Fallback: docker run against the base CUDA image (CUDART & co).
extract_missing_libs() { # $@ = test binary paths (WSL-side)
  for bin in "$@"; do
    [ -x "$bin" ] || continue
    for lib in $(ldd "$bin" 2>/dev/null | awk '/not found/{print $1}'); do
      [ -f "$CUDA13LIBS/$lib" ] && continue
      info "extracting $lib (one-time) ..."
      if ! cexec "mkdir -p /src/.cuda13-libs && for p in /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu /usr/lib /lib /usr/local/cuda; do [ -e \"\$p/$lib\" ] && cp -L \"\$p/$lib\" /src/.cuda13-libs/ && exit 0; done; exit 1" 2>/dev/null; then
        docker run --rm --entrypoint bash -v "$CUDA13LIBS:/out" "$IMAGE" \
          -c "f=\$(find /usr/local/cuda /usr/lib /lib /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu -name '$lib' -not -name '*stubs*' 2>/dev/null | head -1); [ -n \"\$f\" ] && cp -L \"\$f\" /out/ || exit 1"
      fi
      [ -f "$CUDA13LIBS/$lib" ] || warn "could not extract $lib — GPU tests may fail to load"
    done
  done
}

# ---------- stage 0: preflight ----------
log "stage 0: preflight"
if command -v nvidia-smi >/dev/null 2>&1; then
  ok "GPU: $(nvidia-smi -L | head -1)"
else
  bad "nvidia-smi missing in WSL — is the Windows NVIDIA driver up to date?"; exit 1
fi
# SAFETY GATE: this 5090 may be serving the live inference instance. Refuse
# to run (any stage) while something owns the GPU unless overridden — the
# GPU tests would OOM/kill that instance.
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
  ok "incremental build (existing build tree)"
else
  ok "full configure + build"
fi
info "ninja -j$JOBS running; full build ~15-30 min, incremental ~1-5 min"
info "live log: $BUILD/build.log   (WSL: tail -f $BUILD/build.log)"
info "NOTE: Ctrl-C here kills the docker client, not the in-container build;"
info "the script proves the tree is up-to-date (idempotent ninja re-run) and"
info "ABORTS if it is not — stale binaries must never test old code."
build_rc=0
cexec "cd /src && cmake -S . -B /build -G Ninja -DCMAKE_BUILD_TYPE=Release -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=ON -DNINFER_BUILD_BENCHMARKS=OFF > /build/configure.log 2>&1 && ninja -C /build -j$JOBS > /build/build.log 2>&1" || build_rc=$?
# The docker-exec client may be killed (Ctrl-C) while the in-container build
# keeps running — and a FAILED ninja leaves older test binaries on disk, so
# "binaries present" is NOT evidence of a fresh build (that gate once let
# stale binaries silently test old code while ninja rc=1 went unreported).
# Oracle: an idempotent ninja re-run returns 0 only when the tree is fully
# up-to-date — and it resumes/finishes an interrupted build.
if cexec "pgrep -f '[n]inja -C /build' >/dev/null" 2>/dev/null; then
  warn "build still running in container (client interrupted); waiting for it ..."
  for _ in $(seq 1 360); do
    cexec "pgrep -f '[n]inja -C /build' >/dev/null" 2>/dev/null || break
    sleep 10
  done
fi
final_rc=0
cexec "cd /src && ninja -C /build -j$JOBS >> /build/build.log 2>&1" || final_rc=$?
missing=$(missing_test_binaries)
missing_n=${missing%%/*}
missing_t=${missing#*/}
if [ "$final_rc" != "0" ] || [ "$missing_n" != "0" ]; then
  bad "build FAILED (ninja re-run rc=$final_rc; first pass rc=$build_rc; binaries missing: $missing_n/$missing_t)"
  bad "aborting: running stale test binaries would silently test old code"
  cexec "grep -B3 -A10 -iE 'error|FAILED' /build/build.log | tail -60" || true
  report "build" "FAIL"
else
  ok "build verified FRESH: ninja up-to-date re-run rc=0 (first pass rc=$build_rc), all $missing_t test binaries present (log: $BUILD/build.log)"
  report "build" "PASS"
  # One-time image bake: the GPU-mode docker runs (stages 3/4) need the apt
  # toolchain + libav* this container just installed; the base CUDA image lacks
  # them. Committing is cheap and idempotent (fresh layer diff).
  if docker commit "$CONTAINER" "$FULL_IMAGE" >/dev/null 2>&1; then
    ok "committed $CONTAINER -> $FULL_IMAGE (test image for GPU-mode runs)"
  else
    warn "docker commit failed — GPU-mode runs will use the base image (apt libs missing)"
  fi
fi

# ---------- stage 2: CPU tests (no GPU, no artifact) ----------
log "stage 2: CPU tests — linear dispatch shape coverage"
DISPATCH_TEST=$(ctest_name 'linear_dispatch_test')
if [ -z "$DISPATCH_TEST" ]; then
  bad "linear_dispatch test not registered in ctest"; report "cpu dispatch coverage" "FAIL"
elif [ "$(ctest_count "$DISPATCH_TEST")" -ge 1 ] && ctest_run "$DISPATCH_TEST"; then
  ok "dispatch coverage PASS ($DISPATCH_TEST)"
  report "cpu dispatch coverage" "PASS"
else
  bad "dispatch coverage FAILED (output above)"
  report "cpu dispatch coverage" "FAIL"
fi

# ---------- stage 3: GPU op unit tests ----------
log "stage 3: GPU op unit tests (conv + selector)"
CONV_TEST=$(ctest_name 'grouped_dynamic_conv_test')
SEL_TEST=$(ctest_name 'dflash_selector_test')
[ -n "$CONV_TEST" ] && [ -n "$SEL_TEST" ] || { bad "op tests not registered in ctest"; report "gpu op unit tests" "FAIL"; }
# Auto-diagnosis: re-run a failed test under compute-sanitizer inside the GPU
# container (mode A). Names the faulting kernel, address, PC and source line
# (tests are compiled with -lineinfo). Report: /tmp/dflash2-sanitize-<name>.log
sanitize_test() { # $1=binary name $2=timeout-seconds [$3...=extra docker run flags]
  local name="$1" secs="$2"; shift 2
  local out="/tmp/dflash2-sanitize-$name.log"
  [ -x "$BUILD/tests/$name" ] || return 0
  if [ "$GPU_MODE" != docker ]; then
    info "sanitizer auto-run needs mode A (docker --gpus); native mode: run dflash2-sanitize.sh"
    return 0
  fi
  warn "auto-diagnosing $name under compute-sanitizer (timeout ${secs}s) ..."
  timeout "$secs" docker run --rm --gpus all --shm-size=8g -v "$BUILD:/build" -v "$REPO:/src" \
    -e NINFER_LOG_OPS=1 \
    "$@" \
    "$(test_image)" \
    bash -c "cd /build && /usr/local/cuda/bin/compute-sanitizer --tool memcheck --show-backtrace yes --print-limit 30 ./tests/$name" \
    > "$out" 2>&1
  echo "---- sanitizer report (tail) -> $out"
  tail -25 "$out"
}
run_op_tests() {
  FAILED_OPS=""
  if [ "$GPU_MODE" = docker ]; then
    docker run --rm --gpus all --shm-size=8g -v "$BUILD:/build" -v "$REPO:/src" \
      -e NINFER_LOG_OPS=1 \
      "$(test_image)" bash -c "cd /build && ctest -R '^(${CONV_TEST}|${SEL_TEST})\$' --output-on-failure" \
      > /tmp/dflash2-opdocker.log 2>&1
    local rc=$?
    cat /tmp/dflash2-opdocker.log
    FAILED_OPS=$(sed -n 's/^ *[0-9][0-9]* - \([A-Za-z0-9_]*\) (.*/\1/p' /tmp/dflash2-opdocker.log | sort -u | tr '\n' ' ')
    return $rc
  else
    extract_missing_libs "$BUILD/tests/$CONV_TEST" "$BUILD/tests/$SEL_TEST"
    info "running $CONV_TEST (native) ..."
    ( cd "$BUILD/tests" && NINFER_LOG_OPS=1 LD_LIBRARY_PATH="$CUDA13LIBS" ./"$CONV_TEST" ) \
      > /tmp/dflash2-opconv.log 2>&1; rc1=$?
    info "running $SEL_TEST (native) ..."
    ( cd "$BUILD/tests" && NINFER_LOG_OPS=1 LD_LIBRARY_PATH="$CUDA13LIBS" ./"$SEL_TEST" ) \
      > /tmp/dflash2-opselector.log 2>&1; rc2=$?
    tail -3 /tmp/dflash2-opconv.log; tail -3 /tmp/dflash2-opselector.log
    [ "$rc1" -ne 0 ] && FAILED_OPS="$FAILED_OPS $CONV_TEST"
    [ "$rc2" -ne 0 ] && FAILED_OPS="$FAILED_OPS $SEL_TEST"
    [ "$rc1" -eq 0 ] && [ "$rc2" -eq 0 ]
  fi
}
if run_op_tests; then
  ok "op unit tests PASS (conv + selector vs CPU references)"
  report "gpu op unit tests" "PASS"
else
  bad "op unit tests FAILED — logs: /tmp/dflash2-opconv.log /tmp/dflash2-opselector.log"
  report "gpu op unit tests" "FAIL"
  [ -n "${FAILED_OPS:-}" ] || FAILED_OPS="$SEL_TEST"
  for n in $FAILED_OPS; do
    if [ "$n" = "$CONV_TEST" ]; then sanitize_test "$n" 600; else sanitize_test "$n" 900; fi
  done
fi

# ---------- stage 4: artifact tests ----------
log "stage 4: artifact tests (NINFER_QWEN3_8_27B_WEIGHTS)"
ART_ENV="NINFER_QWEN3_8_27B_WEIGHTS=$ARTIFACT"
LOAD_TEST=$(ctest_name 'qwen3_8_27b_dflash_load_plan_test')
REAL_TEST=$(ctest_name 'qwen3_8_27b_dflash_real_test')
run_artifact_test() { # $1=ctest regex $2=mode (docker|native)
  local pat="$1" mode="$2" bin
  if [ "$mode" = docker ]; then
    # The artifact lives on the WSL side; mount its directory and remap the
    # env var to the in-container path (the WSL path does not exist there).
    docker run --rm --gpus all --shm-size=8g -v "$BUILD:/build" -v "$REPO:/src" \
      -v "$(dirname "$ARTIFACT"):/artifacts" \
      -e "NINFER_QWEN3_8_27B_WEIGHTS=/artifacts/$(basename "$ARTIFACT")" \
      -e NINFER_LOG_OPS=1 \
      "$(test_image)" bash -c "cd /build && ctest -R '$pat' --output-on-failure"
    return $?
  fi
  # For these registrations the ctest test name is the binary name; strip the
  # regex decoration (^ $ ( ) |) to get it.
  bin=$(echo "$pat" | sed -e 's/^\^//' -e 's/\$$//' -e 's/[()|]//g' -e 's/\$//g')
  local out="$BUILD/tests/$bin"
  [ -x "$out" ] || out="$BUILD/$bin"
  if [ ! -x "$out" ]; then warn "binary for '$pat' not found under $BUILD"; return 1; fi
  extract_missing_libs "$out"
  info "running $bin (native, artifact: $ARTIFACT) ..."
  ( cd "$(dirname "$out")" && env "$ART_ENV" NINFER_LOG_OPS=1 LD_LIBRARY_PATH="$CUDA13LIBS" "./$(basename "$out")" )
}
# 4a: load-plan (CPU-side binder; the dflash plan half needs the augmented
#     artifact)
if [ "$VARIANT" = augmented ]; then
  if run_artifact_test "$LOAD_TEST" native; then
    ok "load-plan PASS (1124 + 66 dflash objects, no full pool)"
    report "artifact load plan" "PASS"
  else
    bad "load-plan FAILED (augmented artifact — investigate)"; report "artifact load plan" "FAIL"
  fi
else
  out=$(run_artifact_test "$LOAD_TEST" native 2>&1)
  if [ $? -eq 0 ]; then
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
  if [ "$GPU_MODE" = docker ]; then
    run_engine() { run_artifact_test "$REAL_TEST" docker; }
  else
    run_engine() { run_artifact_test "$REAL_TEST" native; }
  fi
  if run_engine; then
    ok "engine dflash real PASS (golden + D4 negative + determinism + long-restore)"
    report "engine dflash real" "PASS"
  else
    bad "engine dflash real FAILED — investigate"; report "engine dflash real" "FAIL"
    sanitize_test "$REAL_TEST" 1800 -v "$(dirname "$ARTIFACT"):/artifacts" \
      -e "NINFER_QWEN3_8_27B_WEIGHTS=/artifacts/$(basename "$ARTIFACT")"
  fi
else
  skip "engine dflash real — needs the dflash-augmented artifact"
  report "engine dflash real" "NOT-RUN (needs augmented artifact)"
fi

# ---------- summary ----------
printf '\n================== SUMMARY ==================\n'
printf '%s\n' "${RESULTS[@]}"
if ls /tmp/dflash2-sanitize-*.log >/dev/null 2>&1; then
  printf '\n== auto-diagnosis (compute-sanitizer reports) ==\n'
  ls -1 /tmp/dflash2-sanitize-*.log
fi
printf '\n----------------------------------------------\n'
if [ "$VARIANT" = fleet ]; then
cat <<'NOTE'
The dflash-augmented artifact (1190 objects) is published at
  https://huggingface.co/phaseonx11/Qwen3.8-27B-nvfp4-DFlash2-NInfer
(qwen3_8_27b_nvfp4.ninfer, sha256 6cc7560ae3427d8fa87b75c17e41328116b71b068c4c4dc06137fb73b656f64e)
or built with the converter's --dflash-model (see docs §14). Re-run this
script with the augmented artifact path as the first argument.
NOTE
fi
fails=0
for r in "${RESULTS[@]}"; do case "$r" in *FAIL) fails=1;; esac; done
exit "$fails"