#!/usr/bin/env bash
# dflash2 minimal repro ladder runner (winbox). Builds dflash2_minimal_repro.cu
# inside the ninfer-wsl container (nvcc 13.1, sm_120a, real repo headers), then
# runs it natively with the extracted CUDA 13.1 libs. HARD GPU-FREE GUARD.
set -uo pipefail
REPO=${NINFER_REPO:-/mnt/f/ninfer-fork}
BUILD="$REPO/build-wsl"
CONTAINER=ninfer-wsl
CUDA13LIBS="$REPO/.cuda13-libs"
REPRO_SRC="$REPO/deploy/qwen3.8-27b/dflash2_minimal_repro.cu"

gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
if [ "${gpu:-0}" -gt 4096 ]; then echo "GPU busy (${gpu} MiB) — refusing."; exit 1; fi
[ -f "$REPRO_SRC" ] || { echo "missing $REPRO_SRC"; exit 1; }

echo "== build (container, nvcc 13.1 sm_120a) =="
docker exec "$CONTAINER" bash -c "cd /build && /usr/local/cuda/bin/nvcc -O3 -std=c++20 -arch=sm_120a -lineinfo -diag-suppress 550 -I/src/include -I/src/src /src/deploy/qwen3.8-27b/dflash2_minimal_repro.cu -o /build/dflash2_minimal_repro" || exit 1
[ -x "$BUILD/dflash2_minimal_repro" ] || { echo "binary missing"; exit 1; }

echo "== run (native) =="
( cd "$BUILD" && LD_LIBRARY_PATH="$CUDA13LIBS" ./dflash2_minimal_repro )
echo "REPRO-DONE rc=$?"