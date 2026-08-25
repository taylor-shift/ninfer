#!/usr/bin/env bash
# dflash2 repro v2 runner (winbox): compiles dflash2_repro_lib.cpp with the
# TEST EXECUTABLE'S EXACT link line (same archives, dynamic + static cudart,
# stubs/libcuda.so), then runs it natively. HARD GPU-FREE GUARD.
set -uo pipefail
REPO=${NINFER_REPO:-/mnt/f/ninfer-fork}
BUILD="$REPO/build-wsl"
CONTAINER=ninfer-wsl
CUDA13LIBS="$REPO/.cuda13-libs"

gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
if [ "${gpu:-0}" -gt 4096 ]; then echo "GPU busy (${gpu} MiB) — refusing."; exit 1; fi

echo "== build (container; exact test link line) =="
docker exec "$CONTAINER" bash -c "cd /build && /usr/bin/c++ -O3 -DNDEBUG -std=gnu++20 -I/src/include -I/src/src -I/src/third_party -I/src/third_party/utf8proc -isystem /usr/local/cuda/targets/x86_64-linux/include /src/deploy/qwen3.8-27b/dflash2_repro_lib.cpp -o /build/dflash2_repro_lib -L/usr/local/cuda/targets/x86_64-linux/lib -Wl,-rpath,/usr/local/cuda-13.1/targets/x86_64-linux/lib src/libninfer_ops.a src/libninfer_nvfp4_tma.a src/libninfer_core.a -ldl /usr/local/cuda-13.1/targets/x86_64-linux/lib/libcudart.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so -lcudadevrt -lcudart_static -lrt -lpthread -ldl" || exit 1
[ -x "$BUILD/dflash2_repro_lib" ] || { echo "binary missing"; exit 1; }

echo "== run (native) =="
( cd "$BUILD" && LD_LIBRARY_PATH="$CUDA13LIBS" ./dflash2_repro_lib )
echo "REPRO-LIB-DONE rc=$?"