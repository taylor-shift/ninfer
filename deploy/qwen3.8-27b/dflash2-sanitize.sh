#!/usr/bin/env bash
# One-shot compute-sanitizer diagnosis for the dflash_selector test (winbox).
# GPU MUST be free (>4 GiB in use => refuse). Extracts compute-sanitizer from the
# CUDA 13.1.2 image (nvidia-container-toolkit is missing, so it cannot run inside
# a docker container with GPU), then runs the test under it natively with the same
# CUDA 13.1 runtime libs the acceptance script uses.
set -uo pipefail
REPO=${NINFER_REPO:-/mnt/f/ninfer-fork}
BUILD="$REPO/build-wsl"
CONTAINER=ninfer-wsl
IMAGE=nvidia/cuda:13.1.2-devel-ubuntu24.04
CUDA13LIBS="$REPO/.cuda13-libs"
SANDIR="$REPO/.cuda13-sanitizer"

gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
if [ "${gpu:-0}" -gt 4096 ]; then echo "GPU busy (${gpu} MiB) — refusing."; exit 1; fi

SEL_TEST=$(docker exec "$CONTAINER" bash -c "cd /build && ctest -N 2>/dev/null" | sed -n 's/^ *Test *#[0-9]*: *//p' | grep -F 'dflash_selector_test' | head -1)
[ -n "$SEL_TEST" ] || { echo "selector test not registered"; exit 1; }
echo "test: $SEL_TEST"

# incremental rebuild of just the selector test target
docker exec "$CONTAINER" bash -c "cd /build && ninja -j32 'tests/$SEL_TEST' 2>&1 | tail -3"
BIN="$BUILD/tests/$SEL_TEST"
[ -x "$BIN" ] || { echo "binary missing after build"; exit 1; }

# extract the REAL compute-sanitizer ELF + whatever WSL cannot provide
# (ldd-driven, one-time). Note: /usr/local/cuda/bin/compute-sanitizer is a
# wrapper script that execs <install-root>/compute-sanitizer/compute-sanitizer —
# copying the wrapper alone breaks (it re-resolves its install root).
mkdir -p "$SANDIR"
if [ ! -x "$SANDIR/compute-sanitizer" ]; then
  echo "== extracting compute-sanitizer from the image =="
  docker run --rm --entrypoint bash -v "$SANDIR:/out" "$IMAGE" \
    -c 'real=$(find /usr/local/cuda -type f -name compute-sanitizer -not -path "*/bin/*" 2>/dev/null | head -1); [ -n "$real" ] || real=/usr/local/cuda/compute-sanitizer/compute-sanitizer; [ -f "$real" ] || { echo "real binary not found"; exit 1; }; cp -L "$real" /out/compute-sanitizer"'
fi
for lib in $(ldd "$SANDIR/compute-sanitizer" 2>/dev/null | awk '/not found/{print $1}'); do
  [ -f "$SANDIR/$lib" ] && continue
  echo "extracting $lib ..."
  docker run --rm --entrypoint bash -v "$SANDIR:/out" "$IMAGE" \
    -c "f=\$(find /usr/local/cuda /usr/lib /lib /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu -name '$lib' -not -name '*stubs*' 2>/dev/null | head -1); [ -n \"\$f\" ] && cp -L \"\$f\" /out/ || exit 1" \
    || echo "could not extract $lib"
done

echo "== plain run (baseline) =="
( cd "$BUILD/tests" && LD_LIBRARY_PATH="$CUDA13LIBS" ./"$SEL_TEST" ) 2>&1 | tail -4
rc=$?

echo "== compute-sanitizer memorycheck =="
( cd "$BUILD/tests" && LD_LIBRARY_PATH="$SANDIR:$CUDA13LIBS" "$SANDIR/compute-sanitizer" \
    --tool memorycheck --show-backtrace 1 ./"$SEL_TEST" ) 2>&1 | tail -50
echo "SANITIZE-DONE (baseline rc=$rc)"