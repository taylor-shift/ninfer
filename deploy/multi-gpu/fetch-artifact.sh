#!/usr/bin/env bash
#
# Fetch the .ninfer artifact fast.
#
# The base image downloads 21.5 GB with a single `curl -fL -C -` stream, which
# measured ~1 MB/s from a RunPod host — over 40 hours. The file is Xet-backed on
# Hugging Face and served from CloudFront, which parallelises well, so the same
# transfer over many connections is dramatically faster.
#
# Order of attempts:
#   1. hf_transfer via huggingface_hub  (Rust multipart; best on Xet-backed files,
#      and the only path that uses HF_TOKEN for higher rate limits)
#   2. aria2c with N connections        (no Python needed; resumes cleanly)
#   3. curl single stream               (last resort; what the base image does)
#
# Usage: ninfer-fetch-artifact <repo_id> <filename> <destination>
#
# Env:
#   HF_TOKEN                      raises rate limits; not required for public repos
#   NINFER_DOWNLOAD_CONNECTIONS   parallel connections for aria2c (default 16)

set -euo pipefail

log() { printf '[fetch-artifact] %s\n' "$*" >&2; }
fail() { log "ERROR: $*"; exit 1; }

repo="${1:?repo id required}"
file="${2:?filename required}"
dest="${3:?destination required}"

conns="${NINFER_DOWNLOAD_CONNECTIONS:-16}"
url="https://huggingface.co/${repo}/resolve/main/${file}"
tmp="${dest}.incomplete"
destdir=$(dirname "$dest")
mkdir -p "$destdir"

# A complete artifact already present is left alone. The base image writes to a
# .incomplete temp and renames, so a finished file is always whole.
if [[ -f "$dest" ]]; then
    log "artifact already present: ${dest} ($(stat -Lc %s "$dest" | awk '{printf "%.1f GiB", $1/1073741824}'))"
    exit 0
fi

# Expected size, used to verify the transfer actually completed. HEAD follows the
# redirect to the CDN, where Content-Length is authoritative.
expected=$(curl -sIL ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} "$url" 2>/dev/null |
    awk 'BEGIN{IGNORECASE=1} /^content-length:/ {v=$2} END{gsub(/\r/,"",v); print v}')
[[ "$expected" =~ ^[0-9]+$ ]] || expected=""
[[ -n "$expected" ]] && log "expected size: $(( expected / 1024 / 1024 )) MiB"

verify_and_finish() {
    [[ -f "$tmp" ]] || return 1
    local got; got=$(stat -Lc %s "$tmp")
    if [[ -n "$expected" && "$got" != "$expected" ]]; then
        log "size mismatch: got ${got}, expected ${expected}"
        return 1
    fi
    # Size alone does NOT prove integrity: a parallel-range download of a
    # Xet-backed file yields the exact right size with wrong bytes. When a known
    # digest is supplied, checking it here is the only thing standing between a
    # corrupt artifact and a confusing engine crash at layer 0.
    if [[ -n "${NINFER_ARTIFACT_SHA256:-}" ]]; then
        log "verifying sha256 (this reads the whole file)..."
        local actual; actual=$(sha256sum "$tmp" | cut -d" " -f1)
        if [[ "$actual" != "$NINFER_ARTIFACT_SHA256" ]]; then
            log "CHECKSUM MISMATCH: got ${actual}"
            log "                   want ${NINFER_ARTIFACT_SHA256}"
            return 1
        fi
        log "sha256 verified"
    fi
    mv -f "$tmp" "$dest"
    log "download complete: ${dest} ($(( got / 1024 / 1024 )) MiB)"
    return 0
}

start=$(date +%s)
report_rate() {
    local end=$(( $(date +%s) - start ))
    (( end > 0 )) || end=1
    if [[ -f "$dest" ]]; then
        local mib=$(( $(stat -Lc %s "$dest") / 1024 / 1024 ))
        log "took ${end}s (~$(( mib / end )) MiB/s)"
    fi
}

# --- 1. hf_transfer ---------------------------------------------------------

if command -v python3 >/dev/null 2>&1 && python3 -c "import huggingface_hub" 2>/dev/null; then
    log "attempt 1/3: huggingface_hub + hf_transfer"
    if HF_HUB_ENABLE_HF_TRANSFER=1 python3 - "$repo" "$file" "$destdir" <<'PY'
import os, shutil, sys
from huggingface_hub import hf_hub_download

repo, filename, destdir = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    path = hf_hub_download(
        repo_id=repo,
        filename=filename,
        token=os.environ.get("HF_TOKEN") or None,
        cache_dir=os.path.join(destdir, ".hf-cache"),
    )
except Exception as exc:                      # fall through to aria2c
    print(f"hf_transfer failed: {exc}", file=sys.stderr)
    sys.exit(1)

target = os.path.join(destdir, os.path.basename(filename) + ".incomplete")
# hf_hub_download returns a symlink into the blob store; move the real blob so
# the cache directory can be removed afterwards without breaking the artifact.
real = os.path.realpath(path)
shutil.move(real, target)
print(target)
PY
    then
        if verify_and_finish; then
            rm -rf "${destdir}/.hf-cache"
            report_rate
            exit 0
        fi
        log "hf_transfer output failed verification; falling back"
    else
        log "hf_transfer unavailable or failed; falling back"
    fi
    rm -rf "${destdir}/.hf-cache"
fi

# --- 2. aria2c --------------------------------------------------------------

# WARNING: aria2c is only safe for plain LFS files. On a Xet-backed repo the
# resolve URL redirects to a per-request signed CDN reconstruction, and 16
# parallel range requests do NOT compose into the original file: the result is
# byte-for-byte the right SIZE but wrong CONTENT, and a different checksum on
# every attempt. That produced three distinct corrupt artifacts before it was
# caught, each failing at load with "divisor must be finite and positive".
# Only use it when the download is verified against a known checksum, or when
# the repo is not Xet-backed.
if [[ "${NINFER_ALLOW_ARIA2:-0}" == "1" ]] && command -v aria2c >/dev/null 2>&1; then
    log "attempt 2/3: aria2c with ${conns} connections (NINFER_ALLOW_ARIA2=1)"
    # --auto-file-renaming=false + --continue makes a retry resume rather than
    # writing a second copy, which matters on a 50 GB container disk.
    if aria2c \
        --max-connection-per-server="$conns" \
        --split="$conns" \
        --min-split-size=16M \
        --continue=true \
        --auto-file-renaming=false \
        --allow-overwrite=true \
        --max-tries=5 \
        --retry-wait=3 \
        --console-log-level=warn \
        --summary-interval=30 \
        ${HF_TOKEN:+--header="Authorization: Bearer ${HF_TOKEN}"} \
        --dir="$destdir" \
        --out="$(basename "$tmp")" \
        "$url"
    then
        if verify_and_finish; then
            report_rate
            exit 0
        fi
        log "aria2c output failed verification; falling back"
    else
        log "aria2c failed; falling back"
    fi
fi

# --- 3. curl ----------------------------------------------------------------

log "attempt 3/3: curl single stream (slow path)"
curl -fL --retry 5 --retry-delay 3 -C - --progress-bar \
    ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} \
    -o "$tmp" "$url" || fail "all download methods failed (partial kept at $tmp)"

verify_and_finish || fail "downloaded file failed verification"
report_rate
