# ninfer-multi — one ninfer-serve per GPU, behind one port

Runs N independent `ninfer-serve` replicas (one per card) on a multi-GPU host and
fronts them with a least-connections TCP balancer, so callers see a single
OpenAI/Anthropic-compatible endpoint.

## Why replicas and not tensor parallelism

ninfer has **no TP path** — no NCCL, no all-reduce anywhere in the tree. So a
multi-GPU host is used by **replication**: each card runs a full engine with its
own copy of the artifact and its own KV pool.

For concurrent agent traffic this is the better shape anyway:

- no interconnect cost, no cross-card sync in the decode loop
- a long 200k-context generation occupies one card, not all of them
- a crashed replica removes 1/N of capacity instead of the whole server
- aggregate throughput scales linearly with cards

The tradeoff: **one request never goes faster than one card**. Per-stream tps is
the single-GPU number; the win is in aggregate throughput and concurrency.

## Requirements

- GPUs at **sm_120a / compute capability 12.0** (RTX 5090, RTX PRO 6000 Blackwell).
  The launcher checks every card and refuses to start on anything else.
- **CUDA 13.1+** — ninfer's CMake hard floor (`CMAKE_CUDA_COMPILER_VERSION < 13.1`
  is a fatal error) and the reason a 13.1 host matters: CUDA forward compat does
  not work on GeForce, so the 13.1 userland needs a native 13.1+ driver.
- `ninfer-serve` on `PATH`, and the `.ninfer` artifact on local disk.

## Usage

```bash
export NINFER_API_KEY='<key>'
./ninfer-multi.sh /workspace/qwen3_8_27b_nvfp4.ninfer
```

On a 7×5090 host that starts 7 engines on `127.0.0.1:18080..18086` (each pinned
with `--device i`) and the balancer on `:8000`. Point any OpenAI client at
`http://<host>:8000/v1`.

### Knobs

| Env | Default | Notes |
|---|---|---|
| `GPUS` | autodetect | number of replicas |
| `PORT` | 8000 | public balancer port |
| `ENGINE_BASE_PORT` | 18080 | first engine port |
| `NINFER_MAX_CONCURRENCY` | 4 | **per engine** — 7 cards × 4 = 28 total streams |
| `NINFER_KV_DTYPE` | int8 | `bf16` doubles KV cost per token |
| `NINFER_MAX_CONTEXT` | autosized | per card, from free VRAM |
| `NINFER_SPEC` / `NINFER_DRAFT_TOKENS` | mtp / 3 | MTP3 measured best |
| `NINFER_EXTRA_ARGS` | — | appended to every engine |
| `READY_TIMEOUT_S` | 900 | per-engine readiness budget |

Context is autosized per card with the same model as `deploy/runpod/entrypoint.sh`
(16 full-attention layers, 4 KV heads, head_dim 256; int8 group-64 ≈ 33 KiB/token,
bf16 ≈ 64 KiB/token; reserve `3151 + 371·C` MiB for workspace and per-batch CUDA
graph families).

## Operations

- Engine logs: `/tmp/ninfer-multi/engine-<i>.log` (override with `NINFER_LOG_DIR`).
- Startup is gated: the launcher polls every replica's `/health` and aborts with
  the failing engine's log path if one dies during load. Nothing serves until all
  replicas are ready.
- Ctrl-C / SIGTERM tears down the balancer and every engine.
- A replica that refuses a connection returns **502** to that caller; the others
  keep serving.

## Load balancer design

`balancer.pl` is a forking **TCP** relay, not an HTTP proxy — it never parses or
buffers the payload, so SSE streaming, chunked encoding, keep-alive, and both the
OpenAI and Anthropic surfaces pass through untouched. Each accepted connection is
routed to the replica with the fewest live sessions (ties → lower port), the
parent tracks session counts and decrements on `SIGCHLD`.

Round-robin was rejected deliberately: request lifetimes vary by orders of
magnitude here, so blind rotation buries one card while another idles.

Verified locally against mock engines: 9 concurrent streaming requests
distributed exactly 3/3/3 across 3 replicas, chunk boundaries preserved.

Consequence of TCP-level routing: a keep-alive connection stays pinned to its
replica for its lifetime — correct for streaming clients, but a client that
funnels *all* traffic down one long-lived connection will pin itself to one card.
Clients that open a connection per request (or a small pool) spread correctly.

## Sizing note for the 7×5090 host

- 32 GB per card, artifact ~20 GiB → autosizer lands near the measured
  configurations (C=4 at ~232k context, int8 KV).
- Per-stream throughput is the single-5090 number (~136 tps measured locally);
  aggregate across 7 cards ≈ **950 tps** at **28 concurrent streams**.
- If you want more concurrency per card instead of more context, lower
  `NINFER_MAX_CONTEXT` and raise `NINFER_MAX_CONCURRENCY` — KV pool is the
  binding constraint.
