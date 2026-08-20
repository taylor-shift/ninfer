# Multi-GPU deployment

One `ninfer-serve` per GPU behind least-connections balancers, packaged as a
single container image. Built and run on an 8×RTX 5090 host.

ninfer has **no tensor parallelism** (no NCCL, no all-reduce in the tree), so a
multi-GPU host is used by **replication**: every card runs a full engine with
its own weights and KV pool. Per-stream speed equals one card; aggregate
throughput and concurrency scale with the card count.

## Contents

| File | Role |
|---|---|
| `Dockerfile` | multi-stage: compiles `ninfer-serve`, ships a runtime-only image |
| `entrypoint.sh` | resolves the artifact, fans out one engine per GPU, runs balancers, drains on stop |
| `balancer.pl` | forking least-connections TCP balancer (not HTTP-aware, so SSE/keep-alive pass through) |
| `fetch-artifact.sh` | sha256-verified artifact download (Xet → aria2c → curl) |
| `refresh-dsh-endpoint.sh` | repoints a DSH provider at the instance's current host port |
| `RUNBOOK-vast.md` | Vast.ai runbook: sizing, rebuild steps, and the gotchas that cost real time |

## Build

From the **repository root** — the build context is the checkout, so the image
always matches the tree it was built from:

```bash
docker build -f deploy/multi-gpu/Dockerfile -t <user>/ninfer-pod-multi:vN .
```

`nvcc` is a **build-time dependency only**: ninfer compiles every kernel ahead
of time (no NVRTC / `cuModuleLoad` anywhere), so the runtime stage needs just
`libcudart` plus the host driver. Keeping the compiler out of the shipped image
costs nothing in capability and ~1.5 GB in size.

## Split fleet

By default the last two cards run `--vision` and the rest are text-only:

| Endpoint | Engines | Context |
|---|---|---|
| `:8000` | text | `NINFER_TEXT_CONTEXT` (262144) |
| `:8001` | vision | `NINFER_VISION_CONTEXT` (196608) |

`--vision` loads fixed GPU allocations (Vision runtime + media cache/live
buffers) *and* the engine's minimum runtime reservation scales with context.
Measured on a 32 GB RTX 5090 at C=2, int8 KV, MTP3:

| Config | Result |
|---|---|
| text 262144 | ✅ KV 274,624 tokens |
| vision 231936 | ❌ needs 11.70 GiB after weights, 11.59 available |
| vision 225280 | ❌ needs 11.46 GiB |
| vision 196608 | ✅ KV ~199k |

So vision costs ~65k context (25%) — worth paying on a couple of cards, not all
of them. Media input is separately capped by the Vision runtime at
`min(--max-context, 32768)` merged tokens, so the smaller vision context does
not reduce usable image capacity.

## Key environment variables

| Variable | Default | Notes |
|---|---|---|
| `NINFER_API_KEY` | — | **required** (or `NINFER_ALLOW_ANONYMOUS=1`) |
| `NINFER_ARTIFACT_SHA256` | — | verified before the engine loads; a partial download has the right *size* but wrong content |
| `NINFER_TEXT_ENGINES` | `GPUS-2` | how many cards stay text-only |
| `NINFER_MAX_CONCURRENCY` | `2` | per engine |
| `NINFER_START_STAGGER_S` | `5` | staggered starts; simultaneous 20 GiB reads thrash one volume |
| `NINFER_SHUTDOWN_GRACE_S` | `30` | drain window before engines are force-killed |
| `NINFER_VISION` | `1` | `0` for an all-text fleet |

## Lifecycle

The entrypoint **supervises** rather than `exec`s the balancer, so a container
stop reaches `shutdown()`, which TERMs every engine and waits for them to
release VRAM (measured ~2 s per card for 30 GiB) before exiting. Engines are
also gated on the target GPU actually being free — checking only the port lets a
half-dead engine make a slot look occupied and strand a card.

## Related fix

`--device N` requires `fix(core): bind the engine's device on every CUDA entry
thread`. Without it, `cudaSetDevice`'s per-thread scope means HTTP worker
threads launch on device 0, and every engine off GPU 0 dies during warmup with
`cudaErrorInvalidValue` at `gqa_attention_prefill.cu:58`.
