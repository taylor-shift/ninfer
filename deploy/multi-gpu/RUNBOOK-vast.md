# Runbook — 8×5090 ninfer fleet on Vast.ai

Split fleet: **6 text engines @ 262,144 ctx** + **2 vision engines @ 196,608 ctx**,
each behind its own least-connections balancer.

| Endpoint | Engines | Context | Use |
|---|---|---|---|
| `:8000` | 18080–18085 (GPU 0–5) | **262,144** (full native) | all text work |
| `:8001` | 18086–18087 (GPU 6–7) | 196,608 | anything with images/video |

Both speak OpenAI **and** Anthropic surfaces; auth via `x-api-key` or
`Authorization: Bearer`. `/health` is unauthenticated.

## Why split rather than vision everywhere

`--vision` loads fixed GPU allocations (Vision runtime + 1 GiB media-cache +
2 GiB media-live) **and** the engine's minimum runtime reservation scales with
context. Measured on a 32 GB RTX 5090 at C=2, int8 KV, MTP3:

| Config | Result |
|---|---|
| text, C=2, 262144 | ✅ KV 274,624 tokens |
| text, C=4, 231936 | ✅ KV 252,032 tokens |
| **vision, C=2, 231936** | ❌ needs 11.70 GiB after weights, 11.59 available |
| **vision, C=2, 225280** | ❌ needs 11.46 GiB (reservation tracks context) |
| **vision, C=2, 196608** | ✅ KV ~199k, 9.79 GiB runtime + media buffers |

**Vision costs ~65k context (25%).** Most work is text, so only 2 cards pay it.

Media input is separately capped by the Vision runtime at
`min(--max-context, 32768)` merged tokens (131,072 raw patches), so a vision
engine's smaller context does not reduce usable image capacity.

## Current deployment

- Instance **48240339**, Vast, Brazil, 8×RTX 5090, **$3.73/hr**, machine 141452
- Driver **595.71.05 = CUDA 13.2** (ninfer needs ≥13.1)
- Image `lroel/ninfer-pod-multi:v4`, disk 80 GB, artifact at `/workspace/qwen3_8_27b_nvfp4.ninfer`
- SSH: `vastai show instance 48240339 --raw` → `ports['22/tcp']`

### Port stability (verified 2026-08-20)

Vast host ports are **stable across stop/start** — this instance kept
`22/tcp → 40215` and `8000/tcp → 40461` through a full stop and start. They are
allocated from the machine's own range (here 40100–40587) and only change on a
**destroy/recreate**, which lands on a different machine. There is no flag to
pin a host port (same limitation on RunPod, where ports *do* rotate on restart).

After a rebuild, repoint DSH with:

```bash
/code/llm-cluster/ninfer-multi/refresh-dsh-endpoint.sh [INSTANCE_ID]
```

It reads the live mapping, health-checks it, and rewrites only the
`ninfer-8x5090` provider's `baseURL` in `~/.dsh/settings.yaml`.

### DSH provider

```yaml
ninfer-8x5090:
  api: openai-completions
  baseURL: http://<ip>:<host-port-for-8000>/v1
  apiKeyEnv: NINFER_VAST_API_KEY      # value lives in ~/.dsh/.credentials.yaml
  models:
    - id: qwen3.8-27b
      contextWindow: 262144
      maxTokens: 32768
```

Use with `provider: ninfer-8x5090`, `model: qwen3.8-27b` (verified via a live
workflow subagent). Vision needs `-p 8001:8001` published at instance-create
time; the commented provider block in `settings.yaml` is ready for it.

## Standing up the fleet by hand

Scripts live on the instance at `/root/`:

```bash
/root/launch-fleet.sh     # engines (reads NINFER_* env)
/root/go-fleet.sh         # wrapper exporting the validated env, then execs the above
/root/bal-text.sh         # :8000 -> 18080..18085
/root/bal-vision.sh       # :8001 -> 18086,18087
```

```bash
# engines (~2 min for all 8)
nohup setsid /root/go-fleet.sh > /var/log/ninfer/fleet.log 2>&1 < /dev/null &

# balancers
nohup setsid /root/bal-text.sh   > /var/log/ninfer/bal-text.log   2>&1 < /dev/null &
nohup setsid /root/bal-vision.sh > /var/log/ninfer/bal-vision.log 2>&1 < /dev/null &

# verify
for p in $(seq 18080 18087); do curl -s -o /dev/null -w "$p:%{http_code} " 127.0.0.1:$p/health; done
curl -s -o /dev/null -w "\ntext:%{http_code} " 127.0.0.1:8000/health
curl -s -o /dev/null -w "vision:%{http_code}\n" 127.0.0.1:8001/health
```

## Hard-won gotchas

1. **`--device N` is broken on multi-GPU hosts.** Engines 1–7 crash warmup with
   `cudaErrorInvalidValue` at `gqa_attention_prefill.cu:58` — a warmup path
   touches device 0's context. **Always** `CUDA_VISIBLE_DEVICES=N ... --device 0`.
   (Real ninfer bug; worth fixing upstream in the fork.)
2. **Processes launched over Vast SSH die with the session.** Use
   `nohup setsid <script> > log 2>&1 < /dev/null &`. Backgrounding a *command*
   is not enough — it must be a script.
3. **Long compound SSH commands silently truncate on this host.** Heredocs
   written through `ssh '...'` frequently never land. `scp` the script instead,
   then run it. This cost real time; do not fight it.
4. **Never kill PID 1** (the container entrypoint) — it takes the container down
   and rotates the SSH port. `kill -STOP` to freeze it instead.
5. **`--draft-tokens` / `--lm-head-draft` require `--spec mtp|dflash`.**
   Passing them without `--spec` is a hard startup error.
6. **Benchmarks must run on the host.** Through a single SSH tunnel the client
   serializes streams: N-way per-stream tps is meaningless (wall times survive).
   `scp llm_bench.py` to the box and run it there.
7. Vast **instance disk survives stop/start** (unlike RunPod's container disk);
   only *destroy* wipes it. No 8×5090 CUDA≥13.1 host currently offers Vast
   volumes (checked all 64 volume-offering machines — zero overlap).

## Rebuilding from scratch

```bash
vastai search offers 'num_gpus>=8 gpu_name=RTX_5090 cuda_vers>=13.1 rentable=true' -o dph
vastai create instance <OFFER_ID> --image lroel/ninfer-pod-multi:v4 --disk 80 --ssh --direct \
  --env '-p 8000:8000 -p 8001:8001 -e NINFER_API_KEY=<key> -e NINFER_MODEL=neroued/Qwen3.8-27B-nvfp4-NInfer -e NINFER_ARTIFACT=qwen3_8_27b_nvfp4.ninfer -e NINFER_VOLUME_PATH=/workspace -e NINFER_ARTIFACT_SHA256=bb3360522a06e136e0367f5703414d26272b7285c8a6ab6194135c17dbd81b32' \
  --onstart-cmd '/usr/local/bin/ninfer-multi-entrypoint'
vastai attach ssh <INSTANCE_ID> "$(cat ~/.ssh/id_ed25519.pub)"
```

Artifact download is sha256-verified (`bb3360522a06e136…`); it lands in ~4 min at
~91 MB/s on a well-connected host. **Never trust size alone** — a partial
`aria2c` file has the exact right byte count with wrong content, which surfaces
later as `text/layers/0/mlp/gate_up: divisor must be finite and positive`.

## Measured performance (on-host, single engine)

- prefill **4,000–5,400 tok/s** @105k ctx
- decode **160–167 tok/s** per stream
- MTP: 3.27–3.48 tokens/round, **75–83% acceptance**
- model load **~3.8 s** from local NVMe
