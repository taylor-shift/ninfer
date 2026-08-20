# NInfer on RunPod Serverless

Runs `ninfer-serve` as a RunPod **load balancing** Serverless endpoint, with the
`.ninfer` artifact supplied by RunPod's cached model feature instead of being baked
into the image.

There is no Python handler. A load balancing endpoint routes HTTP straight to the
worker's own server, and `ninfer-serve` already speaks OpenAI and Anthropic and
exposes an unauthenticated `/health` route, so it *is* the worker. The entrypoint
only resolves the cached artifact and picks a profile before `exec`ing the engine,
which stays PID 1 and receives signals directly.

## Why the model cache

Cached models are ordinary files on the worker at a predictable path:

```
/runpod-volume/huggingface-cache/hub/models--{org}--{name}/snapshots/{hash}/
```

Nothing about that path is framework-specific, so an `.ninfer` artifact works even
though the feature is designed around Hugging Face repositories. RunPod does not
bill for the download, and workers scheduled onto a host that already holds the
model skip it entirely.

The artifact repositories are public, so no Hugging Face token is required:

| Endpoint model field | Artifact |
|---|---|
| `neroued/Qwen3.8-27B-nvfp4-NInfer` | `qwen3_8_27b_nvfp4.ninfer` (20.02 GiB) |
| `neroued/Qwen3.6-27B-nvfp4-NInfer` | `qwen3_6_27b_nvfp4.ninfer` (17.07 GiB) |
| `neroued/Qwen3.6-35B-A3B-NInfer` | `qwen3_6_35b_a3b.ninfer` (21.22 GiB) |

Each endpoint may cache **one** model. Serve several artifacts with several
endpoints sharing this one image.

## Profiles

`NINFER_PROFILE` selects a deployment shape. Concurrency is fixed by the profile;
`--max-context` is then computed from the artifact size and the GPU's real VRAM.

| Profile | GPU | Concurrency | Resulting context |
|---|---|---:|---:|
| `standard` | RTX 5090, 32 GB | 4 | ~232k |
| `highconc` | RTX PRO 6000, 96 GB | 8 | 262,144 (native cap) |

`auto` (the default) picks `highconc` above 64 GiB of VRAM and `standard` otherwise,
so one image serves both endpoint types.

The two profiles differ in capability, not just speed. Both cards are 512-bit GDDR7
with comparable bandwidth, and decode is bandwidth-bound, so a single sequential
request runs at broadly similar speed on either. What 96 GB buys is a KV pool large
enough for eight concurrent requests to each hold a full-length private context,
which does not fit in 32 GB.

### Where the numbers come from

Shared KV costs 33 KiB per token for the 27B hybrid text stack under INT8 group-64
(16 full-attention layers, K and V, 4 KV heads of head_dim 256; GDN layers hold
fixed-size state and do not grow per token). BF16 KV costs 64 KiB per token.

The reserve above weights and KV covers workspace arenas, allocator slack, and the
per-batch CUDA Graph families, which are instantiated once for every batch size in
`[1,C]` and therefore grow with concurrency:

```
reserve_mib = 3151 + 371 * max_concurrency
```

That relation is fitted to two measured RTX 5090 configurations, C=2 at 255,000 and
C=4 at 232,000 context, and reproduces both to within 0.03%. Override it with
`NINFER_RESERVE_MIB`, or bypass sizing entirely with `NINFER_MAX_CONTEXT`.

## Build and push

```bash
docker build -f deploy/runpod/Dockerfile -t <user>/ninfer-runpod:v1.0 .
docker push <user>/ninfer-runpod:v1.0
```

Tag releases with a version rather than deploying `:latest`. A mutable tag makes
deployments unpredictable, complicates rollback, and interacts badly with RunPod's
image caching, since a cached host may hold a different image under the same name.

The build compiles for `sm_120a` only, so the image runs exclusively on compute
capability 12.0 devices. The entrypoint checks this and fails with a clear message
rather than letting the engine abort deep in startup.

## Templates

Two serverless templates exist for this image. Each pins the version tag, exposes
port 80, and carries the environment an endpoint needs; the only difference is
`NINFER_PROFILE`.

| Template | ID | Profile | Pair with GPU pool |
|---|---|---|---|
| `ninfer-standard` | `xcmdo0h3j5` | `standard` | `ADA_32_PRO` (RTX 5090) |
| `ninfer-highconc` | `f4sz9sgcwj` | `highconc` | `BLACKWELL_96` (RTX PRO 6000) |

Both set `containerDiskInGb: 30` and `volumeInGb: 0`. No volume is needed because
the cached model arrives on `/runpod-volume` independently, and the container disk
only holds the 4.4 GiB image plus logs. Both also carry a generated `NINFER_API_KEY`;
see [Authentication](#authentication).

Recreate or update them with `saveTemplate`, whose required fields are `name`,
`imageName`, `containerDiskInGb`, `volumeInGb`, `dockerArgs`, and `env`. Passing an
existing `id` updates that template in place instead of creating another. Leave
`dockerArgs` empty so the image's own `ENTRYPOINT` runs.

## Create the endpoint

The console is the simplest route, and the GraphQL API can do the same thing
unattended; see [Scripted creation](#scripted-creation) below. Note that the REST v2
API cannot: its `CreateEndpointRequest` has no model field, so an endpoint created
that way has no cached model and re-downloads the artifact on every cold start.

In the console, **Serverless → New Endpoint**, then select one of the templates
above rather than importing the image again:

1. **Endpoint type:** Load balancing. (Queue-based would serialize requests per
   worker unless you write an async handler, wasting the engine's own batching.)
2. **Template:** `ninfer-standard` or `ninfer-highconc`, which already carry the
   image, port, and environment.
3. **GPU:** the 32 GB tier (`ADA_32_PRO`) for `standard`, or the 96 GB tier
   (`BLACKWELL_96`) for `highconc`. Both contain only sm_120a parts, so any worker
   the scheduler picks can run this image.
4. **CUDA version:** select 13.1 and every newer version offered.
5. **Model:** the Hugging Face repo id from the table above. This is the field that
   enables caching; it must match `NINFER_MODEL`.
6. **Expose HTTP port:** `80`.
7. **Environment variables:** at minimum `HEALTH_CHECK_PATH=/health` and
   `RUNPOD_INIT_TIMEOUT=800`, plus any overrides from the table below.

`HEALTH_CHECK_PATH` matters. The load balancer polls `/ping` by default, which
`ninfer-serve` does not serve; without it workers never report healthy and return
502 after eight minutes.

Once an endpoint exists, the REST v2 API handles everything else — scaling, worker
counts, and logs.

### Endpoint settings by profile

| Console field | `standard` (burst offload) | `highconc` |
|---|---|---|
| Endpoint type | Load balancing | Load balancing |
| Image | `lroel/ninfer-runpod:v1.0` | same |
| GPU | RTX 5090 (32 GB) | RTX PRO 6000 (96 GB) |
| Model | `neroued/Qwen3.8-27B-nvfp4-NInfer` | same |
| Max workers | 2–3 | 2–3 |
| Active workers | 0 | 0 or 1 |
| Idle timeout | 300s | 300s |
| Expose HTTP port | 80 | 80 |

Environment variables, in both cases:

```
HEALTH_CHECK_PATH=/health
RUNPOD_INIT_TIMEOUT=800
NINFER_MODEL=neroued/Qwen3.8-27B-nvfp4-NInfer
NINFER_PROFILE=standard          # or: highconc
```

`NINFER_PROFILE` may be left at `auto`, which selects by VRAM; naming it explicitly
makes the endpoint's intent obvious and prevents a surprise if it ever schedules onto
a different card.

### Scripted creation

The cached model is reachable from the GraphQL API as `modelReferences` on
`EndpointInput`, so both profiles can be created without the console. The REST v2 API
has no equivalent field.

```bash
# Never commit this key; export it in the shell that runs the request.
read -rsp 'RunPod API key: ' RP_KEY; echo

curl -s -X POST https://api.runpod.io/graphql \
  -H "Authorization: Bearer $RP_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"query":"mutation($i:EndpointInput!){saveEndpoint(input:$i){id name}}",
       "variables":{"i":{
         "name":"ninfer-standard",
         "type":"LB",
         "templateId":"xcmdo0h3j5",
         "gpuIds":"ADA_32_PRO",
         "modelReferences":["neroued/Qwen3.8-27B-nvfp4-NInfer"],
         "workersMin":0, "workersMax":3, "idleTimeout":300,
         "minCudaVersion":"13.1"
       }}}'
```

Field notes, confirmed against the live API:

- `type` takes `LB` or `QB` here, not the REST spelling `LOAD_BALANCER`/`QUEUE`.
- `gpuIds` is required for a GPU endpoint and takes **pool IDs**, not display names.
  Use `ADA_32_PRO` for `standard` and `BLACKWELL_96` for `highconc`. Both pools
  contain only sm_120a parts, verified by probing pool membership: `ADA_32_PRO`
  holds the RTX 5090 alone, and `BLACKWELL_96` holds the three RTX PRO 6000
  Blackwell variants, which share one CUDA target. No exclusions are needed; to
  drop a member from a pool anyway, prefix its GPU id with `-`.
- `templateId` supplies the image, ports, and environment variables. Create the
  template first (console, or `saveTemplate`), since `EndpointInput` carries no
  `image` or `containerDiskInGb` field of its own.
- `modelReferences` is a list, but an endpoint caches one model at a time.

Verify afterwards, and watch `modelStatus` rather than assuming the cache is warm:

```bash
curl -s -X POST https://api.runpod.io/graphql \
  -H "Authorization: Bearer $RP_KEY" -H 'Content-Type: application/json' \
  -d '{"query":"{myself{endpoints{id name type modelReferences modelStatus}}}"}'
```

### Cold starts

Weight load plus CUDA Graph capture takes minutes for a 27B artifact. The model cache
removes download time but not load time, so this cost is paid on every cold start.

**Set `RUNPOD_INIT_TIMEOUT=800`.** RunPod marks a worker unhealthy when its cold start
exceeds seven minutes, which a 20 GiB artifact plus graph capture can approach; this
environment variable raises that ceiling (in seconds). Without it a slow host can put
a worker into a restart loop that never serves a request while still billing.

Two further settings matter more here than for a typical worker:

- **Idle timeout:** raise it to 300s or more so one warm worker absorbs a whole burst
  of subagent traffic instead of reloading between requests.
- **Active workers:** set to 1 if first-request latency matters. This eliminates cold
  starts but bills continuously, which erodes the reason to use Serverless at all.
  With bursty traffic, compare against simply running a Pod.

Graph capture cost also scales with concurrency, since one CUDA Graph family is
instantiated per batch size in `[1,C]`. The `highconc` profile therefore starts more
slowly than `standard`, independent of the artifact.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `NINFER_MODEL` | `neroued/Qwen3.8-27B-nvfp4-NInfer` | Cached Hugging Face repo id |
| `NINFER_PROFILE` | `auto` | `standard`, `highconc`, or `auto` |
| `NINFER_MAX_CONTEXT` | computed | Explicit context; skips sizing |
| `NINFER_MAX_CONCURRENCY` | from profile | Explicit concurrency |
| `NINFER_RESERVE_MIB` | `3151 + 371*C` | VRAM held back from the KV pool |
| `NINFER_KV_DTYPE` | `int8` | `int8` or `bf16` |
| `NINFER_KV_CAPACITY` | `auto` | Passed to `--kv-capacity` |
| `NINFER_SPEC` | `mtp` | `mtp`, `dflash`, or empty to disable |
| `NINFER_DRAFT_TOKENS` | `3` | Speculative draft window |
| `NINFER_LM_HEAD_DRAFT` | `1` | Load the optimized proposal head |
| `NINFER_PRESERVE_THINKING` | `1` | Retain closed-turn reasoning |
| `NINFER_API_KEY` | **required** | Key clients must send as `x-api-key` |
| `NINFER_ALLOW_ANONYMOUS` | `0` | Set `1` to serve without auth (public URL) |
| `NINFER_PENDING_TIMEOUT_MS` | `120000` | Admission wait before a request is rejected |
| `NINFER_VISION` | `0` | Set `1` to enable image and video input |
| `NINFER_CORS` | `0` | Set `1` to send CORS headers |
| `NINFER_ARTIFACT` | unset | File name, when a repo holds several artifacts |
| `NINFER_ARTIFACT_PATH` | unset | Absolute path; bypasses cache discovery |
| `NINFER_EXTRA_ARGS` | unset | Extra `ninfer-serve` flags |
| `PORT` | `80` | Listen port; must match the exposed HTTP port |

DFlash accepts concurrency only in `1..8`; the `highconc` profile sits exactly at
that boundary and MTP has no such limit.

## Authentication

A load balancing endpoint answers on a public URL, and every request it serves bills
GPU time, so the engine must require its own key. `NINFER_API_KEY` supplies it and
the entrypoint refuses to start without one; set `NINFER_ALLOW_ANONYMOUS=1` to serve
openly on purpose.

**Send the engine's key as `x-api-key`, not as a bearer token.** Runpod's proxy uses
the `Authorization` header for account-level auth, so a bearer token intended for the
engine can be consumed before it arrives. `ninfer-serve` accepts either header, and
`x-api-key` is the one that passes through cleanly. Both templates already carry a
generated key; read it back with:

```bash
curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" \
  https://rest.runpod.io/v1/templates |
  python3 -c 'import json,sys; [print(t["name"], {e["key"]:e["value"] for e in t["env"]}.get("NINFER_API_KEY")) for t in json.load(sys.stdin)]'
```

`/health` stays unauthenticated regardless, so the load balancer can keep polling it.
That route reveals nothing but liveness.

## Calling the endpoint

The engine's own routes are served directly, so any OpenAI-compatible client works
by pointing its base URL at the endpoint:

```bash
curl https://<ENDPOINT_ID>.api.runpod.ai/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H "x-api-key: $NINFER_API_KEY" \
  -d '{"model":"ninfer","messages":[{"role":"user","content":"hello"}]}'
```

`/v1/models`, `/v1/responses`, and `/v1/messages` are available on the same host.

For a client that only speaks bearer auth, Anthropic-style tooling maps cleanly:
`ANTHROPIC_API_KEY` becomes `x-api-key`, which is exactly the header to use here.
