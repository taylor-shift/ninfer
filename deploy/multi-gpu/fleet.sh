#!/usr/bin/env bash
#
# fleet.sh — one-command stop/start for the qwen3.8-27b worker pool.
#
#   fleet.sh status              fleet state, health, hourly cost
#   fleet.sh up  [vast|runpod]   start stopped fleet(s), wait for /health,
#                                sync DSH routes + endpoint file (default: both)
#   fleet.sh down [vast|runpod]  stop running fleet(s); GPU billing stops
#   fleet.sh register            one-time: add the ninfer-7x5090 DSH provider
#   fleet.sh pool start|stop|status   LiteLLM pool proxy on :8800 (least-busy
#                                     across both fleets + local 5090; config
#                                     in litellm-config.yaml next to this script)
#
# Managed here:
#   vast:   8x5090 instance (VAST_INSTANCE; state file vast-instance-id,
#           default 48240339)  ($3.75/hr)
#           host ports + instance disk are STABLE across stop/start; the
#           onstart entrypoint relaunches engines + balancers automatically.
#   runpod: 7x5090 pod ee29h260cf8cwh ($6.93/hr)
#           proxy URL https://<podid>-<port>.proxy.runpod.net is stable across
#           stop/start (pinned to pod id); container disk is wiped on stop —
#           the model artifact lives on the attached 30 GB volume.
#   pool:   LiteLLM router (litellm[proxy] venv at /opt/litellm) on the always-
#           on DSH host. Balances text model "qwen3.8-27b" across
#           vast + runpod + local and "qwen3.8-27b-vision" across the vision
#           balancers. Clients send the NINFER_POOL_API_KEY master key.
#           num_workers MUST stay 1 (see litellm-config.yaml) or the router
#           state (least-busy + cooldown) fragments per process.
#
# NOT managed (always on, by design):
#   local winbox 5090        http://192.168.8.15:8000/v1
#   RunPod TP2 pod           bdawwwyiyc1ih3 (DSH agent-default model)
#
# Env overrides: VAST_INSTANCE, RUNPOD_POD, DSH_SETTINGS, ENDPOINTS_FILE,
# WAIT_SECONDS (health-wait budget for `up`, default 600).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# VAST_INSTANCE: env override > state file (written after a recreate) > default.
VAST_STATE_FILE="${VAST_STATE_FILE:-$SCRIPT_DIR/vast-instance-id}"
VAST_INSTANCE="${VAST_INSTANCE:-$(cat "$VAST_STATE_FILE" 2>/dev/null | tr -d '[:space:]' || true)}"
[[ -n "$VAST_INSTANCE" ]] || VAST_INSTANCE=48240339
RUNPOD_POD="${RUNPOD_POD:-ee29h260cf8cwh}"
RUNPOD_PROXY="https://${RUNPOD_POD}"      # + -<port>.proxy.runpod.net
DSH_SETTINGS="${DSH_SETTINGS:-/root/.dsh/settings.yaml}"
CREDENTIALS="${CREDENTIALS:-/root/.dsh/.credentials.yaml}"
ENDPOINTS_FILE="${ENDPOINTS_FILE:-/code/llm-cluster/fleet-endpoints.json}"
WAIT_SECONDS="${WAIT_SECONDS:-600}"
HEALTH_INTERVAL=15
LOCAL_HEALTH="http://192.168.8.15:8000/health"
COST_VAST=3.75
COST_RUNPOD=6.93
# Pool proxy (LiteLLM) on the always-on DSH host.
LITELLM_BIN="${LITELLM_BIN:-/opt/litellm/bin/litellm}"
POOL_CONFIG="${POOL_CONFIG:-$SCRIPT_DIR/litellm-config.yaml}"
POOL_PORT="${POOL_PORT:-8800}"
POOL_LOG="${POOL_LOG:-/code/llm-cluster/litellm.log}"

command -v vastai >/dev/null || { echo "vastai CLI not found" >&2; exit 1; }
command -v runpodctl >/dev/null || { echo "runpodctl not found" >&2; exit 1; }
[[ -n "${RUNPOD_API_KEY:-}" ]] || { set -a; . /root/.config/runpod/env 2>/dev/null || true; set +a; }

# ---------------------------------------------------------------- helpers ---

health_code() { curl -s -m 10 -o /dev/null -w '%{http_code}' "$1" || true; }

# Prints: <ip> <hostport-8000> <hostport-8001> (fields blank when unknown).
vast_mapping() {
  vastai show instance "$VAST_INSTANCE" --raw 2>/dev/null | python3 -c '
import json, sys
d = json.load(sys.stdin)
p = d.get("ports") or {}
def hp(svc):
    m = p.get(svc)
    return str(m[0]["HostPort"]) if m else ""
print(d.get("public_ipaddr") or "", hp("8000/tcp"), hp("8001/tcp"))
' 2>/dev/null || echo "- - -"
}
read -r VAST_IP VAST_P8000 VAST_P8001 < <(vast_mapping) || true
VAST_URL=""
[[ "$VAST_IP" != "-" && -n "$VAST_P8000" ]] && VAST_URL="http://${VAST_IP}:${VAST_P8000}/v1"
VAST_VISION_URL=""
[[ "$VAST_IP" != "-" && -n "$VAST_P8001" ]] && VAST_VISION_URL="http://${VAST_IP}:${VAST_P8001}/v1"

# RunPod endpoint: proxy URL is constant for the life of the pod.
RP_URL="${RUNPOD_PROXY}-8000.proxy.runpod.net/v1"

vast_state() {
  vastai show instance "$VAST_INSTANCE" --raw 2>/dev/null | python3 -c '
import json, sys
d = json.load(sys.stdin)
print(d.get("cur_state") or d.get("actual_status") or "unknown")' 2>/dev/null || echo unknown
}
runpod_state() {
  runpodctl pods list -o json 2>/dev/null | python3 -c '
import json, sys
for p in json.load(sys.stdin):
    if p.get("id") == sys.argv[1]:
        print(p.get("runtimeStatus", "unknown")); break
else:
    print("not-found")' "$RUNPOD_POD" 2>/dev/null || echo unknown
}

wait_health() {
  local label=$1 url=$2
  local deadline code
  deadline=$(( $(date +%s) + WAIT_SECONDS ))
  echo "waiting for ${label} health at ${url%/v1}/health (budget ${WAIT_SECONDS}s)..."
  while :; do
    code="$(health_code "${url%/v1}/health")"
    if [[ "$code" == "200" ]]; then echo "  ${label} healthy"; return 0; fi
    if (( $(date +%s) >= deadline )); then
      echo "  ${label} not healthy after ${WAIT_SECONDS}s (last HTTP ${code:-000})." >&2
      if [[ "$label" == vast* ]]; then
        echo "  diagnose: vastai logs $VAST_INSTANCE" >&2
      else
        echo "  diagnose: runpodctl logs $RUNPOD_POD" >&2
      fi
      return 1
    fi
    sleep "$HEALTH_INTERVAL"
  done
}

# Line-oriented rewrite of one provider's baseURL in settings.yaml (same
# technique as refresh-dsh-endpoint.sh: comments/formatting survive).
rewrite_baseurl() {
  local provider=$1 url=$2
  python3 - "$DSH_SETTINGS" "$provider" "$url" <<'PY'
import re, sys
path, provider, url = sys.argv[1:4]
src = open(path).read()
lines = src.splitlines(keepends=True)
out, in_block, done = [], False, False
for line in lines:
    if re.match(rf'^\s*{re.escape(provider)}:\s*$', line):
        in_block = True
    elif in_block and re.match(r'^\s{0,6}\S', line) and not re.match(r'^\s{6,}', line):
        in_block = False
    if in_block and not done and re.match(r'^\s*baseURL:', line):
        indent = line[:len(line) - len(line.lstrip())]
        line = f'{indent}baseURL: {url}\n'
        done = True
    out.append(line)
if not done:
    sys.exit(f'provider {provider} or its baseURL not found in {path}')
open(path, 'w').write(''.join(out))
print(f'updated {path}: {provider}.baseURL = {url}')
PY
}

write_endpoints() {
  local ts
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  python3 - "$ENDPOINTS_FILE" "$VAST_URL" "$VAST_VISION_URL" "$RP_URL" "$ts" <<'PY'
import json, sys
path, vast_text, vast_vision, rp_text, ts = sys.argv[1:6]
doc = {
    "updated_at": ts,
    "model": "qwen3.8-27b (NVFP4, ninfer)",
    "endpoints": {
        "vast_8x5090_text":     {"url": vast_text, "contextWindow": 262144, "cards": 6},
        "vast_8x5090_vision":   {"url": vast_vision, "contextWindow": 196608, "cards": 2},
        "runpod_7x5090_text":   {"url": rp_text, "contextWindow": 262144},
        "runpod_7x5090_vision": {"url": rp_text.replace("-8000.", "-8001."), "contextWindow": 196608},
        "local_5090":           {"url": "http://192.168.8.15:8000/v1", "contextWindow": 262144},
    },
}
json.dump(doc, open(path, "w"), indent=2)
print(f"wrote {path}")
PY
}

# ---------------------------------------------------------------- status ---

cmd_status() {
  local vs rs vc rc lcode
  vs="$(vast_state)"; rs="$(runpod_state)"
  if [[ -n "$VAST_URL" ]]; then
    vc="$(health_code "${VAST_URL%/v1}/health")"
  else
    vc="-"
  fi
  rc="$(health_code "${RUNPOD_PROXY}-8000.proxy.runpod.net/health")"
  lcode="$(health_code "$LOCAL_HEALTH")"
  printf '%-15s %-10s %-14s %s\n' "fleet" "state" "health(:8000)" "$/hr"
  printf '%-15s %-10s %-14s %s\n' "vast 8x5090"   "$vs" "${vc:-000}" "$COST_VAST"
  printf '%-15s %-10s %-14s %s\n' "runpod 7x5090" "$rs" "${rc:-000}" "$COST_RUNPOD"
  printf '%-15s %-10s %-14s %s\n' "local 5090"    "always-on" "${lcode:-000}" "0.00"
  local pcode
  pcode="$(health_code "http://127.0.0.1:${POOL_PORT}/health/liveliness")"
  if [[ "$pcode" == "200" ]]; then
    echo "pool proxy :8800 UP (least-busy across vast+runpod+local; vision via qwen3.8-27b-vision)"
  else
    echo "pool proxy :8800 DOWN (start: fleet.sh pool start)"
  fi
  echo "DSH: $(grep -A3 'ninfer-8x5090:' "$DSH_SETTINGS" 2>/dev/null | grep baseURL | sed 's/^ *//' || echo 'ninfer-8x5090 route missing')"
  [[ -f "$ENDPOINTS_FILE" ]] && echo "endpoints: $ENDPOINTS_FILE"
  return 0
}

# ------------------------------------------------------------------- up ---

cmd_up() {
  local target="${1:-both}"
  local vs rs
  local -a checks=()

  if [[ "$target" == "vast" || "$target" == "both" ]]; then
    vs="$(vast_state)"
    if [[ "$vs" == "running" ]]; then
      echo "vast instance $VAST_INSTANCE already running"
    else
      echo "starting vast instance $VAST_INSTANCE (was: $vs)..."
      vastai start instance "$VAST_INSTANCE"
      # ports are stable across stop/start, but re-read after (re)create
      read -r VAST_IP VAST_P8000 VAST_P8001 < <(vast_mapping) || true
      VAST_URL=""
      [[ "$VAST_IP" != "-" && -n "$VAST_P8000" ]] && VAST_URL="http://${VAST_IP}:${VAST_P8000}/v1"
      VAST_VISION_URL=""
      [[ "$VAST_IP" != "-" && -n "$VAST_P8001" ]] && VAST_VISION_URL="http://${VAST_IP}:${VAST_P8001}/v1"
    fi
    checks+=(vast "$VAST_URL")
  fi

  if [[ "$target" == "runpod" || "$target" == "both" ]]; then
    rs="$(runpod_state)"
    if [[ "$rs" == "running" ]]; then
      echo "runpod pod $RUNPOD_POD already running"
    else
      echo "starting runpod pod $RUNPOD_POD (was: $rs)..."
      runpodctl pods start "$RUNPOD_POD"
    fi
    checks+=(runpod "$RP_URL")
  fi

  # Wait for health; a warm fleet fast-paths the first probe.
  local i=0
  while (( i < ${#checks[@]} )); do
    if [[ -n "${checks[i+1]}" ]]; then
      wait_health "${checks[i]}" "${checks[i+1]}"
    fi
    i=$(( i + 2 ))
  done

  # Sync config: Vast port is stable across stop/start but a recreate changes
  # it — refresh-dsh-endpoint.sh is the single source of truth for that side.
  if [[ "$target" == "vast" || "$target" == "both" ]]; then
    /code/llm-cluster/ninfer-multi/refresh-dsh-endpoint.sh "$VAST_INSTANCE"
    # A (re)create changes the Vast host ports: restart the pool proxy so
    # its NINFER_POOL_VAST_BASE picks up the new mapping.
    if pool_running; then
      echo "restarting pool proxy with new Vast mapping..."
      local pp; pp="$(pool_pid || true)"; [[ -n "$pp" ]] && kill "$pp"
      sleep 2; cmd_pool start
    fi
  fi
  if [[ "$target" == "runpod" || "$target" == "both" ]]; then
    if grep -q 'ninfer-7x5090:' "$DSH_SETTINGS" 2>/dev/null; then
      rewrite_baseurl ninfer-7x5090 "$RP_URL"
    else
      echo "note: no ninfer-7x5090 provider in $DSH_SETTINGS (run: fleet.sh register)"
    fi
  fi
  write_endpoints
  echo "fleet up."
}

# ------------------------------------------------------------------ down ---

cmd_down() {
  local target="${1:-both}"
  local vs rs

  if [[ "$target" == "vast" || "$target" == "both" ]]; then
    vs="$(vast_state)"
    if [[ "$vs" == "running" ]]; then
      echo "stopping vast instance $VAST_INSTANCE..."
      vastai stop instance "$VAST_INSTANCE"
    else
      echo "vast instance $VAST_INSTANCE already stopped ($vs)"
    fi
  fi

  if [[ "$target" == "runpod" || "$target" == "both" ]]; then
    rs="$(runpod_state)"
    if [[ "$rs" == "running" ]]; then
      echo "stopping runpod pod $RUNPOD_POD..."
      runpodctl pods stop "$RUNPOD_POD"
    else
      echo "runpod pod $RUNPOD_POD already stopped ($rs)"
    fi
  fi
  echo "fleet down. GPU billing stopped (runpod 30GB volume still bills ~\$0.44/mo)."
}

# ------------------------------------------------------------- register ---

cmd_register() {
  if grep -q 'ninfer-7x5090:' "$DSH_SETTINGS" 2>/dev/null; then
    echo "ninfer-7x5090 provider already present in $DSH_SETTINGS"
    return 0
  fi
  python3 - "$DSH_SETTINGS" "$RP_URL" "$RUNPOD_POD" <<'PY'
import sys
path, url, pod = sys.argv[1:4]
src = open(path).read()
anchor = '    ninfer-8x5090:'
if anchor not in src:
    sys.exit(f'anchor line {anchor!r} not found in {path}')
block = f'''    # RunPod 7x5090 ninfer split fleet (pod {pod}, $6.93/hr). Proxy URL is
    # stable across stop/start (pinned to pod id); container disk is wiped on
    # stop, model artifact lives on the attached volume. Distinct key from the
    # Vast fleet: NINFER_RUNPOD_API_KEY (the fleet rejects the Vast key, 401).
    ninfer-7x5090:
      displayName: "ninfer 7x5090 (RunPod, text)"
      api: openai-completions
      baseURL: {url}
      apiKeyEnv: NINFER_RUNPOD_API_KEY
      timeoutMs: 900000
      streamIdleTimeoutMs: 600000
      models:
        - id: qwen3.8-27b
          name: "Qwen3.8-27B NVFP4 (7x5090 ninfer)"
          contextWindow: 262144
          maxTokens: 32768
          input: [ text ]

'''
src = src.replace(anchor, block + anchor, 1)
open(path, 'w').write(src)
print(f'added ninfer-7x5090 provider to {path}')
PY
  rewrite_baseurl ninfer-7x5090 "$RP_URL"
}

# ------------------------------------------------------------------ pool ---

# Load KEY: value lines from the DSH credentials file into the environment.
load_credentials() {
  [[ -f "$CREDENTIALS" ]] || { echo "credentials file $CREDENTIALS missing" >&2; return 1; }
  while IFS= read -r line; do
    if [[ "$line" =~ ^([A-Z0-9_]+):[[:space:]]*(.+)$ ]]; then
      export "${BASH_REMATCH[1]}=${BASH_REMATCH[2]}"
    fi
  done < "$CREDENTIALS"
}

pool_running() {
  [[ "$(health_code "http://127.0.0.1:${POOL_PORT}/health/liveliness")" == "200" ]]
}

pool_pid() {
  ss -tlnp 2>/dev/null | grep ":${POOL_PORT} " | grep -oP 'pid=\K[0-9]+' | head -1
}

cmd_pool() {
  local action="${1:-status}"
  case "$action" in
    start)
      if pool_running; then echo "pool proxy already running on :${POOL_PORT}"; return 0; fi
      command -v "$LITELLM_BIN" >/dev/null || {
        echo "litellm not found at $LITELLM_BIN (install: python3 -m venv /opt/litellm && /opt/litellm/bin/pip install 'litellm[proxy]' + pin fastapi==0.140.0)" >&2
        return 1
      }
      load_credentials
      [[ -n "${NINFER_POOL_API_KEY:-}" ]] || { echo "NINFER_POOL_API_KEY missing from $CREDENTIALS" >&2; return 1; }
      # Vast base: live mapping if reachable, else whatever the DSH route holds.
      local vast_base="${VAST_URL}"
      if [[ -z "$vast_base" ]]; then
        vast_base="$(grep -A3 'ninfer-8x5090:' "$DSH_SETTINGS" 2>/dev/null | grep baseURL | sed 's/^ *baseURL: *//' | head -1)"
      fi
      [[ -n "$vast_base" ]] || { echo "cannot determine Vast base (fleet down?)" >&2; return 1; }
      export NINFER_POOL_VAST_BASE="$vast_base"
      [[ -n "$VAST_VISION_URL" ]] && export NINFER_POOL_VAST_VISION_BASE="$VAST_VISION_URL"
      nohup setsid "$LITELLM_BIN" --config "$POOL_CONFIG" --port "$POOL_PORT" --host 0.0.0.0 > "$POOL_LOG" 2>&1 < /dev/null &
      echo "pool proxy starting on :${POOL_PORT} (log: $POOL_LOG)..."
      for _ in $(seq 1 12); do
        if pool_running; then echo "pool proxy up: text=qwen3.8-27b, vision=qwen3.8-27b-vision"; return 0; fi
        sleep 2
      done
      echo "pool proxy did not come up in 24s — check $POOL_LOG" >&2
      return 1
      ;;
    stop)
      local pid; pid="$(pool_pid || true)"
      if [[ -z "$pid" ]]; then echo "pool proxy not running"; return 0; fi
      kill "$pid" && echo "pool proxy stopped (pid $pid)"
      ;;
    status)
      if pool_running; then
        echo "pool proxy UP on :${POOL_PORT} (pid $(pool_pid))"
      else
        echo "pool proxy DOWN on :${POOL_PORT} (start: fleet.sh pool start)"
      fi
      ;;
    *) echo "usage: fleet.sh pool {start|stop|status}" >&2; return 2 ;;
  esac
}

# --------------------------------------------------------------- vast new ---
# One-command Vast recreate: search offer -> create from the canonical
# template (private template 584160 "ninfer-8x5090" — the same one shown in
# the Vast GUI: image v6, split-fleet env, ports 8000/8001, onstart entrypoint
# with mkdir guard, HF_TOKEN, offer filter 8x5090 CUDA>=13.1) -> attach SSH
# -> state file -> wait for /health -> repoint DSH + pool proxy.
#
# Vast platform quirks this encodes (learned 2026-08-21, both cost money):
#   * --disk is a CREATE-TIME flag; the template's --disk_space does NOT size
#     the container disk (defaulted to 10 GB = guaranteed artifact-pull
#     failure). Default here: 50 GB (20.5 GB artifact + headroom).
#   * `update template` has REPLACE semantics — a partial update (e.g. only
#     --disk_space) WIPES the other fields (image/env/onstart), after which
#     every create fails "400: Invalid args". Always re-issue the FULL spec.
#     See RUNBOOK-vast.md for the full re-issue command.
#   * The console's "recreate container with last template" button creates a
#     NEW instance (double billing) — never use it on a running fleet.

VAST_TEMPLATE_HASH="${VAST_TEMPLATE_HASH:-dd2a1614c1bf092ffb5f84ca330458ed}"
VAST_DISK_GB="${VAST_DISK_GB:-50}"
VAST_OFFER_FILTER="${VAST_OFFER_FILTER:-num_gpus>=8 gpu_name=RTX_5090 cuda_vers>=13.1 rentable=true verified=true}"
VAST_NEW_WAIT="${VAST_NEW_WAIT:-900}"

cmd_vast_new() {
  local existing offers offer out id ip p8000 p8001 waited code
  # 1. Refuse while a Vast instance is running/starting (no double billing).
  existing="$(vastai show instances --raw 2>/dev/null | python3 -c '
import json, sys
d = json.load(sys.stdin)
insts = d if isinstance(d, list) else d.get("instances") or []
print(",".join(str(i.get("id")) for i in insts
               if (i.get("cur_state") or "").lower() in ("running", "starting")))
' 2>/dev/null || true)"
  if [[ -n "$existing" ]]; then
    echo "Vast instance(s) already up: $existing — stop/destroy first (no double billing)" >&2
    return 1
  fi
  # 2. Offers, cheapest first; try each until one takes.
  offers="$(vastai search offers "$VAST_OFFER_FILTER" -o dph 2>/dev/null |
            sed 's/\x1b\[[0-9;]*m//g' | awk 'NR>1 {print $2}')"
  if [[ -z "$offers" ]]; then
    echo "no matching offers ($VAST_OFFER_FILTER)" >&2
    return 1
  fi
  id=""
  while read -r offer; do
    [[ -n "$offer" ]] || continue
    out="$(vastai create instance "$offer" --template_hash "$VAST_TEMPLATE_HASH" \
             --disk "$VAST_DISK_GB" 2>&1 || true)"
    if echo "$out" | grep -q 'success.*True'; then
      id="$(echo "$out" | grep -oP '"new_contract":\s*\K[0-9]+' | head -1)"
      echo "created instance $id from offer $offer (${VAST_DISK_GB} GB disk)"
      break
    else
      echo "offer $offer rejected: $(echo "$out" | head -c 90)"
    fi
  done <<< "$offers"
  if [[ -z "$id" ]]; then
    echo "all offers rejected — template broken? (update template is REPLACE: re-issue the FULL spec, see RUNBOOK-vast.md)" >&2
    return 1
  fi
  vastai attach ssh "$id" "$(cat ~/.ssh/id_ed25519.pub)" >/dev/null 2>&1 || true
  echo "$id" > "$VAST_STATE_FILE"
  VAST_INSTANCE="$id"
  # 3. Wait for host ports + /health (artifact pull ~4 min with HF_TOKEN;
  #    8 engines ~2-3 min more). Ports are allocated a minute or two after start.
  waited=0
  VAST_URL=""; VAST_VISION_URL=""
  while (( waited < VAST_NEW_WAIT )); do
    read -r ip p8000 p8001 < <(vast_mapping) || true
    if [[ -n "$ip" && "$ip" != "-" && -n "$p8000" ]]; then
      code="$(health_code "http://${ip}:${p8000}/health")"
      if [[ "$code" == "200" ]]; then break; fi
    fi
    waited=$((waited + 20))
    echo "  waiting for $id (ip=${ip:-...} :8000=${p8000:-...}) ${waited}/${VAST_NEW_WAIT}s"
    sleep 20
  done
  read -r ip p8000 p8001 < <(vast_mapping) || true
  if [[ -z "${ip:-}" || "$ip" == "-" || -z "${p8000:-}" ]]; then
    echo "timed out waiting for $id — diagnose: vastai logs $id" >&2
    return 1
  fi
  VAST_URL="http://${ip}:${p8000}/v1"
  [[ -n "$p8001" ]] && VAST_VISION_URL="http://${ip}:${p8001}/v1"
  # 4. Repoint: DSH route + endpoints file + pool proxy (fresh base).
  rewrite_baseurl ninfer-8x5090 "$VAST_URL"
  write_endpoints
  cmd_pool stop >/dev/null 2>&1 || true
  sleep 1
  cmd_pool start
  echo "VAST FRESH: instance $id ip=$ip text=$VAST_URL vision=${VAST_VISION_URL:-none}"
  echo "verify: fleet.sh status"
}

# ------------------------------------------------------------------- main ---

cmd="${1:-status}"
case "$cmd" in
  status)   shift; cmd_status "$@" ;;
  up)       shift; cmd_up "${1:-both}" ;;
  down)     shift; cmd_down "${1:-both}" ;;
  register) shift; cmd_register "$@" ;;
  pool)     shift; cmd_pool "${1:-status}" ;;
  vast)     [[ "${2:-}" == "new" ]] && { cmd_vast_new; exit; }
            echo "usage: fleet.sh vast new" >&2; exit 2 ;;
  *) echo "usage: fleet.sh {status|up|down|register|pool|vast new} [vast|runpod]" >&2; exit 2 ;;
esac