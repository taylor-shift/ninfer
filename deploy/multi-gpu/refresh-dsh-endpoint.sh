#!/usr/bin/env bash
#
# Point the DSH `ninfer-8x5090` provider at the instance's current host port.
#
# Vast allocates the container's published ports from the machine's own range
# (e.g. 40100-40587) and offers no way to pin one, so the external port is not
# guaranteed across a recreate. This reads the live mapping and rewrites the
# provider baseURL in ~/.dsh/settings.yaml.
#
# Usage:  ./refresh-dsh-endpoint.sh [INSTANCE_ID]

set -euo pipefail

INSTANCE="${1:-48240339}"
SETTINGS="${DSH_SETTINGS:-/root/.dsh/settings.yaml}"
PROVIDER="${PROVIDER_NAME:-ninfer-8x5090}"

command -v vastai >/dev/null || { echo "vastai CLI not found" >&2; exit 1; }

read -r IP PORT < <(
  vastai show instance "$INSTANCE" --raw 2>/dev/null | python3 -c '
import json, sys
d = json.load(sys.stdin)
ports = d.get("ports") or {}
m = ports.get("8000/tcp")
if not m:
    sys.exit("instance has no 8000/tcp mapping (was -p 8000:8000 set at create time?)")
print(d.get("public_ipaddr"), m[0]["HostPort"])
'
)

URL="http://${IP}:${PORT}/v1"
echo "instance ${INSTANCE} -> ${URL}"

# Verify the endpoint actually serves before rewriting config: a stale-but-working
# baseURL beats a fresh-but-dead one.
code=$(curl -s -m 15 -o /dev/null -w '%{http_code}' "http://${IP}:${PORT}/health" || true)
if [[ "$code" != "200" ]]; then
    echo "health check returned ${code:-000}; refusing to update ${SETTINGS}" >&2
    exit 1
fi

python3 - "$SETTINGS" "$PROVIDER" "$URL" <<'PY'
import re, sys
path, provider, url = sys.argv[1], sys.argv[2], sys.argv[3]
src = open(path).read()

# Rewrite only the baseURL inside this provider's block. Line-oriented rather
# than a YAML round-trip so comments and formatting survive untouched.
lines = src.splitlines(keepends=True)
out, in_block, done = [], False, False
for line in lines:
    if re.match(rf'^\s*{re.escape(provider)}:\s*$', line):
        in_block = True
    elif in_block and re.match(r'^\s{0,6}\S', line) and not re.match(r'^\s{6,}', line):
        in_block = False          # dedented out of the provider block
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

echo "health 200; DSH provider ${PROVIDER} now points at ${URL}"
