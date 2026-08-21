#!/usr/bin/env python3
"""
pool-balancer.py — least-connections pool balancer (stdlib-only FALLBACK).

Primary pool layer is LiteLLM (litellm-config.yaml, run via `fleet.sh pool`).
Keep this around as a zero-dependency fallback: one threaded process, raw TCP
relay with header rewrite, no third-party packages.

Runs on the always-on DSH host. Balances across whole fleets, each of which
has its own internal per-engine least-connections balancer (balancer.pl)
behind :8000 (text) / :8001 (vision):

    :8800 text   -> Vast 8x5090 (6 text engines), RunPod 7x5090 (5), local 5090 (1)
    :8801 vision -> RunPod 7x5090 (2), Vast 8x5090 (2, when published)

Why raw TCP + header rewrite (not a full HTTP proxy):
  - Requests: read the head, rewrite Host + Authorization (each fleet uses a
    DIFFERENT API key), then blind-pipe the body. Keep-alive connections are
    pinned to one upstream for their lifetime — same tradeoff as balancer.pl,
    correct for streaming.
  - Responses: read the head, insert X-Pool-Upstream: <name> (observability),
    then blind-pipe. SSE/chunked/keep-alive semantics preserved; no buffering.

Health: a per-upstream thread probes GET /health every health_interval_sec;
only healthy upstreams receive traffic. If none are healthy the pool answers
503 with the per-upstream health map. GET /health on a pool port answers
locally with the health map (does not consume upstream capacity).

Config: POOL_BALANCER_CONFIG env, else pool-balancer.json next to this script
(chmod 600; carries the per-fleet keys — never commit it; see
pool-balancer.json.example).
"""

import json
import os
import socket
import ssl
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.environ.get("POOL_BALANCER_CONFIG", os.path.join(HERE, "pool-balancer.json"))
HEAD_CAP = 1_000_000  # 1 MB head cap (defensive; ninfer heads are small)
BUFFER = 65536


class Upstream:
    def __init__(self, spec):
        self.name = spec["name"]
        self.scheme = spec["scheme"]
        self.host = spec["host"]
        self.port = int(spec["port"])
        self.key = spec["key"]
        self.up = False
        self.in_flight = 0

    def host_header(self):
        return self.host if self.scheme == "https" else f"{self.host}:{self.port}"

    def connect(self):
        s = socket.create_connection((self.host, self.port), timeout=30)
        s.settimeout(None)
        if self.scheme == "https":
            ctx = ssl.create_default_context()
            s = ctx.wrap_socket(s, server_hostname=self.host)
        return s

    def probe(self, timeout):
        try:
            url = f"{self.scheme}://{self.host_header()}/health"
            req = urllib.request.Request(url, method="GET",
                                         headers={"User-Agent": "curl/8.5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                self.up = r.status == 200
        except Exception:
            self.up = False


def log(msg):
    print(f"[pool-balancer {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def health_loop(u, interval, timeout, stop):
    while not stop.is_set():
        before = u.up
        u.probe(timeout)
        if u.up != before:
            log(f"upstream {u.name} {'UP' if u.up else 'DOWN'}")
        stop.wait(interval)


def rewrite_head(head, u, tag):
    """Rewrite Host/Authorization; append X-Pool-Upstream INSIDE the head.

    `head` ends with the single \r\n\r\n terminator. The tag must be a real
    header line BEFORE that terminator — appending it after would make the
    upstream treat the tag line as the start of the body.
    """
    body = head[:-2] if head.endswith(b"\r\n\r\n") else head
    lines = body.split(b"\r\n")
    out = []
    for i, line in enumerate(lines):
        if i == 0:
            out.append(line)
            continue
        name = line.split(b":", 1)[0].strip().lower()
        if name == b"host":
            out.append(f"Host: {u.host_header()}".encode())
        elif name == b"authorization":
            out.append(f"Authorization: Bearer {u.key}".encode())
        else:
            out.append(line)
    out.append(f"X-Pool-Upstream: {tag}".encode())
    return b"\r\n".join(out) + b"\r\n\r\n"


def pipe(src, dst):
    """Blind pipe until EOF/error in this direction."""
    try:
        while True:
            buf = src.recv(BUFFER)
            if not buf:
                break
            off = 0
            while off < len(buf):
                n = dst.send(buf[off:])
                if n <= 0:
                    return
                off += n
    except (OSError, ssl.SSLError):
        pass
    finally:
        try:
            dst.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def read_head(sock):
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = sock.recv(65536)
        if not chunk:
            return None, None
        buf += chunk
        if len(buf) > HEAD_CAP:
            return None, None
    i = buf.index(b"\r\n\r\n") + 4
    return buf[:i], buf[i:]


def handle(conn, addr, pool, cfg):
    client = conn
    try:
        head, body = read_head(client)
        if head is None:
            return
        first = head.split(b"\r\n", 1)[0].decode("latin1")
        parts = first.split()
        method, path = parts[0] if parts else "", (parts[1] if len(parts) > 1 else "")

        # Local health endpoint: answer without touching upstreams.
        if method == "GET" and path == "/health":
            with pool["lock"]:
                ups = {u.name: {"up": u.up, "in_flight": u.in_flight} for u in pool["ups"]}
            payload = json.dumps({"pool": pool["name"], "upstreams": ups}).encode()
            resp = (f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                    f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n").encode() + payload
            client.sendall(resp)
            return

        with pool["lock"]:
            healthy = [u for u in pool["ups"] if u.up]
            if not healthy:
                payload = json.dumps({"error": "no healthy upstreams",
                                      "upstreams": {u.name: u.up for u in pool["ups"]}}).encode()
                resp = (f"HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\n"
                        f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n").encode() + payload
                client.sendall(resp)
                return
            u = min(healthy, key=lambda x: (x.in_flight, x.name))
            u.in_flight += 1

        try:
            upstream = u.connect()
        except OSError as e:
            with pool["lock"]:
                u.in_flight = max(0, u.in_flight - 1)
            log(f"connect {u.name} failed: {e}; answering 502")
            client.sendall(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n"
                           b"Connection: close\r\n\r\n")
            return

        log(f"{path[:80]} -> {u.name} (in_flight {u.in_flight})")

        # client -> upstream: rewritten head, then blind body pipe.
        def send_up():
            try:
                upstream.sendall(rewrite_head(head, u, u.name))
                if body:
                    upstream.sendall(body)
            except (OSError, ssl.SSLError):
                pass
            pipe(client, upstream)

        # upstream -> client: read response head, insert tag header, blind pipe.
        def recv_down():
            head2, body2 = read_head(upstream)
            if head2 is None:
                return
            out_lines = head2.split(b"\r\n")
            out_lines.insert(1, f"X-Pool-Upstream: {u.name}\r\n".encode())
            head2 = b"\r\n".join(out_lines)
            try:
                client.sendall(head2)
                if body2:
                    client.sendall(body2)
            except OSError:
                pass
            pipe(upstream, client)

        t1 = threading.Thread(target=send_up, daemon=True)
        t2 = threading.Thread(target=recv_down, daemon=True)
        t1.start(); t2.start()
        t1.join(); t2.join()

        with pool["lock"]:
            u.in_flight = max(0, u.in_flight - 1)
        upstream.close()
    except Exception as e:
        log(f"handler error for {addr}: {e!r}")
    finally:
        try:
            client.close()
        except OSError:
            pass


def make_pool(name, specs, cfg, stop):
    ups = [Upstream(s) for s in specs]
    pool = {"name": name, "ups": ups, "lock": threading.Lock()}
    for u in ups:
        threading.Thread(target=health_loop, args=(u, cfg["health_interval_sec"],
                                                   cfg["health_timeout_sec"], stop),
                         daemon=True).start()
    return pool


def serve(listen_port, pool, cfg, stop):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((cfg.get("listen", {}).get("bind", "0.0.0.0"), listen_port))
    srv.listen(512)
    srv.settimeout(1.0)
    log(f"listening on {cfg.get('listen', {}).get('bind', '0.0.0.0')}:{listen_port} ({pool['name']} pool)")
    while not stop.is_set():
        try:
            conn, addr = srv.accept()
        except socket.timeout:
            continue
        threading.Thread(target=handle, args=(conn, addr, pool, cfg), daemon=True).start()
    srv.close()


cfg = json.load(open(CONFIG))
stop = threading.Event()

pools = {
    "text": make_pool("text", cfg["pools"]["text"], cfg, stop),
    "vision": make_pool("vision", cfg["pools"]["vision"], cfg, stop),
}

threads = []
for name, port in {k: v for k, v in cfg["listen"].items() if k != "bind"}.items():
    t = threading.Thread(target=serve, args=(port, pools[name], cfg, stop), daemon=True)
    t.start()
    threads.append(t)

log(f"pool-balancer up: {json.dumps({n: [u.name for u in p['ups']] for n, p in pools.items()})}")
try:
    while True:
        time.sleep(3600)
except KeyboardInterrupt:
    stop.set()