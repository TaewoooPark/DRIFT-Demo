"""Demo HTTP server — the two views, an SSE event stream, and the generate API.

stdlib only (``ThreadingHTTPServer``). One process runs the UDP listener for
worker events, the in-process sink for head events, the SSE fan-out, and the
generation worker thread.

    GET  /            landing page (links to the two views)
    GET  /a           consumer view  (prompt in, tokens out)
    GET  /b           provider view  (compute, receipts, earnings)
    GET  /events      SSE stream of demo events (first frame = state snapshot)
    GET  /api/state   current folded state
    POST /api/generate {"prompt": …, "max_new_tokens"?: …}
"""

from __future__ import annotations

import json
import os
import queue
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import events

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
_CTYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
           ".js": "text/javascript; charset=utf-8", ".svg": "image/svg+xml",
           ".png": "image/png"}

INDEX_HTML = """<!doctype html><meta charset="utf-8"><title>DRIFT demo</title>
<body style="background:#000;color:#fff;font-family:'Courier New',Courier,monospace;
font-weight:700;display:flex;flex-direction:column;align-items:center;
justify-content:center;height:100vh;margin:0;gap:2rem">
<pre style="color:#fff;line-height:1.25;margin:0"> ___  ___ ___ ___ _____
|   \\| _ \\_ _| __|_   _|
| |) |   /| || _|  | |
|___/|_|_\\___|_|   |_|</pre>
<div style="text-transform:uppercase;font-size:11px;letter-spacing:.06em">one model, two machines — open one view per screen</div>
<div style="display:flex;gap:1.4rem">
<a href="/a" style="color:#000;background:#fff;border:2px solid #fff;
padding:.6rem 1.4rem;text-decoration:none">[ A · CONSUMER ]</a>
<a href="/b" style="color:#fff;border:2px solid #fff;padding:.6rem 1.4rem;
text-decoration:none">[ B · PROVIDER ]</a>
</div></body>"""


class Bus:
    """Fan-out of demo events to SSE subscribers, plus a folded state snapshot
    (so a page that connects late — or reloads — renders instantly)."""

    def __init__(self):
        self._subs: list[queue.Queue] = []
        self._lock = threading.Lock()
        self.state = {
            "ready": False, "model": None, "plan": None, "device": None,
            "n_layers": None, "busy": False, "session": None,
            "nodes": {},        # node name -> {port, device, pubkey}
            "ledger": {},       # pubkey16  -> {tokens, layer_tokens}
            "last": {"text": "", "tps": 0.0, "tokens": 0, "prompt": None},
            "receipts_checked": 0, "suspects": [], "sessions": 0,
        }

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=2000)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            try:
                self._subs.remove(q)
            except ValueError:
                pass

    def publish(self, evt: dict) -> None:
        try:
            self._fold(evt)
        except Exception:
            pass
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(evt)
            except queue.Full:
                pass

    def _fold(self, e: dict) -> None:
        s, t = self.state, e.get("t")
        if t == "node.up":
            s["nodes"][e["node"]] = {"port": e.get("port"), "device": e.get("device"),
                                     "pubkey": e.get("pubkey")}
        elif t == "head.plan":
            s.update(ready=True, model=e.get("model"), plan=e.get("nodes"),
                     device=e.get("device"), n_layers=e.get("n_layers"))
        elif t == "head.session_start":
            s["busy"] = True
            s["session"] = e.get("session")
            s["sessions"] += 1
            s["last"] = {"text": "", "tps": 0.0, "tokens": 0, "prompt": e.get("prompt")}
        elif t == "head.token":
            s["last"]["text"] += e.get("text", "")
            s["last"]["tokens"] = e.get("i", 0)
            s["last"]["tps"] = e.get("tps", 0.0)
        elif t == "head.receipts":
            s["receipts_checked"] = e.get("checked", s["receipts_checked"])
            s["suspects"] = e.get("suspects", [])
            for h in e.get("hops", []):
                a = s["ledger"].setdefault(h["node"], {"tokens": 0, "layer_tokens": 0})
                a["tokens"] += 1
                a["layer_tokens"] += int(h["end"]) - int(h["start"])
        elif t in ("head.session_end", "head.error"):
            s["busy"] = False


def udp_listener(bus: Bus, port: int, host: str = "127.0.0.1") -> None:
    """Receive worker events (fire-and-forget UDP JSON) and publish them."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    while True:
        data, _ = sock.recvfrom(65535)
        try:
            evt = json.loads(data.decode())
        except ValueError:
            continue
        bus.publish(evt)


class Handler(BaseHTTPRequestHandler):
    bus: Bus = None      # set by the launcher
    head = None          # set by the launcher (DemoHead)
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # keep the launcher's stdout clean
        pass

    # ---- plumbing -----------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj).encode())

    def _file(self, name: str) -> None:
        fp = os.path.join(STATIC, os.path.basename(name))
        if not os.path.isfile(fp):
            self._json(404, {"error": "not found"})
            return
        with open(fp, "rb") as f:
            body = f.read()
        self._send(200, body, _CTYPES.get(os.path.splitext(fp)[1], "application/octet-stream"))

    # ---- routes ---------------------------------------------------------------
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send(200, INDEX_HTML.encode(), "text/html; charset=utf-8")
        elif path == "/a":
            self._file("a.html")
        elif path == "/b":
            self._file("b.html")
        elif path.startswith("/static/"):
            self._file(path[len("/static/"):])
        elif path == "/events":
            self._sse()
        elif path == "/api/state":
            self._json(200, self.bus.state)
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path != "/api/generate":
            self._json(404, {"error": "not found"})
            return
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            self._json(400, {"error": "bad json"})
            return
        prompt = (body.get("prompt") or "").strip()
        if not prompt:
            self._json(400, {"error": "empty prompt"})
            return
        head = self.head
        if head is None or not head.ready.is_set():
            self._json(503, {"error": "still assembling — workers are loading their layer slices"})
            return
        if not head.busy.acquire(blocking=False):
            self._json(409, {"error": "a generation is already running"})
            return
        session = f"demo-{int(time.time())}"
        max_new = body.get("max_new_tokens")

        def run():
            try:
                head.generate(prompt, max_new, session_id=session)
            except Exception as e:
                events.emit("head.error", session=session,
                            error=f"{type(e).__name__}: {e}"[:300])
            finally:
                head.busy.release()

        threading.Thread(target=run, daemon=True).start()
        self._json(200, {"ok": True, "session": session})

    # ---- SSE --------------------------------------------------------------------
    def _sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        q = self.bus.subscribe()
        try:
            hello = json.dumps({"t": "hello", "state": self.bus.state})
            self.wfile.write(f"data: {hello}\n\n".encode())
            self.wfile.flush()
            while True:
                try:
                    evt = q.get(timeout=15)
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    continue
                self.wfile.write(f"data: {json.dumps(evt)}\n\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.bus.unsubscribe(q)


__all__ = ["Bus", "Handler", "ThreadingHTTPServer", "udp_listener"]
