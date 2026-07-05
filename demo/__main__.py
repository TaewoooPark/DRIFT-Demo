"""``python -m demo`` — launch the whole local demo.

Spawns instrumented DRIFT worker processes, assembles a weightless head over
them (peer-to-peer chain + thin head), serves the two views, and opens them:

    http://127.0.0.1:8800/a   consumer  — prompt in, tokens out
    http://127.0.0.1:8800/b   provider  — compute, receipts, earnings

Verified receipts are journaled to ``.state/journal-<ts>.jsonl`` — audit a demo
run afterwards with ``drift ledger <journal> --verify``.

Startup order matters: every socket (HTTP + the UDP event port) is bound
*before* any worker is spawned, so a port conflict fails fast and can never
orphan a worker process.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
import webbrowser

from . import events
from .head import DemoHead
from .server import Bus, Handler, ThreadingHTTPServer, make_events_socket, udp_listener

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
_KEEP_JOURNALS = 10


def _prune_journals(state_dir: str) -> None:
    """Keep only the newest journals so .state doesn't grow without bound."""
    js = sorted(f for f in os.listdir(state_dir)
                if f.startswith("journal-") and f.endswith(".jsonl"))
    for f in js[: max(0, len(js) - (_KEEP_JOURNALS - 1))]:
        try:
            os.unlink(os.path.join(state_dir, f))
        except OSError:
            pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m demo",
        description="two-screen visual demo of a DRIFT peer-to-peer inference run")
    ap.add_argument("--local", action="store_true",
                    help="run everything on this machine (currently the only mode; default)")
    ap.add_argument("--nodes", type=int, default=2, help="number of local workers (default 2)")
    ap.add_argument("--model", default=os.environ.get("DRIFT_DEMO_MODEL", DEFAULT_MODEL))
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--port", type=int, default=8800, help="HTTP port for the views")
    ap.add_argument("--events-port", type=int, default=0,
                    help="UDP port for worker events (default 0 = pick a free one)")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--no-browser", action="store_true", help="don't open the views")
    args = ap.parse_args(argv)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    state_dir = os.path.join(root, ".state")
    os.makedirs(state_dir, exist_ok=True)
    _prune_journals(state_dir)

    # The head journals every verified receipt (M13's input substrate).
    journal = os.path.join(state_dir, f"journal-{int(time.time())}.jsonl")
    os.environ["DRIFT_JOURNAL"] = journal

    from drift.common import free_port

    # ---- bind every socket FIRST — fail fast while nothing is spawned -------
    bus = Bus()
    events.EMITTER.local = bus.publish  # head events go straight onto the bus
    events_port = args.events_port or free_port()
    try:
        udp_sock = make_events_socket(events_port)
        Handler.bus = bus
        Handler.head = None  # /api/generate answers 503 until the head is up
        httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as e:
        print(f"[demo] cannot bind (http :{args.port} / events udp :{events_port}) — "
              f"is another demo running? {e}", flush=True)
        return 2
    httpd.daemon_threads = True
    threading.Thread(target=udp_listener, args=(bus, udp_sock), daemon=True).start()

    node_ports = [free_port() for _ in range(args.nodes)]
    procs, logs = [], []
    try:
        # ---- workers ---------------------------------------------------------
        for i, p in enumerate(node_ports):
            env = {**os.environ,
                   "DRIFT_DEMO_EVENTS": f"127.0.0.1:{events_port}",
                   # one Ed25519 identity per worker → two distinct contributors
                   "DRIFT_IDENTITY_FILE": os.path.join(state_dir, f"node{i}.identity")}
            log = open(os.path.join(state_dir, f"node{i}.log"), "w")
            logs.append(log)
            procs.append(subprocess.Popen(
                [sys.executable, "-m", "demo.node_main", "--port", str(p)],
                cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT))
        print(f"[demo] spawned {args.nodes} local worker(s) on {node_ports}", flush=True)

        endpoints = [{"name": f"n{i}", "host": "127.0.0.1", "port": p}
                     for i, p in enumerate(node_ports)]
        head = DemoHead(args.model, args.dtype, endpoints,
                        max_new_tokens=args.max_new_tokens)
        Handler.head = head

        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        url_a = f"http://127.0.0.1:{args.port}/a"
        url_b = f"http://127.0.0.1:{args.port}/b"
        print(f"[demo] view A (consumer): {url_a}", flush=True)
        print(f"[demo] view B (provider): {url_b}", flush=True)
        print(f"[demo] receipt journal  : {journal}", flush=True)

        # ---- assemble in the background (workers load their layer slices) ----
        def build():
            try:
                head.build()
                print(f"[demo] ready — {args.model} split across {args.nodes} worker(s)",
                      flush=True)
            except Exception as e:
                events.emit("head.error", error=f"{type(e).__name__}: {e}"[:300])
                print(f"[demo] BUILD FAILED: {type(e).__name__}: {e}", flush=True)

        threading.Thread(target=build, daemon=True).start()

        if not args.no_browser:
            webbrowser.open_new(url_a)
            time.sleep(0.8)
            webbrowser.open_new(url_b)

        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\n[demo] shutting down", flush=True)
    finally:
        for pr in procs:
            pr.terminate()
        for pr in procs:
            try:
                pr.wait(timeout=10)
            except Exception:
                pr.kill()
        for log in logs:
            log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
