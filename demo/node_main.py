"""Instrumented `drift node` for the demo.

The same worker `drift node` runs, minus mDNS/gossip/tunnel (a local demo needs
none of them), plus the demo event instrumentation. The launcher gives each
worker its own ``DRIFT_IDENTITY_FILE`` so two workers on one machine sign
receipts as two distinct contributors.
"""

from __future__ import annotations

import argparse
import sys

from . import events, instrument


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="demo.node_main",
                                 description="instrumented DRIFT worker for the demo")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--device", help="mps | cuda | cpu (default: auto-detect)")
    args = ap.parse_args(argv)

    instrument.patch_worker()

    from drift.common import pick_device
    from drift.shard_server import Node, serve

    device = pick_device(args.device)
    node = Node(name=f"node-{args.port}", model_id="(assigned by head)",
                dtype="float16", device=device)
    events.emit("node.up", node=node.name, port=args.port, device=device,
                pubkey=node._identity()[1][:16])
    banner = f"[demo node] {args.host}:{args.port} device={device} — waiting for configure"
    try:
        serve(node, args.host, args.port, banner=banner)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
