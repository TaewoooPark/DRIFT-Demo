"""Out-of-band demo event bus.

Every process in the demo (worker nodes, the head) emits small JSON events that
drive the browser views. Emission is fire-and-forget UDP so it can never block
or perturb the DRIFT data plane — the parity-gated inference path is untouched.

``DRIFT_DEMO_EVENTS=host:port[,host:port…]`` enables UDP emission (the launcher
sets it for worker subprocesses). The head process instead wires
``EMITTER.local`` straight to the demo server's fan-out bus.
"""

from __future__ import annotations

import json
import os
import socket
import time

# macOS caps a UDP datagram send at ~9 KB by default; stay well under it.
_MAX_DGRAM = 8000


class Emitter:
    def __init__(self, spec: str | None = None):
        spec = spec if spec is not None else os.environ.get("DRIFT_DEMO_EVENTS", "")
        self.targets: list[tuple[str, int]] = []
        for tok in spec.split(","):
            tok = tok.strip()
            if not tok:
                continue
            host, _, port = tok.rpartition(":")
            self.targets.append((host or "127.0.0.1", int(port)))
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if self.targets else None
        if self._sock is not None:
            # emission must never block the data plane — if the send buffer is
            # ever full, drop the datagram instead of waiting
            self._sock.setblocking(False)
        self.local = None  # in-process sink: callable(dict), set by the demo server

    def emit(self, etype: str, **fields) -> None:
        if self._sock is None and self.local is None:
            return
        evt = {"t": etype, "ts": time.time(), **fields}
        if self.local is not None:
            try:
                self.local(evt)
            except Exception:
                pass
        if self._sock is not None:
            data = json.dumps(evt).encode()
            if len(data) > _MAX_DGRAM:
                data = json.dumps({"t": etype, "ts": evt["ts"], "truncated": True}).encode()
            for tgt in self.targets:
                try:
                    self._sock.sendto(data, tgt)
                except OSError:
                    pass


EMITTER = Emitter()


def emit(etype: str, **fields) -> None:
    EMITTER.emit(etype, **fields)
