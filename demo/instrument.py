"""Monkey-patch instrumentation of a stock DRIFT worker.

The DRIFT sources are never edited. This wraps three call sites at process
start and emits demo events around them:

  * ``TorchShardEngine.forward`` — compute-only timing per prefill/decode step
  * ``Node.handle``              — step arrival (lights the view up during a
                                   long prefill, before compute finishes) and
                                   the ``configure`` that assigns a layer range
  * ``Node._relay``              — the peer-to-peer hop: bytes and downstream
                                   target (tensor to the next node, or the
                                   final token to the head's collect sink)

Everything emitted is out-of-band UDP; the DRIFT wire protocol and the math are
untouched, so the run stays exactly the run the parity gate proves.
"""

from __future__ import annotations

import time

from . import events


def patch_worker() -> None:
    from drift import engine_torch, shard_server

    # ---- compute timing ----------------------------------------------------
    orig_forward = engine_torch.TorchShardEngine.forward

    def forward(self, *args, **kw):
        t0 = time.perf_counter()
        out = orig_forward(self, *args, **kw)
        mode = kw.get("mode", args[4] if len(args) > 4 else "?")
        session = kw.get("session_id", args[0] if args else "?")
        pos = kw.get("position_ids", args[2] if len(args) > 2 else None)
        events.emit("node.compute", node=self.name, mode=mode, session=session,
                    start=self.start_layer, end=self.end_layer,
                    n_pos=len(pos) if pos is not None else 0,
                    ms=round((time.perf_counter() - t0) * 1000.0, 2))
        return out

    engine_torch.TorchShardEngine.forward = forward

    # ---- step arrival + configure -------------------------------------------
    orig_handle = shard_server.Node.handle

    def handle(self, msg):
        mtype = msg.get("type")
        if mtype in ("prefill", "decode") and self.engine is not None:
            tensor = msg.get("tensor")
            events.emit("node.step", node=self.name, mode=mtype,
                        seq=msg.get("seq_id"), session=msg.get("session_id"),
                        start=self.engine.start_layer, end=self.engine.end_layer,
                        in_bytes=len(tensor) if isinstance(tensor, (bytes, bytearray)) else 0,
                        embed=bool(msg.get("embed")),
                        n_pos=len(msg.get("position_ids") or []))
        elif mtype == "configure":
            reply = orig_handle(self, msg)
            events.emit("node.configure", node=self.name,
                        start=msg.get("start_layer"), end=msg.get("end_layer"),
                        device=self.device, model_id=msg.get("model_id"),
                        embed_duty=bool(msg.get("embed_duty")),
                        head_duty=bool(msg.get("head_duty")),
                        pubkey=(reply.get("pubkey") or "")[:16])
            return reply
        return orig_handle(self, msg)

    shard_server.Node.handle = handle

    # ---- the peer-to-peer hop ------------------------------------------------
    orig_relay = shard_server.Node._relay

    def _relay(self, msg, payload):
        route = msg["route"]
        target = route[0] if route else msg["collect"]
        tensor = payload.get("tensor")
        events.emit("node.relay", node=self.name, mode=msg.get("type"),
                    seq=msg.get("seq_id"), session=msg.get("session_id"),
                    to=[target[0], int(target[1])], tail=not route,
                    bytes=len(tensor) if isinstance(tensor, (bytes, bytearray)) else 0,
                    token=payload.get("token"))
        return orig_relay(self, msg, payload)

    shard_server.Node._relay = _relay
