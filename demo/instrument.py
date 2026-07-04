"""Monkey-patch instrumentation of a stock DRIFT worker.

The DRIFT sources are never edited. This wraps four call sites at process start
and emits demo events around them — all *observation*, never a change to the
math, so the run stays exactly the run the parity gate proves:

  * ``TorchShardEngine.load``        — after loading, attach read-only forward
        hooks to every kept decoder layer: per layer, ‖Δh‖ of the last position
        (how much that layer rewrote this token's representation).
  * ``TorchShardEngine.forward``     — compute-only timing per step, plus the
        actual residual stream at this shard's boundaries, downsampled to 128
        buckets (the hidden state IS what crosses the wire — the views draw it).
  * ``TorchShardEngine.head_argmax`` — the thin-head tail's own lm_head logits:
        top-k next-token candidates with probabilities (the token returned is
        still the untouched original's argmax).
  * ``Node.handle`` / ``Node._relay`` — step arrival, configure, and the
        peer-to-peer hop (bytes + downstream target).

Everything emitted is out-of-band fire-and-forget UDP.
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from . import events

_HEAT_BUCKETS = 128
_TOPK = 8
_TOKENIZERS: dict = {}


def _tokenizer(model_id: str):
    t = _TOKENIZERS.get(model_id)
    if t is None:
        from transformers import AutoTokenizer

        t = _TOKENIZERS[model_id] = AutoTokenizer.from_pretrained(model_id)
    return t


def _downsample(hidden):
    """Last position of [B,S,H] → 128 mean-|activation| buckets, 0–255."""
    if hidden is None or not torch.is_tensor(hidden):
        return None, 0.0
    v = hidden[0, -1, :].detach().float().abs()
    pooled = F.adaptive_avg_pool1d(v.view(1, 1, -1), _HEAT_BUCKETS)[0, 0]
    mx = float(pooled.max())
    if mx <= 0.0:
        return [0] * _HEAT_BUCKETS, 0.0
    return (pooled / mx * 255).to(torch.uint8).tolist(), round(mx, 3)


def patch_worker() -> None:
    from drift import engine_torch, shard_server

    # ---- read-only per-layer hooks (attached once per engine, at load) --------
    orig_load = engine_torch.TorchShardEngine.load

    def load(self):
        orig_load(self)
        if getattr(self, "_demo_hooked", False) or not self.layers:
            return
        self._demo_hooked = True
        self._demo_deltas = []
        for layer in self.layers:
            def pre(mod, args):
                if args and torch.is_tensor(args[0]):
                    mod._demo_in = args[0][0, -1, :].detach().float()

            def post(mod, args, out, _eng=self):
                o = out if torch.is_tensor(out) else out[0]
                i = getattr(mod, "_demo_in", None)
                if i is not None and torch.is_tensor(o):
                    _eng._demo_deltas.append(
                        (o[0, -1, :].detach().float() - i).norm())
                    mod._demo_in = None

            layer.register_forward_pre_hook(pre)
            layer.register_forward_hook(post)
        if self.head_duty:
            # warm the tokenizer so the first top-k emit doesn't stall a token
            try:
                _tokenizer(self.model_id)
            except Exception:
                pass

    engine_torch.TorchShardEngine.load = load

    # ---- compute timing + the residual stream at this shard's boundaries ------
    orig_forward = engine_torch.TorchShardEngine.forward

    def forward(self, *args, **kw):
        hidden_in = kw.get("hidden", args[1] if len(args) > 1 else None)
        self._demo_deltas = []
        t0 = time.perf_counter()
        out = orig_forward(self, *args, **kw)
        ms = (time.perf_counter() - t0) * 1000.0
        mode = kw.get("mode", args[4] if len(args) > 4 else "?")
        session = kw.get("session_id", args[0] if args else "?")
        pos = kw.get("position_ids", args[2] if len(args) > 2 else None)
        try:
            deltas = ([round(float(x), 2) for x in torch.stack(self._demo_deltas).tolist()]
                      if self._demo_deltas else [])
            hin, hin_max = _downsample(hidden_in)
            hout, hout_max = _downsample(out)
        except Exception:
            deltas, hin, hout, hin_max, hout_max = [], None, None, 0.0, 0.0
        events.emit("node.compute", node=self.name, mode=mode, session=session,
                    start=self.start_layer, end=self.end_layer,
                    n_pos=len(pos) if pos is not None else 0,
                    ms=round(ms, 2), layers=deltas,
                    hin=hin, hout=hout, hin_max=hin_max, hout_max=hout_max)
        return out

    engine_torch.TorchShardEngine.forward = forward

    # ---- the tail's own logits: top-k candidates (display-only) ----------------
    orig_head_argmax = engine_torch.TorchShardEngine.head_argmax

    def head_argmax(self, hidden):
        token = orig_head_argmax(self, hidden)  # the untouched result
        try:
            logits = self.lm_head(self.norm_mod(hidden[:, -1:, :]))[:, -1, :].float()
            vals, ids = torch.topk(torch.softmax(logits, dim=-1)[0], k=_TOPK)
            tok = _tokenizer(self.model_id)
            cand = [[tok.decode([int(i)]), round(float(v), 4)]
                    for i, v in zip(ids, vals)]
            events.emit("node.topk", node=self.name, cand=cand, chosen=int(token))
        except Exception:
            pass
        return token

    engine_torch.TorchShardEngine.head_argmax = head_argmax

    # ---- step arrival + configure ------------------------------------------------
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

    # ---- the peer-to-peer hop ------------------------------------------------------
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
