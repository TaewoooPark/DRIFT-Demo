"""The demo head — a weightless (thin) DRIFT head with a step-wise decode loop.

drift's own ``generate_stream`` doesn't support the thin head, and the demo
needs per-token control anyway (events between steps), so the loop lives here.
It is the same loop shape as ``Orchestrator.generate`` — thin prefill/decode
via ``route_token``, receipt verification per step, ``NodeUnavailable`` →
recover + replay (M9) — with demo events emitted at every step boundary.
"""

from __future__ import annotations

import threading
import time

from . import events


class DemoHead:
    def __init__(self, model_id: str, dtype: str, endpoints: list[dict],
                 max_new_tokens: int = 200):
        from drift.common import pick_device

        self.model_id = model_id
        self.dtype = dtype
        self.device = pick_device(None)
        self.endpoints = endpoints
        self.max_new_tokens = max_new_tokens
        self.orch = None
        self.plan = None
        self.ready = threading.Event()
        self.busy = threading.Lock()   # one generation at a time

    # ---- assembly (slow: every worker loads its layer slice) ----------------
    def build(self) -> None:
        from drift.run import build_over_nodes

        t0 = time.perf_counter()
        events.emit("head.building", model=self.model_id, nodes=len(self.endpoints))
        orch, plan = build_over_nodes(self.model_id, self.dtype, self.device,
                                      self.endpoints, chain=True, thin=True)
        self.orch, self.plan = orch, plan
        self.ready.set()
        events.emit("head.plan", model=self.model_id, thin=True, chain=True,
                    device=self.device, n_layers=orch.n_layers,
                    build_s=round(time.perf_counter() - t0, 1),
                    nodes=[{"name": p["name"], "host": p["host"], "port": p["port"],
                            "start": p["start"], "end": p["end"],
                            "device": p.get("device"),
                            "pubkey": (p.get("pubkey") or "")[:16]} for p in plan])

    # ---- plan re-broadcast (the split changes on failover) --------------------
    def _emit_plan(self) -> None:
        """Emit a fresh head.plan from the live transport — after a recovery the
        survivors hold different layer ranges, and the views rebuild on plan."""
        orch = self.orch
        nodes = []
        for name in orch.order:
            s = orch.transport.shards[name]
            info = {}
            try:
                info = orch.transport.ping(name)
            except Exception:
                pass
            nodes.append({"name": name, "host": s["host"], "port": s["port"],
                          "start": info.get("start_layer"), "end": info.get("end_layer"),
                          "device": info.get("device"),
                          "pubkey": (info.get("pubkey") or "")[:16]})
        events.emit("head.plan", model=self.model_id, thin=True, chain=True,
                    device=self.device, n_layers=orch.n_layers, nodes=nodes)

    # ---- per-step receipt event ----------------------------------------------
    def _emit_receipts(self, session_id: str, mode: str) -> None:
        orch = self.orch
        rs = getattr(orch.transport, "last_receipts", None) or []
        hops = [{"node": r["node"][:16], "start": r["start"], "end": r["end"],
                 "in": r["in_hash"].hex()[:12], "out": r["out_hash"].hex()[:12],
                 "sig": r["sig"].hex()[:12]} for r in rs]
        v = orch.verifier
        events.emit("head.receipts", session=session_id, seq=orch.transport.seq,
                    mode=mode, hops=hops,
                    checked=v.checked if v else 0,
                    suspects=v.suspects() if v else [])

    # ---- generation ------------------------------------------------------------
    def generate(self, prompt: str, max_new_tokens: int | None = None,
                 session_id: str = "s0") -> dict:
        from drift.common import build_input_ids
        from drift.orchestrator import NodeUnavailable

        orch = self.orch
        n_new = max(1, min(1024, int(max_new_tokens or self.max_new_tokens)))
        tok = orch.head.tokenizer
        prompt_ids = build_input_ids(tok, prompt)[0].tolist()
        eos = orch._eos_set(True)

        events.emit("head.session_start", session=session_id, prompt=prompt[:400],
                    prompt_tokens=len(prompt_ids), max_new=n_new)
        t_start = time.perf_counter()
        generated: list[int] = []
        prev_text = ""
        try:
            while True:  # replay loop: a mid-run drop → recover + re-prefill (M9)
                try:
                    seq = prompt_ids + generated
                    events.emit("head.prefill_start", session=session_id, seq_len=len(seq))
                    t0 = time.perf_counter()
                    next_id, _ = orch._prefill(session_id, seq)
                    events.emit("head.prefill_end", session=session_id,
                                ms=round((time.perf_counter() - t0) * 1000.0, 1))
                    self._emit_receipts(session_id, "prefill")
                    p = len(seq)
                    while len(generated) < n_new:
                        if next_id in eos:
                            break
                        generated.append(next_id)
                        text = tok.decode(generated)
                        delta, prev_text = text[len(prev_text):], text
                        dt = time.perf_counter() - t_start
                        events.emit("head.token", session=session_id, i=len(generated),
                                    token_id=next_id, text=delta,
                                    tps=round(len(generated) / dt, 2) if dt > 0 else 0.0)
                        if len(generated) >= n_new:
                            break
                        t0 = time.perf_counter()
                        next_id = orch._decode(session_id, next_id, p)
                        self._emit_receipts(session_id, "decode")
                        events.emit("head.step", session=session_id,
                                    ms=round((time.perf_counter() - t0) * 1000.0, 1))
                        p += 1
                    break
                except NodeUnavailable as e:
                    events.emit("head.recovering", session=session_id, error=str(e)[:200])
                    orch._recover(session_id)
                    self._emit_plan()  # the split changed — views rebuild their panels
                    events.emit("head.recovered", session=session_id,
                                recoveries=orch.recoveries)
        finally:
            for name in orch.order:
                try:
                    orch.transport.reset(name, session_id)
                except Exception:
                    pass
        secs = time.perf_counter() - t_start
        out = {"session": session_id, "text": prev_text, "tokens": len(generated),
               "seconds": round(secs, 2),
               "tps": round(len(generated) / secs, 2) if secs > 0 else 0.0}
        events.emit("head.session_end", **out)
        return out
