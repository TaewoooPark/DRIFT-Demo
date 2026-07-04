# DRIFT-Demo

**A two-screen visual demo of a [DRIFT](https://github.com/TaewoooPark/DRIFT) peer-to-peer inference run — the "For Tokens" economy, both faces at once.**

One model is split layer-by-layer across two DRIFT workers (peer-to-peer chain, weightless head), and each side of the exchange gets its own full-screen view. Every panel shows the network's **actual internals**, not an abstraction:

| view | who | what the screen shows |
|---|---|---|
| **A · consumer** (`/a`) | the one asking | a chat box; **‖Δh‖ per layer** — how much each of its decoder layers (`[0:14)`) actually rewrote this token's representation; the **residual stream itself** (1536 fp16 values/token, drawn as a scrolling heatmap) leaving over the wire; a ticker of the **next-token candidates** the network weighed; a terminal-style **operation log** of every real step |
| **B · provider** (`/b`) | the one contributing | the same live internals for its half (`[14:28)`) — the residual stream **arriving** (bit-identical to what left A's screen), its layer-write bars, the **top-k lm_head probabilities it computes itself** (the head is weightless), every hop's **Ed25519-signed receipt** in the log, and the contribution tally (**layer·tokens**) ticking up |

Staged on two laptops side by side (A left, B right), the consumer's packets exit toward the right edge and the provider's enter from the left — and the two heatmaps draw the **same columns**, because they are the same bytes: the visual proof that two machines are running one forward pass.

**Every pixel is real.** The demo never edits the DRIFT sources and never simulates data: a stock DRIFT worker is instrumented at process start by monkey-patching (`TorchShardEngine.load/forward/head_argmax`, `Node.handle`, `Node._relay`) plus read-only PyTorch forward hooks on each kept decoder layer, all emitting fire-and-forget UDP events *out of band*. The math is untouched — the returned token is always the stock code's result (measured overhead ≈ 15–20 ms/token of display-only extraction). Receipt hashes on screen are the actual receipts the head verifies; the run journals them, so a demo run itself audits with `drift ledger`. Verified live: A's outgoing heatmap columns equal B's incoming ones **30/30 steps** — the fp16 wire round-trip is lossless, exactly as DRIFT's parity gate proves.

## What it looks like

| A · consumer | B · provider |
|---|---|
| ![view A](docs/view-a.png) | ![view B](docs/view-b.png) |

*(captured mid-generation on a live local run — the heatmap is the actual residual stream, the green bars are per-layer ‖Δh‖, the candidate list is the tail's own lm_head softmax, and the op log is one line per real recv/compute/sign/send)*

## Run it

Requires Python 3.12 and [`uv`](https://github.com/astral-sh/uv).

```bash
bash scripts/setup.sh          # vendors DRIFT into vendor/, builds .venv
.venv/bin/python -m demo       # spawns 2 local workers, opens /a and /b
```

Then type a prompt in view A. First launch loads the model shards (~10–60 s); the overlay lifts when the network is assembled.

```
http://127.0.0.1:8800/a   consumer
http://127.0.0.1:8800/b   provider
```

Options:

```
python -m demo --nodes 3            # more workers (view N: /b?node=2)
python -m demo --model <hf-id>      # any model DRIFT runs (default Qwen2.5-1.5B-Instruct)
python -m demo --max-new-tokens 400
python -m demo --no-browser --port 8800
```

## Audit a run

The head journals every verified receipt to `.state/journal-<ts>.jsonl`:

```bash
.venv/bin/drift ledger .state/journal-*.jsonl --verify
```

Two local workers sign with **distinct** Ed25519 identities (`.state/node{0,1}.identity`), so the tally shows two contributors even on one machine.

## How it fits together

```
demo/node_main.py     stock `drift node`, instrumented (no mDNS/gossip — local demo)
demo/instrument.py    the monkey-patches: per-layer ‖Δh‖ hooks, residual-stream
                      downsampling, top-k from the tail's own lm_head, step
                      arrival, compute timing, p2p relay
demo/head.py          weightless (thin) head + step-wise decode loop, events per token
demo/events.py        fire-and-forget UDP JSON emitter (out-of-band, non-blocking)
demo/server.py        stdlib HTTP: /a /b, SSE /events, POST /api/generate, /api/state
demo/__main__.py      launcher: spawn workers → assemble chain+thin head → serve
demo/static/          the two views (plain HTML/CSS/JS, no build step)
```

Topology per generated token (chain + thin head, M7/M10):

```
head ──ids──▶ n0 [0:14) ──hidden 3.1 KB──▶ n1 [14:28) ──token──▶ head
              └─ signs receipt              └─ signs receipt      └─ verifies both, live
```

DRIFT itself is vendored read-only under `vendor/DRIFT` (gitignored; pinned by `scripts/setup.sh`).
