# DRIFT-Demo

**English** · [한국어](./README.ko.md)

**A two-screen visual demo of a [DRIFT](https://github.com/TaewoooPark/DRIFT) peer-to-peer inference run — the "For Tokens" economy, both faces at once.**

One model is split layer-by-layer across two DRIFT workers (peer-to-peer chain, weightless head), and each side of the exchange gets its own full-screen view. Every panel shows the network's **actual internals**, not an abstraction:

| view | who | what the screen shows |
|---|---|---|
| **A · consumer** (`/a`) | the one asking | a chat box; **‖Δh‖ per layer** — how much each of its decoder layers (`[0:14)`) actually rewrote this token's representation; the **residual stream itself** (1536 fp16 values/token, drawn as a scrolling heatmap) leaving over the wire; a ticker of the **next-token candidates** the network weighed; a terminal-style **operation log** of every real step |
| **B · provider** (`/b`) | the one contributing | the same live internals for its half (`[14:28)`) — the residual stream **arriving** (bit-identical to what left A's screen), its layer-write bars, the **top-k lm_head probabilities it computes itself** (the head is weightless), every hop's **Ed25519-signed receipt** in the log, and the contribution tally (**layer·tokens**) ticking up |

Staged on two laptops side by side (A left, B right), the consumer's packets exit toward the right edge and the provider's enter from the left — and the two heatmaps draw the **same columns**, because they are the same bytes: the visual proof that two machines are running one forward pass.

**Every pixel is real.** The demo never edits the DRIFT sources and never simulates data: a stock DRIFT worker is instrumented at process start by monkey-patching (`TorchShardEngine.load/forward/head_argmax`, `Node.handle`, `Node._relay`) plus read-only PyTorch forward hooks on each kept decoder layer, all emitting fire-and-forget UDP events *out of band*. The math is untouched — the next token comes from one `lm_head` pass written as the stock expression, and the demo's greedy output is verified token-identical to the stock path; display-only extraction costs roughly 10–20 ms/token depending on the machine (`DRIFT_DEMO_TOPK=0` disables the top-k tap). Receipt hashes on screen are the actual receipts the head verifies; the run journals them, so a demo run itself audits with `drift ledger`. Verified live: A's outgoing heatmap columns equal B's incoming ones **30/30 steps** — the fp16 wire round-trip is lossless, exactly as DRIFT's parity gate proves.

## Reading the screens

Both figures below were captured on a live local run; the numbers match the badges in the images.

### A · consumer — `/a`

![view A, annotated](docs/view-a.png)

1. **Transcript** — the conversation, in the same grammar as the real `drift run` REPL (`you ›` / `drift ›`). The reply lands token-by-token, one line per completed round trip through the chain.
2. **Candidates ticker** — for the latest step, the next-token candidates the network weighed, with their real probabilities (the tail node's own `lm_head` softmax).
3. **Prompt** — type here; submitting drives the actual orchestrator via `POST /api/generate`.
4. **‖Δh‖ per layer** — this machine's decoder layers `[0:14)`. Bar height = how much that layer just rewrote the current token's hidden representation (measured by read-only forward hooks; the segmented meters snap to each new token).
5. **The residual stream, leaving** — the hidden state that actually crosses the wire: the last position's 1536 fp16 values, downsampled to 128 mean-|activation| buckets, one column per token, rendered 1-bit (Bayer ordered dithering — white-pixel density *is* the magnitude). Node B receives these exact bytes.
6. **The wire** — a filled packet is ~3.0 KB of hidden state leaving for node B; an outlined packet is the single token id coming home. That asymmetry *is* the weightless-head design: tensors flow between nodes, only integers touch the head.
7. **Operation log** — one line per real step on this machine: `<<` recv (bytes off the wire), `::` compute (layer range + ms), `>>` send, `OK` the head's live verification of the full Ed25519 receipt chain (hash prefixes shown).
8. **Session stats** — tok/s, ms/token, wire bytes/token, receipts verified so far, model.

### B · provider — `/b`

![view B, annotated](docs/view-b.png)

1. **The residual stream, arriving** — the same columns as A's outgoing panel, because they are the same bytes (verified 30/30 steps identical in testing): two screens, one forward pass.
2. **‖Δh‖ per layer** — this machine's half, layers `[14:28)`; same live meaning as A's meter.
3. **The wire** — hidden state in from node A; one token id back out toward the head.
4. **Operation log** — this machine's steps, including the `#` sign lines: the Ed25519 receipt this node signs for every hop it computes (`in`/`out` hashes + signature prefix — the same receipts the ledger settles on).
5. **Contribution** — the ledger tally: **layer·tokens** (layers held × tokens carried, M13's settlement unit), tokens carried, sessions served, and this node's identity key.
6. **Verification status** — stays `ALL HOPS VERIFIED` while the head's per-token checks pass; flips to an inverse, blinking `SUSPECT …` the moment any signature / hash-adjacency / anchor check fails.
7. **Next-token candidates** — real probabilities from the `lm_head` **this node runs** (the head holds zero weights): `>` marks the token actually chosen, the `█░` meters are the softmax.
8. **Session stats** — tokens this run, ms/token, wire/token, receipts, model.

## Run it

Requires Python 3.12 and [`uv`](https://github.com/astral-sh/uv).

```bash
bash scripts/setup.sh          # vendors DRIFT into vendor/, builds .venv
.venv/bin/python -m demo       # spawns 2 local workers, opens /a and /b
```

Then type a prompt in view A. First launch loads the model shards (~10–60 s); on a cold machine the model is first downloaded to the Hugging Face cache (~3 GB for the default Qwen). The overlay lifts when the network is assembled.

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

## Models

The demo hardcodes **nothing** about the model: the ‖Δh‖ hooks iterate whatever `engine.layers` holds, the heatmap adaptively pools *any* hidden size into 128 buckets, and the top-k tap uses the introspected `lm_head` + the model's own tokenizer. So any model DRIFT itself runs works here unchanged:

```bash
python -m demo --model Qwen/Qwen2.5-7B-Instruct
python -m demo --model google/gemma-4-E2B-it
```

The constraints are DRIFT's, not the demo's: a decoder-only Hugging Face causal LM whose architecture the installed `transformers` supports, with fp16 weights that fit across the workers' combined memory. The layer panels, wire sizes, and split points all re-derive themselves from the loaded model.

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
tests/                torch-free smoke tests: state fold, exactly-once attach,
                      API validation (`python -m unittest discover -s tests`)
```

Topology per generated token (chain + thin head, M7/M10):

```
head ──ids──▶ n0 [0:14) ──hidden 3.1 KB──▶ n1 [14:28) ──token──▶ head
              └─ signs receipt              └─ signs receipt      └─ verifies both, live
```

**Failover, through the demo loop.** Kill a worker mid-generation and the session survives (M9): the head re-splits over the survivors, replays, and re-broadcasts the plan so the views rebuild their panels — verified live with a `SIGKILL` at token ~30: the finished text was **bitwise-identical** to an uninterrupted run, and the ledger records the story (the survivor's ranges widen to `[0:14),[0:28)` and its share rises accordingly).

DRIFT itself is vendored read-only under `vendor/DRIFT` — gitignored, and pinned to DRIFT **`v1.0.0`** by `scripts/setup.sh` (the demo hooks drift internals, so upgrades are deliberate: `rm -rf vendor/DRIFT && DRIFT_REF=<tag> bash scripts/setup.sh`).
