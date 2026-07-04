"""DRIFT-Demo — a two-screen visual demo of a DRIFT peer-to-peer inference run.

One machine (or one `--local` run) plays both sides of the "For Tokens" economy:

  * view A (`/a`, consumer) — a prompt goes in, tokens stream out, and the
    screen shows the front half of the model computing + the hidden state
    leaving over the wire.
  * view B (`/b`, provider) — the back half of the model computing, every hop's
    signed receipt landing, and the contribution tally (the ledger's input)
    ticking up.

Every pixel is driven by real DRIFT traffic: the demo instruments a stock DRIFT
worker via monkey-patching (the DRIFT sources are never edited) and emits
out-of-band UDP events that the browser views consume over SSE.
"""
