/* DRIFT demo — shared view logic (v2: real internals, no abstract boxes).
 *
 * Every panel is driven by real DRIFT traffic over SSE (/events):
 *   · layer bars   — ‖Δh‖ per decoder layer: how much each layer actually
 *                    rewrote this token's representation (forward hooks)
 *   · heatmap      — the residual stream itself, 1536 fp16 values per token
 *                    downsampled to 128 buckets; A's outgoing columns and B's
 *                    incoming columns are the SAME bytes
 *   · top-k        — the tail node's own lm_head probabilities per step
 *   · op log       — one line per real operation: recv / compute / sign / send
 * Nothing is simulated.
 *
 * The page sets window.VIEW = { role: 'consumer'|'provider', nodeIndex }.
 */
(() => {
  const V = window.VIEW;
  const qi = new URLSearchParams(location.search).get('node');
  if (qi !== null) V.nodeIndex = parseInt(qi, 10);
  const $ = (id) => document.getElementById(id);

  // out/in packet directions: A faces B on its right, B faces A on its left.
  const DIR = V.role === 'consumer' ? { out: 'ltr', in: 'rtl' }
                                    : { out: 'rtl', in: 'ltr' };

  let plan = null, me = null;
  let barFills = [], barVals = [];
  let earnings = { tokens: 0, layer_tokens: 0 };
  let assistantEl = null;
  let busy = false;
  let localPending = false;   // this client submitted the prompt (bubbles exist)

  // The head's plan names workers n0/n1…, but a worker stamps its own events
  // with its self-chosen name (node-<port>). Match on either.
  function isMe(e) {
    return !!me && (e.node === me.name || e.node === 'node-' + me.port);
  }

  // ---------------------------------------------------------------- overlay
  function overlay(text) {
    const ov = $('overlay');
    if (!ov) return;
    if (text === null) { ov.classList.add('hide'); return; }
    ov.classList.remove('hide');
    $('ov-text').textContent = text;
  }

  const MARK = { ok: '[OK]', live: '[RUN]', warn: '[..]', bad: '[ERR]', '': '[--]' };
  function setStatus(text, cls) {
    const d = $('status-dot'), t = $('status-text');
    if (d) {
      d.className = 'mark ' + (cls || '');
      d.textContent = MARK[cls || ''] || '[--]';
    }
    if (t) t.textContent = text;
  }

  // ---------------------------------------------------------------- op log
  const GLYPH = { recv: '<<', compute: '::', send: '>>', sign: '#', verify: 'OK',
                  token: '>', topk: '%', sys: '--', err: '!!' };
  function log(kind, text, ts) {
    const el = $('oplog');
    if (!el) return;
    const line = document.createElement('div');
    line.className = 'll k-' + kind;
    const t = ts ? new Date(ts * 1000) : new Date();
    const stamp = String(t.getMinutes()).padStart(2, '0') + ':' +
                  String(t.getSeconds()).padStart(2, '0') + '.' +
                  String(t.getMilliseconds()).padStart(3, '0');
    line.innerHTML = `<span class="lt">${stamp}</span>` +
                     `<span class="lg">${GLYPH[kind] || '·'}</span>` +
                     `<span class="lx">${text}</span>`;
    el.appendChild(line);
    while (el.children.length > 250) el.firstChild.remove();
    el.scrollTop = el.scrollHeight;
  }
  const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;');
  const fmtB = (n) => n >= 1024 ? (n / 1024).toFixed(1) + ' KB' : n + ' B';

  // ---------------------------------------------------------------- layer bars
  function buildBars() {
    const bars = $('bars');
    bars.innerHTML = '';
    barFills = [];
    for (let i = me.start; i < me.end; i++) {
      const b = document.createElement('div');
      b.className = 'bar';
      const f = document.createElement('div');
      f.className = 'fill';
      const s = document.createElement('span');
      s.textContent = i;
      b.appendChild(f);
      b.appendChild(s);
      bars.appendChild(b);
      barFills.push(f);
    }
    $('node-title').textContent = `worker ${me.name} · decoder layers [${me.start}:${me.end})`;
    $('node-meta').textContent =
      `${me.device || '?'} · ${me.host}:${me.port} · id ${me.pubkey}…`;
  }

  function updateBars(deltas) {
    if (!deltas || !deltas.length || !barFills.length) return;
    barVals = deltas;
    const max = Math.max(...deltas, 1e-6);
    for (let i = 0; i < barFills.length && i < deltas.length; i++) {
      barFills[i].style.height = Math.round((deltas[i] / max) * 100) + '%';
      barFills[i].parentNode.title =
        `layer ${me.start + i} — ‖Δh‖ = ${deltas[i]}`;
    }
    if ($('bars-max')) $('bars-max').textContent = 'max ‖Δh‖ ' + max.toFixed(1);
  }

  // ---------------------------------------------------------------- heatmap
  let heatCtx = null, heatCanvas = null;
  const COLW = 6;
  function heatInit() {
    heatCanvas = $('heat');
    if (!heatCanvas) return;
    heatCanvas.width = 840;
    heatCanvas.height = 128;
    heatCtx = heatCanvas.getContext('2d');
    heatCtx.fillStyle = '#000000';
    heatCtx.fillRect(0, 0, heatCanvas.width, heatCanvas.height);
  }
  function heatColor(v) {
    // straight grayscale (mild gamma so faint channels stay visible)
    const g = Math.round(Math.pow(v / 255, 0.85) * 232);
    return `rgb(${g},${g},${g})`;
  }
  function heatPush(col) {
    if (!heatCtx || !col) return;
    const w = heatCanvas.width, h = heatCanvas.height;
    heatCtx.drawImage(heatCanvas, -COLW, 0);
    const cellH = h / col.length;
    for (let i = 0; i < col.length; i++) {
      heatCtx.fillStyle = heatColor(col[i]);
      heatCtx.fillRect(w - COLW, i * cellH, COLW, Math.ceil(cellH));
    }
  }
  heatInit();

  // ---------------------------------------------------------------- top-k
  const TBAR_W = 24;
  function topkRender(cand, chosen) {
    const box = $('topk');
    if (!box) return;
    box.innerHTML = '';
    cand.forEach(([txt, p], idx) => {
      const row = document.createElement('div');
      row.className = 'trow' + (idx === 0 ? ' chosen' : '');
      const filled = Math.max(idx === 0 ? 1 : 0, Math.round(p * TBAR_W));
      const bar = '█'.repeat(filled) + '░'.repeat(TBAR_W - filled);
      row.innerHTML =
        `<span class="ttok">${idx === 0 ? '&gt; ' : '  '}${esc(JSON.stringify(txt))}</span>` +
        `<span class="tbar-txt">${bar}</span>` +
        `<span class="tp">${(p * 100).toFixed(1)}%</span>`;
      box.appendChild(row);
    });
  }
  function candTicker(cand) {
    const el = $('cand');
    if (!el) return;
    el.textContent = 'weighed: ' + cand.slice(0, 5)
      .map(([t, p]) => `${JSON.stringify(t)} ${(p * 100).toFixed(0)}%`).join(' | ');
  }

  // ---------------------------------------------------------------- packets
  function packet(dirKey, label, kind) {
    const lane = $('lane');
    if (!lane) return;
    const p = document.createElement('div');
    p.className = `packet fly-${DIR[dirKey]}` + (kind ? ' ' + kind : '');
    p.textContent = label;
    lane.appendChild(p);
    p.addEventListener('animationend', () => p.remove());
    setTimeout(() => p.remove(), 1500);
  }

  // ---------------------------------------------------------------- chat (A)
  // transcript lines, same grammar as the real `drift run` REPL
  function transcriptLine(cls, pfx) {
    const logEl = $('chat-log');
    if (!logEl) return null;
    const d = document.createElement('div');
    d.className = 'msg ' + cls;
    const p = document.createElement('span');
    p.className = 'pfx';
    p.textContent = pfx;
    const tx = document.createElement('span');
    tx.className = 'tx';
    d.appendChild(p);
    d.appendChild(tx);
    logEl.appendChild(d);
    logEl.scrollTop = logEl.scrollHeight;
    return d;
  }

  function userBubble(text) {
    const d = transcriptLine('user', 'you ›');
    if (d) d.lastChild.textContent = text;
  }

  function assistantBubble() {
    return transcriptLine('assistant thinking', 'drift ›');
  }

  function setBusy(b) {
    busy = b;
    const input = $('prompt'), btn = $('send');
    if (input) input.disabled = b;
    if (btn) btn.disabled = b;
    if (!b && input) input.focus();
  }

  // ---------------------------------------------------------------- earnings (B)
  function renderEarnings() {
    if ($('earn-lt')) {
      $('earn-lt').textContent = earnings.layer_tokens.toLocaleString();
      $('earn-tokens').textContent = earnings.tokens.toLocaleString();
    }
  }

  function foldEarnings(hops) {
    if (!me) return;
    let changed = false;
    hops.forEach((h) => {
      if (h.node === me.pubkey) {
        earnings.tokens += 1;
        earnings.layer_tokens += (h.end - h.start);
        changed = true;
      }
    });
    if (changed) renderEarnings();
  }

  // ---------------------------------------------------------------- plan/state
  function applyPlan(nodes, model) {
    plan = nodes;
    me = nodes[Math.min(V.nodeIndex, nodes.length - 1)];
    buildBars();
    if ($('model-name')) $('model-name').textContent = model || '';
    if ($('earn-node')) $('earn-node').textContent = `${me.pubkey}…`;
    overlay(null);
    setStatus('network ready — ' + nodes.length + ' worker(s), weightless head', 'ok');
    log('sys', `assigned decoder layers [${me.start}:${me.end}) of ${model}`);
  }

  function applyState(s) {
    if (s.plan) {
      applyPlan(s.plan, s.model);
    } else {
      overlay('assembling — workers are loading their layer slices…');
    }
    if (me && s.ledger && s.ledger[me.pubkey]) {
      earnings = { ...s.ledger[me.pubkey] };
      renderEarnings();
    }
    if ($('earn-sessions')) $('earn-sessions').textContent = s.sessions || 0;
    if ($('stat-verified')) $('stat-verified').textContent = (s.receipts_checked || 0) + ' ✓';
    if (V.role === 'consumer' && s.last && s.last.prompt) {
      userBubble(s.last.prompt);
      assistantEl = assistantBubble();
      assistantEl.classList.remove('thinking');
      assistantEl.lastChild.textContent = s.last.text || '';
    }
    setBusy(!!s.busy);
  }

  // ---------------------------------------------------------------- events
  function handle(e) {
    switch (e.t) {
      case 'hello':
        applyState(e.state);
        break;

      case 'head.building':
        overlay(`assembling — splitting ${e.model} across ${e.nodes} workers…`);
        break;

      case 'node.up': {
        const ov = $('ov-text');
        if (ov && !plan) ov.textContent += `\nworker :${e.port} up (${e.device})`;
        break;
      }

      case 'head.plan':
        applyPlan(e.nodes, e.model);
        break;

      case 'node.step':
        if (isMe(e)) {
          if (e.embed) {
            log('recv', `${e.mode} #${e.seq} · ${e.n_pos} token id(s) — thin entry, ` +
                        `this node embeds them into 1536-d itself`, e.ts);
          } else {
            log('recv', `${e.mode} #${e.seq} · hidden state ${fmtB(e.in_bytes)} ` +
                        `(fp16) off the wire`, e.ts);
          }
        }
        break;

      case 'node.compute':
        if (isMe(e)) {
          updateBars(e.layers);
          heatPush(V.role === 'consumer' ? e.hout : e.hin);
          $('compute-ms').textContent = `${e.ms} ms`;
          $('mode-label').textContent = `${e.mode} · ${e.n_pos} pos`;
          log('compute', `layers ${e.start}–${e.end - 1} forward · attn+mlp ×` +
                         `${e.end - e.start} · ${e.ms} ms @ ${me.device}`, e.ts);
        }
        break;

      case 'node.relay': {
        if (!me) break;
        const label = e.bytes > 0 ? (e.bytes / 1024).toFixed(1) + ' KB'
                                  : 'tok ' + (e.token ?? '');
        const kind = e.bytes > 0 ? 'tensor' : 'token';
        if (isMe(e)) {
          packet('out', label, kind);
          log('send', e.tail
            ? `token id ${e.token} → head collect ${e.to[0]}:${e.to[1]} ` +
              `(only an integer goes back)`
            : `hidden state ${fmtB(e.bytes)} → next node ${e.to[0]}:${e.to[1]}`, e.ts);
        } else if (e.to && e.to[1] === me.port) {
          packet('in', label, kind);
        } else if (V.role === 'consumer' && e.tail) {
          packet('in', label, kind); // the final token coming home to the head
        }
        if (e.bytes > 0 && $('stat-kb')) {
          $('stat-kb').textContent = (e.bytes / 1024).toFixed(2) + ' KB';
        }
        break;
      }

      case 'node.topk':
        topkRender(e.cand, e.chosen);
        candTicker(e.cand);
        if (V.role === 'provider') {
          const top = e.cand.slice(0, 3)
            .map(([t, p]) => `${JSON.stringify(t)} ${(p * 100).toFixed(0)}%`).join(' · ');
          log('topk', `lm_head on THIS node → ${top}`, e.ts);
        }
        break;

      case 'head.session_start':
        setBusy(true);
        if ($('earn-sessions')) {
          $('earn-sessions').textContent = (parseInt($('earn-sessions').textContent) || 0) + 1;
        }
        if (V.role === 'consumer' && !localPending) {
          // a session someone else started — give it its own bubble pair
          userBubble(e.prompt);
          assistantEl = assistantBubble();
        }
        localPending = false;
        log('sys', `session ${e.session} · prompt ${e.prompt_tokens} tokens`, e.ts);
        setStatus('generating…', 'live');
        break;

      case 'head.prefill_start':
        if (assistantEl) assistantEl.classList.add('thinking');
        setStatus(`prefill — ${e.seq_len} tokens through the chain…`, 'live');
        break;

      case 'head.prefill_end':
        log('sys', `prefill done in ${e.ms} ms — every node's KV cache is built`, e.ts);
        setStatus('decoding…', 'live');
        break;

      case 'head.token':
        if (V.role === 'consumer') {
          if (!assistantEl) assistantEl = assistantBubble();
          assistantEl.classList.remove('thinking');
          assistantEl.lastChild.textContent += e.text;
          const logEl = $('chat-log');
          if (logEl) logEl.scrollTop = logEl.scrollHeight;
          log('token', `#${e.i} ${JSON.stringify(e.text)} · ${e.tps} tok/s`, e.ts);
        }
        if ($('stat-tps')) $('stat-tps').textContent = e.tps.toFixed(1);
        if ($('tok-count')) $('tok-count').textContent = e.i;
        break;

      case 'head.step':
        if ($('stat-ms')) $('stat-ms').textContent = e.ms.toFixed(0);
        break;

      case 'head.receipts': {
        foldEarnings(e.hops || []);
        if ($('stat-verified')) $('stat-verified').textContent = e.checked + ' ✓';
        if (V.role === 'provider' && me) {
          const mine = (e.hops || []).find((h) => h.node === me.pubkey);
          if (mine) {
            log('sign', `receipt [${mine.start}:${mine.end}) in ${mine.in}… ` +
                        `out ${mine.out}… sig ${mine.sig}… (Ed25519)`, e.ts);
          }
        } else if (V.role === 'consumer') {
          const chain = (e.hops || []).map((h) => h.out.slice(0, 6)).join('→');
          log('verify', `${(e.hops || []).length} receipts · hash chain ` +
                        `${chain} · ${(e.suspects || []).length} suspects`, e.ts);
        }
        if ($('verify-badge')) {
          const bad = (e.suspects || []).length > 0;
          $('verify-badge').textContent = bad
            ? 'SUSPECT: ' + e.suspects.join(', ')
            : 'all hops verified ✓';
          $('verify-badge').className = 'badge ' + (bad ? 'bad' : 'good');
        }
        break;
      }

      case 'head.recovering':
        log('err', 'a node dropped — re-splitting over survivors…', e.ts);
        setStatus('node dropped — re-splitting over survivors…', 'warn');
        break;

      case 'head.recovered':
        log('sys', `recovered — bitwise replay, recovery #${e.recoveries}`, e.ts);
        setStatus(`recovered (bitwise replay) — recovery #${e.recoveries}`, 'ok');
        break;

      case 'head.session_end':
        setBusy(false);
        assistantEl = null;
        log('sys', `session done — ${e.tokens} tokens · ${e.seconds}s · ${e.tps} tok/s`, e.ts);
        setStatus(`done — ${e.tokens} tokens in ${e.seconds}s (${e.tps} tok/s)`, 'ok');
        break;

      case 'head.error':
        setBusy(false);
        if (assistantEl) {
          assistantEl.classList.remove('thinking');
          assistantEl.lastChild.textContent += `\n[error] ${e.error}`;
          assistantEl = null;
        }
        if (!plan) overlay('failed to assemble: ' + e.error);
        log('err', e.error, e.ts);
        setStatus('error — ' + e.error, 'bad');
        break;
    }
  }

  // ---------------------------------------------------------------- chat form
  const form = $('chat-form');
  if (form) {
    form.addEventListener('submit', async (ev) => {
      ev.preventDefault();
      const input = $('prompt');
      const text = input.value.trim();
      if (!text || busy) return;
      input.value = '';
      userBubble(text);
      assistantEl = assistantBubble();
      localPending = true;
      setBusy(true);
      try {
        const r = await fetch('/api/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt: text }),
        });
        if (!r.ok) {
          const err = (await r.json()).error || r.statusText;
          assistantEl.classList.remove('thinking');
          assistantEl.lastChild.textContent = '[' + err + ']';
          assistantEl = null;
          localPending = false;
          setBusy(false);
        }
      } catch (err) {
        assistantEl.classList.remove('thinking');
        assistantEl.lastChild.textContent = '[unreachable: ' + err + ']';
        assistantEl = null;
        localPending = false;
        setBusy(false);
      }
    });
  }

  // ---------------------------------------------------------------- SSE
  function connect() {
    const es = new EventSource('/events');
    es.onmessage = (m) => {
      try { handle(JSON.parse(m.data)); } catch (e) { /* keep streaming */ }
    };
    es.onerror = () => setStatus('reconnecting…', 'warn');
    es.onopen = () => { if (!plan) setStatus('connected — waiting for the network…', ''); };
  }
  connect();
})();
