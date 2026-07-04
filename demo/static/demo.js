/* DRIFT demo — shared view logic.
 *
 * Every element on screen is driven by real DRIFT traffic: the SSE stream at
 * /events carries out-of-band events emitted by the instrumented workers and
 * the head. Nothing here is simulated.
 *
 * The page sets window.VIEW = { role: 'consumer'|'provider', nodeIndex }.
 * Physical staging: view A on the left screen, view B on the right — the
 * consumer's wire exits rightward, the provider's enters from the left.
 */
(() => {
  const V = window.VIEW;
  const qi = new URLSearchParams(location.search).get('node');
  if (qi !== null) V.nodeIndex = parseInt(qi, 10);
  const $ = (id) => document.getElementById(id);

  // out/in packet directions: A faces B on its right, B faces A on its left.
  const DIR = V.role === 'consumer' ? { out: 'ltr', in: 'rtl' }
                                    : { out: 'rtl', in: 'ltr' };

  let plan = null, me = null, blocks = [];
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

  function setStatus(text, cls) {
    const d = $('status-dot'), t = $('status-text');
    if (d) d.className = 'dot ' + (cls || '');
    if (t) t.textContent = text;
  }

  // ---------------------------------------------------------------- stack
  function buildStack() {
    const stack = $('stack');
    stack.innerHTML = '';
    blocks = [];
    for (let i = me.start; i < me.end; i++) {
      const b = document.createElement('div');
      b.className = 'block';
      b.title = 'decoder layer ' + i;
      const s = document.createElement('span');
      s.textContent = i;
      b.appendChild(s);
      stack.appendChild(b);
      blocks.push(b);
    }
    $('node-title').textContent = `worker ${me.name} · layers [${me.start}:${me.end})`;
    $('node-meta').textContent =
      `${me.device || '?'} · ${me.host}:${me.port} · id ${me.pubkey}…`;
  }

  let waveTimer = [];
  function wave(ms) {
    waveTimer.forEach(clearTimeout);
    waveTimer = [];
    const span = Math.min(420, Math.max(160, (ms || 40) * 5));
    blocks.forEach((b, i) => {
      waveTimer.push(setTimeout(() => {
        b.classList.remove('on');
        void b.offsetWidth;
        b.classList.add('on');
        setTimeout(() => b.classList.remove('on'), 200);
      }, (i / Math.max(1, blocks.length)) * span));
    });
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
    setTimeout(() => p.remove(), 1500); // belt & braces
  }

  // ---------------------------------------------------------------- chat (A)
  function userBubble(text) {
    const log = $('chat-log');
    if (!log) return;
    const d = document.createElement('div');
    d.className = 'msg user';
    d.textContent = text;
    log.appendChild(d);
    log.scrollTop = log.scrollHeight;
  }

  function assistantBubble() {
    const log = $('chat-log');
    if (!log) return null;
    const d = document.createElement('div');
    d.className = 'msg assistant thinking';
    d.textContent = '';
    log.appendChild(d);
    log.scrollTop = log.scrollHeight;
    return d;
  }

  function setBusy(b) {
    busy = b;
    const input = $('prompt'), btn = $('send');
    if (input) input.disabled = b;
    if (btn) btn.disabled = b;
    if (!b && input) input.focus();
  }

  // ---------------------------------------------------------------- receipts (B)
  function receiptRow(e) {
    const list = $('receipts-list');
    if (!list) return;
    const row = document.createElement('div');
    row.className = 'rrow';
    const hops = e.hops.map((h) => {
      const mine = me && h.node === me.pubkey ? ' mine' : '';
      return `<span class="hop${mine}">[${h.start}:${h.end}) ` +
             `<span class="hash">${h.in}</span>→<span class="hash">${h.out}</span> ` +
             `<span class="sig">✓${h.sig.slice(0, 6)}</span></span>`;
    }).join('<span class="link">⛓</span>');
    row.innerHTML = `<span class="seq">#${e.seq}</span>` +
                    `<span class="mode">${e.mode}</span>${hops}`;
    list.prepend(row);
    while (list.children.length > 40) list.lastChild.remove();
  }

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
    buildStack();
    if ($('model-name')) $('model-name').textContent = model || '';
    if ($('earn-node')) $('earn-node').textContent = `${me.pubkey}…`;
    overlay(null);
    setStatus('network ready — ' + nodes.length + ' worker(s), weightless head', 'ok');
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
      assistantEl.textContent = s.last.text || '';
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
          $('stack').classList.add('busy');
          $('mode-label').textContent = `${e.mode} · ${e.n_pos} pos` +
            (e.embed ? ' · embed(thin entry)' : '');
        }
        break;

      case 'node.compute':
        if (isMe(e)) {
          $('stack').classList.remove('busy');
          wave(e.ms);
          $('compute-ms').textContent = `${e.ms} ms compute`;
        }
        break;

      case 'node.relay': {
        if (!me) break;
        const label = e.bytes > 0 ? (e.bytes / 1024).toFixed(1) + ' KB'
                                  : 'tok ' + (e.token ?? '');
        const kind = e.bytes > 0 ? 'tensor' : 'token';
        if (isMe(e)) {
          packet('out', label, kind);
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
        setStatus('generating…', 'live');
        break;

      case 'head.prefill_start':
        if (assistantEl) assistantEl.classList.add('thinking');
        setStatus(`prefill — ${e.seq_len} tokens through the chain…`, 'live');
        break;

      case 'head.prefill_end':
        setStatus('decoding…', 'live');
        break;

      case 'head.token':
        if (V.role === 'consumer') {
          if (!assistantEl) assistantEl = assistantBubble();
          assistantEl.classList.remove('thinking');
          assistantEl.textContent += e.text;
          const log = $('chat-log');
          if (log) log.scrollTop = log.scrollHeight;
        }
        if ($('stat-tps')) $('stat-tps').textContent = e.tps.toFixed(1);
        if ($('tok-count')) $('tok-count').textContent = e.i;
        break;

      case 'head.step':
        if ($('stat-ms')) $('stat-ms').textContent = e.ms.toFixed(0);
        break;

      case 'head.receipts':
        receiptRow(e);
        foldEarnings(e.hops || []);
        if ($('stat-verified')) $('stat-verified').textContent = e.checked + ' ✓';
        if ($('verify-badge')) {
          const bad = (e.suspects || []).length > 0;
          $('verify-badge').textContent = bad
            ? 'SUSPECT: ' + e.suspects.join(', ')
            : 'all hops verified ✓';
          $('verify-badge').className = 'badge ' + (bad ? 'bad' : 'good');
        }
        break;

      case 'head.recovering':
        setStatus('node dropped — re-splitting over survivors…', 'warn');
        break;

      case 'head.recovered':
        setStatus(`recovered (bitwise replay) — recovery #${e.recoveries}`, 'ok');
        break;

      case 'head.session_end':
        setBusy(false);
        assistantEl = null;
        setStatus(`done — ${e.tokens} tokens in ${e.seconds}s (${e.tps} tok/s)`, 'ok');
        break;

      case 'head.error':
        setBusy(false);
        if (assistantEl) {
          assistantEl.classList.remove('thinking');
          assistantEl.textContent += `\n[error] ${e.error}`;
          assistantEl = null;
        }
        if (!plan) overlay('failed to assemble: ' + e.error);
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
          assistantEl.textContent = '[' + err + ']';
          assistantEl = null;
          localPending = false;
          setBusy(false);
        }
      } catch (err) {
        assistantEl.textContent = '[unreachable: ' + err + ']';
        assistantEl = null;
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
