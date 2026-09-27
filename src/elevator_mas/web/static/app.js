/* Smart Elevator Fleet Coordinator — dashboard.
 *
 * Plain ES modules-free JS, no build step. The server streams one snapshot per simulated
 * tick over the WebSocket; this file interpolates between snapshots with
 * requestAnimationFrame so a 1 Hz simulation animates smoothly.
 */
'use strict';

const API = {
  async post(path, body) {
    const res = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {}),
    });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
    return res.json();
  },
  async get(path) {
    const res = await fetch(path);
    if (!res.ok) throw new Error(res.statusText);
    return res.json();
  },
};

/* ------------------------------------------------------------------ state */

const S = {
  snapshot: null,     // latest frame from the server
  previous: null,     // the frame before it, for interpolation
  frameTime: 0,       // when the latest frame arrived (ms)
  tickMs: 100,        // estimated wall-clock ms between ticks
  meta: null,
  selected: null,     // inspected agent address
  charts: {},
  socket: null,
  awt: { labels: [], wait: [], p95: [] },
};

const CAR_COLOURS = ['#58a6ff', '#f0883e', '#3fb950', '#bc8cff', '#39c5cf', '#db61a2', '#d29922', '#7ee787'];
const carColour = (id) => CAR_COLOURS[id % CAR_COLOURS.length];
const $ = (id) => document.getElementById(id);
const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? '—' : Number(v).toFixed(d));

/* -------------------------------------------------------------- websocket */

function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const socket = new WebSocket(`${proto}://${location.host}/ws`);
  S.socket = socket;

  socket.onmessage = (event) => {
    const frame = JSON.parse(event.data);
    if (S.snapshot && frame.tick > S.snapshot.tick) {
      // Track real inter-frame time so interpolation matches the actual speed.
      const dt = performance.now() - S.frameTime;
      if (dt > 8 && dt < 4000) S.tickMs = S.tickMs * 0.7 + dt * 0.3;
      S.previous = S.snapshot;
    } else if (!S.snapshot || frame.tick < S.snapshot.tick) {
      S.previous = null; // a reset or a rewind: do not interpolate across it
    }
    S.snapshot = frame;
    S.frameTime = performance.now();
    onFrame(frame);
  };
  socket.onclose = () => {
    setStatus('disconnected', 'paused');
    setTimeout(connect, 1200);
  };
}

function onFrame(frame) {
  renderKpis(frame);
  renderAuction(frame);
  renderMessages(frame);
  renderRules(frame);
  renderTraffic(frame);
  renderControls(frame);
  pushAwt(frame);
  if (S.selected) refreshInspector();
}

/* ------------------------------------------------------------- the building */

const canvas = $('building');
const ctx = canvas.getContext('2d');

function interpolatedCars() {
  const frame = S.snapshot;
  if (!frame) return [];
  const previous = S.previous;
  const running = frame.session && frame.session.running;
  // Fraction of the way from the previous frame to this one.
  const alpha = running && previous ? Math.min(1, (performance.now() - S.frameTime) / S.tickMs) : 1;

  return frame.cars.map((car) => {
    const before = previous && previous.cars.find((c) => c.car_id === car.car_id);
    let position = car.floor + car.progress * (car.direction === 'UP' ? 1 : car.direction === 'DOWN' ? -1 : 0);
    if (before) {
      const from = before.floor + before.progress * (before.direction === 'UP' ? 1 : before.direction === 'DOWN' ? -1 : 0);
      // Only interpolate short hops; a large jump means a reset or a teleport.
      if (Math.abs(position - from) <= 2.2) position = from + (position - from) * alpha;
    }
    return { ...car, y: position };
  });
}

function draw() {
  requestAnimationFrame(draw);
  const frame = S.snapshot;
  if (!frame) return;

  const floors = frame.building.floors;
  const cars = frame.building.cars;

  // Size the canvas for the device pixel ratio so text stays crisp.
  const dpr = window.devicePixelRatio || 1;
  const cssWidth = canvas.clientWidth || 700;
  const rowHeight = Math.max(22, Math.min(46, 620 / floors));
  const cssHeight = Math.round(floors * rowHeight + 46);
  if (canvas.width !== Math.round(cssWidth * dpr) || canvas.height !== Math.round(cssHeight * dpr)) {
    canvas.width = Math.round(cssWidth * dpr);
    canvas.height = Math.round(cssHeight * dpr);
    canvas.style.height = `${cssHeight}px`;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssWidth, cssHeight);

  const labelWidth = 48;
  const waitWidth = 84;
  const shaftArea = cssWidth - labelWidth - waitWidth - 26;
  const shaftWidth = Math.max(26, Math.min(74, shaftArea / cars - 8));
  const gap = cars > 1 ? (shaftArea - shaftWidth * cars) / (cars - 1 + 0.0001) : 0;
  const top = 34;
  const yOf = (floor) => top + (floors - 1 - floor) * rowHeight;

  // --- floor rows, labels, hall calls and waiting crowds ---
  ctx.font = '11px ui-monospace, Menlo, monospace';
  for (let floor = 0; floor < floors; floor += 1) {
    const y = yOf(floor);
    const floorData = frame.floors[floor] || {};
    ctx.fillStyle = floor % 2 === 0 ? 'rgba(255,255,255,0.016)' : 'transparent';
    ctx.fillRect(labelWidth - 6, y, cssWidth - labelWidth - 6, rowHeight);
    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.beginPath();
    ctx.moveTo(labelWidth - 6, y + rowHeight);
    ctx.lineTo(cssWidth - 8, y + rowHeight);
    ctx.stroke();

    const isLobby = floor === frame.building.lobby;
    ctx.fillStyle = isLobby ? '#58a6ff' : '#8b949e';
    ctx.textAlign = 'right';
    ctx.fillText(isLobby ? 'LOBBY' : String(floor), labelWidth - 12, y + rowHeight / 2 + 4);

    // Hall-call arrows, coloured by the car that won the call.
    const cx = labelWidth + 4;
    drawArrow(cx, y + rowHeight / 2, true, floorData.up, floorData.assigned_up);
    drawArrow(cx + 14, y + rowHeight / 2, false, floorData.down, floorData.assigned_down);

    // Waiting crowd, with the longest wait shown once it gets uncomfortable.
    const waiting = (floorData.waiting_up || 0) + (floorData.waiting_down || 0);
    if (waiting > 0) {
      const x0 = cssWidth - waitWidth + 4;
      const shown = Math.min(waiting, 9);
      for (let i = 0; i < shown; i += 1) {
        const oldest = floorData.oldest_wait || 0;
        ctx.fillStyle = oldest > 60 ? '#f85149' : oldest > 30 ? '#d29922' : '#7ee787';
        ctx.beginPath();
        ctx.arc(x0 + i * 7.5, y + rowHeight / 2, 2.7, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.fillStyle = '#6b7784';
      ctx.textAlign = 'left';
      if (waiting > 9) ctx.fillText(`+${waiting - 9}`, x0 + shown * 7.5 + 2, y + rowHeight / 2 + 4);
      if (floorData.oldest_wait > 45) {
        ctx.fillStyle = floorData.oldest_wait > 60 ? '#f85149' : '#d29922';
        ctx.fillText(`${floorData.oldest_wait}s`, x0 + 58, y + rowHeight / 2 + 4);
      }
    }
  }

  // --- shafts ---
  for (let car = 0; car < cars; car += 1) {
    const x = labelWidth + 30 + car * (shaftWidth + gap);
    ctx.fillStyle = 'rgba(255,255,255,0.022)';
    ctx.fillRect(x, top, shaftWidth, floors * rowHeight);
    ctx.strokeStyle = 'rgba(255,255,255,0.07)';
    ctx.strokeRect(x, top, shaftWidth, floors * rowHeight);
    ctx.fillStyle = carColour(car);
    ctx.textAlign = 'center';
    ctx.font = '600 11px ui-monospace, Menlo, monospace';
    ctx.fillText(`C${car}`, x + shaftWidth / 2, top - 18);
  }

  // --- cars ---
  const carHeight = Math.max(15, rowHeight - 7);
  interpolatedCars().forEach((car) => {
    const x = labelWidth + 30 + car.car_id * (shaftWidth + gap);
    const y = top + (floors - 1 - car.y) * rowHeight + (rowHeight - carHeight) / 2;
    const colour = carColour(car.car_id);

    let fill = colour;
    if (car.out_of_service) fill = '#6e2620';
    else if (car.fire_mode) fill = '#f85149';

    ctx.globalAlpha = car.out_of_service ? 0.55 : 1;
    roundRect(x + 2, y, shaftWidth - 4, carHeight, 3);
    ctx.fillStyle = fill;
    ctx.fill();

    // Load bar along the bottom of the car.
    const loadFraction = car.capacity ? car.load / car.capacity : 0;
    if (loadFraction > 0) {
      ctx.fillStyle = loadFraction > 0.85 ? '#f85149' : 'rgba(255,255,255,0.75)';
      ctx.fillRect(x + 3, y + carHeight - 4, (shaftWidth - 6) * Math.min(1, loadFraction), 2.6);
    }

    // Doors: a light gap that widens as they open.
    const doorOpen = car.door === 'open' ? 1 : car.door === 'opening' || car.door === 'closing' ? 0.5 : 0;
    if (doorOpen > 0) {
      ctx.fillStyle = 'rgba(13,17,23,0.82)';
      const width = (shaftWidth - 8) * doorOpen * 0.5;
      ctx.fillRect(x + (shaftWidth - width) / 2, y + 1.5, width, carHeight - 6);
    }

    ctx.globalAlpha = 1;
    ctx.fillStyle = loadFraction > 0.55 ? '#0d1117' : '#e6edf3';
    ctx.font = '600 10.5px ui-monospace, Menlo, monospace';
    ctx.textAlign = 'center';
    const label = car.out_of_service ? 'OOS' : car.fire_mode ? 'FIRE' : String(car.load);
    ctx.fillText(label, x + shaftWidth / 2, y + carHeight / 2 + 3.5);

    // Direction tick above the car.
    if (car.direction !== 'IDLE' && !car.out_of_service) {
      ctx.fillStyle = colour;
      ctx.fillText(car.direction === 'UP' ? '▲' : '▼', x + shaftWidth / 2, y - 2);
    }
  });

  // --- fire banner ---
  if (frame.fire_alarm) {
    ctx.fillStyle = 'rgba(248,81,73,0.13)';
    ctx.fillRect(0, 0, cssWidth, cssHeight);
    ctx.fillStyle = '#f85149';
    ctx.font = '600 13px -apple-system, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('FIRE ALARM — all cars recalled to lobby, hall calls blocked', cssWidth / 2, 18);
  }
}

function drawArrow(x, y, up, active, assignedCar) {
  const r = 5.5;
  ctx.beginPath();
  if (up) { ctx.moveTo(x, y - r); ctx.lineTo(x + r, y + r * 0.8); ctx.lineTo(x - r, y + r * 0.8); }
  else { ctx.moveTo(x, y + r); ctx.lineTo(x + r, y - r * 0.8); ctx.lineTo(x - r, y - r * 0.8); }
  ctx.closePath();
  if (!active) {
    ctx.fillStyle = 'rgba(255,255,255,0.13)';
    ctx.fill();
    return;
  }
  // An assigned call glows in its car's colour, so "who is coming for me" is readable
  // at a glance; an unassigned one stays white and pulsing-bright to stand out.
  const colour = (assignedCar === null || assignedCar === undefined) ? '#e6edf3' : carColour(assignedCar);
  ctx.shadowColor = colour;
  ctx.shadowBlur = 7;
  ctx.fillStyle = colour;
  ctx.fill();
  ctx.shadowBlur = 0;
}

function roundRect(x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

/* Clicking the building: a floor row adds a passenger, a car opens the inspector. */
canvas.addEventListener('click', async (event) => {
  const frame = S.snapshot;
  if (!frame) return;
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const floors = frame.building.floors;
  const rowHeight = Math.max(22, Math.min(46, 620 / floors));
  const top = 34;
  const floor = floors - 1 - Math.floor((y - top) / rowHeight);
  if (floor < 0 || floor >= floors) return;

  const labelWidth = 48;
  const waitWidth = 84;
  const shaftArea = canvas.clientWidth - labelWidth - waitWidth - 26;
  const shaftWidth = Math.max(26, Math.min(74, shaftArea / frame.building.cars - 8));
  const gap = frame.building.cars > 1 ? (shaftArea - shaftWidth * frame.building.cars) / (frame.building.cars - 1 + 0.0001) : 0;

  for (let car = 0; car < frame.building.cars; car += 1) {
    const cx = labelWidth + 30 + car * (shaftWidth + gap);
    if (x >= cx && x <= cx + shaftWidth) {
      const carData = frame.cars.find((c) => c.car_id === car);
      if (carData && Math.abs(carData.floor - floor) <= 1) { inspect(`car-${car}`); return; }
    }
  }
  try {
    await API.post('/api/passenger', { origin: floor, priority: event.shiftKey });
    toast(`Passenger added on floor ${floor}${event.shiftKey ? ' (priority)' : ''}`);
  } catch (error) { toast(error.message, true); }
});

/* ------------------------------------------------------------------- KPIs */

function renderKpis(frame) {
  const m = frame.metrics;
  const threshold = 60;
  const cls = (value, warnAt, badAt) => (value >= badAt ? 'bad' : value >= warnAt ? 'warn' : 'good');
  $('kpis').innerHTML = [
    kpi('Tick', m.tick, 's', ''),
    kpi('Avg wait', fmt(m.avg_wait), 's', cls(m.avg_wait, 30, 60)),
    kpi('P95 wait', fmt(m.p95_wait), 's', cls(m.p95_wait, 60, 120)),
    kpi('Max wait', fmt(m.max_wait, 0), 's', cls(m.max_wait, 90, 180)),
    kpi(`Waits >${threshold}s`, fmt(m.long_wait_pct), '%', cls(m.long_wait_pct, 10, 25)),
    kpi('Avg ride', fmt(m.avg_ride), 's', ''),
    kpi('Delivered', m.delivered, `/${m.arrived}`, ''),
    kpi('Waiting', m.waiting, '', m.waiting > 20 ? 'warn' : ''),
    kpi('Throughput', fmt(m.throughput, 0), '/h', ''),
    kpi('Energy', fmt(m.energy, 0), '', ''),
    kpi('Messages', m.messages, '', ''),
    kpi('Nodes', m.nodes_expanded, '', ''),
    kpi('ms/tick', fmt(m.compute_ms_per_tick, 2), '', m.compute_ms_per_tick > 40 ? 'warn' : ''),
    kpi('Rules fired', m.rules_fired, '', m.rules_fired > 0 ? 'warn' : ''),
  ].join('');
}

const kpi = (label, value, unit, cls) =>
  `<div class="kpi ${cls}"><div class="label">${label}</div>
   <div class="value">${value}<span class="unit">${unit}</span></div></div>`;

/* --------------------------------------------------------------- auction */

function renderAuction(frame) {
  const auction = frame.auction;
  const box = $('auction-body');
  if (!auction) { box.innerHTML = '<div class="empty">No auction yet. Let traffic arrive, or click a floor to add a passenger.</div>'; return; }

  const viable = auction.bids.filter((b) => !b.refused);
  const refused = auction.bids.filter((b) => b.refused);
  const max = Math.max(...viable.map((b) => b.total), 1);

  const rows = auction.bids.map((bid) => {
    if (bid.refused) {
      return `<tr><td><span class="chip" style="border-color:${carColour(bid.car_id)}">C${bid.car_id}</span></td>
        <td colspan="4" style="color:var(--text-faint)">REFUSE — ${bid.reason}</td></tr>`;
    }
    const won = bid.car_id === auction.winner;
    const pct = (bid.total / max) * 100;
    return `<tr class="${won ? 'best' : ''}">
      <td><span class="chip" style="border-color:${carColour(bid.car_id)}">C${bid.car_id}</span>${won ? ' <b style="color:var(--good)">WIN</b>' : ''}</td>
      <td style="min-width:110px"><div style="background:var(--bg-input);border-radius:3px;height:14px;overflow:hidden">
        <div style="width:${pct}%;height:100%;background:${carColour(bid.car_id)};opacity:${won ? 1 : 0.45}"></div></div></td>
      <td class="num">${fmt(bid.total, 1)}</td>
      <td class="num" style="color:var(--text-dim)">${fmt(bid.wait, 1)}/${fmt(bid.ride, 1)}/${fmt(bid.crowding, 1)}/${fmt(bid.energy, 1)}</td>
      <td class="num">${fmt(bid.eta, 0)}s</td></tr>`;
  }).join('');

  box.innerHTML = `
    <div style="font-size:12px;color:var(--text-dim);margin-bottom:8px">
      Call <b style="color:var(--text)">floor ${auction.floor} ${auction.direction === 'UP' ? '▲' : '▼'}</b>
      at t=${auction.tick} &middot; <code>${auction.conversation_id}</code>
      &middot; ${viable.length} proposed, ${refused.length} refused
    </div>
    <table><thead><tr><th>Car</th><th>Bid</th><th class="num">Total</th>
      <th class="num">W/R/C/E</th><th class="num">ETA</th></tr></thead><tbody>${rows}</tbody></table>
    <div style="font-size:11px;color:var(--text-faint);margin-top:7px">
      Bid = marginal cost of inserting this call into the car's A* plan, split into
      <b>W</b>ait / <b>R</b>ide / <b>C</b>rowding / <b>E</b>nergy and weighted by the current traffic pattern.
    </div>`;
}

/* -------------------------------------------------------------- messages */

let messageFilter = '';

function renderMessages(frame) {
  const rows = frame.messages
    .filter((m) => !messageFilter || m.performative === messageFilter)
    .slice(-60)
    .reverse()
    .map((m) => `<div class="log-row" style="border-left-color:${performativeColour(m.performative)}">
        <span class="tick">${m.tick}s</span>
        <span class="perf p-${m.performative}">${m.performative}</span>
        <span class="who">${m.sender} → ${m.receiver}</span></div>`)
    .join('');
  $('messages').innerHTML = rows || '<div class="empty">No messages yet.</div>';
  $('message-count').textContent = frame.metrics.messages;
}

const PERF_COLOURS = {
  REQUEST: '#39c5cf', CFP: '#58a6ff', PROPOSE: '#bc8cff', REFUSE: '#6b7784',
  ACCEPT_PROPOSAL: '#3fb950', REJECT_PROPOSAL: '#8b6c3f', INFORM: '#9aa7b5',
  CANCEL: '#d29922', FAILURE: '#f85149',
};
const performativeColour = (p) => PERF_COLOURS[p] || '#6b7784';

/* ----------------------------------------------------------------- rules */

function renderRules(frame) {
  const rows = frame.rules.slice().reverse().map((r) =>
    `<div class="rule-row"><span class="rtick">t=${r.tick}</span>
      <div class="rname">${r.rule}</div>
      <div class="reffect">${r.effect || JSON.stringify(r.binding)}</div></div>`).join('');
  $('rules').innerHTML = rows || '<div class="empty">No safety rule has fired. Inject a fault or a fire alarm.</div>';
}

/* --------------------------------------------------------------- traffic */

function renderTraffic(frame) {
  const t = frame.traffic;
  const correct = t.pattern === t.true_pattern;
  $('traffic').innerHTML = `
    <dl class="kv">
      <dt>Detected</dt><dd><span class="chip type">${t.pattern}</span></dd>
      <dt>Ground truth</dt><dd><span class="chip" style="${correct ? '' : 'border-color:var(--warn);color:var(--warn)'}">${t.true_pattern}</span>
        ${correct ? '<span style="color:var(--good)">✓</span>' : '<span style="color:var(--warn)">learning…</span>'}</dd>
      <dt>Why</dt><dd style="font-size:11.5px">${t.reason}</dd>
      <dt>Observed</dt><dd>${t.observed} arrivals</dd>
      <dt>Weights</dt><dd>wait ${fmt(t.weights.wait, 2)} &middot; ride ${fmt(t.weights.ride, 2)}
        &middot; crowd ${fmt(t.weights.crowding, 2)} &middot; energy ${fmt(t.weights.energy, 2)}</dd>
      <dt>Parking</dt><dd>${Object.keys(frame.parking || {}).length
        ? Object.entries(frame.parking).map(([c, f]) => `C${c}→${f}`).join(', ') : 'none'}</dd>
    </dl>`;
}

/* ---------------------------------------------------------- AWT over time */

function pushAwt(frame) {
  const chart = S.charts.awt;
  if (!chart) return;
  const last = S.awt.labels[S.awt.labels.length - 1];
  if (last === frame.tick) return;
  if (last !== undefined && frame.tick < last) { S.awt.labels = []; S.awt.wait = []; S.awt.p95 = []; }
  S.awt.labels.push(frame.tick);
  S.awt.wait.push(frame.metrics.avg_wait);
  S.awt.p95.push(frame.metrics.p95_wait);
  if (S.awt.labels.length > 320) { S.awt.labels.shift(); S.awt.wait.shift(); S.awt.p95.shift(); }
  chart.data.labels = S.awt.labels;
  chart.data.datasets[0].data = S.awt.wait;
  chart.data.datasets[1].data = S.awt.p95;
  chart.update('none');
}

/* ------------------------------------------------------------- inspector */

async function inspect(address) {
  S.selected = address;
  try {
    const data = await API.get(`/api/agent/${address}`);
    renderInspector(data);
  } catch (error) { $('inspector').innerHTML = `<div class="empty">${error.message}</div>`; }
}
async function refreshInspector() {
  try { renderInspector(await API.get(`/api/agent/${S.selected}`)); } catch { /* agent gone */ }
}

function renderInspector(d) {
  const peas = d.peas || {};
  const state = d.internal_state || {};
  const rows = Object.entries(state).map(([k, v]) =>
    `<dt>${k.replace(/_/g, ' ')}</dt><dd>${Array.isArray(v) ? (v.join(', ') || '—') : String(v)}</dd>`).join('');

  $('inspector').innerHTML = `
    <div style="margin-bottom:9px">
      <span class="chip type">${d.agent_type}</span>
      <code>${d.address}</code>
    </div>
    ${['performance', 'environment', 'actuators', 'sensors'].map((k) => peas[k]
      ? `<div class="peas-item"><div class="pk">${k[0].toUpperCase()} — ${k}</div><div class="pv">${peas[k]}</div></div>`
      : '').join('')}
    ${rows ? `<h3 class="sec">Internal state</h3><dl class="kv">${rows}</dl>` : ''}
    ${d.planned_route ? `<h3 class="sec">Planned route (A*)</h3>
      <div>${d.planned_route.length ? d.planned_route.map((s) => `<span class="chip">${s}</span>`).join('') : '<span style="color:var(--text-faint)">idle</span>'}</div>
      <div style="font-size:11.5px;color:var(--text-faint);margin-top:5px">
        plan cost ${fmt(d.plan_cost, 1)} &middot; ${d.plan_nodes_expanded} nodes expanded &middot; ${d.replans} replans</div>` : ''}
    ${d.last_bid ? `<h3 class="sec">Last bid</h3><dl class="kv">
      <dt>total</dt><dd>${fmt(d.last_bid.total, 2)}</dd>
      <dt>wait/ride</dt><dd>${fmt(d.last_bid.wait, 2)} / ${fmt(d.last_bid.ride, 2)}</dd>
      <dt>crowd/energy</dt><dd>${fmt(d.last_bid.crowding, 2)} / ${fmt(d.last_bid.energy, 2)}</dd></dl>` : ''}
    ${d.rules ? `<h3 class="sec">Rule base (salience order)</h3>
      ${d.rules.map((r) => `<div class="rule-row"><div class="rname">${r.name} <span style="color:var(--text-faint)">[${r.salience}]</span></div>
        <div class="reffect">${r.description}</div></div>`).join('')}` : ''}
    ${d.facts ? `<h3 class="sec">Working memory</h3><div>${d.facts.length
      ? d.facts.map((f) => `<span class="chip">${f}</span>`).join('') : '<span style="color:var(--text-faint)">empty</span>'}</div>` : ''}
    ${d.assignments ? `<h3 class="sec">Assignments</h3><div>${Object.keys(d.assignments).length
      ? Object.entries(d.assignments).map(([c, v]) => `<span class="chip">${c}→C${v}</span>`).join('') : '<span style="color:var(--text-faint)">none</span>'}</div>` : ''}
    ${d.pattern ? `<h3 class="sec">Learned demand</h3><dl class="kv">
      <dt>pattern</dt><dd>${d.pattern}</dd><dt>why</dt><dd style="font-size:11.5px">${d.reason}</dd>
      <dt>busiest</dt><dd>${(d.top_floors || []).map((f) => `fl${f.floor}:${f.rate}`).join(' ') || '—'}</dd></dl>` : ''}`;
}

function renderAgentPicker(frame) {
  const options = [
    ...frame.cars.map((c) => `car-${c.car_id}`),
    'dispatcher', 'monitor', 'safety',
    ...frame.floors.map((f) => `floor-${f.floor}`),
  ];
  const select = $('agent-select');
  if (select.options.length !== options.length) {
    select.innerHTML = options.map((o) => `<option value="${o}">${o}</option>`).join('');
  }
}

/* -------------------------------------------------------------- controls */

function renderControls(frame) {
  const running = frame.session.running;
  $('btn-play').textContent = running ? 'Pause' : 'Play';
  $('btn-play').className = running ? 'warn' : 'primary';
  setStatus(
    `t=${frame.tick}/${frame.session.duration}s · ${frame.strategy} · seed ${frame.seed}`,
    frame.fire_alarm ? 'alarm' : running ? 'live' : 'paused',
  );
  renderAgentPicker(frame);
}

function setStatus(text, cls) {
  const pill = $('status');
  pill.textContent = text;
  pill.className = `status-pill ${cls}`;
}

function toast(message, isError) {
  const el = $('toast');
  el.textContent = message;
  el.style.color = isError ? 'var(--bad)' : 'var(--good)';
  clearTimeout(el._timer);
  el._timer = setTimeout(() => { el.textContent = ''; }, 2600);
}

/* ---------------------------------------------------------- search lab */

async function runSearchLab() {
  const box = $('searchlab-body');
  box.innerHTML = '<div class="empty"><span class="spinner"></span> running BFS, UCS, Greedy and A*…</div>';
  try {
    const data = await API.get('/api/search-lab');
    if (data.error) { box.innerHTML = `<div class="empty">${data.error}</div>`; return; }
    const rows = data.results.map((r) => `<tr class="${r.optimal ? 'best' : ''}">
      <td><b>${r.algorithm.toUpperCase()}</b></td>
      <td class="num">${r.found ? fmt(r.cost, 1) : '—'}${r.optimal ? ' ✓' : ''}</td>
      <td class="num">${r.nodes_expanded}</td><td class="num">${r.nodes_generated}</td>
      <td class="num">${r.max_frontier}</td><td class="num">${fmt(r.runtime_ms, 3)}</td>
      <td style="font:11.5px var(--mono)">${r.route.join(' → ') || '—'}</td></tr>`).join('');

    const astar = data.results.find((r) => r.algorithm === 'astar');
    const ucs = data.results.find((r) => r.algorithm === 'ucs');
    const saving = ucs && astar && ucs.nodes_expanded
      ? (100 * (ucs.nodes_expanded - astar.nodes_expanded) / ucs.nodes_expanded).toFixed(0) : '0';

    box.innerHTML = `
      <div class="note"><strong>${data.source}</strong> — car at floor ${data.current_floor},
        ${data.stops.length} pending stop(s), h(start) = ${data.h_at_start}.
        All four algorithms run on this identical problem instance.</div>
      <table><thead><tr><th>Algorithm</th><th class="num">Cost</th><th class="num">Expanded</th>
        <th class="num">Generated</th><th class="num">Max frontier</th><th class="num">ms</th><th>Route</th>
        </tr></thead><tbody>${rows}</tbody></table>
      <div class="note" style="margin-top:12px;border-color:var(--good)">
        <strong>A* vs UCS:</strong> identical cost (both optimal), but A* expanded
        <strong>${saving}% fewer nodes</strong> — the payoff of an admissible, consistent heuristic.
        Greedy is fastest but not cost-optimal; BFS optimises stop count, which is the wrong
        objective when step costs differ.</div>
      <div style="font-size:11.5px;color:var(--text-faint);margin-top:8px">
        Route notation: floor, then ↑/↓ for the hall-call direction, then P(ickup) or D(rop-off).</div>`;
  } catch (error) { box.innerHTML = `<div class="empty">${error.message}</div>`; }
}

async function runAnnealingLab() {
  const box = $('annealing-body');
  box.innerHTML = '<div class="empty"><span class="spinner"></span> annealing…</div>';
  try {
    const data = await API.get('/api/search-lab/annealing');
    if (!data.available) { box.innerHTML = `<div class="empty">${data.reason}</div>`; return; }
    const provenance = data.synthetic
      ? `<div class="note" style="border-color:var(--warn)">Not enough live traffic to re-optimise, so this is a
           <strong>constructed</strong> ${data.calls}-call / ${data.cars}-car instance built from this building's
           geometry. The search code and the objective are the live ones.</div>`
      : `<div class="note">Live instance: <strong>${data.calls} open hall calls</strong> across
           ${data.cars} available cars, taken from the running simulation.</div>`;
    box.innerHTML = provenance + `<div class="legend">
        <span><i style="background:#58a6ff"></i>SA current</span>
        <span><i style="background:#3fb950"></i>SA best</span>
        <span><i style="background:#f0883e"></i>Hill climbing</span></div>
      <div class="chart-box tall"><canvas id="sa-chart"></canvas></div>
      <table><thead><tr><th>Search</th><th class="num">Initial</th><th class="num">Final</th>
        <th class="num">Improvement</th><th class="num">Iterations</th></tr></thead><tbody>
        ${[data.simulated_annealing, data.hill_climbing].map((r) => `<tr>
          <td>${r.algorithm.replace(/_/g, ' ')}</td><td class="num">${fmt(r.initial_cost, 1)}</td>
          <td class="num">${fmt(r.final_cost, 1)}</td><td class="num">${fmt(r.improvement, 1)}</td>
          <td class="num">${r.iterations}</td></tr>`).join('')}</tbody></table>
      <div class="note" style="margin-top:10px">SA accepts worse moves early (the jagged blue line) to
        escape local optima; its running best (green) never worsens. Hill climbing stops at the first
        local optimum it cannot improve on.</div>`;

    const ctx2 = $('sa-chart').getContext('2d');
    if (S.charts.sa) S.charts.sa.destroy();
    S.charts.sa = new Chart(ctx2, {
      type: 'line',
      data: {
        labels: data.simulated_annealing.curve.map((_, i) => i),
        datasets: [
          { label: 'SA current', data: data.simulated_annealing.curve, borderColor: '#58a6ff', borderWidth: 1, pointRadius: 0, tension: 0 },
          { label: 'SA best', data: data.simulated_annealing.best_curve, borderColor: '#3fb950', borderWidth: 2, pointRadius: 0, tension: 0 },
          { label: 'Hill climbing', data: data.hill_climbing.curve, borderColor: '#f0883e', borderWidth: 2, pointRadius: 0, stepped: true },
        ],
      },
      options: chartOptions('iteration', 'assignment cost'),
    });
  } catch (error) { box.innerHTML = `<div class="empty">${error.message}</div>`; }
}

async function runMinimaxLab() {
  const box = $('minimax-body');
  box.innerHTML = '<div class="empty"><span class="spinner"></span> solving the parking game…</div>';
  try {
    const d = await API.get('/api/search-lab/minimax');
    box.innerHTML = `
      <dl class="kv">
        <dt>Cars placed</dt><dd>${d.cars_placed} (of ${d.idle_cars} idle)</dd>
        <dt>Candidate spots</dt><dd>${d.candidate_spots.join(', ')}</dd>
        <dt>Nature's floors</dt><dd>${d.likely_floors.join(', ')}</dd>
        <dt>Best parking</dt><dd><b>${d.best_parking.join(', ')}</b></dd>
        <dt>Minimax value</dt><dd>${fmt(d.value, 1)} (worst-case distance ${fmt(-d.value, 1)} floors)</dd>
        <dt>Worst floor</dt><dd>${d.worst_floor}</dd>
      </dl>
      <h3 class="sec">Pruning</h3>
      <table><thead><tr><th>Search</th><th class="num">Nodes visited</th></tr></thead><tbody>
        <tr><td>Minimax (no pruning)</td><td class="num">${d.nodes_minimax}</td></tr>
        <tr class="best"><td>Alpha-beta</td><td class="num">${d.nodes_alphabeta}</td></tr>
      </tbody></table>
      <div class="note" style="margin-top:10px">Same value, <strong>${d.pruning_saving_pct}% fewer nodes</strong>.
        MAX (the dispatcher) picks where to park; MIN ("nature") then picks the worst floor for the next call,
        so the value is a worst-case guarantee rather than an average.</div>`;
  } catch (error) { box.innerHTML = `<div class="empty">${error.message}</div>`; }
}

function chartOptions(xLabel, yLabel) {
  return {
    responsive: true, maintainAspectRatio: false, animation: false,
    interaction: { intersect: false, mode: 'index' },
    plugins: { legend: { display: false } },
    scales: {
      x: { title: { display: true, text: xLabel, color: '#6b7784' }, ticks: { color: '#6b7784', maxTicksLimit: 10 }, grid: { color: 'rgba(255,255,255,0.05)' } },
      y: { title: { display: true, text: yLabel, color: '#6b7784' }, ticks: { color: '#6b7784' }, grid: { color: 'rgba(255,255,255,0.05)' } },
    },
  };
}

/* -------------------------------------------------------------- benchmark */

async function runBenchmark() {
  const box = $('benchmark-body');
  const seeds = Number($('bench-seeds').value);
  const ticks = Number($('bench-ticks').value);
  $('btn-bench').disabled = true;
  box.innerHTML = `<div class="empty"><span class="spinner"></span> running 4 strategies × 4 scenarios × ${seeds} seeds = ${16 * seeds} runs of ${ticks} ticks. This takes a while.</div>`;
  try {
    const data = await API.post('/api/benchmark', { seeds, ticks });
    const byScenario = {};
    data.summary.forEach((row) => { (byScenario[row.scenario] ||= []).push(row); });

    let html = `<div class="note"><strong>${data.runs} runs</strong> — mean ± sd over
      ${data.seeds.length} seeds, ${data.ticks} ticks each. Lower is better for every column except
      delivered.</div>`;

    for (const [scenario, rows] of Object.entries(byScenario)) {
      const best = Math.min(...rows.map((r) => r.avg_wait_mean));
      html += `<h3 class="sec">${scenario}</h3>
        <table><thead><tr><th>Strategy</th><th class="num">Avg wait</th><th class="num">P95</th>
          <th class="num">Long waits</th><th class="num">Delivered</th><th class="num">Energy</th>
          <th class="num">Nodes</th></tr></thead><tbody>
        ${rows.map((r) => `<tr class="${Math.abs(r.avg_wait_mean - best) < 1e-9 ? 'best' : ''}">
          <td>${r.strategy}</td>
          <td class="num">${fmt(r.avg_wait_mean)} ± ${fmt(r.avg_wait_std)}</td>
          <td class="num">${fmt(r.p95_wait_mean)}</td>
          <td class="num">${fmt(r.long_wait_pct_mean)}%</td>
          <td class="num">${fmt(r.delivered_mean, 0)}</td>
          <td class="num">${fmt(r.energy_mean, 0)}</td>
          <td class="num">${fmt(r.nodes_expanded_mean, 0)}</td></tr>`).join('')}
        </tbody></table>`;
    }

    html += `<h3 class="sec">Full vs NearestCar baseline</h3>
      <table><thead><tr><th>Scenario</th><th class="num">Wait</th><th class="num">P95</th>
        <th class="num">Long waits</th><th>Verdict</th></tr></thead><tbody>
      ${data.comparison.map((c) => {
        const won = data.wins[c.scenario];
        return `<tr><td>${c.scenario}</td>
          <td class="num" style="color:${c.avg_wait_improvement_pct > 0 ? 'var(--good)' : 'var(--bad)'}">${c.avg_wait_improvement_pct > 0 ? '+' : ''}${fmt(c.avg_wait_improvement_pct, 1)}%</td>
          <td class="num" style="color:${c.p95_wait_improvement_pct > 0 ? 'var(--good)' : 'var(--bad)'}">${c.p95_wait_improvement_pct > 0 ? '+' : ''}${fmt(c.p95_wait_improvement_pct, 1)}%</td>
          <td class="num" style="color:${c.long_wait_pct_improvement_pct > 0 ? 'var(--good)' : 'var(--bad)'}">${c.long_wait_pct_improvement_pct > 0 ? '+' : ''}${fmt(c.long_wait_pct_improvement_pct, 1)}%</td>
          <td style="color:${won ? 'var(--good)' : 'var(--text-faint)'}">${won ? 'beats baseline' : 'parity / worse'}</td></tr>`;
      }).join('')}</tbody></table>
      <div class="note" style="margin-top:12px">Positive means the full strategy is better. It wins
        decisively in the <strong>peak</strong> regimes, where coordination has something to exploit;
        in light traffic every strategy is equivalent because no car is ever contended for.
        See <code>docs/TESTING.md</code> for the discussion.</div>`;

    box.innerHTML = html;
  } catch (error) {
    box.innerHTML = `<div class="empty">${error.message}</div>`;
  } finally { $('btn-bench').disabled = false; }
}

/* ------------------------------------------------------------------ theory */

async function renderTheory() {
  const meta = S.meta || (S.meta = await API.get('/api/meta'));
  $('theory-agents').innerHTML = meta.agents.map((a) => `
    <div class="card" style="margin-bottom:12px"><div class="card-head">
      <h2>${a.name}</h2><span class="spacer"></span><span class="chip type">${a.agent_type}</span></div>
      <div class="card-body">
        ${['performance', 'environment', 'actuators', 'sensors'].map((k) => `
          <div class="peas-item"><div class="pk">${k[0].toUpperCase()} — ${k}</div>
          <div class="pv">${a.peas[k] || '—'}</div></div>`).join('')}
      </div></div>`).join('');

  $('theory-env').innerHTML = `<table><thead><tr><th>Dimension</th><th>Classification</th>
    <th>Justification</th></tr></thead><tbody>
    ${meta.environment.map((e) => `<tr><td><b>${e.property}</b></td>
      <td><span class="chip type">${e.value}</span></td>
      <td style="color:var(--text-dim)">${e.justification}</td></tr>`).join('')}</tbody></table>`;

  $('theory-peas').innerHTML = ['performance', 'environment', 'actuators', 'sensors'].map((k) =>
    `<div class="peas-item"><div class="pk">${k[0].toUpperCase()} — ${k}</div>
     <div class="pv">${meta.system_peas[k]}</div></div>`).join('');

  $('theory-rules').innerHTML = `<table><thead><tr><th class="num">Salience</th><th>Rule</th>
    <th>Production</th></tr></thead><tbody>
    ${meta.rules.map((r) => `<tr><td class="num">${r.salience}</td><td><code>${r.name}</code></td>
      <td style="color:var(--text-dim)">${r.description}</td></tr>`).join('')}</tbody></table>`;

  $('theory-strategies').innerHTML = `<table><thead><tr><th>Strategy</th><th>Assignment</th>
    <th>Routing</th><th>Reassign</th><th>Parking</th><th>Learns</th></tr></thead><tbody>
    ${meta.strategies.map((s) => `<tr><td><b>${s.name}</b><div style="color:var(--text-faint);font-size:11px">${s.description}</div></td>
      <td>${s.assignment}</td><td>${s.routing}</td><td>${s.reassignment || '—'}</td>
      <td>${s.parking_policy || '—'}</td><td>${s.adapts_weights ? 'yes' : 'no'}</td></tr>`).join('')}</tbody></table>`;
}

/* -------------------------------------------------------------------- boot */

const TABS = ['live', 'search', 'bench', 'theory'];

function switchTab(name) {
  if (!TABS.includes(name)) name = 'live';
  document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t.dataset.tab === name));
  document.querySelectorAll('.panel-tab').forEach((p) => p.classList.toggle('active', p.id === `tab-${name}`));
  // Keep the tab in the URL, so a demo can be deep-linked and a reload stays put.
  if (location.hash.slice(1) !== name) history.replaceState(null, '', `#${name}`);
  if (name === 'theory') renderTheory();
  if (name === 'search') { runSearchLab(); runAnnealingLab(); runMinimaxLab(); }
}

async function boot() {
  document.querySelectorAll('.tab').forEach((t) => t.addEventListener('click', () => switchTab(t.dataset.tab)));
  window.addEventListener('hashchange', () => switchTab(location.hash.slice(1)));

  $('btn-play').onclick = async () => {
    const running = S.snapshot && S.snapshot.session.running;
    await API.post(running ? '/api/pause' : '/api/play');
  };
  $('btn-step').onclick = () => API.post('/api/step', { ticks: 1 });
  $('btn-step10').onclick = () => API.post('/api/step', { ticks: 10 });
  $('btn-reset').onclick = async () => {
    S.awt = { labels: [], wait: [], p95: [] };
    await API.post('/api/reset', {
      scenario: $('sel-scenario').value,
      strategy: $('sel-strategy').value,
      seed: Number($('in-seed').value),
      floors: Number($('in-floors').value),
      cars: Number($('in-cars').value),
    });
    toast('Simulation reset');
  };
  $('in-speed').oninput = (e) => {
    $('speed-label').textContent = `${e.target.value}×`;
    API.post('/api/speed', { speed: Number(e.target.value) });
  };
  $('btn-fault').onclick = async () => {
    const car = Number($('sel-fault-car').value);
    await API.post('/api/inject', { kind: 'car_fault', car });
    toast(`Car ${car} fault injected — watch rule R4 fire`);
  };
  $('btn-repair').onclick = async () => {
    const car = Number($('sel-fault-car').value);
    await API.post('/api/inject', { kind: 'car_repair', car });
    toast(`Car ${car} repaired`);
  };
  $('btn-fire').onclick = async () => {
    await API.post('/api/inject', { kind: 'fire_alarm' });
    toast('Fire alarm — rules R1/R2/R3 recall the fleet', true);
  };
  $('btn-fire-clear').onclick = async () => {
    await API.post('/api/inject', { kind: 'fire_clear' });
    toast('Alarm cleared — rule R7 restores service');
  };
  $('btn-rush').onclick = async () => {
    await API.post('/api/inject', { kind: 'rush', floor: 0, count: 12 });
    toast('12 passengers surged into the lobby');
  };
  $('agent-select').onchange = (e) => inspect(e.target.value);
  $('msg-filter').onchange = (e) => { messageFilter = e.target.value; if (S.snapshot) renderMessages(S.snapshot); };
  $('btn-searchlab').onclick = runSearchLab;
  $('btn-annealing').onclick = runAnnealingLab;
  $('btn-minimax').onclick = runMinimaxLab;
  $('btn-bench').onclick = runBenchmark;

  const meta = (S.meta = await API.get('/api/meta'));
  $('sel-scenario').innerHTML = meta.scenarios.map((s) => `<option>${s}</option>`).join('');
  $('sel-strategy').innerHTML = meta.strategies.map((s) => `<option value="${s.name}">${s.label}</option>`).join('');

  const state = await API.get('/api/state');
  // Paint at once from REST. The WebSocket may take a moment (or be blocked entirely by a
  // proxy), and a dashboard that shows an empty building until then looks broken.
  S.snapshot = state;
  S.frameTime = performance.now();
  onFrame(state);
  $('sel-scenario').value = state.scenario;
  $('sel-strategy').value = state.strategy;
  $('in-seed').value = state.seed;
  $('in-floors').value = state.building.floors;
  $('in-cars').value = state.building.cars;
  $('sel-fault-car').innerHTML = state.cars.map((c) => `<option value="${c.car_id}">Car ${c.car_id}</option>`).join('');

  S.charts.awt = new Chart($('awt-chart').getContext('2d'), {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: 'Avg wait', data: [], borderColor: '#58a6ff', backgroundColor: 'rgba(88,166,255,0.10)', fill: true, borderWidth: 2, pointRadius: 0, tension: 0.25 },
        { label: 'P95 wait', data: [], borderColor: '#d29922', borderWidth: 1.5, pointRadius: 0, tension: 0.25, borderDash: [4, 3] },
      ],
    },
    options: chartOptions('simulated seconds', 'seconds'),
  });

  connect();
  requestAnimationFrame(draw);
  inspect('dispatcher');
  if (location.hash) switchTab(location.hash.slice(1));

  // Fallback poll: if the WebSocket is not open (still connecting, or blocked), keep the
  // dashboard live over REST rather than showing a frozen building.
  setInterval(async () => {
    if (S.socket && S.socket.readyState === WebSocket.OPEN) return;
    try {
      const frame = await API.get('/api/state');
      if (!S.snapshot || frame.tick !== S.snapshot.tick) {
        S.previous = S.snapshot;
        S.snapshot = frame;
        S.frameTime = performance.now();
        onFrame(frame);
      }
    } catch { /* server restarting */ }
  }, 1000);
}

boot().catch((error) => {
  document.body.insertAdjacentHTML('afterbegin',
    `<div style="padding:14px;background:#2d1412;color:#f85149">Failed to start: ${error.message}</div>`);
});
