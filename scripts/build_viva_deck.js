// Build the case-study review deck (18 slides, white background), aligned to the two reviews:
//   Review 1 — PEAS · Environment & agent analysis · Algorithmic modelling & search · Q&A
//   Review 2 — Tools & setup · Multi-agent execution & interaction · Demo & testing ·
//              Code structure & scalability
//   npm install pptxgenjs   (once, anywhere on NODE_PATH)   then   node scripts/build_viva_deck.js
// Charts come from reports/ (real runs); screenshots from docs/img/ui (scripts/capture_ui.mjs).
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const FIG = (n) => path.join(ROOT, "reports", "viva", n);
const UI = (n) => path.join(ROOT, "docs", "img", "ui", n);
const OUT = path.join(ROOT, "docs", "LiftZero_Viva_Deck.pptx");

const NAVY = "1F3A5F", TEAL = "0F7C7A", SLATE = "4A5560", MUTED = "6B7680";
const TINT = "F2F5F8", TINT2 = "E8F3F2", WHITE = "FFFFFF", LINE = "D5DBE1";
const HEAD = "Cambria", BODY = "Calibri";

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
pres.title = "LiftZero: Multi-Agent Smart Elevator Fleet Coordinator";
pres.author = "Adhikkesh, Sisr Reddy, Kavin Karthic, Akash";
pres.subject = "FOAI case study review";
pres.theme = { headFontFace: HEAD, bodyFontFace: BODY };

function fitImage(slide, file, x, y, maxW, maxH, alt, border = false) {
  if (!fs.existsSync(file)) throw new Error(`missing image ${file}`);
  const { w, h } = pngSize(file);
  let iw = maxW, ih = (maxW * h) / w;
  if (ih > maxH) { ih = maxH; iw = (maxH * w) / h; }
  const ix = x + (maxW - iw) / 2;
  slide.addImage({ path: file, x: ix, y, w: iw, h: ih, altText: alt });
  if (border) slide.addShape(pres.shapes.RECTANGLE, { x: ix, y, w: iw, h: ih, fill: { type: "none" }, line: { color: LINE, width: 1 } });
}

pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: WHITE },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 10.4, h: 0.8,
      fontFace: HEAD, fontSize: 30, bold: true, color: NAVY, align: "left", valign: "middle", margin: 0 }, text: "" } },
    { text: { text: "LiftZero  ·  FOAI case study", options: { x: 0.6, y: 7.0, w: 6, h: 0.3,
      fontFace: BODY, fontSize: 10, color: MUTED, margin: 0 } } },
  ],
  slideNumber: { x: 12.2, y: 7.0, w: 0.5, h: 0.3, fontFace: BODY, fontSize: 10, color: MUTED },
});

let n = 0;
function content(title, rubric, notes) {
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText(title, { placeholder: "title" });
  if (rubric) {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 11.1, y: 0.5, w: 1.6, h: 0.45, fill: { color: TINT2 }, line: { color: TINT2 }, rectRadius: 0.08 });
    s.addText(rubric, { isTextBox: true, x: 11.1, y: 0.5, w: 1.6, h: 0.45, align: "center", valign: "middle", fontFace: BODY, fontSize: 11, bold: true, color: TEAL, margin: 0 });
  }
  if (notes) s.addNotes(notes);
  n += 1;
  return s;
}
function text(s, t, o) { s.addText(t, { isTextBox: true, fontFace: BODY, color: SLATE, margin: 0, ...o }); }
function card(s, x, y, w, h, fill = TINT) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: fill }, rectRadius: 0.08 });
}
function stat(s, x, y, w, big, label, color = NAVY, h = 1.45) {
  card(s, x, y, w, h);
  text(s, big, { x: x + 0.25, y: y + 0.15, w: w - 0.5, h: 0.75, fontFace: HEAD, fontSize: 32, bold: true, color });
  text(s, label, { x: x + 0.25, y: y + 0.88, w: w - 0.5, h: h - 0.95, fontSize: 13, color: SLATE, valign: "top" });
}
function bullets(s, items, o) {
  s.addText(items.map((t, i) => ({ text: t, options: { bullet: { indent: 18 }, breakLine: i < items.length - 1 } })),
    { isTextBox: true, fontFace: BODY, fontSize: 16, color: SLATE, paraSpaceAfter: 8, valign: "top", margin: 0, ...o });
}
function dot(s, x, y, label, fill = TEAL) {
  s.addShape(pres.shapes.OVAL, { x, y, w: 0.5, h: 0.5, fill: { color: fill }, line: { color: fill } });
  text(s, label, { x, y, w: 0.5, h: 0.5, align: "center", valign: "middle", fontSize: 14, bold: true, color: WHITE });
}
function table(s, header, rows, o) {
  const fs_ = o.fontSize || 13;
  const hdr = header.map((h) => ({ text: h, options: { bold: true, color: WHITE, fill: { color: NAVY }, fontFace: BODY, fontSize: fs_ } }));
  const body = rows.map((r, i) => r.map((c, j) => ({ text: c, options: { color: SLATE, fontFace: BODY, fontSize: fs_,
    bold: j === 0, fill: { color: i % 2 ? WHITE : TINT } } })));
  s.addTable([hdr, ...body], { x: o.x, y: o.y, w: o.w, colW: o.colW, border: { type: "solid", color: LINE, pt: 0.5 }, margin: 0.06, valign: "middle", autoPage: false });
}

// 1 ── Title ─────────────────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  n += 1;
  text(s, "FOUNDATIONS OF ARTIFICIAL INTELLIGENCE  ·  CASE STUDY REVIEW", { x: 0.8, y: 0.8, w: 11.5, h: 0.4, fontSize: 13, bold: true, color: TEAL, charSpacing: 2 });
  text(s, "LiftZero", { x: 0.8, y: 1.35, w: 11.5, h: 1.2, fontFace: HEAD, fontSize: 60, bold: true, color: NAVY });
  text(s, "Multi-Agent Smart Elevator Fleet Coordinator", { x: 0.8, y: 2.5, w: 11.5, h: 0.6, fontFace: HEAD, fontSize: 28, color: NAVY });
  text(s, "Autonomous lift agents that negotiate every hall call — search, optimisation, safety rules and learning", { x: 0.8, y: 3.15, w: 11.5, h: 0.9, fontSize: 18, color: SLATE, valign: "top" });
  const team = [
    ["Adhikkesh", "CB.SC.U4CSE23101"], ["Sisr Reddy", "CB.SC.U4CSE23129"],
    ["Kavin Karthic", "CB.SC.U4CSE23161"], ["Akash", "CB.SC.U4CSE23162"],
  ];
  team.forEach(([name, reg], i) => {
    const x = 0.8 + i * 3.0;
    card(s, x, 4.4, 2.8, 1.2);
    text(s, name, { x: x + 0.22, y: 4.55, w: 2.4, h: 0.45, fontFace: HEAD, fontSize: 20, bold: true, color: NAVY });
    text(s, reg, { x: x + 0.22, y: 5.0, w: 2.4, h: 0.4, fontSize: 14, color: TEAL, bold: true });
  });
  text(s, "Department of Computer Science and Engineering", { x: 0.8, y: 6.3, w: 11.5, h: 0.35, fontSize: 14, color: MUTED });
  s.addNotes("Adhikkesh: introduce the team and the one-line idea — every elevator car is an autonomous agent that bids for hall calls, and the system is built and tested as an AIMA multi-agent case study.");
}

// 2 ── Problem statement ─────────────────────────────────────────────────────────
{
  const s = content("Problem statement: elevator group control", "Review 1",
    "Adhikkesh: state the problem in plain words, then the four reasons it is hard. End with what we built: a message-driven multi-agent system where each car bids for calls.");
  text(s, "A building has N floors and M lift cars. People arrive at unpredictable times, press a hall button, and reveal where they are going only after they board. The system must decide — every second — which car serves which call and in what order, while cars break down, fire alarms sound and traffic changes through the day.", { x: 0.6, y: 1.35, w: 7.2, h: 1.9, fontSize: 17 });
  bullets(s, [
    "Combinatorial: N cars and k open calls ⇒ N^k possible assignments",
    "Uncertain: arrivals are random; destinations are hidden until boarding",
    "Sequential: today's assignment changes every car's future route",
    "Safety-critical: doors, capacity, faults and fire recall",
  ], { x: 0.6, y: 3.4, w: 7.2, h: 2.3, fontSize: 16 });
  card(s, 0.6, 5.85, 7.2, 0.9, TINT2);
  text(s, "Our approach: each car is an autonomous agent that bids its own cost in a Contract Net auction; the cheapest credible bid wins.", { x: 0.85, y: 5.9, w: 6.8, h: 0.8, fontSize: 15, bold: true, color: TEAL, valign: "middle" });
  stat(s, 8.3, 1.35, 4.4, "4²⁰ ≈ 10¹²", "assignments for just 4 cars and 20 open calls");
  stat(s, 8.3, 3.0, 4.4, "6 agent types", "passenger, floor, elevator, dispatcher, traffic monitor, safety", TEAL);
  stat(s, 8.3, 4.65, 4.4, "9 scenarios", "peaks, lunch, faults, fire, priority, 40-floor stress test");
}

// 3 ── Why classic fails ──────────────────────────────────────────────────────────
{
  const s = content("Why the obvious answer fails", "Review 1",
    "Adhikkesh: 'send the nearest car' ignores what a car is already committed to, so cars bunch and upper floors starve. Our measured morning up-peak numbers on 100 test seeds: the full agent system halves the wait.");
  fitImage(s, FIG("fig11_classic_failure.png"), 0.6, 1.4, 6.8, 4.9, "Morning up-peak average wait for four classical strategies");
  const rows = [
    ["Nearest car", "ignores existing commitments → bunching and starved floors"],
    ["Collective (LOOK)", "sweeps one way; never estimates future cost"],
    ["Fixed rules", "cannot adapt when traffic flips from up-peak to down-peak"],
  ];
  rows.forEach(([h, d], i) => {
    const y = 1.45 + i * 1.3;
    dot(s, 7.8, y, String(i + 1));
    text(s, h, { x: 8.5, y: y - 0.05, w: 4.2, h: 0.4, fontSize: 17, bold: true, color: NAVY });
    text(s, d, { x: 8.5, y: y + 0.35, w: 4.2, h: 0.8, fontSize: 14, valign: "top" });
  });
  card(s, 7.8, 5.4, 4.9, 0.85, TINT2);
  text(s, "−47 % wait vs nearest car (29.9 s vs 56.2 s)", { x: 8.0, y: 5.45, w: 4.6, h: 0.75, fontSize: 16, bold: true, color: TEAL, valign: "middle" });
}

// 4 ── PEAS system ─────────────────────────────────────────────────────────────────
{
  const s = content("PEAS formulation — the whole system", "PEAS · 3M",
    "Adhikkesh: walk through P, E, A, S for the building as a whole. Stress the multi-objective performance measure: minimising energy alone parks every car; minimising average wait alone starves a floor — so fairness is enforced separately by call aging.");
  const peas = [
    ["P", "Performance", "Average, 95th-percentile and maximum wait · ride and system time · % of waits over 60 s · throughput · energy (floors, stops, reversals) · zero safety violations · no starved call"],
    ["E", "Environment", "A multi-storey building: floors with up/down hall buttons and occupancy sensors, shafts, cars, passengers, and the other cars each car competes with"],
    ["A", "Actuators", "Car motors (up / down / stop) · doors (open / close / hold) · hall lanterns and displays · FIPA-ACL messages on the bus"],
    ["S", "Sensors", "Position encoders · load and door sensors · car and hall buttons · per-floor occupancy sensors · the message inbox"],
  ];
  peas.forEach(([k, h, d], i) => {
    const y = 1.4 + i * 1.32;
    card(s, 0.6, y, 12.1, 1.18);
    s.addShape(pres.shapes.OVAL, { x: 0.85, y: y + 0.24, w: 0.7, h: 0.7, fill: { color: NAVY }, line: { color: NAVY } });
    text(s, k, { x: 0.85, y: y + 0.24, w: 0.7, h: 0.7, align: "center", valign: "middle", fontFace: HEAD, fontSize: 24, bold: true, color: WHITE });
    text(s, h, { x: 1.8, y: y + 0.12, w: 2.3, h: 0.9, fontFace: HEAD, fontSize: 19, bold: true, color: NAVY, valign: "middle" });
    text(s, d, { x: 4.1, y: y + 0.1, w: 8.4, h: 0.98, fontSize: 15, valign: "middle" });
  });
}

// 5 ── PEAS per agent ───────────────────────────────────────────────────────────────
{
  const s = content("PEAS of every agent", "PEAS · 3M",
    "Adhikkesh: one row per agent. Point out that each agent's sensors are local — nobody sees everything — which is what makes this genuinely multi-agent.");
  table(s, ["Agent", "Performance", "Environment", "Actuators", "Sensors"], [
    ["Passenger", "own wait, ride, delivery", "its floor, buttons, arriving cars", "press button, board, alight", "car presence, doors, space"],
    ["Floor", "wait of its callers, no starved call", "landing, waiting people", "hall lamps, REQUEST messages", "hall buttons, occupancy sensor"],
    ["Elevator", "calls served cheaply, safe doors", "its shaft, riders, other cars", "motor, doors, PROPOSE bids", "position, load, door, buttons"],
    ["Dispatcher", "fleet wait, fairness, recovery", "all calls and cars", "CFP, ACCEPT/REJECT, parking", "REQUEST, PROPOSE, REFUSE"],
    ["Traffic monitor", "accurate demand estimate", "arrival stream", "INFORM new cost weights", "observed arrivals per floor"],
    ["Safety", "zero violations, correct recall", "loads, doors, faults, alarm", "recall, hold doors, block calls", "load, door, fault, fire alarm"],
  ], { x: 0.6, y: 1.45, w: 12.1, colW: [1.7, 2.6, 2.6, 2.6, 2.6], fontSize: 15 });
  text(s, "Full table with justifications: docs/DESIGN.md §2 · also live on the dashboard's Theory page", { x: 0.6, y: 6.45, w: 12.1, h: 0.35, fontSize: 12, color: MUTED, italic: true });
}

// 6 ── Environment ─────────────────────────────────────────────────────────────────
{
  const s = content("Environment analysis (AIMA classification)", "Env & Agent · 3M",
    "Akash: go dimension by dimension with the justification. The two that shape the design: partially observable (bid before you know the destination, replan on boarding) and dynamic (never trust a cached plan across an event).");
  table(s, ["Dimension", "Classification", "Why"], [
    ["Observability", "Partially observable", "A passenger's destination is private until boarding; each agent sees only its own sensors and inbox"],
    ["Agents", "Multi-agent, cooperative", "Cars bid against each other but share one fleet objective"],
    ["Determinism", "Stochastic", "Poisson arrivals and injected faults (reproducible from a seed)"],
    ["Episodes", "Sequential", "An assignment now changes which car is well placed for later calls"],
    ["Change", "Dynamic", "People keep arriving while agents deliberate"],
    ["Values", "Discrete", "Floors and 1-second ticks, discretised from continuous motion"],
    ["Knowledge", "Known physics, unknown demand", "Travel and door times are known; arrival rates must be learned online"],
  ], { x: 0.6, y: 1.45, w: 12.1, colW: [2.0, 3.0, 7.1], fontSize: 16 });
}

// 7 ── Agent types ─────────────────────────────────────────────────────────────────
{
  const s = content("Agent analysis — why each agent is its type", "Env & Agent · 3M",
    "Akash: the examiner's favourite question. Simple reflex for the passenger; model-based for the floor because it must remember how long a call has waited; goal- and utility-based for the car because it searches for a plan and must trade off wait against energy; learning for the traffic monitor; knowledge-based for safety.");
  const agents = [
    ["Passenger", "Simple reflex", "if a car is here, doors open, my direction, room → board"],
    ["Floor", "Model-based reflex", "remembers active calls and how long they waited (fairness)"],
    ["Elevator", "Goal- + utility-based", "searches for a stop plan; utility trades wait vs energy; bids it"],
    ["Dispatcher", "Utility-based auctioneer", "asks, compares bids, awards — never commands a car"],
    ["Traffic monitor", "Learning agent", "EWMA demand estimate, detects the pattern, retunes weights"],
    ["Safety", "Knowledge-based", "forward-chaining rule base (fire recall, overload)"],
  ];
  agents.forEach(([a, t, d], i) => {
    const col = i % 3, row = Math.floor(i / 3);
    const x = 0.6 + col * 4.1, y = 1.4 + row * 2.6;
    card(s, x, y, 3.9, 2.4);
    text(s, a, { x: x + 0.25, y: y + 0.2, w: 3.4, h: 0.45, fontFace: HEAD, fontSize: 20, bold: true, color: NAVY });
    text(s, t, { x: x + 0.25, y: y + 0.7, w: 3.4, h: 0.4, fontSize: 15, bold: true, color: TEAL });
    text(s, d, { x: x + 0.25, y: y + 1.15, w: 3.4, h: 1.1, fontSize: 14, valign: "top" });
  });
}

// 8 ── Search formulation ─────────────────────────────────────────────────────────
{
  const s = content("Modelling car routing as state-space search", "Search · 3M",
    "Akash: formulate the problem exactly as AIMA does — state, actions, transition, cost, goal — then the heuristic and why it is admissible and consistent. A* therefore returns the optimal stop order.");
  const parts = [
    ["State", "(car floor, direction, set of remaining stops)"],
    ["Actions", "serve one legal next stop (collective control: no backtracking past a stop)"],
    ["Step cost", "travel + door time, weighted by the people waiting at that stop"],
    ["Goal", "no stops remaining"],
  ];
  parts.forEach(([h, d], i) => {
    const y = 1.4 + i * 0.95;
    dot(s, 0.6, y, String(i + 1), NAVY);
    text(s, h, { x: 1.3, y: y + 0.08, w: 1.6, h: 0.4, fontSize: 17, bold: true, color: NAVY, valign: "top" });
    text(s, d, { x: 2.9, y: y + 0.1, w: 4.6, h: 0.75, fontSize: 15, valign: "top" });
  });
  card(s, 7.9, 1.4, 4.8, 2.35);
  text(s, "Heuristic", { x: 8.15, y: 1.5, w: 4.3, h: 0.4, fontSize: 17, bold: true, color: NAVY });
  text(s, "h = Σ wᵢ · travel(car, fᵢ) + energy lower bound of the floor span", { x: 8.15, y: 1.95, w: 4.3, h: 0.8, fontFace: "Cambria Math", fontSize: 15, color: SLATE });
  text(s, "Two lower bounds on disjoint parts of the cost ⇒ h ≤ h* (admissible) and h(n) ≤ c(n,n′) + h(n′) (consistent)", { x: 8.15, y: 2.75, w: 4.3, h: 0.9, fontSize: 13, color: SLATE, valign: "top" });
  card(s, 7.9, 3.95, 4.8, 1.7, TINT2);
  text(s, "Bounded rationality", { x: 8.15, y: 4.05, w: 4.3, h: 0.4, fontSize: 17, bold: true, color: TEAL });
  text(s, "Above 10 pending stops the car falls back to a LOOK sweep — fast and still legal.", { x: 8.15, y: 4.5, w: 4.3, h: 1.05, fontSize: 14, valign: "top" });
  text(s, "Tested over 200 random instances each: A* cost = UCS cost (optimal), A* never expands more nodes than UCS, h admissible and consistent at every node.", { x: 0.6, y: 5.35, w: 6.9, h: 1.1, fontSize: 14, color: MUTED, italic: true });
}

// 9 ── Search comparison ──────────────────────────────────────────────────────────
{
  const s = content("Search strategy: BFS vs UCS vs Greedy vs A*", "Search · 3M",
    "Akash: same 200 random routing problems for all four algorithms. Uninformed BFS ignores cost; UCS is optimal but expands many nodes; greedy is fast but usually wrong; A* is optimal with 39 % fewer expansions than UCS — that is why the cars use A*. Live version in the Algorithm Lab.");
  fitImage(s, FIG("fig12_search_comparison.png"), 0.6, 1.4, 6.4, 4.65, "Nodes expanded and optimality of BFS, UCS, Greedy and A*");
  table(s, ["Algorithm", "Optimal plans", "Cost gap", "Nodes"], [
    ["BFS (uninformed)", "51.5 %", "+9.2 %", "86.8"],
    ["UCS (uninformed)", "100 %", "0 %", "54.1"],
    ["Greedy best-first", "6.0 %", "+44.9 %", "7.7"],
    ["A* (ours)", "100 %", "0 %", "32.8"],
  ], { x: 7.4, y: 1.5, w: 5.3, colW: [2.0, 1.2, 1.0, 1.1], fontSize: 14 });
  card(s, 7.4, 4.3, 5.3, 1.6, TINT2);
  text(s, "A* is optimal like UCS but expands 39 % fewer nodes — the heuristic pays for itself.", { x: 7.65, y: 4.4, w: 4.8, h: 1.4, fontSize: 16, bold: true, color: TEAL, valign: "middle" });
  text(s, "Source: scripts/search_benchmark.py → reports/search_benchmark_summary.csv", { x: 0.6, y: 6.15, w: 12, h: 0.35, fontSize: 12, color: MUTED, italic: true });
}

// 10 ── Optimisation & reasoning ───────────────────────────────────────────────────
{
  const s = content("Beyond search: optimisation, games and rules", "Search · 3M",
    "Akash: three more AIMA techniques, each with a job. Simulated annealing re-optimises all assignments every 30 s; hill climbing and minimax with alpha-beta choose where idle cars park; forward chaining fires the safety rules. Each is visible in the Algorithm Lab.");
  const cols = [
    ["Local search", "Simulated annealing", ["Re-assigns all open calls every 30 s", "Accepts worse moves with probability e^(−Δ/T) to escape local minima", "Tested: never worse than its input"]],
    ["Adversarial search", "Minimax + alpha-beta", ["Parks idle cars against the worst-case next call", "Alpha-beta gives the same value with fewer nodes", "Hill climbing alternative for parking"]],
    ["Knowledge-based", "Forward chaining", ["Safety rules R1–R7: fire recall, door hold, overload, faults", "Fires to a fixed point by salience", "Every action traceable to its rule"]],
  ];
  cols.forEach(([k, h, items], i) => {
    const x = 0.6 + i * 4.1;
    card(s, x, 1.4, 3.9, 4.9);
    text(s, k, { x: x + 0.25, y: 1.55, w: 3.4, h: 0.4, fontSize: 14, bold: true, color: TEAL });
    text(s, h, { x: x + 0.25, y: 1.95, w: 3.4, h: 0.5, fontFace: HEAD, fontSize: 20, bold: true, color: NAVY });
    bullets(s, items, { x: x + 0.25, y: 2.6, w: 3.45, h: 3.5, fontSize: 15 });
  });
}

// 11 ── Tools ──────────────────────────────────────────────────────────────────────
{
  const s = content("Tool and package selection, and setup", "Tools · 3M",
    "Sisr Reddy: one line per tool — why it, and what we rejected. Then setup is two commands: uv sync and uv run elevator serve. uv fetches Python 3.12 itself, so the lab machine's Python version does not matter.");
  table(s, ["Need", "Chosen", "Why", "Rejected"], [
    ["Language & env", "Python 3.12 + uv", "one command installs Python and pinned packages", "pip/venv (no interpreter), conda"],
    ["Agent framework", "Mesa 3", "standard Python agent-based modelling; seeded RNG", "SPADE (needs XMPP server), JADE (Java)"],
    ["Server", "FastAPI + uvicorn", "REST + WebSocket streaming, validation", "Flask, Django"],
    ["Config", "pydantic + YAML", "scenarios as data, clear errors", "hand-written checks"],
    ["Dashboard", "React 18 + TypeScript + Vite", "60 FPS canvas, typed API, fast build", "pygame, Streamlit"],
    ["Analysis", "pandas + matplotlib", "benchmark tables and charts", "hand-rolled stats"],
    ["Quality", "pytest, Vitest, ruff", "property tests over seeded instances", "unittest"],
    ["Learning (extension)", "PyTorch → ONNX Runtime", "train once, run on CPU without PyTorch", "shipping PyTorch to the demo"],
  ], { x: 0.6, y: 1.4, w: 8.6, colW: [1.6, 2.1, 2.7, 2.2], fontSize: 12 });
  card(s, 9.5, 1.4, 3.2, 4.9, TINT2);
  text(s, "Setup", { x: 9.75, y: 1.55, w: 2.8, h: 0.4, fontSize: 18, bold: true, color: TEAL });
  text(s, [
    { text: "uv sync", options: { fontFace: "Courier New", bold: true, color: NAVY, breakLine: true } },
    { text: "installs Python 3.12 + all packages", options: { breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "uv run elevator serve", options: { fontFace: "Courier New", bold: true, color: NAVY, breakLine: true } },
    { text: "opens the dashboard at localhost:8000", options: { breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "uv run pytest", options: { fontFace: "Courier New", bold: true, color: NAVY, breakLine: true } },
    { text: "runs all 451 tests" },
  ], { x: 9.75, y: 2.05, w: 2.8, h: 4.0, fontSize: 14, valign: "top" });
}

// 12 ── Multi-agent execution ──────────────────────────────────────────────────────
{
  const s = content("Multi-agent execution and interaction", "MAS · 3M",
    "Adhikkesh: agents never call each other's methods — they exchange FIPA-ACL messages and read a public status board. Walk through one Contract Net round on the left, then point at the live Agents page screenshot: the graph and message log are real traffic.");
  const steps = [
    ["REQUEST", "a floor reports a new hall call"],
    ["CFP", "dispatcher broadcasts the call to every car"],
    ["PROPOSE / REFUSE", "each car bids its A* marginal cost, or refuses if full or faulty"],
    ["ACCEPT / REJECT", "lowest bid wins; ties → lowest car id"],
    ["INFORM", "winner updates its route and the board"],
  ];
  steps.forEach(([h, d], i) => {
    const y = 1.4 + i * 0.98;
    dot(s, 0.6, y, String(i + 1), NAVY);
    text(s, h, { x: 1.3, y: y - 0.04, w: 4.4, h: 0.4, fontSize: 16, bold: true, color: NAVY });
    text(s, d, { x: 1.3, y: y + 0.34, w: 4.4, h: 0.55, fontSize: 14, valign: "top" });
  });
  text(s, "Staged tick: sense → negotiate → decide → act, so every agent works from one consistent snapshot.", { x: 0.6, y: 6.25, w: 5.2, h: 0.6, fontSize: 13, color: MUTED, italic: true });
  fitImage(s, UI("agents_graph.png"), 6.1, 1.6, 6.6, 4.6, "Agents page: live multi-agent communication network", true);
}

// 13 ── Dashboard demo ─────────────────────────────────────────────────────────────
{
  const s = content("Live demo: Mission Control dashboard", "Demo · 3M",
    "Sisr Reddy: this is what the examiners will see live. Shafts and cars animate at 60 FPS; the auction panel shows each bid; faults and fire alarms can be injected from the chaos panel. Then switch to the live demo.");
  fitImage(s, UI("mission_control.png"), 0.6, 1.35, 8.4, 5.3, "Mission Control: building shafts, cars, KPIs, auction panel and message stream", true);
  const items = [
    ["Mission Control", "shafts, cars, doors, KPIs, live auctions"],
    ["Agents", "agent graph, FIPA-ACL message flow, PEAS"],
    ["Algorithm Lab", "BFS / UCS / Greedy / A* step-through, SA, minimax"],
    ["Experiments", "multi-seed benchmarks, compare mode"],
    ["Theory · Story", "PEAS & environment tables, guided tour"],
  ];
  items.forEach(([h, d], i) => {
    const y = 1.4 + i * 1.04;
    card(s, 9.3, y, 3.4, 0.92);
    text(s, h, { x: 9.5, y: y + 0.06, w: 3.1, h: 0.35, fontSize: 15, bold: true, color: NAVY });
    text(s, d, { x: 9.5, y: y + 0.42, w: 3.1, h: 0.45, fontSize: 12, valign: "top" });
  });
}

// 14 ── Testing scenarios ──────────────────────────────────────────────────────────
{
  const s = content("Testing scenarios and automated tests", "Testing · 3M",
    "Kavin: nine scenarios, each a YAML file, each run to completion and drained — everyone delivered, zero safety violations. Then the automated tests: correctness properties are asserted over hundreds of seeded random instances, not hand-picked cases.");
  table(s, ["Scenario", "What it tests", "Avg wait", "Delivered"], [
    ["morning_up_peak", "lobby rush", "12.1 s", "135 / 135"],
    ["evening_down_peak", "rush to the lobby", "22.4 s", "164 / 164"],
    ["lunch_two_way", "both directions", "12.7 s", "123 / 123"],
    ["interfloor_light", "light mixed traffic", "9.7 s", "52 / 52"],
    ["car_breakdown", "car fault + recovery", "10.9 s", "136 / 136"],
    ["fire_emergency", "fire recall (rules R1–R3, R7)", "71.5 s", "130 / 130"],
    ["priority_passenger", "priority callers", "14.1 s", "138 / 138"],
    ["stress_scale", "40 floors × 8 cars, 1 hour", "18.5 s", "377 / 377"],
    ["demo_story", "surge + fault + fire", "72.4 s", "68 / 68"],
  ], { x: 0.6, y: 1.4, w: 7.6, colW: [2.2, 2.9, 1.2, 1.3], fontSize: 12 });
  stat(s, 8.6, 1.4, 4.1, "451 + 72", "automated tests (Python + dashboard), all passing");
  stat(s, 8.6, 3.0, 4.1, "0", "safety violations — invariants checked every tick of every run", TEAL);
  card(s, 8.6, 4.6, 4.1, 1.75);
  text(s, "Search, optimisation, safety rules, Contract Net protocol, message passing, whole-run invariants, API, scenarios, determinism.", { x: 8.85, y: 4.7, w: 3.6, h: 1.55, fontSize: 13, valign: "top" });
}

// 15 ── Results ─────────────────────────────────────────────────────────────────────
{
  const s = content("Results across traffic patterns", "Testing · 3M",
    "Akash: 100 paired test seeds per pattern. The full multi-agent system is best overall and best by far at morning up-peak; at evening down-peak simple collective control is competitive — reported honestly. Error bars are 95 % confidence intervals.");
  fitImage(s, FIG("fig1_wait_time_comparison.png"), 0.6, 1.3, 12.1, 4.75, "Average wait for six strategies in four traffic patterns");
  text(s, "Up-peak: 29.9 s (full) vs 56.2 s (nearest car)  ·  every scenario delivered everyone  ·  0 safety violations in 7,800 runs", { x: 0.6, y: 6.2, w: 12.1, h: 0.5, fontSize: 15, bold: true, color: NAVY });
}

// 16 ── Code structure & scalability ────────────────────────────────────────────────
{
  const s = content("Code structure and scalability", "Code · 1M",
    "Sisr Reddy: the engine is independent of the server and UI; everything changeable is YAML; strategies plug in through a registry. The 40-floor, 8-car stress scenario is only a config file and simulates an hour in under 4 seconds.");
  const mods = [
    ["agents/", "the six agents"], ["comms/", "messages, bus, status board"],
    ["planning/", "BFS / UCS / Greedy / A*, routing"], ["optimization/", "SA, hill climbing, minimax"],
    ["rules/", "forward-chaining safety rules"], ["strategies/", "pluggable dispatch strategies"],
    ["traffic/ · sim/", "arrivals, headless runs, benchmark"], ["api/ · web/", "FastAPI server, React dashboard"],
  ];
  mods.forEach(([m, d], i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 0.6 + col * 3.6, y = 1.4 + row * 1.1;
    card(s, x, y, 3.45, 0.95);
    text(s, m, { x: x + 0.2, y: y + 0.08, w: 3.1, h: 0.38, fontFace: "Courier New", fontSize: 14, bold: true, color: NAVY });
    text(s, d, { x: x + 0.2, y: y + 0.47, w: 3.1, h: 0.4, fontSize: 13 });
  });
  text(s, "src/elevator_mas/ — the engine never imports the server or UI", { x: 0.6, y: 5.85, w: 7.1, h: 0.4, fontSize: 13, color: MUTED, italic: true });
  stat(s, 8.0, 1.4, 4.7, "< 4 s", "to simulate one hour of a 40-floor, 8-car building (stress_scale)");
  stat(s, 8.0, 3.0, 4.7, "YAML only", "floors, cars, timings, traffic, faults, strategy — no code change", TEAL);
  stat(s, 8.0, 4.6, 4.7, "Registry", "a new dispatch strategy is one entry; agents unchanged");
}

// 17 ── LiftZero extension ───────────────────────────────────────────────────────────
{
  const s = content("Extension: LiftZero — a learned bidder", null,
    "Kavin Karthic: our extension beyond the syllabus. A small neural network learned to imitate the A* bid from 1.5 million recorded decisions; it agrees with A* about 90 % of the time, prices a call in 0.19 ms and keeps the same waiting times. The Brain page shows its bids and a 'Why?' explanation live. Optional look-ahead search can override close calls.");
  fitImage(s, UI("brain_panel.png"), 0.6, 1.35, 7.4, 5.3, "LiftZero Brain page: learned bids, attention and look-ahead search", true);
  stat(s, 8.4, 1.35, 4.3, "89.9 %", "agreement with the A* bid on unseen test decisions");
  stat(s, 8.4, 2.95, 4.3, "0.19 ms", "per decision on a laptop CPU", TEAL);
  stat(s, 8.4, 4.55, 4.3, "13 / 13", "scenarios within ±5 % of A*'s waiting time (plain Contract Net)");
}

// 18 ── Conclusion ───────────────────────────────────────────────────────────────────
{
  const s = content("Conclusion", null,
    "Kavin Karthic: summarise against the rubric — PEAS and environment analysis, search and optimisation, tools, multi-agent execution, testing, structure — then invite questions. Each member answers on their own slides.");
  stat(s, 0.6, 1.4, 3.85, "−47 %", "up-peak wait vs nearest car");
  stat(s, 4.73, 1.4, 3.85, "A* optimal", "with 39 % fewer nodes than UCS", TEAL);
  stat(s, 8.85, 1.4, 3.85, "0", "safety violations across all scenarios");
  text(s, "What we delivered", { x: 0.6, y: 3.15, w: 6, h: 0.45, fontSize: 18, bold: true, color: NAVY });
  bullets(s, [
    "Six agent types with PEAS, communicating only by FIPA-ACL messages",
    "Contract Net auctions with A* bids, annealing, minimax parking, safety rules",
    "Live dashboard, 9 test scenarios, 523 automated tests",
  ], { x: 0.6, y: 3.65, w: 6.1, h: 2.4, fontSize: 15 });
  text(s, "Future work", { x: 6.9, y: 3.15, w: 5.8, h: 0.45, fontSize: 18, bold: true, color: NAVY });
  bullets(s, ["Reinforcement learning on the real objective", "Real building traffic traces", "Multiple lobbies and express zones"], { x: 6.9, y: 3.65, w: 5.8, h: 1.8, fontSize: 15 });
  card(s, 6.9, 5.55, 5.8, 0.85, TINT2);
  text(s, "Thank you — questions?", { x: 7.1, y: 5.6, w: 5.4, h: 0.75, fontFace: HEAD, fontSize: 22, bold: true, color: TEAL, valign: "middle" });
}

pres.writeFile({ fileName: OUT }).then(() => console.log(`wrote ${OUT} (${n} slides)`));
