// Build the LiftZero viva deck (14 slides, white background) from the generated charts.
//   node scripts/build_viva_deck.js            (needs `npm install pptxgenjs` once)
// Output: docs/LiftZero_Viva_Deck.pptx. Charts come from reports/viva/*.png
// (scripts/generate_viva_charts.py), i.e. from real simulation results only.
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const FIG = (n) => path.join(ROOT, "reports", "viva", n);
const OUT = path.join(ROOT, "docs", "LiftZero_Viva_Deck.pptx");

const NAVY = "1F3A5F", TEAL = "0F7C7A", SLATE = "4A5560", MUTED = "6B7680";
const TINT = "F2F5F8", TINT2 = "E8F3F2", LINE = "D5DBE1", WHITE = "FFFFFF";
const HEAD = "Cambria", BODY = "Calibri";

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
function fitImage(slide, file, x, y, maxW, maxH, alt) {
  const { w, h } = pngSize(file);
  let iw = maxW, ih = (maxW * h) / w;
  if (ih > maxH) { ih = maxH; iw = (maxH * w) / h; }
  slide.addImage({ path: file, x: x + (maxW - iw) / 2, y, w: iw, h: ih, altText: alt });
}

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
pres.title = "LiftZero: Multi-Agent Smart Elevator Fleet Coordinator";
pres.author = "Adhikkesh, Sisr Reddy, Kavin Karthic, Akash";
pres.subject = "FOAI Semester 7 capstone viva";
pres.theme = { headFontFace: HEAD, bodyFontFace: BODY };

pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: WHITE },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.8,
      fontFace: HEAD, fontSize: 30, bold: true, color: NAVY, valign: "middle", margin: 0 },
      text: "" } },
    { text: { text: "LiftZero  ·  FOAI Semester 7", options: { x: 0.6, y: 7.0, w: 6, h: 0.3,
      fontFace: BODY, fontSize: 10, color: MUTED, margin: 0 } } },
  ],
  slideNumber: { x: 12.2, y: 7.0, w: 0.5, h: 0.3, fontFace: BODY, fontSize: 10, color: MUTED },
});

let n = 0;
function content(title, notes) {
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText(title, { placeholder: "title" });
  if (notes) s.addNotes(notes);
  n += 1;
  return s;
}
function text(s, t, o) { s.addText(t, { isTextBox: true, fontFace: BODY, color: SLATE, margin: 0, ...o }); }
function card(s, x, y, w, h, fill = TINT) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: fill }, rectRadius: 0.08 });
}
function stat(s, x, y, w, big, label, color = NAVY) {
  card(s, x, y, w, 1.45);
  text(s, big, { x: x + 0.25, y: y + 0.15, w: w - 0.5, h: 0.75, fontFace: HEAD, fontSize: 34, bold: true, color });
  text(s, label, { x: x + 0.25, y: y + 0.88, w: w - 0.5, h: 0.5, fontSize: 13, color: SLATE });
}
function bullets(s, items, o) {
  s.addText(items.map((t, i) => ({ text: t, options: { bullet: { indent: 18 }, breakLine: i < items.length - 1 } })),
    { isTextBox: true, fontFace: BODY, fontSize: 16, color: SLATE, paraSpaceAfter: 8, valign: "top", margin: 0, ...o });
}
function numberDot(s, x, y, label, fill = TEAL) {
  s.addShape(pres.shapes.OVAL, { x, y, w: 0.5, h: 0.5, fill: { color: fill }, line: { color: fill } });
  text(s, label, { x, y, w: 0.5, h: 0.5, align: "center", valign: "middle", fontSize: 14, bold: true, color: WHITE });
}

// 1 ── Title ───────────────────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  n += 1;
  text(s, "FOUNDATIONS OF ARTIFICIAL INTELLIGENCE  ·  SEMESTER 7 CAPSTONE", { x: 0.8, y: 0.8, w: 11.5, h: 0.4, fontSize: 13, bold: true, color: TEAL, charSpacing: 2 });
  text(s, "LiftZero", { x: 0.8, y: 1.35, w: 11.5, h: 1.2, fontFace: HEAD, fontSize: 60, bold: true, color: NAVY });
  text(s, "Multi-Agent Smart Elevator Fleet Coordinator", { x: 0.8, y: 2.5, w: 11.5, h: 0.6, fontFace: HEAD, fontSize: 28, color: NAVY });
  text(s, "Contract Net agents · A* search · imitation-learned bidding · cooperative RL", { x: 0.8, y: 3.15, w: 11.5, h: 0.45, fontSize: 18, color: SLATE });
  const team = [
    ["Adhikkesh", "CB.SC.U4CSE23101", "Architecture & multi-agent coordination"],
    ["Sisr Reddy", "CB.SC.U4CSE23129", "Mission Control & frontend"],
    ["Kavin Karthic", "CB.SC.U4CSE23161", "Simulation environments & imitation learning"],
    ["Akash", "CB.SC.U4CSE23162", "Deep RL, search & evaluation"],
  ];
  team.forEach(([name, reg, role], i) => {
    const x = 0.8 + i * 3.0;
    card(s, x, 4.35, 2.8, 1.75);
    text(s, name, { x: x + 0.22, y: 4.5, w: 2.4, h: 0.45, fontFace: HEAD, fontSize: 20, bold: true, color: NAVY });
    text(s, reg, { x: x + 0.22, y: 4.95, w: 2.4, h: 0.35, fontSize: 14, color: TEAL, bold: true });
    text(s, role, { x: x + 0.22, y: 5.3, w: 2.4, h: 0.7, fontSize: 13, color: SLATE, valign: "top" });
  });
  text(s, "Department of Computer Science and Engineering", { x: 0.8, y: 6.55, w: 11.5, h: 0.35, fontSize: 14, color: MUTED });
  s.addNotes("Presenter: Adhikkesh. Introduce the team and the one-line idea: every elevator car is an autonomous agent that bids for hall calls; we then learn how to bid.");
}

// 2 ── Problem ────────────────────────────────────────────────────────────────────
{
  const s = content("The Elevator Group Control Problem (EGCP)",
    "Presenter: Adhikkesh. Define EGCP: assign every hall call to one car, online, under uncertainty about future arrivals and destinations. Stress the combinatorics (N^k) and that the objective trades wait time, fairness and energy.");
  text(s, "Every time someone presses a hall button, the fleet must decide — in real time — which car serves it. Decisions interact: today's assignment shapes every car's future route.", { x: 0.6, y: 1.35, w: 7.0, h: 1.2, fontSize: 18 });
  bullets(s, [
    "Online: future passengers and their destinations are unknown (partially observable, stochastic)",
    "Combinatorial: with N cars and k open calls there are N^k assignments",
    "Multi-objective: average wait, worst-case (p95) wait, ride time, energy",
    "Safety-critical: doors, capacity, faults and fire recall must never be violated",
  ], { x: 0.6, y: 2.75, w: 7.0, h: 3.6 });
  stat(s, 8.2, 1.4, 4.5, "4²⁰ ≈ 10¹²", "assignments for 4 cars and 20 open calls");
  stat(s, 8.2, 3.05, 4.5, "1 s", "decision cycle: one simulation tick = one second", TEAL);
  stat(s, 8.2, 4.7, 4.5, "6 + N", "cooperating agents per building (N cars)");
}

// 3 ── Why classic approaches fail ─────────────────────────────────────────────────
{
  const s = content("Why classic dispatchers fail at peak hours",
    "Presenter: Adhikkesh. Use the measured up-peak numbers: nearest car ignores what a car is already committed to, so cars bunch and upper floors starve; collective control avoids reversing but still has no look-ahead. Planning (A*) and global re-optimisation halve the wait.");
  s.addChart(pres.charts.BAR, [{ name: "Average wait (s)", labels: ["Nearest car", "Collective", "CNP + A*", "Full (CNP+A*+SA)"], values: [56.2, 47.1, 42.9, 29.9] }], {
    x: 0.6, y: 1.35, w: 6.6, h: 4.9, barDir: "bar", chartColors: ["2A78D6", "EB6834", "1BAF7A", "EDA100"], varyColors: true,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 12, dataLabelColor: "3A3A3A", dataLabelFontFace: BODY, dataLabelFormatCode: "0.0",
    catAxisLabelFontFace: BODY, valAxisLabelFontFace: BODY, catAxisLabelFontSize: 13, valAxisLabelFontSize: 11,
    catAxisLabelColor: "4A5560", valAxisLabelColor: "6B7680", valGridLine: { color: "E6E5E0", size: 0.75 }, catGridLine: { style: "none" },
    showLegend: false, showTitle: true, title: "Morning up-peak: average wait (s), 100 test seeds", titleFontFace: BODY, titleFontSize: 13, titleColor: "1F3A5F",
    catAxisOrientation: "maxMin", valAxisMinVal: 0,
  });
  const rows = [
    ["Nearest car", "Ignores each car's existing commitments → bunching, upper floors starve"],
    ["Collective (LOOK)", "Sweeps in one direction; no estimate of future cost"],
    ["Fixed rules", "Cannot adapt when traffic flips from up-peak to lunch to down-peak"],
  ];
  rows.forEach(([h, d], i) => {
    const y = 1.45 + i * 1.3;
    numberDot(s, 7.6, y, String(i + 1));
    text(s, h, { x: 8.3, y: y - 0.05, w: 4.4, h: 0.4, fontSize: 17, bold: true, color: NAVY });
    text(s, d, { x: 8.3, y: y + 0.35, w: 4.4, h: 0.8, fontSize: 14, valign: "top" });
  });
  card(s, 7.6, 5.4, 5.1, 0.85, TINT2);
  text(s, "Full system cuts up-peak wait by 47 % vs nearest car (29.9 s vs 56.2 s)", { x: 7.8, y: 5.45, w: 4.8, h: 0.75, fontSize: 15, bold: true, color: TEAL, valign: "middle" });
}

// 4 ── MAS architecture ──────────────────────────────────────────────────────────
{
  const s = content("Multi-agent architecture and the Contract Net Protocol",
    "Presenter: Adhikkesh. Six agent types, each with a PEAS description and an AIMA agent type. Agents never call each other's methods: they exchange FIPA-ACL messages through a bus and read a public status board. Walk through one Contract Net round on the right.");
  const agents = [
    ["Elevator (×N)", "utility-based: bids marginal cost, A* routing"],
    ["Dispatcher", "auctioneer: CFP, award, SA re-optimisation"],
    ["Floor", "senses waiting passengers, raises hall calls"],
    ["Traffic monitor", "learning agent: detects pattern, retunes weights"],
    ["Safety", "rule-based: forward-chaining safety rules"],
    ["Passenger", "simple reflex: board, ride, alight"],
  ];
  agents.forEach(([a, d], i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 0.6 + col * 3.1, y = 1.4 + row * 1.6;
    card(s, x, y, 2.9, 1.4);
    text(s, a, { x: x + 0.2, y: y + 0.15, w: 2.5, h: 0.4, fontSize: 16, bold: true, color: NAVY });
    text(s, d, { x: x + 0.2, y: y + 0.58, w: 2.5, h: 0.75, fontSize: 13, valign: "top" });
  });
  const steps = [
    ["CFP", "Dispatcher broadcasts the hall call (floor, direction, urgency)"],
    ["PROPOSE / REFUSE", "Each car prices the call from its own state; full or faulty cars refuse"],
    ["ACCEPT / REJECT", "Lowest bid wins (ties → lowest car id)"],
    ["INFORM", "Winner updates its route; board and dashboard update"],
  ];
  text(s, "One auction round (FIPA-ACL)", { x: 7.1, y: 1.4, w: 5.6, h: 0.4, fontSize: 17, bold: true, color: TEAL });
  steps.forEach(([h, d], i) => {
    const y = 1.95 + i * 1.18;
    numberDot(s, 7.1, y, String(i + 1), NAVY);
    text(s, h, { x: 7.8, y: y - 0.04, w: 4.9, h: 0.4, fontSize: 16, bold: true, color: NAVY });
    text(s, d, { x: 7.8, y: y + 0.36, w: 4.9, h: 0.7, fontSize: 14, valign: "top" });
  });
}

// 5 ── Physical model ──────────────────────────────────────────────────────────────
{
  const s = content("Physical model, search and safety",
    "Presenter: Adhikkesh. Explain the discrete-time kinematics (1 tick = 1 s), the A* stop-sequencing problem with an admissible heuristic, and the safety invariants checked every tick. Mention that our 100-seed evaluation found and fixed a fire-recall bug (car counted as moving with doors open).");
  const cols = [
    ["Kinematics", ["1 tick = 1 simulated second", "2 s per floor, 2 s door open / close", "1 s boarding per passenger, capacity 6–16", "Door states: closed → opening → open → closing"]],
    ["Routing as search", ["State: car floor, direction, remaining stops", "A* with an admissible, consistent heuristic", "Bid = marginal cost of inserting the call", "LOOK fallback above 10 stops (bounded rationality)"]],
    ["Safety", ["Never move with doors open", "Never exceed capacity or floor bounds", "Fire recall to lobby, fault ejection", "Invariants checked every tick of every run"]],
  ];
  cols.forEach(([h, items], i) => {
    const x = 0.6 + i * 4.1;
    card(s, x, 1.4, 3.9, 4.5);
    text(s, h, { x: x + 0.25, y: 1.55, w: 3.4, h: 0.5, fontFace: HEAD, fontSize: 20, bold: true, color: NAVY });
    bullets(s, items, { x: x + 0.25, y: 2.2, w: 3.45, h: 3.5, fontSize: 15 });
  });
  card(s, 0.6, 6.1, 12.1, 0.7, TINT2);
  text(s, "Evaluation at scale found a real safety bug (fire alarm mid-move); fixed and pinned by regression tests — 0 violations in 7,800 test runs", { x: 0.85, y: 6.15, w: 11.7, h: 0.6, fontSize: 15, bold: true, color: TEAL, valign: "middle" });
}

// 6 ── Features & learning environment ────────────────────────────────────────────
{
  const s = content("Feature engineering and the learning environment",
    "Presenter: Kavin Karthic. A decision is encoded as tokens: one call token, one global token, one token per car — so the network sees a set of cars of any size. Schema v2 added the public cost weights, which removed conflicting labels between the two teachers. Be honest about the twin: measured 3.7x faster than the real simulator and not faithful, so RL trains on the real simulator.");
  const toks = [["Call token", "14", "floor, direction, waiting, urgency, traffic pattern"], ["Car token (each car)", "26", "position, direction, load, doors, plan summary, geometry to the call"], ["Global token", "10", "time, building, fleet load, public cost weights W1–W4"]];
  toks.forEach(([h, k, d], i) => {
    const y = 1.4 + i * 1.35;
    card(s, 0.6, y, 6.6, 1.2);
    text(s, k, { x: 0.8, y: y + 0.15, w: 1.1, h: 0.9, fontFace: HEAD, fontSize: 36, bold: true, color: TEAL, valign: "middle" });
    text(s, h, { x: 2.0, y: y + 0.12, w: 5.0, h: 0.4, fontSize: 16, bold: true, color: NAVY });
    text(s, d, { x: 2.0, y: y + 0.52, w: 5.0, h: 0.6, fontSize: 14, valign: "top" });
  });
  text(s, "Features are public only: what any car can read from the status board.", { x: 0.6, y: 5.55, w: 6.6, h: 0.5, fontSize: 14, color: MUTED, italic: true });
  stat(s, 7.7, 1.4, 5.0, "1.47 M", "expert decisions recorded from the real simulator (train / val / test)");
  stat(s, 7.7, 3.05, 5.0, "2 M", "extra learner-state decisions from two DAgger rounds", TEAL);
  stat(s, 7.7, 4.7, 5.0, "3.7×", "measured twin speed-up only → RL trains on the real simulator");
}

// 7 ── BC + DAgger ─────────────────────────────────────────────────────────────────
{
  const s = content("LiftZero-BC: a set-Transformer that learns to bid",
    "Presenter: Kavin Karthic. The network replaces only the bid computation. It is permutation-equivariant (no car ids), 237k parameters. Behaviour cloning on 595k decisions reached 88.3% agreement over 3 seeds; DAgger fixed the one closed-loop regime that was outside the 5% band. ONNX inference is 0.19 ms per decision.");
  fitImage(s, FIG("fig5_imitation_learning_curves.png"), 0.6, 1.35, 7.6, 3.6, "BC training loss and validation agreement over epochs for 3 seeds");
  stat(s, 8.6, 1.35, 4.1, "88.3 ± 0.4 %", "validation agreement, 3 seeds (gate 85 %)");
  stat(s, 8.6, 2.95, 4.1, "89.9 %", "agreement on unseen test decisions", TEAL);
  stat(s, 8.6, 4.55, 4.1, "0.19 ms", "CPU inference per decision (8 cars)");
  bullets(s, [
    "4-layer set-Transformer, 237,575 parameters, permutation-equivariant over cars",
    "Loss: listwise CE + log-cost regression + rank + pairwise hinge on hard decisions",
    "DAgger: learner drives, A* teacher labels — closed-loop gap shrank from +6.2 % to +1.1 %",
  ], { x: 0.6, y: 5.2, w: 7.6, h: 1.6, fontSize: 15 });
}

// 8 ── PPO ─────────────────────────────────────────────────────────────────────────
{
  const s = content("LiftZero-PPO: cooperative reinforcement learning (CTDE)",
    "Presenter: Akash. Imitation can only match the teacher; RL optimises the real objective. Explain the SMDP (irregular decision times, gamma^(dt/5)), the shared team reward (cooperative), the asymmetric critic that sees privileged state during training only, and the KL anchor that keeps the policy close to the imitation policy. Status: implemented and unit-tested; full training runs on Kaggle.");
  card(s, 0.6, 1.4, 6.3, 2.2);
  text(s, "Objective", { x: 0.85, y: 1.5, w: 5.8, h: 0.4, fontSize: 17, bold: true, color: NAVY });
  text(s, "L = −E[min(ρA, clip(ρ, 1±ε)A)] + c_v·(V_priv − R)² − c_e·H[π] + β·KL(π_BC ‖ π)", { x: 0.85, y: 1.95, w: 5.8, h: 0.8, fontFace: "Cambria Math", fontSize: 15, color: SLATE });
  text(s, "SMDP discount γ^(Δt/τ), GAE λ = 0.95, β: 0.2 → 0", { x: 0.85, y: 2.85, w: 5.8, h: 0.5, fontSize: 14, color: MUTED });
  bullets(s, [
    "Team reward: −(Σ waiting time + ½ Σ riding time)/100 − 0.02·floors − 0.5·long waits",
    "Decentralised execution: every car runs the same public-information actor",
    "Centralised training: the critic also sees per-floor queues and waiting ages",
    "Trains directly on the real simulator (no sim-to-real gap)",
  ], { x: 0.6, y: 3.85, w: 6.3, h: 2.9, fontSize: 15 });
  card(s, 7.4, 1.4, 5.3, 2.6, TINT2);
  text(s, "Status", { x: 7.65, y: 1.5, w: 4.8, h: 0.4, fontSize: 17, bold: true, color: TEAL });
  text(s, "Implemented from scratch and unit-tested (GAE, clipping, KL, critic isolation, seed guards). End-to-end smoke run passes. Full training runs on Kaggle; results are added once available.", { x: 7.65, y: 1.95, w: 4.8, h: 1.95, fontSize: 15, valign: "top" });
  const pts = [["1", "Warm start from LiftZero-BC"], ["2", "Collect 8k decisions in parallel workers"], ["3", "Advantages from the privileged critic"], ["4", "4 epochs of clipped updates, KL early stop"]];
  pts.forEach(([k, t], i) => {
    const y = 4.3 + i * 0.62;
    numberDot(s, 7.4, y, k, NAVY);
    text(s, t, { x: 8.1, y: y + 0.03, w: 4.6, h: 0.45, fontSize: 15, valign: "middle" });
  });
}

// 9 ── MCTS ─────────────────────────────────────────────────────────────────────────
{
  const s = content("Look-ahead arbitration with PUCT Monte-Carlo tree search",
    "Presenter: Akash. This is the designed next stage: the dispatcher, which sees all bids, runs a short look-ahead over contested decisions using the network as prior and value. It is not implemented yet — say so clearly and present the design and the formula.");
  card(s, 0.6, 1.4, 6.3, 1.7);
  text(s, "Selection rule", { x: 0.85, y: 1.5, w: 5.8, h: 0.4, fontSize: 17, bold: true, color: NAVY });
  text(s, "a* = argmaxₐ [ Q(s,a) + c_puct · P(s,a) · √(Σ_b N(s,b)) / (1 + N(s,a)) ]", { x: 0.85, y: 1.98, w: 5.8, h: 0.9, fontFace: "Cambria Math", fontSize: 16, color: SLATE });
  bullets(s, [
    "Prior P from the LiftZero network, leaf value from its public value head",
    "Only cars that proposed are candidates; refusals stay classical",
    "Hidden passenger destinations are sampled (determinisation, partial observability)",
    "Hard real-time budget: ≤ 50 ms per contested decision",
  ], { x: 0.6, y: 3.35, w: 6.3, h: 3.2, fontSize: 15 });
  card(s, 7.4, 1.4, 5.3, 2.0, TINT2);
  text(s, "Status: designed, not yet implemented", { x: 7.65, y: 1.5, w: 4.8, h: 0.4, fontSize: 17, bold: true, color: TEAL });
  text(s, "The reflex learner (bids) and the deliberative planner (search) form the AIMA contrast between reflex and model-based agents in one system.", { x: 7.65, y: 1.95, w: 4.8, h: 1.35, fontSize: 15, valign: "top" });
  const why = [["Reflex", "learned bid: 0.19 ms, no look-ahead"], ["Deliberative", "search: simulates the next minute"], ["Hybrid", "search only when bids are close"]];
  why.forEach(([h, d], i) => {
    const y = 3.75 + i * 0.95;
    card(s, 7.4, y, 5.3, 0.8);
    text(s, h, { x: 7.6, y: y + 0.08, w: 1.6, h: 0.64, fontSize: 15, bold: true, color: NAVY, valign: "middle" });
    text(s, d, { x: 9.2, y: y + 0.08, w: 3.4, h: 0.64, fontSize: 14, valign: "middle" });
  });
}

// 10 ── Dashboard ───────────────────────────────────────────────────────────────────
{
  const s = content("Mission Control: the live React dashboard",
    "Presenter: Sisr Reddy. Launch with `uv run elevator serve` and open localhost:8000. Show the shaft canvas, the auction theatre (live CFP/PROPOSE/ACCEPT with bid breakdown), compare mode, and story mode. Switch the strategy to LiftZero-BC: the decision trace shows the network's bids.");
  const tabs = [
    ["Mission Control", "60 FPS canvas of shafts, cars, doors and passengers; live KPI sparklines"],
    ["Auction theatre", "every Contract Net round: bids, breakdown, winner and reason"],
    ["Agents", "agent graph, FIPA-ACL swimlanes, PEAS inspector"],
    ["Search lab", "BFS / UCS / Greedy / A* step-through, SA and minimax demos"],
    ["Experiments", "multi-seed benchmarks and side-by-side compare mode"],
    ["Story mode", "guided 10-beat walkthrough for examiners"],
  ];
  tabs.forEach(([h, d], i) => {
    const col = i % 3, row = Math.floor(i / 3);
    const x = 0.6 + col * 4.1, y = 1.45 + row * 2.0;
    card(s, x, y, 3.9, 1.8);
    numberDot(s, x + 0.25, y + 0.25, String(i + 1));
    text(s, h, { x: x + 0.9, y: y + 0.25, w: 2.8, h: 0.5, fontSize: 17, bold: true, color: NAVY, valign: "middle" });
    text(s, d, { x: x + 0.25, y: y + 0.88, w: 3.45, h: 0.85, fontSize: 14, valign: "top" });
  });
  card(s, 0.6, 5.65, 12.1, 0.9, TINT2);
  text(s, "React 19 + TypeScript, WebSocket streaming from FastAPI · 66 Vitest unit tests · LiftZero strategies selectable live", { x: 0.85, y: 5.7, w: 11.7, h: 0.8, fontSize: 15, bold: true, color: TEAL, valign: "middle" });
}

// 11 ── Results ─────────────────────────────────────────────────────────────────────
{
  const s = content("Results: average wait across traffic patterns",
    "Presenter: Akash. 100 paired test seeds per regime in the real simulator. The full multi-agent system is strongest overall; LiftZero-BC tracks its teacher closely. Point out down-peak, where simple collective control is competitive — the motivation for RL.");
  fitImage(s, FIG("fig1_wait_time_comparison.png"), 0.6, 1.3, 12.1, 4.75, "Grouped bar chart of average wait for six strategies in four traffic patterns");
  text(s, "Up-peak: Full 29.9 s vs Nearest 56.2 s (−47 %)  ·  LiftZero-BC within ±3 % of its teacher in 3 of 4 patterns  ·  0 safety violations in 7,800 runs", { x: 0.6, y: 6.2, w: 12.1, h: 0.5, fontSize: 15, bold: true, color: NAVY });
}

// 12 ── Ablation ────────────────────────────────────────────────────────────────────
{
  const s = content("Does the learned bidder match the A* teacher?",
    "Presenter: Akash. Each learned strategy differs from its teacher only in the bidder (single-variable comparison). The bare-CNP pair is within the ±5% band in all 13 regimes and scenarios; the full-system pair misses in three, where passenger streams diverge because annealing shares the random stream — wide confidence intervals.");
  fitImage(s, FIG("fig4_learned_vs_teacher.png"), 0.6, 1.3, 7.4, 5.5, "Forest plot of percentage difference in average wait between learned and teacher strategies");
  stat(s, 8.4, 1.4, 4.3, "13 / 13", "regimes within ±5 % for the bare-CNP learned bidder");
  stat(s, 8.4, 3.0, 4.3, "10 / 13", "regimes within ±5 % for LiftZero-BC vs Full", TEAL);
  card(s, 8.4, 4.6, 4.3, 2.0);
  text(s, "Ablation ladder (up-peak wait)", { x: 8.6, y: 4.7, w: 3.9, h: 0.4, fontSize: 15, bold: true, color: NAVY });
  text(s, "Nearest 56.2 → Collective 47.1 → CNP+A* 42.9 → Full 29.9 → LiftZero-BC 31.6 s", { x: 8.6, y: 5.12, w: 3.9, h: 1.4, fontSize: 14, valign: "top" });
}

// 13 ── Team ────────────────────────────────────────────────────────────────────────
{
  const s = content("Team contributions",
    "Each member presents their own slides: Adhikkesh 1–5, Kavin 6–7, Akash 8–9 and 11–12, Sisr 10 and the live demo; everyone fields questions on their modules.");
  const team = [
    ["Adhikkesh", "CB.SC.U4CSE23101", ["Mesa 3 multi-agent engine", "FIPA-ACL + Contract Net", "Kinematics, A* routing, safety", "Slides 1–5, demo launch"]],
    ["Sisr Reddy", "CB.SC.U4CSE23129", ["React 19 Mission Control", "60 FPS canvas, auction theatre", "Compare & story modes", "Slide 10, live UI demo"]],
    ["Kavin Karthic", "CB.SC.U4CSE23161", ["Feature schema, expert recorder", "Set-Transformer, BC + DAgger", "ONNX runtime, learned bidder", "Slides 6–7, CLI demo"]],
    ["Akash", "CB.SC.U4CSE23162", ["Cooperative PPO (CTDE)", "Search design (PUCT-MCTS)", "Benchmarks & statistics", "Slides 8–9, 11–12"]],
  ];
  team.forEach(([name, reg, items], i) => {
    const x = 0.6 + i * 3.08;
    card(s, x, 1.4, 2.9, 5.2);
    text(s, name, { x: x + 0.22, y: 1.55, w: 2.5, h: 0.5, fontFace: HEAD, fontSize: 21, bold: true, color: NAVY });
    text(s, reg, { x: x + 0.22, y: 2.05, w: 2.5, h: 0.4, fontSize: 14, bold: true, color: TEAL });
    bullets(s, items, { x: x + 0.22, y: 2.65, w: 2.5, h: 3.8, fontSize: 14 });
  });
}

// 14 ── Conclusion ──────────────────────────────────────────────────────────────────
{
  const s = content("Conclusion and next steps",
    "Presenter: Adhikkesh (with all). Summarise: a working multi-agent system with search and safety, a learned bidder that matches the A* teacher at a fraction of the compute, and an RL pipeline ready to train. Then invite questions.");
  stat(s, 0.6, 1.4, 3.85, "−47 %", "up-peak wait vs nearest car (full system)");
  stat(s, 4.73, 1.4, 3.85, "0.19 ms", "learned bid vs an A* search per car", TEAL);
  stat(s, 8.85, 1.4, 3.85, "0", "safety violations in 7,800 test runs");
  text(s, "What we built", { x: 0.6, y: 3.15, w: 5.8, h: 0.45, fontSize: 18, bold: true, color: NAVY });
  bullets(s, [
    "Message-driven multi-agent system with Contract Net, A* and safety rules",
    "Imitation-learned bidder (BC + DAgger) matching its teacher, shipped as ONNX",
    "Live dashboard and a reproducible evaluation pipeline (444 + 66 tests)",
  ], { x: 0.6, y: 3.65, w: 5.9, h: 2.6, fontSize: 15 });
  text(s, "Next steps", { x: 6.9, y: 3.15, w: 5.8, h: 0.45, fontSize: 18, bold: true, color: NAVY });
  bullets(s, [
    "Complete PPO training on GPU and test against the teacher",
    "Implement PUCT look-ahead for contested calls",
    "Real building traces and multi-lobby buildings",
  ], { x: 6.9, y: 3.65, w: 5.8, h: 2.0, fontSize: 15 });
  card(s, 6.9, 5.75, 5.8, 0.85, TINT2);
  text(s, "Thank you — questions?", { x: 7.1, y: 5.8, w: 5.4, h: 0.75, fontFace: HEAD, fontSize: 22, bold: true, color: TEAL, valign: "middle" });
}

pres.writeFile({ fileName: OUT }).then(() => console.log(`wrote ${OUT} (${n} slides)`));
