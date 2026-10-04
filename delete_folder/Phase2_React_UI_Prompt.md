# PHASE 2 — LiftZero Control Room: a React UI for the Elevator Fleet Simulator

> Paste this whole file into Antigravity as the task (Planning mode, your strongest model).
> Everything you need is in here. Where a number or a path is given, it was read from the
> real code at commit `2ef99c0` on branch `v2` — do not guess a different one.

---

## 0. Who you are working for, and what "done" means

You are building the front end for a university AI case study (course 23CSE401 *Fundamentals
of AI*, AIMA 4th ed.): **a multi-agent smart-elevator fleet coordinator.** It will be
graded by examiners who watch a live demo and ask viva questions. So the UI has two jobs:

1. **Make the multi-agent system visible**: cars, floors, a dispatcher, a safety agent and a
   traffic-learning agent *negotiating by messages* (FIPA-ACL / Contract Net), with every
   decision explained.
2. **Look and feel like a real product** (a control-room / ops console), not a student demo.

The Python back end already exists, is tested (201 tests passing) and must **not change
behaviour**. You build a new React app in a new `frontend/` folder, served by the existing
FastAPI server. The old plain-JS dashboard stays alive at `/classic` as a fallback.

**Done = every item in §10 (acceptance checklist) is ticked, with evidence.**

---

## 1. Working agreement

* Work on a new git branch `phase-2-ui` created from `v2`. Small commits, imperative
  messages. **Do not push, do not touch `main`, do not rewrite history.**
* First produce a short written plan (files, components, store shape, test list) and then
  execute it. Re-plan if something in this prompt contradicts the code you find — **the
  code wins**; mention the discrepancy in your final report.
* Python rules: Python 3.12, managed by `uv`. `uv run pytest` and `uv run ruff check .` and
  `uv run ruff format --check .` must stay green. Line length 100.
* Do **not** change engine semantics (anything under `agents/`, `planning/`, `optimization/`,
  `rules/`, `traffic/`, `model.py` step logic). Back-end changes are allowed **only** in
  `src/elevator_mas/api/` and `src/elevator_mas/web/`, are **additive**, and each gets a test.
* The finished demo must work **offline** on an exam-hall laptop: no CDN, no Google Fonts,
  no network calls at runtime. Bundle fonts with `@fontsource/*`.
* The built app must be **committed** under `src/elevator_mas/web/dist/` so the demo runs
  with only Python installed (no Node needed at demo time). `frontend/node_modules` is
  git-ignored. Keep the committed bundle reasonably small (< 2.5 MB gzipped total);
  code-split the heavy pages (Experiments, Algorithm Lab, Agents graph).

---

## 2. Tech stack (fixed — don't substitute without a written reason)

| Concern | Choice |
| --- | --- |
| Build / dev server | **Vite** + **React 18/19** + **TypeScript (strict, `noUncheckedIndexedAccess`)** |
| Styling | **Tailwind CSS v4** + **shadcn/ui** (Radix primitives) + `lucide-react` icons |
| Animation | **Motion** (`motion/react`, the successor to framer-motion) |
| Charts | **Recharts** (consistent theme tokens; no chart on a raw `<canvas>` unless it's the building) |
| Agent graph | **@xyflow/react** (React Flow) |
| Client state | **Zustand** (live frames, UI prefs) |
| Server state | **TanStack Query** (REST: meta, benchmark, search-lab, run) |
| Routing | **TanStack Router** or **React Router v7** (hash-free, SPA, history mode) |
| Tests | **Vitest** + **@testing-library/react**; **Playwright** for end-to-end |
| Lint/format | **ESLint (typescript-eslint) + Prettier**, `tsc --noEmit` clean |
| Package manager | **pnpm** (fall back to npm if pnpm is unavailable; keep one lockfile) |

Directory: `frontend/` at the repo root (`frontend/src`, `frontend/e2e`, `frontend/public`).
Vite `build.outDir` = `../src/elevator_mas/web/dist`, `emptyOutDir: true`, `base: "/"`.
Dev: `pnpm dev` on :5173 with a Vite proxy: `/api` → `http://127.0.0.1:8000`, `/ws` →
`ws://127.0.0.1:8000` (with `ws: true`).

---

## 3. The back end you are talking to (read-only reference)

Run it with `uv sync && uv run elevator serve --scenario demo_story` →
`http://127.0.0.1:8000`. It starts **paused**. The app is `create_app()` in
`src/elevator_mas/api/server.py`; the session in `api/session.py`.

### 3.1 Live stream — `WS /ws`

On connect you receive one full snapshot, then one **full snapshot per simulated tick**
while the sim runs (or per `step`/`reset`/`inject`). Frames are JSON, **strict** (no
`Infinity`/`NaN`; non-finite numbers arrive as `null`). The client never sends on the socket.
Speed is a multiplier on 1 tick/second (UI range 0.1×–50×; at 50× expect ~50 frames/s, so
**batch store updates with `requestAnimationFrame`** and keep React renders cheap).

Snapshot shape (real example at tick 120 of `demo_story`; arrays shortened):

```jsonc
{
  "tick": 120, "scenario": "demo_story", "strategy": "full", "seed": 7,
  "building": { "floors": 15, "cars": 4, "capacity": 10, "lobby": 0 },
  "cars": [{
    "car_id": 0, "floor": 8, "direction": "UP",          // UP | DOWN | IDLE
    "door": "closed",                                    // closed|opening|open|closing
    "state": "moving_up",    // idle|moving_up|moving_down|doors|out_of_service|fire_recall
    "load": 2, "capacity": 10, "out_of_service": false, "fire_mode": false,
    "route":    [{ "floor": 8, "kind": "dropoff", "direction": "IDLE" }],   // kind: pickup|dropoff
    "car_calls": [8, 14],                                // buttons pressed inside the car
    "assigned": [{ "floor": 6, "direction": "DOWN" }],   // hall calls this car won
    "progress": 0.5,        // 0..1 of the way to the next floor -> interpolate smoothly
    "plan_cost": 33.5, "plan_nodes": 2                   // A* result for the current plan
  }],
  "floors": [{
    "floor": 0, "up": true, "down": false,               // hall lanterns/buttons lit
    "waiting_up": 13, "waiting_down": 0,
    "assigned_up": 3, "assigned_down": null,             // car id serving that call
    "eta_up": 11.0, "eta_down": null, "escalations": 0, "oldest_wait": 25
  }],
  "metrics": {                                           // live Metrics (see domain below)
    "tick": 120, "arrived": 35, "delivered": 9, "waiting": 13, "riding": 13,
    "avg_wait": 11.4, "p95_wait": 25.0, "max_wait": 25.0, "avg_ride": 22.6,
    "avg_system": 27.8, "long_wait_pct": 0.0, "throughput": 270.0, "energy": 136.0,
    "floors_travelled": 104, "stops": 16, "reversals": 0, "messages": 190,
    "messages_per_call": 11.9, "calls": 16, "nodes_expanded": 363, "replans": 52,
    "compute_ms_per_tick": 0.309, "rules_fired": 0
  },
  "auction": {                                           // the LAST Contract Net round, or null
    "tick": 113, "conversation_id": "c30", "floor": 6, "direction": "DOWN",
    "bids": [{ "car_id": 2, "total": 2.95, "wait": 2.8, "ride": 0.0, "crowding": 0.0,
               "energy": 0.15, "eta": 2.0, "refused": false, "reason": "" },
             { "car_id": 3, "total": null, "refused": true, "reason": "full" /* ... */ }],
    "winner": 2,
    "reason": "car 2 wins 6D at cost 2.9 (mostly wait 2.8, ETA 2s); next best car 1 at 10.0 (+7.1); car 3 refused (full)"
  },
  "messages": [{                                         // last 40, newest last
    "seq": 270, "tick": 204, "performative": "INFORM",
    "sender": "car-0", "receiver": "floor-5",            // or "broadcast"
    "conversation_id": "c47",
    "content": { "car_id": 0, "direction": "DOWN", "eta": 5.0 }   // HallCall -> {floor, direction}
  }],
  "rules": [{ "tick": 160, "rule": "R4_car_fault_out_of_service",
              "binding": { "car": 1 }, "effect": "car 1 out of service; 0 call(s) handed back for re-auction" }],
  "traffic": {                                           // the learning agent
    "pattern": "up_peak",                                // what it INFERRED
    "reason": "81% of trips start at the lobby", "observed": 42,
    "true_pattern": "two_way",                           // ground truth (show as 'hidden truth')
    "weights": { "wait": 1.4, "ride": 0.3, "crowding": 0.5, "energy": 0.1 }
  },
  "events": [{ "tick": 95, "kind": "rush", "floor": 0, "count": 12 }],
  "fire_alarm": false,
  "reassignment": null,                                  // last simulated-annealing result (object) or null
  "parking": { "2": 0 },                                 // car_id -> parking floor
  "session": { "running": false, "speed": 10.0, "duration": 330 }
}
```

Performatives (colour-code them consistently everywhere): `REQUEST, CFP, PROPOSE, REFUSE,
ACCEPT_PROPOSAL, REJECT_PROPOSAL, INFORM, CANCEL, FAILURE`.
Agent addresses: `dispatcher`, `monitor`, `safety`, `car-<n>`, `floor-<n>`.

**Future field (not present yet):** Phase 6 will add an optional `brain` object to the
snapshot. Type it as `brain?: BrainFrame | null` with a *placeholder* interface now (see
§6.6) so the Brain panel can light up later without a rewrite.

### 3.2 REST (all JSON)

| Method + path | Body / query | Returns |
| --- | --- | --- |
| `GET /api/state` | – | one snapshot |
| `GET /api/meta` | – | `{scenarios: string[], strategies: Strategy[], agents: AgentInfo[], environment: {...}[], system_peas: {...}, rules: {name, salience, description}[]}` |
| `POST /api/play` · `/api/pause` | – | `{running}` |
| `POST /api/step` | `{ticks: 1..600}` | state |
| `POST /api/speed` | `{speed: 0.1..50}` | state |
| `POST /api/reset` | `{scenario?, strategy?, seed?, floors? (2..200), cars? (1..32)}` | state |
| `POST /api/passenger` | `{origin, destination?, priority?}` | detail |
| `POST /api/inject` | `{kind, car?, floor?, count?}`, kind ∈ `car_fault, car_repair, fire_alarm, fire_clear, rush` | detail |
| `GET /api/agent/{address}` | – | agent inspector (`address, agent_type, peas, internal_state/…, planned_route, last_bid, energy, assignments…` — shape differs per agent type; render generically as a typed key/value tree, plus bespoke views for car and dispatcher) |
| `GET /api/messages?limit=` | ≤500 | `{messages, total}` |
| `GET /api/auctions?limit=` | – | `{auctions: Auction[]}` |
| `GET /api/search-lab?car=` | – | BFS/UCS/Greedy/A* on a live routing problem: `{source, car_id, current_floor, stops[], h_at_start, results: [{algorithm, found, cost, nodes_expanded, nodes_generated, max_frontier, runtime_ms, depth, route: string[], optimal}]}` |
| `GET /api/search-lab/annealing` | – | `{available, synthetic, calls, cars, simulated_annealing: {...}, hill_climbing: {...}}` (inspect the `as_dict()` of `LocalSearchResult` in `optimization/local_search.py` for the exact fields incl. the cost curve) |
| `GET /api/search-lab/minimax` | – | minimax vs alpha-beta: node counts, `best_parking`, `value`, `pruning_saving_pct`, candidate spots |
| `POST /api/benchmark` | `{scenarios?, strategies?, seeds (1..10), ticks (60..3600)}` | `{runs, scenarios, strategies, seeds, ticks, summary: row[], comparison: row[], wins: {...}}` — **CPU-bound, takes seconds–minutes; show progress/skeleton and never block the UI** |
| `GET /api/health` | – | `{status, tick, scenario, strategy}` |

Strategies (the "ladder", in order): `nearest_car` (reflex baseline) → `collective` (LOOK;
a genuinely strong baseline) → `cnp_astar` (Contract Net + A*) → `full` (adds simulated
annealing reassignment, smart parking, online weight learning). Labels/descriptions come
from `/api/meta`.

Scenarios (from `configs/scenarios/*.yaml`): `demo_story` (330 s scripted story: calm →
up-peak rush → breakdown → fire drill), `morning_up_peak`, `evening_down_peak`,
`lunch_two_way`, `interfloor_light`, `car_breakdown`, `fire_emergency`,
`priority_passenger`, `stress_scale` (40 floors, 8 cars). Don't hard-code the list; read it
from `/api/meta`.

### 3.3 Back-end additions you ARE asked to make (small, additive, tested)

1. **Serve the new app, keep the old.**
   * `GET /` → `web/dist/index.html` if it exists, else fall back to the classic page.
   * `GET /classic` → the existing `web/index.html` (keep `/static/*` mounted for it).
   * `/assets/*` (Vite's hashed bundle) mounted from `web/dist/assets`.
   * **SPA fallback**: any non-`/api`, non-`/ws`, non-`/static`, non-`/assets` GET path
     returns `dist/index.html` (so deep links like `/lab` work on refresh).
2. **`POST /api/run`** — a *fast headless* run for Compare Mode and charts.
   Body: `{scenario: string, strategy: string, seed: int, ticks?: int (60..3600), sample_every?: int (1..60, default 10)}`.
   Implementation: reuse `elevator_mas.sim.run_scenario` / build an `ElevatorModel` and step
   it in `asyncio.to_thread`; sample the `Metrics` every `sample_every` ticks.
   Returns `{scenario, strategy, seed, ticks, runtime_s, final: Metrics, series: {tick: number[], avg_wait: number[], p95_wait: number[], waiting: number[], riding: number[], delivered: number[], energy: number[], long_wait_pct: number[], messages: number[]}, rules_fired: string[], violations: string[]}`.
   Must be deterministic for a given (scenario, strategy, seed). Validate names → HTTP 400.
3. **`GET /api/board`** — the Phase-1 status board as JSON (`model.board.as_dict()`), used by
   the Agents page "Shared status board" panel.
4. (Optional but nice) `GET /api/version` → `{app: "2.0.0-dev", git: <short sha or null>}` for the footer.

Tests for every addition in `tests/test_api.py` (or a new `tests/test_web.py`): SPA fallback,
`/classic`, `/api/run` determinism + 400s, `/api/board`. Keep the strict-JSON guarantee
(`json.loads(text, parse_constant=raise)` on every new endpoint's output).

---

## 4. Product design

### 4.1 Look & feel

A **dark "control room"** theme by default with a **light** theme toggle (persist in
`localStorage`, guarded by try/catch). Calm, dense, professional — think Linear / Grafana /
Vercel dashboards, not a toy.

* One accent hue for "system / selected", plus a **fixed categorical palette** for the
  nine performatives and a **semantic palette** (ok / warn / danger / info). Define all
  colours as CSS variables (design tokens) used by Tailwind **and** Recharts, so charts
  and UI match in both themes. Check contrast ≥ 4.5:1 for text.
* Typography: Inter (UI) + JetBrains Mono (numbers, addresses, message payloads), both
  via `@fontsource`, `font-variant-numeric: tabular-nums` for every live number so values
  don't jitter.
* Motion with purpose: car movement is *interpolated* between floors using `progress`;
  doors animate; message "packets" travel along edges in the agent graph; a new auction
  flashes the winning bar. Everything respects `prefers-reduced-motion`.
* Responsive down to 1280×720 (the exam projector). Below that, degrade gracefully.
  Keyboard accessible (focus rings, labelled controls, `aria-live="polite"` for the
  decision trace).
* A persistent **top bar**: scenario select, strategy select, seed input, floors/cars
  steppers (disabled with a tooltip for scripted scenarios if you think that's wise),
  Play/Pause, Step (1 / 10 / 60), speed slider (log scale 0.1–50×), Reset, tick/time
  display `mm:ss`, connection status dot (WS), theme toggle, and a **Present** button.
* A persistent **left rail** with the five pages + Story mode.
* **Keyboard shortcuts** (show a `?` cheat-sheet dialog): `Space` play/pause, `→` step,
  `Shift+→` step 10, `R` reset, `F` inject fire alarm, `B` break a car, `1–5` pages,
  `P` present mode, `T` theme.

### 4.2 Pages

#### Page 1 — Mission Control (`/`)  ← the hero page; spend the most effort here

Layout (≥1280 px): building in the centre, intelligence panels around it.

* **Building view** (SVG or DOM, your call, but 60 fps while 40 floors × 8 cars):
  one shaft per car, floor labels, lobby marked. Each car is a rounded rectangle that
  glides to `floor + direction_sign * progress` (interpolate with the Motion spring or
  rAF lerp), shows: id, load bar (`load/capacity`, turns amber ≥80 %, red when full),
  door state (animated two-panel door), direction chevron, its route as small chips
  (pickup ▲/▼ vs drop-off ●), and a state ring colour (idle/moving/doors/fault/fire).
  Out-of-service cars are greyed with a wrench badge; fire-recall cars flash red.
  Floors: left gutter with hall lanterns ▲▼ lit from `floors[].up/down`, **a crowd of
  dots/pips for `waiting_up/waiting_down`** (cap at ~12 + "+N"), ETA badge from
  `eta_*`, assigned car id colour-tied to that car, an *aging* ring when `oldest_wait`
  grows (and a flame badge when `escalations > 0`). Clicking a floor adds a passenger
  (`POST /api/passenger`, shift-click = priority); clicking a car selects it (opens the
  inspector). For `stress_scale` (40 floors) auto-compress floor height and enable
  scroll/zoom of the shafts.
* **Live KPI strip**: avg wait, p95 wait, max wait, long-wait %, throughput /h, delivered
  / arrived, waiting, riding, energy, messages per call, compute ms/tick. Each tile has a
  sparkline from the last ~120 samples and a delta arrow vs 60 ticks ago. (Use the
  ring buffer in the store; don't refetch.)
* **Auction panel** ("Why did car 2 get this call?"): for the latest `auction`, draw one
  horizontal **stacked bar per car** split into the four utility terms
  (`wait, ride, crowding, energy` — fixed colours, legend shows the live weights from
  `traffic.weights`), refused cars shown as a hatched "REFUSED — full / out of service"
  row, the winner highlighted, and the **decision trace** sentence (`auction.reason`)
  rendered verbatim in a quote block with `aria-live`. A history strip of the last 20
  auctions (use `GET /api/auctions` once on mount and then append from the stream when
  `auction.conversation_id` changes) lets the user click back through earlier rounds.
* **Message stream** (compact): last ~40 messages, virtualised, colour-coded by
  performative, sender → receiver chips, conversation-id chips that highlight every
  message of the same conversation on hover, pause-on-hover, filter by performative /
  agent. Click → opens a detail drawer with the pretty-printed content.
* **Learning-agent card** ("Traffic monitor"): inferred pattern vs `true_pattern`
  (green tick when equal, amber when not — this is the *partial observability* story),
  the textual `reason`, the four live weights as small animated bars, and the parking
  targets (`parking`) drawn as ghost markers in the shafts.
* **Safety agent card**: the last 12 `rules` (R1…R7) as a timeline with their `effect`,
  a big **FIRE ALARM** banner when `fire_alarm`, and the invariant badge
  "no violations ✓".
* **Scenario controls / chaos panel**: buttons for *Break car N* (select car), *Repair*,
  *Fire alarm*, *Clear alarm*, *Rush at floor F (count)* → `POST /api/inject`. Event
  history list from `events`.
* A **footer progress bar** `tick / duration` that is scrubbable *only* by Reset/Step
  (no rewind in the engine) — show the scenario's phases as coloured segments when the
  scenario is `demo_story` (you may hard-code the 4 phase boundaries 60/150/240/330 with
  a comment pointing at the YAML).

#### Page 2 — Agents (`/agents`)

* **Agent graph** (React Flow): nodes for `dispatcher`, `monitor`, `safety`, every car, and
  a collapsed "floors" group node (expand to show floor agents when there are ≤ 20).
  Node cards show the AIMA *agent type* (from `/api/meta` → `agents[].agent_type`) and a
  tiny live status. **Edges animate with a travelling dot whenever a message of that
  pair appears in the stream**, coloured by performative; edge thickness decays over a few
  seconds so busy channels glow. Layout: dispatcher centre, cars ring/column, floors
  below, safety/monitor above. Provide a "Replay last conversation" button that
  re-animates one `conversation_id` slowly (CFP → PROPOSE×N → ACCEPT/REJECT → INFORM).
* **Sequence diagram** of the selected conversation (lifelines + arrows, drawn in SVG),
  e.g. `floor-6 → dispatcher REQUEST`, `dispatcher ⇒ cars CFP`, … built from the
  messages with that `conversation_id`.
* **Agent inspector** (right drawer, opened from any agent): header with address and
  agent type; the **PEAS table** (`peas.performance/environment/actuators/sensors`);
  AIMA agent-type badge; live internal state via `GET /api/agent/{address}` polled every
  500 ms *only while the drawer is open* (TanStack Query `refetchInterval`); car view
  shows planned route, last bid breakdown, replans, energy; dispatcher view shows open
  calls, `assignments`, last reassignment, last minimax, parking.
* **Shared status board** panel (from `GET /api/board`, polled while visible): a table of
  each car's *public* status — floor, direction, load/space, availability, assigned
  calls, plan end ETA — captioned *"what every agent is allowed to see; nobody reads
  another agent's private state."* This is the blackboard the learned model will read in
  later phases; highlight which fields were published this tick.
* **Message log** page section: full searchable table (virtualised) with the same filters,
  `GET /api/messages?limit=500` as the initial load then live appends.

#### Page 3 — Algorithm Lab (`/lab`)  — makes Review 1 "algorithmic modelling & search" obvious

Three tabs, each with a "Run on the live simulation" button and a "What am I looking at?"
explainer (2–3 sentences, plain English):

1. **Routing search (`/api/search-lab`)** — BFS vs UCS vs Greedy vs A* on the *same* routing
   problem. Table + grouped bar chart of `nodes_expanded`, `nodes_generated`,
   `max_frontier`, `runtime_ms`; optimal badge; route chips; show `h_at_start`. Add a
   **step-through visualiser** for A*: given `stops` and `current_floor` returned by the
   API, compute (client-side, in a Web Worker) the expansion order for a *small* instance
   (≤ 6 stops) and animate the open/closed list and f = g + h for each node, with a
   speed slider and Back/Next. Implement the **same** problem definition the server uses
   (state = (floor, set of served stops); action = go to a pending stop subject to the
   collective-control legality rules; step cost = travel + dwell weighted by remaining
   stops + energy; heuristic admissible — read `planning/routing.py` and replicate
   faithfully, then **assert in a Vitest test that your client-side A* optimal cost equals
   the server's `astar.cost`** for 20 fixed fixtures you generate by calling the real API
   in a one-off script and saving JSON in `frontend/src/lab/fixtures/`).
2. **Assignment optimisation (`/api/search-lab/annealing`)** — line chart of the SA `curve` and
   `best_curve` vs hill-climbing (payload fields: `initial_cost, final_cost, improvement,
   iterations, accepted, curve, best_curve, assignment`), final-cost comparison, and the
   `accepted` move count (SA accepts some worse moves early — that is how it escapes local
   minima; explain it in the panel). Show the `synthetic` flag honestly ("constructed instance — the algorithms
   and objective are the real ones").
3. **Adversarial parking (`/api/search-lab/minimax`)** — minimax vs alpha-beta node counts
   (bars) with `pruning_saving_pct`, the chosen `best_parking` drawn on a mini building,
   candidate spots and `likely_floors`.

#### Page 4 — Experiments (`/experiments`)

* **Benchmark runner**: choose scenarios (multi), strategies (multi, default all four),
  seeds 1–10, ticks 60–3600 (default 900) → `POST /api/benchmark`. Show a determinate-looking
  progress UI (estimated from runs × typical runtime; the call itself is one request),
  cancellation via `AbortController`. Result: sortable **summary table**, the **comparison
  table** (champion `full` vs baseline `nearest_car`, % improvement, colour-coded;
  **if a strategy loses, show it in red — never hide a loss**), small multiples of
  grouped bars for avg wait / p95 / long-wait % / energy per scenario, and the `wins`
  badge row. Export CSV and PNG (use `html-to-image` or SVG serialisation).
* **Compare Mode** (`/experiments/compare`): pick scenario + seed + two (or up to four)
  strategies; call `POST /api/run` for each in parallel; then render a **synchronised
  replay**: N building mini-views are *not* available from `/api/run`, so instead render
  a shared scrubber + play button that reveals the **time series** (avg wait, waiting,
  p95, energy) progressively for every strategy on the same axes, with a live "leader"
  badge and a final verdict card ("full beats nearest_car by 31% on avg wait; loses to
  collective on p95 — and here is why" is the honest template; the *why* is a static
  hint map keyed by (scenario, strategy) you may leave empty).
* **Saved report viewer**: static charts from `reports/benchmark_*.png` are *not* served
  by the API; skip them. (Do not add endpoints for this.)

#### Page 5 — Theory (`/theory`)  — the viva cheat-sheet, always one click away

Populate from `/api/meta` (`system_peas`, `environment`, `agents`, `rules`) and static
copy you write (keep it accurate; cross-check with `docs/DESIGN.md`):

* **PEAS** for the whole system and, in tabs, each agent (Performance / Environment /
  Actuators / Sensors) as a clean 4-column card.
* **Environment classification** table (partially observable, multi-agent
  cooperative+competitive, stochastic, sequential, dynamic, discrete, unknown…) with the
  one-line justification for each.
* **Agent-type ladder**: simple reflex → model-based reflex → goal-based → utility-based →
  learning, with which agent in this system is which, and a mini-diagram.
* **Why multi-agent?** 4–5 crisp points (distributed sensing/control, local autonomy,
  fault tolerance, scalability by adding cars, negotiation yields better global cost).
* **Algorithms** cards: A* (admissible + consistent heuristic, why optimal), Contract Net,
  simulated annealing, hill climbing, minimax + alpha-beta, forward-chaining rules R1–R7
  (list from `rules`), EWMA learning. Each card links to the matching Lab tab.
* **Message protocol** diagram: the Contract Net sequence (static SVG you draw) plus the
  nine performatives with a one-line meaning each.
* **Glossary** + **"Likely viva questions"** accordion (write 12 good Q&As).

#### Story / Presentation mode (`/story`, also the **Present** button)

For the live demo. A full-screen, minimal-chrome mode:

* A **guided script** over `demo_story` (330 s): a left caption card advances through ~10
  beats with a "Next" button / `→` key, each beat does a *real action* and explains it.
  Suggested beats (adapt to what the engine really does; verify by running it):
  1 "The building: 15 floors, 4 cars" → 2 "Calm traffic — cars park where demand is"
  (parking markers) → 3 "Rush at the lobby — watch the auction" (auto-focus the auction
  panel when a new auction appears and the dispatcher↔car messages) → 4 "Why car 2? the
  decision trace" → 5 "The learning agent re-tunes the weights" → 6 "Breakdown: car 1 dies,
  its calls are re-auctioned" (fires `car_fault` through the API) → 7 "Fire alarm: safety
  overrides everyone" → 8 "Normal service restored" → 9 "Results vs the baseline"
  (small `/api/run` comparison: `nearest_car` vs `full` on the same seed) → 10 "What's next:
  the learned bidder (LiftZero)".
* Big typography, hides the top bar, shows a slim progress dots indicator, `Esc` exits.
* Beats must be **robust**: they trigger actions via the API and wait for the *condition*
  (e.g. "an auction with ≥3 bids exists"), not for a fixed tick, with a visible fallback
  "Skip" if the condition doesn't occur within 20 s.

---

## 5. Front-end architecture

* `src/api/` — typed REST client (`fetch` wrapper that throws on non-2xx with the `detail`
  string; zod schemas for **runtime validation** of `/api/meta`, `/api/run`,
  `/api/benchmark`, `/api/search-lab*`; the snapshot is validated cheaply — shape-check
  the top-level keys only, to stay fast at 50 fps). All response types live in
  `src/api/types.ts` and mirror §3.
* `src/live/socket.ts` — a resilient WebSocket client: auto-reconnect with exponential
  backoff (cap 5 s), connection state in the store, drops frames when the tab is hidden,
  and an **rAF coalescer** so at most one store update per frame.
* `src/store/` — Zustand slices: `live` (latest snapshot, derived ring buffers of metrics
  ≤ 600 samples, last 200 messages, auction history ≤ 100), `ui` (theme, selected
  agent/car/conversation, drawer state, shortcuts, present mode), `story`. Use selectors
  and `useShallow` so a floor lamp change doesn't re-render the message list.
* `src/features/<page>/…`, `src/components/ui/…` (shadcn), `src/components/viz/…`
  (Building, CarShaft, KpiTile, Sparkline, BidBars, MessageChip, AgentGraph, SequenceDiagram).
* Pure helpers (`format.ts`, `colors.ts`, `interpolate.ts`, `sequence.ts`) are unit-tested.
* **Performance budget**: at 8 cars × 40 floors × 50 fps the main thread stays under
  ~8 ms/frame on a mid laptop; verify with the React Profiler / Chrome performance panel
  and note the numbers in `docs/UI.md`. Memoise per-car and per-floor components; animate
  with transforms (not layout properties).
* Error/empty/loading states for **every** async view; a global error boundary with
  "Reconnect" and "Open classic dashboard" buttons.
* Accessibility: semantic landmarks, tab order, `aria-label`s on icon buttons, no
  colour-only encoding (add icons/patterns for refused bids, faults, fire).

---

## 6. Details that are easy to get wrong

1. **Strict JSON.** The server maps non-finite numbers to `null`. A refused bid has
   `total: null`; don't `Math.max(...)` over nulls; filter `!refused` first.
2. **`progress` semantics.** It is the fraction of the way to the *next floor in the car's
   direction of travel*. Interpolated y = `floor + sign(direction) * progress` (IDLE →
   `floor`). Don't extrapolate between frames at high speed; at ≤ 2× speed ease between
   frames, above that snap (frames are too frequent for easing to matter).
3. **Floors index 0 = lobby = bottom.** Draw top-down with the highest floor at the top.
4. **Conversation ids** like `c47` thread one protocol round; `INFORM car→floor` carries the
   ETA; reassignments appear as `CANCEL` + `ACCEPT_PROPOSAL` with
   `content.reason == "global reassignment"`.
5. **Hall-call objects** in message `content` serialise to `{floor, direction}`; enums are
   upper-case names (`"UP"`). Orders from the safety agent are `REQUEST` messages whose
   `content.order` is one of `FIRE_RECALL, HOLD_DOORS_OPEN, BLOCK_HALL_CALLS, RESTORE_SERVICE,
   OUT_OF_SERVICE, RETURN_TO_SERVICE, REFUSE_BOARDING, REOPEN_DOORS` — render these as
   "ORDER · FIRE_RECALL" chips.
6. **Brain placeholder** (for Phase 6): define
   ```ts
   export interface BrainFrame {            // NOT sent by the server yet
     model: string; params: number; latency_ms: number;
     decision?: { call: {floor:number; direction:'UP'|'DOWN'};
                  scores: { car_id:number; score:number; teacher?:number }[];
                  chosen:number; teacher_choice?:number; mcts?: { visits:number[]; value:number } };
   }
   ```
   and a **Brain panel** on Mission Control that, when `snapshot.brain` is absent, shows a
   tasteful empty state: *"LiftZero model not loaded — running the classical Contract-Net
   bidder."* with a small diagram. Phase 6 will only have to populate the field.
7. **Don't poll what the stream already carries.** Poll only the inspector, board and
   search-lab endpoints, and only while their UI is visible.
8. **Speed 50× on `stress_scale`** is heavy for the *server*; if frames back up the server
   drops them (bounded queue) — the UI must tolerate gaps in `tick`.

---

## 7. Testing (all of it is graded — "demo quality & testing")

**Python** (`uv run pytest`): all 201 existing tests still pass, plus the new API tests from
§3.3.

**Vitest** (`pnpm test`): ≥ 60 meaningful tests, ≥ 80 % statement coverage on `src/api`,
`src/store`, `src/lib` and `src/lab`. Must include: store reducers/ring buffers; rAF
coalescer; message filtering; colour/format helpers; `interpolate` (floor + progress);
sequence-diagram builder; story beat condition waiters (fake timers); the A* parity test
(§4 Lab tab 1) over 20 fixtures; component tests for `KpiTile`, `BidBars` (refused rows,
null totals), `MessageChip`, `Building` (renders N shafts, car position, lamps).

**Playwright** (`pnpm e2e`, starts `uv run elevator serve --scenario demo_story` on a
free port via `webServer` config; Chromium only): at least these specs —
1. app loads, WS shows *connected*, building shows 4 shafts and 15 floors;
2. Play → tick counter increases; Pause → stops; Step +10 advances exactly 10;
3. Speed slider changes `session.speed` (check via `/api/health` or the UI);
4. Reset with a different scenario/strategy changes the header chips;
5. Inject *car fault* → a Safety timeline entry `R4…` appears and car shows the wrench
   badge; *Repair* clears it;
6. Inject *fire alarm* → banner appears; *Clear* → R7 entry;
7. Click a floor → waiting pip count increases;
8. Auction panel shows a decision trace after a rush; clicking history changes the panel;
9. Agents page: graph renders all agents; inspector opens for `car-0` with PEAS table;
10. Algorithm Lab: all three tabs return data and render without console errors;
11. Experiments: run a tiny benchmark (1 scenario, 2 strategies, 1 seed, 120 ticks) and see
    both tables; Compare Mode renders ≥ 2 series;
12. Story mode: advance through every beat using `→` (fallback Skip allowed) and reach the
    end; `Esc` exits;
13. Classic dashboard still loads at `/classic`; `/lab` deep link works after a hard reload;
14. **No console errors or failed network requests** during specs 1–13 (assert via
    `page.on('console'|'requestfailed')`).

Also add an **axe-core** accessibility smoke test (`@axe-core/playwright`) on Mission Control
and Theory: no *serious/critical* violations.

Take **Playwright screenshots** (1440×900, dark and light) of every page and Story mode into
`docs/img/ui/` and reference them from `docs/UI.md`.

---

## 8. Documentation to deliver

* `docs/UI.md` — architecture diagram, how to run (dev / prod), page-by-page tour with the
  screenshots, the data-flow (WS → store → views), performance numbers, accessibility
  notes, and a "how to extend" section (how Phase 6 populates the Brain panel).
* `frontend/README.md` — scripts table (`dev`, `build`, `test`, `e2e`, `lint`, `typecheck`).
* Update the root `README.md` quick-start: `uv run elevator serve` now opens the React app;
  `/classic` is the old one; developer flow is `cd frontend && pnpm i && pnpm dev`.
* Update `docs/DEMO_SCRIPT.md` to match Story mode's beats.
* Update `docs/TESTING.md` with the new test layers and counts.
* Add a `Makefile` or `justfile` target `ui-build` (`pnpm -C frontend build`) and make sure
  `uv run elevator serve` has **no** dependency on Node.

---

## 9. Suggested order of work (so you always have something demoable)

1. Scaffold Vite/React/TS/Tailwind/shadcn, theme tokens, shell (top bar, rail, routes),
   proxy, back-end serving + `/classic` + SPA fallback + tests.
2. Typed API client + WebSocket + store + connection indicator. Prove live ticking.
3. Building view with interpolation, KPI strip, controls. **Commit a working Mission Control.**
4. Auction panel, message stream, learning/safety cards, chaos panel.
5. `/api/run`, `/api/board` + tests. Agents page (graph, sequence, inspector, board).
6. Algorithm Lab (3 tabs + A* visualiser + parity test).
7. Experiments + Compare Mode.
8. Theory page, Story mode.
9. Polish (empty/error states, shortcuts, a11y, reduced motion, perf pass), Playwright suite,
   screenshots, docs, production build committed to `src/elevator_mas/web/dist`.

---

## 10. Acceptance checklist (tick each with evidence in your final report)

- [ ] `uv run pytest` → all green (201 existing + new), `ruff check` + `ruff format --check` clean.
- [ ] `pnpm typecheck`, `pnpm lint`, `pnpm test` (≥ 60 tests, coverage thresholds met), `pnpm e2e` (all specs + axe) green.
- [ ] `uv run elevator serve` (no Node installed) serves the React app at `/`; `/classic` serves the old one; deep links survive refresh.
- [ ] Mission Control shows: animated cars, door states, load bars, route chips, hall lanterns, waiting crowds, KPI sparklines, auction stacked bars + decision trace, message stream, learning-agent card (inferred vs true pattern), safety timeline, chaos controls, Brain placeholder.
- [ ] Agents page: live message-pulse graph, conversation sequence diagram, inspector with PEAS, status-board table.
- [ ] Algorithm Lab: 3 tabs with real data; client-side A* parity test passes on 20 fixtures.
- [ ] Experiments: benchmark table + chart + CSV/PNG export; Compare Mode with synchronous series replay; losses shown in red.
- [ ] Theory page complete; Story mode runs end-to-end and is robust to timing.
- [ ] Works fully offline (verify: block network in DevTools, reload, run a scenario).
- [ ] 50× speed on `stress_scale` stays smooth (note frame-time numbers in `docs/UI.md`).
- [ ] `docs/UI.md`, `frontend/README.md`, root README, DEMO_SCRIPT, TESTING updated; screenshots committed.
- [ ] Work is on branch `phase-2-ui`, committed in logical steps, not pushed.

## 11. Final report format (what to send back)

A concise report: what was built (per page), test counts and coverage, bundle size (gzip),
frame-time measurements, any deviation from this prompt with the reason, known gaps, and
the exact commands for me to verify (`git checkout phase-2-ui`, `uv sync`, `uv run pytest`,
`uv run elevator serve`, `cd frontend && pnpm i && pnpm test && pnpm e2e`).
