# Testing and results

```bash
uv run pytest                 # 432 tests (incl. slow), ~2.5 min; -m 'not slow' for a quick run
uv run pytest --cov           # coverage of the core logic
uv run elevator verify        # run every scenario, assert the invariants
uv run elevator bench         # regenerate reports/
```

**432 Python tests pass** (Phase 1–2: 253 → Phase 4: 432) **and 66 Vitest tests.** Coverage of the core package is 96 %; of `learning.lift` 90 % (targets 85 %).

Fire-recall regression: `tests/test_fire_recall_regression.py` (6) pins a Phase 1 safety bug found by the Phase 4 test matrix (see `docs/lift/PHASE4_IMITATION.md` §9).

---

## 1. What the tests establish

### 1.1 Search — `tests/test_search.py` (18 tests)

The rubric's central claims, asserted over **200 seeded random instances each** rather
than hand-picked examples.

| Property | How it is checked |
| --- | --- |
| A* is optimal | `A* cost == UCS cost` on every solvable instance |
| A* is efficient | `A* nodes expanded ≤ UCS nodes expanded`, always |
| Heuristic is **admissible** | `h(n) ≤` true remaining optimal cost at every node of an optimal path |
| Heuristic is **consistent** | `h(n) ≤ c(n,a,n') + h(n')` for every legal transition |
| `h` is zero at the goal | otherwise admissibility is trivially broken |
| Greedy is not optimal but never cheats | `greedy cost ≥ optimal cost` |
| BFS is complete | finds a solution exactly when A* does |
| Plans obey collective control | every action is in `legal_stops(state)` |
| A DOWN sweep descends | `[14,12,11,8]`, never `[8,11,12,14]` |
| An UP sweep ascends | mirror case |
| Search is deterministic | identical results on repeated runs |
| LOOK fallback engages | no search runs above 10 pending stops |
| Negative step costs are rejected | would silently break optimality |

### 1.2 Optimization — `tests/test_optimization.py` (14 tests)

| Property | How it is checked |
| --- | --- |
| **SA is never worse than its input** | `final_cost ≤ initial_cost` over 40 random instances |
| SA returns what it claims | `objective(result.assignment) == result.final_cost` |
| SA improves a bad start | piling every call on one car is measurably improved |
| SA's running best is monotone | the green Search Lab curve can only fall |
| SA is reproducible | same seed ⇒ same assignment |
| Hill climbing never worsens | it only takes improving moves |
| **Alpha-beta == minimax** | identical value across car counts |
| Alpha-beta prunes | strictly fewer nodes on a branching game |
| The minimax value is a real guarantee | no floor nature may pick is further than the reported value |
| Move generation is deterministic | otherwise node counts are not comparable |

### 1.3 Safety rules — `tests/test_safety_rules.py` (15 tests)

Engine mechanics (firing, chaining to a fixed point, salience ordering, retraction,
termination of a self-satisfying rule) and the behaviour they produce:

- **Fire:** every in-service car reaches the lobby; hall calls are blocked; assignments are
  cleared; doors are held open; clearing the alarm restores service (R1, R2, R3, R7).
- **Fault:** the car goes out of service, empties, releases its calls, and **refuses** new
  ones rather than bidding a large number (R4). Stranded riders **keep their original
  arrival time** — a breakdown cannot flatter the metrics. Repair fully restores the car.
- **Overload:** boarding is refused and `space == 0` (R5).

### 1.4 Protocol — `tests/test_protocol.py` (12 tests)

| Property | How it is checked |
| --- | --- |
| **Exactly one ACCEPT per call** | counted over a whole run; more than one means double service |
| Every CFP is answered by every car | `replies == len(cars)`; silence would stall the protocol |
| The lowest bid wins | the award matches `min(total)` in every completed auction |
| Out-of-service cars REFUSE | with reason `"out of service"` |
| A failure re-auctions its calls | no call stays assigned to a dead car |
| Fire mode stops new awards | no `ACCEPT_PROPOSAL` after the alarm |
| Every message is threaded | all carry a `conversation_id` |
| Aging escalates | a single overloaded car produces urgent re-requests |

> This suite found a real bug: each car was replying to the broadcast CFP *and* being
> polled directly by the dispatcher, so every proposal was logged twice. Fixing it cut
> messages per call from 16.4 to **12.3**.

### 1.4b Message-driven interaction — `tests/test_messaging.py` (18 tests)

| Property | Check |
| --- | --- |
| No direct calls | a spy proves the dispatcher and safety agent are never on the stack of a car's bid/accept/drop/out-of-service routine |
| Reassignment by message | every global reassignment is a `CANCEL` plus an `ACCEPT_PROPOSAL` |
| One INFORM per award | the winning car tells the floor exactly once |
| One reply per car per CFP | every car answers, once, to the auctioneer |
| Decision trace | every award carries a non-empty explanation |
| Stage order | `sense, communicate, negotiate, decide, act, learn`; no round left open across ticks |
| Safety by orders | faults and fire alarms appear as `REQUEST` orders and show on the status board |
| Status board | immutable snapshots, availability filter, policy published by the monitor, never stale |

### 1.5 Whole-run invariants — `tests/test_simulation.py` (23 tests)

Checked **at every tick**, for **all four strategies**:

- capacity is never exceeded;
- doors are never open while a car is moving;
- cars stay inside the building;
- an out-of-service car carries nobody;
- no passenger is ever in two places at once.

Plus **no starvation** (after a drain, every single passenger has been delivered — under
every strategy) and **determinism** (same seed ⇒ identical metrics, identical car
trajectories and an identical message log; different seeds ⇒ different runs).

### 1.6 Scenarios, API, benchmark

- `tests/test_scenarios.py` (35): every scenario runs clean, meets the `expected:` block in
  its own YAML, and delivers everyone after a drain; plus the specific behaviour each
  scenario exists to demonstrate.
- `tests/test_api.py` (37): every REST endpoint, validation failures, the agent inspector,
  all three Search Lab panels, and WebSocket streaming.
- `tests/test_benchmark.py` (11): the matrix, the aggregates, and that the PNG charts are
  genuine, non-empty images.
- `tests/test_config.py` (17): schema validation, the strategy registry, domain invariants
  and the metric helpers.

### 1.7 Learning environment — `tests/learning/` (54 tests, Phase 3 + Phase 4 fixes)

Feature schema v2 (incl. the four public cost weights), encoder truth tables and bounds
(Hypothesis), recorder shards (teacher/source/executed arrays, finite sentinels, labels =
argmin of teacher bids, split codes by run seed, shadow-bid labelling for DAgger), the
hook-neutrality check, twin invariants, and the real-backend evaluation path.

### 1.8 LiftZero learned bidder — `tests/lift/` (133 tests, Phase 4)

| file | tests | what it proves |
| --- | --- | --- |
| `test_model.py` | 22 | parameter budget 0.2–0.4 M; permutation **equivariance** and **padding invariance** to 1e-5; finite for N = 1…32; ineligible offset; determinism; gradient reaches every parameter (except the Phase 5 value head); tie-break to lowest id |
| `test_losses.py` | 16 | each loss term against a hand computation; padded/ineligible cars contribute exactly 0 to every term; tie-aware soft targets; hard decisions weigh ×2; trivial decisions carry no ranking loss |
| `test_augment.py` | 11 | permutation keeps labels; dropout never drops a (tied) winner and keeps ≥ 2 cars; noise only on continuous features of real cars |
| `test_metrics.py` | 6 | agreement (strict/tie-aware), regret, Kendall τ-b on cases with known answers |
| `test_train.py` | 8 | the pipeline **overfits** 256 decisions to ≥ 99 %; same seed ⇒ identical first 20 losses; warm-up/cosine schedule; run artefacts; schema-mismatch checkpoints refused |
| `test_onnx_runtime.py` | 14 | torch ↔ ONNX **parity < 1e-4** at N = 1, 2, 8, 16, 32 and on 1 000 real decisions; card/schema/hash mismatch refuses to load; trimming padding is exact; the runtime path imports neither torch, onnx nor gymnasium |
| `test_bidder.py` | 10 | the bidder reads **only the public board** (stub whose other attributes raise); `shared` ≡ `per_car`; live CFP-time features ≡ recorded announce-time features; shadow fields only when enabled and never change the run; ineligible cars still REFUSE; β-mixing ≈ β ± 0.05 over 2 000 auctions; DAgger RNG never touches the arrival stream |
| `test_strategy_api.py` | 6 | registry entries differ from their teacher only in the bidder; `/api/meta` exposes `bidder`/`available`; a missing model ⇒ unavailable (never silently classical); frames stay strict JSON |
| `test_safety_matrix.py` | 28 | `liftzero_bc` on **all 9 scenarios × 3 seeds**: zero invariant violations every tick, everyone delivered after a drain, no ineligible car ever wins (faults, fire, rush included) |
| `test_dagger_eval.py` | 12 | DAgger seeds never leak into val/test, rounds use disjoint fresh seeds, shards carry `source` tags and teacher labels; closed-loop runner, paired bootstrap comparison, offline report, CLI |

`tests/test_golden_metrics.py` (25) pins `Metrics.as_dict()` of the four classical strategies
× 3 scenarios × 2 seeds, generated **before** any Phase 4 engine change: the learned bidders
are provably additive. Results of the learned bidder are in
[`docs/lift/PHASE4_IMITATION.md`](lift/PHASE4_IMITATION.md).

---

## 2. Scenario results

Every scenario, run to completion and then drained (arrivals off, run on until everyone is
delivered). **Zero invariant violations anywhere.**

| Scenario | Runtime | Avg wait | P95 | Max | Waits > 60 s | Delivered | Rules fired |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `morning_up_peak` | 0.22 s | 16.0 s | 52.9 s | 59 s | 0.0 % | **135/135** | — |
| `evening_down_peak` | 0.49 s | 23.3 s | 62.2 s | 109 s | 5.7 % | **158/158** | — |
| `lunch_two_way` | 0.30 s | 13.4 s | 45.0 s | 69 s | 3.1 % | **127/127** | — |
| `interfloor_light` | 0.13 s | 9.7 s | 24.4 s | 37 s | 0.0 % | **57/57** | — |
| `car_breakdown` | 0.22 s | 11.1 s | 38.5 s | 62 s | 0.7 % | **136/136** | R4 |
| `fire_emergency` | 0.44 s | 71.2 s | 262.5 s | 331 s | 31.9 % | **144/144** | R1, R2, R3, R7 |
| `priority_passenger` | 0.29 s | 12.5 s | 35.9 s | 77 s | 0.7 % | **142/142** | — |
| `stress_scale` | 2.17 s | 17.1 s | 66.0 s | 194 s | 7.0 % | **359/359** | — |
| `demo_story` | 0.10 s | 72.4 s | 164.5 s | 174 s | 54.4 % | **68/68** | R1, R2, R3, R4, R7 |

Notes on the two apparent outliers, both expected:

- **`fire_emergency`** has high waits *by construction*: for 270 simulated seconds every car
  is recalled to the lobby and hall calls are blocked, so everybody waiting accrues time.
  A low average here would mean the evacuation was not being taken seriously.
- **`demo_story`** compresses a surge, a breakdown **and** a fire into 330 seconds. It is
  built for demonstration density, not for good service.

**`stress_scale`** is the scalability result: 40 floors, 8 cars, a **full simulated hour**
in **2.17 s** — roughly 1 600× real time, at 4.6 ms per tick — against a 120 s budget.
It differs from the default only by its YAML file.

---

## 3. Strategy benchmark

4 strategies × 4 scenarios × 5 seeds × 900 ticks = 80 runs. Identical seeds for every
strategy, so differences are the coordination mechanism, not the traffic.

| Scenario | Strategy | Avg wait (mean ± sd) | P95 | Waits > 60 s |
| --- | --- | ---: | ---: | ---: |
| `morning_up_peak` | nearest_car | 54.06 ± 54.07 | 121.8 | 27.9 % |
| | collective | 23.92 ± 14.13 | 65.1 | 11.7 % |
| | cnp_astar | 48.71 ± 28.54 | 104.9 | 34.3 % |
| | **full** | **15.89 ± 4.39** | **49.1** | **2.7 %** |
| `evening_down_peak` | nearest_car | 21.40 ± 2.16 | 68.4 | 8.7 % |
| | **collective** | **18.03 ± 1.83** | **58.7** | **5.6 %** |
| | cnp_astar | 21.99 ± 1.44 | 61.0 | 5.8 % |
| | full | 22.85 ± 2.44 | 64.6 | 7.7 % |
| `lunch_two_way` | nearest_car | 17.51 ± 3.25 | 60.6 | 6.8 % |
| | collective | 16.70 ± 2.50 | 52.2 | 3.8 % |
| | cnp_astar | 14.24 ± 1.92 | 45.6 | 2.5 % |
| | **full** | **13.25 ± 1.31** | **36.8** | **1.7 %** |
| `interfloor_light` | **nearest_car** | **8.52 ± 1.16** | **19.7** | **0.0 %** |
| | collective | 8.69 ± 1.47 | 21.4 | 0.3 % |
| | cnp_astar | 8.96 ± 0.69 | 21.7 | 0.0 % |
| | full | 8.67 ± 1.56 | 22.7 | 0.3 % |

### Full vs the NearestCar baseline (positive = better)

| Scenario | Avg wait | P95 wait | Waits > 60 s | Verdict |
| --- | ---: | ---: | ---: | --- |
| `morning_up_peak` | **+70.6 %** | **+59.7 %** | **+90.3 %** | beats baseline |
| `lunch_two_way` | **+24.4 %** | **+39.3 %** | **+75.3 %** | beats baseline |
| `evening_down_peak` | −6.8 % | +5.6 % | +11.9 % | mixed |
| `interfloor_light` | −1.8 % | −15.0 % | 0.0 % | parity |

### Reading the results honestly

**Where it wins, it wins decisively.** In the peak regimes — the ones that matter, because
they are when a lift system is actually under strain — the full agent system cuts average
wait by 70.6 % and long waits by 90.3 % against the reflex baseline. That is the result the
specification asks for, and it holds.

**It is also far more *reliable*.** Look at the standard deviations in `morning_up_peak`:
the baseline is 54.06 ± **54.07** — its performance is a coin toss, ranging from tolerable
to dreadful depending on the seed. The full system is 15.89 ± **4.39**. Predictability is
worth as much to a passenger as the mean.

**Three results that do *not* flatter the design, reported as found:**

1. **`interfloor_light` is a tie, and could not be anything else.** At 0.06 arrivals per
   second with four cars, a car is almost always free the moment a button is pressed.
   There is no allocation problem to solve, so no allocation policy can help. All four
   strategies sit within 0.5 s of each other. Coordination has value only under contention.

2. **`evening_down_peak` is the honest loss: `collective` beats `full` by 4.8 s.**
   Down-peak traffic fills cars at the top of the building and empties them at the lobby,
   which is precisely the pattern a plain LOOK sweep is optimal for. The extra machinery
   then has nothing to exploit and the SA reassignment's cheap insertion *estimate* cannot
   perfectly predict what each car's own A* will do, so occasionally it moves a call that
   should have stayed. We tuned the hysteresis (80) and the reassignment interval (30 s) to
   shrink this, and disabled idle-car parking in down-peak and two-way traffic once
   measurement showed it was harmful — but a gap remains, and we are reporting it rather
   than quietly dropping the scenario.

3. **`cnp_astar` is worse than `collective` in up-peak (48.7 vs 23.9).** This looks wrong
   until you see what parking does: auctions alone leave cars wherever their last trip
   ended, scattered up the building, while everyone arrives in the lobby. Adding smart
   parking is what turns 48.7 into 15.89. It is a good illustration that a more
   sophisticated *allocation* policy is worthless if the fleet is in the wrong *place* —
   and that the techniques interact rather than simply stacking.

Energy is roughly flat (within ±7 % in three of four scenarios; `interfloor_light` costs
21.7 % more energy because parking repositions otherwise-idle cars). The system trades a
little electricity for a large reduction in waiting, which is the correct trade for a
passenger-facing service — and the weights are configurable if a building disagrees.

---

## 4. Tuning decisions, and the evidence for them

Every constant below was chosen by measurement, not taste.

| Parameter | Value | Evidence |
| --- | --- | --- |
| Reassignment hysteresis | 80 | Swept 4 / 25 / 80 / 200. At 4 the fleet churned assignments for trivial gain (50.2 s); 80 gave the best result (44.5 s) and stable hall lanterns. |
| Reassignment interval | 30 s | 15 s reassigned faster than a car can complete a leg (47.8 s); 30 s measured better (44.5 s). |
| `light` threshold | 0.07 / 0.10 per s | A single threshold made the classifier flip-flop 6 times in 150 ticks, retuning the fleet's weights each time. A hysteresis band fixed it; `tests/test_scenarios.py::test_classification_does_not_flip_flop` now guards it. |
| Parking in down-peak / two-way | **disabled** | Measured harmful: 78.9 s with parking vs 71.9 s without. Idle cars are rare in heavy spread traffic, and every repositioning move is a journey that must be undone. |
| LOOK fallback threshold | 10 stops | Keeps `stress_scale` inside its budget; above this the branching factor makes per-tick search impractical. |
| `stress_scale` arrival rates | 0.07–0.12 per s | At 0.30 per s the 40-floor building saturated completely (974 s waits, 183/851 delivered) — every strategy looks identical in saturation, so the scenario measured nothing. |

---

## 5. Bugs the tests and measurements caught

Recorded because the process is itself part of the case study.

| Bug | Symptom | Fix |
| --- | --- | --- |
| **Wrong-direction pickups** | A* planned `8↓, 11↓, 12↓, 14↓` from floor 7 — boarding down-travelling passengers while driving *up*. Down-peak was worse than the reflex baseline. | Added the second collective-control constraint: a DOWN call may only be taken when no pickup lies further up. |
| **Door-cycle deadlock** | A zero-length dwell left the timer at 0, and the car sat in `OPEN` forever. A drained building never emptied: 7 passengers stranded, 3 074 s max wait. | The door timer now always advances at least one tick. |
| **Stale assigned calls** | Cars kept pickups nobody was waiting for, stopping for no one and inflating their own bids. | `prune_stale_calls()` every tick. Stops fell from 2 388 to 147; average wait 158.9 s → 39.4 s. |
| **Silent floors** | A landing whose call was already "pressed" never re-requested when its assigned car filled and left, so a queue grew with nobody coming. | The floor re-requests when a live call has no assigned car. |
| **Refusing passengers at drop-off stops** | A car stopping only to let riders out reported *no* serving direction and refused everyone waiting. Worst in down-peak, where nearly every stop is a drop-off. | Infer the onward direction from the remaining plan when the direction flag is momentarily clear. |
| **Duplicate CFP replies** | Every car answered the broadcast CFP *and* was polled by the dispatcher, double-logging every proposal. | Superseded in v2: the dispatcher no longer computes bids at all; each car answers its CFP once from its own inbox, which `test_messaging.py` checks. |
| **SA optimising the wrong thing** | The reassignment objective ignored work a car was already committed to, so SA "improved" its own estimate while worsening real waits (33.8 s → 47.9 s). | The objective now starts from the car's existing plan and orders calls the way the car will actually serve them. |
| **Classifier flip-flop** | The traffic pattern oscillated around a single threshold, retuning fleet policy every few ticks. | Hysteresis band (0.07 / 0.10). |

---

## 6. Frontend testing and E2E validation

The Phase 2 LiftZero Control Room web client is verified across three layers: fast Vitest unit/component tests, exact TypeScript–Python algorithm numerical parity, and full headless browser integration with Playwright.

```bash
cd frontend
pnpm test               # 66 Vitest unit and component tests
pnpm test:coverage      # Statement coverage: 88.91 % (target > 80 %)
pnpm e2e                # 16 Playwright browser specs across 6 views
```

### 6.1 Unit and component tests — `frontend/src/` (66 tests)

| Test Suite | Tests | What is checked |
| --- | ---: | --- |
| `src/api/client.test.ts` | 14 | REST endpoints (`/api/version`, `/api/scenarios`, `/api/run`, `/api/board`), WebSocket connection lifecycles, reconnect backoff, message demultiplexing, and error handling |
| `src/store/simulationStore.test.ts` | 12 | Zustand reactive state, tick advancement, WebSocket telemetry ingestion, car status updates, replay buffer, and filter toggles |
| `src/lib/colors.test.ts` | 8 | Theme token mapping, car color assignments, contrast ratios, and HSL badge utilities |
| `src/lab/routing.test.ts` | 20 | Complete A* graph search, heuristic admissibility, consistency, collective-control legality, and fallback mechanisms |
| `src/lab/optimization.test.ts` | 12 | Simulated annealing temperature cooling schedule, cost improvement, minimax tree evaluation, and alpha-beta pruning |

**Overall frontend unit coverage: 88.91 % statements, 80.70 % branches, 91.80 % functions.**

### 6.2 TypeScript A* numerical parity against Python

To ensure the client-side interactive Search Lab faithfully visualizes the real multi-agent planning engine, `routing.test.ts` runs against **20 deterministic Python-generated problem instances** serialized from `elevator_mas.planning.search`:

| Fixture Parameter | Values Tested | Parity Property Checked | Result |
| --- | --- | --- | --- |
| Building heights | 5, 10, 16, 24 floors | Exact cost equality `abs(ts_cost - py_cost) < 1e-4` | **20/20 PASS** |
| Direction sweeps | UP, DOWN, IDLE | Identical stop sequences and visit order | **20/20 PASS** |
| Dwell & transit physics | `transit=1.0s`, `dwell=4.0s` | State transition timings identical to tick | **20/20 PASS** |
| Heuristic values | Admissible Manhattan + stops | Zero discrepancy in nodes expanded | **20/20 PASS** |

### 6.3 End-to-end integration — `frontend/e2e/app.spec.ts` (16 tests)

Playwright runs against the integrated FastAPI server serving the compiled SPA static bundle (`src/elevator_mas/web/dist/`):

| Test Case | Route | What is verified |
| --- | --- | --- |
| `mission control loads and displays header` | `/` | Header, status badges, telemetry cards, and building canvas render |
| `building canvas renders shafts and cars` | `/` | Shaft columns, floor markers, and 4 animated car cabins render |
| `live telemetry stream updates kpi strip` | `/` | WebSocket message streaming updates wait time and delivered count |
| `playback controls trigger simulation commands` | `/` | Play, Pause, Step, and Speed multiplier buttons communicate with backend |
| `scenario switcher switches active scenario` | `/` | Selecting `evening_down_peak` updates active scenario and re-initializes engine |
| `chaos injection triggers faults and alarms` | `/` | Injecting Car Breakdown updates car status to `OUT_OF_SERVICE`; Fire triggers lobby recall |
| `agent inspector drawer opens on car click` | `/` | Clicking Car 0 in building canvas opens inspector drawer with PEAS and state |
| `agents page renders topology and sequence diagram` | `/agents` | Flow topology canvas and live FIPA-ACL sequence swimlanes render |
| `status board displays blackboard records` | `/agents` | System-wide blackboard table shows fleet policies, active reservations, and status |
| `search lab runs A* interactive simulation` | `/lab` | Custom floor stop inputs generate step-by-step search tree and frontier list |
| `search lab simulated annealing visualizes temperature` | `/lab` | SA optimization runs and produces cooling curve and cost descent |
| `experiments page displays benchmark matrix` | `/experiments` | 4x4 scenario-strategy matrix displays with radar charts and metrics |
| `experiments page run button triggers live execution` | `/experiments` | Triggering a headless benchmark run displays progress indicator and updates table |
| `theory page renders educational primers and math` | `/theory` | All 5 pedagogical modules (A*, CNP, SA, Minimax, Rules) render with math formulas |
| `story mode runs scripted demo` | `/story` | Step-by-step interactive narrative loads chapters and highlights timeline events |
| `legacy dashboard fallback is accessible` | `/classic` | Navigating to `/classic` renders the original SVG template dashboard |

### 6.4 Automated accessibility audit (`@axe-core/playwright`)

All primary routes were scanned with Axe-core adhering to **WCAG 2.1 Level AA** standards:

| View Tested | Route | Critical Violations | Serious Violations | Minor/Notice |
| --- | --- | ---: | ---: | ---: |
| Mission Control | `/` | **0** | **0** | Color contrast on dark badges addressed |
| Multi-Agent Fleet | `/agents` | **0** | **0** | ARIA labels provided on diagram controls |
| Algorithm Lab | `/lab` | **0** | **0** | Form inputs have explicit labels and descriptions |
| Experiments Matrix | `/experiments` | **0** | **0** | Accessible table headers and sparkline descriptions |
| Theory Primer | `/theory` | **0** | **0** | Semantic heading hierarchy (`h1` -> `h2` -> `h3`) |
| Story Walkthrough | `/story` | **0** | **0** | Button controls and timeline landmarks navigable |

