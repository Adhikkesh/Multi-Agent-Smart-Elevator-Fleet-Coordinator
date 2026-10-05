# LiftZero — Case-Study Review Study Material

**Multi-Agent Smart Elevator Fleet Coordinator** · Foundations of Artificial Intelligence

| Member | Register no. | Presents (deck slides) | Live demo part |
| --- | --- | --- | --- |
| Adhikkesh | CB.SC.U4CSE23101 | 1–5 (problem, PEAS), 12 (multi-agent execution) | Mission Control, auctions |
| Sisr Reddy | CB.SC.U4CSE23129 | 11 (tools & setup), 13 (dashboard), 16 (code structure) | Agents page, faults & fire |
| Kavin Karthic | CB.SC.U4CSE23161 | 14 (testing), 17 (LiftZero extension), 18 (conclusion) | scenarios, tests, Brain page |
| Akash | CB.SC.U4CSE23162 | 6–7 (environment & agents), 8–10 (search), 15 (results) | Algorithm Lab, Experiments |

Every number in this document comes from our own runs; sources are given so you can show
them if asked. Read §1 and §11 first, then your own sections.

---

## 0. Rubric map — where each mark is earned

| Review | Rubric item (marks) | What to show | Deck | Dashboard page | Section |
| --- | --- | --- | --- | --- | --- |
| 1 | PEAS formulation (3) | system PEAS + PEAS of all 6 agents | 4, 5 | Theory | §2 |
| 1 | Environment & agent analysis (3) | 7-property classification, why each agent is its type | 6, 7 | Theory | §3, §4 |
| 1 | Algorithmic modelling & search (3) | routing as search, A\* heuristic, BFS/UCS/Greedy/A\* comparison, SA, minimax, rules | 8, 9, 10 | Algorithm Lab | §5 |
| 1 | Q&A & presentation (1) | clear answers, timing | all | — | §12 |
| 2 | Tool/package selection & setup (3) | why each tool, what was rejected, 2-command setup | 11 | — | §7 |
| 2 | Multi-agent execution & interaction (3) | FIPA-ACL messages, Contract Net round, live message flow | 12 | Agents, Mission Control | §6 |
| 2 | Demo quality & testing scenarios (3) | live demo, 9 scenarios, 523 automated tests | 13, 14, 15 | all | §8, §13 |
| 2 | Code structure & scalability (1) | module layout, YAML config, registry, 40-floor stress test | 16 | — | §9 |

---

## 1. The problem, and how we solved it

### 1.1 Problem statement (say this in your own words)

A building has **N floors** and **M lift cars**. People arrive at random times, press a hall
button (up or down), and only reveal where they are going **after they board** and press a
button inside the car. The control system must decide, every second, **which car answers
which call and in what order it visits its stops** — while cars break down, fire alarms
sound, and traffic changes through the day (in the morning everyone goes up from the lobby,
in the evening everyone comes down, at lunch both).

This is the **Elevator Group Control Problem (EGCP)**. It is hard because it is:

* **Combinatorial** — with *k* open calls and *N* cars there are *N^k* possible assignments
  (4 cars and 20 calls ≈ 10¹²).
* **Uncertain** — arrivals are random and destinations are hidden until boarding.
* **Sequential** — today's assignment decides where every car will be for the next minutes.
* **Multi-objective** — short average wait, nobody waiting very long (fairness), low energy.
* **Safety-critical** — never move with doors open, never overload, evacuate on fire.

The obvious answer — "send the nearest free car" — measurably fails: in the morning up-peak
it gives a **56.2 s** average wait against **29.9 s** for our full system (100 test seeds).

### 1.2 How we built the solution (the story, step by step)

1. **A simulated building** (Mesa 3, 1 tick = 1 second): floors, shafts and cars with real
   door and travel timings; passengers arrive by a Poisson process whose rate and pattern
   change during the day. Everything (floors, cars, timings, traffic, faults) is a YAML file.
2. **Six kinds of agents**, each with its own sensors and actuators (§2): passengers, floors,
   elevator cars, a dispatcher, a traffic monitor and a safety agent.
3. **Agents talk only by messages** (FIPA-ACL performatives over a message bus) and read a
   shared **status board** — no agent calls another agent's code (§6).
4. **Contract Net auctions**: when a floor reports a call, the dispatcher announces it, every
   car bids what serving it would cost, and the cheapest credible bid wins (§6.2).
5. **Each car's bid comes from search**: the car plans its stop order with **A\*** and bids the
   *extra* cost of adding the new call (§5.1–5.4).
6. **Global improvement**: **simulated annealing** periodically re-assigns calls; **hill
   climbing / minimax with alpha-beta** decides where idle cars wait (§5.4).
7. **Safety by rules**: a **forward-chaining** rule base handles fire recall, faults, overload
   and blocked doors (§5.5).
8. **Learning**: the traffic monitor **learns** the arrival rate of every floor and retunes the
   cost weights when the traffic pattern changes (§4).
9. **A live dashboard** (React) shows the building, every auction and every message (§7, §13).
10. **Testing**: 9 scenarios plus 451 Python and 72 dashboard tests; safety invariants are
    checked every tick of every run (§8).
11. **Extension**: *LiftZero*, a small neural network that learned to imitate the A\* bid (§10).

---

## 2. PEAS formulation (Review 1 — 3 marks)

### 2.1 System level

| | |
| --- | --- |
| **Performance** | average, 95th-percentile (p95) and maximum wait; ride and total system time; % of waits over 60 s; throughput (passengers/hour); energy (floors travelled + 2·stops + reversals); **zero safety violations**; fairness — no starved call |
| **Environment** | a multi-storey building: floors with up/down hall buttons and occupancy sensors, shafts, cars, passengers, and the other cars each car competes with |
| **Actuators** | car motors (up / down / stop), doors (open / close / hold), hall lanterns and displays, FIPA-ACL messages on the bus |
| **Sensors** | position encoders, load and door sensors, car and hall buttons, per-floor occupancy sensors, the message inbox |

**Key point for the viva:** the performance measure is *multi-objective and partly in
conflict*. Minimising energy alone would park every car and let people wait; minimising
average wait alone lets one unlucky floor starve. That is why the cars are **utility-based**
(they trade the objectives with weights W1–W4) and why fairness is enforced separately by
**call aging** (an old call gets an urgency discount so it eventually wins an auction).

### 2.2 Per agent

| Agent | AIMA type | Performance | Environment | Actuators | Sensors |
| --- | --- | --- | --- | --- | --- |
| Passenger | simple reflex | own wait, ride, being delivered | its floor, buttons, arriving cars | press hall button, board, press car button, alight | car present, direction, doors, space |
| Floor | model-based reflex | wait of its callers, no starved call | the landing, waiting people | hall lamps, lanterns, REQUEST messages | hall buttons, occupancy sensor, messages |
| Elevator | goal-based + utility-based | calls served cheaply, safe doors, capacity | its shaft, its riders, other cars | motor, doors, PROPOSE / REFUSE | position, load, door sensor, car buttons |
| Dispatcher | utility-based coordinator | fleet wait, p95, fairness, fault recovery | all calls and cars | CFP, ACCEPT / REJECT, CANCEL, parking orders | REQUEST, PROPOSE, REFUSE, FAILURE |
| Traffic monitor | learning agent | accurate demand estimate → lower wait | the arrival stream | INFORM new weights and parking policy | arrivals per floor and direction |
| Safety | knowledge-based | zero violations, correct recall | loads, doors, faults, alarm | recall, hold doors, out-of-service, block calls | load, door, fault sensors, fire alarm |

Each agent's PEAS is also written in its code docstring and shown on the dashboard's
**Theory** page.

---

## 3. Environment analysis (Review 1 — 3 marks, together with §4)

| Dimension | Our environment | Why |
| --- | --- | --- |
| Observable? | **Partially observable** | a passenger's destination is private until boarding; each agent sees only its own sensors and inbox |
| Agents | **Multi-agent, cooperative** | cars bid against each other, but all share one fleet objective |
| Deterministic? | **Stochastic** | Poisson arrivals, injected faults (a seed still replays a run exactly) |
| Episodic? | **Sequential** | an assignment now changes which car is well placed later |
| Static? | **Dynamic** | people keep arriving while the agents deliberate |
| Discrete? | **Discrete** | floors and 1-second ticks, discretised from continuous motion |
| Known? | **Known physics, unknown demand** | travel and door times are known; arrival rates must be learned |

**The two that shape the design:** *partially observable* → a car must commit to a call
before knowing where the person is going, then replan when they board; *dynamic* → never
trust a stored plan across an event; replan when something changes.

---

## 4. Agent analysis — why each agent is its type

* **Passenger — simple reflex.** One condition-action rule: *if a car is here, its doors are
  open, it serves my direction and has room → board.* No memory, no planning.
* **Floor — model-based reflex.** A simple reflex agent cannot do this job: the right action
  depends on history that is not in the current percept (has this call already been
  reported? how long has it waited?). So it keeps internal state (`active_calls`,
  `call_since`) — the textbook model-based extension — and that is what makes fairness
  possible.
* **Elevator — goal-based and utility-based.** *Goal-based* because it holds a set of stops to
  reach and **searches** for a plan; *utility-based* because many plans reach the goals and it
  must prefer one — quicker for riders, kinder to waiting people, cheaper in energy — by a
  weighted utility. That same utility is its auction bid.
* **Dispatcher — utility-based coordinator, not a controller.** It never commands a car: it
  asks (CFP), cars answer with their own costs, and it awards. So a broken car degrades
  service gracefully instead of breaking a central plan.
* **Traffic monitor — learning agent.** All four AIMA parts: *performance element* (the fleet
  doing the work), *critic* (observed arrivals), *learning element* (an EWMA estimate of each
  floor's arrival rate), *problem generator* (the pattern classifier that proposes new
  weights).
* **Safety — knowledge-based.** Its behaviour is a declarative rule base plus an inference
  engine, so every safety action is traceable to a rule and can be audited without reading
  code.

**AIMA definitions to remember:** *simple reflex* — acts on the current percept only;
*model-based* — keeps an internal state of the world; *goal-based* — plans to reach goals;
*utility-based* — chooses between plans by a utility function; *learning* — improves its
performance element from feedback.

---

## 5. Algorithmic modelling and search strategy (Review 1 — 3 marks)

### 5.1 Car routing as a state-space search problem

| Component | Definition |
| --- | --- |
| State | (car floor, direction, set of remaining stops) |
| Initial state | the car's current floor and direction with its pending stops |
| Actions | serve one *legal* next stop (collective control: keep sweeping in one direction) |
| Transition | move to that floor, open the doors, remove the stop |
| Step cost | travel time + door time, weighted by the people waiting at that stop |
| Goal test | no stops remaining |

### 5.2 The heuristic (why A\* is optimal here)

`h(state) = Σ wᵢ · travel(car, fᵢ) + energy lower bound of the floor span`

* **Admissible** (`h ≤ h*`): every pending stop costs at least its weighted straight-line
  travel time, and the car must at least sweep the span of pending floors. These are lower
  bounds on *disjoint* parts of the true cost, so their sum never overestimates.
* **Consistent** (`h(n) ≤ c(n, n′) + h(n′)`): serving one stop removes its term, and the travel
  to it is part of the step cost, so h never drops by more than the step cost.
  Consistent ⇒ admissible ⇒ A\* returns the optimal stop order (full proof in
  `docs/DESIGN.md` §6.1).
* **Bounded rationality:** with more than 10 pending stops the car uses a LOOK sweep instead
  (fast, always legal).

### 5.3 Search strategy comparison (our measurement)

200 random 7-stop routing problems; all four algorithms solve the *same* problem
(`uv run python scripts/search_benchmark.py` → `reports/search_benchmark_summary.csv`):

| Algorithm | Type | Optimal plans | Avg cost gap | Nodes expanded |
| --- | --- | --- | --- | --- |
| BFS | uninformed, fewest steps | 51.5 % | +9.2 % | 86.8 |
| UCS | uninformed, lowest cost | 100 % | 0 % | 54.1 |
| Greedy best-first | informed, h only | 6.0 % | +44.9 % | 7.7 |
| **A\*** | informed, g + h | **100 %** | **0 %** | **32.8** |

**Read it as:** BFS ignores cost, so it is often wrong; UCS is always right but slower; greedy
is fast but almost always wrong; **A\* is always right and expands 39 % fewer nodes than UCS**
— the heuristic pays for itself. The tests check this on 200 random instances: A\* cost = UCS
cost, A\* never expands more nodes than UCS, h admissible and consistent at every node.

| Property | BFS | UCS | Greedy | A\* |
| --- | --- | --- | --- | --- |
| Complete | yes | yes | yes (finite space) | yes |
| Optimal | only for equal step costs | yes | no | yes (admissible h) |
| Time / space | O(b^d) | O(b^(1+⌊C*/ε⌋)) | O(b^m) | exponential worst case, far less with a good h |

### 5.4 The bid, local search and adversarial search

* **The bid (utility):** `bid = W1·wait + W2·ride + W3·crowding + W4·energy − urgency·bonus`,
  where each term is the *marginal* cost of adding the call to the car's A\* plan. The urgency
  discount (call aging) makes a long-waiting call cheaper for everyone → no starvation.
* **Simulated annealing** (every 30 s): re-assign open calls between cars; accept a worse
  assignment with probability `e^(−Δ/T)` while T cools, to escape local minima. Tested: never
  worse than its starting assignment.
* **Hill climbing** chooses parking floors for idle cars toward expected demand.
* **Minimax with alpha-beta:** treat "where the next call appears" as an adversary and park
  idle cars to minimise the worst-case response distance; alpha-beta gives the same value
  while expanding fewer nodes (tested).

### 5.5 Knowledge-based safety (forward chaining)

Rules fire on facts, highest salience first, until nothing new can be derived:

| Rule (salience) | IF … THEN … |
| --- | --- |
| R1 fire recall (100) | fire alarm and car in service → recall the car to the lobby |
| R2 fire doors (90) | fire alarm and car at the lobby → hold its doors open |
| R3 block calls (80) | fire alarm → block all hall calls |
| R4 car fault (70) | car fault → out of service, re-auction its calls |
| R5 overload (60) | load > capacity → hold doors, refuse boarding |
| R6 door obstruction (50) | doors blocked too long → re-open them |
| R7 fire cleared (40) | no fire alarm and cars recalling → restore normal service |

---

## 6. Multi-agent execution and interaction (Review 2 — 3 marks)

### 6.1 How agents communicate

* **FIPA-ACL messages** on a bus: each message has a *performative* (REQUEST, CFP, PROPOSE,
  REFUSE, ACCEPT_PROPOSAL, REJECT_PROPOSAL, INFORM, CANCEL, FAILURE), a sender, a receiver, a
  conversation id and content.
* **Status board (blackboard):** every car publishes its own status (floor, direction, load,
  doors, plan); the traffic monitor publishes the current pattern and weights.
* **The rule:** no agent calls another agent's methods — everything goes through messages or
  the board. That is what makes this a genuine multi-agent system.

### 6.2 One Contract Net round (per hall call)

```
Floor       ──REQUEST(call)────────────────────────▶ Dispatcher
Dispatcher  ──CFP(call, urgency, waiting)──────────▶ every car
Car i       ──PROPOSE(bid_i)  or  REFUSE(reason)───▶ Dispatcher
Dispatcher  ──ACCEPT_PROPOSAL──▶ winner (lowest bid, ties → lowest car id)
Dispatcher  ──REJECT_PROPOSAL──▶ the other bidders
Winner      ──INFORM / status update───────────────▶ board, floor lantern
```

* A car **refuses** if it is full, out of service or in fire mode.
* Calls are auctioned **one at a time**, so each auction sees the previous awards.
* **Staged tick:** sense → negotiate (announce, bid, award) → decide → act. In every stage all
  agents work from the same snapshot of the building, so a run is deterministic.
* About **12 messages per call** in a 4-car building (shown live as "MSGS / CALL").

### 6.3 Interaction under faults

* **Car fault:** the car goes out of service, its calls are released and re-auctioned to the
  other cars, its riders leave at the current floor (rule R4).
* **Fire alarm:** rules R1–R3 recall every car to the lobby, hold the doors open and block hall
  calls; R7 restores service when the alarm clears.

---

## 7. Tool / package selection and setup (Review 2 — 3 marks)

| Need | Chosen | Why | Rejected (and why) |
| --- | --- | --- | --- |
| Language & environment | Python 3.12 + **uv** | `uv sync` installs Python itself and the exact package versions | pip/venv (cannot install Python), conda (heavier) |
| Agent framework | **Mesa 3** | the standard Python agent-based modelling library; seeded randomness and data collection | SPADE (needs an XMPP server), JADE (Java) |
| Server | **FastAPI + uvicorn** | REST for control, WebSocket for live streaming, input validation | Flask (WebSockets need add-ons), Django (too heavy) |
| Configuration | **pydantic + YAML** | scenarios are data; a bad file fails with a clear error | hand-written validation |
| Dashboard | **React 18 + TypeScript + Vite** (Tailwind CSS, Zustand, Zod) | smooth canvas animation, typed API, fast builds | pygame (no dashboard widgets), Streamlit (re-runs the whole script) |
| Analysis | **pandas + matplotlib** | benchmark tables and 300-DPI charts | hand-rolled statistics |
| Quality | **pytest, Vitest, ruff** | property tests over hundreds of seeded random cases; one fast linter | unittest (more boilerplate) |
| Learning extension | **PyTorch → ONNX Runtime** | train once, run on CPU without PyTorch at demo time | shipping PyTorch to the demo machine |

### 7.1 Setup (Windows, macOS or Linux)

```bash
git clone https://github.com/Adhikkesh/Multi-Agent-Smart-Elevator-Fleet-Coordinator.git
cd Multi-Agent-Smart-Elevator-Fleet-Coordinator
pip install uv              # once, if uv is not installed (macOS: brew install uv)
uv sync                     # downloads Python 3.12 + all packages
uv run elevator serve       # then open http://localhost:8000
```

The dashboard is pre-built and committed, so **Node.js is not needed** to run the demo.

---

## 8. Testing scenarios and results (Review 2 — 3 marks)

### 8.1 The nine scenarios (each a YAML file in `configs/scenarios/`)

Each is run to completion, then *drained* (arrivals off, run until everyone is delivered).

| Scenario | What it tests | Avg wait | p95 | Delivered | Safety rules fired |
| --- | --- | --- | --- | --- | --- |
| morning_up_peak | lobby rush | 12.1 s | 39.6 s | 135/135 | — |
| evening_down_peak | rush to the lobby | 22.4 s | 69.8 s | 164/164 | — |
| lunch_two_way | both directions | 12.7 s | 42.8 s | 123/123 | — |
| interfloor_light | light mixed traffic | 9.7 s | 24.0 s | 52/52 | — |
| car_breakdown | car fault and recovery | 10.9 s | 33.2 s | 136/136 | R4 |
| fire_emergency | fire recall | 71.5 s | 265.8 s | 130/130 | R1, R2, R3, R7 |
| priority_passenger | priority callers | 14.1 s | 44.1 s | 138/138 | — |
| stress_scale | 40 floors × 8 cars, 1 hour | 18.5 s | 63.2 s | 377/377 | — |
| demo_story | surge + fault + fire | 72.4 s | 164.5 s | 68/68 | R1, R2, R3, R4, R7 |

Fire-scenario waits are high **by design**: for 270 s every car is recalled to the lobby and
calls are blocked — a low wait there would mean the evacuation was not taken seriously.

### 8.2 Automated tests — 451 Python + 72 dashboard = 523

| Area (count) | What is proven |
| --- | --- |
| Search (18) | A\* = UCS cost on 200 random instances; A\* expands ≤ UCS; h admissible and consistent; plans obey collective control |
| Optimisation (14) | SA is never worse than its input; alpha-beta = minimax value with fewer nodes |
| Safety rules (15) | fire recall, door hold, overload, faults; rules reach a fixed point |
| Protocol (12) & messaging (18) | correct Contract Net sequence; agents never call each other directly |
| Simulation invariants (29) | every tick: no motion with doors open, load ≤ capacity, floors in range, everyone delivered |
| Scenarios (35) | every scenario meets the `expected:` block in its own YAML |
| API (45) | every endpoint, validation errors, the WebSocket stream |
| Golden metrics (25) | the four classical strategies give bit-identical results across code changes |
| LiftZero extension (≈200) | the learned bidder: model, training, runtime, safety on every scenario |
| Dashboard (72, Vitest) | data schemas, stores, A\* parity with Python, Brain panel |

**0 safety violations** in every scenario and in 7,800 evaluation runs.

### 8.3 Strategy comparison (100 paired test seeds per traffic pattern)

| Average wait (s) | Nearest car | Collective | CNP + A\* | **Full system** |
| --- | --- | --- | --- | --- |
| Morning up-peak | 56.2 | 47.1 | 42.9 | **29.9** |
| Evening down-peak | 22.4 | **21.1** | 22.0 | 23.7 |
| Lunch two-way | 18.7 | 17.4 | **14.5** | **14.5** |
| Inter-floor | 12.5 | 11.6 | 11.8 | **11.0** |

**Be honest about down-peak:** plain collective control is slightly better there, because
down-peak (cars fill at the top, empty at the lobby) is exactly what a simple sweep is optimal
for. Our system wins where it matters most — the morning peak (−47 %) — and is far more
consistent across seeds.

---

## 9. Code structure and scalability (Review 2 — 1 mark)

```
src/elevator_mas/
  agents/        the six agents (each with its PEAS docstring)
  comms/         FIPA-ACL messages, message bus, status board
  planning/      BFS / UCS / Greedy / A*, the car routing problem
  optimization/  simulated annealing, hill climbing, minimax + alpha-beta
  rules/         forward-chaining engine and safety rules
  strategies/    pluggable dispatch strategies (registry)
  traffic/       Poisson arrivals, traffic patterns
  model.py       the Mesa model: staged activation, wiring
  sim/           headless runner and benchmark
  api/           FastAPI REST + WebSocket server
  web/dist/      the built React dashboard
  learning/      (extension) the LiftZero learned bidder
configs/scenarios/*.yaml    tests/    docs/    frontend/ (dashboard source)
```

* **Separation:** the simulation engine never imports the server or the UI — it runs headless
  in tests and benchmarks.
* **Configuration, not code:** floors, cars, capacity, timings, traffic, faults and strategy
  are YAML. The 40-floor × 8-car stress test is only a config file.
* **Extensible:** a new dispatch strategy is one registry entry; a new safety policy is one rule.
* **Scales:** 40 floors × 8 cars, one simulated hour in **under 4 s** (≈1 000× real time); replanning is event-driven;
  above 10 stops the search falls back to LOOK.
* **Quality:** type hints throughout, `ruff`-clean, all randomness seeded (every run is
  reproducible).

---

## 10. Extension: LiftZero — a learned bidder (present briefly; answer if asked)

* A small neural network (a *set-Transformer*, 237,575 parameters) learned to **imitate the A\*
  bid** from about 1.5 million recorded auction decisions — behaviour cloning, then DAgger
  (the network drives, A\* labels the situations it creates).
* It agrees with the A\* winner **89.9 %** of the time on unseen decisions, prices a call in
  **0.19 ms**, and keeps waiting times within ±5 % of A\* in 13 of 13 scenarios (plain Contract
  Net version), with 0 safety violations. Eligibility (full / faulty) is still decided by the
  rules, never by the network.
* An optional **look-ahead** (Monte-Carlo tree search) re-checks close auctions by simulating
  the next minute; on 20 test seeds it lowered waits in all four patterns, but not
  significantly.
* Shown live on the dashboard's **LiftZero Brain** page, with a "Why?" explanation of each
  decision.
* Not finished: training with reinforcement learning (implemented, not trained) — future work.

---

## 11. Commands — navigate and run everything

Run these from the project folder.

| Goal | Command |
| --- | --- |
| Install / update | `uv sync` |
| Start the dashboard | `uv run elevator serve` → http://localhost:8000 |
| Start on a chosen scenario | `uv run elevator serve --scenario fire_emergency` |
| List scenarios and strategies | `uv run elevator scenarios` |
| Run one scenario in the terminal | `uv run elevator run --scenario morning_up_peak` |
| …with a chosen strategy / seed | `uv run elevator run --scenario lunch_two_way --strategy nearest_car --seed 7` |
| Check every scenario for safety violations | `uv run elevator verify` |
| Benchmark strategies (CSV + charts in `reports/`) | `uv run elevator bench` |
| Search algorithm comparison | `uv run python scripts/search_benchmark.py` |
| Run all tests | `uv run pytest` (quick: `uv run pytest -m "not slow"`) |
| Lint | `uv run ruff check .` |
| Learned bidder: model card / speed | `uv run elevator lift card` · `uv run elevator lift bench-infer` |
| Regenerate the charts | `uv run python scripts/generate_viva_charts.py` |

**Dashboard pages:**

| Page | URL | Use it for |
| --- | --- | --- |
| Mission Control | http://localhost:8000/ | building, cars, auctions, messages, faults & fire |
| Agents | http://localhost:8000/agents | agent graph, message sequence, status board |
| Algorithm Lab | http://localhost:8000/lab | BFS / UCS / Greedy / A\* step-through, annealing, minimax |
| Experiments | http://localhost:8000/experiments | multi-seed benchmarks, compare two strategies |
| Theory | http://localhost:8000/theory | PEAS tables, environment classification |
| LiftZero Brain | http://localhost:8000/brain | learned bids, attention, look-ahead, "Why?" |
| Story Mode | http://localhost:8000/story | guided walkthrough |

**Keyboard on the dashboard:** `Space` play/pause, `→` step one tick (the `?` button lists all
shortcuts).

**Important files:** `docs/DESIGN.md` (full design and proofs), `docs/TESTING.md` (all test
results), `docs/DEMO_SCRIPT.md` (the demo), `configs/scenarios/` (scenarios),
`docs/LiftZero_Viva_Deck.pptx` (slides).

---

## 12. Viva questions and answers

### PEAS and environment

1. **What is PEAS?** Performance measure, Environment, Actuators, Sensors — the AIMA way to
   specify a task environment before designing the agent.
2. **What is your performance measure?** Average, p95 and maximum wait, % of waits over 60 s,
   throughput, energy, and zero safety violations — several objectives traded off by weights.
3. **Why is the environment partially observable?** A passenger's destination is unknown until
   they board; each agent sees only its own sensors and messages.
4. **Why stochastic and not deterministic?** Arrivals follow a Poisson process and faults are
   injected; with a fixed seed a run replays exactly, so it is *reproducibly* stochastic.
5. **Why sequential?** Assigning a call now decides where the cars will be for later calls.
6. **Why dynamic?** People arrive while the agents deliberate, so plans go stale; we replan on
   every event.
7. **Is it cooperative or competitive?** Cooperative: cars compete in auctions only as a way to
   allocate work; all share one fleet objective.

### Agents

8. **Why is the floor model-based and not simple reflex?** It must remember whether a call was
   already reported and how long it has waited — information not in the current percept.
9. **Why is the elevator utility-based?** Several plans reach its goals; it needs a utility to
   prefer the plan that best trades wait, ride time and energy — and that utility is its bid.
10. **What makes the traffic monitor a learning agent?** It estimates each floor's arrival rate
    from observations (EWMA), classifies the traffic pattern and changes the cost weights.
11. **Is the dispatcher a central controller?** No. It only announces calls and compares bids;
    it never commands a car or reads its internal state.
12. **Why a knowledge-based safety agent?** Safety rules are declarative, auditable and
    traceable; priorities between rules are explicit (salience).

### Search and algorithms

13. **How is routing formulated as search?** State = (floor, direction, remaining stops);
    action = serve a legal next stop; cost = weighted travel + door time; goal = no stops left.
14. **Why A\* and not UCS?** Same optimal answer, 39 % fewer nodes expanded in our benchmark,
    because the heuristic guides the search.
15. **Prove your heuristic is admissible.** It adds two lower bounds on separate parts of the
    remaining cost (straight-line travel to every stop, and sweeping the floor span), so it can
    never overestimate.
16. **What is consistency and why does it matter?** `h(n) ≤ c(n,n′) + h(n′)`; it guarantees A\*
    never re-opens a node, and it implies admissibility.
17. **Why not greedy best-first?** It is fast but found the optimal plan only 6 % of the time
    (+44.9 % cost on average).
18. **What does simulated annealing do here?** Every 30 s it re-assigns open calls between cars,
    sometimes accepting a worse move (probability e^(−Δ/T)) to escape local minima.
19. **Where is minimax used?** For parking idle cars: the next call's location plays the
    adversary; alpha-beta pruning gives the same answer with fewer nodes.
20. **What is bounded rationality in your system?** Above 10 pending stops a car uses a LOOK
    sweep instead of A\* — good enough and always fast.
21. **How do you prevent starvation?** Call aging: the longer a call waits, the bigger its
    urgency discount, so some car eventually wins it.

### Multi-agent interaction

22. **What is the Contract Net Protocol?** The manager sends a call for proposals; contractors
    propose or refuse; the manager accepts the best and rejects the rest; the winner informs.
23. **What is FIPA-ACL?** A standard agent communication language: each message carries a
    performative (the speech act) plus sender, receiver, conversation id and content.
24. **What if a car fails?** It goes out of service, refuses all calls, its calls are
    re-auctioned and its riders leave at the current floor (rule R4).
25. **What happens during a fire alarm?** R1 recalls the cars to the lobby, R2 holds the doors
    open, R3 blocks hall calls; R7 restores service afterwards.
26. **Why auctions instead of the dispatcher computing everything?** Each car knows its own route
    best; auctions keep that knowledge local, scale linearly with cars and tolerate failures.

### Tools, testing, structure

27. **Why Mesa?** The standard Python agent-based modelling framework, with seeded randomness
    and data collection; SPADE needs an XMPP server and JADE is Java.
28. **Why FastAPI?** Native WebSockets for streaming every tick to the dashboard, plus REST and
    validation.
29. **How do you know the system is correct?** 523 automated tests; correctness properties are
    asserted on hundreds of random seeded cases; safety invariants checked every tick.
30. **How does it scale?** 40 floors × 8 cars, one simulated hour in under 4 seconds; only a YAML file
    changes.
31. **Where does your system lose?** Evening down-peak: plain collective control is slightly
    better, because a simple sweep is already optimal for that pattern. We report it.
32. **How do you make runs reproducible?** All randomness comes from the model's seeded random
    generator; the same seed gives the same run (tested).

### Extension

33. **What is LiftZero?** A neural network that learned to imitate the A\* bid; it agrees with
    A\* about 90 % of the time and is much faster per decision.
34. **Why imitation learning, then DAgger?** Imitation learns from A\*'s situations; DAgger lets
    the network drive and asks A\* to label the new situations it creates, fixing drift.
35. **Did you use reinforcement learning?** It is implemented but not trained in this
    submission; it is our future work.

---

## 13. Live demonstration

The full step-by-step script with timings and who does what is in
[`docs/DEMO_SCRIPT.md`](DEMO_SCRIPT.md). In short: start the server → Mission Control (play
morning up-peak, explain an auction) → inject a car fault and a fire alarm → Agents page
(message flow) → Algorithm Lab (A\* vs UCS) → Experiments (nearest car vs full) → Theory
(PEAS) → LiftZero Brain (optional).

---

## 14. Numbers to remember

* Up-peak average wait: nearest car **56.2 s** → full system **29.9 s** (**−47 %**).
* A\*: optimal **100 %**, **32.8** nodes vs UCS **54.1** (**−39 %**); greedy optimal only **6 %**.
* **9** scenarios, everyone delivered, **0** safety violations; **7** safety rules.
* **523** automated tests (451 Python + 72 dashboard).
* **40 floors × 8 cars**: one hour simulated in **under 4 s**.
* About **12 messages per call**; **6** agent types; **9** FIPA performatives.
* LiftZero (extension): **237,575** parameters, **89.9 %** agreement, **0.19 ms** per decision.
