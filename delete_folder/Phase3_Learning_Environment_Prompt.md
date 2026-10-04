# PHASE 3 — The learning environment for LiftZero: feature schema, expert-data recorder, fast twin simulator, Gym env

> Paste this whole file into Antigravity as the task (Planning mode, your strongest model).
> Facts below were read from the real code at commit `2ef99c0` on branch `v2`.
> If you run Phase 2 and Phase 3 in parallel, use **separate git worktrees/branches** (see §1);
> the two phases touch different files and merge cleanly.

---

## 0. Context and goal

Project: a university AI case study — **a multi-agent smart-elevator fleet** built on Mesa 3
(`src/elevator_mas/`). Today every car computes its own *bid* (marginal cost of taking a hall
call) with an A* search, and a dispatcher runs a Contract Net auction (lowest bid wins).
That classical system is our **teacher**.

**LiftZero** (the next phases) replaces each car's bid computation with a small neural
network (a permutation-invariant set-Transformer, ~0.3 M parameters, CPU only). It will be:

* **Phase 4** — trained by *imitation* of the teacher (needs a dataset of expert decisions),
* **Phase 5** — improved by cooperative **PPO** (needs a *fast* simulator and a Gym env),
* **Phase 6** — combined with MCTS look-ahead and plugged into the live system.

**Your job in Phase 3 is to build everything those phases stand on — and nothing else:**

1. a **versioned feature schema** + encoder (the single definition of "what a car sees"),
2. a **decision recorder** that harvests ~500 000 expert decisions from the *real* simulator,
3. a **fast numpy/numba "twin" simulator** (≥ 20× faster than the real one, ±10 % fidelity),
4. a **Gymnasium environment** (semi-MDP: one decision per hall call) + a vectorised wrapper,
5. **regimes + baselines + evaluation harness**, with honest fidelity and speed reports.

Do **not** build any neural network, training loop, ONNX export or UI. Do not use PyTorch.
Do not change any behaviour of the existing simulator (see §2 rule on the hook).

**Done = §11 acceptance checklist, with evidence.**

---

## 1. Working agreement

* New branch `phase-3-learning-env` from `v2` (if Phase 2 runs concurrently, create it with
  `git worktree add ../elevator-p3 -b phase-3-learning-env v2` and work there). Small commits.
  **Do not push, do not touch `main`.**
* Python 3.12, `uv`. Keep `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .` green
  (line length 100, ruff rules in `pyproject.toml`). Existing suite = **201 tests, all must pass**.
* New runtime deps go in a **new optional dependency group** so the base install stays light:
  in `pyproject.toml` add `[dependency-groups] learn = ["numpy>=1.26", "numba>=0.60", "gymnasium>=1.0", "scipy>=1.13", "hypothesis>=6.100"]`
  (numpy is already a pandas dependency). `uv sync --group learn` installs them. Base
  `uv sync` + `uv run pytest` must still work **without** the group: guard the learning
  tests with `pytest.importorskip` and keep `elevator_mas.learning` import-light (lazy
  numba import) so the dashboard never needs it. If `numba` cannot be installed on the
  user's machine, the twin must still run in pure numpy/Python (slower) — keep a
  `ELEVATOR_NO_NUMBA=1` escape hatch and make the speed test skip (not fail) in that case.
* First write a short plan (modules, data flow, test list); then execute. **The code wins over
  this prompt** where they disagree — note discrepancies in the final report.

---

## 2. What exists today (read these files first)

| File | Why you need it |
| --- | --- |
| `src/elevator_mas/model.py` | `ElevatorModel.step()`: per tick `tick += 1` → `_apply_events` → `_generate_arrivals` → stages `sense, communicate, negotiate, decide, act, learn` (negotiate = repeated `announce → bid → award → commit` rounds, one hall call per round). `model.board` is the public `StatusBoard`. |
| `src/elevator_mas/comms/board.py` | `CarStatus` (the **public** per-car state — the *only* thing a car's network may see about others) and `FleetPolicy` (pattern, weights, EWMA demand). |
| `src/elevator_mas/agents/dispatcher.py` | `announce()` sends the CFP `{call: HallCall, urgency: int, waiting: int}`; `award()` collects PROPOSE/REFUSE, builds `list[Bid]`, picks `min((total, car_id))` among non-refused, records an `AuctionRound`. |
| `src/elevator_mas/agents/elevator.py` | The teacher's bid: `compute_bid()` / `marginal_cost()`; physics in `_advance_physics()` (doors → motion → boarding), `seconds_per_floor`, door open/close, `boarding_per_passenger`, `dwell_min`; LOOK fallback routing for `nearest_car` / `collective` strategies. |
| `src/elevator_mas/domain.py` | `HallCall`, `Stop`, `Bid` (`total` is `inf` when `refused`), `AuctionRound`, `PassengerRecord`, `Direction`, `DoorState`. |
| `src/elevator_mas/config.py` | `ScenarioConfig` and friends. Note defaults: capacity 10, `seconds_per_floor=2`, `door_open=2`, `door_close=2`, `boarding_per_passenger=1`, cost weights `wait=1.0, ride=0.5, crowding=0.3, energy=0.2`. `1 tick = 1 simulated second`. |
| `src/elevator_mas/traffic/{generator,profiles}.py` | Poisson arrivals with a time-varying rate; 5 patterns (`up_peak, down_peak, two_way, interfloor, light`) defined by `P(origin = lobby)` and `P(destination = lobby)`. The twin must reproduce these distributions exactly. |
| `src/elevator_mas/strategies/registry.py` | The four strategies: `nearest_car` (reflex, LOOK routing), `collective` (LOOK, car already sweeping towards call takes it), `cnp_astar` (Contract Net + A* bids), `full` (+ SA reassignment, parking, weight learning). **`cnp_astar` and `full` are the teachers.** |
| `src/elevator_mas/metrics.py`, `sim/runner.py`, `sim/benchmark.py` | Metrics (`avg_wait, p95_wait, max_wait, long_wait_pct, throughput, energy, …`), `run_scenario`, `run_benchmark`. |
| `configs/scenarios/*.yaml` | 9 scenarios; durations 330–3600; 15 floors × 4 cars except `stress_scale` (40 × 8). |
| `tests/` | 201 tests; conventions: classes of tests, `small_config` fixture, strict typing. Mirror the style. |

**Allowed change to the engine — exactly one, and it must be behaviour-neutral:**
add an *observer hook* so a recorder can see each auction without touching dispatch logic.

```python
# model.py
self.decision_hooks: list[Callable[[DecisionEvent], None]] = []   # empty by default

# dispatcher.announce(): just before sending the CFP, remember the public view
# CarStatus entries are frozen; FleetPolicy is mutable, so COPY it (shallow copy + dict(demand))
self._round_view = (self.model.board.cars(), copy_policy(self.model.board.policy))

# dispatcher.award(): after the winner is chosen (or all refused), if hooks are set:
for hook in self.model.decision_hooks:
    hook(DecisionEvent(tick=..., call=call, urgency=..., waiting=..., statuses=view_cars,
                       policy=view_policy, bids=bids, winner=winner_id_or_None,
                       building=self.model.config.building, seed=self.model.seed_value))
```

`DecisionEvent` is a frozen dataclass (put it in `comms/board.py` or a new `events.py`).
**Proof of neutrality is required**: a test that runs 3 scenarios × 2 strategies × 2 seeds
with and without a no-op hook and asserts identical `Metrics.as_dict()` and identical
`bus.total_sent`, plus the old suite passing.

---

## 3. Package layout to create

```
src/elevator_mas/learning/
  __init__.py          # lazy; exports the public API listed in §9
  schema.py            # FEATURE_VERSION, dims, names, normalisation constants, MAX_CARS
  view.py              # FleetView (SoA numpy) + DecisionContext + from_board()/from_event()
  features.py          # encode_decision(...) -> (call[Kc], cars[N,Kcar], mask[N], glob[Kg])
  recorder.py          # DecisionRecorder (hook) + shard writer + CLI worker
  dataset.py           # shard reader, split logic, stats, 1 000-decision fixture builder
  twin/
    __init__.py
    state.py           # structure-of-arrays state for B parallel envs
    kernels.py         # numba @njit kernels (pure-python fallback via same functions)
    traffic.py         # Poisson + OD profiles, same distributions as traffic/generator.py
    policies.py        # nearest, collective, cost_greedy, random (scorers on the encoded obs)
  env.py               # ElevatorDecisionEnv (gymnasium.Env) + VectorDecisionEnv
  regimes.py           # loads configs/learning/regimes.yaml
  evaluate.py          # evaluate(policy, regimes, backend="twin"|"real", seeds)
configs/learning/regimes.yaml
docs/LEARNING_ENV.md
tests/learning/ (test_schema.py, test_features.py, test_recorder.py, test_hook_neutrality.py,
                 test_twin_invariants.py, test_twin_fidelity.py, test_env_api.py, test_regimes.py,
                 test_evaluate.py)
```

and a `learn` sub-command group on the existing Typer app in `src/elevator_mas/cli.py`:

```
elevator learn record        --teacher cnp_astar|full|mixed --decisions 500000 --workers N --out data/expert --seed-start 0
elevator learn dataset-info  --path data/expert
elevator learn validate-twin --seeds 10 --ticks 900 --report docs/LEARNING_ENV_fidelity.md
elevator learn bench-env     --envs 64 --steps 20000
elevator learn baselines     --backend twin|real --seeds 20 --split val|test
```

`data/` is git-ignored (add to `.gitignore`). Only the 1 000-decision fixture
(`tests/learning/fixtures/expert_sample.npz`, < 300 kB) is committed.

---

## 4. The feature schema (version 1) — the contract with Phase 4–6

This is the most important deliverable. Phase 4 builds the network *against these exact
names and shapes*, so don't improvise — implement as written, expose everything via
`schema.py`, and bump `FEATURE_VERSION` for any later change.

A **decision** = one CFP: "who should take hall call *(floor, direction)*?" The network
sees one **call token**, `N ≤ MAX_CARS` **car tokens**, and one **global token**, plus a
boolean mask. `MAX_CARS = 16` in recorded shards and in the env (the model itself will
support up to 32; keep `MAX_CARS` a module constant so it can be raised).

Everything is derived **only** from public information: the CFP content, the cars'
`CarStatus` entries and the `FleetPolicy` on the board, and static building config. It
must **never** use the teacher's bids, a car's private planner state, or passenger
objects. (That is the "shared view": cars in the real system publish `CarStatus` each tick.)

All features are `float32`, scaled to ~[-1, 1] or [0, 1] as stated, clipped, and never
NaN/inf. `F` = number of floors, `cap` = car capacity, `Fm1 = max(F-1, 1)`.

**Call token — `Kc = 14`**

| # | name | definition |
| --- | --- | --- |
| 0 | `call_floor` | `floor / Fm1` |
| 1 | `call_dir` | `+1` UP, `-1` DOWN |
| 2 | `call_is_lobby` | 1 if `floor == lobby` |
| 3 | `call_waiting` | `min(waiting, 20) / 20` (CFP `waiting`) |
| 4 | `call_urgency` | `min(urgency, 4) / 4` (CFP `urgency`) |
| 5 | `call_dist_to_lobby` | `abs(floor - lobby) / Fm1` |
| 6 | `call_demand_share` | `policy.demand[floor] / max(sum(demand), 1e-6)`, clipped to [0,1]; 0 if demand empty |
| 7–11 | `pattern_*` | one-hot of `policy.pattern` over `[up_peak, down_peak, two_way, interfloor, light]` |
| 12 | `n_calls_open_norm` | `min(open hall calls in the building, 20) / 20` (from `Σ len(assigned_calls)` over statuses + 1) |
| 13 | `fleet_size_norm` | `n_cars / MAX_CARS` |

**Car token — `Kcar = 26`** (from `CarStatus`; `d` = signed distance car→call)

| # | name | definition |
| --- | --- | --- |
| 0 | `floor` | `floor / Fm1` |
| 1–3 | `dir_*` | one-hot `[UP, DOWN, IDLE]` |
| 4 | `load` | `load / cap` |
| 5 | `space` | `space / cap` |
| 6 | `available` | 1/0 |
| 7 | `out_of_service` | 1/0 |
| 8 | `fire_mode` | 1/0 |
| 9–12 | `door_*` | one-hot `[closed, opening, open, closing]` |
| 13 | `door_blocked` | `min(door_blocked_ticks, 10) / 10` |
| 14 | `n_assigned` | `min(len(assigned_calls), 10) / 10` |
| 15 | `n_car_calls` | `min(len(car_calls), 10) / 10` |
| 16 | `riders` | `min(riders, cap) / cap` |
| 17 | `plan_stops` | `min(planned_stops, 12) / 12` |
| 18 | `plan_end_floor` | `plan_end_floor / Fm1`, or the car's own floor if `None` |
| 19 | `plan_end_eta` | `min(plan_end_eta, 240) / 240` |
| 20 | `dist_to_call` | `abs(d) / Fm1` |
| 21 | `signed_dist` | `d / Fm1` where `d = call_floor - car_floor` |
| 22 | `heading_to_call` | 1 if the car is moving (or committed) toward the call floor, −1 away, 0 idle |
| 23 | `call_on_route` | 1 if `call_floor` lies between the car's floor and `plan_end_floor` in its direction of travel AND the call direction matches (a "free" pickup on the sweep) |
| 24 | `has_same_call` | 1 if this car already has the *same* hall call assigned |
| 25 | `park_target_dist` | `abs(park_target - call_floor)/Fm1` if `park_target` set else `abs(floor - call_floor)/Fm1` |

**Global token — `Kg = 6`**: `tick_in_scenario` (`min(tick, 3600)/3600`), `F/40`, `cap/20`,
`n_available/ n_cars`, `fleet_total_load` (`Σload / Σcap`), `fleet_idle_frac`.

Mask `mask[i] = True` iff car `i` exists (padding is `False`). Cars that are **not
eligible** (out of service, fire mode, no space) stay *in* the token set with their
features visible (the network can learn to refuse them) but are flagged in a second
boolean `eligible[i]`; the teacher's `REFUSE` ⇔ `~eligible[i]` for the recorded data
(verify this holds ≥ 99.9 % and report the exceptions — they are bugs or races).

Provide `encode_decision(ctx: DecisionContext) -> Encoded` returning a frozen dataclass with
`call (Kc,)`, `cars (MAX_CARS, Kcar)`, `glob (Kg,)`, `mask (MAX_CARS,)`, `eligible (MAX_CARS,)`.
Also `encode_batch(list[DecisionContext])` vectorised. `DecisionContext` is built either
from a `DecisionEvent` (real sim) or from the twin state — **both code paths must call the
same encoder** so a policy trained on one runs on the other. Include a **parity test**:
build a hand-made `FleetView` two ways (from a real `CarStatus` list, and from twin arrays)
and assert `encode` outputs are bit-identical.

Car ordering: tokens are in `car_id` order; the model is permutation-invariant, but tests
must also check that shuffling cars (and the label) leaves the *set* of tokens unchanged.

---

## 5. Decision recorder and expert dataset

`DecisionRecorder` registers on `model.decision_hooks`; for each auction (including ones
where everyone refuses) it stores the encoded tokens, the **teacher costs**
(`teacher_cost[N]` = each car's `Bid.total`; `1e6` where refused/padded), the **winner**
(`-1` if nobody could take it), and metadata.

Shard format (`.npz`, `np.savez_compressed`, ~50 000 decisions per shard):

```
call          float32 [M, 14]       cars          float32 [M, 16, 26]
glob          float32 [M, 6]        mask          bool    [M, 16]
eligible      bool    [M, 16]       teacher_cost  float32 [M, 16]
teacher_wait/ride/crowding/energy  float32 [M, 16]   # the bid breakdown (auxiliary targets)
winner        int16   [M]           run_id        int32   [M]
tick          int32   [M]           split         int8    [M]   # 0 train, 1 val, 2 test
meta_json     str     (scalar)      # feature_version, teacher, schema hash, git sha, config ranges
```

**Domain randomisation** (so the learned policy generalises — this is graded as "scalability"):
each recorded run draws, from a seeded RNG, a building and traffic from the ranges in
`configs/learning/regimes.yaml` `training:` block —
floors 6–40, cars 2–8 (≥ 1 car per ~8 floors), capacity 6–16, `seconds_per_floor` 1–3,
rate per car-per-second within a band so the fleet is neither idle nor saturated
(use `rate = u * cars * 0.04`, `u ~ U(0.6, 1.8)`), a random sequence of 1–4 phases drawn
from the five patterns, `priority_probability ∈ {0, 0.05}`, and with probability 0.25 a
scripted disturbance (car fault + repair, or fire alarm + clear) at a random tick.
Teacher strategy per run: `cnp_astar` or `full` (`--teacher mixed` = 50/50). Duration 600–1800 ticks.

**Splits are by run, never by decision** (decisions in one run are strongly correlated):
`run_seed < 8000` → train, `8000–8499` → val, `8500–8999` → test; held-out **building
sizes** too: reserve (floors=33–40, cars=7–8) *only* for the test split to measure scaling
(report separately as `test_large`). Never let a test run's seed appear in train.

Throughput: use `multiprocessing` (spawn), one process per worker, each writing its own
shards; the parent only coordinates and prints a progress bar with decisions/s. Target:
**≥ 500 000 decisions** in < 40 min on 8 cores (document your measured rate; at the very
least produce the sample + a 50 000-decision dataset on this machine and the exact command
the user runs for the full set). The recorder must be resumable (skip existing shards) and
deterministic given `--seed-start`.

`dataset-info` prints: decisions per split, per teacher, per pattern, fraction all-refused,
fraction with a single eligible car (trivial decisions!), distribution of winner margin
(`best vs 2nd best cost`), and the **"hard decision" fraction** (margin < 10 % of best cost) —
Phase 4 will report accuracy on hard decisions separately, so this number matters.

Also output a **teacher-baselines table** on the recorded data: how often `nearest-car by
distance` and `lowest ETA` happen to equal the teacher's winner (this is the floor the
network must beat — expected ~50–65 %; report what you measure).

---

## 6. The twin simulator

### 6.1 Purpose and scope

The real simulator is a Mesa agent model with A* planning: about 0.5–2 ms per tick.
PPO needs tens of millions of ticks. The **twin** is a stripped-down, array-based
re-implementation of the *physical* world and the *assignment decision*; it is **not** a
Mesa model and has **no** messaging, planners, safety rules or UI.

What the twin **must** model faithfully:

* Poisson arrivals, 5 OD patterns, optional priority (weight 3.0) exactly as `traffic/`.
* Cars: floor, direction, capacity/load, riders' destinations, hall assignments,
  door state machine (`opening → open(dwell) → closing → closed`) with the same tick
  durations as `config.timing`, motion at `seconds_per_floor`, boarding
  `boarding_per_passenger` per person, `dwell_min`, full cars skipping pickups
  (people left behind re-press → re-decision), doors never open while moving.
* **Intra-car stop sequencing = LOOK/collective sweep** (serve stops in the current
  direction, reverse when none ahead) — *not* A*. The learned policy only decides **which
  car takes each new hall call** (and optionally parking in a later phase — leave a clean
  extension point, don't implement it).
* Fault & fire events minimal: `car_fault` (car stops, riders ejected to waiting, its hall
  assignments re-decided), `car_repair`, `fire_alarm` (cars recall to lobby, hall calls
  blocked), `fire_clear`. Invariants: no over-capacity, no motion with open doors.
* Passenger timing: record arrival, boarding and alighting ticks per passenger so the
  *same metrics* can be computed (`avg_wait`, `p95_wait`, `max_wait`, `avg_ride`,
  `long_wait_pct` with the 60 s threshold, `throughput`, `energy` using
  `energy_per_floor=0.5`, `energy_per_stop=1.0`).
* Hall-call lifecycle: a call exists while someone waits for (floor, direction); first
  press → decision request; if its assigned car fills/leaves without clearing the queue,
  the call is re-requested (same as the real `FloorAgent`); calls unserved for
  `escalate_after = 60` s raise `urgency` by 1 each further 60 s (urgency feeds the call token).
* Public car state: for each decision the twin must produce, in `FleetView`, exactly the
  `CarStatus` fields that `features.py` uses (floor, direction, load, space, available,
  out_of_service, fire_mode, door, door_blocked_ticks, n_assigned, n_car_calls, riders,
  planned_stops, plan_end_floor, plan_end_eta, park_target). `plan_end_eta` = LOOK-route
  time to finish all current commitments (use the same per-leg arithmetic as
  `ElevatorAgent._eta_to`: `travel = floors * seconds_per_floor`, `+ dwell` per stop where
  `dwell = door_open + door_close`).

What it may omit: A* planning, bids/messages, safety-rule engine, SA reassignment,
parking, minimax, the traffic *monitor* (the twin should still provide `policy.pattern`
and `policy.demand` to the encoder — use the ground-truth pattern and an EWMA of
per-floor arrivals with `alpha = 0.02`, mirroring `TrafficMonitorAgent`; document that the
real monitor *infers* the pattern, and offer a flag `infer_pattern=True` that reuses the
same rule-based classifier if you can port it cheaply — optional).

### 6.2 Implementation guidance

* Structure-of-arrays state for `B` parallel environments (`float32/int16/int8` arrays,
  shape `[B, cars, …]`, passengers in preallocated ring buffers sized from rate × duration).
* Hot loops in `@numba.njit(cache=True, fastmath=False)` kernels in `kernels.py`
  (`step_tick(state, rng_state, …)`, `next_decision(state, …)`). With
  `ELEVATOR_NO_NUMBA=1` the same functions run as plain Python (use a no-op decorator).
* RNG: numba-friendly `np.random.Generator`-free PCG/xorshift stored in state so a seed
  reproduces a run bit-for-bit; document it.
* **Semi-MDP stepping.** The unit of agent time is a *decision*, not a tick: `advance()`
  simulates ticks until at least one hall call needs assignment (or the episode ends), then
  hands back that call. Several calls in one tick are decided sequentially, each seeing the
  assignments just made (same as the real sequential Contract Net).
* Memory/time budgets: 64 envs × 1800 ticks in < 1 s on a laptop core after JIT warm-up.

### 6.3 Speed requirement (measured, reported)

`elevator learn bench-env` reports **ticks per second** and **decisions per second** for
(a) the real Mesa model with `nearest_car`, (b) the real model with `cnp_astar`, (c) the
twin single-env, (d) the twin with 64 envs (per core and total). Acceptance: **twin single-env
≥ 20× the real model's ticks/s on the 15×4 `morning_up_peak` scenario and ≥ 20× on `stress_scale`**
(real `cnp_astar` is the denominator, since that's what RL would otherwise have to run);
state the numba JIT warm-up time separately.

### 6.4 Fidelity requirement (measured, reported honestly)

`elevator learn validate-twin` compares the twin against the **real simulator running the
strategies that share the twin's intra-car logic**:

* Real `nearest_car` ↔ twin policy `nearest` and real `collective` ↔ twin policy `collective`
  (both use LOOK sequencing in the real sim, so the twin should match closely).
* Scenarios: all 9 YAMLs; seeds 1–10 each; 900 ticks (or the YAML duration if shorter);
  compare **means over seeds** of `avg_wait, p95_wait, long_wait_pct, throughput, energy`.
* **Pass** = mean `avg_wait` within **±10 %** (relative) on ≥ 7 of 9 scenarios for each of the
  two policies; the *ranking* `collective < nearest_car` (lower wait) agrees with the real
  sim in the same ≥ 7 of 9; throughput within ±10 %. Report every number in
  `docs/LEARNING_ENV_fidelity.md` as a table (scenario, real, twin, % error, pass/fail) —
  **including failures**. If it fails, first debug for real mismatches (compare
  per-passenger wait distributions, door timing, boarding order, call re-requests); tune
  constants only for genuine modelling differences and write down what you changed.
* Additional informational comparison (no pass/fail): twin `cost_greedy` (a scorer that
  mimics the teacher: wait ≈ `plan_end_eta + travel to call`, with weights
  `wait 1.0, ride 0.5, crowding 0.3, energy 0.2` approximated from public features) vs real
  `cnp_astar`. Report the gap.
* Add a pytest `test_twin_fidelity.py` marked `@pytest.mark.slow` that checks a reduced
  version (3 scenarios × 5 seeds × 600 ticks) with the ±15 % tolerance, so CI stays fast.

---

## 7. The Gym environment

`ElevatorDecisionEnv(gymnasium.Env)` over the twin:

* `reset(seed=None, options={"regime": name | dict})` → `obs, info`.
* `step(action)` → `obs, reward, terminated, truncated, info`; `action` = index of the chosen
  car (`Discrete(MAX_CARS)`), invalid choices (padding / ineligible) are **masked**: provide
  `info["action_mask"]` (bool `[MAX_CARS]`) and, if an invalid action is nonetheless sent,
  map it to the lowest-cost eligible car by `cost_greedy` and set `info["invalid_action"]=True`
  (never raise mid-training).
* `observation_space = Dict(call=Box(Kc), cars=Box(MAX_CARS, Kcar), glob=Box(Kg), mask=MultiBinary(MAX_CARS), eligible=MultiBinary(MAX_CARS))`.
  The `encode_decision` output is the observation, key for key.
* **Reward (team reward, shared by all cars)** between consecutive decisions, `Δt` ticks:
  `r = -( Σ_waiting_passengers Δt  +  0.5 · Σ_riding Δt ) / 100  -  0.02 · floors_moved_by_all_cars  -  0.5 · (new passengers crossing the 60 s wait threshold)`.
  Coefficients live in a dataclass `RewardConfig` (default as above) so Phase 5 can tune
  them; document why each term exists (wait-time is the objective; ride-time and energy are
  secondary; the threshold term fights starvation). Provide `info["episode_metrics"]` at the
  end with the *true* evaluation metrics (`avg_wait`, `p95_wait`, …) — evaluation uses these,
  never the shaped reward.
* `terminated=True` at the scenario duration after draining arrivals is **off**; episodes are
  fixed-horizon, `truncated=True` at `duration` (keep serving until the last decision of the
  horizon). Also support `info["teacher_action"]` when `teacher="cost_greedy"` is requested,
  for DAgger-style use later.
* **Determinism:** `reset(seed=s)` + the same action sequence ⇒ identical observations and
  returns. Test it.
* Must pass `gymnasium.utils.env_checker.check_env(env, skip_render_check=True)` and a
  100-random-episode soak with no NaN/inf in obs or reward.
* `VectorDecisionEnv(n_envs, ...)` — numba-batched stepping of B envs in lockstep with
  auto-reset, `reset()/step(actions[B])` returning stacked dict-of-arrays, plus
  `info["final_metrics"]` for finished episodes. Envs have different numbers of cars/floors
  (domain randomisation) → padded to `MAX_CARS` with `mask`. This is the PPO workhorse; it
  must reach **≥ 150 000 decisions/s total** on 4 cores or document the achieved rate and the
  bottleneck honestly.

---

## 8. Regimes, baselines and the evaluation harness

`configs/learning/regimes.yaml` defines:

* `training:` — the randomisation ranges from §5 (shared by the recorder and the env).
* `eval_regimes:` exactly **four named, fixed regimes** on the standard 15-floor × 4-car,
  capacity-10 building, 900 ticks, used for every headline claim from now on:

  | name | pattern / rate | note |
  | --- | --- | --- |
  | `up_peak` | `up_peak`, 0.20/s | morning rush |
  | `down_peak` | `down_peak`, 0.20/s | the regime where the current "full" strategy loses to `collective` — keep it, it's the honest test |
  | `two_way` | `two_way`, 0.18/s | lunch |
  | `interfloor` | `interfloor`, 0.10/s | light mixed |

  plus `scale:` regimes (not counted in the four): `large_40x8` (40 floors, 8 cars, rate 0.12/s,
  capacity 12) and `small_6x2`.
* Evaluation seeds: `val` = 8000–8049, `test` = 8500–8599 (never used for training or tuning).

`evaluate.py`: `evaluate(policy, regimes, backend, seeds) -> pandas.DataFrame` with columns
`regime, backend, policy, seed, avg_wait, p95_wait, max_wait, long_wait_pct, throughput, energy, decisions, wall_s`;
and `summarise(df)` giving mean ± 95 % bootstrap CI (1 000 resamples) per regime/policy, and a
paired comparison (`policy A − policy B` on identical seeds, CI of the difference, win-rate).

A **policy** is any object with
```python
class Policy(Protocol):
    name: str
    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:   # obs batched [B, ...]; returns action per env
```
Built-in policies in `twin/policies.py`, all operating **only on the encoded observation**:
`nearest` (smallest `dist_to_call` among eligible), `collective` (prefer `call_on_route` &
`heading_to_call`, then nearest), `cost_greedy` (the teacher-mimicking scorer above),
`random_eligible`. Backends: `"twin"` (fast, B-batched) and `"real"` (drives the real Mesa
model: for `real` the harness runs the actual strategies by name `nearest_car, collective,
cnp_astar, full`; running an *arbitrary* `Policy` inside the real sim is Phase 4/6 work —
leave a clear `NotImplementedError("Phase 4")` with a docstring describing the hook).

`elevator learn baselines` prints and saves (`reports/learning_baselines.csv/.md`) the four
regimes × {nearest, collective, cost_greedy, random} on the twin **and** the four real
strategies on the real sim, validation split, 20 seeds. This table is the benchmark LiftZero
must later beat; include the paired-difference columns.

---

## 9. Public API (what Phase 4/5/6 will import — keep it stable)

```python
from elevator_mas.learning import (
    FEATURE_VERSION, MAX_CARS, KC, KCAR, KG, FEATURE_NAMES,   # schema.py
    DecisionContext, Encoded, encode_decision, encode_batch,    # features.py / view.py
    DecisionRecorder, load_shards, ExpertDataset,               # recorder.py / dataset.py
    ElevatorDecisionEnv, VectorDecisionEnv, RewardConfig,       # env.py
    Policy, evaluate, summarise, load_regimes,                  # evaluate.py / regimes.py
)
```
`ExpertDataset` loads shards lazily (memory-map where possible), exposes `train/val/test`
views with `__len__`, `batch(indices) -> dict[str, np.ndarray]`, `iter_batches(batch_size, shuffle, seed)`,
`hard_mask()` and `stats()`.

---

## 10. Testing (graded: "demo quality & testing")

Run with `uv sync --group learn && uv run pytest`. Target **≥ 70 new tests**, ≥ 85 % statement
coverage of `elevator_mas.learning`. Required, at minimum:

* **Schema:** dims/names consistent (`len(FEATURE_NAMES['call']) == KC` …), version hash stable.
* **Features:** shapes, dtypes, ranges, no NaN/inf (property-based with `hypothesis` is welcome);
  padding rows are exactly zero and masked; permutation test; `call_on_route` /
  `heading_to_call` truth tables on hand-built cases; real-vs-twin **parity** test (bit-identical).
* **Hook neutrality:** metrics identical with/without hook (§2).
* **Recorder:** on a seeded run, #records == #`AuctionRound`s; recorded `winner` equals
  `AuctionRound.winner`; `argmin(teacher_cost over eligible)` equals the winner (ties by car id) in
  100 % of records; refused ⇔ `~eligible` ≥ 99.9 %; deterministic across two runs; resumable;
  splits are disjoint by run id; test seeds never in train.
* **Twin invariants (property-style, many seeds & randomised configs):** passenger
  conservation (`arrived = waiting + riding + delivered`), load ≤ capacity, no motion with open
  doors, floors in range, every passenger eventually delivered after a drain, determinism
  by seed, fault ejects riders and re-decides calls, fire recall blocks calls then restores.
* **Twin fidelity:** the reduced slow test of §6.4.
* **Env:** `check_env`, action masking & invalid-action fallback, determinism, soak, reward
  decomposition test (hand-computed reward over a scripted 3-decision episode), vector env
  equals B independent single envs (same seeds ⇒ same trajectories).
* **Evaluate:** paired comparison math on synthetic data; bootstrap CI contains the true mean in
  ≥ 90 % of 100 synthetic trials.
* **CLI smoke tests** with Typer's `CliRunner` for every `learn` sub-command (tiny sizes).
* Regression: all 201 pre-existing tests still pass.

---

## 11. Acceptance checklist (evidence required in the final report)

- [ ] `uv sync` (no group) → `uv run pytest` green (201 + the tests that don't need the group);
      `uv sync --group learn` → full suite green; `ruff check` + `ruff format --check` clean.
- [ ] Hook neutrality proven; engine diff limited to the hook + `DecisionEvent`.
- [ ] Feature schema v1 implemented exactly as §4; parity test (real vs twin) passes.
- [ ] `elevator learn record` produced a ≥ 50 000-decision dataset here (or ≥ 500 000 if time allows),
      with `dataset-info` output pasted in `docs/LEARNING_ENV.md`; exact command for the full 500 k run documented with a measured decisions/s rate.
- [ ] Twin speed: ≥ 20× real `cnp_astar` on both `morning_up_peak` and `stress_scale` (numbers in the report), vector env rate reported.
- [ ] Twin fidelity: report table committed; pass criteria of §6.4 met **or** failure analysed honestly with the residual error stated.
- [ ] `ElevatorDecisionEnv` passes `check_env`; `VectorDecisionEnv` equals B single envs.
- [ ] `regimes.yaml` + baselines table (`reports/learning_baselines.md`) with CIs and paired differences, twin and real.
- [ ] `docs/LEARNING_ENV.md` (design, schema table, dataset format, twin scope + known simplifications, reward rationale, how to reproduce) and `docs/LEARNING_ENV_fidelity.md`.
- [ ] `.gitignore` covers `data/`; only the small fixture is committed.
- [ ] Branch `phase-3-learning-env`, logical commits, not pushed.

## 12. Final report format

What was built per module; test counts + coverage; the recorded dataset statistics (incl. the
hard-decision fraction and the trivial-decision fraction); speed table; fidelity table with
verdict; the baseline table; any deviation from this prompt with the reason; known limitations
of the twin (be specific); and the exact commands for me to verify, ending with:

```
git checkout phase-3-learning-env
uv sync --group learn
uv run pytest
uv run elevator learn dataset-info --path data/expert
uv run elevator learn bench-env
uv run elevator learn validate-twin
uv run elevator learn baselines --backend twin --split val
```
