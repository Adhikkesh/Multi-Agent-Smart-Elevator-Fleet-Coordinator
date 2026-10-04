# PHASE 6 — Look-ahead and explainability: PUCT-MCTS arbitration, the "Brain" API, and the Brain panel in the React UI

> **Prerequisites:** Phase 2 (React UI, `frontend/`, `BrainFrame` placeholder, `/api/*`), Phase 3
> (twin, env, features), Phase 4 (`LiftRuntime`, ONNX, learned bidder, `liftzero_bc`) and Phase 5
> (`liftzero_ppo`, `V_pub` value head, `models/liftzero_ppo_v1.onnx`) are all merged into `v2`.
> The user gives you the base commit. **The merged code wins over this prompt**; list every adaptation.

Paste this whole file into Antigravity (Planning mode, strongest model).

---

## 0. Context and goal

LiftZero today is a *reflex* bidder: each car runs a neural network on the public board and bids
its score. It never "thinks ahead". This phase adds **deliberation** — an AlphaZero-style
**PUCT Monte-Carlo tree search** that, for *contested* hall-call decisions, simulates the next
minute of the building under each candidate assignment before the dispatcher awards the call —
and, just as important for the demo, makes the whole learned system **visible and explainable**
in the UI (the **Brain panel**).

Deliverables:

1. a **world model** for search: build a twin state from the *public* board (partial observability handled by sampled hidden information),
2. a **PUCT MCTS** with the network as prior and the public value head as leaf evaluator, with common random numbers and a hard time budget (≤ 50 ms/decision),
3. its **integration** into the multi-agent system as dispatcher-side *arbitration over the cars' learned bids* — strategy `liftzero_mcts`,
4. the **Brain API** (`/api/brain*`) and the `snapshot.brain` stream field (strict JSON),
5. the **Brain panel** and related UI work in `frontend/` (React), incl. runtime model switching, search controls, saliency "Why?" view, and a Theory-page LiftZero section,
6. anytime/latency–quality analysis, tests (Python, Vitest, Playwright), docs.

**Acceptance headline:** MCTS decisions ≤ 50 ms (hard cap), deterministic given the seed, zero
safety violations, and an honest measurement of whether look-ahead improves on `liftzero_ppo`.

---

## 1. Working agreement

* Branch `phase-6-mcts-brain` from `v2`. Small commits. **No push, no `main`.**
* Python 3.12 / `uv`; ruff clean (100 cols); `pnpm typecheck/lint/test/e2e` green in `frontend/`. All earlier tests green; golden strategy metrics unchanged (Phase 4's `tests/golden/strategy_metrics.json`).
* No torch at runtime: search uses `LiftRuntime` (ONNX) and the numba twin. **Dependency change (required):** the search runs inside the live server, so `numpy` and `numba` move from the Phase 3 `learn` group into the **base** `dependencies` (a fresh `uv sync` on the demo laptop must run `liftzero_mcts` with no extra groups; torch/gymnasium stay out of base). Numba JIT cost must never hit the demo: use `cache=True`, and add a `warm_up()` that compiles the twin and search kernels at server start-up (in the FastAPI lifespan, in a thread, before the first request is served; `/api/health` reports `warming: true/false`), plus a test that the first MCTS decision after `warm_up()` meets the 50 ms budget. Keep the `ELEVATOR_NO_NUMBA=1` pure-Python fallback working (search then silently degrades to `sims=0`/network argmin and the Brain panel says so).
* Search lives in `src/elevator_mas/learning/lift/search/`; engine changes limited to the dispatcher award step, the strategy registry, and `api/`. Deterministic: seeded from `(model.seed_value, tick, conversation_id)`.
* Write a plan first. The code wins over this prompt where they conflict.

---

## 2. The architecture decision (write it into the docs — viva material)

*Who deliberates, and with what information?* The multi-agent rules from Phase 1 still hold: no agent reads another's private state.

* **Cars** (learned bidders) still bid: each computes its network score from the public board → `PROPOSE`.
* **Dispatcher** (auctioneer) is the only agent that *sees all bids* and *owns the global picture* (open calls, assignments, waiting counts it was told in `REQUEST`s, the public board). It runs the look-ahead **as a tie-breaker/arbiter over the bids** — it never recomputes a car's bid, and it may only choose among cars that *proposed* (not refused).
* The search sees **only public information** plus what the dispatcher itself has been told. Hidden information (who is waiting where and where they want to go) is **sampled**, not peeked: this is *determinisation* / **perfect-information Monte-Carlo (PIMC)** over a belief — a direct, explainable link to *partially observable* environments in AIMA.

So the hybrid is: *decentralised learned bidding (reflex)* + *centralised model-based look-ahead (deliberation)* — exactly the AIMA contrast between reflex and model-based planning agents, in one system.

---

## 3. The world model for search — `search/worldmodel.py`

### 3.1 Building a twin state from the public view

`TwinState.from_public(view: FleetView, pending: list[PendingCall], policy: FleetPolicy, building: BuildingConfig, timing: TimingConfig, rng_seed: int) -> TwinState`.

(Phase 3's twin must expose, or you must add additively: `clone()`, `apply_action(car_id)`, `advance(max_ticks) -> (reward, done, next_decision | None)`, `from_public(...)`; implement them in `twin/` with numba kernels if missing, with the same semantics as the training env so the value head's training distribution matches search.)

Inputs available to the dispatcher (all public or dispatcher-owned):
* per car, from `CarStatus`: floor, direction, load, door state, `car_calls` (destination buttons — known), `assigned_calls`, `plan_end_*`, availability, park target;
* open and recently-assigned hall calls with the **waiting counts the floors reported in their REQUESTs** (the dispatcher already stores them in `pending_calls`/`unassigned` records; keep a small `last_waiting[(floor,dir)]` map updated from each REQUEST/INFORM it sees — additive, in the dispatcher);
* `policy.pattern`, `policy.demand` (EWMA), the building and timing configs.

Hidden, so **sampled per simulation** (determinisation):
1. each waiting person's destination ~ the OD profile of `policy.pattern` conditioned on origin/direction (reuse `traffic/profiles`),
2. riders already aboard: the car's `riders` spread over its `car_calls` (≥ 1 rider per distinct button, remainder uniformly),
3. future arrivals over the search horizon ~ Poisson with the EWMA rates `policy.demand` (use `max(rate, small)`), destinations from the pattern profile.

### 3.2 Common random numbers (CRN) — required

Simulation index `s = 0..S−1` fixes **one** sampled hidden world and **one** future-arrival stream (seeded by `hash(root_seed, s)`). *Every* root action is evaluated against the same `s`-th world, so differences between actions are not drowned in sampling noise. A test must demonstrate lower variance of the action-value difference with CRN than without (paired t-test or variance ratio on 200 trials).

### 3.3 Fidelity of the world model (measured, reported)

Take 300 mid-run states from real-sim runs (`full` and `liftzero_ppo`, all four regimes). For each state, build the twin via `from_public`, apply the *real* decision, run 45 ticks, and compare with what the real sim did over the same 45 ticks: number delivered, number still waiting, mean wait accrued, floors travelled. Report MAE/MAPE and correlation, plus **ranking accuracy**: for decisions with ≥ 2 candidates, does the twin's predicted return rank the *teacher's best* alternative's outcome the same way real counterfactual re-runs do? (Real counterfactuals: fork the real model with `copy.deepcopy` at the decision, force each candidate, run 45 ticks. Do this for 200 decisions; it is slow — parallelise.) Put the result in `docs/lift/PHASE6_SEARCH.md`. This is the honest answer to "is the search model good enough to help?".

---

## 4. The search — `search/mcts.py`

### 4.1 Tree

* **Node** = a decision point: the twin state at the moment a hall call needs assignment (+ the call), with cached encoded tokens, network `priors[a]` over eligible cars, value `V_pub(s)`.
* **Edge** = (state, action = car). Stats: `N, W (sum of returns), Q = W/N, P` (prior).
* **Expansion** of an edge: `s' = clone(s); apply_action(a); advance(until next decision or T_max)`; edge reward `r` = the Phase 3 team reward accumulated over the interval; the child node is created lazily with `priors` and `V_pub` from **one batched ONNX call** per leaf.
* **Horizon**: depth ≤ `D = 3` decisions **or** `T_max = 60` simulated ticks since the root, whichever first; at the horizon the leaf value is `V_pub(s_leaf)` (or 0 if the episode ended). Value is a discounted return with the same SMDP discount `γ_eff = 0.99^(Δt/5)` used in Phase 5.
* **Selection (PUCT)**: `a* = argmax_a [ Q̄(s,a) + c_puct · P(s,a) · √(Σ_b N(s,b)) / (1 + N(s,a)) ]`, `c_puct = 1.25`; `Q̄` = min-max-normalised Q over the tree (as in MuZero) to [0,1]; unvisited edges use `Q̄ = parent's mean Q` (not 0). **No Dirichlet noise** (we want reproducible deterministic decisions); document that.
* **Root candidates**: only cars that sent `PROPOSE` (never refused); further restricted to the `top_k = 3` by network score, always including the network's argmin and, if different, the classical nearest car by distance (a cheap safety net; configurable).
* **When to search** (compute saver, measured in the ablation): search only if `|candidates| ≥ 2` **and** the score margin is small: `score_2 − score_1 < τ_margin · max(1, |score_1|)` with `τ_margin = 0.5` (tuned on VAL; report the tuned value and the fraction of decisions searched). Otherwise award the network's argmin as before.
* **Budget**: `sims = 32` default (config 0–256), plus a **hard wall-clock cap** `time_budget_ms = 50` checked after every simulation (the search returns the best result so far — *anytime*). The cap includes world-model construction. Never exceed the cap by more than 5 ms except for the ONNX call in flight.
* **Final move**: root action with the highest **visit count**; ties → higher `Q`, then lower network score, then lower car id. If the search errors or runs 0 simulations, fall back to the network argmin (log a counter `search_fallbacks` exposed in `/api/brain`).
* **Batching (optional, +speed)**: virtual loss with leaf batches of 4 to amortise ONNX calls — implement only if profiling shows ONNX dominates; keep the simple sequential version as the reference and prove equal results on a fixed seed with batch=1.

### 4.2 Search outputs (for the UI)

`SearchResult(chosen, visits[a], q[a], prior[a], root_value, sims_done, depth_max, time_ms, world_model_ms, nn_ms, overridden: bool, leaf_eval: "value"|"rollout")`.

### 4.3 Two leaf evaluators (ablation material, both required)

* `value` (default): `V_pub` from the Phase 5 network.
* `rollout`: complete the horizon with the `cost_greedy` policy on the twin (no network value). Works with the Phase 4 (BC) model, which has no trained value. Strategy `liftzero_bc_mcts` uses `rollout`.

---

## 5. Integration into the multi-agent system

### 5.1 Strategies (additive registry entries)

| name | bidder | arbitration | model |
| --- | --- | --- | --- |
| `liftzero_bc_mcts` | learned | PUCT, leaf = rollout | `liftzero_bc_v1.onnx` |
| `liftzero_mcts` | learned | PUCT, leaf = value | `liftzero_ppo_v1.onnx` |

(Everything else — A* routing inside cars, SA reassignment, parking, adaptive weights — identical to `full`, as in Phase 4/5.) `DispatchStrategy` gets `arbiter: str = "min_bid" | "mcts"` and `search: dict` (sims, top_k, time_budget_ms, tau_margin, leaf).

### 5.2 Dispatcher change

In `DispatcherAgent.award()`, after collecting `PROPOSE`s: if `strategy.arbiter == "mcts"` → `self.arbiter.choose(call, bids, board_view)` returns `(winner_car_id, SearchResult|None)`; the rest of `award()` (ACCEPT/REJECT, `AuctionRound`, `unassigned` handling) is unchanged. The **decision trace** is extended: e.g.
`"car 2 wins 6D (net 3.1 vs car 1 4.0); look-ahead (32 sims, 18 ms) preferred car 1 → OVERRIDE: value +0.9 vs +0.3"`.
The arbiter is passed only the board, the bids and the dispatcher's own records — enforce with a stub-based test (no model-object access).

### 5.3 Brain recorder (ring buffers)

`BrainLog` on the model: last 200 decisions, each: tick, conversation id, call, candidates (car id, eligibility/refusal, net score, aux breakdown, attention weight, teacher cost if shadow on, prior/visits/Q if searched), chosen, net choice, teacher choice, `SearchResult`, latency split (net / search / total), plus the **encoded tensors** for the last 50 (for saliency). Running stats: decisions, % agreement with the net argmin, % agreement with shadow teacher, % overridden by search, mean/p95/p99 latencies, fallbacks. Memory bounded; reset on `/api/reset`.

### 5.4 Brain API (strict JSON, additive)

```
GET  /api/brain                → { available: bool, strategies: [{name,label,available,reason?}], active: {...model card subset...},
                                  stats: {...running stats...}, config: {sims, top_k, time_budget_ms, tau_margin, leaf, shadow_teacher} }
GET  /api/brain/decisions?limit=50  → { decisions: BrainDecision[] }
POST /api/brain/config   { sims?, top_k?, time_budget_ms?, tau_margin?, shadow_teacher? } → config  (applies to the running model; validated ranges; 400 otherwise)
POST /api/brain/explain  { conversation_id }  → { groups: [{group, delta_score_chosen, delta_score_runner_up, flips_decision: bool}], method: "occlusion" }
GET  /api/brain/model-card?name=liftzero_ppo_v1 → model card JSON
```
`snapshot.brain` (in every WebSocket frame, `null` when the active strategy is not learned):
```ts
interface BrainFrame {
  strategy: string; mode: 'classical'|'imitation'|'ppo'|'ppo+mcts';
  model: { name: string; version: string; params: number; feature_version: number; sha?: string };
  latency_ms: { net: number; search: number; total: number; p95_total: number };
  stats: { decisions: number; agree_net_pct: number; agree_teacher_pct: number|null; overridden_pct: number; fallbacks: number; searched_pct: number };
  decision: BrainDecision | null;            // the latest
}
interface BrainDecision {
  tick: number; conversation_id: string; call: { floor: number; direction: 'UP'|'DOWN' };
  candidates: { car_id: number; eligible: boolean; refused_reason?: string; score: number|null; teacher_cost?: number|null;
                aux?: { wait: number; ride: number; crowding: number; energy: number }; attention: number;
                prior?: number; visits?: number; q?: number }[];
  chosen: number; net_choice: number|null; teacher_choice?: number|null; overridden: boolean;
  search?: { sims: number; depth_max: number; time_ms: number; root_value: number; leaf: 'value'|'rollout'; fallback: boolean };
  reason: string;
}
```
Keep the previous Phase 2 `BrainFrame` placeholder compatible (this is the filled-in version). All floats finite or `null`.

### 5.5 Runtime model switching

`POST /api/reset` already accepts `strategy`; make the learned strategies selectable there (and listed with availability in `/api/meta`). Switching **resets** the simulation with the same scenario+seed (the UI states this) so A/B comparisons are same-seed.

---

## 6. Explainability: saliency

`explain(decision)`: **feature-group occlusion** on the stored encoded tensors. Groups: `position` (floor, signed/abs distance, park distance), `motion` (direction, heading_to_call, call_on_route), `load` (load, space, riders), `doors` (door one-hots, blocked), `plan` (n_assigned, n_car_calls, plan_stops, plan_end_floor, plan_end_eta, has_same_call), `call` (call token features), `pattern` (pattern one-hot + demand share), `global` (global token). For each group: replace that group's features of **all** car tokens (or the relevant token) by the **dataset mean** (store per-feature means in the model card) — *not* zero, which is out-of-distribution — re-run ONNX, report Δscore for the chosen car and for the runner-up, and whether the decision flips. Cheap: ≤ 10 ONNX calls. Document limits (occlusion ≠ causality; group interactions).

Also expose the last-layer **attention of the call token over cars** (`attn` output) as `attention` per candidate (head-averaged), plainly labelled "where the model looks", with a caveat that attention is not an explanation by itself.

---

## 7. The Brain panel and UI work (in `frontend/`, Phase 2 conventions)

Replace the placeholder with a real panel; keep design tokens, dark/light, a11y, reduced motion.

1. **Header**: model badge (name, version, params "0.27 M", feature v1), mode chip (Classical / Imitation / PPO / PPO + MCTS), a **strategy switcher** (segmented control) that calls `POST /api/reset {strategy}` with the current scenario+seed after a confirm toast ("restarts the run"), and a latency pill (`net 0.4 ms · search 18 ms`) turning amber > 35 ms, red > 50 ms.
2. **Decision card** for the latest decision: "Call 6↓ — who goes?" with, per candidate car, a horizontal **score bar** (network cost, lower = better; refused cars hatched with the reason), a **teacher marker** when shadow is on (◆ at the teacher's cost, with ✓/✗ agree badge), the **aux breakdown** as a mini stacked bar (wait/ride/crowding/energy using the same colours as the Auction panel), the **attention strip** (small intensity bar; hovering a candidate highlights its car in the building view and vice-versa), and for searched decisions the **visit-count bars with Q values** next to the score bars, a crown on the chosen car, and an **"OVERRIDE" badge** when search ≠ network (with the value gap).
3. **Search inspector** (collapsible): sims done, depth, root value, time split (world model / NN / tree), leaf evaluator; a tiny tree diagram (root → candidate edges → best child) drawn in SVG from `visits/q`. If the backend provides only root-level stats, draw the root fan-out only (don't fake deeper levels).
4. **"Why?" popover**: calls `POST /api/brain/explain` and shows the group-occlusion bars (chosen vs runner-up), flipping groups marked, with the one-line caveat.
5. **History strip**: last 30 decisions as coloured pips (agree-with-teacher / disagree / overridden-by-search) — click to load that decision in the card (from `GET /api/brain/decisions`).
6. **Stats row**: decisions, agreement with teacher %, overridden %, fallbacks, p95 latency, with sparklines from the stream.
7. **Search controls**: sims slider (0–128; 0 disables search), top-k, time budget, margin τ, shadow-teacher toggle → `POST /api/brain/config`; show applied values.
8. **Model card dialog**: renders the JSON card (training data, metrics, limitations).
9. **Compare Mode & Experiments**: add the LiftZero strategies to the strategy pickers; labelled with a "learned" tag; unavailable ones disabled with the reason.
10. **Theory page — new "LiftZero" section**: a clean architecture figure (tokens → encoder → heads), the training pipeline (teacher → imitation → DAgger → PPO → MCTS) as a diagram, numbers pulled live from the model card (`/api/brain/model-card`), "reflex vs deliberative agent", "what is learned vs what is classical" table, limitations.
11. **Story mode**: add two beats after "Why car 2?": *"The network's view: scores, attention"* (switches to `liftzero_ppo` by reset with the same seed, waits for the first contested decision) and *"Looking ahead: MCTS overrides the network"* (switches to `liftzero_mcts`, waits for the first `overridden` decision, up to 30 s, with a Skip). Update the closing beat to show the **A/B result** from `/api/run` (`full` vs `liftzero_ppo` vs `liftzero_mcts`, same seed). Keep the old beats' robustness rules (wait on conditions, not ticks).
12. **Performance**: Brain updates must not re-render the building view; memoise by `decision.conversation_id`.

Tests: Vitest for the Brain store slice/selectors/formatters and components (`DecisionCard` with refused/null scores, override badge, teacher markers; `SearchInspector` with root-only data; `ExplainPopover`); Playwright specs: (1) switch to `liftzero_ppo` → Brain card appears and updates while playing; (2) switch to `liftzero_mcts` → latency pill present and ≤ red threshold, an OVERRIDE eventually appears on `stress`-like busy scenario or the test falls back to asserting `searched_pct > 0`; (3) "Why?" returns groups; (4) search slider to 0 sets `sims=0` in `/api/brain`; (5) model card dialog opens; (6) no console errors; (7) axe: no serious violations. Rebuild and commit `src/elevator_mas/web/dist`.

---

## 8. Analysis deliverables

* **Anytime curve**: sims ∈ {0, 4, 8, 16, 32, 64, 128} × the four regimes × 50 val seeds (real simulator): avg wait (paired % vs `sims=0`), decision latency (median/p95), fraction searched. Plot wait-vs-latency Pareto; choose the shipped default (32 sims, 50 ms) from the curve, not from taste.
* **Where does search help?** Break down overrides: fraction of decisions overridden, and the *realised* benefit of overrides (counterfactual re-runs on 300 overridden decisions: fork the real model, run both choices 60 ticks, compare team reward); report mean gain and the fraction of overrides that were harmful.
* **Ablations (leaf evaluator, top-k, CRN on/off, margin gate on/off, horizon D=1/2/3)** on VAL, 30 seeds.
* **Final comparison on TEST seeds** (once): `liftzero_ppo` vs `liftzero_mcts` vs `liftzero_bc_mcts` vs `full`, 100 paired seeds × 4 regimes, with the Phase 5 statistics (paired CI, Wilcoxon, Holm).
* Honest verdict: if MCTS doesn't help (the PPO policy is already near the twin's optimum, or the world model is too coarse), say so with the evidence and keep `liftzero_ppo` as the headline strategy; the search remains a documented, working extension.

---

## 9. Python tests (≥ 55 new, ≥ 85 % coverage of `learning.lift.search` + new API code)

* PUCT formula & min-max normalisation: hand-computed numbers; unvisited-edge `Q̄` rule; visit-count final move and tie-breaks.
* Backup: discounted return accumulation on a hand-built 3-step tree; SMDP discount with `Δt = 0`.
* Anytime/budget: with a mocked slow evaluator the search stops inside `budget + 5 ms`; `sims=0` ⇒ network argmin; error injection ⇒ fallback + counter.
* Determinism: same seed ⇒ identical `SearchResult` (visits/q/choice); different tick ⇒ may differ.
* CRN variance test (§3.2); batching equivalence (if implemented).
* Candidate hygiene: never selects a refused/ineligible car; always includes the network argmin; `top_k` respected.
* World model: `from_public` conserves counts (riders, waiting, car_calls), respects capacity/doors/eligibility; hidden-info sampling distribution matches the OD profile (χ² test, p > 0.01); `clone()` independence (mutating a clone does not change the original).
* Board-only guard: the arbiter fed a stub board/bids works; any access to a model/car/floor object raises.
* Dispatcher integration: `ACCEPT/REJECT` messages unchanged in structure; trace string contains OVERRIDE when overridden; `AuctionRound.reason` present.
* **Safety matrix**: `liftzero_bc_mcts` and `liftzero_mcts` × 9 scenarios × 3 seeds ⇒ `violations()==[]` every tick, all delivered after drain, ineligible never wins.
* API: `/api/brain*` schemas, validation errors (400), strict JSON, `snapshot.brain` null for classical strategies, populated for learned; config changes take effect; explain output sums/flip logic on a synthetic decision; reset-with-strategy keeps scenario+seed.
* Golden: classical strategies' metrics unchanged.

---

## 10. Documentation

* `docs/lift/PHASE6_SEARCH.md`: architecture decision (§2), world model & determinisation, PUCT details, CRN, budget/anytime, integration into the agent protocol, the measured fidelity (§3.3), the anytime curve, ablations, final comparison, honest verdict, limitations.
* Update `docs/DESIGN.md` (agent-type discussion: reflex bidder + model-based deliberation; PEAS update for the dispatcher), `docs/TESTING.md`, `docs/UI.md` (Brain panel tour with screenshots in `docs/img/ui/`), `docs/DEMO_SCRIPT.md` (the new beats), `README.md`.
* `scripts/reproduce_phase6.sh`: smoke pipeline (≤ 15 min) = fidelity check on 50 states + 4-regime anytime curve with 5 seeds.

---

## 11. Pitfalls

1. **Search must not change who is *eligible*.** Refusals are classical; MCTS only reorders among proposers.
2. Calls are auctioned sequentially; the twin you build at the root must already include awards made earlier **in the same tick** (they are on the board by then).
3. `waiting` counts are *the dispatcher's last report*, possibly stale; clamp and add a small Poisson jitter per simulation rather than trusting them exactly.
4. Keep the world model **cheap**: `clone()` should be a few `ndarray.copy()` calls; do not deep-copy Python objects in the hot path.
5. Wall-clock budgets make runs machine-dependent: the **simulation-time** behaviour must stay deterministic given sims; use `time_budget_ms` only as a safety cap, and record `sims_done` so results are reproducible by fixing `sims` (CI uses `time_budget_ms = ∞`).
6. At 50× playback the server may fall behind with MCTS; the UI shows latency and the session already drops frames — don't "fix" this by lowering the budget silently.
7. Attention maps and occlusion are *indicative*; say so in the UI and docs (an examiner will ask).
8. Don't call `/api/brain/explain` per frame; only on demand.

---

## 12. Acceptance checklist

- [ ] World model fidelity study (§3.3) done and reported (MAE/MAPE, ranking accuracy).
- [ ] PUCT-MCTS with CRN, two leaf evaluators, hard 50 ms budget (p99 ≤ 60 ms at N=8 on the user's machine), deterministic; unit tests green.
- [ ] `liftzero_bc_mcts`, `liftzero_mcts` registered, safe on the 9-scenario matrix; classical goldens unchanged.
- [ ] Brain API + `snapshot.brain` strict JSON; saliency endpoint; config endpoint validated.
- [ ] Brain panel complete (items 1–12 of §7), Vitest + Playwright + axe green, dist rebuilt and committed; docs with screenshots.
- [ ] Anytime curve, override-benefit study, ablations and the single final TEST comparison with paired statistics; honest verdict written.
- [ ] Docs/README/DESIGN/TESTING/DEMO_SCRIPT updated; `scripts/reproduce_phase6.sh` works.
- [ ] Branch `phase-6-mcts-brain`, logical commits, not pushed.

## 13. Final report format

Results first (fidelity, anytime curve, final table with CIs, verdict), then what was built (backend, UI), test counts & coverage,
latency numbers, deviations and why, limitations, verification commands ending with:

```
git checkout phase-6-mcts-brain
uv sync
uv run pytest
uv run elevator serve --scenario demo_story      # open http://127.0.0.1:8000 → Mission Control → Brain panel → strategy: LiftZero + MCTS
cd frontend && pnpm i && pnpm test && pnpm e2e
```
