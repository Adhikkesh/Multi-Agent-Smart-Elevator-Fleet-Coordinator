# LiftZero — Viva Study Material

*Team: Adhikkesh (CB.SC.U4CSE23101) · Sisr Reddy (CB.SC.U4CSE23129) · Kavin Karthic
(CB.SC.U4CSE23161) · Akash (CB.SC.U4CSE23162). FOAI, Semester 7.*

Read this once end to end, then re-read your own sections. Every number here comes from our
own runs (sources in `docs/lift/` and `reports/`). If an examiner asks about something we
did **not** build, say so plainly — see §9.

---

## 1. The problem in one minute

**Elevator Group Control Problem (EGCP).** A building has *N* cars. People press hall buttons
at random times; each hall call (floor, direction) must be assigned **online** to one car.
We do not know future arrivals or where waiting people want to go.

**Why it is hard**
* **Combinatorial**: with *k* open calls and *N* cars there are *N^k* assignments
  (4 cars, 20 calls ≈ 1.1 × 10¹²).
* **Sequential and coupled**: an assignment changes a car's route, which changes the cost of
  every later call.
* **Stochastic and partially observable**: arrivals are Poisson; destinations are hidden
  until people board.
* **Multi-objective**: average wait, worst-case (p95) wait, ride time, energy, fairness.
* **Safety-critical**: never move with doors open, never exceed capacity, fire recall.

**Objective we optimise** (per run): minimise average wait (AWT) while keeping p95 wait and
long-wait % (waits > 60 s) low and energy reasonable.
Energy = floors travelled + 2 · stops + reversals.

### Why classic heuristics fail (our measured numbers, morning up-peak, 100 seeds)

| strategy | how it assigns | avg wait |
| --- | --- | --- |
| Nearest car | closest free car | 56.2 s |
| Collective (LOOK) | car already sweeping towards the call | 47.1 s |
| CNP + A\* | auction; each car bids its A\* marginal cost | 42.9 s |
| Full (CNP + A\* + SA + parking + adaptive weights) | + global re-optimisation | **29.9 s** |

* *Nearest car* ignores what a car is already committed to → **bunching** (several cars chase
  the same area) and **starvation** of upper floors.
* *Collective* avoids reversals but has no notion of future cost.
* Fixed rules cannot adapt when traffic changes (up-peak → lunch → down-peak).

---

## 2. AIMA foundations you must be able to state

**Agent** = perceives through sensors, acts through actuators. **PEAS** = Performance,
Environment, Actuators, Sensors.

| agent | AIMA type | P / E / A / S (short) |
| --- | --- | --- |
| ElevatorAgent | utility-based (and model-based) | P: low marginal cost; E: shaft, riders; A: move, doors, PROPOSE; S: floor sensor, inbox, board |
| DispatcherAgent | goal/utility-based coordinator | P: all calls served cheaply; A: CFP, ACCEPT/REJECT, reassignment; S: bids, requests |
| FloorAgent | simple reflex with state | senses waiting people, raises hall calls |
| TrafficMonitorAgent | **learning agent** | estimates demand (EWMA), detects pattern, retunes weights W1–W4 |
| SafetyAgent | knowledge-based (forward chaining) | fires safety rules (fire recall, overload) |
| PassengerAgent | simple reflex | board, ride, alight |

**Environment classification**: partially observable, stochastic, sequential, dynamic,
discrete-time (1 tick = 1 s), multi-agent (cooperative).

**Learning agent components** (for LiftZero): performance element = the network that bids;
critic = imitation loss vs the A\* teacher (Phase 4) or the team reward (Phase 5); learning
element = the trainer; problem generator = domain randomisation and DAgger rollouts.

---

## 3. Multi-agent system: FIPA-ACL and the Contract Net Protocol

Agents never call each other's methods. They exchange **FIPA-ACL messages** (performative +
sender + receiver + conversation id + content) through a message bus, and read a public
**status board** (blackboard) where each car publishes its own status.

**One Contract Net round (per hall call):**

```
Dispatcher ──CFP(call, urgency, waiting)──────────▶ all cars
car i      ──PROPOSE(bid_i)  or  REFUSE(reason)───▶ Dispatcher
Dispatcher ──ACCEPT_PROPOSAL──▶ winner (lowest bid, ties → lowest car id)
Dispatcher ──REJECT_PROPOSAL──▶ the others
winner     ──INFORM / status update──────────────▶ board
```

* **Refusal** (eligibility) is classical: out of service, fire mode, or full.
* Calls are auctioned **one at a time**, so later rounds see earlier awards.
* **Bid** (classical) = marginal cost of inserting the call into the car's A\* route:
  `bid = W1·wait + W2·ride + W3·crowding + W4·energy − urgency·escalation_bonus`.
  The urgency discount makes a starving call cheaper for everyone → anti-starvation.

**Why CNP?** Decentralised: each car prices with its own private knowledge (its route); the
dispatcher only compares numbers. Scales linearly in N, robust to a car failing (it refuses).

---

## 4. Search, optimisation and safety (Phases 1–2)

* **Routing as state-space search.** State = (car floor, direction, remaining stops). Actions
  = serve a legal next stop. Cost = time × waiting weight. **A\*** with an admissible,
  consistent heuristic (lower bound on remaining travel) → optimal stop order. Above 10 stops
  we fall back to LOOK (bounded rationality).
  *Admissible*: h(n) ≤ true cost. *Consistent*: h(n) ≤ c(n,n') + h(n'). Consistent ⇒ admissible.
* **Simulated annealing** periodically reassigns calls globally (escape local minima by
  accepting worse moves with probability e^{−Δ/T}).
* **Hill climbing / minimax** choose where idle cars park.
* **Forward chaining** safety rules (Prolog-style) fire on facts like `fire_alarm`.
* **Invariants checked every tick**: no motion with doors open, load ≤ capacity, floors in
  range, no riders in an out-of-service car. Our 100-seed evaluation found a real bug (fire
  alarm mid-move left a stale move timer) — fixed and pinned by regression tests. Final test:
  **0 violations in 7,800 runs.**

---

## 5. LiftZero-BC — learning to bid by imitation (Phase 4)

**Idea.** Replace only the bid computation (one A\* search per car per call) with a neural
network that predicts the bid from public information.

### Features (schema v2)
* Call token (14): floor, direction, waiting, urgency, distance to lobby, traffic pattern…
* Car token (26, one per car): floor, direction, load, door state, plan summary, distance
  and direction to the call, "call on route"…
* Global token (10): time, building size, fleet load, idle fraction, **public cost weights
  W1–W4** (added in v2: the two teachers use different weights; without them identical inputs
  had conflicting labels).

### Set-Transformer (237,575 parameters)
Tokens `[global, call, car_1 … car_N]` → 4 pre-LayerNorm Transformer layers (d = 64, 4 heads)
→ per-car score (log-cost, lower = better), 4-part bid breakdown, value head, attention.

* **Permutation-equivariant**: no positional encoding, no car-id embedding ⇒ permuting cars
  permutes outputs. One network serves any fleet size.
* Self-attention: `Attention(Q,K,V) = softmax(QKᵀ/√d_k + mask) V`; padded cars get −10⁹ in the
  mask.

### Losses
`L = 1.0·L_list + 0.5·L_reg + 0.5·L_rank + 0.2·L_aux + 0.3·L_pair`
* Listwise cross-entropy: `−log softmax(−s_i e^{−τ})[winner]` (uniform over exact ties).
* Regression on log-costs (Huber), centred rank regression, bid-breakdown regression.
* Pairwise hinge winner vs runner-up, ×2 weight on **hard** decisions (runner-up within 10 %).

### DAgger (Dataset Aggregation)
BC learns on states the **teacher** visits; when the learner drives, small errors lead to new
states → compounding error. DAgger: let the **learner drive**, ask the **teacher** to label
every state it reaches, aggregate, retrain. β-mixing (β = 0.5 then 0.25) lets the teacher
execute some awards for safety. We ran 2 rounds (1.05 M + 0.94 M decisions).

### Results (real numbers)
| | value |
| --- | --- |
| Validation agreement, 3 seeds | **88.3 ± 0.4 %** (gate 85 %) |
| Test agreement (tie-aware) | **89.9 % (91.7 %)** |
| Unseen 33–40-floor buildings | 83.7 % (87.2 %) |
| Best classical scoring rule (cost-greedy) | 54.6 % |
| Closed loop, bare-CNP pair | within ±5 % of teacher in **13 / 13** regimes |
| Closed loop, full-system pair | within ±5 % in 10 / 13 |
| Inference latency (8 cars, CPU, ONNX) | **0.19 ms** |

Why not 100 %? The public features summarise each car's plan; the teacher sees every queued
stop and exact waiting counts/priorities. Errors concentrate on crowded lobby calls in tall
buildings.

---

## 6. LiftZero-PPO — cooperative reinforcement learning (Phase 5)

**Why RL?** Imitation can at best match the teacher. RL optimises the real objective.

**Semi-MDP formulation**
* Decision = assign one hall call; decisions arrive at irregular times (Δt seconds apart).
* State: the public tokens. Action: one eligible car.
  Policy `π(a|s) = softmax_{a∈E}(−score_a · e^{−τ})`.
* **Team reward** (identical for all cars ⇒ cooperative):
  `r = −(Σ_waiting Δt + 0.5·Σ_riding Δt)/100 − 0.02·floors − 0.5·new long waits`.
* Time-aware discount `γ_t = 0.99^{Δt/5}`.

**GAE**: `δ_t = r_t + γ_t V(s_{t+1}) − V(s_t)`, `A_t = δ_t + γ_t λ A_{t+1}` (λ = 0.95).
The episode horizon **truncates** (bootstrap V), it does not terminate.

**PPO clipped objective**: `L_clip = E[min(ρ_t A_t, clip(ρ_t, 1−ε, 1+ε) A_t)]`, ρ = π/π_old,
ε = 0.2. Prevents destructively large policy updates.

**Full loss**: `L = −L_clip + 0.5·Huber(V_priv − R) + 0.5·MSE(V_pub − V_priv) − c_e·H[π] + β·KL(π_BC ‖ π)`.
* **KL anchor** to the imitation policy (β 0.2 → 0 over 60 % of training) stops early noisy
  advantages from destroying the good warm start (no catastrophic forgetting).
* **CTDE**: *decentralised execution* — every car runs the same public-information actor;
  *centralised training* — the critic `V_priv` also sees privileged state (per-floor queues,
  oldest-waiter ages, riders per destination). The actor never sees it.
* Entropy bonus 0.01 → 0.001; Adam 3e-4 → 3e-5; early stop if approx-KL > 0.03.

**Where it trains.** Directly on the real simulator (the fast twin was measured at only 3.7×
the speed and was not faithful), so there is no sim-to-real gap.

**Status (be honest):** implemented from scratch and unit-tested; a smoke run passes end to
end; full training runs on Kaggle (`docs/lift/KAGGLE_TRAINING.md`). No PPO result is claimed
until that run finishes.

---

## 7. Look-ahead search: PUCT-MCTS (designed next step)

**PUCT selection**: `a* = argmax_a [ Q(s,a) + c_puct · P(s,a) · √(Σ_b N(s,b)) / (1 + N(s,a)) ]`
* Q = mean return of action a; P = network prior; N = visit counts; c_puct ≈ 1.25.
* Exploitation (Q) vs exploration (prior × uncertainty bonus that shrinks with visits).
* Design: the dispatcher (which sees all bids) searches only contested calls, among cars that
  proposed; hidden destinations are sampled (determinisation for partial observability);
  ≤ 50 ms budget. **Not implemented** in this submission.

---

## 8. Twenty viva questions with answers

1. **Why a multi-agent system instead of one central optimiser?** Each car holds private
   knowledge (its route) and computes its own bid; the dispatcher only compares numbers. It
   scales linearly in cars, tolerates a failed car (it refuses), and maps to the CNP literature.
2. **What is the Contract Net Protocol?** A FIPA interaction protocol: manager sends CFP,
   contractors PROPOSE or REFUSE, manager sends ACCEPT/REJECT, winner INFORMs. We use one round
   per hall call.
3. **How is a car's bid computed?** Classically: A\* plans the route with and without the new
   call; the bid is the weighted marginal cost (wait, ride, crowding, energy) minus an urgency
   discount. In LiftZero the network predicts it.
4. **Why A\* and why is your heuristic admissible?** A\* is optimal with an admissible
   heuristic. Ours is `h = Σ_i w_i · travel(cur, f_i) + energy_lower_bound(span)`: every pending
   stop costs at least its weighted straight-line travel, and the car must at least sweep the
   span of pending floors — two lower bounds on disjoint parts of the cost, so `h ≤ h*`. It is
   also consistent (proof in `docs/DESIGN.md` §6.1).
5. **How do you guarantee safety?** Eligibility is never learned (full/faulty/fire cars refuse
   classically); motion only happens with doors closed; invariants are checked every tick; the
   safety agent fires forward-chaining rules. 0 violations in 7,800 test runs.
6. **What happens if a car fails mid-run?** It goes out of service, releases its calls (they are
   re-auctioned), and ejects riders to the floor; it refuses all CFPs until repaired.
7. **What does the traffic monitor learn?** Per-floor arrival rates (EWMA), the traffic pattern,
   and it retunes the cost weights W1–W4 — a learning agent in AIMA's sense.
8. **Why a set-Transformer?** Cars form a *set*: order is meaningless and fleet size varies.
   Self-attention without positional encoding is permutation-equivariant, so one network works
   for 2–32 cars and lets each car's price depend on the others.
9. **Why imitation before RL?** RL from scratch explores badly and can learn unsafe behaviour;
   BC gives a competent starting policy quickly (88 % agreement in ~20 min of training).
10. **Why is DAgger needed after BC?** BC trains on teacher states only; at deployment the
    learner's own mistakes lead to unseen states (covariate shift). DAgger collects teacher
    labels on learner-visited states.
11. **Your agreement is 88 %, not 100 % — why?** The features are public summaries; the teacher
    sees every queued stop and exact waiting counts. Many disagreements are near-ties with
    almost zero cost difference (regret ≈ 2 cost units).
12. **How do you know the learned bidder is not just memorising?** Splits are by run seed; it
    scores 89.9 % on unseen test runs and 83.7 % on larger buildings it never saw.
13. **Why PPO and not DQN?** The action set changes per decision (eligible cars, variable fleet);
    a policy-gradient method with a masked softmax handles that naturally; PPO's clipping gives
    stable updates.
14. **What is CTDE?** Centralised training (critic may use privileged global state) with
    decentralised execution (each car acts only on public information).
15. **Why the KL anchor?** Early critic estimates are noisy; without an anchor PPO can drift
    far from the good BC policy and forget it. The KL penalty to π_BC decays to zero.
16. **What is the SMDP discount?** Decisions are irregularly spaced, so the discount per step is
    γ^{Δt/τ}: a long gap discounts the future more than two calls in the same second.
17. **Why not use centralised RL that assigns all calls jointly?** The joint action space is
    N^k and the system would no longer be decentralised; CTDE keeps decisions local and the
    actor small enough to run per car.
18. **What if a car loses communication?** It sends no PROPOSE and is simply not awarded calls;
    calls it held are released when it goes out of service and re-auctioned.
19. **How is your evaluation fair?** 100 paired seeds per regime, identical code paths, the
    learned strategy differs from its teacher only in the bidder, 95 % bootstrap CIs, test seeds
    used once after selection on validation seeds.
20. **What are the limitations?** Simulated buildings only (single lobby, 6–40 floors); routing
    still classical A\*; PPO training and MCTS are not finished; the fast twin is not faithful.

---

## 9. What we did not finish (say it before they ask)

* PPO training results (code ready; training on Kaggle).
* PUCT-MCTS search and the React "Brain panel".
* A faithful fast twin (measured 3.7× faster, 27–126 % wait error) — so RL uses the real simulator.

---

## 10. Live demonstration script (≈ 6 minutes)

**Before the viva:** `uv sync`, then `uv run elevator serve`, open http://localhost:8000.
Keep a terminal ready in the repo.

1. **Adhikkesh (1 min)** — Mission Control. Pick scenario `morning_up_peak`, strategy `full`,
   press play. Point at the shafts: cars, doors, waiting passengers. "Each car is an agent."
2. **Sisr (1.5 min)** — Auction theatre: show one CFP → PROPOSE/REFUSE → ACCEPT round with the
   bid breakdown and the winner's reason. Then Compare mode: `nearest_car` vs `full` side by side
   — watch the wait sparkline diverge. Trigger a car fault and a fire alarm from the disturbance
   panel; show refusals and fire recall.
3. **Kavin (1.5 min)** — switch strategy to `liftzero_bc`: the decision trace now shows
   "[LiftZero] net bids…". In the terminal:
   `uv run elevator lift card` (model card) and `uv run elevator lift bench-infer` (≈ 0.2 ms).
4. **Akash (1.5 min)** — open `reports/viva/fig1_wait_time_comparison.png` and
   `fig4_learned_vs_teacher.png`; explain the paired test and CIs. Show
   `uv run elevator lift train-ppo --help` and the PPO design.
5. **All (30 s)** — Story mode for a guided recap, then questions.

**Fallback if the server fails:** `uv run elevator run --scenario morning_up_peak --strategy
liftzero_bc` prints the full metric summary in the terminal.

---

## 11. Numbers to remember

* Up-peak avg wait: nearest 56.2 s → full 29.9 s (**−47 %**).
* LiftZero-BC: **237,575** parameters, **88.3 ± 0.4 %** val agreement, **89.9 %** test,
  **0.19 ms** per decision, **0** safety violations in **7,800** test runs.
* Data: **1.47 M** expert decisions + **2 M** DAgger decisions.
* Tests: **444** Python + **66** frontend.
