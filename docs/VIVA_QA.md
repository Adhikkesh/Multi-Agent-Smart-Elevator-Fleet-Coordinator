# Viva preparation — 20 likely questions

Crisp answers, with the evidence to point at. Numbers are from `docs/TESTING.md`.

---

### 1. Why is the elevator a *goal-based and utility-based* agent rather than just one?

**Goal-based** because it holds a set of stops it must reach and *searches* for an action
sequence that achieves them — change the goals and the behaviour follows without rewriting
any rule. But "achieve the goals" does not say which plan to *prefer*, and the candidate
plans differ in kind, not just degree: one is quicker for the people aboard, another kinder
to the crowd waiting downstairs, another cheaper in electricity. Choosing between
incommensurable outcomes needs a **utility** function, so the car scores plans with
`W1·wait + W2·ride + W3·crowding + W4·energy`. That same function is what it bids with.

### 2. Why can't the FloorAgent be a simple reflex agent?

Because the correct action depends on information that is *not in the current percept*. A
simple reflex agent seeing "three people waiting" cannot tell whether it has already
requested service, nor how long they have been there. It would either spam the dispatcher
every tick or never re-request. So the agent keeps internal state — `active_calls`,
`call_since`, `assigned_car` — and updates it from percepts. That is exactly AIMA's
model-based extension, and it is what makes the fairness guarantee possible.

### 3. Prove your heuristic is admissible.

`h = Σᵢ wᵢ·travel(cur, fᵢ) + energy_lower_bound(line span)` — two independent lower bounds
on *disjoint* parts of the true cost:

1. Every pending stop must eventually be reached, and cannot be reached faster than
   travelling straight to it, so it costs at least `wᵢ·travel(cur, fᵢ)`. Dwell time and the
   delay other stops inflict are both dropped, which only lowers the estimate.
2. The car must visit the lowest *and* the highest pending floor, so it travels at least
   the span of that line through its position, and opens its doors at least once per stop.

Neither double-counts the other, so the sum is still `≤ h*`. Admissible, therefore A* is
optimal. Asserted numerically over 200 random instances in
`tests/test_search.py::TestHeuristic`.

### 4. And consistent? Why does that matter separately?

For a transition `s --a--> s'` serving stop *j*: the travel term loses `wⱼ·travel(cur, fⱼ)`
exactly, and each remaining term grows by at most `wᵢ·travel(cur, fⱼ)` by the triangle
inequality on a line — while the step cost charges `(travel + dwell) × Σ w` over everything
still pending, which covers both. So `h(s) ≤ c(s,a,s') + h(s')`.

It matters because admissibility alone only gives optimality; **consistency** additionally
guarantees A* never re-expands a node, so it expands no more nodes than UCS. That is the
gap we measure: 98 nodes for UCS versus **41** for A* on a live 7-stop instance.

### 5. Why is A* better here than Dijkstra/UCS if they find the same answer?

They find the identical cost — we assert exactly that. The difference is *work*: A* uses
the heuristic to avoid exploring stop orderings that cannot beat the incumbent, expanding
40–60 % fewer nodes. In a dynamic environment where cars replan on every event, that is the
difference between planning being free and planning being the bottleneck.

### 6. Why is BFS in the comparison at all if it is not optimal here?

To make the point that **optimality is relative to the cost function**. BFS optimises the
*number of actions* — the number of stops — which would only be the right objective if
every stop cost the same. Here a stop's cost depends on the distance travelled, the dwell
time and how many people are waiting on the outcome, so BFS returns a shallow but expensive
route. It is the control that shows why UCS and A* are needed.

### 7. Why Contract Net rather than a central optimiser?

Three reasons. **Robustness:** no central plan exists to be invalidated, so a car failing is
just a bidder dropping out — the calls are re-auctioned and the fleet degrades gracefully.
**Locality of information:** each car knows its own load, doors and committed plan; making a
central optimiser reason about all of that means duplicating every car's state. **Truthful
costs:** a car bids its *marginal* cost, the genuine opportunity cost of taking the call,
which is exactly the information needed to allocate well. The cost is coordination
overhead: ~12 messages per call, which we measure.

### 8. Contract Net is greedy — one call at a time. Doesn't that give a poor global result?

Yes, and that is precisely why the dispatcher also runs **simulated annealing** every 30 s
over the calls nobody has picked up yet. CNP gives an immediate, robust answer; local search
repairs the drift. The two operate at different time scales deliberately. We also protect
against churn with a hysteresis threshold of 80 — without it the fleet reshuffles for
trivial gain and passengers watch their hall lantern flicker between cars.

### 9. Why simulated annealing rather than hill climbing?

Hill climbing halts at the first local optimum. The assignment space is full of them: moving
any single call often makes things worse even when swapping *two* would help. SA accepts
worsening moves with probability `exp(−ΔE/T)` and so escapes them. We keep hill climbing in
the Search Lab precisely so the contrast is visible — its curve flattens early and higher.
Crucially, SA tracks the best configuration it ever saw, so **it can never return something
worse than it started with** — asserted in the tests, which is what makes it safe to run on
a live fleet.

### 10. Where is the adversarial search, and isn't nature-as-adversary unrealistic?

Idle-car parking. MAX (the dispatcher) chooses where to park; MIN ("nature") then picks the
floor of the next hall call, choosing the one that hurts most. Alpha-beta returns the
identical value with 55 % fewer nodes (105 → 47).

On realism: nature is not malicious, so this is pessimistic — and that is the *point*. It
buys a **worst-case guarantee**: whatever floor calls next, the response distance is no
worse than the minimax value. Hill-climbing parking optimises the *expected* response and
will happily leave a quiet wing completely uncovered. The honest alternative is
**expectimax**, weighting each floor by the learned arrival probability instead of taking
the minimum; that optimises the average rather than the worst case. We use minimax because
the guarantee is what a safety-adjacent service wants, and because it is the clean textbook
illustration of alpha-beta.

### 11. Why a rule-based system for safety instead of `if` statements?

Three reasons that are specific to safety. **Auditability:** the rules are data, reviewable
by a safety engineer who does not read Python. **Explanation:** every action is traceable to
the rule and the variable bindings that produced it — the rules-fired log is a real
explanation, not a print statement. **Explicit priority:** `salience` states that fire
recall (100) outranks overload (60), rather than leaving it implicit in the order of some
`if` branches. Adding a policy means adding a rule, not editing the agent. The same rules
are mirrored in Prolog in `docs/safety_rules.pl` to show the knowledge is independent of the
inference procedure.

### 12. What exactly makes the TrafficMonitor a *learning* agent?

It has AIMA's four components. *Performance element:* the dispatcher and cars doing the
work. *Critic:* the observed arrivals, which are the feedback signal. *Learning element:* an
EWMA estimate of the per-floor arrival rate λ_f. *Problem generator:* the pattern classifier,
which proposes a different policy to try when the shape of demand changes.

The key point is that the true arrival rates are **hidden** — only the generator knows them,
and the agents never see them. And its output is not a statistic on a dashboard: it retunes
the fleet's cost weights and its parking policy, so learning visibly changes behaviour.

### 13. Your environment is "partially observable". Where does that actually bite?

At the moment of assignment. A passenger's destination is private until they board and press
a car button, so the dispatcher must commit a car to a call **before** knowing whether that
call is going one floor up or fourteen. The system therefore bids on expected cost, and then
**replans once boarding reveals the truth** — closed-loop, not open-loop. It is also why a
car's plan is re-derived on a boarding event rather than trusted.

### 14. Why staged activation? What would break without it?

Each tick runs `sense → communicate → decide → act`, with every agent of a type completing a
stage before the next begins. Without it, an agent activated early would sense a world that
agents activated later had not yet acted on, so the outcome would depend on **activation
order** rather than on the agents' reasoning — and the run would not reproduce from its
seed. Reproducibility is what makes the benchmark meaningful, and it is asserted:
`tests/test_simulation.py::TestDeterminism` checks that the same seed gives identical
metrics, identical car trajectories *and* an identical message log.

### 15. How do you prevent starvation? Isn't lowest-bid-wins inherently unfair?

It is — a floor that is always expensive to reach would never win, and there is always a
cheaper call somewhere. So fairness is enforced separately. The FloorAgent tracks how long
each call has been outstanding and **re-escalates** it every 60 s with rising urgency; the
urgency becomes a discount on every car's bid, so a starving call eventually outbids fresher,
closer ones. The reassignment objective also carries an aging term. The guarantee is tested
directly: after a drain, **every single passenger has been delivered**, under every strategy.

### 16. Your full strategy loses in `evening_down_peak`. Explain that.

It does, by 4.8 s against `collective`, and we report it rather than dropping the scenario.
Down-peak fills cars at the top and empties them at the lobby — which is exactly the pattern
a plain LOOK sweep is already optimal for, so the extra machinery has little to exploit.
Worse, the SA objective is a *cheap estimate* of insertion cost, and it cannot perfectly
predict what each car's own A* will do, so occasionally it moves a call that should have
stayed. We shrank the gap by tuning the hysteresis and interval, and by disabling idle-car
parking in down-peak once measurement showed it actively harmful (78.9 s with, 71.9 s
without). The honest conclusion is that **no single policy dominates every traffic regime** —
which is itself the argument for having a learning agent that switches policy.

### 17. `interfloor_light` shows no difference between any strategy. Doesn't that undermine the project?

No — it identifies the *boundary of applicability*, which is a result, not a failure. At
0.06 arrivals per second with four cars, a car is almost always free the moment a button is
pressed. There is no contention, so there is no allocation problem, so no allocation policy
can help. Coordination only has value under contention. That is why the peak scenarios are
the ones that matter, and there the full system cuts average wait by 70.6 % and long waits
by 90.3 %.

### 18. Why is `cnp_astar` *worse* than `collective` in up-peak? That seems backwards.

It was the most instructive result we got. Auctions allocate calls well but leave cars
wherever their last trip ended — scattered up the building — while in a morning rush
everybody arrives in the lobby. The allocation is good; the fleet is simply in the wrong
place. Adding smart parking takes it from 48.7 s to 15.9 s. The lesson is that these
techniques **interact rather than stack**: a better allocation policy is worthless if the
cars are badly positioned.

### 19. How does this scale, and what did you do to make it scale?

`stress_scale` is 40 floors and 8 cars — **a YAML file, not a code change** — and simulates a
full hour in 2.17 s, about 1 600× real time, at 4.6 ms per tick. Four things make that work:
replanning is **event-driven** rather than per tick; above 10 pending stops the car falls
back to an O(n log n) **LOOK sweep** (AIMA's bounded rationality — the optimal computation is
not worth its cost under real-time constraints); delivered passenger agents are retired while
their records stay with the collector so metrics remain complete; and the message and rule
logs are bounded. Floors, cars, strategies and scenarios are all configuration.

### 20. If you had another month, what would you do next?

Four things, in order of expected value. **(a)** Replace the SA objective's cheap insertion
estimate with the cars' real A* cost — that is the direct cause of the down-peak regression,
though it costs far more compute per evaluation. **(b)** Make the learning agent *choose* the
strategy, not just the weights: the benchmark shows different regimes want different
policies, and the monitor already detects the regime. **(c)** Switch parking from minimax to
**expectimax**, weighting floors by the learned arrival probability, and compare worst-case
against expected-case performance directly. **(d)** Add destination-entry (passengers declare
their floor in the lobby), which converts the environment from partially to **fully
observable** for up-peak traffic and would let the dispatcher group passengers by destination
— the single biggest real-world win available here.

---

## Rapid-fire facts

| Question | Answer |
| --- | --- |
| How many agents? | 6 types; ~22 instances by default (15 floors + 4 cars + 3 singletons) |
| Which AIMA chapters? | 2 (agents, PEAS), 3 (search), 4 (local search), 5 (adversarial), 7 & 9 (logic), plus multiagent §2.4 |
| Tests? | 200 passing, 96 % coverage |
| A* vs UCS, measured? | 41 vs 98 nodes expanded, identical cost 375 |
| Alpha-beta saving? | 105 → 47 nodes, identical value |
| Messages per call? | ~12 |
| Tick cost? | 0.14 ms (15 floors), 4.6 ms (40 floors) |
| Best result? | 70.6 % lower average wait, 90.3 % fewer long waits in up-peak |
| Is the run reproducible? | Yes — same seed gives identical metrics, trajectories and message log |
| What is the weakest part? | The SA objective is an estimate, not the true A* cost — it causes the down-peak regression |
