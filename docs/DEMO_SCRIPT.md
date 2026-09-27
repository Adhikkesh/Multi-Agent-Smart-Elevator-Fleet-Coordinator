# Five-minute live demo

**Before you start**

```bash
uv sync                                    # once, beforehand
uv run elevator serve                      # → http://localhost:8000
```

Open the browser **full screen**. The server starts **paused** on `demo_story`, so the
building is still and you can talk over it. Everything works offline.

The scenario is scripted, so the beats land at the same ticks every time (seed 7):

| Tick | Event |
| --- | --- |
| 95 | 12 passengers surge into the lobby |
| 160 | car 1 breaks down |
| 235 | car 1 is repaired |
| 265 | fire alarm |
| 310 | alarm cleared |

Set **Speed** to about 10× — 330 simulated seconds then take roughly half a minute of real
time per segment, which fits the five minutes with room to talk.

---

## 0:00 — 0:40 · The building and the agents

> "This is a 15-floor building with four lift cars. Every moving part is an autonomous
> agent, and each one is a different AIMA agent type."

- Point at the shafts: four cars, colour-coded. Arrows on the left are hall calls; a call
  turns the colour of the car that won it.
- Open the **Agent inspector** (right) and pick `car-0`. Read out the chip: *goal-based +
  utility-based agent*, and its PEAS.
- Switch to `safety` — *knowledge-based agent*, showing its seven production rules.
- Switch to `monitor` — *learning agent*.

> "Six agent types, and the dashboard reads the PEAS straight out of the code, so the
> documentation cannot drift from the implementation."

**Press Play.**

---

## 0:40 — 1:40 · Contract Net in action

> "Nobody is in charge of the fleet. When a hall button is pressed, the call is
> *auctioned*."

- Watch the **Contract Net auction** panel. Each round shows all four cars bidding, the
  cost split into **W**ait / **R**ide / **C**rowding / **E**nergy, and the winner in green.
- Point out a `REFUSE` if one appears (a full car).

> "A car's bid is the *marginal* cost of inserting the call into the plan its own A* search
> just produced — so a car already passing the floor bids almost nothing, and one that
> would have to reverse bids a lot. That is exactly the information needed to choose well."

- Scroll the **Message log**: colour-coded `REQUEST → CFP → PROPOSE → ACCEPT/REJECT →
  INFORM`. That is FIPA-ACL, threaded by conversation id.

> "Around twelve messages per call, and every one is logged and auditable."

---

## 1:40 — 2:20 · Learning, and the lobby surge

At **t ≈ 95** twelve people appear in the lobby. (If you are running behind, press
**Lobby rush** yourself.)

- Point at the **Traffic monitor** panel: *Detected: up_peak*, with the reason — "85 % of
  trips start at the lobby" — next to the ground truth.

> "The agents are never told the arrival rates. This agent estimates them online with an
> EWMA and classifies the regime with a small rule base. When it decides this is a morning
> rush it retunes the fleet's cost weights and tells the dispatcher to park spare cars at
> the lobby — so a learning agent's output here is a *different dispatching policy*, not a
> number on a dashboard."

- Watch the **Average wait over time** chart absorb the surge and recover.

---

## 2:20 — 3:10 · A car breaks down

At **t ≈ 160** car 1 fails. (Or press **Inject fault** with Car 1 selected.)

- The car greys out and shows `OOS`.
- **Safety rules fired** shows `R4_car_fault_out_of_service` with its effect: *"car 1 out of
  service; N call(s) re-auctioned"*.
- The **message log** shows a `FAILURE`, then fresh `CFP`s for the released calls.

> "The SafetyAgent is knowledge-based: it forward-chains over declarative rules. Rule R4
> fires, the car stops at the next floor, its riders are turned out and re-queued **keeping
> their original arrival time** — so the metrics cannot flatter a failure — and its calls
> are re-auctioned with raised urgency. No central plan had to be repaired: the remaining
> three cars simply bid on the work."

At **t ≈ 235** the car returns to service.

---

## 3:10 — 4:00 · Fire alarm

At **t ≈ 265** the alarm sounds. (Or press **Fire alarm**.)

- The building tints red with a banner.
- Every car abandons its calls and runs to the lobby.
- **Rules fired** shows `R1_fire_recall`, `R2_fire_doors_open`, `R3_block_hall_calls`.

> "Safety overrides coordination. The SafetyAgent runs before the dispatcher in every tick,
> so once the alarm is raised no auction can award a call at all. Hall calls are blocked,
> the cars recall, and their doors are held open at the lobby. Salience makes that priority
> explicit — fire recall is 100, overload is 60 — rather than leaving it implicit in the
> order of some `if` statements."

At **t ≈ 310** clear it: `R7_fire_cleared` fires and normal service resumes.

> "The same rules are written as Horn clauses in `docs/safety_rules.pl`, and Prolog derives
> the same conclusions — the knowledge is independent of the inference engine."

---

## 4:00 — 4:40 · Search Lab

Click **Search Lab**.

> "This takes the routing problem out of the car that is running right now and gives the
> identical instance to all four search algorithms."

Read the table:

- **BFS, UCS and A\*** all reach the same optimal cost.
- **A\*** expands far fewer nodes than UCS — typically 40–60 % fewer.
- **Greedy** is fastest but returns a worse route.

> "That gap is the payoff of an admissible, consistent heuristic — proved in the design
> document and asserted in the tests over two hundred random instances. BFS optimises the
> *number* of stops, which is the wrong objective when the step costs differ."

Point at the two lower panels:

- **Simulated annealing vs hill climbing** — the jagged blue line accepts worse moves to
  escape local optima; the green running-best never worsens; hill climbing flattens early.
- **Minimax vs alpha-beta** — identical value, roughly half the nodes. *"The dispatcher is
  MAX choosing where to park; nature is MIN choosing the worst next call floor. The value
  is a worst-case guarantee, not an average."*

---

## 4:40 — 5:00 · Benchmark and close

Click **Benchmark**. If time is short, show the pre-generated
`reports/benchmark_avg_wait.png` instead of running it live.

> "Four strategies from a pure reflex baseline up to the full agent system, on four traffic
> regimes, several seeds each, same seeds for every strategy. In the peak regimes the full
> system cuts average wait by about 70 % against the nearest-car baseline and nearly
> eliminates waits over a minute. In light traffic every strategy is equivalent — there is
> nothing to coordinate when no car is ever contended for, and we report that honestly
> rather than hiding it."

Close on the **Theory** tab:

> "PEAS per agent, the environment classification with justifications, the rule base and
> the architecture — all read live from the running system. A hundred and eighty-two tests
> pass, including that A* matches uniform-cost search on two hundred random instances, that
> no passenger is ever starved, and that the same seed reproduces a run exactly."

---

## If something goes wrong

| Problem | Fix |
| --- | --- |
| Nothing animates | Check the pill top-right. If it says *disconnected* the page still polls over REST, so it keeps working — just less smoothly. Reload. |
| The demo drifts off the script | Every event has a button: **Inject fault**, **Fire alarm**, **Lobby rush**. Trigger them by hand. |
| You run out of time | Skip the Benchmark tab and show `reports/benchmark_avg_wait.png`. |
| A question needs a still frame | **Pause**, then **Step** one tick at a time. |
| You need to restart cleanly | Set scenario `demo_story`, seed 7, press **Reset**. |
