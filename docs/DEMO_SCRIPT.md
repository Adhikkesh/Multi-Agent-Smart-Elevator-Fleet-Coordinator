# Live demo script (≈ 7 minutes)

The demo is marked under **Review 2: Multi-Agent Execution & Interaction** and **Demo Quality
& Testing Scenarios**. Every step below names who does it, what to click, and what to say.

## Before the review (10 minutes earlier)

```bash
cd Multi-Agent-Smart-Elevator-Fleet-Coordinator
uv sync                                   # once; installs Python 3.12 + packages
uv run elevator verify                    # every scenario, safety check — expect "0 violations"
uv run elevator serve                     # keep this terminal open
```

* Open **http://localhost:8000** in Chrome, full screen (`F11` / `Ctrl+Cmd+F`).
* Press `T` if you want the light theme for the projector.
* Keep a second terminal open in the project folder for step 6.
* **Keyboard:** `Space` play/pause · `→` step 1 tick · `Shift+→` step 10 · `F` fire alarm ·
  `B` break car 1 · `R` reset · `1`–`6` switch pages · `?` all shortcuts.

## 1. Mission Control — the building and the agents (Adhikkesh, 1.5 min)

1. Top bar: **Scenario** = `morning_up_peak`, **Strategy** = `full`, press **Play** (speed
   about 10×).
2. Point at the **Building & Shafts** panel: "15 floors, 4 cars. Each car is an autonomous
   agent; the up/down arrows are hall calls raised by floor agents; the dots are waiting
   passengers."
3. Point at the **KPI strip**: average wait, p95, long waits, throughput, energy, messages per
   call — "this is our performance measure from the PEAS."
4. **Contract Net Auction** panel: "Every hall call is auctioned. The dispatcher sends a CFP,
   each car bids the extra cost of adding the call to its A\*-planned route, the lowest bid
   wins." Read the sentence that explains why the winner won (cost, its biggest component,
   margin over the next car).
5. **Live Message Bus**: "These are real FIPA-ACL messages — REQUEST, CFP, PROPOSE, REFUSE,
   ACCEPT_PROPOSAL, REJECT_PROPOSAL, INFORM. Agents never call each other directly."

## 2. Faults and fire — safety rules in action (Sisr Reddy, 1.5 min)

1. In **Disturbance & Chaos Injection**, choose **Car 1**, press **Break Car** (or key `B`).
   "Car 1 goes out of service; rule R4 fires; its calls are re-auctioned to the other cars —
   watch it refuse new CFPs in the message bus."
2. Press **Repair Car**.
3. Press **Fire Alarm** (or key `F`). "The Safety agent's forward-chaining rules fire: R1
   recalls every car to the lobby, R2 holds the doors open, R3 blocks hall calls." Point at
   the **Safety Agent** card listing the fired rules.
4. Press **Clear Alarm** — "R7 restores normal service." The invariants badge stays green
   ("Invariants OK"): no car ever moves with its doors open.

## 3. Agents page — multi-agent interaction (Sisr Reddy, 1 min)

1. Press `2` (Agents).
2. **Multi-Agent Communication Network**: "Six agent types; messages pulse along the edges."
3. Click **Replay Last Auction Conversation**: the **Protocol Sequence Diagram** shows one
   full Contract Net round message by message.
4. Scroll to **Shared Status Board**: "the blackboard each car publishes its status to."

## 4. Algorithm Lab — search strategy (Akash, 1.5 min)

1. Press `3` (Algorithm Lab), tab **1. Car Routing (BFS/UCS/Greedy/A\*)**, click
   **Run on Live Car**.
2. "The same live routing problem solved by four algorithms. BFS and UCS find the optimal
   route but expand many nodes; greedy is fast but its route costs far more; **A\* finds the
   optimal route expanding the fewest nodes** — because our heuristic is admissible and
   consistent." (Typical live numbers: A\* 15 nodes, UCS 40, BFS 158; greedy cost 512 vs
   237.)
3. Scroll to **A\* Step-Through Search Trace**, press **Play**: the open and closed lists
   change node by node.
4. Tabs **2. Assignment (Simulated Annealing)** and **3. Adversarial Parking
   (Minimax/Alpha-Beta)**: "annealing re-assigns calls fleet-wide; minimax with alpha-beta
   decides where idle cars wait — same value with fewer nodes."

## 5. Experiments — testing and comparison (Akash, 1 min)

1. Press `4` (Experiments) → **Compare Mode**: compare `nearest_car` against `full` on
   `morning_up_peak`. "Same scenario, same seed, same traffic pattern — the difference is the coordination.
   Nearest car bunches cars and lets the wait grow; the multi-agent system keeps it low."
2. Optionally run the **Monte-Carlo Benchmark** with 3 seeds.

## 6. Testing scenarios in the terminal (Kavin Karthic, 1 min)

In the second terminal:

```bash
uv run elevator scenarios                                   # the 9 scenarios + strategies
uv run elevator run --scenario car_breakdown                # one scenario, full metric summary
uv run pytest -m "not slow" -q                              # the fast test suite (≈ 1 min)
```

Say: "Nine scenarios — peaks, lunch, faults, fire, priority, a 40-floor stress test. Every
one delivers everyone with zero safety violations, and 523 automated tests check search
optimality, the protocol and the safety invariants on hundreds of random cases."

## 7. Theory page and the extension (Kavin Karthic, 1 min)

1. Press `5` (Theory): system PEAS, each agent's PEAS (tabs), the environment classification
   table — "the Review 1 analysis is built into the product."
2. Optional — press `6` (LiftZero Brain), click **LiftZero (imitation)**, go back to Mission
   Control and press Play for a few seconds, then return: "our extension — a neural network
   that learned to imitate the A\* bid. Each decision shows the network's bids and attention;
   **Why?** shows which features drove it."
3. Press `R` to reset. Hand over to questions.

## If something goes wrong

| Problem | Fix |
| --- | --- |
| Page does not load | check the server terminal is running; reload the page |
| Port 8000 busy | `uv run elevator serve --port 8001` and open http://localhost:8001 |
| Simulation looks stuck | press **Reset** (`R`), then **Play** |
| No internet | everything runs offline — no CDN, no external services |
| Laptop fails entirely | screenshots in `docs/img/ui/` and slides 12, 13, 15, 17 show every page |
| Terminal fallback | `uv run elevator run --scenario demo_story` prints the full metrics |

## Optional: guided Story Mode

Press `P` (or open http://localhost:8000/story) for a 10-step guided tour that triggers the
surge, the breakdown and the fire automatically; use `→` / `←` to move between steps and
`Esc` to leave.
