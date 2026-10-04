# LiftZero Control Room — Phase 2 Frontend Architecture

The **LiftZero Control Room** is an enterprise-grade, high-density operations center and AI laboratory interface for the Multi-Agent Smart Elevator Fleet Coordinator. Built with modern web technologies, it visualizes real-time distributed multi-agent negotiations, state-space search algorithms, local search optimizations, and adversarial games.

---

## 1. Architecture & Tech Stack

- **Framework**: [React 18](https://react.dev/) with [TypeScript 5.7](https://www.typescriptlang.org/) in strict mode.
- **Build Tool**: [Vite 6](https://vite.dev/) with Rollup code-splitting chunks:
  - `vendor` (`react`, `react-dom`, `react-router-dom`, `zustand`, `@tanstack/react-query`)
  - `charts` (`recharts`)
  - `graph` (`@xyflow/react`)
  - `motion` (`motion`)
- **Styling**: [Tailwind CSS v4](https://tailwindcss.com/) with custom CSS custom properties (HSL palette), dark theme default with light theme toggle.
- **Icons & Typography**: [Lucide React](https://lucide.dev/), bundled offline fonts (`@fontsource/inter` and `@fontsource/jetbrains-mono`). Zero external CDN requests.
- **State Management**:
  - **Live Stream Store** ([`src/store/liveStore.ts`](file:///home/adhikkesh/adhikkesh/SEM7/FOAI/Multi-Agent-Smart-Elevator-Fleet-Coordinator/frontend/src/store/liveStore.ts)): High-frequency ring buffers for 300 ticks of telemetry, 120 messages, and 20 auction rounds with `requestAnimationFrame` coalescing.
  - **UI Store** ([`src/store/uiStore.ts`](file:///home/adhikkesh/adhikkesh/SEM7/FOAI/Multi-Agent-Smart-Elevator-Fleet-Coordinator/frontend/src/store/uiStore.ts)): Theme, selected car, active inspection address, modal dialogs, and presentation mode.
  - **Story Store** ([`src/store/storyStore.ts`](file:///home/adhikkesh/adhikkesh/SEM7/FOAI/Multi-Agent-Smart-Elevator-Fleet-Coordinator/frontend/src/store/storyStore.ts)): 10-beat interactive demo sequence synchronizing simulation actions with narration.
- **Network Resilience**:
  - Resilient WebSocket client ([`src/live/socket.ts`](file:///home/adhikkesh/adhikkesh/SEM7/FOAI/Multi-Agent-Smart-Elevator-Fleet-Coordinator/frontend/src/live/socket.ts)) with exponential backoff (500ms to 10s), jitter, Page Visibility API throttling, and fallback polling.
- **Bundle Metrics**:
  - Gzipped JavaScript + CSS total size: **~320 kB** (well below the 2.5MB constraint).
  - Production build outputs to `src/elevator_mas/web/dist/`, served directly by FastAPI without Node.js required at runtime.

---

## 2. Views & Modules

### 2.1 Mission Control (`/`)
The primary live monitoring operations center:
- **Building Schematic**: Animated elevator shafts rendering real-time interpolated cabin positions, directional indicators, passenger load bars with color thresholds, door states (opening, open, closing, closed), waiting passenger queues per floor with wait times, priority indicator pips, strategic ghost parking targets, and fire evacuation smoke particles.
- **KPI Strip**: 8 real-time sparkline telemetry cards tracking:
  - Average Passenger Wait Time ($W_1$)
  - 95th Percentile Wait Time
  - Delivered Passengers Count
  - System Throughput (passengers/min)
  - Fleet Energy Consumption ($W_3$)
  - Inter-floor Wait Imbalance
  - Collective Legality Violations ($0$)
  - Contract Net Dispatch Latency
- **Contract Net Auction Panel**: Live visualization of the distributed auction mechanism. Inspects call origin floor/direction, round number, all car agent bid vectors ($b_i = \mathbf{w}^T \mathbf{c}_i$), and highlights the winning car badge. Includes auction history replay selector.
- **Agent Communication Stream**: Filterable live table of FIPA-ACL protocol messages (`REQUEST`, `CFP`, `PROPOSE`, `REFUSE`, `ACCEPT_PROPOSAL`, `REJECT_PROPOSAL`, `INFORM`). Supports pause-on-hover to inspect rapid exchanges and triggers full-screen sequence diagrams.
- **Simulation Control Bar**: Play, Pause, Single-step (+1, +10, +60 ticks), Reset with scenario/strategy/seed selectors, speed slider (0.1x to 50x), and theme toggle.
- **Disturbance & Chaos Panel**: Live injection of car hardware faults, repairs, lobby passenger rushes, and building fire evacuation drill.
- **Status & Connection Bar**: Real-time tick counter, ETA, wall-clock simulation time, and WebSocket status indicator pill.

### 2.2 Multi-Agent Architecture (`/agents`)
- **Interactive Multi-Agent Network**: Built using `@xyflow/react`. Nodes represent Dispatcher, Cars, Safety Agent, Traffic Monitor, and Floor Agents. Directed edges pulse with real-time FIPA-ACL performative colors upon message transmission.
- **FIPA-ACL Sequence Diagram**: Swimlane visualization of agent interactions across conversations. Highlights CFP broadcast, car bid submissions, dispatcher award/rejections, and arrival informs.
- **Shared Status Blackboard**: Public facts viewer (car locations, loads, operational modes, commitments) demonstrating agent privacy preservation.
- **Agent Inspector Drawer**: Comprehensive slide-out inspector detailing:
  - Formal PEAS specification (Performance, Environment, Actuators, Sensors)
  - AIMA 4e Agent Taxonomy Classification
  - Internal state variables (target floor, motor direction, door timer, passenger dropoffs)
  - Planned A* route sequence and search metrics (nodes expanded, runtime, energy breakdown)

### 2.3 Algorithm & State-Space Search Lab (`/lab`)
- **Car Routing Tab**:
  - Live side-by-side comparison of **BFS**, **Uniform-Cost Search (UCS)**, **Greedy Best-First**, and **A\*** on the exact identical routing problem.
  - Interactive **Step-Through Visualizer**: Step forward, backward, or auto-play through node expansions, showing the state, parent, $g(n)$, $h(n)$, and $f(n)$ frontier queues.
  - Nodes expanded, generated, and maximum frontier bar charts.
- **Simulated Annealing Tab**:
  - Local search reassignment optimization comparing standard Greedy Hill Climbing vs Simulated Annealing.
  - Temperature cooling schedule curve, energy surface convergence chart, and accepted worse moves counter (demonstrating local minima escape).
- **Adversarial Game Tab**:
  - Minimax with $\alpha$-$\beta$ pruning versus greedy passenger dispatch in an adversarial peak traffic game.
  - Pruning efficiency charts demonstrating state evaluation reductions.

### 2.4 Experiments & Empirical Validation (`/experiments`)
- **Monte-Carlo Benchmark Matrix**: Run multi-seed, multi-scenario simulations across all dispatch strategies (`nearest_car`, `collective`, `cnp_astar`, `full`).
- **Synchronized Compare Mode**: Side-by-side synchronized replay comparing two competing strategies on identical seeds, complete with scrubbable timeline, difference indicators, and leader callouts.
- **Data Export**: Export results to CSV, JSON, or capture high-resolution chart images as PNG.

### 2.5 Theoretical Primer (`/theory`)
- Comprehensive AIMA 4e reference guide for academic evaluation and viva defense.
- Environment properties table (Partially Observable, Stochastic, Sequential, Dynamic, Continuous, Multi-Agent).
- Formal PEAS formulation for all system agents.
- Contract Net Protocol finite state machine specifications.
- Mathematical proofs of heuristic admissibility and consistency.
- 12 curated Viva Voce defense questions and answers.

### 2.6 Mission Control Story Mode (`/story`)
- Full-screen guided presentation tour designed for evaluations.
- 10 structured acts walking through normal operation, peak rush, contract net bidding, fault failover, fire evacuation, heuristic search optimality, and energy conservation.

### 2.7 Classic Fallback (`/classic`)
- Preserves the original lightweight single-page HTML dashboard for low-resource environments and regression testing.

---

## 3. Parity & Validation

The frontend includes an exact TypeScript replication of the server's state-space routing physics ([`frontend/src/lab/routing.ts`](file:///home/adhikkesh/adhikkesh/SEM7/FOAI/Multi-Agent-Smart-Elevator-Fleet-Coordinator/frontend/src/lab/routing.ts)) with:
- Identical state expansion rules, collective constraints, and motion kinematics.
- Admissible heuristic calculation matching Python:
  $$h(n) = \max\left( \max_{p \in \text{rem}} w_p \cdot |f_p - f_{\text{dest}}|, \frac{\text{span}(\text{stops})}{v_{\max}} \right)$$
- **20 Deterministic Fixtures**: Tested in [`frontend/src/lab/routing.test.ts`](file:///home/adhikkesh/adhikkesh/SEM7/FOAI/Multi-Agent-Smart-Elevator-Fleet-Coordinator/frontend/src/lab/routing.test.ts) against Python outputs generated across diverse scenarios and ticks, verifying 100% numerical parity on optimal cost, nodes expanded, and sequence path.

---

## 4. Development & Build

### Running Dev Server
```bash
# Terminal 1: Backend
uv run elevator serve

# Terminal 2: Frontend (with hot module replacement)
cd frontend
pnpm dev
```

### Running Tests
```bash
# Vitest unit & integration tests with coverage
cd frontend
pnpm test

# Playwright end-to-end tests (launches server and headless browser)
pnpm e2e
```

### Production Build
```bash
cd frontend
pnpm build
```
Builds assets into `src/elevator_mas/web/dist/`. The Python package can then be distributed and served immediately without Node or pnpm.
