# LiftZero: Multi-Agent Smart Elevator Fleet Coordinator
## Master Handover & Execution Guide for Claude Pro (Phases 4 – 8)

> **Course:** FOAI (Foundations of Artificial Intelligence) — Semester 7  
> **Repository:** Multi-Agent-Smart-Elevator-Fleet-Coordinator (`main` branch)  
> **Toolchain:** Python 3.12 (`uv`), Node.js ≥ 20 (`pnpm`), React 19, Mesa 3, PyTorch, ONNX  

---

### Project Team

| Name | Register Number | Core Technical Ownership & Viva Focus |
| :--- | :--- | :--- |
| **Adhikkesh** | `CB.SC.U4CSE23101` | **System Architecture & Multi-Agent Coordination Lead**<br>• Phase 1 Multi-Agent Core & Mesa 3 Engine<br>• FIPA-ACL Contract Net Protocol (CNP) & Kinematics<br>• System Integration, CLI & End-to-End Orchestration |
| **Sisr Reddy** | `CB.SC.U4CSE23129` | **Mission Control & Frontend Experience Lead**<br>• Phase 2 React Control Room UI (`frontend/`)<br>• 60 FPS HTML5 Canvas 2D Shaft & Car Visualizer<br>• Live Auction Theatre, Compare Mode & Story/Present Tour |
| **Kavin Karthic** | `CB.SC.U4CSE23161` | **Simulation Environments & Imitation Learning Lead**<br>• Phase 3 48-Feature Schema & Fast Numba Twin Engine<br>• Gymnasium Single & Vectorized Environments<br>• Phase 4 LiftZero Set-Transformer, BC & DAgger Pipeline |
| **Akash** | `CB.SC.U4CSE23162` | **Deep RL, MCTS Arbitration & Benchmark Evaluation Lead**<br>• Phase 5 Cooperative Multi-Agent PPO (CTDE + KL Anchor)<br>• Phase 6 PUCT-MCTS Tree Search & Brain Panel API<br>• Phase 7 Quantitative Benchmarking, Ablations & Performance Metrics |

---

## 1. Ten Golden Directives for Claude Pro

Claude, read these 10 rules carefully before performing any action. These instructions take precedence over any individual phase prompt:

1. **NO BRANCHES — WORK STRICTLY ON `main`**:
   - **Do NOT create any branches** (ignore any prompt text saying `git checkout -b phase-...`).
   - All work across Phase 4, Phase 5, Phase 6, Phase 7, and Phase 8 must be developed and merged directly on `main`.
2. **COMMIT AND PUSH AFTER EACH PHASE**:
   - Immediately after completing and testing each phase, run git commands to stage all files, write a clean conventional commit (e.g., `feat(learning): complete Phase 4 transformer imitation and ONNX runtime`), and push directly to `origin main`.
3. **DO PHASE-BY-PHASE (DO NOT OVERLOAD YOURSELF)**:
   - Execute strictly **one phase at a time**.
   - Do NOT try to batch multiple phases into a single session. Give each phase your full focus and attention to detail.
4. **MAKE EVERYTHING ROBUST, CLEAN & PRODUCTION-GRADE**:
   - Zero shortcuts, zero unhandled exceptions, zero stubbed mocks in production code paths.
   - Enforce Python 3.12 type hints, docstrings with AIMA agent types and PEAS specifications, strict line length 100, and ensure `uv run ruff check .` and `uv run ruff format --check .` pass with 0 errors.
   - Maintain 100% test pass rate across `uv run pytest` and `pnpm test`.
5. **PRE-PHASE 4 AUDIT (MANDATORY BEFORE WRITING PHASE 4 CODE)**:
   - Before writing any Phase 4 code, audit Phases 1, 2, and 3. Verify that tests pass, check the folder structure, and inspect existing models. If any minor bug, lint, or missing detail is found, fix it immediately as step 1 of Phase 4.
6. **OBSERVE ALL README FILES & FOLDER STRUCTURE FIRST**:
   - Before generating code, explore the entire project tree:
     - `src/elevator_mas/` (core engine, agents, traffic, strategies, rules, learning, api, web)
     - `frontend/` (React 19 control room, components, stores, hooks)
     - `tests/` (unit, integration, invariants, benchmarks)
     - `AGENTS.md` (strict repo conventions)
7. **HIGH SCIENTIFIC RIGOR & SEEDED REPRODUCIBILITY**:
   - Never call `random.*` or `numpy.random.*` at the module level.
   - All randomness must flow through `model.random` / `model.rng` to preserve seed determinism.
8. **CREATE PHASE 8: PROFESSIONAL PPT PLAN & EMPIRICAL VISUALIZATIONS**:
   - Design a complete presentation plan (12–15 slides) for the 4-member team.
   - **Clean, human-like design with a crisp white background** (NO dark gaming slides, NO generic AI-generated clipart or robotic imagery).
   - Use real, empirical graphs generated via `matplotlib` / `seaborn` directly from the simulation benchmark results.
9. **COMPREHENSIVE PROJECT STUDY MATERIAL (`docs/STUDY_MATERIAL.md`)**:
   - Author a crisp, high-yield study dossier explaining:
     - The core Problem Statement (Elevator Group Control Problem - EGCP).
     - Why classic heuristic dispatchers (Nearest Car, Collective) fail in peak traffic.
     - Multi-Agent System architecture (FIPA-ACL, Contract Net Protocol, Hall Call Allocation).
     - The Machine Learning / RL / MCTS mathematical foundations and algorithms.
     - Step-by-step guide on how to present and explain the system during the viva using both the PPT and the live React frontend.
10. **CONTINUOUSLY ENHANCE & POLISH THE FRONTEND**:
    - The React UI is the centerpiece of the evaluation and viva demonstration (`uv run elevator serve`).
    - Antigravity created the foundational UI, but Claude Pro must elevate it: ensure smooth 60fps animations, intuitive controls, crystal-clear auction visualizations, and in Phase 6, connect the visual Brain Panel.
    - Always rebuild and verify the production bundle: `cd frontend && pnpm build`, ensuring `src/elevator_mas/web/dist/` is always committed and synced.

---

## 2. Complete Context: What Has Been Done So Far

### Phase 1: Message-Driven Multi-Agent Core (Status: COMPLETED & VERIFIED)
- **Engine:** Mesa 3 discrete-event simulation engine with staged activation (`model.agents_by_type[Cls].do(...)`).
- **Agents:**
  - `ElevatorAgent`: Autonomous car agent with internal state machine, continuous kinematics (acceleration, jerk, floor transit, door dwell), and A* search for marginal cost calculation.
  - `HallCallManager`: Tracks building-wide passenger requests and coordinates Contract Net auctions.
  - `DispatcherAgent`: Fleet coordinator managing auction rounds, bid evaluation, and tie-breaking.
  - `PassengerAgent`: Dynamic passenger generation based on Poisson traffic profiles.
- **Protocols:** Formal FIPA-ACL messaging protocol (`CFP` -> `PROPOSE` -> `ACCEPT_PROPOSAL` -> `REJECT_PROPOSAL` -> `INFORM`).
- **Telemetry:** Real-time WebSocket streaming via `StatusBoard` and FastAPI server (`uv run elevator serve`).

### Phase 2: React Control Room UI (Status: COMPLETED & VERIFIED)
- **Tech Stack:** React 19, TypeScript, Vite, Tailwind/CSS variables, Lucide icons, Vitest.
- **Features Implemented:**
  - **Mission Control View:** Real-time 60 FPS HTML5 Canvas 2D visualization of elevator shafts, cars, passengers, door states, and kinematics.
  - **Live Auction Theatre:** Visualizes Contract Net Protocol bidding in real time, displaying car bids, marginal cost evaluations, and winning assignment traces.
  - **Algorithm Lab / Compare Mode:** Runs side-by-side simulations of different dispatch algorithms (Nearest Car vs A* vs Contract Net) with real-time comparative performance graphs.
  - **Story / Present Mode:** Guided presentation tour designed for viva examiners, explaining the system step-by-step.
- **Delivery:** Pre-compiled production bundle committed at `src/elevator_mas/web/dist/`. 66 Vitest unit tests passing.

### Phase 3: Learning Environment & Fast Twin (Status: COMPLETED & VERIFIED)
- **Schema:** 48-feature schema v1 (`schema.py`) encoding car positions, velocities, loads, door states, hall call distributions, and spatial urgency.
- **Expert Recorder:** Multi-worker parallel dataset recorder (`recorder.py`) generating sharded `.npz` files of expert A* heuristic decisions across standard traffic scenarios (morning up-peak, evening down-peak, lunch rush, inter-floor).
- **Fast Numba Twin Simulator:** Ultra-fast vector simulator (`twin.py`) executing 10,000x faster than the full agent sim for high-throughput RL rollouts.
- **Gymnasium Environments:** Single-agent and vectorized Gymnasium environments (`env.py`) supporting standard RL training interfaces.
- **Evaluation Harness:** Multi-scenario benchmark runner (`evaluate.py`) computing Average Wait Time (AWT), 95th Percentile Wait Time (P95), energy consumption, and throughput.
- **CLI Commands:** `elevator learn record`, `elevator learn eval`, `elevator learn benchmark-twin`. Total pytest count: 253 tests passing.

---

## 3. Execution Roadmap: Remaining Phases (Phases 4 – 8)

All phase prompts are stored in `prompt/`. Follow them strictly in sequence:

```
[Phase 1, 2, 3: Completed on main]
              │
              ▼
    [Phase 4: Transformer Imitation (BC + DAgger)] ──▶ Commit & Push to main
              │
              ▼
    [Phase 5: Cooperative Multi-Agent PPO (CTDE)] ──▶ Commit & Push to main
              │
              ▼
    [Phase 6: MCTS Arbitration & React Brain Panel] ──▶ Commit & Push to main
              │
              ▼
    [Phase 7: Full Evaluation, Ablations & Release v2.0] ──▶ Commit & Push to main
              │
              ▼
    [Phase 8: Master PPT Plan, Matplotlib Charts & Study Material] ──▶ Commit & Push to main
```

---

### Phase 4: Transformer Imitation Learning (BC + DAgger)
* **Prompt File:** `prompt/Phase4_Transformer_Imitation_Prompt.md`
* **Key Tasks:**
  1. Build `LiftZeroPolicy` (PyTorch set-Transformer architecture, ~0.27M parameters, permutation-equivariant).
  2. Implement Behavioral Cloning (BC) training pipeline with pairwise ranking loss + cross-entropy.
  3. Implement DAgger (Dataset Aggregation) loop to collect online car decisions and eliminate distributional drift.
  4. Export model to ONNX (`liftzero_bc.onnx`) with CPU `onnxruntime` inference (< 2 ms latency).
  5. Integrate into simulation engine as registered strategy `liftzero_bc`.
* **Acceptance Criteria:** ≥ 85% teacher agreement on validation split, closed-loop AWT within ±5% of A* teacher, zero safety violations.

---

### Phase 5: Cooperative Multi-Agent PPO (CTDE)
* **Prompt File:** `prompt/Phase5_Cooperative_PPO_Prompt.md`
* **Key Tasks:**
  1. Centralized Training, Decentralized Execution (CTDE) architecture.
  2. Centralized Critic observing global fleet state; decentralized Actor computing local car value/bid.
  3. Implement custom PPO training loop with **KL-divergence anchor** to the BC policy to prevent catastrophic forgetting and erratic elevator behaviors.
  4. High-throughput training on Numba Fast Twin vectorized environment.
  5. Validate on full Mesa simulation and register strategy `liftzero_ppo`.
* **Acceptance Criteria:** Outperforms pure BC and baseline heuristic in peak traffic wait times and energy consumption.

---

### Phase 6: MCTS Arbitration & React Brain Panel
* **Prompt File:** `prompt/Phase6_MCTS_Brain_Panel_Prompt.md`
* **Key Tasks:**
  1. Implement PUCT-MCTS (Predictor Upper Confidence bounds applied to Trees) for dispatch arbitration.
  2. Use the trained policy-value network as heuristic prior and rollout evaluator.
  3. Search budget control: 10–50 rollouts per dispatch within strict real-time deadline (< 20 ms).
  4. Expose search tree telemetry via FastAPI REST API (`/api/brain`).
  5. **Frontend Enhancement:** Build the **Brain Panel** in React, displaying an interactive tree graph, visit counts, and car value heatmaps.
  6. Register strategy `liftzero_mcts`.
* **Acceptance Criteria:** Demonstrable search improvement over greedy policy; live visual rendering of MCTS tree in React dashboard.

---

### Phase 7: Evaluation, Ablations & Documentation
* **Prompt File:** `prompt/Phase7_Evaluation_Docs_Release_Prompt.md`
* **Key Tasks:**
  1. Run exhaustive benchmarks across 4 traffic patterns:
     - Morning Up-Peak (heavy lobby ingress)
     - Evening Down-Peak (heavy egress to ground)
     - Lunch Rush (high inter-floor two-way traffic)
     - Off-Peak / Random Poisson
  2. Ablation Matrix comparing:
     - Nearest Car Heuristic
     - Collective Control Heuristic
     - Pure A* Search
     - LiftZero-BC (Imitation)
     - LiftZero-PPO (Reinforcement Learning)
     - LiftZero-MCTS (Full Hybrid AI)
  3. Log comprehensive metrics: Average Wait Time (AWT), P95 Wait Time, Energy Consumption (kWh), Door Cycles, Compute Latency.
  4. Generate release artifacts, documentation, and verify fresh clone execution.

---

### Phase 8: Master PPT Plan, Empirical Visualizations & Study Material (NEW)

Claude Pro must create and execute Phase 8 upon completion of Phase 7:

#### 1. Presentation Deck (PPT) Plan
- **Aesthetic:** Clean, minimalist, human-designed with a **crisp white background**, modern typography (Inter/Calibri/Segoe UI), subtle accent borders (slate/navy/emerald), and high contrast.
- **Rule on Visuals:** **NO AI-generated clipart or futuristic robot illustrations.** Every chart must be a genuine empirical plot generated via Matplotlib/Seaborn from the simulation benchmark data.
- **Slide Deck Structure (14 Slides):**
  1. **Title Slide:** Project LiftZero: Multi-Agent Smart Elevator Fleet Coordinator | Team Members & Register Numbers | Course & Department.
  2. **The Problem:** The Elevator Group Control Problem (EGCP) — high passenger wait times, energy waste, and exponential combinatorial complexity in high-rise buildings.
  3. **Why Classic Approaches Fail:** Flaws of Nearest Car and rule-based dispatchers during peak hours (car bunching, starvation of upper floors).
  4. **Multi-Agent System Architecture:** Autonomous cars as intelligent agents, FIPA-ACL Contract Net Protocol (CFP, Propose, Accept, Reject, Inform).
  5. **Kinematic & Physical Modeling:** Realistic physics — jerk limits, acceleration curves, door dwell times, capacity constraints.
  6. **Feature Engineering & Fast Twin Engine:** 48-feature observation space and 10,000x accelerated Numba twin simulator.
  7. **Behavioral Cloning & DAgger (LiftZero-BC):** Set-Transformer neural architecture, imitation learning, and solving covariate shift.
  8. **Cooperative Multi-Agent RL (LiftZero-PPO):** Centralized Training with Decentralized Execution (CTDE), reward formulation, KL-divergence safety anchor.
  9. **MCTS Lookahead Arbitration (LiftZero-MCTS):** PUCT search tree, real-time lookahead dispatch, bridging neural policy with online search.
  10. **Mission Control React Dashboard:** Real-time 60 FPS Canvas visualization, live auction theatre, algorithm comparison lab, and brain inspection panel.
  11. **Empirical Results & Benchmark Comparisons:** Matplotlib comparative bar charts and box plots (AWT, P95 wait times, energy efficiency across 4 traffic patterns).
  12. **Ablation Studies:** Quantifying the incremental gain of each component (Heuristic vs BC vs PPO vs MCTS).
  13. **Team Contribution Breakdown:** Individual responsibilities, technical modules owned, and personal contributions of each member.
  14. **Conclusion & Future Directions:** Summary of achievements, real-world deployment viability, and Q&A.

#### 2. Matplotlib Visualizations Script (`scripts/generate_viva_charts.py`)
- Claude Pro must write a standalone Python script using `matplotlib` and `seaborn` with `style='whitegrid'`, professional color palettes (slate blue, teal, coral), clear axis labels, and high DPI (300 DPI).
- Visuals to generate:
  - `fig1_wait_time_comparison.png`: Grouped bar chart comparing AWT across all 6 strategies in 4 traffic regimes.
  - `fig2_p95_service_guarantee.png`: P95 wait time distribution showing fairness and passenger anti-starvation.
  - `fig3_energy_efficiency.png`: Total energy consumption (kWh) across fleet.
  - `fig4_mcts_search_dynamics.png`: Search tree depth vs decision quality curve.
  - `fig5_learning_convergence.png`: PPO reward curve and KL divergence stability over training iterations.

#### 3. Comprehensive Study Material (`docs/STUDY_MATERIAL.md`)
- A comprehensive, structured study document covering:
  - **Problem Statement & Theory:** Formal definition of EGCP, mathematical formulation of objective functions (wait time minimization, energy optimization).
  - **Multi-Agent Systems:** Detailed explanation of FIPA-ACL, Contract Net Protocol sequence diagram, bid calculation mechanics.
  - **Machine Learning & RL Formulations:**
    - State space, action space, reward function formulation.
    - Set-Transformer mathematical intuition (permutation equivariance across elevator cars).
    - PPO clipped objective, generalized advantage estimation (GAE), KL penalty anchor.
    - PUCT-MCTS formula: $Q(s, a) + c_{puct} P(s, a) \frac{\sqrt{\sum N}}{1 + N(s, a)}$.
  - **Viva Q&A Cheat Sheet:** 20 critical questions examiners ask (e.g., "Why not use centralized RL?", "How do you guarantee elevator safety?", "What happens if a car loses communication?", "Why is DAgger needed after BC?").
  - **Live Demonstration Script:** Step-by-step walkthrough script for examiners using the React UI (`uv run elevator serve`) and live scenario switching.

#### 4. Frontend Enhancement & Polish
- Ensure the React UI is fully responsive, polished, and bug-free.
- Add real-time tooltips, clear legend indicators, and smooth state transitions.
- Verify Story Mode seamlessly walks through the project highlights.
- Run `pnpm build` in `frontend/` and commit the updated `src/elevator_mas/web/dist/`.

---

## 4. Team Presentation & Viva Distribution

| Member | Viva Presentation Responsibilities | Live UI Demo Segments |
| :--- | :--- | :--- |
| **Adhikkesh** (`CB.SC.U4CSE23101`) | • Slides 1–5: Introduction, EGCP problem statement, MAS theory, FIPA-ACL protocol & kinematics.<br>• Explains overall software architecture and Mesa 3 lifecycle. | • Launches server (`uv run elevator serve`).<br>• Demonstrates Mission Control Canvas, car physics, and FIPA-ACL message flow. |
| **Sisr Reddy** (`CB.SC.U4CSE23129`) | • Slides 6 & 10: Frontend architecture, real-time WebSocket protocol, UI design principles.<br>• Explains client-server state synchronization and 60 FPS rendering. | • Leads the Live Auction Theatre walkthrough.<br>• Demonstrates Compare Mode / Algorithm Lab side-by-side run.<br>• Guides Story / Present Mode for examiners. |
| **Kavin Karthic** (`CB.SC.U4CSE23161`) | • Slides 6–7: Feature extraction (48-feature schema), fast twin simulation physics, Behavioral Cloning & DAgger pipeline.<br>• Explains Set-Transformer architecture and ONNX deployment. | • Demonstrates `elevator learn` CLI commands.<br>• Shows model inference latency benchmarks and ONNX CPU execution. |
| **Akash** (`CB.SC.U4CSE23162`) | • Slides 8–9, 11–12: Cooperative PPO formulation, CTDE, PUCT-MCTS lookahead arbitration, empirical benchmark results & ablations.<br>• Defends mathematical formulations and experimental metrics. | • Demonstrates the Phase 6 React Brain Panel.<br>• Inspects live MCTS tree search rollouts and car value heatmaps.<br>• Presents benchmark summary plots. |

---

## 5. Quick Command Reference

```bash
# Setup & Environment
uv sync
cd frontend && pnpm install && cd ..

# Run Development Server (Backend + UI)
uv run elevator serve

# Run Headless Simulation
uv run elevator run --scenario morning_up_peak --strategy liftzero_mcts

# Run Automated Benchmark Suite
uv run elevator bench

# Run Learning CLI Suite
uv run elevator learn record --help
uv run elevator learn eval --help
uv run elevator learn benchmark-twin

# Testing & Quality Assurance
uv run pytest                         # Run all 250+ Python tests
cd frontend && pnpm test && cd ..     # Run React Vitest suite
uv run ruff check .                   # Lint check
uv run ruff format --check .          # Formatting check

# Build and Sync Frontend Production Bundle
cd frontend && pnpm build && cd ..
```

---

## 6. Handover Checklist for Claude Pro

Before starting:
- [ ] Read this entire `00_START_HERE.md`.
- [ ] Run `uv run pytest` and `uv run ruff check .` to verify Phase 1–3 stability.
- [ ] Confirm you are on branch `main` (`git status`).
- [ ] Open `prompt/Phase4_Transformer_Imitation_Prompt.md` and start Phase 4!
