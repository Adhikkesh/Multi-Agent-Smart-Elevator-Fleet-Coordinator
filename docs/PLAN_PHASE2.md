# Phase 2 — LiftZero Control Room: Implementation Plan

## 1. Backend Additions (`src/elevator_mas/api/`)
- **`src/elevator_mas/api/schemas.py`**:
  - Add `RunRequest` schema: `scenario: str`, `strategy: str`, `seed: int`, `ticks: int = 600 (ge=60, le=3600)`, `sample_every: int = 10 (ge=1, le=60)`.
- **`src/elevator_mas/api/server.py`**:
  - Update `GET /`: serve `web/dist/index.html` if it exists, fallback to `web/index.html`.
  - Add `GET /classic`: serve `web/index.html`.
  - Mount `/assets` to `web/dist/assets` if present.
  - Add SPA fallback handler: any non-API, non-WS, non-static, non-assets path returns `web/dist/index.html` (or `web/index.html`).
  - Add `POST /api/run`: headless simulation with periodic sampling of `Metrics`, returning `{scenario, strategy, seed, ticks, runtime_s, final, series, rules_fired, violations}`.
  - Add `GET /api/board`: return `session.model.board.as_dict()`.
  - Add `GET /api/version`: return `{"app": "2.0.0-dev", "git": <commit-sha>}`.
- **`tests/test_api.py` / `tests/test_web.py`**:
  - Tests for `/classic`, SPA fallback, `/api/run` (determinism, error validation, metrics series), `/api/board`, `/api/version`.

## 2. Frontend Architecture (`frontend/`)
- **Tooling**: Vite + React + TypeScript (strict mode) + Tailwind CSS + Lucide React + Recharts + @xyflow/react + Zustand + TanStack Query + React Router.
- **Offline Fonts**: `@fontsource/inter` and `@fontsource/jetbrains-mono`.
- **Store Shape (`Zustand`)**:
  - `liveStore`: `snapshot`, `connected`, `metricsHistory` (ring buffer <= 600), `recentMessages` (ring buffer <= 200), `auctionHistory` (ring buffer <= 100), `brain`.
  - `uiStore`: `theme` ('dark' | 'light'), `selectedCarId`, `selectedFloor`, `selectedConversationId`, `inspectorAgentAddress`, `inspectorOpen`, `shortcutsOpen`, `presentMode`.
  - `storyStore`: `activeBeatIndex`, `beatStatus`, `isAutoPlaying`.
- **WebSocket Streaming**:
  - `src/live/socket.ts` with exponential backoff auto-reconnect, visibility change pause, and `requestAnimationFrame` store update coalescing.

## 3. Pages & Features
1. **Mission Control (`/`)**:
   - Central Shaft/Building view: car glide interpolation, doors, load bars, route chips, waiting crowd pips, hall lanterns, ETA badges.
   - Live KPI Strip: sparklines + deltas for avg wait, p95, throughput, energy, etc.
   - Auction Panel: stacked utility bars per car, refused rows, winner badge, live decision trace quote, auction history selector.
   - Message Stream: virtualized, performative badges, conversation ID grouping, filter chips, payload modal.
   - Traffic Monitor card: inferred vs true pattern, weights bars, ghost parking targets.
   - Safety Agent card: R1..R7 timeline, fire alarm alert banner, invariant status.
   - Chaos Panel: car fault, repair, fire alarm, clear alarm, floor rush injection.
   - Brain Panel: Phase 6 placeholder ("LiftZero model not loaded — running classical Contract-Net bidder").
   - Footer: scenario progress with phase segments for `demo_story`.
2. **Agents (`/agents`)**:
   - Graph view (@xyflow/react): dispatcher, cars, monitor, safety, floors; animated message pulses along edges.
   - Sequence Diagram: SVG swimlane view for the selected `conversation_id`.
   - Agent Inspector: slide-out drawer with PEAS table, AIMA agent type, live polled state.
   - Shared Status Board: table from `GET /api/board` showing public blackboard state.
   - Full searchable/filterable message table.
3. **Algorithm Lab (`/lab`)**:
   - Tab 1: Routing Search (`/api/search-lab`) with BFS/UCS/Greedy/A* comparisons + client-side step-through A* visualizer.
   - Tab 2: Assignment Optimization (`/api/search-lab/annealing`) with SA vs HC convergence curves.
   - Tab 3: Adversarial Parking (`/api/search-lab/minimax`) with minimax vs alpha-beta node comparisons and floor layout.
4. **Experiments (`/experiments`)**:
   - Benchmark Runner (`POST /api/benchmark`): scenario/strategy/seeds selector, summary + comparison tables (losses in red), CSV and PNG export.
   - Compare Mode (`/experiments/compare`): parallel headless runs via `POST /api/run`, multi-line synchronized time-series scrubbers.
5. **Theory (`/theory`)**:
   - PEAS 4-column cards for system and all 6 agents.
   - Environment classification table.
   - Agent-type ladder and multi-agent rationale.
   - Algorithm explainer cards linked to Lab tabs.
   - FIPA-ACL Contract Net protocol diagram.
   - Viva Q&A accordion (12 questions).
6. **Story / Presentation Mode (`/story` or `Present` button)**:
   - Full-screen minimal-chrome walkthrough over `demo_story` across ~10 beats with API actions and state-driven progress.

## 4. Testing Plan
- **Python**: `uv run pytest` passing 201 existing + new endpoints tests; `uv run ruff check .` and `uv run ruff format --check .` 100% clean.
- **Vitest (`pnpm test`)**: >= 60 unit tests covering:
  - Store ring buffers, reducers, rAF coalescer.
  - Message filtering and conversation grouping.
  - Color palettes and tabular formatting helpers.
  - Car position and progress interpolation.
  - Sequence diagram event extraction.
  - Client-side A* parity assertion against 20 server routing fixtures.
  - Component unit tests: `KpiTile`, `BidBars`, `Building`, `MessageChip`.
- **Playwright E2E (`pnpm e2e`)**:
  - Full specs for controls, building animation, injections, auction inspection, agents graph, lab tabs, benchmark, compare mode, story mode, `/classic` fallback, axe accessibility tests.
