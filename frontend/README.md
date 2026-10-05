# LiftZero Control Room Frontend

Modern web frontend for the **Multi-Agent Smart Elevator Fleet Coordinator**.

## Quick Start

Ensure [pnpm](https://pnpm.io/) is installed.

```bash
# Install dependencies
pnpm install

# Start Vite dev server on port 5173 with proxy to backend on port 8000
pnpm dev

# Run Vitest test suite with V8 code coverage
pnpm test

# Run Playwright E2E test suite (16 specs + accessibility checks)
pnpm e2e

# Build production bundle into src/elevator_mas/web/dist/
pnpm build
```

## Tech Stack

- **Framework**: React 18 + TypeScript 5.7 (strict mode)
- **Tooling**: Vite 6, Vitest, Playwright, Tailwind CSS v4, Biome/ESLint, Prettier
- **Visualization**: Lucide Icons, Recharts, @xyflow/react
- **State**: Zustand (coalesced 60fps live store), TanStack Query
- **Fonts**: Self-hosted Inter & JetBrains Mono via Fontsource (zero CDN dependencies)

## Test Coverage

- **Vitest Unit Tests**: 72 tests passing with >88% statement coverage across client, stores, formatters, and routing algorithms.
- **Algorithm Parity**: 20 deterministic Python-generated routing fixtures verified with 100% numerical parity for A* sequence, cost, and node expansions.
- **Playwright E2E**: 16 end-to-end tests validating live controls, car selection, fault injection, fire drills, auction visualization, message filtering, agent graphs, inspector drawers, step-through search lab, benchmark runners, compare mode, story mode, classic fallback, and WCAG 2.1 AA axe accessibility audits.
