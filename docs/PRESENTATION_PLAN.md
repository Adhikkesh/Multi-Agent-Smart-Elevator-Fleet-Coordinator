# Case-study review — presentation plan

**Deck:** [`docs/LiftZero_Viva_Deck.pptx`](LiftZero_Viva_Deck.pptx) — 18 slides, white
background, structured on the review rubric (each slide carries its rubric tag, e.g.
"PEAS · 3M"). Speaker notes in every slide name the presenter and what to say.
Charts are real results; screenshots are of the running dashboard.

| # | Slide | Rubric item | Presenter |
| --- | --- | --- | --- |
| 1 | Title, team, register numbers | — | Adhikkesh |
| 2 | Problem statement: elevator group control | Review 1 | Adhikkesh |
| 3 | Why the obvious answer fails (measured) | Review 1 | Adhikkesh |
| 4 | PEAS — the whole system | PEAS · 3M | Adhikkesh |
| 5 | PEAS of every agent | PEAS · 3M | Adhikkesh |
| 6 | Environment analysis (7 properties) | Env & Agent · 3M | Akash |
| 7 | Why each agent is its type | Env & Agent · 3M | Akash |
| 8 | Modelling car routing as state-space search (heuristic, admissibility) | Search · 3M | Akash |
| 9 | BFS vs UCS vs Greedy vs A\* (200 problems) | Search · 3M | Akash |
| 10 | Simulated annealing, minimax + alpha-beta, forward chaining | Search · 3M | Akash |
| 11 | Tool and package selection, setup | Tools · 3M | Sisr Reddy |
| 12 | Multi-agent execution: FIPA-ACL + Contract Net (+ Agents page) | MAS · 3M | Adhikkesh |
| 13 | Live demo: Mission Control | Demo · 3M | Sisr Reddy |
| 14 | Testing scenarios and automated tests | Testing · 3M | Kavin Karthic |
| 15 | Results across traffic patterns | Testing · 3M | Akash |
| 16 | Code structure and scalability | Code · 1M | Sisr Reddy |
| 17 | Extension: LiftZero learned bidder | — | Kavin Karthic |
| 18 | Conclusion, questions | Q&A · 1M | Kavin Karthic |

**Timing:** about 1 minute per slide (≈ 18 min) + the 7-minute live demo
([`DEMO_SCRIPT.md`](DEMO_SCRIPT.md)) + questions ([`STUDY_MATERIAL.md`](STUDY_MATERIAL.md) §12).

**Rebuild after changes:**

```bash
uv run python scripts/search_benchmark.py          # slide 9 numbers
uv run python scripts/generate_viva_charts.py      # charts in reports/viva/
uv run elevator serve --port 8765 &                # then: node scripts/capture_ui.mjs (screenshots)
npm install pptxgenjs && node scripts/build_viva_deck.js
```
