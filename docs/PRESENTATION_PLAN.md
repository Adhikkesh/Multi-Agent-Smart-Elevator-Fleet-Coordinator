# Phase 8 — Viva presentation plan

**Deck:** [`docs/LiftZero_Viva_Deck.pptx`](LiftZero_Viva_Deck.pptx) — 14 slides, white
background, Cambria headings / Calibri body, navy + teal palette, no clip-art. Every chart is a
real result rendered by `scripts/generate_viva_charts.py` (300 DPI, `reports/viva/`). Speaker
notes in the deck name the presenter of each slide. Rebuild after new results:

```bash
uv run python scripts/generate_viva_charts.py       # figures from reports/ and docs/lift/
npm install pptxgenjs && node scripts/build_viva_deck.js
```

| # | Slide | Presenter | Visual |
| --- | --- | --- | --- |
| 1 | Title, team and register numbers | Adhikkesh | team cards |
| 2 | The Elevator Group Control Problem | Adhikkesh | stat callouts (N^k, 1 s cycle) |
| 3 | Why classic dispatchers fail | Adhikkesh | native bar chart, up-peak waits |
| 4 | Multi-agent architecture + Contract Net | Adhikkesh | agent cards + 4-step CNP flow |
| 5 | Physical model, search and safety | Adhikkesh | three-column cards |
| 6 | Feature engineering & learning environment | Kavin Karthic | token cards + data stats |
| 7 | LiftZero-BC: set-Transformer, BC + DAgger | Kavin Karthic | `fig5_imitation_learning_curves.png` |
| 8 | LiftZero-PPO: cooperative RL (CTDE) | Akash | loss, process steps, honest status |
| 9 | PUCT-MCTS look-ahead (design) | Akash | formula, reflex vs deliberative |
| 10 | Mission Control dashboard | Sisr Reddy | six feature cards (then live demo) |
| 11 | Results: average wait | Akash | `fig1_wait_time_comparison.png` |
| 12 | Learned vs teacher (ablation) | Akash | `fig4_learned_vs_teacher.png` |
| 13 | Team contributions | all | four member cards |
| 14 | Conclusion & next steps | Adhikkesh | stat callouts, Q&A |

Additional figures for backup slides / questions: `fig2_p95_service_guarantee.png`,
`fig3_energy_efficiency.png`, `fig6_dagger_rounds.png`, `fig7_offline_agreement.png`,
`fig8_inference_latency.png`; after Kaggle PPO training, `fig9_ppo_learning_curves.png`.

Timing: ~1.5 min per slide (≈ 20 min) + 6 min live demo (script in
[`STUDY_MATERIAL.md`](STUDY_MATERIAL.md) §10).
