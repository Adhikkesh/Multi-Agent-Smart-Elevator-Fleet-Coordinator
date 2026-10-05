# Phases 6–7 — status at submission

## Phase 6 — look-ahead and the Brain panel (implemented, case-study scale)

* `learning/lift/search/`: a world model built from the dispatcher's public view (hidden
  destinations sampled per simulation, common random numbers), anytime **PUCT-MCTS**
  (network priors, chance nodes on the next call, cost-greedy rollouts or value leaves,
  depth ≤ 3, horizon 60 s, 50 ms budget), the dispatcher **arbiter** (searches only contested
  calls, only among cars that proposed), and the **Brain log**.
* Strategies: `liftzero_bc_mcts` (imitation + look-ahead) and `liftzero_mcts` (PPO + value
  leaves; available once the PPO model exists).
* API: `GET /api/brain`, `GET /api/brain/decisions`, `POST /api/brain/config`,
  `POST /api/brain/explain` (occlusion saliency over 9 feature groups),
  `GET /api/brain/model-card`.
* UI: **LiftZero Brain** page (`/brain`) — strategy switch, live stats, search budget sliders,
  per-decision bids / attention / visits / Q, and the "Why?" explanation.
* Measured on `lunch_two_way`: ~22 simulations to depth 3 per searched call within 50 ms;
  ~70 % of auctions searched, ~15–20 % overridden; 0 safety violations.
* Also fixed: the twin's scenario events never fired (it read `ev.type`/`ev.car_id`).

### Closed-loop result (20 paired test seeds per regime, `docs/lift/mcts_closed_loop.md`)

| regime | LiftZero-BC | + look-ahead | Δ | Wilcoxon p |
| --- | --- | --- | --- | --- |
| up_peak | 36.3 s | 31.9 s | −12.0 % | 0.12 |
| down_peak | 24.3 s | 23.5 s | −3.4 % | 0.81 |
| two_way | 14.2 s | 13.9 s | −1.9 % | 0.65 |
| interfloor | 10.8 s | 10.7 s | −0.8 % | 0.93 |

Lower average wait in all four regimes, but **not statistically significant** with 20 seeds;
compute rises from 1.2 to 9.6 ms per simulated second; 0 safety violations. Figure:
`reports/viva/fig10_mcts_lookahead.png`.


## Phase 7 — evaluation

The evaluation pipeline was run for the classical and imitation strategies: 7,800
real-simulator test runs (`docs/lift/closed_loop.md`), offline evaluation and latency. Not
done: PPO (not trained) and a 100-seed look-ahead evaluation (20 seeds above). Release:
version 2.0.0, `CHANGELOG.md`.
