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

## Phase 7 — evaluation

The evaluation pipeline was run for the classical and imitation strategies: 7,800
real-simulator test runs (`docs/lift/closed_loop.md`), offline evaluation and latency. Not
done: a 100-seed evaluation of `liftzero_bc_mcts` and of PPO (not trained). Run with
`uv run elevator lift eval-sim --strategy liftzero_bc_mcts --strategy liftzero_bc --strategy full --n 20`.
