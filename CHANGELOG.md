# Changelog

## 2.0.0 — LiftZero (learning and search on top of the multi-agent system)

### Added
- **Phase 3** learning environment: 50-feature token schema (v2), expert decision recorder,
  expert dataset, twin simulator, Gymnasium environments, evaluation harness.
- **Phase 4** LiftZero-BC: permutation-equivariant set-Transformer (237,575 parameters),
  behaviour cloning with listwise/regression/rank/pairwise losses, two DAgger rounds, ONNX
  export with parity gate and model card, CPU runtime, strategies `liftzero_bc` and
  `liftzero_bc_cnp`, shadow teacher, offline and closed-loop evaluation (7,800 test runs).
- **Phase 5** cooperative PPO (CTDE critic, KL anchor to BC, SMDP-GAE) trained on the real
  simulator through a dispatcher award hook; `train-ppo` CLI and Kaggle guide; strategies
  `liftzero_ppo` / `liftzero_ppo_cnp` become available once the model is trained.
- **Phase 6** PUCT-MCTS look-ahead arbitration (`liftzero_bc_mcts`, `liftzero_mcts`), the Brain
  log and `/api/brain*` endpoints, occlusion explanations, and the **LiftZero Brain** page.
- **Phase 8** viva deck, study material, presentation plan and 300-DPI result charts.
- `elevator lift` CLI: train-bc, eval-offline, export, bench-infer, eval-sim, dagger, card,
  train-ppo.

### Fixed
- Fire recall during a move left a stale move timer (car "moving with doors open").
- Twin simulator never applied scenario events.
- `evaluate(backend="real")` crashed; `learn bench-env` always printed PASS.
- Expert harvest seeds could run into validation/test seed ranges.
- `elevator_mas.learning` imported Gymnasium eagerly, breaking a base install.

### Unchanged by design
- The four classical strategies are bit-identical to 1.0.0 (golden-metrics test).
