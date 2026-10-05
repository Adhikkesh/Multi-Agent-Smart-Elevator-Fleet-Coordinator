# Phases 6–7 — status at submission

* **Phase 6 (PUCT-MCTS arbitration + React Brain panel): not implemented.** The design is in
  `prompt/Phase6_MCTS_Brain_Panel_Prompt.md` and summarised in `docs/STUDY_MATERIAL.md` §7 and
  deck slide 9. Prerequisites now exist: the ONNX runtime, the network's value head (distilled
  from the PPO critic once trained), the dispatcher hook point (`ElevatorModel.award_hook`). The
  twin needs fixing first (scenario events never fire; it reads `ev.type`/`ev.car_id` but
  `EventConfig` has `kind`/`car`) before it can serve as a search world model.
* **Phase 7 (full evaluation & release):** the evaluation pipeline is complete and was run for
  the classical and imitation strategies — 7,800 real-simulator test runs
  (`docs/lift/closed_loop.md`, `reports/lift/closed_loop_test_runs.csv`), offline evaluation
  and latency. Adding PPO rows needs only the Kaggle-trained model (`docs/lift/KAGGLE_TRAINING.md` §4).
