# Training LiftZero-PPO on Kaggle (Phase 5)

The PPO code is complete and smoke-tested locally (`elevator lift train-ppo --preset smoke`
runs collection → SMDP-GAE → PPO update → real-simulator validation → checkpoint in ~30 s).
Full training was not run on the laptop; run it on Kaggle as below.

**Important:** rollouts run the *real* multi-agent simulator, which is CPU-bound. The GPU
only speeds up the PPO update (small). Kaggle notebooks have 4 CPU cores — use `--workers 3`.
Expect ~600–900 decisions/s, i.e. the default 2 M-decision budget takes ~45–60 min per seed.

## 1. Notebook setup (first cell)

```bash
!git clone https://github.com/Adhikkesh/Multi-Agent-Smart-Elevator-Fleet-Coordinator.git lz
%cd lz
!pip install -q uv
!uv sync --group learn --group lift-train
!uv run pytest tests/lift/rl -q          # sanity: 12 tests
```

## 2. Smoke run (~1 min) — check it works on Kaggle

```bash
!uv run elevator lift train-ppo --preset smoke --workers 3
```

## 3. Training (one cell per seed; each ~45–60 min)

```bash
!uv run elevator lift train-ppo --preset default --seed 0 --workers 3
!uv run elevator lift train-ppo --preset default --seed 1 --workers 3   # optional
!uv run elevator lift train-ppo --preset default --seed 2 --workers 3   # optional
```

Each run writes `runs/ppo_default_s<seed>_<time>/` with `metrics.csv` (per update: return,
true avg wait of training episodes, policy/value loss, entropy, approx-KL, KL to BC, explained
variance, β), `evals.csv` (validation closed-loop vs the `full` teacher), `last.pt`
(resumable: `--resume runs/.../last.pt`) and `best.pt` (best *eligible* validation score —
p95/long-wait guard ±10 % vs the imitation policy).

If the session is about to time out, the run can be resumed in a new session with
`--resume`. A budget override is available: `--decisions 1000000`.

## 4. Export and evaluate (pick the seed with the best `best_score_pct` in run.json)

```bash
!uv run elevator lift export --ckpt runs/ppo_default_s0_*/best.pt --out models/liftzero_ppo_v1.onnx
!uv run elevator lift card --model models/liftzero_ppo_v1.onnx --markdown docs/lift/MODEL_CARD_PPO.md
!uv run elevator lift eval-sim --strategy liftzero_ppo --strategy full --strategy collective \
   --strategy cnp_astar --strategy liftzero_bc --regimes regimes --seeds test --n 100 --workers 3 \
   --out docs/lift/ppo_closed_loop.md --csv reports/lift/ppo_closed_loop.csv
!uv run python scripts/generate_viva_charts.py     # adds the PPO learning-curve figure
```

## 5. Bring the results back

Download `models/liftzero_ppo_v1.onnx`, `models/liftzero_ppo_v1.json`,
`runs/ppo_default_s*/{metrics.csv,evals.csv,run.json}`, `docs/lift/ppo_closed_loop.md`,
`reports/lift/ppo_closed_loop.csv` and `reports/viva/*.png`; copy them into the repo at the
same paths and commit. The `liftzero_ppo` strategy becomes available automatically once
`models/liftzero_ppo_v1.onnx` exists (the dashboard lists it).
