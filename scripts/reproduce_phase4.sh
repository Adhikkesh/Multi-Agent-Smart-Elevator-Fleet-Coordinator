#!/usr/bin/env bash
# Phase 4 smoke pipeline, end to end in ~10-15 minutes on a laptop:
#   record 20k expert decisions -> train the smoke preset -> export ONNX -> closed loop.
# Everything goes to $OUT (default: a temp dir); the shipped model is never touched.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${OUT:-$(mktemp -d -t liftzero-phase4-XXXX)}"
WORKERS="${WORKERS:-4}"
echo "== LiftZero Phase 4 smoke pipeline -> $OUT"

uv sync --group learn --group lift-train

echo "== 1/5 record expert decisions (train / val)"
uv run elevator learn record --decisions 20000 --workers "$WORKERS" --out "$OUT/expert/train" --seed-start 0
uv run elevator learn record --decisions 6000 --workers "$WORKERS" --out "$OUT/expert/val" --seed-start 8000

echo "== 2/5 train (smoke preset)"
cat > "$OUT/bc_smoke.yaml" <<YAML
name: bc_smoke_repro
data_train: [$OUT/expert/train]
data_val: [$OUT/expert/val]
max_train_decisions: 50000
epochs: 3
warmup_steps: 100
seed: 0
YAML
uv run elevator lift train-bc --config "$OUT/bc_smoke.yaml" --out "$OUT/runs"
CKPT="$(ls -d "$OUT"/runs/bc_smoke_repro_s0_*/ | tail -1)best.pt"

echo "== 3/5 export ONNX + model card"
uv run elevator lift export --ckpt "$CKPT" --out "$OUT/liftzero_smoke.onnx" --version smoke
uv run elevator lift card --model "$OUT/liftzero_smoke.onnx" | tail -1

echo "== 4/5 inference latency"
uv run elevator lift bench-infer --model "$OUT/liftzero_smoke.onnx" --reps 500

echo "== 5/5 closed loop (5 val seeds x 4 regimes, real simulator)"
uv run elevator lift eval-sim --strategy liftzero_bc --strategy full --regimes regimes \
  --seeds val --n 5 --workers "$WORKERS" --model "$OUT/liftzero_smoke.onnx" \
  --out "$OUT/closed_loop.md" --csv "$OUT/closed_loop.csv"

echo "== done: see $OUT/closed_loop.md"
