"""Regenerate `tests/learning/fixtures/expert_sample.npz` (1 000 expert decisions).

Deterministic: harvests training-split runs (seeds 0, 100, 200, ...) with the mixed teacher
until 1 000 decisions are collected, then truncates. Re-run after any feature-schema change:

    uv run python scripts/generate_learning_fixture.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from elevator_mas.learning.recorder import DecisionRecorder, harvest_run, sample_random_regime

N_DECISIONS = 1000
OUT = (
    Path(__file__).resolve().parent.parent / "tests" / "learning" / "fixtures" / "expert_sample.npz"
)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        recorder = DecisionRecorder(tmp, shard_capacity=10**9, teacher="mixed")
        run = 0
        while recorder.buffered < N_DECISIONS:
            harvest_run(sample_random_regime(run, run * 100, "mixed"), recorder)
            run += 1
        shard = recorder.flush()
        assert shard is not None
        with np.load(shard) as data:
            arrays = {k: data[k][:N_DECISIONS] for k in data.files if k != "meta_json"}
            meta = json.loads(str(data["meta_json"]))
    meta["count"] = N_DECISIONS
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, meta_json=json.dumps(meta), **arrays)
    print(f"wrote {N_DECISIONS} decisions from {run} runs to {OUT}")


if __name__ == "__main__":
    main()
