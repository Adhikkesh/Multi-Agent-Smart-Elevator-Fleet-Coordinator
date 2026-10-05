"""ONNX export parity, dynamic shapes, the model card contract, and the CPU runtime."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from lift_helpers import numpy_batch

from elevator_mas.learning.lift.bidder import StrategyUnavailableError, get_runtime
from elevator_mas.learning.lift.card import ModelCardError, card_path_for, load_card
from elevator_mas.learning.lift.model import INELIGIBLE_SCORE
from elevator_mas.learning.lift.runtime import LiftRuntime, bench_latency


def _torch_outputs(net: Any, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
    torch = pytest.importorskip("torch")
    with torch.no_grad():
        o = net(*(torch.from_numpy(feeds[k]) for k in ("call", "cars", "glob", "mask", "eligible")))
    return [o.score.numpy(), o.aux.numpy(), o.value.numpy(), o.attn.numpy()]


def _assert_parity(got: Any, ref: list[np.ndarray], feeds: dict[str, np.ndarray]) -> None:
    valid = feeds["mask"] & feeds["eligible"]
    mask = feeds["mask"]
    assert np.abs(got.score - ref[0])[valid].max() < 1e-4
    assert np.abs(got.aux - ref[1])[mask].max() < 1e-4
    assert np.abs(got.value - ref[2]).max() < 1e-4
    assert np.abs(got.attn - ref[3])[mask].max() < 1e-4
    # Sentinel scores agree to float32 precision and stay far above every real score.
    np.testing.assert_allclose(got.score[~valid], ref[0][~valid], rtol=1e-6)
    assert np.all(got.score[~valid] > INELIGIBLE_SCORE / 2)


@pytest.mark.parametrize("n", [1, 2, 8, 16, 32])
def test_parity_on_edge_shapes(tiny_export: tuple[Any, Path], n: int) -> None:
    net, path = tiny_export
    feeds = numpy_batch(16, n, seed=n)
    _assert_parity(LiftRuntime(path).score(feeds, trim=False), _torch_outputs(net, feeds), feeds)


def test_parity_on_1000_real_decisions(
    tiny_export: tuple[Any, Path], fixture_decisions: Any
) -> None:
    net, path = tiny_export
    d = fixture_decisions
    feeds = {"call": d.call, "cars": d.cars, "glob": d.glob, "mask": d.mask, "eligible": d.eligible}
    assert len(d) == 1000
    _assert_parity(LiftRuntime(path).score(feeds, trim=False), _torch_outputs(net, feeds), feeds)


def test_trimming_padding_does_not_change_scores(shipped_model: Path) -> None:
    rt = LiftRuntime(shipped_model)
    feeds = numpy_batch(8, 16, seed=3)
    feeds["mask"][:, 6:] = False
    feeds["eligible"][:, 6:] = False
    a, b = rt.score(feeds, trim=True), rt.score(feeds, trim=False)
    np.testing.assert_allclose(a.score[:, :6], b.score[:, :6], atol=1e-5)
    assert a.score.shape == b.score.shape == (8, 16)


def test_choose_respects_eligibility(shipped_model: Path) -> None:
    rt = LiftRuntime(shipped_model)
    feeds = numpy_batch(64, 6, seed=4)
    choice = rt.choose(feeds)
    valid = feeds["mask"] & feeds["eligible"]
    assert np.all(valid[np.arange(64), choice])


def test_card_matches_the_shipped_model(shipped_model: Path) -> None:
    card = load_card(shipped_model)
    assert 200_000 <= card.param_count <= 400_000
    assert card.inputs == ["call", "cars", "glob", "mask", "eligible"]
    assert card.outputs == ["score", "aux", "value", "attn"]
    assert card.metrics["onnx_parity_max_abs"] < 1e-4
    assert card.limitations


def _copy_model(src: Path, dst_dir: Path) -> Path:
    dst = dst_dir / src.name
    dst.write_bytes(src.read_bytes())
    card_path_for(dst).write_text(card_path_for(src).read_text())
    return dst


def test_schema_mismatch_refuses_to_load(shipped_model: Path, tmp_path: Path) -> None:
    m = _copy_model(shipped_model, tmp_path)
    data = json.loads(card_path_for(m).read_text())
    data["schema_hash"] = "0000000000000000"
    card_path_for(m).write_text(json.dumps(data))
    with pytest.raises(ModelCardError, match="schema"):
        LiftRuntime(m)


def test_tampered_model_file_is_refused(shipped_model: Path, tmp_path: Path) -> None:
    m = _copy_model(shipped_model, tmp_path)
    m.write_bytes(m.read_bytes() + b"\0")
    with pytest.raises(ModelCardError, match="sha256"):
        LiftRuntime(m)


def test_missing_model_or_card(tmp_path: Path) -> None:
    with pytest.raises(ModelCardError):
        LiftRuntime(tmp_path / "nope.onnx")
    (tmp_path / "x.onnx").write_bytes(b"x")
    with pytest.raises(ModelCardError, match="card"):
        LiftRuntime(tmp_path / "x.onnx")
    with pytest.raises(StrategyUnavailableError):
        get_runtime(str(tmp_path / "nope.onnx"))


def test_latency_benchmark_shape(shipped_model: Path) -> None:
    rows = bench_latency(LiftRuntime(shipped_model), sizes=(2, 8), reps=50)
    assert [r["n"] for r in rows] == [2, 8]
    assert all(0 < r["median_ms"] <= r["p99_ms"] for r in rows)


def test_runtime_path_never_imports_torch_or_onnx(shipped_model: Path) -> None:
    code = (
        "import sys\n"
        "from elevator_mas.learning.lift.runtime import LiftRuntime\n"
        "from elevator_mas.learning.lift import bidder\n"
        "import elevator_mas.cli\n"
        f"LiftRuntime({str(shipped_model)!r}).score({{'call': [[0.0]*14], "
        "'cars': [[[0.0]*26]], 'glob': [[0.0]*10], 'mask': [[True]], 'eligible': [[True]]})\n"
        "bad = [m for m in ('torch', 'onnx', 'gymnasium') if m in sys.modules]\n"
        "assert not bad, bad\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
