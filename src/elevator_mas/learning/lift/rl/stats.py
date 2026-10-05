"""Paired statistics for strategy comparisons over shared seeds.

For a metric measured on the same seeds under strategies A and B, with paired differences
``d_s = A_s − B_s``: mean, percentile-bootstrap 95 % CI, Wilcoxon signed-rank p-value
(two-sided), win-rate (fraction of seeds where A is better, i.e. lower), Cohen's ``d_z``
(mean / sd of the differences), and Holm–Bonferroni adjustment across several claims.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


def bootstrap_ci(
    x: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile-bootstrap CI of the mean."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        v = float(x[0]) if len(x) else float("nan")
        return v, v
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return float(np.percentile(means, 100 * alpha / 2)), float(
        np.percentile(means, 100 * (1 - alpha / 2))
    )


def wilcoxon_p(d: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value (1.0 when every difference is zero)."""
    from scipy.stats import wilcoxon

    d = np.asarray(d, dtype=float)
    if np.allclose(d, 0.0):
        return 1.0
    return float(wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)


def holm(pvalues: list[float]) -> list[float]:
    """Holm–Bonferroni adjusted p-values (same order as the input)."""
    m = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[idx]))
        adjusted[idx] = running
    return adjusted.tolist()


@dataclass
class PairedResult:
    n: int
    mean_a: float
    mean_b: float
    mean_diff: float
    rel_pct: float
    ci_low: float
    ci_high: float
    p_wilcoxon: float
    win_rate: float
    cohens_dz: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def paired(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> PairedResult:
    """Paired comparison of A against B (lower is better for wait metrics)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = a - b
    lo, hi = bootstrap_ci(d, n_boot=n_boot, seed=seed)
    sd = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
    mb = float(b.mean())
    return PairedResult(
        n=len(d),
        mean_a=float(a.mean()),
        mean_b=mb,
        mean_diff=float(d.mean()),
        rel_pct=(float(a.mean()) / mb - 1.0) * 100.0 if mb else float("nan"),
        ci_low=lo,
        ci_high=hi,
        p_wilcoxon=wilcoxon_p(d),
        win_rate=float(np.mean(d < 0)),
        cohens_dz=float(d.mean() / sd) if sd > 0 else 0.0,
    )
