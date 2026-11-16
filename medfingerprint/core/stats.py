"""Streaming / mergeable statistics.

* :class:`Moments` holds count, mean and central moment sums (M2..M4) and merges
  exactly across chunks, cases and datasets (Pébay 2008). Fields may be numpy
  arrays, giving vectorised per-label moments.
* Histogram helpers compute exact quantiles for integer data binned at width 1,
  and within-bin-uniform interpolated quantiles otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import HistogramSpec

PERCENTILES = (0.5, 1, 5, 25, 50, 75, 95, 99, 99.5)


def pkey(q: float) -> str:
    """Percentile key, e.g. 0.5 -> 'p00_5', 99.5 -> 'p99_5', 50 -> 'p50'."""
    if float(q).is_integer():
        return f"p{int(q):02d}"
    whole, frac = str(float(q)).split(".")
    return f"p{int(whole):02d}_{frac}"


@dataclass
class Moments:
    n: np.ndarray
    mean: np.ndarray
    M2: np.ndarray
    M3: np.ndarray
    M4: np.ndarray

    @classmethod
    def empty(cls, shape=()) -> "Moments":
        z = lambda: np.zeros(shape, dtype=np.float64)
        return cls(z(), z(), z(), z(), z())

    @classmethod
    def from_values(cls, x: np.ndarray) -> "Moments":
        x = np.asarray(x, dtype=np.float64).ravel()
        n = float(x.size)
        if n == 0:
            return cls.empty()
        mean = x.mean()
        d = x - mean
        d2 = d * d
        return cls(np.float64(n), np.float64(mean), d2.sum(), (d2 * d).sum(), (d2 * d2).sum())

    @classmethod
    def from_labels(cls, x: np.ndarray, labels: np.ndarray, nlabels: int) -> "Moments":
        """Per-label moments in one pass over a chunk (labels in [0, nlabels))."""
        x = np.asarray(x, dtype=np.float64).ravel()
        labels = labels.ravel()
        n = np.bincount(labels, minlength=nlabels).astype(np.float64)
        s = np.bincount(labels, weights=x, minlength=nlabels)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(n > 0, s / np.maximum(n, 1), 0.0)
        d = x - mean[labels]
        d2 = d * d
        M2 = np.bincount(labels, weights=d2, minlength=nlabels)
        M3 = np.bincount(labels, weights=d2 * d, minlength=nlabels)
        M4 = np.bincount(labels, weights=d2 * d2, minlength=nlabels)
        return cls(n, mean, M2, M3, M4)

    def merge(self, o: "Moments") -> "Moments":
        na, nb = self.n, o.n
        n = na + nb
        safe = np.where(n > 0, n, 1.0)
        delta = o.mean - self.mean
        mean = np.where(n > 0, self.mean + delta * nb / safe, 0.0)
        M2 = self.M2 + o.M2 + delta ** 2 * na * nb / safe
        M3 = (self.M3 + o.M3 + delta ** 3 * na * nb * (na - nb) / safe ** 2
              + 3.0 * delta * (na * o.M2 - nb * self.M2) / safe)
        M4 = (self.M4 + o.M4
              + delta ** 4 * na * nb * (na * na - na * nb + nb * nb) / safe ** 3
              + 6.0 * delta ** 2 * (na * na * o.M2 + nb * nb * self.M2) / safe ** 2
              + 4.0 * delta * (na * o.M3 - nb * self.M3) / safe)
        return Moments(n, mean, M2, M3, M4)

    def summary(self) -> dict:
        """mean, std (population, like np.std), skewness and excess kurtosis."""
        n = self.n
        with np.errstate(invalid="ignore", divide="ignore"):
            var = np.where(n > 0, self.M2 / np.where(n > 0, n, 1), np.nan)
            std = np.sqrt(var)
            skew = np.where(self.M2 > 0, np.sqrt(n) * self.M3 / self.M2 ** 1.5, np.nan)
            kurt = np.where(self.M2 > 0, n * self.M4 / self.M2 ** 2 - 3.0, np.nan)
            mean = np.where(n > 0, self.mean, np.nan)
        return {"count": n, "mean": mean, "std": std, "skewness": skew, "kurtosis": kurt}

    def to_list(self) -> list:
        return [float(self.n), float(self.mean), float(self.M2), float(self.M3), float(self.M4)]

    @classmethod
    def from_list(cls, v) -> "Moments":
        return cls(*(np.float64(a) for a in v))

    def item(self, i: int) -> "Moments":
        return Moments(self.n[i], self.mean[i], self.M2[i], self.M3[i], self.M4[i])


def hist_value_at_rank(counts: np.ndarray, spec: HistogramSpec, ranks: np.ndarray,
                       vmin: float, vmax: float, integer_data: bool) -> np.ndarray:
    """Value of the order statistic(s) at 0-based ``ranks`` from a counts array."""
    cum = np.cumsum(counts)
    b = np.searchsorted(cum, ranks, side="right")
    b = np.clip(b, 0, len(counts) - 1)
    before = np.where(b > 0, cum[np.maximum(b - 1, 0)], 0)
    left = spec.lo + (b - 1) * spec.width
    if integer_data and spec.width == 1:
        val = left.astype(np.float64)
    else:
        cnt = np.maximum(counts[b], 1)
        val = left + (ranks - before + 0.5) / cnt * spec.width
    val = np.where(b == 0, vmin, val)
    val = np.where(b == len(counts) - 1, vmax, val)
    return np.clip(val, vmin, vmax)


def hist_percentiles(counts: np.ndarray, spec: HistogramSpec, qs=PERCENTILES,
                     vmin: float = -np.inf, vmax: float = np.inf,
                     integer_data: bool = True) -> dict:
    """Percentiles matching ``np.percentile(..., method='linear')`` for integer data."""
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return {pkey(q): float("nan") for q in qs}
    qs = np.asarray(qs, dtype=np.float64)
    pos = qs / 100.0 * (total - 1)
    k = np.floor(pos)
    frac = pos - k
    lo_v = hist_value_at_rank(counts, spec, k, vmin, vmax, integer_data)
    hi_v = hist_value_at_rank(counts, spec, np.minimum(k + 1, total - 1), vmin, vmax, integer_data)
    vals = lo_v + frac * (hi_v - lo_v)
    return {pkey(q): float(v) for q, v in zip(qs, vals)}


def hist_entropy_bits(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    tot = counts.sum()
    if tot <= 0:
        return float("nan")
    p = counts[counts > 0] / tot
    return float(-(p * np.log2(p)).sum())


def describe_from(m: Moments, counts: np.ndarray, spec: HistogramSpec,
                  vmin: float, vmax: float, integer_data: bool) -> dict:
    """Full descriptive statistics for a region given its moments + histogram."""
    s = {k: float(v) for k, v in m.summary().items()}
    s["count"] = int(round(s["count"]))
    s["min"] = float(vmin) if s["count"] else float("nan")
    s["max"] = float(vmax) if s["count"] else float("nan")
    s.update(hist_percentiles(counts, spec, vmin=vmin, vmax=vmax, integer_data=integer_data))
    s["iqr"] = s["p75"] - s["p25"]
    s["entropy_bits"] = hist_entropy_bits(counts)
    tot = float(np.sum(counts))
    s["frac_below_range"] = float(counts[0] / tot) if tot else float("nan")
    s["frac_above_range"] = float(counts[-1] / tot) if tot else float("nan")
    return s


def summarize(values) -> dict:
    """Summary of a per-case feature across cases (NaNs ignored)."""
    v = np.asarray([x for x in values if x is not None], dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    p5, p25, p50, p75, p95 = np.percentile(v, [5, 25, 50, 75, 95])
    return {
        "n": int(v.size), "mean": float(v.mean()), "std": float(v.std()),
        "min": float(v.min()), "p05": float(p5), "p25": float(p25), "median": float(p50),
        "p75": float(p75), "p95": float(p95), "max": float(v.max()),
    }
