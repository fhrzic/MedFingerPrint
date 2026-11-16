"""User-defined and preset intensity bins, applied after the scan to the fine
histograms (no image re-reading needed).

Allocation rule: counts are spread uniformly within each fine bin, so bin edges
that fall on fine-bin edges (e.g. integer HU with 1-HU fine bins) are exact.
Values outside the histogram range are attributed to the bin that crosses the
range boundary (e.g. -2048 padding goes to a bin whose lo is < -1024 or null).
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import yaml

from .config import HistogramSpec

PRESET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "presets")
DEFAULT_COLORS = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
                  "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"]


@dataclass
class Bin:
    name: str
    lo: Optional[float]   # None = -inf
    hi: Optional[float]   # None = +inf
    color: Optional[str] = None


@dataclass
class BinSet:
    name: str
    bins: list
    unit: str = ""
    description: str = ""

    def validate(self) -> list[str]:
        """Return a list of problems (empty = OK). Overlaps and gaps are reported."""
        problems = []
        edges = [(-np.inf if b.lo is None else b.lo, np.inf if b.hi is None else b.hi, b.name)
                 for b in self.bins]
        for lo, hi, n in edges:
            if not lo < hi:
                problems.append(f"bin '{n}': lo must be < hi")
        s = sorted(edges)
        for (l1, h1, n1), (l2, h2, n2) in zip(s, s[1:]):
            if l2 < h1:
                problems.append(f"bins '{n1}' and '{n2}' overlap")
            elif l2 > h1:
                problems.append(f"gap between '{n1}' and '{n2}' ({h1} to {l2})")
        return problems

    def to_dict(self) -> dict:
        return {"name": self.name, "unit": self.unit, "description": self.description,
                "bins": [asdict(b) for b in self.bins]}

    @classmethod
    def from_dict(cls, d: dict) -> "BinSet":
        bins = [Bin(b["name"], b.get("lo"), b.get("hi"), b.get("color")) for b in d["bins"]]
        for i, b in enumerate(bins):
            b.color = b.color or DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
        return cls(d.get("name", "custom"), bins, d.get("unit", ""), d.get("description", ""))

    @classmethod
    def from_edges(cls, edges, name="custom", names=None, unit="", open_ends=False) -> "BinSet":
        """BinSet from edges, e.g. [0, 120, 180, 216] -> 3 bins.
        ``open_ends`` adds (-inf, e0) and (eN, +inf) bins."""
        e = [float(v) for v in edges]
        pairs = list(zip(e[:-1], e[1:]))
        if open_ends:
            pairs = [(None, e[0])] + pairs + [(e[-1], None)]
        bins = []
        for i, (lo, hi) in enumerate(pairs):
            nm = names[i] if names and i < len(names) else _range_name(lo, hi)
            bins.append(Bin(nm, lo, hi, DEFAULT_COLORS[i % len(DEFAULT_COLORS)]))
        return cls(name, bins, unit)


def _range_name(lo, hi) -> str:
    f = lambda v: f"{v:g}"
    if lo is None:
        return f"< {f(hi)}"
    if hi is None:
        return f">= {f(lo)}"
    return f"[{f(lo)}, {f(hi)})"


# ------------------------------------------------------------------ presets
def load_presets(path: Optional[str] = None) -> dict:
    out = {}
    files = [path] if path else [os.path.join(PRESET_DIR, f) for f in sorted(os.listdir(PRESET_DIR))
                                 if f.endswith("_bins.yaml")]
    for fp in files:
        with open(fp) as fh:
            data = yaml.safe_load(fh) or {}
        for name, d in data.items():
            out[name] = BinSet.from_dict({**d, "name": name})
    return out


def load_windows() -> dict:
    with open(os.path.join(PRESET_DIR, "ct_windows.yaml")) as fh:
        return yaml.safe_load(fh)


def save_binset(bs: BinSet, path: str) -> None:
    with open(path, "w") as fh:
        yaml.safe_dump({bs.name: {k: v for k, v in bs.to_dict().items() if k != "name"}}, fh,
                       sort_keys=False)


def equal_width(lo: float, hi: float, n: int, unit: str = "") -> BinSet:
    return BinSet.from_edges(np.linspace(lo, hi, n + 1), f"equal_{n}", unit=unit)


def quantile_bins(counts: np.ndarray, spec: HistogramSpec, n: int, unit: str = "") -> BinSet:
    """Bins holding ~equal numbers of voxels in the given (dataset) histogram."""
    cdf = _cdf_at_edges(np.asarray(counts, dtype=np.float64), spec)
    edges = spec.edges()
    targets = np.linspace(0, cdf[-1], n + 1)[1:-1]
    inner = np.interp(targets, cdf, edges)
    e = np.unique(np.round(np.concatenate([[spec.lo], inner, [spec.hi]]), 6))
    return BinSet.from_edges(e, f"quantile_{n}", unit=unit)


def otsu_threshold(counts: np.ndarray, spec: HistogramSpec) -> float:
    c = np.asarray(counts[1:-1], dtype=np.float64)
    centers = spec.edges()[:-1] + spec.width / 2
    w0 = np.cumsum(c)
    w1 = w0[-1] - w0
    m0 = np.cumsum(c * centers)
    mu0 = np.divide(m0, w0, out=np.zeros_like(m0), where=w0 > 0)
    mu1 = np.divide(m0[-1] - m0, w1, out=np.zeros_like(m0), where=w1 > 0)
    between = w0 * w1 * (mu0 - mu1) ** 2
    return float(centers[int(np.argmax(between))] + spec.width / 2)


# ------------------------------------------------------------------ application
def _cdf_at_edges(counts: np.ndarray, spec: HistogramSpec) -> np.ndarray:
    """Cumulative count at each regular edge; F(lo) includes the underflow bin."""
    return counts[0] + np.concatenate([[0.0], np.cumsum(counts[1:-1])])


def _cdf(counts: np.ndarray, spec: HistogramSpec, x: np.ndarray) -> np.ndarray:
    """Counts below x (vectorised over the last axis of counts)."""
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum(axis=-1, keepdims=True)
    inner = np.cumsum(counts[..., 1:-1], axis=-1)
    cdf = counts[..., :1] + np.concatenate([np.zeros_like(total), inner], axis=-1)  # at edges
    edges = spec.edges()
    out = np.empty(counts.shape[:-1] + (len(x),))
    for j, v in enumerate(x):
        if v == -np.inf:
            out[..., j] = 0.0
        elif v == np.inf:
            out[..., j] = total[..., 0]
        elif v < spec.lo:
            out[..., j] = 0.0
        elif v > spec.hi:
            out[..., j] = total[..., 0]
        else:
            pos = (v - spec.lo) / spec.width
            k = min(int(np.floor(pos)), spec.nbins - 1)
            frac = pos - k
            out[..., j] = cdf[..., k] + frac * (cdf[..., k + 1] - cdf[..., k])
    return out


def apply_binset(counts: np.ndarray, spec: HistogramSpec, bs: BinSet) -> np.ndarray:
    """Counts per bin. ``counts`` may be (size,) or (n_cases, size) -> (..., n_bins)."""
    lo = np.array([-np.inf if b.lo is None else b.lo for b in bs.bins], dtype=np.float64)
    hi = np.array([np.inf if b.hi is None else b.hi for b in bs.bins], dtype=np.float64)
    return _cdf(counts, spec, hi) - _cdf(counts, spec, lo)


def binset_table(counts: np.ndarray, spec: HistogramSpec, bs: BinSet) -> list[dict]:
    c = apply_binset(counts, spec, bs)
    total = float(np.sum(counts))
    return [{"name": b.name, "lo": b.lo, "hi": b.hi, "color": b.color,
             "count": float(v), "fraction": float(v / total) if total else float("nan")}
            for b, v in zip(bs.bins, c)]
