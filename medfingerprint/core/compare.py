"""Compare two or more fingerprints.

Everything is computed from the fingerprint JSONs (pooled fine histograms and summaries);
no images are re-read. Histograms with different specs (e.g. two 2D datasets with
different automatic ranges) are compared through their CDFs on a common grid.

Distances between pooled intensity distributions
------------------------------------------------
wasserstein  W1 (earth mover's) distance in intensity units (HU for CT):
             the average shift needed to turn one distribution into the other.
ks           Kolmogorov-Smirnov statistic: max |F_a - F_b|, 0..1.
js           Jensen-Shannon divergence (base 2) on common bins, 0 (identical)..1 (disjoint).
"""
from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Optional

import numpy as np

from .bins import BinSet, _cdf, apply_binset
from .config import HistogramSpec

MAX_GRID = 20000
DISTANCES = ("wasserstein", "ks", "js")


def load_for_compare(path: str) -> dict:
    """Load a fingerprint, dropping the (possibly huge) per-case table."""
    with open(path) as fh:
        fp = json.load(fh)
    if not str(fp.get("schema", "")).startswith("medfingerprint/"):
        raise ValueError(f"{path} is not a MedFingerPrint fingerprint")
    fp.pop("cases", None)
    fp["_path"] = path
    return fp


# ---------------------------------------------------------------- common intensity scale
# Datasets stored with different value ranges (8-bit PNG 0..255 vs 16-bit DICOM 0..65535,
# HU, [0, 1] floats) can be compared on one scale: each dataset is mapped linearly
# x' = (x - lo) / (hi - lo) * 255 with its OWN reference range [lo, hi]. Intensity values
# (mean, percentiles, min, max) are shifted and scaled, spreads (std, IQR) only scaled, and
# shape measures (skewness, kurtosis, entropy, fractions) are unchanged; histograms keep
# their counts on rescaled bin edges. No image is re-read.
SCALES = {
    "original": "Original values (HU for CT)",
    "robust": "0–255 from each dataset's p0.5–p99.5 (recommended: ignores rare extreme pixels; "
              "about 1 % of values fall slightly below 0 / above 255)",
    "minmax": "0–255 from each dataset's minimum–maximum (one extreme pixel changes the scale)",
    "bits": "0–255 from each dataset's bit depth (e.g. 12-bit 0–4095, 16-bit 0–65535, 8-bit 0–255): "
            "like converting the files to 8 bit",
}
SCALE_TARGET = 255.0
_LOC = re.compile(r"^(mean|min|max|median|p\d+(_\d+)?|percentile_\d+_\d+)$")
_SPREAD = {"std", "iqr"}


def intensity_range(fp: dict, mode: str):
    """Reference range [lo, hi] of a dataset for ``mode`` (region "all": every voxel)."""
    inten = fp.get("intensity") or {}
    reg = inten.get("all") or next(iter(inten.values()), {})
    pooled = reg.get("pooled") or {}
    if mode == "robust":
        lo, hi = pooled.get("p00_5"), pooled.get("p99_5")
    elif mode == "minmax":
        lo, hi = pooled.get("min"), pooled.get("max")
    elif mode == "bits":
        eb = (fp.get("geometry") or {}).get("effective_bits") or {}
        bits = max((int(k) for k in eb if str(k).lstrip("-").isdigit()), default=None)
        if bits is None:
            dt = _top(_g(fp, "geometry", "original_dtype")) or ""
            bits = 8 if "8" in dt else 16 if "16" in dt else None
        mn = pooled.get("min")
        if bits is None or mn is None:
            lo, hi = pooled.get("min"), pooled.get("max")
        else:
            lo, hi = min(0.0, float(mn)), float(2 ** bits - 1) + min(0.0, float(mn))
    else:
        return None
    if lo is None or hi is None or not hi > lo:
        return None
    return float(lo), float(hi)


def _tx(v, a, k, spread=False):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return v
    return v * k if spread else (v - a) * k


def _tx_flat(d: dict, a: float, k: float):
    """Pooled statistics / nnU-Net properties: intensities shift+scale, spreads scale."""
    if not isinstance(d, dict):
        return
    for key, v in d.items():
        if key in _SPREAD:
            d[key] = _tx(v, a, k, True)
        elif _LOC.match(key):
            d[key] = _tx(v, a, k)


def _tx_summary(d: dict, a: float, k: float, spread: bool):
    """summarize() of a per-case quantity: its location keys follow the quantity."""
    if not isinstance(d, dict):
        return
    for key, v in d.items():
        if key == "n":
            continue
        d[key] = _tx(v, a, k, spread or key == "std")


def rescale_fp(fp: dict, mode: str) -> dict:
    """Copy of a fingerprint with all intensity quantities on the common 0–255 scale."""
    if mode in (None, "original"):
        return fp
    rng = intensity_range(fp, mode)
    if rng is None:
        out = dict(fp)
        out["_scale"] = {"mode": mode, "error": "no usable intensity range"}
        return out
    a, b = rng
    k = SCALE_TARGET / (b - a)
    out = deepcopy({key: v for key, v in fp.items() if key not in ("cases", "outliers", "bins")})
    hs = out.get("histogram_spec") or {}
    for key in ("lo", "hi"):
        if key in hs:
            hs[key] = (hs[key] - a) * k
    if "width" in hs:
        hs["width"] = hs["width"] * k
    for reg in (out.get("intensity") or {}).values():
        h = reg.get("histogram") or {}
        for key in ("lo", "hi"):
            if key in h:
                h[key] = (h[key] - a) * k
        if "width" in h:
            h["width"] = h["width"] * k
        _tx_flat(reg.get("pooled"), a, k)
        for q, summ in (reg.get("per_case") or {}).items():
            if q in _SPREAD or _LOC.match(q):
                _tx_summary(summ, a, k, spread=q in _SPREAD)
    for key in ("nnunet_fingerprint", "nnunet_fingerprint_body"):
        for props in ((out.get(key) or {}).get("foreground_intensity_properties_per_channel") or {}).values():
            _tx_flat(props, a, k)
    for lab in (out.get("labels") or {}).values():
        _tx_flat(lab.get("pooled_intensity"), a, k)
        _tx_summary(lab.get("case_mean_intensity"), a, k, spread=False)
    out["_scale"] = {"mode": mode, "from": [a, b], "to": [0.0, SCALE_TARGET], "factor": k}
    return out


def scale_note(fps: list) -> Optional[str]:
    sc = [fp.get("_scale") for fp in fps]
    if not any(sc):
        return None
    mode = next(s["mode"] for s in sc if s)
    parts = []
    for fp, s in zip(fps, sc):
        if s and s.get("from"):
            parts.append(f"{name_of(fp)}: {_num(s['from'][0])}–{_num(s['from'][1])} → 0–255")
        elif s:
            parts.append(f"{name_of(fp)}: not rescaled ({s.get('error')})")
    return f"Intensities rescaled to a common scale ({SCALES[mode]}). " + "; ".join(parts) + "."


def _num(v):
    return f"{v:.6g}"


def name_of(fp: dict) -> str:
    return fp.get("dataset", {}).get("name") or fp.get("_path", "?")


def common_regions(fps: list) -> list:
    regs = [list(fp.get("intensity", {}).keys()) for fp in fps]
    return [r for r in regs[0] if all(r in x for x in regs[1:])] if regs else []


# ---------------------------------------------------------------- histograms
def region_hist(fp: dict, region: str):
    h = fp["intensity"][region]["histogram"]
    return np.asarray(h["counts"], dtype=np.float64), HistogramSpec.from_dict(fp["histogram_spec"])


def _grid(specs: list, counts: list) -> np.ndarray:
    """Common grid covering the occupied range of all histograms."""
    los, his = [], []
    for c, s in zip(counts, specs):
        nz = np.flatnonzero(c[1:-1])
        e = s.edges()
        los.append(e[nz[0]] if nz.size else s.lo)
        his.append(e[nz[-1] + 1] if nz.size else s.hi)
    lo, hi = min(los), max(his)
    step = max(min(s.width for s in specs), (hi - lo) / MAX_GRID)
    return np.arange(lo, hi + step / 2, step)


def cdfs_on_grid(items: list):
    """items: [(counts, spec)] -> (grid, [normalised CDF on grid])."""
    counts = [c for c, _ in items]
    specs = [s for _, s in items]
    x = _grid(specs, counts)
    F = []
    for c, s in items:
        tot = c.sum()
        F.append(_cdf(c, s, x) / tot if tot > 0 else np.zeros_like(x))
    return x, F


def distance(a: tuple, b: tuple, metric: str = "wasserstein") -> float:
    x, (Fa, Fb) = cdfs_on_grid([a, b])
    if metric == "wasserstein":
        mid = 0.5 * (np.abs(Fa[1:] - Fb[1:]) + np.abs(Fa[:-1] - Fb[:-1]))
        return float(np.sum(mid * np.diff(x)))
    if metric == "ks":
        return float(np.max(np.abs(Fa - Fb)))
    if metric == "js":
        # probabilities on the common bins (plus mass below/above the grid)
        pa = np.concatenate([[Fa[0]], np.diff(Fa), [1 - Fa[-1]]]).clip(0)
        pb = np.concatenate([[Fb[0]], np.diff(Fb), [1 - Fb[-1]]]).clip(0)
        m = 0.5 * (pa + pb)

        def kl(p, q):
            k = p > 0
            return float(np.sum(p[k] * np.log2(p[k] / q[k])))
        return 0.5 * kl(pa, m) + 0.5 * kl(pb, m)
    raise ValueError(f"metric must be one of {DISTANCES}")


def distance_matrix(fps: list, region: str, metric: str = "wasserstein") -> np.ndarray:
    items = [region_hist(fp, region) for fp in fps]
    n = len(items)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = distance(items[i], items[j], metric)
    return D


def density(fp: dict, region: str):
    """(bin centres, density per intensity unit) of the pooled histogram (regular bins only)."""
    c, s = region_hist(fp, region)
    tot = c.sum()
    e = s.edges()
    return e, (c[1:-1] / (tot * s.width) if tot > 0 else c[1:-1])


# ---------------------------------------------------------------- tables
def _g(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _axes(fps):
    dims = 3 if any(len(fp.get("geometry", {}).get("spacing") or []) == 3 for fp in fps) else 2
    return ("z", "y", "x") if dims == 3 else ("y (height)", "x (width)")


def stats_rows(fps: list, region: Optional[str] = None) -> list:
    """Rows (section, metric, [value per dataset], help) for a side-by-side table."""
    region = region or ("body" if "body" in common_regions(fps) else (common_regions(fps) or ["all"])[0])
    rows = []

    def add(section, metric, vals, tip=""):
        if any(v is not None for v in vals):
            rows.append((section, metric, vals, tip))

    add("Dataset", "modality", [_g(fp, "dataset", "modality") for fp in fps])
    add("Dataset", "cases (ok)", [_g(fp, "dataset", "n_ok") for fp in fps])
    add("Dataset", "failed cases", [_g(fp, "dataset", "n_failed") for fp in fps])
    add("Dataset", "has labels", [_g(fp, "dataset", "has_labels") for fp in fps])
    add("Dataset", "original dtype", [_top(_g(fp, "geometry", "original_dtype")) for fp in fps])
    if any(fp.get("_scale") for fp in fps):
        add("Dataset", "intensity scale (original → 0–255)",
            [(f"{_num(fp['_scale']['from'][0])}–{_num(fp['_scale']['from'][1])} → 0–255"
              if (fp.get("_scale") or {}).get("from") else "not rescaled") for fp in fps],
            "each dataset is mapped linearly with its own reference range")
    for i, ax in enumerate(_axes(fps)):
        add("Geometry", f"spacing {ax} median (mm)", [_idx(_g(fp, "geometry", "spacing"), i, "median") for fp in fps])
    for i, ax in enumerate(_axes(fps)):
        add("Geometry", f"shape {ax} median", [_idx(_g(fp, "geometry", "shape"), i, "median") for fp in fps])
    for i, ax in enumerate(_axes(fps)):
        add("Geometry", f"extent {ax} median (mm)", [_idx(_g(fp, "geometry", "extent_mm"), i, "median") for fp in fps])
    add("Geometry", "orientation", [_top(_g(fp, "geometry", "orientation")) for fp in fps])

    sec = f"Intensity ({region}, pooled)"
    for k, tip in (("mean", "mean of all voxels"), ("std", "spread"), ("p00_5", "0.5th percentile"),
                   ("p05", ""), ("p25", ""), ("p50", "median"), ("p75", ""), ("p95", ""),
                   ("p99_5", "99.5th percentile"), ("iqr", "p75 - p25"), ("skewness", ""),
                   ("kurtosis", ""), ("entropy_bits", "grey-level variety"),
                   ("frac_below_range", "share of voxels below the histogram range (CT: < -1024 HU)")):
        add(sec, k, [_g(fp, "intensity", region, "pooled", k) for fp in fps], tip)
    sec = f"Per-case variability ({region})"
    for k in ("mean", "std", "p50"):
        add(sec, f"case {k}: median", [_g(fp, "intensity", region, "per_case", k, "median") for fp in fps],
            f"median over cases of each case's {k}")
        add(sec, f"case {k}: IQR", [_iqr(_g(fp, "intensity", region, "per_case", k)) for fp in fps],
            f"spread between cases: p75 - p25 of the per-case {k}")
    add("Body region", "body fraction median", [_g(fp, "flags", "body_fraction", "median") for fp in fps],
        "median share of the image covered by the body region")
    add("Body region", "body method", [_g(fp, "flags", "body_method") for fp in fps])
    for key, title in (("nnunet_fingerprint", "nnU-Net (labels)"), ("nnunet_fingerprint_body", "nnU-Net-style (body)")):
        for k in ("mean", "std", "median", "percentile_00_5", "percentile_99_5"):
            add(title, k, [_g(fp, key, "foreground_intensity_properties_per_channel", "0", k) for fp in fps])
        add(title, "median rel. size after crop", [_g(fp, key, "median_relative_size_after_cropping") for fp in fps])
    add("Flags", "cases not in HU", [len(_g(fp, "flags", "not_hu_suspected") or []) if fp.get("flags") else None
                                     for fp in fps])
    # metadata: match keys across sources, e.g. DICOM "ConvolutionKernel" == meta.csv "convolution_kernel"
    groups: dict = {}
    for fp in fps:
        for k in (fp.get("metadata") or {}):
            groups.setdefault(_norm_key(k), k)
    for nk, display in sorted(groups.items()):
        vals = []
        for fp in fps:
            md = fp.get("metadata") or {}
            k = next((k for k in md if _norm_key(k) == nk), None)
            vals.append(_meta_summary(md.get(k)) if k else None)
        add("Metadata", display, vals)
    return rows


def _norm_key(k: str) -> str:
    return "".join(ch for ch in str(k).lower() if ch.isalnum())


def label_rows(fps: list) -> list:
    """(label, metric, [value per dataset]) for labels present in at least two datasets."""
    names: dict = {}
    for fp in fps:
        for k in (fp.get("labels") or {}):
            names[k] = names.get(k, 0) + 1
    out = []
    for lab in sorted(k for k, n in names.items() if n >= min(2, len(fps))):
        ls = [(fp.get("labels") or {}).get(lab) for fp in fps]
        out.append((lab, "cases present", [_g(x, "n_cases_present") for x in ls]))
        out.append((lab, "volume ml (median)", [_g(x, "volume_ml", "median") for x in ls]))
        out.append((lab, "mean intensity", [_g(x, "pooled_intensity", "mean") for x in ls]))
        out.append((lab, "std intensity", [_g(x, "pooled_intensity", "std") for x in ls]))
        out.append((lab, "median intensity", [_g(x, "pooled_intensity", "p50") for x in ls]))
    return out


def bin_fractions(fps: list, region: str, bs: BinSet) -> np.ndarray:
    """(n_datasets, n_bins) fractions of voxels per bin."""
    out = []
    for fp in fps:
        c, s = region_hist(fp, region)
        v = apply_binset(c, s, bs)
        out.append(v / c.sum() if c.sum() else v)
    return np.array(out)


def difference(a, b):
    """(absolute, relative %) difference b - a for numbers, else (None, None)."""
    if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, (int, float)) \
            or not isinstance(b, (int, float)):
        return None, None
    d = b - a
    rel = 100.0 * d / abs(a) if a not in (0, None) and np.isfinite(a) else None
    return d, rel


def _idx(lst, i, key):
    try:
        return lst[i][key]
    except (TypeError, IndexError, KeyError):
        return None


def _iqr(d):
    if not isinstance(d, dict) or d.get("p75") is None or d.get("p25") is None:
        return None
    return d["p75"] - d["p25"]


def _top(counts):
    if not isinstance(counts, dict) or not counts:
        return None
    k, v = max(counts.items(), key=lambda kv: kv[1])
    tot = sum(counts.values())
    return f"{k} ({100 * v / tot:.0f} %)" if len(counts) > 1 else str(k)


def _meta_summary(m):
    if not isinstance(m, dict):
        return None
    if m.get("type") == "numeric" or "median" in m:
        return m.get("median")
    c = m.get("counts")
    if isinstance(c, dict) and c:
        items = sorted(c.items(), key=lambda kv: -kv[1])
        tot = sum(c.values())
        s = ", ".join(f"{k} {100 * v / tot:.0f}%" for k, v in items[:3])
        return s + (f" (+{len(items) - 3})" if len(items) > 3 else "")
    return None
