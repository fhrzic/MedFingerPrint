"""Outlier detection over per-case features and per-case histograms.

Methods
-------
robust_z   modified z-score 0.6745 (x - median) / MAD  (Iglewicz & Hoaglin; flag > 3.5)
zscore     (x - mean) / std                               (flag > 3)
iqr        Tukey fences, distance beyond Q1/Q3 in IQR units (flag > 1.5; > 3 = far out)
histogram  Wasserstein-1 distance of each case's normalised histogram to the
           dataset median histogram, then robust-z scored
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .config import HistogramSpec

METHODS = {"robust_z": 3.5, "zscore": 3.0, "iqr": 1.5}


def get_path(d: dict, path: str):
    for p in path.split("."):
        if d is None:
            return None
        if isinstance(d, list):
            d = d[int(p)] if int(p) < len(d) else None
        else:
            d = d.get(p)
    return d


def default_features(fp: dict) -> list[str]:
    regions = list(fp.get("intensity", {}).keys())
    is_ct = fp.get("config", {}).get("modality") == "CT"
    dims = 3 if any(c.get("geometry", {}).get("dims") == 3 for c in fp.get("cases", [])) else 2
    feats = []
    main = "body" if "body" in regions else "all"
    for k in ("mean", "std", "p50", "p00_5", "p99_5", "skewness"):
        feats.append(f"regions.{main}.{k}")
    if "foreground" in regions:
        feats += ["regions.foreground.mean", "regions.foreground.std"]
    if dims == 3:
        feats += ["geometry.spacing.0", "geometry.spacing.1", "geometry.spacing.2",
                  "geometry.shape.0", "geometry.extent_mm.0"]
    else:
        feats += ["geometry.shape.0", "geometry.shape.1", "regions.all.entropy_bits"]
    if "body" in regions:
        feats.append("flags.body_fraction")
    if is_ct:
        feats.append("regions.all.frac_below_range")
    else:
        feats += ["flags.frac_at_image_min", "flags.frac_at_image_max"]
    if any(c.get("quality") for c in fp.get("cases", [])[:50]):
        feats += ["quality.sharpness", "quality.noise"]
        if dims == 2 and not is_ct:
            feats += ["quality.bright_background", "quality.frac_border_padding"]
    return feats


# Spread below this fraction of |median| is treated as numerical noise (e.g. spacing
# 1.5 vs 1.5000001), so it can never produce a large score.
REL_SCALE_FLOOR = 1e-3
# Fractions such as "share of pixels at the image maximum" are zero-inflated and
# heavily right-skewed: score them as log10(x + eps) and flag only high values.
FRACTION_EPS = 1e-4
MIN_GROUP = 10  # smaller groups are scored against the whole dataset
# In large datasets heavy tails make any fixed z threshold flag thousands of cases. A case is
# therefore flagged on a feature only if it also lies in the most extreme ``max_frac`` of
# that feature (per direction; at least MIN_CAP cases, so small datasets are unaffected).
DEFAULT_MAX_FRAC = 0.01
MIN_CAP = 5


def _cap(s: np.ndarray, hit: np.ndarray, max_frac: Optional[float]) -> np.ndarray:
    if not max_frac:
        return hit
    n_ok = int(np.isfinite(s).sum())
    k = max(MIN_CAP, int(np.ceil(max_frac * n_ok)))
    out = np.zeros_like(hit)
    for sign in (1, -1):
        idx = np.flatnonzero(hit & (np.sign(s) == sign))
        if idx.size > k:
            idx = idx[np.argsort(-np.abs(s[idx]))[:k]]
        out[idx] = True
    return out


def is_fraction_feature(feat: str) -> bool:
    last = feat.split(".")[-1]
    return last.startswith("frac_")


def _robust_scale(v: np.ndarray, med: float) -> float:
    dev = np.abs(v - med)
    mad = np.median(dev)
    if mad > 0:
        return mad / 0.6745
    # >50% of the values are tied at the median (e.g. images resized to 512 px).
    rest = dev[dev > 0]
    if rest.size >= max(MIN_GROUP, 0.1 * v.size):
        # The non-tied values form a real minority mode (MURA: 21 % landscape images):
        # use their MAD so the whole mode is not reported as outliers.
        return float(np.median(rest) / 0.6745)
    # Only a few cases deviate: each of them is unusual. Mean absolute deviation.
    return float(1.253314 * np.mean(dev))


def _scores(x: np.ndarray, method: str, ref: Optional[np.ndarray] = None) -> np.ndarray:
    """Score ``x`` against the distribution of ``ref`` (default: ``x`` itself)."""
    finite = np.isfinite(x)
    s = np.full_like(x, np.nan, dtype=np.float64)
    r = (x if ref is None else np.asarray(ref, dtype=np.float64))
    r = r[np.isfinite(r)]
    if r.size < 3:
        return s
    v = x[finite]
    med = float(np.median(r))
    floor = max(REL_SCALE_FLOOR * abs(med), 1e-12)
    if method == "robust_z":
        s[finite] = (v - med) / max(_robust_scale(r, med), floor)
    elif method == "zscore":
        s[finite] = (v - r.mean()) / max(r.std(), floor)
    elif method == "iqr":
        q1, q3 = np.percentile(r, [25, 75])
        d = np.where(v < q1, v - q1, 0.0) + np.where(v > q3, v - q3, 0.0)
        iqr = q3 - q1
        if iqr > floor:
            s[finite] = d / iqr
        else:
            s[finite] = np.where(np.abs(d) > floor, np.sign(d) * 99.0, 0.0)
    else:
        raise ValueError(f"Unknown method {method!r}")
    return s


def case_groups(fp: dict, cases: list, group_by: Optional[str]) -> Optional[list]:
    """Group label per case. ``group_by``: ``level:N`` (N-th component of the case id
    path, e.g. MURA ``train/XR_WRIST/...`` -> level:1) or ``meta:KEY`` (metadata
    column, e.g. ``meta:institute`` or ``meta:ConvolutionKernel``)."""
    if not group_by:
        return None
    kind, _, key = group_by.partition(":")
    out = []
    for c in cases:
        if kind == "level":
            parts = str(c["case_id"]).replace("\\", "/").split("/")
            n = int(key)
            out.append(parts[n] if -len(parts) <= n < len(parts) - 1 or n == -1 else "")
        elif kind == "meta":
            out.append(str((c.get("meta") or {}).get(key, "")))
        else:
            raise ValueError(f"group_by must be 'level:N' or 'meta:KEY', got {group_by!r}")
    return out


def _grouped_scores(x: np.ndarray, method: str, groups: Optional[list]):
    """Scores and, per case, the median of its reference distribution."""
    if groups is None:
        return _scores(x, method), np.full(x.shape, np.nanmedian(x))
    s = np.full_like(x, np.nan, dtype=np.float64)
    meds = np.full_like(x, np.nan, dtype=np.float64)
    g = np.asarray(groups, dtype=object)
    for name in set(groups):
        idx = np.flatnonzero(g == name)
        ref = x[idx] if idx.size >= MIN_GROUP else x
        s[idx] = _scores(x[idx], method, ref)
        meds[idx] = np.nanmedian(ref) if np.isfinite(ref).any() else np.nan
    return s, meds


def wasserstein_to_reference(hists: np.ndarray, spec: HistogramSpec,
                             reference: Optional[np.ndarray] = None) -> np.ndarray:
    """W1 distance (in intensity units) of each row of ``hists`` to ``reference``
    (default: per-bin median of the normalised rows)."""
    h = np.asarray(hists, dtype=np.float64)
    tot = h.sum(axis=1, keepdims=True)
    p = np.divide(h, tot, out=np.zeros_like(h), where=tot > 0)
    if reference is None:
        ref = np.median(p, axis=0)
    else:
        ref = np.asarray(reference, dtype=np.float64)
    ref = ref / ref.sum() if ref.sum() > 0 else ref
    return np.abs(np.cumsum(p, axis=1) - np.cumsum(ref)).sum(axis=1) * spec.width


def detect_outliers(fp: dict, sidecar: Optional[dict] = None, method: str = "robust_z",
                    threshold: Optional[float] = None, features: Optional[list] = None,
                    hist_region: Optional[str] = None, group_by: Optional[str] = None,
                    max_frac: Optional[float] = DEFAULT_MAX_FRAC) -> dict:
    cases = [c for c in fp.get("cases", []) if c.get("status") == "ok"]
    threshold = METHODS.get(method, 3.5) if threshold is None else threshold
    features = features or default_features(fp)
    ids = [c["case_id"] for c in cases]
    groups = case_groups(fp, cases, group_by)
    flagged, per_feature = [], {}

    def _add(i, feat, value, med, score, direction):
        r = {"case_id": ids[i], "feature": feat, "value": float(value),
             "dataset_median": med, "score": float(score), "direction": direction}
        if groups is not None:
            r["group"] = groups[i]
        flagged.append(r)

    for feat in features:
        x = np.array([_f(get_path(c, feat)) for c in cases], dtype=np.float64)
        if np.isfinite(x).sum() < 3:
            continue
        frac = is_fraction_feature(feat)
        xs = np.log10(np.clip(x, 0, None) + FRACTION_EPS) if frac else x
        s, meds = _grouped_scores(xs, method, groups)
        if frac:
            meds = 10 ** meds - FRACTION_EPS
        over = s > threshold if frac else np.abs(s) > threshold
        hit = _cap(s, over, max_frac)
        med = float(np.nanmedian(x))
        per_feature[feat] = {"median": med, "n_flagged": int(hit.sum()),
                             "n_over_threshold": int(over.sum()),
                             "scale": "log10" if frac else "linear"}
        for i in np.flatnonzero(hit):
            _add(i, feat, x[i], float(meds[i]), s[i], "high" if s[i] > 0 else "low")

    hist_dist = None
    if sidecar is not None:
        region = hist_region or ("body" if "hist_body" in sidecar else "all")
        key = f"hist_{region}"
        if key in sidecar and len(sidecar.get("case_ids", [])) == len(fp.get("cases", [])):
            ok = np.array([c.get("status") == "ok" for c in fp["cases"]])
            spec = HistogramSpec.from_dict(fp["histogram_spec"])
            H = sidecar[key][ok]
            d = np.full(len(cases), np.nan)
            if groups is None:
                d = wasserstein_to_reference(H, spec)
            else:
                g = np.asarray(groups, dtype=object)
                for name in set(groups):
                    idx = np.flatnonzero(g == name)
                    tot = H.sum(axis=1, keepdims=True).astype(np.float64)
                    P = np.divide(H, tot, out=np.zeros(H.shape), where=tot > 0)
                    ref = np.median(P[idx] if idx.size >= MIN_GROUP else P, axis=0)
                    d[idx] = wasserstein_to_reference(H[idx], spec, ref)
            s, meds = _grouped_scores(d, "robust_z", groups)
            feat = f"histogram_w1.{region}"
            med = float(np.median(d))
            over = s > METHODS["robust_z"]
            hit = _cap(s, over, max_frac)
            per_feature[feat] = {"median": med, "n_flagged": int(hit.sum()),
                                 "n_over_threshold": int(over.sum()), "scale": "linear"}
            hist_dist = {cid: float(v) for cid, v in zip(ids, d)}
            for i in np.flatnonzero(hit):
                _add(i, feat, d[i], float(meds[i]), s[i], "high")

    flagged.sort(key=lambda r: -abs(r["score"]))
    by_case: dict = {}
    for r in flagged:
        e = by_case.setdefault(r["case_id"], {"case_id": r["case_id"], "max_score": 0.0,
                                               "n_features": 0, "features": []})
        if "group" in r:
            e["group"] = r["group"]
        e["max_score"] = max(e["max_score"], abs(r["score"]))
        e["n_features"] += 1
        e["features"].append(r["feature"])
    ranked = sorted(by_case.values(), key=lambda e: (-e["n_features"], -e["max_score"]))
    return {"method": method, "threshold": threshold, "features": features,
            "group_by": group_by, "max_frac": max_frac, "n_cases": len(cases),
            "per_feature": per_feature, "flags": flagged, "cases": ranked,
            "histogram_w1": hist_dist}


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan
