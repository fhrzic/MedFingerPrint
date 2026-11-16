"""Image quality measures and duplicate detection.

Per case (:func:`case_quality`, cheap, computed during the scan):

sharpness          mean |Laplacian| of the image resized to <= 512 px, divided by the
                   image's p5-p95 range (CT: in HU). Low = blurry / smoothed.
noise              Immerkaer (1996) noise sigma on a native-resolution central crop,
                   divided by the p5-p95 range (CT: in HU, mid axial slice). High = noisy,
                   e.g. low dose or a sharp (lung/bone) reconstruction kernel.
bright_background  (median of the outer border - median of the central third) / p5-p95 range.
                   X-rays normally have a dark background (< 0); clearly positive values
                   mean a bright background: inverted grey scale (e.g. MONOCHROME1 not
                   handled) or large collimated areas. 2D only.
frac_border_padding  share of the image in constant rows/columns at the edges (black or
                   white bars, padding to a square).
colourfulness      mean channel difference / range for RGB images (0 = grey stored as RGB).
frac_corner_bright share of corner pixels brighter than the image's 99.5th percentile:
                   burned-in text and L/R markers.
dynamic_range      p0.5-p99.5 range divided by the range of the original data type
                   (integer images). Very low = flat / under-exposed.
dhash, dhash_fine difference hashes on 8x8 and 16x16 grids (64 / 256 bits per view; 3D:
                   axial, coronal, sagittal): images that look the same have hashes that
                   differ in few bits, even after re-saving, resizing or brightness changes.
pixel_sha1         hash of the exact pixel values (exact duplicates).

Dataset level (:func:`find_duplicates`, :func:`summarize_quality`).
"""
from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np

PROC_MAX = 512
HASH_W, HASH_H = 9, 8


def _resize(a: np.ndarray, w: int, h: int) -> np.ndarray:
    from PIL import Image
    a = np.nan_to_num(np.asarray(a, dtype=np.float32))
    return np.asarray(Image.fromarray(a, mode="F").resize((w, h), Image.BOX), dtype=np.float32)


def _small(a: np.ndarray, max_side: int = PROC_MAX) -> np.ndarray:
    h, w = a.shape
    s = max(h, w)
    if s <= max_side:
        return np.nan_to_num(a.astype(np.float32))
    f = max_side / s
    return _resize(a, max(1, round(w * f)), max(1, round(h * f)))


def dhash(a2d: np.ndarray, n: int = 8) -> str:
    """Difference hash of a 2D image on an n x n grid (n*n bits) as a hex string."""
    sm = _resize(a2d, n + 1, n)
    bits = (sm[:, 1:] > sm[:, :-1]).ravel()
    return np.packbits(bits).tobytes().hex()


def _noise_sigma(a: np.ndarray) -> float:
    """Immerkaer fast noise variance estimation, on a central crop at native resolution."""
    from scipy.signal import convolve2d
    h, w = a.shape
    c = 256
    y0, x0 = max(0, h // 2 - c), max(0, w // 2 - c)
    crop = np.nan_to_num(a[y0: y0 + 2 * c, x0: x0 + 2 * c].astype(np.float64))
    if min(crop.shape) < 3:
        return float("nan")
    m = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    conv = np.abs(convolve2d(crop, m, mode="valid"))
    H, W = crop.shape
    return float(np.sqrt(np.pi / 2) * conv.sum() / (6 * (W - 2) * (H - 2)))


def _border_padding(a: np.ndarray) -> float:
    """Share of the image covered by constant rows/columns at the four edges."""
    h, w = a.shape
    rows_const = np.ptp(a, axis=1) == 0
    cols_const = np.ptp(a, axis=0) == 0

    def run(flags):
        n = 0
        for f in flags:
            if not f:
                break
            n += 1
        return n
    top, bottom = run(rows_const), run(rows_const[::-1])
    left, right = run(cols_const), run(cols_const[::-1])
    if top >= h:
        return 1.0
    inner_h = max(0, h - top - bottom)
    inner_w = max(0, w - left - right)
    return float(1.0 - inner_h * inner_w / float(h * w))


def _views(gray: np.ndarray) -> list:
    if gray.ndim == 2:
        return [gray]
    z, y, x = gray.shape
    return [gray[z // 2], gray[:, y // 2, :], gray[:, :, x // 2]]


def case_quality(gray: np.ndarray, rgb: Optional[np.ndarray], original_dtype: str,
                 is_ct: bool) -> dict:
    """Quality measures + hashes for one case (gray: 2D or 3D analysis channel)."""
    from scipy import ndimage
    out: dict = {}
    finite = np.isfinite(gray)
    sub = gray[(slice(None, None, 2),) * gray.ndim] if gray.size > 8_000_000 else gray
    v = sub[np.isfinite(sub)]
    if v.size == 0:
        return {"flat": True}
    p0, p5, p95, p100 = np.percentile(v, (0.5, 5, 95, 99.5))
    rng = float(p95 - p5)
    out["flat"] = bool(np.ptp(v) == 0 or (rng == 0 and float(p100 - p0) == 0))
    dt = np.dtype(original_dtype)
    if np.issubdtype(dt, np.integer) and not is_ct:
        info = np.iinfo(dt)
        full = float(info.max) - float(info.min)
        out["dynamic_range"] = float(p100 - p0) / full if full > 0 else None
    # CT is calibrated (HU): sharpness and noise in absolute HU, comparable between scans and
    # unaffected by padding/air. Other images: relative to the image's own contrast.
    scale = 1.0 if is_ct else (rng if rng > 0 else (float(p100 - p0) or 1.0))

    # hashes
    views = _views(gray)
    clean = [np.where(np.isfinite(vw), vw, 0) for vw in views]
    out["dhash"] = "".join(dhash(vw, 8) for vw in clean)          # 64 bits per view: candidates
    out["dhash_fine"] = "".join(dhash(vw, 16) for vw in clean)    # 256 bits per view: verification
    h = hashlib.sha1()
    h.update(str(gray.shape).encode())
    h.update(np.ascontiguousarray(np.nan_to_num(gray), dtype=np.float32).tobytes())
    out["pixel_sha1"] = h.hexdigest()

    # sharpness / noise on the (middle) 2D view
    a = views[0] if gray.ndim == 2 else views[0]
    sm = _small(a)
    out["sharpness"] = float(np.mean(np.abs(ndimage.laplace(sm)))) / scale
    out["noise"] = _noise_sigma(a) / scale

    if gray.ndim == 2 and not is_ct:
        hh, ww = a.shape
        bh, bw = max(1, hh // 20), max(1, ww // 20)
        border = np.concatenate([a[:bh].ravel(), a[-bh:].ravel(), a[:, :bw].ravel(), a[:, -bw:].ravel()])
        centre = a[hh // 3: 2 * hh // 3, ww // 3: 2 * ww // 3]  # anatomy is usually central
        out["bright_background"] = float(np.nanmedian(border) - np.nanmedian(centre)) / scale
        out["frac_border_padding"] = _border_padding(np.nan_to_num(a))
        ch, cw = max(1, hh // 8), max(1, ww // 8)
        corners = np.concatenate([a[:ch, :cw].ravel(), a[:ch, -cw:].ravel(),
                                  a[-ch:, :cw].ravel(), a[-ch:, -cw:].ravel()])
        thr = np.percentile(v, 99.5)
        out["frac_corner_bright"] = float(np.mean(corners > thr)) if p100 > p5 else 0.0
        if rgb is not None and rgb.shape[0] >= 3:
            c = rgb[:3].astype(np.float32)
            step = max(1, int(np.ceil(max(c.shape[1:]) / PROC_MAX)))
            c = c[:, ::step, ::step]
            diff = (np.abs(c[0] - c[1]) + np.abs(c[1] - c[2]) + np.abs(c[0] - c[2])) / 3
            out["colourfulness"] = float(diff.mean()) / scale
    del finite
    return out


# ====================================================================== duplicates
def _hash_array(hexes: list) -> np.ndarray:
    """Hex strings (equal length) -> (n, bytes) uint8 array."""
    return np.array([np.frombuffer(bytes.fromhex(h), dtype=np.uint8) for h in hexes])


def _hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.bitwise_count(a ^ b).sum(axis=-1)


# Verified near-duplicate: the 256-bit (per view) hashes differ in at most 5.5 % of their
# bits (14 of 256). Measured on CheXpert (223k chest X-rays): re-saved / resized / JPEG-
# recompressed copies differ in <= 6 bits, +10 % brightness in <= 12 bits, while the closest
# 1 in 100,000 pairs of *different* X-rays differ in 32 bits. (The former 16 % = 41 bits let
# millions of different but similar X-rays through.)
NEAR_FINE_FRAC = 0.055
MAX_NEAR_PAIRS = 200_000   # stop the search beyond this: memory stays bounded


def _chunks16(F: np.ndarray) -> np.ndarray:
    if F.shape[1] % 2:
        F = np.pad(F, ((0, 0), (0, 1)))
    return (F[:, 0::2].astype(np.uint32) << 8) | F[:, 1::2]


def _near_candidates(F: np.ndarray, thr: int, max_block: int = 2_000_000, cap: int = MAX_NEAR_PAIRS):
    """All pairs (a < b) whose hashes differ in <= thr bits, exhaustively: split the hash
    into 16-bit pieces; with thr < number of pieces, such a pair has at least one piece in
    common (pigeonhole), so only images sharing a piece value are compared. Returns
    (a, b, distance) arrays and whether the search stopped at ``cap`` pairs."""
    n = F.shape[0]
    W = _chunks16(F)
    if thr >= W.shape[1]:
        raise ValueError("near-duplicate threshold too large for the hash size")
    A, B, D = [], [], []
    seen = set()
    n_found = 0
    for c in range(W.shape[1]):
        key = W[:, c]
        order = np.argsort(key, kind="stable")
        vals = key[order]
        starts = np.flatnonzero(np.r_[True, vals[1:] != vals[:-1]])
        ends = np.r_[starts[1:], n]
        multi = (ends - starts) >= 2
        for s_, e_ in zip(starts[multi], ends[multi]):
            mem = order[s_:e_]
            m = e_ - s_
            sub = F[mem]
            step = max(1, max_block // m)
            for a0 in range(0, m - 1, step):
                blk = sub[a0:a0 + step]
                dist = np.bitwise_count(blk[:, None, :] ^ sub[None, a0:, :]).sum(-1, dtype=np.int32)
                ii, jj = np.nonzero(dist <= thr)
                sel = jj > ii                     # jj is relative to a0 as well
                if not sel.any():
                    continue
                ii, jj = ii[sel], jj[sel]
                pa, pb = mem[ii + a0], mem[jj + a0]
                lo, hi = np.minimum(pa, pb), np.maximum(pa, pb)
                dd = dist[ii, jj]
                for x, y, d in zip(lo.tolist(), hi.tolist(), dd.tolist()):
                    if (x, y) in seen:
                        continue
                    seen.add((x, y))
                    A.append(x)
                    B.append(y)
                    D.append(d)
                    n_found += 1
                    if n_found >= cap:
                        return np.array(A), np.array(B), np.array(D), True
    return np.array(A, dtype=np.int64), np.array(B, dtype=np.int64), np.array(D, dtype=np.int64), False


def find_duplicates(case_ids: list, dhashes: list, sha1s: list, flat: list,
                    groups: Optional[list] = None, fine: Optional[list] = None) -> dict:
    """Exact duplicates (same pixels) and near-duplicates (same picture after re-saving,
    resizing or a brightness change). Near-duplicates: 256-bit-per-view difference hashes
    differing in at most NEAR_FINE_FRAC of their bits, found exhaustively (see
    :func:`_near_candidates`). Memory is bounded for any dataset size."""
    n = len(case_ids)
    keep = [i for i in range(n) if not flat[i] and dhashes[i]]
    res = {"method": "", "n_exact_groups": 0, "n_exact_cases": 0, "exact_groups": [],
           "n_near_pairs": 0, "near_pairs": [], "n_cross_group_pairs": 0,
           "n_excluded_flat": sum(1 for f in flat if f)}
    by: dict = {}
    for i in keep:
        by.setdefault(sha1s[i], []).append(i)
    exact = sorted((sorted(v) for v in by.values() if len(v) > 1), key=lambda g: -len(g))
    res["n_exact_groups"] = len(exact)
    res["n_exact_cases"] = sum(len(g) for g in exact)
    res["exact_groups"] = [[case_ids[i] for i in g] for g in exact[:200]]
    cross_exact = [g for g in exact if groups and len({groups[i] for i in g}) > 1]
    if groups:
        res["exact_cross_groups"] = [[case_ids[i] for i in g] for g in cross_exact[:100]]
    n_cross_exact = sum(1 for g in cross_exact for j in g[1:] if groups[j] != groups[g[0]])
    res["n_cross_group_pairs"] = n_cross_exact
    # one representative per exact group for the near-duplicate search
    rep = sorted(g[0] for g in by.values()) if by else []
    use_fine = fine is not None and any(fine[i] for i in rep)
    hashes = fine if use_fine else dhashes
    if not rep:
        return res
    L = max({len(hashes[i] or "") for i in rep}, key=lambda l: sum(1 for i in rep if len(hashes[i] or "") == l))
    idx = np.array([i for i in rep if hashes[i] and len(hashes[i]) == L])
    if idx.size < 2:
        return res
    F = _hash_array([hashes[i] for i in idx])
    bits = F.shape[1] * 8
    thr = int(NEAR_FINE_FRAC * bits) if use_fine else 3
    a, b, d, truncated = _near_candidates(F, thr)
    H = None
    if use_fine:
        try:
            H = _hash_array([dhashes[i] for i in idx])
        except ValueError:
            H = None
    near = []
    for x, y, dist in zip(a.tolist(), b.tolist(), d.tolist()):
        i, j = int(idx[x]), int(idx[y])
        cross = bool(groups and groups[i] != groups[j])
        dc = int(_hamming(H[x], H[y])) if H is not None and H.ndim == 2 else (None if use_fine else dist)
        near.append({"a": case_ids[i], "b": case_ids[j], "distance": dc,
                     "distance_fine": dist if use_fine else None, "cross": cross})
    near.sort(key=lambda p: (not p["cross"], p["distance_fine"] if p["distance_fine"] is not None
                             else p["distance"]))
    res["n_near_pairs"] = len(near)
    res["n_cross_group_pairs"] = n_cross_exact + sum(1 for p in near if p["cross"])
    res["near_pairs"] = near[:500]
    res["near_threshold_bits"] = thr
    res["near_hash_bits"] = bits
    if truncated:
        res["near_search_stopped"] = MAX_NEAR_PAIRS
    res["method"] = (f"Exact duplicates: identical pixel values. Near-duplicates: the same picture after "
                     f"re-saving, resizing or a brightness change ({bits}-bit difference hash differing in "
                     f"at most {thr} bits, searched exhaustively). Crops and other edits are not "
                     f"detected. Blank/flat images are excluded."
                     + (" ⚠ marks pairs in different top-level folders." if groups else "")
                     + (f" The search stopped after {MAX_NEAR_PAIRS:,} pairs: the counts are a lower "
                        "bound." if truncated else ""))
    return res


# ====================================================================== dataset summary
PROBLEMS = (
    ("flat", "Blank / flat images (no contrast)"),
    ("low_dynamic_range", "Very low dynamic range (< 2 % of the data type range)"),
    ("bright_background", "Bright background: possibly inverted grey scale (or large collimated area)"),
    ("padding", "Large uniform borders / padding (> 25 % of the image)"),
    ("colour", "Colour images (channels differ)"),
    ("blurry", "Much less sharp than the dataset (< 1/3 of the median sharpness)"),
    ("noisy", "Much noisier than the dataset (> 3x the median noise)"),
    ("saturated", "Saturated: > 20 % of pixels at the data-type maximum"),
    ("markers", "Bright marks in the corners (burned-in text / L-R markers; informational)"),
)
QUALITY_MEASURES = ("sharpness", "noise", "bright_background", "frac_border_padding", "colourfulness",
                    "frac_corner_bright", "dynamic_range")


# Reported, but not counted as problems: L/R markers and burned-in text are on almost every
# radiograph (59 % of CheXpert), so they describe the dataset rather than flag images.
INFO_KEYS = ("markers",)


def summarize_quality(rows: list) -> Optional[dict]:
    """rows: [{"case_id", "quality": {...}, "flags": {...}}] for ok cases. Adds the list of
    problem keys to each row's ``quality["problems"]``."""
    from .stats import summarize
    rows = [r for r in rows if r.get("quality")]
    if not rows:
        return None
    q = [r["quality"] for r in rows]
    med_sharp = np.nanmedian([x.get("sharpness", np.nan) for x in q] + [np.nan])
    med_noise = np.nanmedian([x.get("noise", np.nan) for x in q] + [np.nan])
    tests = {
        "flat": lambda x, f: bool(x.get("flat")),
        "low_dynamic_range": lambda x, f: not x.get("flat") and (x.get("dynamic_range") or 1) < 0.02,
        "bright_background": lambda x, f: (x.get("bright_background") or -1) > 0.3,
        "padding": lambda x, f: not x.get("flat") and (x.get("frac_border_padding") or 0) > 0.25,
        "colour": lambda x, f: (x.get("colourfulness") or 0) > 0.02,
        "blurry": lambda x, f: np.isfinite(med_sharp) and not x.get("flat")
        and (x.get("sharpness") or 0) < med_sharp / 3,
        "noisy": lambda x, f: np.isfinite(med_noise) and (x.get("noise") or 0) > 3 * med_noise > 0,
        "saturated": lambda x, f: (f.get("frac_at_dtype_max") or 0) > 0.2,
        "markers": lambda x, f: 0.002 < (x.get("frac_corner_bright") or 0) < 0.2,
    }
    problems = {}
    for r in rows:
        r["quality"]["problems"] = []
    for key, title in PROBLEMS:
        hits = []
        info = key in INFO_KEYS
        for r in rows:
            if tests[key](r["quality"], r.get("flags") or {}):
                hits.append(r["case_id"])
                if not info:
                    r["quality"]["problems"].append(key)
        problems[key] = {"title": title, "count": len(hits), "examples": hits[:20]}
        if info:
            problems[key]["informational"] = True
    return {
        "n_cases": len(rows),
        "problems": problems,
        "measures": {k: summarize([x.get(k) for x in q]) for k in QUALITY_MEASURES},
    }
