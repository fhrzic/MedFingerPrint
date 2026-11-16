"""Analyse one case: geometry, intensity statistics per region, fine histograms,
per-label statistics, quality flags, previews, and the nnU-Net fingerprint
quantities (identical semantics to nnU-Net's DatasetFingerprintExtractor).

Regions
-------
all         every voxel/pixel of the image
body        anatomy region from intensities, no masks needed: largest connected
            component above a threshold, holes filled slice-wise (so lungs and
            bowel gas are inside). Threshold: -500 HU for CT, per-image Otsu for
            other modalities, or a user value (``ScanConfig.body_method``).
foreground  voxels whose label value is > 0 (when masks are available)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import ndimage

from .config import ScanConfig
from .cropping import crop_to_nonzero
from .discovery import CaseSpec
from .io import LoadedImage, load_image, read_label_array
from .stats import Moments, describe_from, hist_percentiles

PREVIEW_MAX = 256


# --------------------------------------------------------------------------- helpers
def analysis_channel(img: LoadedImage, gray_mode: str = "luminance") -> np.ndarray:
    a = img.array
    if a.shape[0] == 1:
        return a[0]
    if gray_mode == "luminance" and a.shape[0] >= 3:
        return (0.299 * a[0] + 0.587 * a[1] + 0.114 * a[2]).astype(np.float32)
    if gray_mode.startswith("channel"):
        return a[int(gray_mode.replace("channel", "") or 0)]
    return a.mean(axis=0).astype(np.float32)


def otsu_value(arr: np.ndarray, nbins: int = 128, classes: int = 3) -> float:
    """Background/anatomy threshold (subsampled for speed).

    ``classes=3`` (default): 3-class Otsu (background / soft tissue / dense),
    returning the LOWER threshold. Plain 2-class Otsu on CT-like data often splits
    soft tissue from bone instead of body from background (e.g. head CT).
    ``classes=2``: classic Otsu.
    """
    x = arr[(slice(None, None, 2),) * arr.ndim] if arr.size > 4_000_000 else arr
    x = x[np.isfinite(x)].ravel()
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return lo
    c, edges = np.histogram(x, bins=nbins, range=(lo, hi))
    p = c.astype(np.float64) / c.sum()
    centers = (edges[:-1] + edges[1:]) / 2
    P = np.cumsum(p)
    S = np.cumsum(p * centers)
    if classes == 2:
        w0, w1 = P, 1 - P
        mu0 = np.divide(S, w0, out=np.zeros_like(S), where=w0 > 0)
        mu1 = np.divide(S[-1] - S, w1, out=np.zeros_like(S), where=w1 > 0)
        return float(edges[int(np.argmax(w0 * w1 * (mu0 - mu1) ** 2)) + 1])
    t1, t2 = np.triu_indices(nbins - 1, 1)          # class 0: bins<=t1, 1: (t1,t2], 2: >t2
    w0, w1, w2 = P[t1], P[t2] - P[t1], 1 - P[t2]
    s0, s1, s2 = S[t1], S[t2] - S[t1], S[-1] - S[t2]
    with np.errstate(divide="ignore", invalid="ignore"):
        between = np.where(w0 > 0, s0 ** 2 / w0, 0) + np.where(w1 > 0, s1 ** 2 / w1, 0) \
            + np.where(w2 > 0, s2 ** 2 / w2, 0)
    return float(edges[t1[int(np.argmax(between))] + 1])


def body_mask(arr: np.ndarray, threshold: float) -> np.ndarray:
    """Largest connected component above ``threshold`` with holes filled
    (per slice for volumes). 2D images and 3D volumes (z, y, x)."""
    if arr.ndim == 2:
        above = arr > threshold
        lab, n = ndimage.label(above)
        if n == 0:
            return above
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        return ndimage.binary_fill_holes(lab == sizes.argmax())
    return body_mask_ct(arr, threshold)


def body_mask_ct(vol: np.ndarray, threshold: float = -500.0) -> np.ndarray:
    """Patient body mask for a CT volume (z, y, x). Removes table/air, keeps lungs."""
    above = vol > threshold
    if not above.any():
        return above
    small = above[:, ::2, ::2]  # in-plane downsampling keeps connected components cheap
    lab, n = ndimage.label(small)
    if n == 0:
        return np.zeros_like(above)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    keep = lab == sizes.argmax()
    del lab
    keep = keep.repeat(2, axis=1).repeat(2, axis=2)[:, : vol.shape[1], : vol.shape[2]]
    body = above & keep
    for z in range(body.shape[0]):
        if body[z].any():
            body[z] = ndimage.binary_fill_holes(body[z])
    return body


IGNORE_KEY = "__ignore__"   # label key of voxels excluded from label statistics


def _labelmap_on_image_grid(path: str, geom: dict, info: dict) -> tuple[np.ndarray, dict]:
    """Read a label map stored on a different grid than the image (e.g. SAROS masks on
    a 5 mm grid) and place it on the image grid by physical position, nearest neighbour.

    ``info["ignore"]``: value marking unannotated voxels (SAROS: 255). Voxels outside
    the mask get it too. With ``info["sparse_slices"]`` only slices of the mask that
    are annotated (not entirely ``ignore``) count; each is put on the single nearest
    image slice and every other image slice is set to ``ignore``. Returns the array and
    {"slices", "annotated_slices", "volume_factor"}."""
    import SimpleITK as sitk

    from .io import sitk_read
    ign = int(info.get("ignore", 255))
    mask = sitk_read(path)
    marr = sitk.GetArrayViewFromImage(mask)           # (z, y, x)
    ref = sitk.Image([int(v) for v in geom["size"]], sitk.sitkUInt8)
    ref.SetSpacing(geom["spacing"])
    ref.SetOrigin(geom["origin"])
    ref.SetDirection(geom["direction"])
    small = {sitk.sitkUInt8, sitk.sitkInt8, sitk.sitkUInt16, sitk.sitkInt16}
    ptype = mask.GetPixelID() if mask.GetPixelID() in small and 0 <= ign <= 255 else sitk.sitkInt32
    out = sitk.GetArrayFromImage(sitk.Resample(
        sitk.Cast(mask, ptype), ref, sitk.Transform(), sitk.sitkNearestNeighbor, ign, ptype))
    n_img = out.shape[0]
    stats = {"slices": int(n_img), "annotated_slices": int(n_img), "volume_factor": 1.0}
    if info.get("sparse_slices"):
        keep = set()
        cx, cy = (mask.GetSize()[0] - 1) / 2.0, (mask.GetSize()[1] - 1) / 2.0
        for k in np.flatnonzero((marr != ign).reshape(marr.shape[0], -1).any(axis=1)):
            pt = mask.TransformContinuousIndexToPhysicalPoint((cx, cy, float(k)))
            z = int(round(ref.TransformPhysicalPointToContinuousIndex(pt)[2]))
            if 0 <= z < n_img:
                keep.add(z)
        drop = np.setdiff1d(np.arange(n_img), sorted(keep))
        out[drop] = ign
        stats["annotated_slices"] = len(keep)
        # each annotated slice stands for the stretch of scan up to the next one
        stats["volume_factor"] = n_img / len(keep) if keep else float("nan")
    return out, stats


def load_case_labels(spec: CaseSpec, shape: tuple, cfg: ScanConfig, geom: Optional[dict] = None):
    """Return (local_index_map, keys, is_fg, sampling) or (None, None, None, None).

    ``local_index_map`` holds 0..L-1; ``keys[i]`` is the label name for index i
    (``keys[0] == 'background'``); ``is_fg[i]`` is True where the original label
    value is > 0 (nnU-Net's foreground definition). Voxels whose value is the mask's
    ignore value get the last index, key :data:`IGNORE_KEY`, not foreground.
    ``sampling`` is None, or for sparsely annotated masks the slice counts (see
    :func:`_labelmap_on_image_grid`).
    """
    m = spec.mask
    if m is None:
        return None, None, None, None
    if m.kind == "objects":
        from .annotations import objects_label_map
        lab, keys, _, _ = objects_label_map(m, shape)
        return lab, keys, np.array([False] + [True] * (len(keys) - 1)), None
    if m.kind == "binary_dir":
        import SimpleITK as sitk

        from .io import sitk_read
        names = sorted(m.files)
        local = np.zeros(shape, dtype=np.int16)
        flat = local.reshape(-1)
        for i, name in enumerate(names, start=1):
            img = sitk_read(m.files[name])
            arr = sitk.GetArrayViewFromImage(img)
            if arr.shape != shape:
                raise ValueError(f"Mask '{name}' shape {arr.shape} != image shape {shape}")
            # Most structures span few slices: locate them with a fast max-reduction,
            # then search only those slices.
            per = int(np.prod(shape[1:]))
            zs = np.flatnonzero(arr.reshape(shape[0], -1).max(axis=1))
            if zs.size:
                z0, z1 = int(zs[0]), int(zs[-1]) + 1
                nz = np.flatnonzero(arr[z0:z1] != 0)
                flat[z0 * per + nz] = i
            del arr, img
        keys = ["background"] + names
        is_fg = np.array([False] + [True] * len(names))
        return local, keys, is_fg, None

    info = m.info or {}
    sampling, ign_mask = None, None
    if info.get("resample"):
        if geom is None:
            raise ValueError("This mask must be placed by physical position, but the image has no geometry")
        arr, sampling = _labelmap_on_image_grid(m.path, geom, info)
    else:
        arr = read_label_array(m.path)
    if arr.ndim == shape.__len__() + 1 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.shape != shape:
        raise ValueError(f"Label map shape {arr.shape} != image shape {shape}")
    if "ignore" in info:
        ign_mask = arr == int(info["ignore"])
        if ign_mask.any():
            arr = np.where(ign_mask, 0, arr)
        else:
            ign_mask = None
    vmin = int(arr.min())
    if int(arr.max()) - vmin <= 65535:
        counts = np.bincount((arr - vmin).ravel())
        values = np.flatnonzero(counts) + vmin
        if 0 not in values:
            values = np.sort(np.append(values, 0))
        values = np.concatenate([[0], values[values != 0]])  # background first
        lut = np.zeros(int(arr.max()) - vmin + 1, dtype=np.int16)
        for i, v in enumerate(values):
            if v - vmin < lut.size and v - vmin >= 0:
                lut[v - vmin] = i
        local = lut[arr - vmin]
    else:  # sparse large codes (colour-coded masks)
        u, inv = np.unique(arr, return_inverse=True)
        values = np.concatenate([[0], u[u != 0]])
        pos = np.searchsorted(values[1:], u) + 1
        pos[u == 0] = 0
        local = pos[inv.reshape(arr.shape)].astype(np.int16)
    from .io import RGB_LABEL_OFFSET
    names = dict(cfg.label_names or {})
    for v in values:
        if v >= RGB_LABEL_OFFSET and int(v) not in names:
            names[int(v)] = f"colour #{int(v) - RGB_LABEL_OFFSET:06x}"
    if not names and set(int(v) for v in values) == {0, 255}:
        names = {255: "255 (mask)"}  # 8-bit binary mask saved as 0/255
    keys = ["background"] + [names.get(int(v), str(int(v))) for v in values[1:]]
    is_fg = values > 0
    if ign_mask is not None:
        local[ign_mask] = len(keys)
        keys.append(IGNORE_KEY)
        is_fg = np.append(is_fg, False)
    return local, keys, is_fg, sampling


def _preview(arr2d: np.ndarray) -> np.ndarray:
    """Small preview (<= PREVIEW_MAX per side), area-averaged (not every n-th pixel) so
    edges stay smooth."""
    a = np.asarray(arr2d, dtype=np.float32)
    step = max(1, int(np.ceil(max(a.shape) / PREVIEW_MAX)))
    if step == 1:
        return np.ascontiguousarray(a)
    h, w = (a.shape[0] // step) * step, (a.shape[1] // step) * step
    if h == 0 or w == 0:
        return np.ascontiguousarray(a[::step, ::step])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN blocks stay NaN
        return np.ascontiguousarray(np.nanmean(a[:h, :w].reshape(h // step, step, w // step, step), axis=(1, 3)))


def _effective_bits(vmin: float, vmax: float) -> int:
    rng = max(0.0, vmax - min(vmin, 0.0))
    return int(np.ceil(np.log2(rng + 1))) if rng > 0 else 1


class _Region:
    """Accumulates histogram + moments + min/max for one region across chunks."""

    def __init__(self, size: int):
        self.hist = np.zeros(size, dtype=np.int64)
        self.mom = Moments.empty()
        self.vmin, self.vmax = np.inf, -np.inf

    def add(self, x: np.ndarray, idx: np.ndarray):
        if x.size == 0:
            return
        self.hist += np.bincount(idx, minlength=self.hist.size)
        self.mom = self.mom.merge(Moments.from_values(x))
        self.vmin = min(self.vmin, float(x.min()))
        self.vmax = max(self.vmax, float(x.max()))

    def describe(self, spec, integer_data) -> dict:
        return describe_from(self.mom, self.hist, spec, self.vmin, self.vmax, integer_data)


def _nnunet_case(data: np.ndarray, fg: np.ndarray, is3d: bool, spacing: tuple,
                 image_kind: str, cfg: ScanConfig):
    """nnU-Net's per-case fingerprint step (crop to nonzero, sample foreground
    with RandomState(1234)) for a given boolean foreground mask."""
    seg = fg.astype(np.int8)[None]
    if not is3d:  # nnU-Net natural-image IO yields (C, 1, Y, X)
        data = data[:, None]
        seg = seg[:, None]
        nn_spacing = [999.0, 1.0, 1.0] if image_kind == "raster" else [max(spacing) * 999.0, *spacing]
    else:
        nn_spacing = list(spacing)
    data_c, seg_c, bbox = crop_to_nonzero(data, seg)
    fgm = seg_c[0] > 0
    rs = np.random.RandomState(cfg.nnunet_seed)
    k = int(cfg.nnunet_samples_per_case)
    samples = np.zeros((data.shape[0], k if fgm.any() else 0), dtype=np.float32)
    for c in range(data.shape[0]):
        fp = data_c[c][fgm]
        if fp.size:
            samples[c] = rs.choice(fp, k, replace=True)
    shape_before, shape_after = data.shape[1:], data_c.shape[1:]
    info = {
        "spacing": nn_spacing,
        "shape_after_crop": [int(v) for v in shape_after],
        "relative_size_after_cropping": float(np.prod(shape_after) / np.prod(shape_before)),
        "bbox": bbox,
    }
    return info, samples


# --------------------------------------------------------------------------- main
def analyze_case(spec: CaseSpec, cfg: ScanConfig) -> tuple[dict, dict]:
    hs = cfg.histogram
    img = load_image(spec.image_kind, spec.image, spec.series_uid)
    gray = analysis_channel(img, cfg.gray_mode)
    is3d = gray.ndim == 3
    shape = tuple(int(s) for s in gray.shape)

    local, keys, is_fg, sampling = load_case_labels(spec, shape, cfg, img.geom)
    has_labels = local is not None
    L = len(keys) if has_labels else 0

    body, body_thr = None, None
    method = cfg.resolved_body_method()
    if method != "none":
        if method == "hu":
            body_thr = float(cfg.body_threshold_hu)
        elif method == "otsu":
            body_thr = otsu_value(gray)
        else:
            body_thr = float(method)
        body = body_mask(gray, body_thr)

    regions = {"all": _Region(hs.size)}
    if body is not None:
        regions["body"] = _Region(hs.size)
    if has_labels:
        regions["foreground"] = _Region(hs.size)
        lab_hist = np.zeros(L * hs.size, dtype=np.int64)
        lab_mom = Moments.empty((L,))
        lab_min = np.full(L, np.inf)
        lab_max = np.full(L, -np.inf)

    nan_count = 0
    integer_data = True
    under_values: dict = {}
    per_row = int(np.prod(shape[1:]))
    step = max(1, cfg.chunk_voxels // max(per_row, 1))

    for s in range(0, shape[0], step):
        sl = slice(s, s + step)
        x = gray[sl].ravel()
        finite = np.isfinite(x)
        all_finite = bool(finite.all())
        if not all_finite:
            nan_count += int((~finite).sum())
            x = x[finite]
        if integer_data and x.size:
            integer_data = bool(np.array_equal(x, np.round(x)))
        idx = hs.index(x)
        regions["all"].add(x, idx)

        if cfg.is_ct and x.size and x.min() < hs.lo:
            xu = x[x < hs.lo]
            if integer_data:
                off = int(xu.min())
                c = np.bincount((xu - off).astype(np.int64))
                u = np.flatnonzero(c)
                c = c[u]
                u = u + off
            else:
                u, c = np.unique(xu, return_counts=True)
            top = np.argsort(c)[-20:]  # most frequent sub-range values (padding candidates)
            for uv, cv in zip(u[top], c[top]):
                under_values[float(uv)] = under_values.get(float(uv), 0) + int(cv)

        if body is not None:
            b = body[sl].ravel()
            if not all_finite:
                b = b[finite]
            regions["body"].add(x[b], idx[b])

        if has_labels:
            lv = local[sl].ravel().astype(np.int64)
            if not all_finite:
                lv = lv[finite]
            fg = is_fg[lv]
            regions["foreground"].add(x[fg], idx[fg])
            lab_hist += np.bincount(lv * hs.size + idx, minlength=L * hs.size)
            lab_mom = lab_mom.merge(Moments.from_labels(x, lv, L))
            out_rng = (idx == 0) | (idx == hs.size - 1)
            if out_rng.any():
                xo, lo_ = x[out_rng], lv[out_rng]
                ix = np.unique(lo_)
                lab_min[ix] = np.minimum(lab_min[ix], ndimage.minimum(xo, lo_, index=ix))
                lab_max[ix] = np.maximum(lab_max[ix], ndimage.maximum(xo, lo_, index=ix))

    # ---------------------------------------------------------------- geometry
    spacing = tuple(float(v) for v in img.spacing)
    if len(spacing) != len(shape):
        spacing = tuple([1.0] * (len(shape) - len(spacing))) + spacing[-len(shape):]
    voxel_volume = float(np.prod(spacing))
    geometry = {
        "dims": len(shape),
        "shape": list(shape),
        "spacing": list(spacing),
        "extent_mm": [float(a * b) for a, b in zip(shape, spacing)],
        "n_voxels": int(np.prod(shape)),
        "voxel_volume_mm3": voxel_volume,
        "anisotropy": float(max(spacing) / min(spacing)) if min(spacing) > 0 else float("nan"),
        "orientation": img.orientation,
        "channels": int(img.array.shape[0]),
        "original_dtype": img.original_dtype,
        "photometric": img.photometric,
    }

    # ---------------------------------------------------------------- regions
    region_stats, region_moments, hists = {}, {}, {}
    for name, r in regions.items():
        region_stats[name] = r.describe(hs, integer_data)
        region_moments[name] = r.mom.to_list()
        hists[name] = r.hist
    a = regions["all"]

    # ---------------------------------------------------------------- flags
    n_all = max(int(a.mom.n), 1)
    flags = {"nan_count": nan_count, "integer_valued": integer_data, "notes": list(img.notes)}
    if cfg.is_ct:
        flags["not_hu_suspected"] = bool(a.vmin > -500 or a.vmax <= 2.0)
        # Padding = one dominant constant below the histogram range (e.g. -2048, -3024)
        # covering >= 0.1% of voxels and >= 50% of all sub-range voxels. Air noise
        # spread over -1025, -1030, ... does not qualify.
        n_under = int(regions["all"].hist[0])
        pv, pc = max(under_values.items(), key=lambda kv: kv[1]) if under_values else (None, 0)
        if pv is not None and pc / n_all >= 1e-3 and pc >= 0.5 * n_under:
            flags["padding_value"] = pv
            flags["padding_fraction"] = pc / n_all
        else:
            flags["padding_value"] = None
            flags["padding_fraction"] = 0.0
    else:
        dt = np.dtype(img.original_dtype)
        if np.issubdtype(dt, np.integer):
            lo_v, hi_v = float(np.iinfo(dt).min), float(np.iinfo(dt).max)
            geometry["effective_bits"] = _effective_bits(a.vmin, a.vmax)
        else:
            lo_v, hi_v = a.vmin, a.vmax
        flags["frac_at_dtype_min"] = float((gray == lo_v).sum()) / n_all
        flags["frac_at_dtype_max"] = float((gray == hi_v).sum()) / n_all
        flags["frac_at_image_min"] = float((gray == a.vmin).sum()) / n_all
        flags["frac_at_image_max"] = float((gray == a.vmax).sum()) / n_all

    flags["body_method"] = method
    flags["body_threshold"] = body_thr
    if body is not None:
        flags["body_fraction"] = float(body.sum()) / n_all

    # ---------------------------------------------------------------- labels
    labels_out, label_hist_rows, label_keys_out = {}, [], []
    if has_labels:
        lab_hist = lab_hist.reshape(L, hs.size)
        inner = lab_hist[:, 1:-1] > 0
        has_in = inner.any(axis=1)
        first = np.argmax(inner, axis=1)
        last = hs.nbins - 1 - np.argmax(inner[:, ::-1], axis=1)
        w = 0.0 if integer_data and hs.width == 1 else hs.width
        hmin = np.where(has_in, hs.lo + first * hs.width, np.inf)
        hmax = np.where(has_in, hs.lo + last * hs.width + w, -np.inf)
        lab_min = np.minimum(lab_min, hmin)
        lab_max = np.maximum(lab_max, hmax)
        lsum = lab_mom.summary()
        fg_total = max(int(regions["foreground"].mom.n), 1)
        # Fractions refer to the annotated voxels; with sparse annotation the volume is
        # estimated from the annotated slices (Cavalieri: each stands for its stretch).
        n_ign = int(lab_mom.n[keys.index(IGNORE_KEY)]) if IGNORE_KEY in keys else 0
        n_lab = max(n_all - n_ign, 1)
        vfac = float(sampling["volume_factor"]) if sampling else 1.0
        for i in range(1, L):
            n_i = int(lab_mom.n[i])
            if n_i == 0 or keys[i] == IGNORE_KEY:
                continue
            pct = hist_percentiles(lab_hist[i], hs, qs=(5, 50, 95), vmin=lab_min[i],
                                   vmax=lab_max[i], integer_data=integer_data)
            labels_out[keys[i]] = {
                "voxels": n_i,
                "volume_ml": n_i * voxel_volume * vfac / 1000.0 if is3d else None,
                "frac_of_image": n_i / n_lab,
                "frac_of_foreground": n_i / fg_total,
                "mean": float(lsum["mean"][i]),
                "std": float(lsum["std"][i]),
                "min": float(lab_min[i]),
                "max": float(lab_max[i]),
                **pct,
                "_moments": lab_mom.item(i).to_list(),
            }
            label_hist_rows.append(lab_hist[i])
            label_keys_out.append(keys[i])

    # ---------------------------------------------------------------- nnU-Net
    # Real nnU-Net block (foreground = label > 0) and an nnU-Net-style block with
    # the intensity-derived body mask as foreground (available without masks).
    nnunet, samples, nnunet_body, body_samples = None, None, None, None
    if cfg.compute_nnunet and has_labels:
        nnunet, samples = _nnunet_case(img.array, is_fg[local], is3d, spacing, spec.image_kind, cfg)
    if cfg.compute_nnunet and cfg.nnunet_body_block and body is not None and body.any():
        nnunet_body, body_samples = _nnunet_case(img.array, body, is3d, spacing, spec.image_kind, cfg)

    # ---------------------------------------------------------------- previews
    if is3d:
        previews = {
            "preview_axial": _preview(gray[shape[0] // 2]),
            "preview_coronal": _preview(gray[:, shape[1] // 2, :]),
            "preview_sagittal": _preview(gray[:, :, shape[2] // 2]),
        }
    else:
        previews = {"preview_axial": _preview(gray)}

    # ---------------------------------------------------------------- quality
    quality = None
    if cfg.compute_quality:
        from .quality import case_quality
        quality = case_quality(gray, img.array if img.array.shape[0] >= 3 else None,
                               img.original_dtype, cfg.is_ct)

    # ---------------------------------------------------------------- annotations
    objects, object_notes = None, []
    if spec.mask is not None and spec.mask.kind == "objects" and not is3d:
        from .annotations import _scale_objects, measure_objects
        objs, object_notes = _scale_objects(spec.mask.objects, spec.mask.info or {}, shape)
        objects = measure_objects(gray, objs, spacing)
        for o, src in zip(objects, objs):
            if src.get("rle"):
                o["rle"] = True

    result = {
        "case_id": spec.case_id,
        "status": "ok",
        "geometry": geometry,
        "regions": region_stats,
        "region_moments": region_moments,
        "flags": flags,
        "labels": labels_out,
        "label_sampling": sampling,
        "label_keys": label_keys_out,
        "nnunet": nnunet,
        "nnunet_body": nnunet_body,
        "meta": {**img.meta, **spec.meta},
    }
    if quality is not None:
        result["quality"] = quality
    if objects is not None:
        result["objects"] = objects
        result["object_notes"] = object_notes
        result["annotation_listed"] = bool((spec.mask.info or {}).get("listed"))
    arrays = {f"hist_{k}": v.astype(np.int64) for k, v in hists.items()}
    if label_hist_rows:
        arrays["label_hist"] = np.stack(label_hist_rows).astype(np.int64)
    if samples is not None:
        arrays["nnunet_samples"] = samples
    if body_samples is not None:
        arrays["nnunet_body_samples"] = body_samples
    arrays.update(previews)
    return result, arrays
