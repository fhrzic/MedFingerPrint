"""Combine per-case results into the dataset fingerprint (JSON + sidecar arrays)."""
from __future__ import annotations

import datetime as _dt
import hashlib
import os
import platform
from collections import Counter
from typing import Optional

import numpy as np

from .. import SCHEMA_VERSION, __version__
from .bins import BinSet, binset_table, load_presets
from .config import ScanConfig
from .annotations import summarize_annotations
from .outliers import detect_outliers
from .quality import find_duplicates, summarize_quality
from .runner import ScanOutput, load_case_cache
from .stats import Moments, describe_from, summarize

PER_CASE_SUMMARY_KEYS = ("mean", "std", "p00_5", "p50", "p99_5", "skewness", "kurtosis",
                         "entropy_bits", "min", "max")


def _counts(values) -> dict:
    c = Counter(str(v) for v in values)
    return dict(c.most_common(50))


def _axis_summary(rows: list) -> list:
    if not rows:
        return []
    ndim = max(len(r) for r in rows)
    return [summarize([r[i] for r in rows if len(r) == ndim]) for i in range(ndim)]


def _metadata_summary(metas: list) -> dict:
    keys = sorted({k for m in metas for k in m})
    out = {}
    for k in keys:
        vals = [m.get(k) for m in metas if m.get(k) not in (None, "")]
        nums = []
        for v in vals:
            try:
                nums.append(float(v))
            except (TypeError, ValueError):
                nums = None
                break
        if nums is not None and len(set(nums)) > 12:
            out[k] = {"type": "numeric", **summarize(nums)}
        else:
            out[k] = {"type": "categorical", "counts": _counts(vals), "n": len(vals)}
    return out


def _duplicates(dup):
    if not dup["ids"]:
        return None
    tops = [str(c).replace("\\", "/").split("/")[0] if "/" in str(c) else "" for c in dup["ids"]]
    groups = tops if len(set(tops)) > 1 else None
    return find_duplicates(dup["ids"], dup["dhash"], dup["sha1"], dup["flat"], groups, fine=dup["fine"])


def _csv_report(rep):
    if not rep:
        return None
    out = {}
    for k, v in rep.items():
        if k == "path":
            v = "; ".join(os.path.basename(x.strip()) for x in str(v).split(";"))
        elif k == "files":
            v = [_csv_report(r) for r in v]
        out[k] = v
    return out


def _sampling_summary(rows: list, info: Optional[dict]) -> Optional[dict]:
    """Sparse annotation (e.g. SAROS): how many slices per case carry labels."""
    if not rows and not info:
        return None
    out = dict(info or {})
    if rows:
        ann = [r["annotated_slices"] for r in rows]
        tot = [r["slices"] for r in rows]
        out.update({
            "n_cases": len(rows),
            "annotated_slices": summarize(ann),
            "slices": summarize(tot),
            "frac_slices_annotated": float(sum(ann) / max(sum(tot), 1)),
            "volume_factor": summarize([r["volume_factor"] for r in rows]),
        })
    return out


def sampling_note(fp: dict) -> Optional[str]:
    """One-sentence explanation of sparse label sampling (None for ordinary masks)."""
    ls = fp.get("label_sampling") or {}
    if not ls.get("n_cases"):
        return None
    src = ls.get("source") or "the masks"
    return (f"Labels from {src} {ls.get('mask') or ''} masks are sparse: only "
            f"{100 * ls.get('frac_slices_annotated', 0):.0f}% of CT slices are annotated (median "
            f"{ls['annotated_slices'].get('median', 0):.0f} of {ls['slices'].get('median', 0):.0f} per case; "
            f"value {ls.get('ignore_value', '?')} = not annotated, ignored). Label intensities use "
            "the annotated slices only; volumes are estimated by scaling each annotated slice "
            "to the stretch of scan it represents (median factor "
            f"{ls['volume_factor'].get('median', 0):.1f}); voxel counts and fractions refer to the "
            "annotated slices.").replace("  ", " ")


def build_fingerprint(scan: ScanOutput, name: str = "dataset", anonymize: bool = False,
                      binsets: Optional[list] = None, outlier_method: str = "robust_z",
                      stage=None):
    """``stage(message, done, total)`` (optional) reports the post-processing steps;
    ``total`` is 0 when a step has no known length."""
    def _stage(msg, done=0, total=0):
        if stage:
            stage(msg, done, total)

    cfg: ScanConfig = scan.config
    disc = scan.discovery
    spec = cfg.histogram
    root = disc.root

    cases_out, failed = [], []
    region_hist: dict = {}
    region_case_hists: dict = {}
    region_mom: dict = {}
    region_min: dict = {}
    region_max: dict = {}
    label_hist: dict = {}
    label_mom: dict = {}
    label_rows: dict = {}
    sampling_rows: list = []
    nn_blocks = {key: {"spacings": [], "shapes_after_crop": [], "rel": [], "samples": [], "n_missing": 0}
                 for key in ("nnunet", "nnunet_body")}
    integer_all = True
    case_ids, cache_files = [], {}
    id_map = {}
    ann_rows = []
    q_rows, dup = [], {"ids": [], "dhash": [], "fine": [], "sha1": [], "flat": []}

    ok_specs = {c.case_id: c for c in disc.cases}
    n_all = len(disc.cases)
    every = max(1, n_all // 200)
    for k, (spec_case, outcome) in enumerate(zip(disc.cases, scan.outcomes)):
        if k % every == 0:
            _stage("Reading the results of every image from the cache", k, n_all)
        cid = spec_case.case_id
        pub_id = f"case_{k + 1:05d}" if anonymize else cid
        id_map[pub_id] = cid
        if outcome is None or outcome["status"] != "ok":
            failed.append({"case_id": pub_id, "error": (outcome or {}).get("error", "not run")})
            cases_out.append({"case_id": pub_id, "status": "error"})
            case_ids.append(pub_id)
            continue
        res, arr = load_case_cache(outcome["cache"], skip_previews=True)
        cache_files[pub_id] = outcome["cache"]
        integer_all &= bool(res["flags"].get("integer_valued", True))

        for region, stats in res["regions"].items():
            h = arr[f"hist_{region}"]
            region_hist[region] = region_hist.get(region, 0) + h
            region_case_hists.setdefault(region, {})[len(case_ids)] = h
            m = Moments.from_list(res["region_moments"][region])
            region_mom[region] = region_mom[region].merge(m) if region in region_mom else m
            if stats.get("count"):
                region_min[region] = min(region_min.get(region, np.inf), stats["min"])
                region_max[region] = max(region_max.get(region, -np.inf), stats["max"])

        if res.get("label_sampling"):
            sampling_rows.append(res["label_sampling"])
        lh = arr.get("label_hist")
        for j, lk in enumerate(res.get("label_keys", [])):
            label_hist[lk] = label_hist.get(lk, 0) + lh[j]
        for lk, ls in res.get("labels", {}).items():
            m = Moments.from_list(ls.pop("_moments"))
            label_mom[lk] = label_mom[lk].merge(m) if lk in label_mom else m
            r = label_rows.setdefault(lk, {"voxels": [], "volume_ml": [], "frac_of_foreground": [],
                                           "mean": [], "p50": [], "min": [], "max": []})
            for key in r:
                r[key].append(ls.get(key))

        for key, nn in nn_blocks.items():
            if res.get(key):
                nn["spacings"].append(res[key]["spacing"])
                nn["shapes_after_crop"].append(res[key]["shape_after_crop"])
                nn["rel"].append(res[key]["relative_size_after_cropping"])
                if f"{key}_samples" in arr:
                    nn["samples"].append(arr[f"{key}_samples"])
            else:
                nn["n_missing"] += 1

        qual = res.get("quality")
        if qual:
            dup["ids"].append(pub_id)
            dup["dhash"].append(qual.get("dhash"))
            dup["fine"].append(qual.get("dhash_fine"))
            dup["sha1"].append(qual.get("pixel_sha1"))
            dup["flat"].append(bool(qual.get("flat")))
            qual = {k: v for k, v in qual.items() if k not in ("dhash", "dhash_fine", "pixel_sha1")}
            q_rows.append({"case_id": pub_id, "quality": qual, "flags": res.get("flags") or {}})
        if res.get("objects") is not None:
            ann_rows.append({"case_id": pub_id, "objects": res["objects"],
                             "notes": res.get("object_notes"), "meta": res.get("meta") or {},
                             "listed": res.get("annotation_listed")})

        image_rel = None if anonymize else (
            os.path.relpath(spec_case.image, root) if root else spec_case.image)
        row = {"case_id": pub_id, "status": "ok", "image": image_rel}
        for key in ("geometry", "regions", "flags", "labels", "meta"):
            row[key] = res.get(key)
        for key in ("nnunet", "nnunet_body"):
            if res.get(key):
                row[key] = dict(res[key])
        if qual:
            row["quality"] = qual
        if res.get("objects") is not None:
            row["annotations"] = {"n_objects": len(res["objects"]),
                                  "categories": sorted({o["category"] for o in res["objects"]})}
        cases_out.append(row)
        case_ids.append(pub_id)

    _stage("Reading the results of every image from the cache", n_all, n_all)
    ok_rows = [c for c in cases_out if c["status"] == "ok"]
    if not ok_rows:
        raise RuntimeError("All cases failed; see failed_cases. First error: "
                           + (failed[0]["error"] if failed else "?"))

    # ---------------------------------------------------------------- geometry
    _stage("Summarising geometry, intensities and labels")
    g = [c["geometry"] for c in ok_rows]
    geometry = {
        "dims": _counts([x["dims"] for x in g]),
        "spacing": _axis_summary([x["spacing"] for x in g]),
        "shape": _axis_summary([x["shape"] for x in g]),
        "extent_mm": _axis_summary([x["extent_mm"] for x in g]),
        "anisotropy": summarize([x["anisotropy"] for x in g]),
        "voxel_volume_mm3": summarize([x["voxel_volume_mm3"] for x in g]),
        "n_voxels": summarize([x["n_voxels"] for x in g]),
        "orientation": _counts([x.get("orientation") for x in g]),
        "original_dtype": _counts([x["original_dtype"] for x in g]),
        "channels": _counts([x["channels"] for x in g]),
        "photometric": _counts([x.get("photometric") for x in g]),
    }
    if any("effective_bits" in x for x in g):
        geometry["effective_bits"] = _counts([x.get("effective_bits") for x in g])

    # ---------------------------------------------------------------- intensity
    intensity = {}
    for region, h in region_hist.items():
        pooled = describe_from(region_mom[region], h, spec, region_min.get(region, np.nan),
                               region_max.get(region, np.nan), integer_all)
        per_case = {k: summarize([c["regions"].get(region, {}).get(k) for c in ok_rows])
                    for k in PER_CASE_SUMMARY_KEYS}
        intensity[region] = {
            "n_cases": len(region_case_hists[region]),
            "pooled": pooled,
            "per_case": per_case,
            "histogram": {**spec.to_dict(), "counts": np.asarray(h, dtype=np.int64)},
        }

    # ---------------------------------------------------------------- labels
    labels = {}
    n_ok = len(ok_rows)
    for lk in sorted(label_mom, key=lambda s: (not s.isdigit(), int(s) if s.isdigit() else 0, s)):
        r = label_rows[lk]
        lmin = float(np.nanmin(np.array(r["min"], dtype=float)))
        lmax = float(np.nanmax(np.array(r["max"], dtype=float)))
        labels[lk] = {
            "n_cases_present": len(r["voxels"]),
            "frac_cases_present": len(r["voxels"]) / n_ok,
            "voxels": summarize(r["voxels"]),
            "volume_ml": summarize(r["volume_ml"]),
            "frac_of_foreground": summarize(r["frac_of_foreground"]),
            "case_mean_intensity": summarize(r["mean"]),
            "pooled_intensity": describe_from(label_mom[lk], label_hist[lk], spec, lmin, lmax,
                                              integer_all),
        }

    # ---------------------------------------------------------------- nnU-Net block
    def _nn_block(nn, source):
        if not nn["spacings"]:
            return None
        props = {}
        if nn["samples"]:
            nch = nn["samples"][0].shape[0]
            fg = np.array([np.concatenate([s[c] for s in nn["samples"]]) for c in range(nch)])
            for c in range(nch):
                if fg[c].size == 0:  # every mask empty (e.g. only negative images)
                    continue
                p005, med, p995 = np.percentile(fg[c], np.array((0.5, 50.0, 99.5)))
                props[str(c)] = {
                    "mean": float(np.mean(fg[c])), "median": float(med), "std": float(np.std(fg[c])),
                    "min": float(np.min(fg[c])), "max": float(np.max(fg[c])),
                    "percentile_99_5": float(p995), "percentile_00_5": float(p005),
                }
            del fg
        return {
            "spacings": nn["spacings"],
            "shapes_after_crop": nn["shapes_after_crop"],
            "foreground_intensity_properties_per_channel": props,
            "median_relative_size_after_cropping": float(np.median(nn["rel"])),
            "_foreground_source": source,
            "_n_cases_used": len(nn["spacings"]),
            "_n_cases_missing": nn["n_missing"],
            "_samples_per_case": cfg.nnunet_samples_per_case,
        }

    _stage("Computing the nnU-Net fingerprint values")
    nnunet = _nn_block(nn_blocks["nnunet"], "labels > 0 (identical to nnU-Net)")
    nnunet_body = _nn_block(nn_blocks["nnunet_body"],
                            f"intensity body mask ({cfg.resolved_body_method()}); NOT nnU-Net's "
                            "label-based definition, use to compare datasets without masks")

    # ---------------------------------------------------------------- flags / meta
    fl = [c["flags"] for c in ok_rows]
    flags = {
        "cases_with_nan": sum(1 for f in fl if f.get("nan_count")),
        "notes": _counts([n for f in fl for n in f.get("notes", [])]),
    }
    if cfg.is_ct:
        flags["not_hu_suspected"] = [c["case_id"] for c in ok_rows if c["flags"].get("not_hu_suspected")]
        flags["padding_values"] = _counts([f.get("padding_value") for f in fl if f.get("padding_value") is not None])
    else:
        for k in ("frac_at_dtype_min", "frac_at_dtype_max", "frac_at_image_min", "frac_at_image_max"):
            flags[k] = summarize([f.get(k) for f in fl])

    flags["body_method"] = cfg.resolved_body_method()
    flags["body_fraction"] = summarize([f.get("body_fraction") for f in fl])
    flags["body_threshold"] = summarize([f.get("body_threshold") for f in fl])

    _stage("Summarising metadata, annotations and quality checks")
    meta_summary = _metadata_summary([c.get("meta") or {} for c in ok_rows])
    ann_summary = summarize_annotations(ann_rows, getattr(disc, "annotations", None))
    quality_summary = summarize_quality(q_rows)
    _stage(f"Searching for duplicate images among {len(dup['ids']):,} images (about 1 min per 200,000)")
    duplicates = _duplicates(dup)
    _stage("Hashing the file list")
    # file list hash (reproducibility; detects silent dataset changes)
    hasher = hashlib.sha256()
    for c in disc.cases:
        try:
            size = os.path.getsize(c.image) if os.path.isfile(c.image) else 0
        except OSError:
            size = 0
        rel = os.path.relpath(c.image, root) if root else c.image
        hasher.update(f"{rel}:{size}\n".encode())

    fp = {
        "schema": SCHEMA_VERSION,
        "software": {"name": "medfingerprint", "version": __version__,
                     "python": platform.python_version(), "numpy": np.__version__},
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "dataset": {
            "name": name, "layout": None, "modality": cfg.modality,
            "n_cases": len(disc.cases), "n_ok": n_ok, "n_failed": len(failed),
            "file_list_sha256": hasher.hexdigest(), "anonymized": anonymize,
            "has_labels": bool(labels),
            "metadata_csv": _csv_report(getattr(disc, "metadata_csv", None)),
            "case_filter": getattr(disc, "case_filter", None),
        },
        "config": cfg.to_dict(),
        "histogram_spec": spec.to_dict(),
        "geometry": geometry,
        "intensity": intensity,
        "labels": labels,
        "label_names": {str(k): v for k, v in (cfg.label_names or {}).items()},
        "label_sampling": _sampling_summary(sampling_rows, getattr(disc, "label_info", None)),
        "nnunet_fingerprint": nnunet,
        "nnunet_fingerprint_body": nnunet_body,
        "flags": flags,
        "metadata": meta_summary,
        "annotations": ann_summary,
        "quality": quality_summary,
        "duplicates": duplicates,
        "discovery_warnings": disc.warnings[:200],
        "failed_cases": failed,
        "cases": cases_out,
        "local": None if anonymize else {"root": root, "cache_dir": scan.cache_dir,
                                         "cache_files": cache_files},
    }

    # ---------------------------------------------------------------- sidecar
    _stage("Collecting the per-image histograms")
    n = len(case_ids)
    sidecar = {"case_ids": np.array(case_ids), "hist_spec": np.array([spec.lo, spec.hi, spec.width])}
    for region, rows in region_case_hists.items():
        m = np.zeros((n, spec.size), dtype=np.int64)
        for i, h in rows.items():
            m[i] = h
        sidecar[f"hist_{region}"] = m.astype(np.int32) if m.max() < 2 ** 31 else m
    if label_hist:
        names = list(labels.keys())
        sidecar["label_names"] = np.array(names)
        sidecar["label_hist"] = np.stack([label_hist[k] for k in names]).astype(np.int64)

    # ---------------------------------------------------------------- bins + outliers
    if binsets is None:
        presets = load_presets()
        hu = cfg.is_ct and spec.lo <= -1000 and spec.hi >= 2000
        binsets = [presets["ct_tissue"], presets["ct_body_composition"]] if hu else []
    _stage("Predicting the nnU-Net plan and intensity bins")
    try:
        from .plan import predict_plan
        fp["nnunet_plan"] = predict_plan(fp)
    except Exception as e:  # the plan is an extra; never fail the fingerprint for it
        fp["nnunet_plan"] = {"error": f"{type(e).__name__}: {e}"}
    fp["bins"] = apply_binsets(fp, binsets)
    _stage("Finding outliers")
    fp["outliers"] = detect_outliers(fp, sidecar, method=outlier_method)
    return fp, sidecar, id_map


def apply_binsets(fp: dict, binsets: list) -> list:
    from .config import HistogramSpec
    spec = HistogramSpec.from_dict(fp["histogram_spec"])
    out = []
    for bs in binsets:
        entry = {"binset": bs.to_dict(), "problems": bs.validate(), "regions": {}}
        for region, d in fp["intensity"].items():
            entry["regions"][region] = binset_table(np.asarray(d["histogram"]["counts"]), spec, bs)
        out.append(entry)
    return out
