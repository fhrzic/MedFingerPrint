"""Command line interface.

    mfp scan --layout totalsegmentator --root /data/Totalsegmentator_dataset_v201 \
             --modality CT --out fingerprints/ts.json --workers 8
    mfp info fingerprints/ts.json
    mfp bins fingerprints/ts.json --preset ct_tissue --region body
    mfp bins fingerprints/ts.json --edges -1000 -500 -190 -30 100 300 --region body
    mfp outliers fingerprints/ts.json --method robust_z --top 20
    mfp report fingerprints/ts.json                      # PDF report
    mfp compare a.json b.json --pdf comparison.pdf
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np


def _progress_printer():
    t0 = time.time()

    def cb(done, total, case_id, status):
        el = time.time() - t0
        rate = done / el if el > 0 and done else 0
        msg = f"\r[{done}/{total}] {status:<6} {str(case_id or '')[:50]:<50} {rate:5.2f} cases/s"
        sys.stderr.write(msg)
        sys.stderr.flush()
        if done == total:
            sys.stderr.write("\n")
    return cb


def _stage_printer():
    """One line per post-processing step (with the time the previous one took)."""
    st = {"msg": None, "t": time.time()}

    def cb(msg, done, total):
        now = time.time()
        if msg != st["msg"]:
            if st["msg"] and not total:
                sys.stderr.write(f"   ({now - st['t']:.0f}s)\n")
            st["msg"], st["t"] = msg, now
            if msg is None:          # finished
                return
            if not total:
                sys.stderr.write(f"Post-processing: {msg}")
                sys.stderr.flush()
        if total:
            sys.stderr.write(f"\rPost-processing: {msg} [{done}/{total}]")
            if done == total:
                sys.stderr.write(f"   ({now - st['t']:.0f}s)\n")
                st["msg"] = None
            sys.stderr.flush()
    return cb


def cmd_scan(a):
    from .core.aggregate import build_fingerprint
    from .core.config import HistogramSpec, ScanConfig
    from .core.discovery import discover
    from .core.runner import run_scan
    from .core.schema import save_fingerprint

    disc = discover(a.layout, a.modality, root=a.root, exts=a.ext, images_dir=a.images,
                    masks_dir=a.masks, labels_json=a.labels_json, match=a.match,
                    meta_csv=a.meta_csv, meta_id_col=a.meta_id_col, annotations=a.annotations,
                    saros_mask=a.saros_mask, only=a.only)
    if disc.case_filter:
        f = disc.case_filter
        print(f"Only cases where {f['rule']}: kept {f['kept']}, left out {f['removed']}")
    if a.limit:
        disc.cases = disc.cases[: a.limit]
    print(f"Found {len(disc.cases)} cases ({sum(1 for c in disc.cases if c.mask)} with masks)")
    for w in disc.warnings[:10]:
        print("  warning:", w)
    if len(disc.warnings) > 10:
        print(f"  ... {len(disc.warnings) - 10} more warnings")
    if not disc.cases:
        sys.exit(1)

    cfg = ScanConfig(
        modality=a.modality,
        histogram=HistogramSpec(*a.hist) if a.hist else None,
        body_method="none" if a.no_body_mask else a.body,
        compute_nnunet=not a.no_nnunet,
        body_threshold_hu=a.body_threshold,
    )
    cache_dir = a.cache or os.path.splitext(a.out)[0] + ".cache"
    scan = run_scan(disc, cfg, cache_dir, workers=a.workers, progress=_progress_printer(),
                    use_cache=not a.no_cache, notify=lambda m: sys.stderr.write(f"\n  NOTE: {m}\n"))
    for w in disc.warnings:
        if w.startswith("CT selected"):
            print("  WARNING:", w)
    errs = [o for o in scan.outcomes if o["status"] != "ok"]
    for e in errs[:5]:
        print(f"  FAILED {e['case_id']}: {e['error']}")
    name = a.name or os.path.splitext(os.path.basename(a.out))[0]
    stage = _stage_printer()
    fp, side, id_map = build_fingerprint(scan, name=name, anonymize=a.anonymize, stage=stage)
    fp["dataset"]["layout"] = a.layout
    stage("Saving the fingerprint (JSON + histograms)", 0, 0)
    save_fingerprint(fp, side, a.out)
    stage(None, 0, 0)
    if a.anonymize:
        with open(os.path.splitext(a.out)[0] + ".idmap.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["public_id", "original_id"])
            w.writerows(id_map.items())
    print(f"Wrote {a.out} ({fp['dataset']['n_ok']} ok, {fp['dataset']['n_failed']} failed)")


def _fmt(v, nd=2):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def cmd_info(a):
    from .core.schema import load_fingerprint
    fp = getattr(a, "_fp", None) or load_fingerprint(a.fingerprint, with_sidecar=False)[0]
    d = fp["dataset"]
    print(f"{d['name']}  modality={d['modality']}  cases={d['n_ok']}/{d['n_cases']}  "
          f"labels={d['has_labels']}  schema={fp['schema']}")
    g = fp["geometry"]
    if g["spacing"]:
        print("spacing median:", [_fmt(s.get("median"), 3) for s in g["spacing"]],
              " shape median:", [_fmt(s.get("median"), 0) for s in g["shape"]])
    print("orientation:", g["orientation"], " dtype:", g["original_dtype"])
    keys = ("count", "mean", "std", "min", "p00_5", "p25", "p50", "p75", "p99_5", "max",
            "skewness", "kurtosis", "entropy_bits")
    print(f"\n{'region':<12}" + "".join(f"{k:>12}" for k in keys))
    for region, r in fp["intensity"].items():
        p = r["pooled"]
        print(f"{region:<12}" + "".join(f"{_fmt(p.get(k)):>12}" for k in keys))
    for key, title in (("nnunet_fingerprint", "nnU-Net (labels)"),
                       ("nnunet_fingerprint_body", "nnU-Net-style (body mask)")):
        nn = fp.get(key)
        if nn:
            props = nn["foreground_intensity_properties_per_channel"].get("0", {})
            print(f"\n{title}: " + "  ".join(f"{k}={_fmt(v)}" for k, v in props.items())
                  + f"  median_rel_size_after_crop={_fmt(nn['median_relative_size_after_cropping'], 3)}")
    fl = fp["flags"]
    if fl.get("body_fraction", {}).get("n"):
        print(f"body region ({fl['body_method']}): median fraction of image "
              f"{_fmt(fl['body_fraction']['median'], 3)}, median threshold "
              f"{_fmt(fl['body_threshold'].get('median'))}")
    if fp["labels"]:
        from .core.aggregate import sampling_note
        note = sampling_note(fp)
        if note:
            print("\n" + note)
        print(f"\n{'label':<28}{'cases':>7}{'med vol ml':>12}{'mean':>10}{'std':>10}{'p50':>10}")
        for k, l in list(fp["labels"].items())[: a.max_labels]:
            pi = l["pooled_intensity"]
            print(f"{k[:27]:<28}{l['n_cases_present']:>7}{_fmt(l['volume_ml'].get('median')):>12}"
                  f"{_fmt(pi['mean']):>10}{_fmt(pi['std']):>10}{_fmt(pi['p50']):>10}")
    plan = fp.get("nnunet_plan")
    if plan and plan.get("configurations"):
        print(f"\nPredicted nnU-Net plan ({plan['foreground_source']}; approximate):")
        for name, c in plan["configurations"].items():
            print(f"  {name:<11} spacing {[round(v, 3) for v in c['spacing']]}  median shape {c['median_shape']}"
                  f"  patch {c['patch_size']}  batch {c['batch_size']}")
        print(f"  normalization: {plan['normalization']}")
    an = fp.get("annotations")
    if an:
        print(f"\nAnnotations ({an.get('format')}): {an['n_objects']} objects in {an['n_images_with_objects']} "
              f"of {an['n_images']} images")
        for cat, c in an["categories"].items():
            print(f"  {cat:<20} objects {c['n_objects']:<6} images {c['n_images']:<6} median area "
                  f"{_fmt(c['area_px'].get('median'), 0)} px  contrast {_fmt(c['contrast'].get('median'))}")
        for k, v in an["checks"].items():
            if v["count"]:
                print(f"  check: {v['title']}: {v['count']}")
    q = fp.get("quality")
    if q:
        hits = [(v["title"], v["count"]) for v in q["problems"].values()
                if v["count"] and not v.get("informational")]
        print("\nQuality checks:" + ("" if hits else " no problems found"))
        for t, n in hits:
            print(f"  {n:>6}  {t}")
        for v in q["problems"].values():
            if v.get("informational") and v["count"]:
                print(f"  (info) {v['title']}: {v['count']} images "
                      f"({100 * v['count'] / max(q.get('n_cases', 1), 1):.0f} %)")
    du = fp.get("duplicates")
    if du:
        print(f"\nDuplicates: {du['n_exact_groups']} exact groups ({du['n_exact_cases']} images), "
              f"{du['n_near_pairs']} near-duplicate pairs, {du['n_cross_group_pairs']} across top-level folders")
    fl = fp["flags"]
    if fl.get("not_hu_suspected"):
        print(f"\nWARNING: {len(fl['not_hu_suspected'])} case(s) may not be in HU:",
              fl["not_hu_suspected"][:10])
    if fp["failed_cases"]:
        print(f"\n{len(fp['failed_cases'])} failed case(s):", fp["failed_cases"][:3])


def cmd_bins(a):
    from .core.bins import BinSet, binset_table, load_presets
    from .core.config import HistogramSpec
    from .core.schema import load_fingerprint
    fp, _ = load_fingerprint(a.fingerprint, with_sidecar=False)
    presets = load_presets()
    if a.list:
        for n, b in presets.items():
            print(f"{n:<22} {b.description}")
        return
    if a.edges:
        bs = BinSet.from_edges(a.edges, open_ends=a.open_ends)
    elif a.file:
        bs = load_presets(a.file)[a.preset] if a.preset else next(iter(load_presets(a.file).values()))
    else:
        bs = presets[a.preset or "ct_tissue"]
    for p in bs.validate():
        print("note:", p)
    spec = HistogramSpec.from_dict(fp["histogram_spec"])
    region = a.region or ("body" if "body" in fp["intensity"] else "all")
    counts = np.asarray(fp["intensity"][region]["histogram"]["counts"])
    print(f"bin set '{bs.name}' on region '{region}'")
    for r in binset_table(counts, spec, bs):
        print(f"  {r['name']:<34}{100 * r['fraction']:8.3f} %   ({r['count']:.0f})")


def cmd_outliers(a):
    from .core.outliers import detect_outliers
    from .core.schema import load_fingerprint
    fp, side = load_fingerprint(a.fingerprint)
    res = detect_outliers(fp, side, method=a.method, threshold=a.threshold, group_by=a.group_by,
                          max_frac=a.max_frac or None)
    n = res["n_cases"]
    multi = sum(1 for e in res["cases"] if e["n_features"] >= 2)
    print(f"method={res['method']} threshold={res['threshold']} group_by={res['group_by']}  "
          f"flagged cases: {len(res['cases'])}/{n} ({100 * len(res['cases']) / max(n, 1):.1f} %), "
          f"on >=2 features: {multi}")
    if a.per_feature:
        for k, v in res["per_feature"].items():
            print(f"    {k:<36} flagged={v['n_flagged']:<7} over_threshold={v['n_over_threshold']:<7} "
                  f"median={v['median']:.4g} ({v['scale']})")
    for e in res["cases"][: a.top]:
        grp = f" [{e['group']}]" if "group" in e else ""
        print(f"  {e['case_id']:<40}{grp} n_features={e['n_features']:<3} max|score|={e['max_score']:.1f}")
        for r in [r for r in res["flags"] if r["case_id"] == e["case_id"]][:4]:
            print(f"      {r['feature']:<34} value={r['value']:.4g}  median={r['dataset_median']:.4g}"
                  f"  score={r['score']:.1f}")


def cmd_compare(a):
    from .core import compare as C
    fps = [C.rescale_fp(C.load_for_compare(p), a.scale) for p in a.fingerprints]
    names = [C.name_of(fp) for fp in fps]
    if C.scale_note(fps):
        print(C.scale_note(fps) + "\n")
    region = a.region or ("body" if "body" in C.common_regions(fps) else C.common_regions(fps)[0])
    rows = C.stats_rows(fps, region)
    w = max(12, *(len(n) for n in names)) + 2
    print(f"{'metric':<44}" + "".join(f"{n[:w - 2]:>{w}}" for n in names)
          + (f"{'delta %':>10}" if len(fps) == 2 else ""))
    sec = None
    for s_, metric, vals, _ in rows:
        if s_ != sec:
            print(f"\n[{s_}]")
            sec = s_
        line = f"  {metric[:42]:<42}" + "".join(f"{_fmt(v, 3)[:w - 1]:>{w}}" for v in vals)
        if len(fps) == 2:
            rel = C.difference(vals[0], vals[1])[1]
            line += f"{'' if rel is None else f'{rel:+.1f}':>10}"
        print(line)
    if len(fps) >= 2:
        for m in C.DISTANCES:
            D = C.distance_matrix(fps, region, m)
            print(f"\ndistance {m} ({region}):")
            for i, n in enumerate(names):
                print(f"  {n[:30]:<30}" + "".join(f"{D[i, j]:>12.4g}" for j in range(len(names))))
    if a.pdf:
        from .report import compare_report
        compare_report(fps, a.pdf, region)
        print(f"\nWrote {a.pdf}")
    if a.csv:
        import csv as _csv
        with open(a.csv, "w", newline="") as fh:
            wr = _csv.writer(fh)
            wr.writerow(["section", "metric"] + names)
            for s_, metric, vals, _ in rows:
                wr.writerow([s_, metric] + ["" if v is None else v for v in vals])
        print(f"\nWrote {a.csv}")


def cmd_report(a):
    from .core.outliers import detect_outliers
    from .core.schema import load_fingerprint
    from .report import fingerprint_report
    fp, side = load_fingerprint(a.fingerprint)
    res = None
    if a.method != "robust_z" or a.group_by:
        res = detect_outliers(fp, side, method=a.method, group_by=a.group_by)
    out = a.out or os.path.splitext(a.fingerprint)[0] + ".pdf"
    fingerprint_report(fp, out, side, outliers=res, max_outliers=a.max_outliers)
    print(f"Wrote {out}")


def cmd_split(a):
    import json as _json

    from .core.split import describe_choice, group_choices, save_split, split_fingerprint
    with open(a.fingerprint) as fh:
        fp = _json.load(fh)
    if not a.by:
        print("Possible --by values:")
        for g in group_choices(fp):
            print(f"  {g:<28} {describe_choice(fp, g, 5)}")
        return
    parts = split_fingerprint(fp, a.by, min_cases=a.min_cases)
    out_dir = a.out_dir or os.path.splitext(a.fingerprint)[0] + "_split_" + a.by.replace(":", "_")
    for p in save_split(parts, out_dir):
        print("Wrote", p)


def cmd_gui(a):
    try:
        from .gui.app import main as gui_main
    except ImportError as e:
        sys.exit(f"GUI dependencies missing ({e}). Install with: pip install -e '.[gui]'")
    sys.exit(gui_main([a.fingerprint] if a.fingerprint else []))


def build_parser():
    from .core.discovery import EXTENSION_GROUPS, LAYOUTS
    p = argparse.ArgumentParser(prog="mfp", description="Medical imaging dataset fingerprinting")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="scan a dataset and write a fingerprint JSON")
    s.add_argument("--layout", choices=LAYOUTS, default="files")
    s.add_argument("--root", nargs="+", help="dataset root (files / nnunet / totalsegmentator / saros layouts); "
                   "several folders (e.g. train valid test) are scanned as one dataset")
    s.add_argument("--saros-mask", choices=("body-regions", "body-parts"), default="body-regions",
                   help="saros layout: which SAROS label map to use (default body-regions)")
    s.add_argument("--images", nargs="+", help="images folder(s) (paired layout)")
    s.add_argument("--masks", nargs="+", help="masks folder(s) (paired layout), one per images folder")
    s.add_argument("--match", choices=("name", "sorted"), default="name")
    s.add_argument("--labels-json", help="label mapping JSON ({value: name} or nnU-Net dataset.json)")
    s.add_argument("--meta-csv", nargs="+", help="metadata CSV(s), one row per image/case (e.g. FracAtlas "
                                      "dataset.csv; CheXpert train.csv valid.csv); the id column is "
                                      "detected automatically for each CSV")
    s.add_argument("--meta-id-col", help="id column of --meta-csv (default: automatic)")
    s.add_argument("--annotations", nargs="+", help="object annotations: COCO/VGG .json, or a YOLO (.txt) or "
                                         "Pascal VOC (.xml) folder (2D images)")
    s.add_argument("--ext", nargs="+", default=["nifti"],
                   help=f"extensions or groups: {', '.join(EXTENSION_GROUPS)} (or e.g. .png)")
    s.add_argument("--modality", choices=("CT", "2D", "3D"), default="CT")
    s.add_argument("--hist", nargs=3, type=float, metavar=("LO", "HI", "WIDTH"),
                   help="fine histogram range (default: CT -1024 3072 1; auto otherwise)")
    s.add_argument("--body-threshold", type=float, default=-500.0)
    s.add_argument("--body", default="auto",
                   help="body/foreground region without masks: auto (CT: -500 HU, else Otsu), "
                        "hu, otsu, none, or a threshold value, e.g. 30")
    s.add_argument("--no-body-mask", action="store_true", help="same as --body none")
    s.add_argument("--no-nnunet", action="store_true")
    s.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    s.add_argument("--out", required=True)
    s.add_argument("--name")
    s.add_argument("--cache", help="cache dir (default: <out>.cache)")
    s.add_argument("--no-cache", action="store_true")
    s.add_argument("--anonymize", action="store_true",
                   help="replace case ids with case_00001..., drop paths (writes <out>.idmap.csv)")
    s.add_argument("--limit", type=int, help="only the first N cases (testing)")
    s.add_argument("--only", help="scan only cases whose metadata matches, e.g. modality=CT, "
                                  "'split=train,validation', 'modality!=MRI; split=train'")
    s.set_defaults(func=cmd_scan)

    s = sub.add_parser("info", help="print a fingerprint summary")
    s.add_argument("fingerprint")
    s.add_argument("--max-labels", type=int, default=40)
    s.set_defaults(func=cmd_info)

    s = sub.add_parser("bins", help="apply a bin set to a fingerprint")
    s.add_argument("fingerprint")
    s.add_argument("--preset")
    s.add_argument("--file", help="YAML file with bin sets")
    s.add_argument("--edges", nargs="+", type=float, help="custom edges, e.g. 0 120 180 216")
    s.add_argument("--open-ends", action="store_true")
    s.add_argument("--region")
    s.add_argument("--list", action="store_true")
    s.set_defaults(func=cmd_bins)

    s = sub.add_parser("outliers", help="list outlier cases")
    s.add_argument("fingerprint")
    s.add_argument("--method", choices=("robust_z", "zscore", "iqr"), default="robust_z")
    s.add_argument("--threshold", type=float)
    s.add_argument("--top", type=int, default=20)
    s.add_argument("--group-by", help="score within groups: level:N (N-th path component of the "
                                      "case id) or meta:KEY (e.g. meta:institute)")
    s.add_argument("--per-feature", action="store_true", help="print flagged counts per feature")
    s.add_argument("--max-frac", type=float, default=0.01,
                   help="flag at most this fraction of cases per feature and direction "
                        "(min 5 cases); 0 = threshold only")
    s.set_defaults(func=cmd_outliers)

    s = sub.add_parser("compare", help="compare two or more fingerprints")
    s.add_argument("fingerprints", nargs="+")
    s.add_argument("--region", help="all | body | foreground (default: body if present)")
    s.add_argument("--csv", help="write the statistics table to a CSV file")
    s.add_argument("--pdf", help="write a PDF comparison report")
    s.add_argument("--scale", choices=("original", "robust", "minmax", "bits"), default="original",
                   help="put all datasets on a common 0-255 scale: robust = each dataset's p0.5-p99.5 "
                        "(recommended), minmax = min-max, bits = bit depth (like converting to 8 bit)")
    s.set_defaults(func=cmd_compare)

    s = sub.add_parser("report", help="write a PDF report for one fingerprint")
    s.add_argument("fingerprint")
    s.add_argument("--out", help="PDF path (default: <fingerprint>.pdf)")
    s.add_argument("--method", choices=("robust_z", "zscore", "iqr"), default="robust_z",
                   help="outlier method used in the report")
    s.add_argument("--group-by", help="outlier groups, e.g. level:1 or meta:institute")
    s.add_argument("--max-outliers", type=int, default=24, help="thumbnails of the top N outliers")
    s.set_defaults(func=cmd_report)

    s = sub.add_parser("split", help="one fingerprint per group (folder level or metadata column)")
    s.add_argument("fingerprint")
    s.add_argument("--by", help="level:N or meta:COLUMN (omit to list the possibilities)")
    s.add_argument("--out-dir")
    s.add_argument("--min-cases", type=int, default=3)
    s.set_defaults(func=cmd_split)

    s = sub.add_parser("gui", help="start the graphical application")
    s.add_argument("fingerprint", nargs="?", help="optional fingerprint JSON to open")
    s.set_defaults(func=cmd_gui)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
