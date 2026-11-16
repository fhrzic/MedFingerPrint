"""Object annotations for 2D datasets: COCO JSON, YOLO txt folders, Pascal VOC XML folders.

Annotations are read into one neutral structure::

    {"format": "coco", "path": ..., "categories": [names],
     "images": {file key: {"width": W | None, "height": H | None,
                           "objects": [{"category": name, "polygon": [[x, y], ...] | None,
                                        "bbox": [x, y, w, h]}]}}}

and attached to the discovered cases as ``MaskSource(kind="objects")``. During the scan
each object is drawn into a label map (so the usual per-label statistics and the
nnU-Net fingerprint work unchanged) and measured on its own: size, shape, position and
contrast against a ring of surrounding pixels (:func:`measure_objects`).

Images that the annotation file does not mention get an empty object list (e.g.
FracAtlas ``Non_fractured``): they count as "no objects", which is what a detection or
segmentation model trained on the dataset would see.
"""
from __future__ import annotations

import glob
import json
import os
import xml.etree.ElementTree as ET
from typing import Optional

import numpy as np

from .discovery import Discovery, MaskSource, _id_variants, _norm_id
from .io import split_ext
from .outliers import _robust_scale

MAX_OBJECTS_PER_IMAGE = 2000


# ====================================================================== readers
def load_annotations(path) -> dict:
    """Read one or several (";"-separated / list) annotation sources and merge them.
    Each entry remembers the folder it came from ("src"), used to pick the right entry
    when the same file name occurs in several splits."""
    from .discovery import split_paths
    paths = split_paths(path)
    if len(paths) == 1:
        return _load_one(paths[0])
    if not paths:
        raise ValueError("No annotation file or folder given")
    parts = [_load_one(p) for p in paths]
    images, orphans = {}, []
    for i, a in enumerate(parts):
        for k, e in a["images"].items():
            e.setdefault("match", [k])
            images[k if k not in images else f"{paths[i]}::{k}"] = e
        orphans += a.get("orphan_annotations", [])
    fmts = sorted({a["format"] for a in parts})
    return {"format": "+".join(fmts), "path": "; ".join(paths),
            "categories": sorted({c for a in parts for c in a["categories"]}),
            "images": images, "orphan_annotations": orphans}


def _load_one(path: str) -> dict:
    """Read COCO (.json), VGG VIA (.json), YOLO (folder of .txt) or VOC (folder of .xml)."""
    if os.path.isdir(path):
        if glob.glob(os.path.join(path, "**", "*.xml"), recursive=True):
            return _read_voc(path)
        if glob.glob(os.path.join(path, "**", "*.txt"), recursive=True):
            return _read_yolo(path)
        raise ValueError(f"{path}: folder contains neither VOC .xml nor YOLO .txt files")
    with open(path) as fh:
        d = json.load(fh)
    if isinstance(d, dict) and "images" in d and "annotations" in d:
        return _read_coco(d, path)
    if isinstance(d, dict) and d and all(isinstance(v, dict) and "regions" in v for v in d.values()):
        return _read_vgg(d, path)
    if isinstance(d, dict) and "_via_img_metadata" in d:
        return _read_vgg(d["_via_img_metadata"], path)
    raise ValueError(f"{path}: not a COCO or VGG annotation file (COCO needs 'images' and 'annotations')")


def _read_coco(d: dict, path: str) -> dict:
    cats = {c["id"]: str(c.get("name", c["id"])) for c in d.get("categories", [])}
    images, by_id = {}, {}
    for im in d["images"]:
        key = _norm_id(im.get("file_name", str(im.get("id"))))
        entry = {"width": im.get("width"), "height": im.get("height"), "objects": [],
                 "src": os.path.dirname(os.path.abspath(path))}
        images[key] = entry
        by_id[im["id"]] = entry
    orphans = []
    for a in d["annotations"]:
        entry = by_id.get(a.get("image_id"))
        if entry is None:
            orphans.append(a.get("id"))
            continue
        cat = cats.get(a.get("category_id"), str(a.get("category_id")))
        seg = a.get("segmentation")
        polys = []
        if isinstance(seg, list):
            for p in seg:
                if isinstance(p, list) and len(p) >= 2:
                    polys.append(np.asarray(p, dtype=float).reshape(-1, 2).tolist())
        rle = isinstance(seg, dict)
        bbox = [float(v) for v in a["bbox"]] if a.get("bbox") else None
        if not polys:
            entry["objects"].append({"category": cat, "polygon": None, "bbox": bbox, "rle": rle,
                                     "id": a.get("id")})
        for poly in polys:  # a multi-part object is counted per part (as drawn)
            entry["objects"].append({"category": cat, "polygon": poly, "bbox": bbox,
                                     "id": a.get("id")})
    return {"format": "coco", "path": path, "categories": sorted(set(cats.values())),
            "images": images, "orphan_annotations": orphans}


def _read_vgg(d: dict, path: str) -> dict:
    images, cats = {}, set()
    for v in d.values():
        objs = []
        regions = v.get("regions") or []
        regions = regions.values() if isinstance(regions, dict) else regions
        for r in regions:
            sa = r.get("shape_attributes", {})
            ra = r.get("region_attributes", {}) or {}
            cat = str(next(iter(ra.values()), "object")) if ra else "object"
            cats.add(cat)
            if sa.get("name") == "polygon":
                poly = list(zip(sa.get("all_points_x", []), sa.get("all_points_y", [])))
                objs.append({"category": cat, "polygon": [list(map(float, p)) for p in poly], "bbox": None})
            elif sa.get("name") == "rect":
                objs.append({"category": cat, "polygon": None,
                             "bbox": [float(sa["x"]), float(sa["y"]), float(sa["width"]), float(sa["height"])]})
        images[_norm_id(v.get("filename", ""))] = {"width": None, "height": None, "objects": objs,
                                                   "src": os.path.dirname(os.path.abspath(path))}
    return {"format": "vgg", "path": path, "categories": sorted(cats), "images": images}


def _read_voc(folder: str) -> dict:
    images, cats = {}, set()
    for f in sorted(glob.glob(os.path.join(folder, "**", "*.xml"), recursive=True)):
        try:
            root = ET.parse(f).getroot()
        except ET.ParseError:
            continue
        fn = root.findtext("filename") or os.path.basename(f)[:-4]
        size = root.find("size")
        w = int(float(size.findtext("width") or 0)) if size is not None else None
        h = int(float(size.findtext("height") or 0)) if size is not None else None
        objs = []
        for o in root.findall("object"):
            cat = (o.findtext("name") or "object").strip()
            cats.add(cat)
            bb = o.find("bndbox")
            if bb is None:
                continue
            x0, y0 = float(bb.findtext("xmin")), float(bb.findtext("ymin"))
            x1, y1 = float(bb.findtext("xmax")), float(bb.findtext("ymax"))
            objs.append({"category": cat, "polygon": None, "bbox": [x0, y0, x1 - x0, y1 - y0]})
        key = _norm_id(fn)
        if key in images:   # same file name in two sub-folders (train/valid): keep both
            key = _norm_id(os.path.relpath(f, folder))[:-4]
        images[key] = {"width": w or None, "height": h or None, "objects": objs,
                       "src": os.path.dirname(os.path.abspath(f)), "match": [_norm_id(fn)]}
    return {"format": "voc", "path": folder, "categories": sorted(cats), "images": images}


def _read_yolo(folder: str) -> dict:
    """YOLO: one ``<stem>.txt`` per image, lines ``class cx cy w h`` (normalised 0..1) or
    ``class x1 y1 x2 y2 ...`` (normalised polygon, YOLO-seg). Class names from
    ``classes.txt`` / ``*.names`` / ``data.yaml`` if present."""
    names = {}
    for cand in ("classes.txt", "obj.names", "classes.names"):
        p = os.path.join(folder, cand)
        if os.path.isfile(p):
            with open(p) as fh:
                names = {i: ln.strip() for i, ln in enumerate(fh) if ln.strip()}
            break
    for y in (glob.glob(os.path.join(folder, "*.yaml")) + glob.glob(os.path.join(folder, "..", "*.yaml"))
              + glob.glob(os.path.join(folder, "..", "..", "*.yaml"))):
        try:
            import yaml
            with open(y) as fh:
                dd = yaml.safe_load(fh) or {}
            nm = dd.get("names")
            if isinstance(nm, list):
                names = dict(enumerate(map(str, nm)))
            elif isinstance(nm, dict):
                names = {int(k): str(v) for k, v in nm.items()}
            if names:
                break
        except Exception:
            continue
    images, cats = {}, set()
    for f in sorted(glob.glob(os.path.join(folder, "**", "*.txt"), recursive=True)):
        if os.path.basename(f) in ("classes.txt",):
            continue
        objs = []
        with open(f) as fh:
            for ln in fh:
                p = ln.split()
                if len(p) < 5:
                    continue
                cat = names.get(int(float(p[0])), p[0])
                cats.add(cat)
                v = [float(x) for x in p[1:]]
                if len(v) == 4:
                    cx, cy, w, h = v
                    objs.append({"category": cat, "polygon": None, "bbox_norm": [cx - w / 2, cy - h / 2, w, h]})
                else:
                    objs.append({"category": cat, "polygon_norm": np.asarray(v[: len(v) // 2 * 2]).reshape(-1, 2).tolist(),
                                 "polygon": None, "bbox": None})
        stem = split_ext(os.path.basename(f))[0]
        key = stem if stem not in images else _norm_id(os.path.relpath(f, folder))[:-4]
        images[key] = {"width": None, "height": None, "objects": objs,
                       "src": os.path.dirname(os.path.abspath(f)), "match": [stem]}
    return {"format": "yolo", "path": folder, "categories": sorted(cats), "images": images}


# ====================================================================== attach to cases
def attach_annotations(disc: Discovery, path: str) -> dict:
    """Match annotation entries to cases (relative path, file name or stem) and set
    ``case.mask = MaskSource("objects", ...)``. Returns and stores a report."""
    ann = load_annotations(path)
    index: dict = {}
    for key, e in ann["images"].items():
        names = {key, *e.get("match", [])}
        for n in list(names):
            names |= {os.path.basename(n), split_ext(os.path.basename(n))[0]}
        for v in names:
            if key not in index.get(v, []):
                index.setdefault(v, []).append(key)
    used: dict = {}
    ambiguous = []
    n_listed = 0

    def closeness(case, key):
        """Shared leading path parts of the image and the annotation source folder
        (train/images/x.jpg vs train/labels -> shares .../train)."""
        src = ann["images"][key].get("src")
        if not src or not case.image:
            return 0
        a = os.path.abspath(case.image).split(os.sep)
        b = src.split(os.sep)
        n = 0
        while n < min(len(a), len(b)) and a[n] == b[n]:
            n += 1
        return n

    for c in disc.cases:
        key = None
        for v in _id_variants(c)[:4]:
            cand = index.get(v)
            if cand:
                if len(cand) > 1:
                    sc = sorted(((closeness(c, k), k) for k in cand), reverse=True)
                    key = sc[0][1]
                    if sc[0][0] == sc[1][0]:
                        ambiguous.append(v)
                else:
                    key = cand[0]
                break
        entry = ann["images"].get(key) if key else None
        objs = (entry or {}).get("objects", [])[:MAX_OBJECTS_PER_IMAGE]
        if key:
            used[key] = used.get(key, 0) + 1
            n_listed += 1
        info = {"listed": bool(key), "format": ann["format"],
                "width": (entry or {}).get("width"), "height": (entry or {}).get("height")}
        if c.mask is not None and c.mask.kind != "objects":
            disc.warnings.append(f"{c.case_id}: has a mask and annotations; the annotations are used")
        c.mask = MaskSource("objects", ann["path"], objects=objs, info=info)
    ambiguous = list(dict.fromkeys(ambiguous))
    missing = [k.split("::")[-1] for k in ann["images"] if k not in used and ann["images"][k]["objects"]]
    rep = {
        "path": ann["path"], "format": ann["format"], "categories": ann["categories"],
        "n_entries": len(ann["images"]),
        "n_objects_in_file": sum(len(v["objects"]) for v in ann["images"].values()),
        "n_cases_listed": n_listed, "n_cases_not_listed": len(disc.cases) - n_listed,
        "entries_without_image": missing[:200], "n_entries_without_image": len(missing),
        "n_entries_matched_twice": sum(1 for v in used.values() if v > 1),
        "orphan_annotations": len(ann.get("orphan_annotations", [])),
        "ambiguous_names": ambiguous[:20],
    }
    if missing:
        disc.warnings.append(f"Annotations: {len(missing)} annotated image(s) not found among the scanned "
                             f"files (e.g. {missing[0]})")
    if n_listed == 0:
        disc.warnings.append("Annotations: no scanned image matches the annotation file names")
    disc.annotations = rep
    return rep


# ====================================================================== per case
def _scale_objects(objs: list, info: dict, shape: tuple) -> tuple[list, list]:
    """Pixel-coordinate polygons/boxes; normalised YOLO coordinates use the image size."""
    h, w = shape
    notes = []
    aw, ah = info.get("width"), info.get("height")
    if aw and ah and (int(aw), int(ah)) != (w, h):
        if (int(aw), int(ah)) == (h, w):
            notes.append(f"annotation size {aw}x{ah} is the image size rotated by 90° (EXIF orientation?)")
        else:
            notes.append(f"annotation size {aw}x{ah} != image size {w}x{h}")
    out = []
    for o in objs:
        o = dict(o)
        if o.get("bbox_norm"):
            x, y, bw, bh = o["bbox_norm"]
            o["bbox"] = [x * w, y * h, bw * w, bh * h]
        if o.get("polygon_norm"):
            o["polygon"] = [[px * w, py * h] for px, py in o["polygon_norm"]]
        if o.get("polygon") is None and o.get("bbox"):
            x, y, bw, bh = o["bbox"]
            o["polygon"] = [[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]]
            o["from_box"] = True
        if o.get("polygon") is None:
            notes.append("object without polygon or box skipped")
            continue
        out.append(o)
    return out, notes


def _raster(poly, h, w, off=(0, 0), size=None):
    from PIL import Image, ImageDraw
    sh, sw = size or (h, w)
    im = Image.new("L", (sw, sh), 0)
    if len(poly) >= 3:
        ImageDraw.Draw(im).polygon([(x - off[1], y - off[0]) for x, y in poly], fill=1, outline=1)
    elif len(poly) >= 1:
        ImageDraw.Draw(im).line([(x - off[1], y - off[0]) for x, y in poly] * 2, fill=1)
    return np.asarray(im, dtype=bool)


def objects_label_map(mask: MaskSource, shape: tuple):
    """(label map int16, category keys incl. background, scaled objects, notes)."""
    if len(shape) != 2:
        raise ValueError("Object annotations are supported for 2D images only")
    objs, notes = _scale_objects(mask.objects, mask.info or {}, shape)
    cats = sorted({o["category"] for o in objs})
    lut = {c: i + 1 for i, c in enumerate(cats)}
    lab = np.zeros(shape, dtype=np.int16)
    h, w = shape
    for o in objs:
        m = _raster(o["polygon"], h, w)
        lab[m] = lut[o["category"]]
    return lab, ["background"] + cats, objs, notes


def measure_objects(gray: np.ndarray, objs: list, spacing: tuple) -> list[dict]:
    """Per-object measurements on a 2D image (pixel coordinates)."""
    from scipy import ndimage
    h, w = gray.shape
    img_std = float(np.nanstd(gray)) or 1.0
    px_area = float(spacing[-1] * spacing[-2]) if len(spacing) >= 2 else 1.0
    out = []
    boxes = []
    for o in objs:
        poly = np.asarray(o["polygon"], dtype=float).reshape(-1, 2)
        x0, y0 = poly.min(axis=0)
        x1, y1 = poly.max(axis=0)
        outside = bool(x0 < -1 or y0 < -1 or x1 > w + 1 or y1 > h + 1)
        n_vert = int(len(poly)) if not o.get("from_box") else None
        r = {"category": o["category"], "n_vertices": n_vert, "outside_image": outside,
             "from_box": bool(o.get("from_box")), "bbox_w": float(x1 - x0), "bbox_h": float(y1 - y0),
             "cx": float((x0 + x1) / 2 / w), "cy": float((y0 + y1) / 2 / h)}
        r["aspect"] = r["bbox_w"] / r["bbox_h"] if r["bbox_h"] > 0 else None
        # crop with margin for the ring
        side = max(x1 - x0, y1 - y0)
        ring_w = int(np.clip(round(0.5 * side), 3, 64))
        cx0, cy0 = int(max(0, np.floor(x0) - ring_w - 1)), int(max(0, np.floor(y0) - ring_w - 1))
        cx1, cy1 = int(min(w, np.ceil(x1) + ring_w + 2)), int(min(h, np.ceil(y1) + ring_w + 2))
        if cx1 <= cx0 or cy1 <= cy0:
            r.update(area_px=0, degenerate=True)
            out.append(r)
            continue
        m = _raster(poly.tolist(), h, w, off=(cy0, cx0), size=(cy1 - cy0, cx1 - cx0))
        area = int(m.sum())
        r["area_px"] = area
        r["area_mm2"] = area * px_area
        r["area_frac_image"] = area / float(h * w)
        r["degenerate"] = bool(area == 0 or (n_vert is not None and n_vert < 3))
        r["touches_border"] = bool(x0 <= 1 or y0 <= 1 or x1 >= w - 2 or y1 >= h - 2)
        if area:
            crop = gray[cy0:cy1, cx0:cx1]
            ring = ndimage.binary_dilation(m, iterations=ring_w) & ~m
            inside = crop[m]
            r["mean"] = float(np.nanmean(inside))
            r["std"] = float(np.nanstd(inside))
            if ring.any():
                r["ring_mean"] = float(np.nanmean(crop[ring]))
                r["contrast"] = r["mean"] - r["ring_mean"]
                r["contrast_z"] = r["contrast"] / img_std
        boxes.append((x0, y0, x1, y1))
        out.append(r)
    # duplicated objects (same category, IoU > 0.9): double-clicked or merged annotations
    for i in range(len(boxes)):
        for j in range(i):
            if out[i]["category"] != out[j]["category"]:
                continue
            a, b = boxes[i], boxes[j]
            iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
            inter = iw * ih
            union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
            if union > 0 and inter / union > 0.9:
                out[i]["duplicate_of"] = j
    return out


# ====================================================================== dataset summary
HEATMAP_BINS = 20
TINY_PX = 25
OBJECT_KEYS = ("area_px", "area_mm2", "area_frac_image", "bbox_w", "bbox_h", "aspect", "n_vertices",
               "contrast", "contrast_z")


def summarize_annotations(rows: list, report: Optional[dict], anonymize: bool = False) -> Optional[dict]:
    """``rows``: per ok case {"case_id", "objects": [...], "notes": [...], "meta": {...},
    "listed": bool}."""
    from .stats import summarize
    if not rows:
        return None
    cats: dict = {}
    n_with = 0
    per_image = []
    co: dict = {}
    checks = {k: {"title": t, "count": 0, "examples": []} for k, t in (
        ("degenerate", "Degenerate objects (fewer than 3 points or zero area)"),
        ("outside", "Objects extending outside the image"),
        ("tiny", f"Very small objects (< {TINY_PX} pixels)"),
        ("duplicate", "Duplicated objects (same category, boxes overlap > 90 %)"),
        ("size_mismatch", "Images whose annotated size differs from the real size"),
        ("rle", "Objects stored as RLE masks (measured from their box only)"),
    )}

    def hit(k, ex):
        checks[k]["count"] += 1
        if len(checks[k]["examples"]) < 10:
            checks[k]["examples"].append(ex)

    all_objs = []
    for r in rows:
        objs = r.get("objects") or []
        if objs:
            n_with += 1
            per_image.append(len(objs))
        names = sorted({o["category"] for o in objs})
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                co[(a, b)] = co.get((a, b), 0) + 1
        for n in r.get("notes") or []:
            if "annotation size" in n:
                hit("size_mismatch", r["case_id"])
        for o in objs:
            c = cats.setdefault(o["category"], {"n_objects": 0, "images": set(), "vals": {k: [] for k in OBJECT_KEYS},
                                                "hm": np.zeros((HEATMAP_BINS, HEATMAP_BINS), dtype=np.int64)})
            c["n_objects"] += 1
            c["images"].add(r["case_id"])
            for k in OBJECT_KEYS:
                v = o.get(k)
                if isinstance(v, (int, float)) and np.isfinite(v):
                    c["vals"][k].append(float(v))
            ix = int(np.clip(o["cx"] * HEATMAP_BINS, 0, HEATMAP_BINS - 1))
            iy = int(np.clip(o["cy"] * HEATMAP_BINS, 0, HEATMAP_BINS - 1))
            c["hm"][iy, ix] += 1
            ex = r["case_id"]
            if o.get("degenerate"):
                hit("degenerate", ex)
            if o.get("outside_image"):
                hit("outside", ex)
            if 0 < (o.get("area_px") or 0) < TINY_PX:
                hit("tiny", ex)
            if o.get("duplicate_of") is not None:
                hit("duplicate", ex)
            if o.get("rle"):
                hit("rle", ex)
            all_objs.append((r["case_id"], o))

    # count columns in metadata (e.g. FracAtlas fracture_count) vs objects per image
    count_cols = {}
    metas = [r.get("meta") or {} for r in rows]
    for k in sorted({k for m in metas for k in m}):
        if "count" not in k.lower() and not k.lower().startswith(("n_", "num")):
            continue
        mism, n = [], 0
        for r, m in zip(rows, metas):
            try:
                v = int(float(m.get(k)))
            except (TypeError, ValueError):
                continue
            n += 1
            if v != len(r.get("objects") or []):
                mism.append(r["case_id"])
        if n >= 0.5 * len(rows):
            count_cols[k] = mism
    for k, mism in count_cols.items():
        checks[f"count_{k}"] = {"title": f"Images where metadata '{k}' ≠ number of objects",
                                "count": len(mism), "examples": mism[:10]}

    if report:
        checks["not_found"] = {"title": "Annotated images not found among the scanned files",
                               "count": report.get("n_entries_without_image", 0),
                               "examples": report.get("entries_without_image", [])[:10]}
        checks["orphans"] = {"title": "Annotations pointing to a non-existing image id",
                             "count": report.get("orphan_annotations", 0), "examples": []}

    categories = {}
    for name, c in sorted(cats.items()):
        e = {"n_objects": c["n_objects"], "n_images": len(c["images"])}
        for k, v in c["vals"].items():
            e[k] = summarize(v)
        e["position_heatmap"] = c["hm"].tolist()
        area = np.asarray(c["vals"]["area_px"])
        if area.size > 5000:
            area = np.random.default_rng(0).choice(area, 5000, replace=False)
        e["_area_px_values"] = area.round(1).tolist()
        categories[name] = e

    # unusual objects: robust z of log10(area) and log(aspect) per category
    unusual = []
    for name in categories:
        objs = [(cid, o) for cid, o in all_objs if o["category"] == name and (o.get("area_px") or 0) > 0]
        if len(objs) < 20:
            continue
        for key, f in (("area_px", np.log10), ("aspect", np.log)):
            v = np.array([f(o[key]) if o.get(key) else np.nan for _, o in objs], dtype=float)
            ok = np.isfinite(v)
            if ok.sum() < 20:
                continue
            med = float(np.median(v[ok]))
            # log units: a spread below 0.05 (about 5-12 %) is treated as no spread
            z = (v - med) / max(_robust_scale(v[ok], med), 0.05)
            cap = max(5, int(0.01 * ok.sum()))
            for i in np.argsort(-np.nan_to_num(np.abs(z)))[:cap]:
                if np.isfinite(z[i]) and abs(z[i]) > 3.5:
                    cid, o = objs[i]
                    unusual.append({"case_id": cid, "category": name, "feature": key,
                                    "value": o[key], "score": float(z[i])})
    unusual.sort(key=lambda u: -abs(u["score"]))

    return {
        "source": "; ".join(os.path.join(os.path.basename(os.path.dirname(p)), os.path.basename(p))
                            for p in report["path"].split("; ")) if report else None,
        "format": report.get("format") if report else None,
        "n_images": len(rows),
        "n_images_listed": sum(1 for r in rows if r.get("listed")),
        "n_images_with_objects": n_with,
        "n_objects": sum(c["n_objects"] for c in cats.values()),
        "objects_per_image": summarize(per_image),
        "categories": categories,
        "cooccurrence": sorted(([a, b, n] for (a, b), n in co.items()), key=lambda t: -t[2]),
        "checks": checks,
        "unusual_objects": unusual[:200],
    }
