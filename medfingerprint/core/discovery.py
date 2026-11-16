"""Find cases under a root folder.

Layouts
-------
files             every matching file under root (recursive). DICOM files are
                  grouped into series per folder when the modality is 3D/CT.
paired            images folder + masks folder, matched by file stem
                  (nnU-Net style ``_0000`` channel suffixes are stripped).

Several folders (e.g. the train / valid / test parts of a downloaded dataset) can be
scanned as one dataset: give them separated by ";" (or as a list). Case ids are then
relative to the common parent folder, so they start with the split name
(``train/img1.png``) and the split can be used to group, split and compare, and
duplicates across splits are flagged as possible leakage. For the paired layout the
n-th images folder is matched with the n-th masks folder.
nnunet            nnU-Net raw dataset folder (imagesTr, labelsTr, dataset.json).
totalsegmentator  ``<root>/<case>/ct.nii.gz`` + ``<root>/<case>/segmentations/*.nii.gz``
                  (one binary mask per structure) + optional ``<root>/meta.csv``.
saros             SAROS (TCIA): CT as DICOM series in collection folders, masks
                  ``case_XXX/body-regions.nii.gz`` (or body-parts) on a 5 mm grid with
                  every 5th slice annotated (255 = not annotated), linked through the
                  series UID in ``Segmentation-Info*.csv``.
"""
from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Optional

from .io import DICOM_EXTS, RASTER_EXTS, SITK_EXTS, image_kind_for, list_dicom_series, split_ext

LAYOUTS = ("files", "paired", "nnunet", "totalsegmentator", "saros")

# SAROS label values (UMEssen/saros-dataset, training/util.py: BodyRegions, BodyParts).
# 255 marks unannotated slices; it is ignored, not counted as a label.
SAROS_MASKS = {
    "body-regions": {1: "subcutaneous_tissue", 2: "muscle", 3: "abdominal_cavity",
                     4: "thoracic_cavity", 5: "bone", 6: "parotid_glands", 7: "pericardium",
                     8: "breast_implant", 9: "mediastinum", 10: "brain", 11: "spinal_cord",
                     12: "thyroid_glands", 13: "submandibular_glands"},
    "body-parts": {1: "torso", 2: "head", 3: "right_leg", 4: "left_leg", 5: "right_arm",
                   6: "left_arm"},
}
SAROS_IGNORE = 255

EXTENSION_GROUPS = {
    "nifti": (".nii", ".nii.gz"),
    "dicom": DICOM_EXTS,
    "mha/nrrd": (".mha", ".mhd", ".nrrd", ".nhdr"),
    "png": (".png",),
    "jpg": (".jpg", ".jpeg"),
    "tiff": (".tif", ".tiff"),
    "bmp": (".bmp",),
}


@dataclass
class MaskSource:
    kind: str                         # "labelmap" | "binary_dir" | "objects"
    path: str
    files: dict = field(default_factory=dict)  # binary_dir: {structure name: path}
    objects: list = field(default_factory=list)  # objects: polygons/boxes (see annotations.py)
    info: dict = field(default_factory=dict)     # objects: {"listed", "format", "width", "height"}


@dataclass
class CaseSpec:
    case_id: str
    image: str
    image_kind: str                   # "sitk" | "dicom_series" | "dicom2d" | "raster"
    series_uid: Optional[str] = None
    mask: Optional[MaskSource] = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CaseSpec":
        d = dict(d)
        if d.get("mask"):
            d["mask"] = MaskSource(**d["mask"])
        return cls(**d)


@dataclass
class Discovery:
    cases: list
    label_names: Optional[dict] = None   # {int value: name}, for label maps
    warnings: list = field(default_factory=list)
    root: Optional[str] = None
    metadata_csv: Optional[dict] = None      # report of attach_metadata_csv
    annotations: Optional[dict] = None       # report of attach_annotations
    label_info: Optional[dict] = None        # how labels were sampled (e.g. SAROS sparse slices)
    case_filter: Optional[dict] = None       # report of filter_cases ("only cases where …")


def normalize_exts(exts) -> tuple:
    out = []
    for e in exts or ():
        e = e.lower().strip()
        if e in EXTENSION_GROUPS:
            out.extend(EXTENSION_GROUPS[e])
        else:
            out.append(e if e.startswith(".") else "." + e)
    return tuple(dict.fromkeys(out))


def _matches(name: str, exts: tuple) -> bool:
    if name.startswith("."):  # hidden files, incl. macOS AppleDouble "._x.png" forks
        return False
    low = name.lower()
    return any(low.endswith(e) for e in exts)


_CHANNEL_SUFFIX = re.compile(r"_\d{4}$")
_MASK_SUFFIX = re.compile(r"[_\-.](mask|masks|seg|segmentation|label|labels|gt|annotation)$", re.I)


def case_key(path: str) -> str:
    stem = split_ext(path)[0]
    return _CHANNEL_SUFFIX.sub("", stem)


def mask_key(path: str) -> str:
    """Key of a mask file: like :func:`case_key`, also dropping ``_mask``/``_seg``/... suffixes."""
    return _MASK_SUFFIX.sub("", case_key(path))


def split_paths(v) -> list[str]:
    """``"a; b"`` / ``["a", "b"]`` / ``None`` -> list of paths."""
    if v is None:
        return []
    items = v if isinstance(v, (list, tuple)) else str(v).split(";")
    return [os.path.expanduser(str(x).strip()) for x in items if str(x).strip()]


def common_root(paths: list[str]) -> str:
    """Folder that case ids are relative to: the folder itself, or the common parent."""
    paths = [os.path.abspath(p) for p in paths]
    if len(paths) == 1:
        return paths[0]
    c = os.path.commonpath(paths)
    return c if c not in paths else os.path.dirname(c)


def _walk(root: str, exts: tuple) -> list[str]:
    out = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for f in sorted(filenames):
            if _matches(f, exts):
                out.append(os.path.join(dirpath, f))
    return out


def discover_files(root, exts, modality: str) -> Discovery:
    exts = normalize_exts(exts)
    if not exts:
        raise ValueError("No file extensions selected")
    roots = split_paths(root)
    if not roots:
        raise ValueError("No dataset folder given")
    for r in roots:
        if not os.path.isdir(r):
            raise ValueError(f"Not a folder: {r}")
    files = sorted(dict.fromkeys(f for r in roots for f in _walk(r, exts)))
    root = common_root(roots)
    cases, warnings = [], []
    dicom_files = [f for f in files if _matches(f, DICOM_EXTS)]
    other = [f for f in files if not _matches(f, DICOM_EXTS)]

    if dicom_files and modality in ("CT", "3D"):
        for d in sorted({os.path.dirname(f) for f in dicom_files}):
            uids = list_dicom_series(d)
            if not uids:
                warnings.append(f"No readable DICOM series in {d}")
            rel = os.path.relpath(d, root)
            for i, uid in enumerate(uids):
                cid = rel if len(uids) == 1 else f"{rel}#{i}"
                cases.append(CaseSpec(cid, d, "dicom_series", series_uid=uid))
    else:
        other = files

    for f in other:
        cases.append(CaseSpec(os.path.relpath(f, root), f, image_kind_for(f, modality)))
    return Discovery(cases, warnings=warnings, root=root)


def _pair(images_dir: str, masks_dir: Optional[str], exts: tuple, match: str, warnings: list) -> list:
    images = _walk(images_dir, exts)
    masks = _walk(masks_dir, exts) if masks_dir else []
    if match == "sorted":
        if len(images) != len(masks):
            raise ValueError(f"'sorted' matching needs equal counts: {len(images)} images vs {len(masks)} masks"
                             f" ({images_dir})")
        return list(zip(images, masks))
    # same sub-folder first (images/train/a.png <-> masks/train/a.png), then the name alone
    by_key: dict = {}
    for keyf in (case_key, mask_key):
        for m in masks:
            sub = os.path.dirname(os.path.relpath(m, masks_dir))
            by_key.setdefault((sub, keyf(m)), m)
    for keyf in (case_key, mask_key):
        for m in masks:
            by_key.setdefault(("*", keyf(m)), m)
    pairs = []
    for im in images:
        sub = os.path.dirname(os.path.relpath(im, images_dir))
        m = by_key.get((sub, case_key(im))) or by_key.get(("*", case_key(im)))
        if m is None:
            warnings.append(f"No mask for {os.path.relpath(im, images_dir)}")
        pairs.append((im, m))
    return pairs


def discover_paired(images_dir, masks_dir, exts, modality: str, match: str = "name") -> Discovery:
    exts = normalize_exts(exts)
    imgs, msks = split_paths(images_dir), split_paths(masks_dir)
    if not imgs:
        raise ValueError("No images folder given")
    if msks and len(msks) != len(imgs):
        raise ValueError(f"{len(imgs)} images folders but {len(msks)} masks folders: give one masks folder "
                         "per images folder, in the same order")
    warnings: list = []
    pairs = []
    for i, d in enumerate(imgs):
        pairs += _pair(d, msks[i] if msks else None, exts, match, warnings)
    images_dir = common_root(imgs)
    cases = [
        CaseSpec(
            case_id=os.path.relpath(im, images_dir),
            image=im,
            image_kind=image_kind_for(im, modality),
            mask=MaskSource("labelmap", m) if m else None,
        )
        for im, m in pairs
    ]
    return Discovery(cases, warnings=warnings, root=images_dir)


def discover_nnunet(dataset_dir: str) -> Discovery:
    with open(os.path.join(dataset_dir, "dataset.json")) as fh:
        ds = json.load(fh)
    ending = ds.get("file_ending", ".nii.gz")
    img_dir = os.path.join(dataset_dir, "imagesTr")
    lab_dir = os.path.join(dataset_dir, "labelsTr")
    warnings = []
    channels = ds.get("channel_names", ds.get("modality", {"0": "?"}))
    if len(channels) > 1:
        warnings.append(f"{len(channels)} channels in dataset.json; only channel 0000 is analysed")
    cases = []
    for f in sorted(os.listdir(img_dir)):
        if not f.endswith("_0000" + ending):
            continue
        cid = f[: -len("_0000" + ending)]
        lab = os.path.join(lab_dir, cid + ending)
        cases.append(CaseSpec(
            case_id=cid,
            image=os.path.join(img_dir, f),
            image_kind=image_kind_for(f, "3D"),
            mask=MaskSource("labelmap", lab) if os.path.isfile(lab) else None,
        ))
        if not os.path.isfile(lab):
            warnings.append(f"No label for {cid}")
    if not cases:
        # Medical Segmentation Decathlon style (also AMOS): files without the _0000 channel
        # suffix, listed in dataset.json "training" (and "validation") as image/label pairs
        for split in ("training", "validation"):
            for e in ds.get(split) or []:
                if not isinstance(e, dict) or not e.get("image"):
                    continue
                img = os.path.normpath(os.path.join(dataset_dir, e["image"]))
                lab = os.path.normpath(os.path.join(dataset_dir, e["label"])) if e.get("label") else None
                if not os.path.isfile(img):
                    warnings.append(f"Listed in dataset.json but missing: {e['image']}")
                    continue
                cid = split_ext(os.path.basename(img))[0]
                has_lab = bool(lab and os.path.isfile(lab))
                if not has_lab:
                    warnings.append(f"No label for {cid}")
                cases.append(CaseSpec(cid, img, image_kind_for(img, "3D"),
                                      mask=MaskSource("labelmap", lab) if has_lab else None,
                                      meta={"split": "train" if split == "training" else "validation"}))
        if cases:
            warnings.append(f"dataset.json in Decathlon style: {len(cases)} labelled cases from its "
                            "training/validation lists (test images without labels are not scanned)")
        if cases and is_amos(ds, cases):
            n_mr = 0
            for c in cases:
                mr = _amos_id(c.case_id) >= 500        # AMOS readme: ids < 500 are CT, the rest MRI
                c.meta = {**c.meta, "modality": "MRI" if mr else "CT"}
                n_mr += mr
            warnings.append(f"AMOS: {len(cases) - n_mr} CT and {n_mr} MRI cases (ids >= 500 are MRI, "
                            "per the AMOS readme), tagged as meta 'modality'; scan one modality with "
                            "'only cases where' modality=CT (or MRI)")
    if not cases:
        warnings.append(f"No cases found: expected imagesTr/<case>_0000{ending} (nnU-Net v2) or "
                        "'training' entries in dataset.json (Decathlon style)")
    labels = ds.get("labels", {})
    label_names = {}
    for name, val in labels.items():
        if isinstance(val, int) and val > 0:   # nnU-Net v2: {"liver": 1}; skip region tuples
            label_names[val] = name
        elif isinstance(val, str) and str(name).isdigit() and int(name) > 0:   # Decathlon: {"1": "liver"}
            label_names[int(name)] = val
    return Discovery(cases, label_names=label_names or None, warnings=warnings, root=dataset_dir)


def _amos_id(case_id: str) -> int:
    m = re.search(r"amos_(\d+)", str(case_id))
    return int(m.group(1)) if m else -1


def is_amos(ds: dict, cases) -> bool:
    return str(ds.get("name", "")).strip().upper() == "AMOS" or \
        (len(cases) > 0 and all(_amos_id(c.case_id) >= 0 for c in cases))


def parse_case_filter(rule) -> list:
    """``"modality=CT"``, ``"split=train,validation; modality!=MRI"`` -> [(key, op, {values})]."""
    out = []
    for part in str(rule or "").split(";"):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^\s*(?:meta:)?([^=!]+?)\s*(!=|=)\s*(.*)$", part)
        if not m:
            raise ValueError(f"Cannot read the case filter {part!r}: use key=value (e.g. modality=CT)")
        vals = [v.strip() for v in m.group(3).split(",") if v.strip()]
        out.append((m.group(1).strip(), m.group(2), vals))
    return out


def filter_cases(disc: "Discovery", rule) -> Optional[dict]:
    """Keep only the cases whose metadata matches ``rule`` (see parse_case_filter). Values
    compare case-insensitively; a case without the key is dropped by '=' and kept by '!='."""
    conds = parse_case_filter(rule)
    if not conds:
        return None
    n0 = len(disc.cases)
    keys = {k for c in disc.cases for k in c.meta}
    for k, _, _ in conds:
        if k not in keys:
            disc.warnings.append(f"Case filter: no case has the metadata column {k!r} "
                                 f"(available: {', '.join(sorted(keys)[:15]) or 'none'})")

    def ok(c):
        for k, op, vals in conds:
            v = str(c.meta.get(k, "")).strip().lower() if k in c.meta else None
            hit = v is not None and v in {x.lower() for x in vals}
            if (op == "=" and not hit) or (op == "!=" and hit):
                return False
        return True

    disc.cases = [c for c in disc.cases if ok(c)]
    rep = {"rule": "; ".join(f"{k}{op}{','.join(v)}" for k, op, v in conds),
           "kept": len(disc.cases), "removed": n0 - len(disc.cases)}
    disc.case_filter = rep
    if not disc.cases:
        disc.warnings.append(f"Case filter {rep['rule']!r} removed all {n0} cases")
    return rep


def read_csv_rows(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as fh:  # TotalSegmentator v2+ has a BOM
        sample = fh.read(4096)
        fh.seek(0)
        delim = max((";", ",", "\t"), key=sample.count)
        return [{(k or "").strip(): (v or "").strip() for k, v in r.items() if k is not None}
                for r in csv.DictReader(fh, delimiter=delim)]


def _read_meta_csv(path: str) -> dict:
    rows = read_csv_rows(path)
    if not rows:
        return {}
    id_col = "image_id" if "image_id" in rows[0] else next(iter(rows[0]))
    return {r[id_col]: {k: v for k, v in r.items() if k != id_col} for r in rows}


def _id_variants(case: "CaseSpec") -> list[str]:
    """Strings a metadata CSV may use to refer to a case (most specific first)."""
    cid = str(case.case_id).replace("\\", "/").split("#")[0]
    base = os.path.basename(cid.rstrip("/"))
    out = [cid, base, split_ext(base)[0], case_key(base)]
    parts = [p for p in cid.split("/") if p]
    out += parts[:-1][::-1]                   # parent folders (e.g. patient or study id)
    return list(dict.fromkeys(v for v in out if v))


def _norm_id(v: str) -> str:
    v = str(v).strip().replace("\\", "/")
    return v[2:] if v.startswith("./") else v


def _csv_keys(v: str) -> list[str]:
    """Index keys for one CSV id value: the value, without extension, and for paths every
    trailing part (``CheXpert-v1.0-small/train/p1/s1/view1.jpg`` -> ``train/p1/s1/view1.jpg``,
    ``p1/s1/view1.jpg``, ...), so ids written relative to another folder still match.
    Ambiguous keys (same key, different rows) are dropped later."""
    v = _norm_id(v)
    if not v:
        return []
    out = []
    parts = [p for p in v.split("/") if p]
    for i in range(len(parts)):
        k = "/".join(parts[i:])
        out.append(k)
        stem, ext = split_ext(k)
        if ext:
            out.append(k[: -len(ext)])
    return out


def _csv_index(rows: list, col: str) -> dict:
    """key -> row index, or -1 where the key points to several different rows."""
    index: dict = {}
    for j, r in enumerate(rows):
        for k in dict.fromkeys(_csv_keys(r.get(col, ""))):
            prev = index.get(k)
            if prev is None:
                index[k] = j
            elif prev != j and prev != -1:
                index[k] = -1
    return index


def _match_rows(variants: list, index: dict) -> tuple[list, int]:
    """Per case the row index (or None), using its most specific id variant that points
    to exactly one row; also the number of cases with only ambiguous matches."""
    out, ambiguous = [], 0
    for vs in variants:
        hit, amb = None, False
        for v in vs:
            j = index.get(v)
            if j is None:
                continue
            if j == -1:
                amb = True
                continue
            hit = j
            break
        out.append(hit)
        ambiguous += hit is None and amb
    return out, ambiguous


def _attach_one_csv(disc: "Discovery", path: str, id_col: Optional[str] = None):
    """Match one CSV to the cases and merge matched rows into ``case.meta``.
    Returns (report, set of matched case indices, set of ambiguous-only case indices)."""
    rows = read_csv_rows(path)
    if not rows:
        return ({"path": path, "id_col": None, "rows": 0, "matched": 0, "unmatched": len(disc.cases),
                 "rows_used": 0, "columns": []}, set(), set())
    cols = list(rows[0].keys())
    if id_col and id_col not in cols:
        id_col = None                    # several CSVs: the given column may exist in only some
    variants = [_id_variants(c) for c in disc.cases]
    all_variants = set().union(*map(set, variants)) if variants else set()

    def plausible(col):    # an id column has many distinct values that look like case ids
        vals = {_norm_id(r.get(col, "")) for r in rows}
        if len(vals) < 0.1 * min(len(rows), max(len(disc.cases), 1)):
            return False
        return any(k in all_variants for v in list(vals)[:2000] for k in _csv_keys(v))

    results = {}
    for col in ([id_col] if id_col else [c for c in cols if plausible(c)]):
        idx = _csv_index(rows, col)
        hits, amb = _match_rows(variants, idx)
        amb_set = {i for i, (h, vs) in enumerate(zip(hits, variants))
                   if h is None and any(idx.get(v) == -1 for v in vs)}
        results[col] = (sum(h is not None for h in hits), -amb, hits, amb_set)
    if not results:
        id_col = id_col or cols[0]
        hits, amb_set = [None] * len(disc.cases), set()
    else:
        id_col = max(results, key=lambda c: results[c][:2])
        _, _, hits, amb_set = results[id_col]
    matched = set()
    for i, (c, j) in enumerate(zip(disc.cases, hits)):
        if j is not None:
            c.meta = {**c.meta, **{k: v for k, v in rows[j].items() if k != id_col}}
            matched.add(i)
    rep = {"path": path, "id_col": id_col, "rows": len(rows), "matched": len(matched),
           "unmatched": len(disc.cases) - len(matched),
           "rows_used": len({j for j in hits if j is not None}),
           "columns": [c for c in cols if c != id_col]}
    return rep, matched, amb_set


def attach_metadata_csv(disc: "Discovery", path, id_col: Optional[str] = None) -> dict:
    """Join one or several CSVs (one row per case/image) into ``case.meta``.

    ``path`` is one path, ``"a.csv; b.csv"`` or a list, e.g. CheXpert ``train.csv`` and
    ``valid.csv`` when both split folders are scanned together. Each CSV is matched on its
    own (its own id column); a case may get columns from several CSVs.

    The id column is found automatically: the column that matches the most cases to
    exactly one row each, by relative path (also when the CSV path starts higher up, e.g.
    CheXpert ``CheXpert-v1.0-small/train/...``), file name, file stem or a parent folder
    name (e.g. FracAtlas ``image_id`` = ``IMG0000019.jpg``). A value that refers to several
    different rows (CheXpert ``view1_frontal.jpg`` in every study; a patient id with one row
    per series) is never used. Returns a small report dict.
    """
    paths = split_paths(path)
    reps, matched, amb = [], set(), set()
    for p in paths:
        rep, m, a = _attach_one_csv(disc, p, id_col)
        reps.append(rep)
        if rep["rows"] == 0:
            disc.warnings.append(f"Metadata CSV {p} is empty")
        elif rep["matched"] == 0:
            disc.warnings.append(f"Metadata CSV {os.path.basename(p)}: no row matched any case "
                                 f"(id column guessed: {rep['id_col']!r})")
        matched |= m
        amb |= a
    if not reps:
        return {"matched": 0, "id_col": None}
    names = ", ".join(os.path.basename(p) for p in paths)
    n_un = len(disc.cases) - len(matched)
    if matched and n_un:
        amb_un = len(amb - matched)
        msg = (f"Metadata CSV {names}: {n_un} of {len(disc.cases)} cases have no row (id column "
               f"{', '.join(dict.fromkeys(repr(r['id_col']) for r in reps if r['matched']))})")
        if amb_un:
            msg += (f"; {amb_un} of them share only a file or folder name with several rows, "
                    "which is not used to match")
        disc.warnings.append(msg)
    if len(reps) == 1:
        rep = reps[0]
    else:
        rep = {"path": "; ".join(paths),
               "id_col": ", ".join(dict.fromkeys(str(r["id_col"]) for r in reps)),
               "rows": sum(r["rows"] for r in reps), "matched": len(matched), "unmatched": n_un,
               "rows_used": sum(r["rows_used"] for r in reps),
               "columns": list(dict.fromkeys(c for r in reps for c in r["columns"])),
               "files": reps}
    disc.metadata_csv = rep
    return rep


_MURA_STUDY = re.compile(r"^study\d+_(positive|negative)$")


def find_study_label_csvs(root) -> list[str]:
    """MURA's ``train_labeled_studies.csv`` / ``valid_labeled_studies.csv`` next to the
    chosen folder(s) or one level above (when train/ or valid/ itself is chosen)."""
    out = []
    for r in split_paths(root):
        for d in dict.fromkeys((os.path.abspath(r), os.path.dirname(os.path.abspath(r)))):
            try:
                names = sorted(os.listdir(d))
            except OSError:
                continue
            out += [os.path.join(d, n) for n in names if n.lower().endswith("_labeled_studies.csv")]
    return list(dict.fromkeys(out))


def attach_study_labels(disc: "Discovery", csv_paths: list) -> Optional[dict]:
    """MURA layout ``<split>/XR_<BODYPART>/patientNNNNN/studyN_<positive|negative>/imageK.png``
    with study-level labels in ``*_labeled_studies.csv`` (rows ``<study path>/,<0|1>``, no
    header). Adds per image: split, body_part, patient_id, study, abnormal (abnormal /
    normal). The CSV label is used; the folder name is the fallback."""
    labels = {}
    for f in csv_paths:
        with open(f, newline="") as fh:
            for row in csv.reader(fh):
                if len(row) >= 2 and row[1].strip() in ("0", "1"):
                    parts = [x for x in row[0].replace("\\", "/").split("/") if x]
                    labels[tuple(parts[-4:])] = row[1].strip()
    matched = from_csv = 0
    for c in disc.cases:
        parts = [x for x in os.path.abspath(c.image).replace("\\", "/").split("/") if x]
        if len(parts) < 5 or not _MURA_STUDY.match(parts[-2]) or not parts[-3].startswith("patient"):
            continue
        key = tuple(parts[-5:-1])                 # split, XR_*, patient, study
        lab = labels.get(key)
        if lab is not None:
            from_csv += 1
        else:
            lab = "1" if parts[-2].endswith("_positive") else "0"
        bp = parts[-4]
        c.meta = {**c.meta, "split": parts[-5],
                  "body_part": bp[3:].lower() if bp.upper().startswith("XR_") else bp,
                  "patient_id": parts[-3], "study": f"{parts[-3]}/{parts[-2].split('_')[0]}",
                  "abnormal": "abnormal" if lab == "1" else "normal"}
        matched += 1
    if not matched:
        return None
    rep = {"path": "; ".join(os.path.basename(f) for f in csv_paths), "id_col": "study folder",
           "rows": len(labels), "matched": matched, "unmatched": len(disc.cases) - matched,
           "columns": ["split", "body_part", "patient_id", "study", "abnormal"]}
    if from_csv < matched:
        disc.warnings.append(f"Study labels: {matched - from_csv} of {matched} images not in "
                             f"{rep['path']}; label taken from the folder name")
    disc.metadata_csv = rep
    return rep


def discover_totalsegmentator(root: str, image_name: str = "ct.nii.gz") -> Discovery:
    meta_path = os.path.join(root, "meta.csv")
    meta = _read_meta_csv(meta_path) if os.path.isfile(meta_path) else {}
    cases, warnings = [], []
    for d in sorted(os.listdir(root)):
        cdir = os.path.join(root, d)
        img = os.path.join(cdir, image_name)
        if not os.path.isfile(img):
            continue
        seg_dir = os.path.join(cdir, "segmentations")
        mask = None
        if os.path.isdir(seg_dir):
            files = {split_ext(f)[0]: os.path.join(seg_dir, f)
                     for f in sorted(os.listdir(seg_dir)) if _matches(f, SITK_EXTS)}
            mask = MaskSource("binary_dir", seg_dir, files)
        else:
            warnings.append(f"No segmentations folder for {d}")
        if meta and d not in meta:
            warnings.append(f"{d} missing from meta.csv")
        cases.append(CaseSpec(d, img, "sitk", mask=mask, meta=meta.get(d, {})))
    if not cases:
        warnings.append(f"No '<case>/{image_name}' found under {root}")
    return Discovery(cases, warnings=warnings, root=root)


def _series_uid_of_dir(d: str, files: list) -> Optional[str]:
    """SeriesInstanceUID from the header of one DICOM file in the folder."""
    import pydicom
    for f in files[:3]:
        try:
            ds = pydicom.dcmread(os.path.join(d, f), stop_before_pixels=True,
                                 specific_tags=["SeriesInstanceUID"])
            uid = str(getattr(ds, "SeriesInstanceUID", "") or "").strip()
            if uid:
                return uid
        except Exception:
            continue
    return None


def discover_saros(root: str, mask: str = "body-regions") -> Discovery:
    """SAROS: ``root`` holds the DICOM collection folders, the NIfTI mask folder
    (``case_XXX/<mask>.nii.gz``, possibly nested, e.g. in an unzipped archive folder) and
    ``Segmentation-Info*.csv``. Cases are SAROS cases whose CT series is present."""
    if mask not in SAROS_MASKS:
        raise ValueError(f"SAROS mask must be one of {tuple(SAROS_MASKS)}")
    warnings: list = []
    csv_path, mask_dirs, series = None, {}, {}
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x != "__MACOSX" and not x.startswith("."))
        base = os.path.basename(d)
        if base.startswith("case_") and any(f.endswith(".nii.gz") and not f.startswith("._") for f in files):
            mask_dirs.setdefault(base, d)
            dirs[:] = []
            continue
        if csv_path is None:
            hit = sorted(f for f in files if f.lower().startswith("segmentation-info") and f.lower().endswith(".csv"))
            if hit:
                csv_path = os.path.join(d, hit[0])
        dcm = sorted(f for f in files if _matches(f, DICOM_EXTS))
        if dcm:
            uid = _series_uid_of_dir(d, dcm)
            if uid:
                series.setdefault(uid, d)
            else:
                warnings.append(f"Could not read a series UID in {d}")
    if csv_path is None:
        raise FileNotFoundError(f"No Segmentation-Info*.csv under {root} (it links SAROS cases to CT series)")
    if not mask_dirs:
        raise FileNotFoundError(f"No case_XXX/*.nii.gz mask folders under {root}")
    rows = read_csv_rows(csv_path)
    cases, no_img, no_mask = [], [], []
    used = set()
    info = {"mask": mask, "ignore": SAROS_IGNORE, "resample": True, "sparse_slices": True}
    for r in rows:
        cid = (r.get("id") or "").strip()
        uid = (r.get("tcia_series_instance_uid") or "").strip()
        if not cid:
            continue
        d = series.get(uid)
        if d is None:
            no_img.append(cid)
            continue
        used.add(uid)
        mpath = os.path.join(mask_dirs.get(cid, ""), mask + ".nii.gz")
        m = MaskSource("labelmap", mpath, info=dict(info)) if cid in mask_dirs and os.path.isfile(mpath) else None
        if m is None:
            no_mask.append(cid)
        meta = {
            "collection": r.get("tcia_collection"), "patient_id": r.get("tcia_case_id"),
            "sex": r.get("gender"), "anatomic_region": r.get("anatomic_region"),
            "breast_implant": r.get("has_breast_implant"), "split": r.get("split"),
        }
        cases.append(CaseSpec(cid, d, "dicom_series", series_uid=uid, mask=m,
                              meta={k: v for k, v in meta.items() if v not in (None, "")}))
    if no_img:
        warnings.append(f"{len(no_img)} of {len(rows)} SAROS cases have no CT series in the folder "
                        f"(not downloaded?), e.g. {', '.join(no_img[:5])}")
    if no_mask:
        warnings.append(f"{len(no_mask)} cases have no {mask}.nii.gz (scanned without labels), "
                        f"e.g. {', '.join(no_mask[:5])}")
    extra = len(set(series) - used)
    if extra:
        warnings.append(f"{extra} DICOM series in the folder are not SAROS cases (ignored)")
    disc = Discovery(cases, label_names=dict(SAROS_MASKS[mask]), warnings=warnings, root=root)
    disc.label_info = {"source": "SAROS", "mask": mask, "ignore_value": SAROS_IGNORE,
                       "sparse": True, "segmentation_csv": os.path.basename(csv_path)}
    return disc


def load_label_names(path: str) -> dict:
    """Load a label mapping JSON. Accepts {"1": "liver"} or {"liver": 1}, or nnU-Net's
    dataset.json (uses its "labels" entry)."""
    with open(path) as fh:
        d = json.load(fh)
    if "labels" in d and isinstance(d["labels"], dict):
        d = d["labels"]
    out = {}
    for k, v in d.items():
        if isinstance(v, int):
            out[int(v)] = str(k)
        else:
            try:
                out[int(k)] = str(v)
            except ValueError:
                continue
    out.pop(0, None)
    return out


def discover(layout: str, modality: str, root: Optional[str] = None, exts=None,
             images_dir: Optional[str] = None, masks_dir: Optional[str] = None,
             labels_json: Optional[str] = None, match: str = "name",
             meta_csv: Optional[str] = None, meta_id_col: Optional[str] = None,
             annotations: Optional[str] = None, saros_mask: str = "body-regions",
             only: Optional[str] = None) -> Discovery:
    if layout == "files":
        disc = discover_files(root, exts, modality)
    elif layout == "paired":
        disc = discover_paired(images_dir or root, masks_dir, exts, modality, match)
    elif split_paths(root) and len(split_paths(root)) > 1:
        raise ValueError(f"The {layout} layout takes one folder (several folders: files or paired layout)")
    elif layout == "nnunet":
        disc = discover_nnunet(split_paths(root)[0])
    elif layout == "totalsegmentator":
        disc = discover_totalsegmentator(split_paths(root)[0])
    elif layout == "saros":
        disc = discover_saros(split_paths(root)[0], saros_mask)
    else:
        raise ValueError(f"layout must be one of {LAYOUTS}")
    if labels_json:
        disc.label_names = load_label_names(labels_json)
    if layout == "files" and root:
        study_csvs = find_study_label_csvs(root)
        if study_csvs:
            attach_study_labels(disc, study_csvs)
    if meta_csv:
        attach_metadata_csv(disc, meta_csv, meta_id_col)
    if annotations:
        from .annotations import attach_annotations
        attach_annotations(disc, annotations)
    if only:
        filter_cases(disc, only)
    return disc
