"""Desktop GUI (PySide6 + pyqtgraph).

    mfp gui                 # or: python -m medfingerprint.gui
    mfp gui fp.json         # open an existing fingerprint

Tabs: Scan (choose folder, live progress) | Summary | Histogram & bins | Outliers |
Quality & duplicates | Compare.
The scan runs in a background thread that drives the same multiprocess runner as the CLI.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import time
import traceback
from argparse import Namespace
from collections import Counter

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, QRectF, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QFont, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGridLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QSpinBox, QSplitter, QTableWidget,
    QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from ..core.discovery import EXTENSION_GROUPS, LAYOUTS, SAROS_MASKS, common_root, split_paths
from . import help as H

LAYOUT_HELP = {
    "files": "Every matching file under the folder is one case (DICOM: one case per series).",
    "paired": "Separate image and mask folders, matched by file name.",
    "nnunet": "nnU-Net raw dataset folder (dataset.json, imagesTr, labelsTr).",
    "totalsegmentator": "TotalSegmentator: <case>/ct.nii.gz + <case>/segmentations/*.nii.gz + meta.csv.",
    "saros": "SAROS (TCIA): DICOM CT series + case_XXX/body-regions.nii.gz masks (5 mm, every 5th slice "
             "annotated, 255 = not annotated), linked by Segmentation-Info*.csv. Choose the folder that "
             "holds all of them.",
}
BODY_CHOICES = ("auto", "hu", "otsu", "none")
MONO = QFont("Monospace")
MONO.setStyleHint(QFont.TypeWriter)


# ====================================================================== helpers
def guess_dataset(root: str, max_files: int = 4000) -> dict:
    """Guess layout, modality and file types from a folder (fast, samples files)."""
    out = {"layout": "files", "modality": None, "exts": [], "zips": 0, "n_sampled": 0}
    if not os.path.isdir(root):
        return out
    entries = os.listdir(root)
    if os.path.isfile(os.path.join(root, "dataset.json")) and os.path.isdir(os.path.join(root, "imagesTr")):
        out.update(layout="nnunet", modality="CT", exts=["nifti"])
        return out
    # the nnU-Net / Decathlon folder one level down (e.g. amos22/amos22, an unzipped archive)
    subs = [e for e in sorted(entries) if e != "__MACOSX" and not e.startswith(".")
            and os.path.isdir(os.path.join(root, e))]
    for e in subs[:20]:
        sub = os.path.join(root, e)
        if os.path.isfile(os.path.join(sub, "dataset.json")) and os.path.isdir(os.path.join(sub, "imagesTr")):
            out.update(layout="nnunet", modality="CT", exts=["nifti"], root=sub)
            return out
    if any(e.lower().startswith("segmentation-info") and e.lower().endswith(".csv") for e in entries):
        out.update(layout="saros", modality="CT", exts=["dicom"])
        return out
    sub = [e for e in entries[:200] if os.path.isfile(os.path.join(root, e, "ct.nii.gz"))]
    if sub:
        out.update(layout="totalsegmentator", modality="CT", exts=["nifti"])
        return out
    counts: Counter = Counter()
    n = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d != "__MACOSX")
        # a sample per folder, so a big annotation folder sorted first (FracAtlas/Annotations,
        # 8000 files) cannot use up the budget before the image folders are reached
        for f in filenames[:200]:
            if f.startswith("."):
                continue
            low = f.lower()
            if low.endswith(".zip"):
                out["zips"] += 1
            for g, exts in EXTENSION_GROUPS.items():
                if any(low.endswith(e) for e in exts):
                    counts[g] += 1
            n += 1
        if n >= max_files:
            break
    out["n_sampled"] = n
    if counts:
        top = counts.most_common()
        out["exts"] = [g for g, c in top if c >= 0.05 * top[0][1]]
        main = top[0][0]
        out["modality"] = "2D" if main in ("png", "jpg", "tiff", "bmp") else "CT"
    return out


IMG_DIR_NAMES = ("images", "image", "imgs", "img")
MASK_DIR_NAMES = ("masks", "mask", "labels", "label", "gt", "groundtruth", "ground_truth",
                  "segmentations", "segmentation", "annotations")
SPLIT_NAMES = ("train", "training", "valid", "val", "validation", "test", "testing", "dev")
_IMG_EXTS = tuple(e for g, es in EXTENSION_GROUPS.items() if g != "dicom" for e in es)


def _has_files(d: str, exts: tuple, n: int = 50) -> bool:
    try:
        for i, f in enumerate(os.scandir(d)):
            if f.is_file() and f.name.lower().endswith(exts):
                return True
            if f.is_dir() and i < n and _has_files(f.path, exts, 5):
                return True
            if i >= n:
                break
    except OSError:
        pass
    return False


def _child(d: str, names: tuple):
    try:
        subs = {e.name.lower(): e.path for e in os.scandir(d) if e.is_dir()}
    except OSError:
        return None
    return next((subs[n] for n in names if n in subs), None)


def detect_splits(root: str) -> dict:
    """Downloaded datasets that are already split, scanned as one dataset:

    * ``root/images`` + ``root/masks``                        -> paired (one pair)
    * ``root/<split>/images`` + ``root/<split>/masks``         -> paired, one pair per split
    * ``root/<split>/labels/*.txt`` (YOLO)                     -> annotations per split
    * ``root/<split>/*.json`` COCO (e.g. Roboflow ``_annotations.coco.json``) -> per split
    """
    out = {"splits": [], "images": [], "masks": [], "annotations": []}
    im, mk = _child(root, IMG_DIR_NAMES), _child(root, MASK_DIR_NAMES)
    if im and mk and _has_files(mk, _IMG_EXTS):
        out.update(images=[im], masks=[mk])
        return out
    try:
        subs = sorted(e.path for e in os.scandir(root) if e.is_dir() and not e.name.startswith("."))
    except OSError:
        return out
    splits = [s for s in subs if os.path.basename(s).lower() in SPLIT_NAMES]
    if not splits:
        return out
    out["splits"] = [os.path.basename(s) for s in splits]
    pairs, anns = [], []
    for s in splits:
        im, mk = _child(s, IMG_DIR_NAMES), _child(s, MASK_DIR_NAMES)
        if im and mk and _has_files(mk, _IMG_EXTS):
            pairs.append((im, mk))
        elif mk and _has_files(mk, (".txt", ".xml")):
            anns.append(mk)
        else:
            js = sorted(f for f in os.listdir(s) if f.lower().endswith(".json") and
                        any(w in f.lower() for w in ("coco", "annot", "instances", "via")))
            if js:
                anns.append(os.path.join(s, js[0]))
    if pairs and len(pairs) == len(splits):
        out["images"], out["masks"] = [p[0] for p in pairs], [p[1] for p in pairs]
    out["annotations"] = anns
    return out


SIDE_CSV_NAMES = ("dataset.csv", "meta.csv", "metadata.csv", "labels.csv", "train.csv", "data.csv")


def find_side_files(root: str) -> dict:
    """Metadata CSV / annotation files next to (or one level above) an image folder."""
    out = {"csv": None, "annotations": None}
    root = os.path.abspath(root)
    for d in (root, os.path.dirname(root)):
        try:
            names = os.listdir(d)
        except OSError:
            continue
        if out["csv"] is None:
            for n in SIDE_CSV_NAMES:
                if n in names and not os.path.isfile(os.path.join(d, "dataset.json")):
                    out["csv"] = os.path.join(d, n)
                    break
        if out["annotations"] is None:
            for dirpath, dirnames, filenames in os.walk(d):
                depth = dirpath[len(d):].count(os.sep)
                if depth >= 2:
                    dirnames[:] = []
                if os.path.basename(dirpath).lower().startswith(("annot", "coco")) or dirpath != d:
                    js = [f for f in filenames if f.lower().endswith(".json") and
                          ("coco" in f.lower() or "annot" in f.lower() or "instances" in f.lower())]
                    if js:
                        out["annotations"] = os.path.join(dirpath, sorted(js)[0])
                        break
                dirnames[:] = [x for x in dirnames if x.lower().startswith(("annot", "coco", "label"))]
    if out["csv"] and os.path.basename(out["csv"]) == "meta.csv" and \
            any(os.path.isfile(os.path.join(root, e, "ct.nii.gz")) for e in os.listdir(root)[:50]):
        out["csv"] = None  # TotalSegmentator reads its meta.csv itself
    return out


def _is_amos_folder(d: str) -> bool:
    try:
        with open(os.path.join(d, "dataset.json")) as fh:
            return str(json.load(fh).get("name", "")).strip().upper() == "AMOS"
    except (OSError, ValueError, AttributeError):
        return False


def find_side_csvs(folders: list, splits=()) -> list:
    """Metadata CSVs for the chosen folder(s): per folder ``<folder>.csv`` next to it (CheXpert
    ``train/`` -> ``train.csv``, ``valid/`` -> ``valid.csv``) or the usual names; for a folder
    with split subfolders, ``<split>.csv`` of every split found in it."""
    out = []
    for f in folders:
        f = os.path.abspath(f)
        per_split = [os.path.join(f, s + ".csv") for s in splits if os.path.isfile(os.path.join(f, s + ".csv"))]
        if len(per_split) > 1:
            out += per_split
            continue
        own = os.path.join(os.path.dirname(f), os.path.basename(f) + ".csv")
        if os.path.isfile(own):
            out.append(own)
            continue
        c = find_side_files(f)["csv"]
        if c:
            out.append(c)
    return list(dict.fromkeys(out))


def info_text(fp: dict) -> str:
    from ..cli import cmd_info
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cmd_info(Namespace(fingerprint=None, max_labels=200, _fp=fp))
    return buf.getvalue()


def fmt_time(sec: float) -> str:
    if not np.isfinite(sec) or sec < 0:
        return "–"
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def browse_row(edit: QLineEdit, button_text="Browse…", on_click=None) -> QWidget:
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.addWidget(edit, 1)
    b = QPushButton(button_text)
    b.clicked.connect(on_click)
    lay.addWidget(b)
    return w


# ====================================================================== scan worker
class ScanWorker(QObject):
    stage = Signal(str)
    progress = Signal(int, int, str, str)
    post = Signal(str, int, int)     # post-processing step: message, done, total (0 = unknown)
    log = Signal(str)
    finished = Signal(str)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, params: dict):
        super().__init__()
        self.p = params
        self._cancel = False

    def cancel(self):
        self._cancel = True

    @Slot()
    def run(self):
        from ..core.aggregate import build_fingerprint
        from ..core.config import ScanConfig
        from ..core.discovery import discover
        from ..core.runner import ScanCancelled, run_scan
        from ..core.schema import save_fingerprint
        p = self.p
        try:
            self.stage.emit("Finding files…")
            disc = discover(p["layout"], p["modality"], root=p["root"], exts=p["exts"] or None,
                            images_dir=p.get("images"), masks_dir=p.get("masks"),
                            labels_json=p.get("labels_json") or None,
                            meta_csv=p.get("meta_csv") or None, annotations=p.get("annotations") or None,
                            saros_mask=p.get("saros_mask") or "body-regions", only=p.get("only") or None)
            if disc.case_filter:
                cf = disc.case_filter
                self.log.emit(f"Only cases where {cf['rule']}: kept {cf['kept']}, left out {cf['removed']}")
            if disc.metadata_csv:
                m = disc.metadata_csv
                for r in m.get("files", []):
                    self.log.emit(f"Metadata CSV {os.path.basename(r['path'])}: {r['matched']} cases matched "
                                  f"(id column '{r['id_col']}')")
                self.log.emit(f"Metadata CSV: {m['matched']} of {len(disc.cases)} cases matched "
                              f"(id column '{m['id_col']}', columns: {', '.join(m['columns'][:12])})")
            if disc.annotations:
                a = disc.annotations
                self.log.emit(f"Annotations ({a['format']}): {a['n_cases_listed']} images listed, "
                              f"{a['n_objects_in_file']} objects, categories: {', '.join(a['categories'][:10])}")
            if p.get("limit"):
                disc.cases = disc.cases[: p["limit"]]
            n_mask = sum(1 for c in disc.cases if c.mask)
            self.log.emit(f"Found {len(disc.cases)} cases ({n_mask} with masks)")
            for w in disc.warnings[:20]:
                self.log.emit(f"warning: {w}")
            if not disc.cases:
                self.failed.emit("No cases found. Check the folder, layout and file types.")
                return
            if self._cancel:
                self.cancelled.emit()
                return
            cfg = ScanConfig(modality=p["modality"], body_method=p["body"])
            self.stage.emit("Checking intensity range…")
            out_dir = os.path.dirname(os.path.abspath(p["out"]))
            os.makedirs(out_dir, exist_ok=True)
            cache_dir = os.path.splitext(p["out"])[0] + ".cache"

            def cb(done, total, case_id, status):
                self.progress.emit(done, total, str(case_id or ""), status)

            n_warn = len(disc.warnings)
            scan = run_scan(disc, cfg, cache_dir, workers=p["workers"], progress=cb,
                            cancel=lambda: self._cancel, notify=lambda m: self.log.emit(f"NOTE: {m}"))
            for w in disc.warnings[n_warn:]:
                self.log.emit(f"WARNING: {w}")
            for o in scan.outcomes:
                if o and o["status"] != "ok":
                    self.log.emit(f"FAILED {o['case_id']}: {o.get('error')}")
            name = os.path.splitext(os.path.basename(p["out"]))[0]
            fp, side, id_map = build_fingerprint(scan, name=name, anonymize=p["anonymize"],
                                                 stage=lambda m, d, t: self.post.emit(m, d, t))
            fp["dataset"]["layout"] = p["layout"]
            self.post.emit("Saving the fingerprint (JSON + histograms)", 0, 0)
            save_fingerprint(fp, side, p["out"])
            del fp, side
            if p["anonymize"]:
                import csv
                with open(os.path.splitext(p["out"])[0] + ".idmap.csv", "w", newline="") as fh:
                    w = csv.writer(fh)
                    w.writerow(["public_id", "original_id"])
                    w.writerows(id_map.items())
            self.finished.emit(p["out"])
        except ScanCancelled:
            self.cancelled.emit()
        except Exception as e:  # report everything in the window
            self.log.emit(traceback.format_exc())
            self.failed.emit(f"{type(e).__name__}: {e}")


# ====================================================================== scan tab
class ScanTab(QWidget):
    scan_done = Signal(str)

    def __init__(self):
        super().__init__()
        self.thread = None
        self.worker = None
        lay = QVBoxLayout(self)

        # ---- dataset
        box = QGroupBox("1. Dataset")
        f = QFormLayout(box)
        self.root = QLineEdit()
        self.root.setPlaceholderText("Folder that contains the images (several folders: separate with ;)")
        self.root.setToolTip("All images under this folder (and its sub-folders) are scanned.\n"
                             "A dataset split into train / valid / test: choose the folder that contains "
                             "them, or add each folder with 'Add folder…'. The split name becomes part of "
                             "each case name, so you can group, split and compare by it, and duplicates "
                             "between the splits are flagged.")
        self.root.editingFinished.connect(lambda: self.on_folder(self.root.text()))
        row = browse_row(self.root, on_click=self.pick_root)
        b = QPushButton("Add folder…")
        b.setToolTip("Add another folder to scan together with the ones already listed")
        b.clicked.connect(lambda: self.add_dir(self.root, folder_changed=True))
        row.layout().addWidget(b)
        f.addRow("Dataset folder(s):", row)
        self.layout_cb = QComboBox()
        self.layout_cb.addItems(LAYOUTS)
        self.layout_cb.currentTextChanged.connect(self.on_layout)
        self.layout_help = QLabel()
        self.layout_help.setWordWrap(True)
        self.layout_help.setStyleSheet("color: gray")
        f.addRow("Layout:", self.layout_cb)
        f.addRow("", self.layout_help)
        self.masks = QLineEdit()
        self.masks.setToolTip("Several image folders: list one masks folder per images folder, "
                              "in the same order, separated by ;")
        self.masks_row = browse_row(self.masks, on_click=lambda: self.pick_dir(self.masks))
        b = QPushButton("Add folder…")
        b.clicked.connect(lambda: self.add_dir(self.masks))
        self.masks_row.layout().addWidget(b)
        f.addRow("Masks folder(s):", self.masks_row)
        self.saros_mask = QComboBox()
        self.saros_mask.addItems(list(SAROS_MASKS))
        self.saros_mask.setToolTip("body-regions: 13 tissue/cavity classes (fat, muscle, bone, lungs…); "
                                   "body-parts: torso, head, arms, legs. Scan twice for both.")
        self.saros_mask.currentTextChanged.connect(self.on_saros_mask)
        self.saros_label = QLabel("SAROS masks:")
        f.addRow(self.saros_label, self.saros_mask)
        self.labels_json = QLineEdit()
        self.labels_json.setPlaceholderText("usually leave empty — click ? for when you need it")
        self.labels_json.setToolTip("Only names label numbers in masks (1 = liver, 2 = spleen, …).\n"
                                    "Not needed without masks, for TotalSegmentator or nnU-Net folders.")
        lab_row = browse_row(self.labels_json, on_click=self.pick_labels)
        lab_row.layout().addWidget(H.help_button(self, "labels"))
        f.addRow("Label names (optional):", lab_row)
        self.meta_csv = QLineEdit()
        self.meta_csv.setPlaceholderText("optional: CSV with one row per image (e.g. dataset.csv; "
                                         "several: separate with ;)")
        self.meta_csv.setToolTip("Extra information per image (body part, view, scanner, split…). The column "
                                 "with the file names is found automatically. Its columns can be used to "
                                 "group outliers and to split the dataset for comparison.\n"
                                 "Several dataset folders: add one CSV per folder (e.g. CheXpert train.csv "
                                 "and valid.csv); each CSV is matched on its own.")
        row = browse_row(self.meta_csv, on_click=lambda: self.pick_file(self.meta_csv, "Metadata CSV",
                                                                          "CSV (*.csv *.tsv *.txt)"))
        b = QPushButton("Add CSV…")
        b.setToolTip("Add another metadata CSV (e.g. one per dataset folder)")
        b.clicked.connect(self.add_meta_csv)
        row.layout().addWidget(b)
        row.layout().addWidget(H.help_button(self, "metadata"))
        f.addRow("Metadata CSV (optional):", row)
        self.annotations = QLineEdit()
        self.annotations.setPlaceholderText("optional: COCO/VGG .json file, or a YOLO / Pascal VOC folder "
                                            "(several: separate with ;)")
        self.annotations.setToolTip("Object annotations (polygons or boxes) for 2D images, e.g. FracAtlas "
                                    "COCO_fracture_masks.json. Every object is measured: size, shape, "
                                    "position, contrast; plus consistency checks.")
        row = browse_row(self.annotations, "File…", on_click=lambda: self.pick_file(
            self.annotations, "Annotation file", "Annotations (*.json);;All files (*)"))
        b = QPushButton("Folder…")
        b.setToolTip("Choose a YOLO (.txt) or Pascal VOC (.xml) folder")
        b.clicked.connect(lambda: self.pick_dir(self.annotations))
        row.layout().addWidget(b)
        b = QPushButton("Add…")
        b.setToolTip("Add another annotation file or folder (e.g. one per split)")
        b.clicked.connect(self.add_annotation)
        row.layout().addWidget(b)
        row.layout().addWidget(H.help_button(self, "annotations"))
        f.addRow("Annotations (optional):", row)
        self.only = QLineEdit()
        self.only.setPlaceholderText("optional: e.g. modality=CT   or   split=train,validation; sex!=F")
        self.only.setToolTip("Scan only the cases whose metadata matches (columns from the metadata CSV, "
                             "dataset.json split, AMOS modality, …).\nkey=value (several values: a,b), "
                             "key!=value; several conditions separated by ; must all hold.")
        self._auto_only = ""
        f.addRow("Only cases where:", self.only)
        ext_w = QWidget()
        el = QHBoxLayout(ext_w)
        el.setContentsMargins(0, 0, 0, 0)
        self.ext_boxes = {}
        for g in EXTENSION_GROUPS:
            cb = QCheckBox(g)
            self.ext_boxes[g] = cb
            el.addWidget(cb)
        el.addStretch(1)
        f.addRow("File types:", ext_w)
        self.detect_label = QLabel()
        self.detect_label.setWordWrap(True)
        f.addRow("", self.detect_label)
        lay.addWidget(box)

        # ---- settings
        box = QGroupBox("2. Settings")
        g = QGridLayout(box)
        self.modality = QComboBox()
        self.modality.addItems(["CT", "2D", "3D"])
        self.body = QComboBox()
        self.body.setEditable(True)
        self.body.addItems(BODY_CHOICES)
        self.body.setToolTip("Body region without masks: auto (CT: -500 HU, otherwise Otsu), "
                             "hu, otsu, none, or type a threshold value")
        self.workers = QSpinBox()
        self.workers.setRange(1, max(1, os.cpu_count() or 1))
        self.workers.setValue(min(8, os.cpu_count() or 1))
        self.workers.setToolTip("Parallel CPU processes. More is faster, but each needs memory "
                                "(about 1.5 GB for a large CT with many masks).")
        self.modality.setToolTip("CT: values in Hounsfield units. 2D: X-rays, photos. "
                                 "3D: other volumes (e.g. MRI).")
        self.limit = QSpinBox()
        self.limit.setRange(0, 10_000_000)
        self.limit.setSpecialValueText("all")
        self.limit.setToolTip("Scan only the first N cases (for a quick test)")
        self.anonymize = QCheckBox("Anonymize (case_00001…, no paths)")
        self.anonymize.setToolTip("Replace case names with case_00001… and drop file paths; the "
                                  "mapping is saved to <name>.idmap.csv. Use for shared fingerprints.")
        g.addWidget(QLabel("Modality:"), 0, 0)
        g.addWidget(self.modality, 0, 1)
        g.addWidget(QLabel("Body region:"), 0, 2)
        g.addWidget(self.body, 0, 3)
        g.addWidget(QLabel("Parallel workers:"), 1, 0)
        g.addWidget(self.workers, 1, 1)
        g.addWidget(QLabel("Limit cases:"), 1, 2)
        g.addWidget(self.limit, 1, 3)
        g.addWidget(self.anonymize, 2, 0, 1, 4)
        lay.addWidget(box)

        # ---- output + run
        box = QGroupBox("3. Output and run")
        f = QFormLayout(box)
        self.out = QLineEdit()
        f.addRow("Save fingerprint as:", browse_row(self.out, "Save as…", self.pick_out))
        brow = QHBoxLayout()
        self.start_btn = QPushButton("▶  Start scan")
        self.start_btn.setMinimumHeight(34)
        self.start_btn.clicked.connect(self.start)
        self.cancel_btn = QPushButton("■  Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel)
        brow.addWidget(self.start_btn, 1)
        brow.addWidget(self.cancel_btn)
        brow.addWidget(H.help_button(self, "scan"))
        f.addRow(brow)
        lay.addWidget(box)

        # ---- progress
        box = QGroupBox("Progress")
        v = QVBoxLayout(box)
        self.stage = QLabel("Idle")
        self.stage.setStyleSheet("font-weight: bold")
        self.bar = QProgressBar()
        self.bar.setMinimumHeight(26)
        self.bar.setFormat("%v / %m  (%p%)")
        self.stats = QLabel("")
        self.current = QLabel("")
        self.current.setStyleSheet("color: gray")
        self.logbox = QPlainTextEdit()
        self.logbox.setReadOnly(True)
        self.logbox.setFont(MONO)
        self.logbox.setMaximumBlockCount(5000)
        v.addWidget(self.stage)
        v.addWidget(self.bar)
        v.addWidget(self.stats)
        v.addWidget(self.current)
        v.addWidget(self.logbox, 1)
        lay.addWidget(box, 1)

        self._state = None
        self.timer = QTimer(self)  # refresh labels at 5 Hz, not on every case
        self.timer.timeout.connect(self.refresh)
        self.on_layout(self.layout_cb.currentText())

    # ---------------------------------------------------------------- pickers
    def pick_root(self):
        d = QFileDialog.getExistingDirectory(self, "Dataset folder", self.root.text() or os.path.expanduser("~"))
        if d:
            self.root.setText(d)
            self.on_folder(d)

    def pick_dir(self, edit):
        d = QFileDialog.getExistingDirectory(self, "Folder", edit.text() or self.root.text())
        if d:
            edit.setText(d)

    def add_dir(self, edit, folder_changed=False):
        cur = split_paths(edit.text())
        start = os.path.dirname(cur[-1]) if cur else (self.root.text() or os.path.expanduser("~"))
        d = QFileDialog.getExistingDirectory(self, "Add folder", start)
        if d and d not in cur:
            edit.setText("; ".join(cur + [d]))
            if folder_changed:
                self.on_folder(edit.text())

    def add_meta_csv(self):
        cur = split_paths(self.meta_csv.text())
        start = os.path.dirname(cur[-1]) if cur else (split_paths(self.root.text()) or [""])[0]
        f, _ = QFileDialog.getOpenFileName(self, "Add metadata CSV", start, "CSV (*.csv *.tsv *.txt)")
        if f and f not in cur:
            self.meta_csv.setText("; ".join(cur + [f]))

    def add_annotation(self):
        cur = split_paths(self.annotations.text())
        start = os.path.dirname(cur[-1]) if cur else split_paths(self.root.text())[:1] or [""]
        start = start if isinstance(start, str) else start[0]
        f, _ = QFileDialog.getOpenFileName(self, "Add annotation file (cancel to choose a folder)", start,
                                           "Annotations (*.json);;All files (*)")
        if not f:
            f = QFileDialog.getExistingDirectory(self, "Add YOLO / VOC folder", start)
        if f and f not in cur:
            self.annotations.setText("; ".join(cur + [f]))

    def pick_file(self, edit, title, filt):
        start = edit.text() or self.root.text()
        f, _ = QFileDialog.getOpenFileName(self, title, start, filt)
        if f:
            edit.setText(f)

    def pick_labels(self):
        f, _ = QFileDialog.getOpenFileName(self, "Label names JSON", self.root.text(), "JSON (*.json)")
        if f:
            self.labels_json.setText(f)

    def pick_out(self):
        f, _ = QFileDialog.getSaveFileName(self, "Save fingerprint", self.out.text() or
                                           os.path.expanduser("~/fingerprints"), "JSON (*.json)")
        if f:
            self.out.setText(f if f.endswith(".json") else f + ".json")

    def on_saros_mask(self, mask):
        out = self.out.text()
        for other in SAROS_MASKS:
            if other != mask and out.endswith(f"_{other}.json"):
                self.out.setText(out[: -len(f"_{other}.json")] + f"_{mask}.json")

    def on_layout(self, layout):
        self.layout_help.setText(LAYOUT_HELP.get(layout, ""))
        self.masks_row.setEnabled(layout == "paired")
        self.saros_mask.setVisible(layout == "saros")
        self.saros_label.setVisible(layout == "saros")

    def on_folder(self, d):
        paths = split_paths(d)
        if not paths or not all(os.path.isdir(p) for p in paths):
            return
        multi = len(paths) > 1
        d = paths[0]
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            gs = guess_dataset(d)
            if gs.get("root") and not multi:      # dataset folder is one level down
                d = gs["root"]
                self.root.setText(d)
            sp = detect_splits(d) if not multi else {"images": [], "annotations": [], "splits": []}
        finally:
            QApplication.restoreOverrideCursor()
        if self.layout_cb.currentText() == "paired" and self.masks.text().strip():
            gs["layout"] = "paired"
        self.layout_cb.setCurrentText(gs["layout"])
        for gname, cb in self.ext_boxes.items():
            cb.setChecked(gname in gs["exts"])
        if gs["modality"]:
            self.modality.setCurrentText(gs["modality"])
        msg = f"Detected: layout <b>{gs['layout']}</b>"
        if gs.get("root") and not multi:
            msg += f" in the subfolder <b>{os.path.basename(gs['root'])}</b> (folder field updated)"
        if gs["exts"]:
            msg += f", files <b>{', '.join(gs['exts'])}</b>"
        if gs["modality"]:
            msg += f", modality <b>{gs['modality']}</b> (check: NIfTI/DICOM could also be MRI → 3D)"
        if gs["zips"] and not gs["exts"]:
            msg += (f"<br><span style='color:#c0392b'>Only {gs['zips']} .zip files found. "
                    "Unzip them first; zipped data cannot be scanned.</span>")
        elif not gs["exts"] and gs["layout"] == "files":
            msg += "<br><span style='color:#c0392b'>No supported image files found.</span>"
        if multi:
            msg += (f"<br>{len(paths)} folders, scanned as one dataset; case names start with the folder "
                    f"name below <b>{common_root(paths)}</b>")
        if sp["images"]:
            self.layout_cb.setCurrentText("paired")
            self.root.setText("; ".join(sp["images"]))
            self.masks.setText("; ".join(sp["masks"]))
            msg += (f"<br>Images and masks found{' in ' + ', '.join(sp['splits']) if sp['splits'] else ''}: "
                    "layout set to <b>paired</b>, folders filled in (all scanned as one dataset)")
        elif sp["splits"]:
            msg += (f"<br>Split folders <b>{', '.join(sp['splits'])}</b>: all are scanned as one dataset; "
                    "the split is the first folder of each case name (group / split / compare by "
                    "<i>level:0</i>)")
        if sp["annotations"] and not self.annotations.text():
            self.annotations.setText("; ".join(sp["annotations"]))
            msg += (f"<br>Annotations found in {len(sp['annotations'])} split folder(s): filled in "
                    "(clear it if unwanted)")
        from ..core.discovery import find_study_label_csvs
        study_csvs = find_study_label_csvs(self.root.text())
        if study_csvs:
            msg += ("<br>MURA study labels found (<b>" + ", ".join(os.path.basename(x) for x in study_csvs)
                    + "</b>): every image gets <i>split, body_part, patient_id, study, abnormal</i> "
                    "automatically (group / split / compare by <i>abnormal</i> or <i>body_part</i>)")
        extra = find_side_files(d)
        csvs = find_side_csvs(paths if multi else [d], sp["splits"])
        if csvs and not self.meta_csv.text():
            self.meta_csv.setText("; ".join(csvs))
            msg += (f"<br>Metadata CSV{'s' if len(csvs) > 1 else ''} found: <b>"
                    + ", ".join(os.path.basename(x) for x in csvs) + "</b> (filled in; clear it if unwanted)")
        if extra["annotations"] and not self.annotations.text():
            self.annotations.setText(extra["annotations"])
            msg += (f"<br>Annotations found: <b>{os.path.basename(extra['annotations'])}</b> "
                    "(filled in; clear it if unwanted)")
        amos = gs["layout"] == "nnunet" and not multi and _is_amos_folder(d)
        if amos and (not self.only.text().strip() or self.only.text() == self._auto_only):
            self.only.setText("modality=CT")
            self._auto_only = "modality=CT"
            msg += ("<br>AMOS: ids &ge; 500 are <b>MRI</b> (60 of the 360 labelled cases). "
                    "<i>Only cases where</i> set to <b>modality=CT</b> so MRI does not mix into the HU "
                    "statistics; use <b>modality=MRI</b> (and modality 3D) for the MRI part, or clear it for all")
        elif not amos and self._auto_only and self.only.text() == self._auto_only:
            self.only.setText("")
            self._auto_only = ""
        self.detect_label.setText(msg)
        if not self.out.text():
            d = common_root(paths) if multi else d
            name = os.path.basename(os.path.normpath(d))
            if name.lower() in ("images", "image", "imgs", "img", "data", "dataset", "files", "raw"):
                name = os.path.basename(os.path.dirname(os.path.normpath(d))) or name  # FracAtlas/images
            if gs["layout"] == "saros":
                name += "_" + self.saros_mask.currentText()
            if amos:
                name = "AMOS_CT"
            self.out.setText(os.path.join(os.path.expanduser("~/fingerprints"), name + ".json"))

    # ---------------------------------------------------------------- run
    def params(self) -> dict:
        layout = self.layout_cb.currentText()
        return {
            "layout": layout, "root": self.root.text().strip(),
            "images": self.root.text().strip() if layout == "paired" else None,
            "masks": self.masks.text().strip() or None,
            "labels_json": self.labels_json.text().strip(),
            "meta_csv": self.meta_csv.text().strip(), "annotations": self.annotations.text().strip(),
            "only": self.only.text().strip(),
            "exts": [g for g, cb in self.ext_boxes.items() if cb.isChecked()],
            "modality": self.modality.currentText(), "body": self.body.currentText().strip() or "auto",
            "workers": self.workers.value(), "limit": self.limit.value(),
            "anonymize": self.anonymize.isChecked(), "out": self.out.text().strip(),
            "saros_mask": self.saros_mask.currentText(),
        }

    def start(self):
        p = self.params()
        bad = [x for x in split_paths(p["root"]) if not os.path.isdir(x)]
        if not split_paths(p["root"]) or bad:
            QMessageBox.warning(self, "Scan", "Choose an existing dataset folder." +
                                (f"\n\nNot found: {bad[0]}" if bad else ""))
            return
        if len(split_paths(p["root"])) > 1 and p["layout"] not in ("files", "paired"):
            QMessageBox.warning(self, "Scan", f"The {p['layout']} layout takes one folder. Several folders "
                                "work with the files and paired layouts.")
            return
        if p["layout"] == "paired" and p["masks"] and \
                len(split_paths(p["masks"])) != len(split_paths(p["root"])):
            QMessageBox.warning(self, "Scan", "Give one masks folder per images folder, in the same order.")
            return
        if p["layout"] in ("files", "paired") and not p["exts"]:
            QMessageBox.warning(self, "Scan", "Tick at least one file type.")
            return
        if not p["out"]:
            QMessageBox.warning(self, "Scan", "Choose where to save the fingerprint.")
            return
        try:
            from ..core.config import ScanConfig
            ScanConfig(modality=p["modality"], body_method=p["body"])
        except ValueError as e:
            QMessageBox.warning(self, "Scan", f"Invalid body region setting: {e}")
            return
        self.logbox.clear()
        self.log(f"Scanning {p['root']}  →  {p['out']}")
        self.log("(Cases already in the cache are reused, so a cancelled scan resumes.)")
        self._state = {"t0": time.time(), "done": 0, "total": 0, "case": "", "ok": 0, "err": 0,
                       "cached": 0, "first_done": None, "first_t": None}
        self.bar.setRange(0, 0)  # busy until the total is known
        self.set_running(True)
        self.thread = QThread(self)
        self.worker = ScanWorker(p)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.stage.connect(self.stage.setText)
        self.worker.progress.connect(self.on_progress)
        self.worker.post.connect(self.on_post)
        self.worker.log.connect(self.log)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.cancelled.connect(self.on_cancelled)
        for s in (self.worker.finished, self.worker.failed, self.worker.cancelled):
            s.connect(self.thread.quit)
        self.thread.start()
        self.timer.start(200)

    def cancel(self):
        if self.worker:
            self.worker.cancel()
            self.stage.setText("Cancelling… (waiting for running cases)")
            self.cancel_btn.setEnabled(False)

    def is_running(self) -> bool:
        return self.thread is not None and self.thread.isRunning()

    def set_running(self, running: bool):
        self.start_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)

    @Slot(int, int, str, str)
    def on_progress(self, done, total, case_id, status):
        st = self._state
        if st is None:
            return
        if case_id:
            st["case"] = case_id
            if status == "ok":
                st["ok"] += 1
            else:
                st["err"] += 1
        elif status == "cached":
            st["cached"] = done
            st["first_done"], st["first_t"] = done, time.time()
            self.stage.setText("Scanning…")
        st["done"], st["total"] = done, total

    def on_post(self, msg, done, total):
        """Post-processing after the last image: show the step, a bar when its length is
        known, and log each step with the time it took."""
        st = self._state
        if st is None:
            return
        now = time.time()
        if st.get("post_msg") != msg:
            if st.get("post_msg"):
                self.log(f"  done in {fmt_time(now - st['post_t'])}: {st['post_msg']}")
            else:
                self.log(f"All {st['total']} images analysed. Post-processing:")
                st["post_start"] = now
            st["post_msg"], st["post_t"] = msg, now
            st["post_n"] = st.get("post_n", 0) + 1
            self.current.setText("")
        self.stage.setText(f"Post-processing (step {st['post_n']}): {msg}…")
        if total:
            if self.bar.maximum() != total:
                self.bar.setRange(0, total)
            self.bar.setValue(done)
        elif self.bar.maximum() != 0:
            self.bar.setRange(0, 0)          # busy indicator: length unknown
        self.stats.setText(f"post-processing for {fmt_time(now - st['post_start'])}   |   "
                           f"total elapsed {fmt_time(now - st['t0'])}")

    def refresh(self):
        st = self._state
        if not st or not st["total"] or st.get("post_msg"):
            return
        if self.bar.maximum() != st["total"]:
            self.bar.setRange(0, st["total"])
        self.bar.setValue(st["done"])
        el = time.time() - st["t0"]
        new = st["done"] - (st["first_done"] or 0)
        run_t = time.time() - (st["first_t"] or st["t0"])
        rate = new / run_t if run_t > 0 and new > 0 else 0.0
        eta = (st["total"] - st["done"]) / rate if rate > 0 else float("nan")
        self.stats.setText(
            f"ok {st['ok']}   failed {st['err']}   from cache {st['cached']}   |   "
            f"{rate:.1f} cases/s   elapsed {fmt_time(el)}   remaining ≈ {fmt_time(eta)}")
        self.current.setText(f"last: {st['case'][-110:]}")

    def log(self, msg: str):
        self.logbox.appendPlainText(msg)

    def _end(self):
        self.timer.stop()
        self.refresh()
        self.set_running(False)

    def on_finished(self, path):
        st = self._state or {}
        if st.get("post_msg"):
            now = time.time()
            self.log(f"  done in {fmt_time(now - st['post_t'])}: {st['post_msg']}")
            self.log(f"Post-processing took {fmt_time(now - st['post_start'])}.")
            st["post_msg"] = None
        self._end()
        self.bar.setRange(0, 1)
        self.bar.setValue(1)
        self.log(f"Wrote {path}")
        self.stage.setText("Opening the fingerprint in the other tabs…")
        QApplication.processEvents()
        self.scan_done.emit(path)
        self.stage.setText(f"Done: {path}")

    def on_failed(self, msg):
        self._end()
        self.bar.setRange(0, 1)
        self.stage.setText("Failed")
        self.log("ERROR: " + msg)
        QMessageBox.critical(self, "Scan failed", msg)

    def on_cancelled(self):
        self._end()
        self.stage.setText("Cancelled. Press Start to resume (finished cases are cached).")


# ====================================================================== result tabs
class SummaryTab(QWidget):
    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel("Dataset summary (same as `mfp info`)"))
        row.addStretch(1)
        self.pdf_btn = QPushButton("Export PDF report…")
        self.pdf_btn.setToolTip("PDF with overview, geometry, histograms, bins, nnU-Net values, labels, "
                                "quality, duplicates and outliers with thumbnails. The outliers use "
                                "the settings of the Outliers tab.")
        row.addWidget(self.pdf_btn)
        self.json_btn = QPushButton("Export JSON…")
        self.json_btn.setToolTip("Save a copy of this fingerprint (JSON + .arrays.npz with the histograms), "
                                 "e.g. to share it or compare it later. You can remove the local folder "
                                 "and cache paths from the copy.")
        row.addWidget(self.json_btn)
        row.addWidget(H.help_button(self, "summary"))
        v.addLayout(row)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setFont(MONO)
        self.text.setLineWrapMode(QPlainTextEdit.NoWrap)
        v.addWidget(self.text)

    def set_fp(self, fp, side, path):
        try:
            self.text.setPlainText(f"{path}\n\n" + info_text(fp))
        except Exception:
            self.text.setPlainText(traceback.format_exc())


class HistogramTab(QWidget):
    def __init__(self):
        super().__init__()
        self.fp = None
        self.spec = None
        self.lines = []
        v = QVBoxLayout(self)
        top = QHBoxLayout()
        self.region = QComboBox()
        self.region.currentTextChanged.connect(self.redraw)
        self.logy = QCheckBox("log scale")
        self.logy.setChecked(True)
        self.logy.toggled.connect(self.redraw)
        top.addWidget(QLabel("Region:"))
        top.addWidget(self.region)
        top.addWidget(self.logy)
        top.addSpacing(20)
        top.addWidget(QLabel("Bins:"))
        self.preset = QComboBox()
        top.addWidget(self.preset)
        self.nq = QSpinBox()
        self.nq.setRange(2, 50)
        self.nq.setValue(5)
        self.nq.setPrefix("n=")
        top.addWidget(self.nq)
        b = QPushButton("Apply")
        b.clicked.connect(self.apply_preset)
        top.addWidget(b)
        top.addStretch(1)
        top.addWidget(H.help_button(self, "histogram"))
        v.addLayout(top)
        self.region.setToolTip("all = every voxel; body = body found from intensities; "
                               "foreground = inside the annotation masks")
        self.logy.setToolTip("Log scale shows rare intensities (metal, padding) that are invisible "
                             "on a linear axis")
        self.preset.setToolTip("quantiles = n bins with equal counts; equal width = n equal bins; "
                               "ct_* = HU presets (tissue types, body composition)")
        row = QHBoxLayout()
        self.edges = QLineEdit()
        self.edges.setPlaceholderText("custom edges, e.g. 0 120 180 216   (Enter to apply; "
                                      "drag the lines in the plot to adjust)")
        self.edges.returnPressed.connect(self.apply_edges)
        self.open_ends = QCheckBox("open ends (<first, ≥last)")
        self.open_ends.setChecked(True)
        self.open_ends.toggled.connect(self.update_table)
        row.addWidget(self.edges, 1)
        row.addWidget(self.open_ends)
        v.addLayout(row)
        split = QSplitter(Qt.Vertical)
        self.plot = pg.PlotWidget(background="w")
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        split.addWidget(self.plot)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["bin", "range", "fraction %", "count"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        split.addWidget(self.table)
        split.setSizes([500, 220])
        v.addWidget(split, 1)

    def set_fp(self, fp, side, path):
        from ..core.bins import load_presets
        from ..core.config import HistogramSpec
        self.fp = fp
        self.spec = HistogramSpec.from_dict(fp["histogram_spec"])
        self.region.blockSignals(True)
        self.region.clear()
        self.region.addItems(list(fp["intensity"].keys()))
        if "body" in fp["intensity"]:
            self.region.setCurrentText("body")
        self.region.blockSignals(False)
        self.preset.clear()
        self.preset.addItems(["quantiles (equal counts)", "equal width"])
        if fp["config"].get("modality") == "CT" and self.spec.lo <= -1000:
            self.preset.addItems(list(load_presets().keys()))
        self.redraw()
        self.set_edges([])

    def counts(self):
        return np.asarray(self.fp["intensity"][self.region.currentText()]["histogram"]["counts"],
                          dtype=np.float64)

    def redraw(self):
        if not self.fp or not self.region.currentText():
            return
        c = self.counts()
        self.plot.clear()
        x = self.spec.edges()
        y = c[1:-1]
        if self.logy.isChecked():
            y = np.log10(y + 1)
            self.plot.setLabel("left", "log10(count + 1)")
        else:
            self.plot.setLabel("left", "count")
        self.plot.setLabel("bottom", "intensity (HU)" if self.fp["config"].get("modality") == "CT"
                           else "intensity")
        self.plot.plot(x, y, stepMode="center", fillLevel=0, brush=(70, 130, 180, 120),
                       pen=pg.mkPen((40, 90, 140), width=1))
        nz = np.flatnonzero(c[1:-1])
        if nz.size:
            self.plot.setXRange(x[nz[0]], x[nz[-1] + 1], padding=0.02)
        for ln in self.lines:
            self.plot.addItem(ln)
        self.update_table()

    def apply_preset(self):
        from ..core.bins import equal_width, load_presets, quantile_bins
        name = self.preset.currentText()
        c = self.counts()
        if name.startswith("quantiles"):
            bs = quantile_bins(c, self.spec, self.nq.value())
            edges = [b.lo for b in bs.bins[1:]]
            self.open_ends.setChecked(True)
        elif name == "equal width":
            nz = np.flatnonzero(c[1:-1])
            lo, hi = self.spec.edges()[nz[0]], self.spec.edges()[nz[-1] + 1]
            bs = equal_width(lo, hi, self.nq.value())
            edges = [b.lo for b in bs.bins] + [bs.bins[-1].hi]
            self.open_ends.setChecked(False)
        else:
            bs = load_presets()[name]
            vals = sorted({v for b in bs.bins for v in (b.lo, b.hi) if v is not None and np.isfinite(v)})
            edges = vals
            self.open_ends.setChecked(True)
        self.set_edges(edges)

    def apply_edges(self):
        try:
            edges = sorted(float(t) for t in self.edges.text().replace(",", " ").split())
        except ValueError:
            QMessageBox.warning(self, "Bins", "Edges must be numbers separated by spaces.")
            return
        self.set_edges(edges)

    def set_edges(self, edges):
        for ln in self.lines:
            self.plot.removeItem(ln)
        self.lines = []
        for e in edges:
            ln = pg.InfiniteLine(pos=e, angle=90, movable=True, pen=pg.mkPen((200, 40, 40), width=2),
                                 hoverPen=pg.mkPen((255, 120, 0), width=3),
                                 label="{value:.0f}", labelOpts={"position": 0.95, "color": (200, 40, 40)})
            ln.sigPositionChangeFinished.connect(self.lines_moved)
            ln.sigPositionChanged.connect(self.update_table)
            self.plot.addItem(ln)
            self.lines.append(ln)
        self._sync_text()
        self.update_table()

    def _current_edges(self):
        w = self.spec.width if self.spec else 1.0
        return sorted(round(ln.value() / w) * w for ln in self.lines)

    def lines_moved(self):
        for ln in self.lines:  # snap to histogram bin edges, so counts are exact
            w = self.spec.width
            ln.blockSignals(True)
            ln.setValue(self.spec.lo + round((ln.value() - self.spec.lo) / w) * w)
            ln.blockSignals(False)
        self._sync_text()
        self.update_table()

    def _sync_text(self):
        self.edges.setText(" ".join(f"{e:g}" for e in self._current_edges()))

    def update_table(self):
        from ..core.bins import BinSet, binset_table
        edges = self._current_edges() if self.lines else []
        self.table.setRowCount(0)
        if not self.fp or len(edges) < (1 if self.open_ends.isChecked() else 2):
            return
        rows = binset_table(self.counts(), self.spec,
                            BinSet.from_edges(edges, open_ends=self.open_ends.isChecked()))
        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            lo = "−∞" if r["lo"] is None or not np.isfinite(r["lo"]) else f"{r['lo']:g}"
            hi = "∞" if r["hi"] is None or not np.isfinite(r["hi"]) else f"{r['hi']:g}"
            for j, val in enumerate((r["name"], f"[{lo}, {hi})", f"{100 * r['fraction']:.3f}",
                                     f"{r['count']:.0f}")):
                it = QTableWidgetItem(val)
                if j >= 2:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(i, j, it)


class OutliersTab(QWidget):
    def __init__(self):
        super().__init__()
        self.fp = self.side = None
        self.res = None
        self.dims = 3
        v = QVBoxLayout(self)
        top = QHBoxLayout()
        self.method = QComboBox()
        self.method.addItems(["robust_z", "zscore", "iqr"])
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0, 1000)
        self.threshold.setSpecialValueText("default")
        self.threshold.setSingleStep(0.5)
        self.group = QComboBox()
        self.group.setEditable(True)
        self.group.setMinimumWidth(230)
        self.group.lineEdit().setPlaceholderText("group by (optional)")
        self.maxfrac = QDoubleSpinBox()
        self.maxfrac.setRange(0, 100)
        self.maxfrac.setValue(1.0)
        self.maxfrac.setSuffix(" %")
        self.maxfrac.setToolTip("Flag at most this share of cases per feature and direction "
                                "(min 5 cases); 0 = threshold only")
        b = QPushButton("Recompute")
        b.clicked.connect(self.compute)
        self.method.setToolTip("<b>robust_z</b>: median/MAD, default and most reliable<br>"
                               "<b>zscore</b>: mean/std, distorted by the outliers themselves<br>"
                               "<b>iqr</b>: boxplot rule (distance beyond the quartiles)")
        self.threshold.setToolTip("Score beyond which an image is flagged on a feature. "
                                  "Default: 3.5 (robust_z), 3 (zscore), 1.5 (iqr). "
                                  "Higher = only extreme cases.")
        self.group.setToolTip("Compare each image only with similar images.<br>"
                              "<b>level:1</b> = 2nd folder of the path (MURA train/XR_WRIST/… → body part)<br>"
                              "<b>meta:institute</b>, <b>meta:ConvolutionKernel</b>, "
                              "<b>meta:Manufacturer</b> = a metadata column<br>Empty = whole dataset.")
        self.group.lineEdit().returnPressed.connect(self.compute)
        self.group.activated.connect(lambda _: self.compute())
        for w in (QLabel("Method:"), self.method, QLabel("Threshold:"), self.threshold,
                  QLabel("Max share:"), self.maxfrac, self.group, b, H.help_button(self, "outliers")):
            top.addWidget(w)
        v.addLayout(top)
        self.summary = QLabel("")
        v.addWidget(self.summary)
        split = QSplitter(Qt.Horizontal)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["rank", "case", "group", "# features", "max |score|",
                                              "features"])
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setWordWrap(False)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 45)
        self.table.setColumnWidth(1, 380)
        self.table.setColumnWidth(2, 90)
        self.table.setColumnWidth(3, 80)
        self.table.itemSelectionChanged.connect(self.show_case)
        for col, tip in enumerate((
                "Rank: images unusual on more features come first, then by score.",
                "Case (file or series). Click a row to see the preview and why it was flagged.",
                "Group the image was compared with (when Group by is used).",
                "On how many features the image is unusual. Most important column: 4+ is almost "
                "always a broken image, 1 is often just normal variation.",
                "How extreme the worst feature is (0 = typical, 3.5 = cutoff, 10+ = far off).",
                "Which features were unusual. Hover over a cell for explanations.")):
            self.table.horizontalHeaderItem(col).setToolTip(tip)
        split.addWidget(self.table)
        right = QWidget()
        rv = QVBoxLayout(right)
        self.views = []
        vrow = QHBoxLayout()
        for _ in range(3):
            gv = pg.GraphicsLayoutWidget()
            vb = gv.addViewBox(lockAspect=True)
            vb.invertY(True)
            img = pg.ImageItem()
            vb.addItem(img)
            self.views.append((gv, vb, img))
            vrow.addWidget(gv)
        rv.addLayout(vrow, 3)
        self.detail = QPlainTextEdit()
        self.detail.setReadOnly(True)
        self.detail.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.detail.setFont(MONO)
        rv.addWidget(self.detail, 2)
        split.addWidget(right)
        split.setSizes([650, 650])
        v.addWidget(split, 1)

    def set_fp(self, fp, side, path):
        from ..core.split import describe_choice, group_choices
        self.fp, self.side = fp, side
        self.dims = 3 if any(c.get("geometry", {}).get("dims") == 3 for c in fp.get("cases", [])) else 2
        self.group.blockSignals(True)
        self.group.clear()
        self.group.addItem("")
        for g in group_choices(fp):
            self.group.addItem(f"{g}  —  {describe_choice(fp, g, 3)}")
        self.group.setCurrentIndex(0)
        self.group.blockSignals(False)
        self.compute()

    def compute(self):
        from ..core.outliers import detect_outliers
        if not self.fp:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.res = detect_outliers(
                self.fp, self.side, method=self.method.currentText(),
                threshold=self.threshold.value() or None, group_by=self.group.currentText().split("  ")[0].strip() or None,
                max_frac=self.maxfrac.value() / 100 or None)
        except Exception as e:
            QMessageBox.warning(self, "Outliers", str(e))
            return
        finally:
            QApplication.restoreOverrideCursor()
        r = self.res
        multi = sum(1 for e in r["cases"] if e["n_features"] >= 2)
        self.summary.setText(f"Flagged {len(r['cases'])} of {r['n_cases']} cases "
                             f"({100 * len(r['cases']) / max(r['n_cases'], 1):.1f} %), "
                             f"{multi} on ≥2 features. Select a row to see the image.")
        rows = r["cases"][:5000]
        self.table.setSortingEnabled(False)  # must be off while filling, or rows get scrambled
        self.table.clearContents()
        self.table.setRowCount(len(rows))
        for i, e in enumerate(rows):
            vals = (i + 1, e["case_id"], e.get("group", ""), e["n_features"], round(e["max_score"], 1),
                    ", ".join(f.split(".")[-1] if not f.startswith("histogram") else "histogram"
                              for f in e["features"]))
            for j, val in enumerate(vals):
                it = QTableWidgetItem()
                it.setData(Qt.DisplayRole, val)  # numbers sort numerically
                if j == 5:
                    it.setToolTip("<br><br>".join(H.feature_tooltip(f, self.dims) for f in e["features"]))
                self.table.setItem(i, j, it)
        self.table.setSortingEnabled(True)
        self.table.sortItems(0, Qt.AscendingOrder)
        if len(rows) > 5000:
            self.summary.setText(self.summary.text() + "  (table shows the top 5000)")

    def show_case(self):
        items = self.table.selectedItems()
        if not items or not self.res:
            return
        cid = self.table.item(items[0].row(), 1).text()
        flags = [f for f in self.res["flags"] if f["case_id"] == cid]
        entry = next((e for e in self.res["cases"] if e["case_id"] == cid), None)
        lines = [cid, ""]
        if entry:
            lines += [f"VERDICT: {H.verdict(entry['n_features'], entry['max_score'])}", "",
                      "WHY FLAGGED:"]
        case = next((c for c in self.fp["cases"] if c["case_id"] == cid), None)
        if case:
            obvious = H.obvious_problems(case)
            if obvious:
                lines[2:2] = obvious + [""]
        for f in flags:
            lines.append(H.explain_flag(f, self.dims))
        qp = ((case or {}).get("quality") or {}).get("problems") or []
        if qp:
            titles = {k: v["title"] for k, v in ((self.fp.get("quality") or {}).get("problems") or {}).items()}
            lines += ["", "QUALITY CHECKS:"] + [f"! {titles.get(k, k)}" for k in qp]
        an = (case or {}).get("annotations")
        if an is not None:
            lines += ["", f"Annotated objects: {an['n_objects']}"
                      + (f" ({', '.join(an['categories'])})" if an.get("categories") else "")]
        lines.append("")
        lines.append("Details:")
        if case:
            g = case.get("geometry", {})
            lines += ["", f"shape {g.get('shape')}  spacing {[round(s, 3) for s in g.get('spacing', [])]}"]
            for k, val in case.get("flags", {}).items():
                if k != "notes":
                    lines.append(f"  {k:<22} {val:.4g}" if isinstance(val, float) else f"  {k:<22} {val}")
            if case.get("meta"):
                lines.append("meta: " + ", ".join(f"{k}={v}" for k, v in case["meta"].items() if v))
            if case.get("flags", {}).get("notes"):
                lines.append(f"notes: {case['flags']['notes']}")
        self.detail.setPlainText("\n".join(lines))
        self.show_previews(cid, case)

    def show_previews(self, cid, case):
        for _, _, img in self.views:
            img.clear()
        from .quality_tab import case_previews
        prev = case_previews(self.fp, cid)
        if not prev:
            self.detail.appendPlainText("\n(no preview: original file and scan cache not available, "
                                        "e.g. anonymized, moved or deleted)")
            return
        sp = (case or {}).get("geometry", {}).get("spacing", [])
        is_ct = self.fp["config"].get("modality") == "CT"
        for k, (gv, vb, img) in enumerate(self.views):
            gv.setVisible(k < len(prev))
            if k >= len(prev):
                continue
            a = prev[k]
            lo, hi = (-1000, 1000) if is_ct and np.nanmin(a) < -500 else np.nanpercentile(a, [0.5, 99.5])
            img.setImage(a.T, levels=(lo, hi))  # pyqtgraph uses (x, y) order
            aspect = 1.0
            if len(sp) == 3 and k > 0:  # coronal/sagittal: rows are z
                aspect = sp[0] / sp[1] if sp[1] else 1.0
            img.setRect(QRectF(0, 0, a.shape[1], a.shape[0] * aspect))
            vb.autoRange()


# ====================================================================== main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MedFingerPrint: dataset fingerprinting")
        self.resize(1300, 900)
        self.tabs = QTabWidget()
        self.scan = ScanTab()
        self.summary = SummaryTab()
        self.hist = HistogramTab()
        self.outl = OutliersTab()
        from .compare_tab import CompareTab
        from .quality_tab import QualityTab
        self.qual = QualityTab()
        self.current_path = None
        self.compare = CompareTab(lambda: self.current_path)
        self.tabs.addTab(self.scan, "Scan")
        self.tabs.addTab(self.summary, "Summary")
        self.tabs.addTab(self.hist, "Histogram && bins")
        self.tabs.addTab(self.outl, "Outliers")
        self.tabs.addTab(self.qual, "Quality && duplicates")
        self.tabs.addTab(self.compare, "Compare")
        for i in (1, 2, 3, 4):
            self.tabs.setTabEnabled(i, False)
        self.setCentralWidget(self.tabs)
        self.scan.scan_done.connect(self.open_fp)
        self.summary.pdf_btn.clicked.connect(self.export_pdf)
        self.summary.json_btn.clicked.connect(self.export_json)
        self.fp, self.side = None, None
        m = self.menuBar().addMenu("&File")
        a = QAction("&Open fingerprint…", self)
        a.setShortcut(QKeySequence.Open)
        a.triggered.connect(self.open_dialog)
        m.addAction(a)
        a = QAction("Export &PDF report…", self)
        a.setShortcut(QKeySequence.Print)
        a.triggered.connect(self.export_pdf)
        m.addAction(a)
        a = QAction("Export &JSON…", self)
        a.setShortcut(QKeySequence.SaveAs)
        a.triggered.connect(self.export_json)
        m.addAction(a)
        a = QAction("&Compare fingerprints…", self)
        a.triggered.connect(lambda: (self.tabs.setCurrentWidget(self.compare), self.compare.add_dialog()))
        m.addAction(a)
        a = QAction("&Quit", self)
        a.setShortcut(QKeySequence.Quit)
        a.triggered.connect(self.close)
        m.addAction(a)
        hm = self.menuBar().addMenu("&Help")
        for topic, (title, _) in H.PAGES.items():
            a = QAction(title, self)
            a.triggered.connect(lambda _=False, t=topic: setattr(self, "_help_dlg", H.show_help(self, t)))
            hm.addAction(a)
        self.statusBar().showMessage("Choose a dataset folder on the Scan tab, or File → Open fingerprint. "
                                     "Hover over controls for help, or use the ? buttons.")

    def open_dialog(self):
        f, _ = QFileDialog.getOpenFileName(self, "Open fingerprint", os.path.expanduser("~/fingerprints"),
                                           "Fingerprint JSON (*.json)")
        if f:
            self.open_fp(f)

    def open_fp(self, path):
        from ..core.schema import load_fingerprint
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.statusBar().showMessage(f"Loading {path}…")
        QApplication.processEvents()
        try:
            fp, side = load_fingerprint(path)
            if not str(fp.get("schema", "")).startswith("medfingerprint/"):
                raise ValueError("not a MedFingerPrint fingerprint JSON")
            for tab in (self.summary, self.hist, self.outl, self.qual):
                tab.set_fp(fp, side, path)
            self.fp, self.side = fp, side
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Open fingerprint", f"{path}\n\n{type(e).__name__}: {e}")
            return
        QApplication.restoreOverrideCursor()
        for i in (1, 2, 3, 4):
            self.tabs.setTabEnabled(i, True)
        self.tabs.setCurrentIndex(1)
        self.current_path = path
        d = fp["dataset"]
        self.statusBar().showMessage(f"{path}   |   {d['name']}: {d['n_ok']} cases, modality {d['modality']}")
        self.setWindowTitle(f"MedFingerPrint: {d['name']}")

    def export_pdf(self):
        if not self.fp:
            QMessageBox.information(self, "PDF report", "Open or scan a fingerprint first.")
            return
        f, _ = QFileDialog.getSaveFileName(self, "Save PDF report",
                                           os.path.splitext(self.current_path)[0] + ".pdf", "PDF (*.pdf)")
        if not f:
            return
        f = f if f.lower().endswith(".pdf") else f + ".pdf"
        from ..report import fingerprint_report
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.statusBar().showMessage(f"Writing {f}…")
        QApplication.processEvents()
        try:
            fingerprint_report(self.fp, f, self.side, outliers=getattr(self.outl, "res", None))
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "PDF report", f"{type(e).__name__}: {e}")
            return
        QApplication.restoreOverrideCursor()
        self.statusBar().showMessage(f"Saved {f}")
        if QMessageBox.question(self, "PDF report", f"Saved {f}\n\nOpen it now?") == QMessageBox.Yes:
            from PySide6.QtCore import QUrl
            from PySide6.QtGui import QDesktopServices
            QDesktopServices.openUrl(QUrl.fromLocalFile(f))

    def export_json(self):
        if not self.fp:
            QMessageBox.information(self, "Export JSON", "Open or scan a fingerprint first.")
            return
        f, _ = QFileDialog.getSaveFileName(self, "Export fingerprint JSON",
                                           os.path.splitext(self.current_path)[0] + "_export.json",
                                           "Fingerprint JSON (*.json)")
        if not f:
            return
        f = f if f.lower().endswith(".json") else f + ".json"
        if os.path.abspath(f) == os.path.abspath(self.current_path):
            QMessageBox.warning(self, "Export JSON", "Choose a different file name than the open fingerprint.")
            return
        box = QMessageBox(self)
        box.setWindowTitle("Export JSON")
        box.setText("Remove local paths from the copy?")
        box.setInformativeText(
            "For sharing (paper supplement, colleagues): removes the dataset folder and scan-cache paths "
            "of this computer. Statistics, histograms and case names are kept.\n\n"
            "Keep them if you will use the copy on this computer: image previews in Outliers / Quality "
            "and 'Split open dataset' need them.")
        share = box.addButton("Remove (for sharing)", QMessageBox.AcceptRole)
        box.addButton("Keep (this computer)", QMessageBox.NoRole)
        box.addButton(QMessageBox.Cancel)
        box.exec()
        if box.clickedButton() is box.button(QMessageBox.Cancel):
            return
        from copy import deepcopy
        from ..core.schema import save_fingerprint
        fp = deepcopy(self.fp)
        if box.clickedButton() is share:
            fp["local"] = None
        fp.pop("sidecar", None)
        try:
            save_fingerprint(fp, self.side, f)
        except Exception as e:
            QMessageBox.critical(self, "Export JSON", f"{type(e).__name__}: {e}")
            return
        extra = f"\n+ {os.path.basename(f)[:-5]}.arrays.npz (keep it next to the JSON)" if self.side else ""
        self.statusBar().showMessage(f"Saved {f}")
        QMessageBox.information(self, "Export JSON", f"Saved {f}{extra}")

    def closeEvent(self, ev):
        if self.scan.is_running():
            if QMessageBox.question(self, "Quit", "A scan is running. Cancel it and quit?\n"
                                    "(Finished cases stay cached; the scan can be resumed.)") \
                    != QMessageBox.Yes:
                ev.ignore()
                return
            self.scan.cancel()
            self.scan.thread.wait()
        ev.accept()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    pg.setConfigOptions(antialias=True, imageAxisOrder="col-major")
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("MedFingerPrint")
    w = MainWindow()
    w.show()
    if argv and argv[0].endswith(".json") and os.path.isfile(argv[0]):
        QTimer.singleShot(100, lambda: w.open_fp(argv[0]))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
