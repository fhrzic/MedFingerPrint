"""Quality & duplicates tab: rule-based image problems, duplicates and annotation checks,
each with image previews (one image, or both images of a duplicate pair)."""
from __future__ import annotations

import os

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QPlainTextEdit, QSplitter, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from . import help as H


DISPLAY_MAX = 1024   # longest side of a 2D image shown from the original file


def _original_2d(fp: dict, cid: str):
    """Full-resolution grey image of a 2D case read from its original file (fast for
    PNG/JPG/2D DICOM), area-averaged to <= DISPLAY_MAX. None if not available."""
    from ..core.case_analysis import analysis_channel
    from ..core.io import image_kind_for, load_image
    cfg = fp.get("config") or {}
    if cfg.get("modality") != "2D":
        return None
    local = fp.get("local") or {}
    case = next((c for c in fp.get("cases") or [] if c.get("case_id") == cid), None)
    rel = (case or {}).get("image")
    if not rel or not local.get("root"):
        return None
    path = os.path.join(local["root"], rel)
    if not os.path.isfile(path):
        return None
    try:
        kind = image_kind_for(path, "2D")
        if kind not in ("raster", "dicom2d"):
            return None
        g = analysis_channel(load_image(kind, path), cfg.get("gray_mode", "luminance"))
    except Exception:
        return None
    g = np.squeeze(g)
    if g.ndim != 2:
        return None
    step = int(np.ceil(max(g.shape) / DISPLAY_MAX))
    if step > 1:
        h, w = (g.shape[0] // step) * step, (g.shape[1] // step) * step
        g = g[:h, :w].reshape(h // step, step, w // step, step).mean(axis=(1, 3))
    return g.astype(np.float32)


def case_previews(fp: dict, cid: str) -> list:
    """Images to show for a case: the original file for 2D images (sharp), otherwise the
    small previews stored in the scan cache (CT/3D: axial, coronal, sagittal)."""
    full = _original_2d(fp, cid)
    if full is not None:
        return [full]
    cf = ((fp.get("local") or {}).get("cache_files") or {}).get(cid)
    if not cf or not os.path.isfile(cf):
        return []
    with np.load(cf) as z:
        return [z[k] for k in ("preview_axial", "preview_coronal", "preview_sagittal") if k in z.files]


class _View:
    def __init__(self):
        self.gv = pg.GraphicsLayoutWidget()
        self.vb = self.gv.addViewBox(lockAspect=True)
        self.vb.invertY(True)
        self.img = pg.ImageItem()
        self.vb.addItem(self.img)
        self.title = QLabel("")
        self.title.setWordWrap(True)
        self.w = QWidget()
        v = QVBoxLayout(self.w)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(self.title)
        v.addWidget(self.gv, 1)

    def show(self, fp, cid):
        self.img.clear()
        self.title.setText(cid or "")
        if not cid:
            return False
        prev = case_previews(fp, cid)
        if not prev:
            self.title.setText(f"{cid}\n(no preview: scan cache not available)")
            return False
        a = prev[0]
        ct = fp.get("config", {}).get("modality") == "CT"
        lo, hi = (-1000, 1000) if ct and np.nanmin(a) < -500 else np.nanpercentile(a, [0.5, 99.5])
        if hi <= lo:
            hi = lo + 1
        self.img.setImage(a.T, levels=(lo, hi))
        self.img.setRect(QRectF(0, 0, a.shape[1], a.shape[0]))
        self.vb.autoRange()
        return True


class QualityTab(QWidget):
    COLS = ["type", "problem", "case", "other case / detail"]

    def __init__(self):
        super().__init__()
        self.fp = None
        self.rows = []
        v = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel("Show:"))
        self.filter = QComboBox()
        self.filter.setMinimumWidth(420)
        self.filter.currentIndexChanged.connect(self.fill)
        top.addWidget(self.filter)
        top.addStretch(1)
        self.summary = QLabel("")
        top.addWidget(self.summary)
        top.addWidget(H.help_button(self, "quality"))
        v.addLayout(top)
        split = QSplitter(Qt.Horizontal)
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 110)
        self.table.setColumnWidth(1, 260)
        self.table.setColumnWidth(2, 300)
        self.table.itemSelectionChanged.connect(self.show_row)
        split.addWidget(self.table)
        right = QWidget()
        rv = QVBoxLayout(right)
        prow = QHBoxLayout()
        self.v1, self.v2 = _View(), _View()
        prow.addWidget(self.v1.w)
        prow.addWidget(self.v2.w)
        rv.addLayout(prow, 3)
        self.detail = QPlainTextEdit()
        self.detail.setReadOnly(True)
        rv.addWidget(self.detail, 1)
        split.addWidget(right)
        split.setSizes([700, 600])
        v.addWidget(split, 1)

    # ---------------------------------------------------------------- data
    def set_fp(self, fp, side, path):
        self.fp = fp
        rows = []
        q = fp.get("quality") or {}
        titles = {k: v["title"] for k, v in (q.get("problems") or {}).items()}
        for c in fp.get("cases") or []:
            for k in (c.get("quality") or {}).get("problems") or []:
                rows.append(("Quality", titles.get(k, k), c["case_id"], "", None))
        du = fp.get("duplicates") or {}
        tops = {}
        for g in du.get("exact_groups") or []:
            for other in g[1:]:
                cross = g[0].split("/")[0] != other.split("/")[0] and "/" in g[0]
                t = "Exact duplicate (identical pixels)" + (" ⚠ across folders" if cross else "")
                rows.append(("Duplicate", t, g[0], other, other))
        del tops
        for p in du.get("near_pairs") or []:
            t = "Near-duplicate" + (" ⚠ across folders" if p.get("cross") else "")
            d = p.get("distance_fine", p["distance"])
            rows.append(("Duplicate", t, p["a"], f"{p['b']}   (differs in {d} bits)", p["b"]))
        an = fp.get("annotations") or {}
        case_ids = {c["case_id"] for c in fp.get("cases") or []}
        for k, chk in (an.get("checks") or {}).items():
            for ex in chk.get("examples") or []:
                rows.append(("Annotation", chk["title"], ex if ex in case_ids else "",
                             "" if ex in case_ids else str(ex), None))
        for u in an.get("unusual_objects") or []:
            word = {"area_px": "size", "aspect": "shape (aspect)"}.get(u["feature"], u["feature"])
            rows.append(("Annotation", f"Unusual object {word}", u["case_id"],
                         f"{u['category']}: {u['feature']} = {u['value']:.4g} (score {u['score']:+.1f})", None))
        self.rows = rows
        kinds = {}
        for r in rows:
            kinds[(r[0], r[1])] = kinds.get((r[0], r[1]), 0) + 1
        self.filter.blockSignals(True)
        self.filter.clear()
        self.filter.addItem(f"everything ({len(rows)})", None)
        for (t, p), n in sorted(kinds.items()):
            self.filter.addItem(f"{t}: {p} ({n})", (t, p))
        self.filter.blockSignals(False)
        n_cases = fp["dataset"].get("n_ok") or 1
        probs = {r[2] for r in rows if r[0] == "Quality"}
        info = "; ".join(f"info: {v['title'].split('(')[0].strip().lower()} in "
                         f"{100 * v['count'] / max(q.get('n_cases', 1), 1):.0f} %"
                         for v in (q.get("problems") or {}).values() if v.get("informational") and v["count"])
        self.summary.setText(f"{len(probs)} images with quality findings, "
                             f"{du.get('n_exact_groups', 0)} exact duplicate groups, "
                             f"{du.get('n_near_pairs', 0)} near-duplicate pairs  (of {n_cases} images)"
                             + (f"\n{info}" if info else ""))
        self.fill()

    def fill(self):
        sel = self.filter.currentData()
        rows = [r for r in self.rows if sel is None or (r[0], r[1]) == sel][:20000]
        self.table.setSortingEnabled(False)
        self.table.clearContents()
        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            for j, val in enumerate(r[:4]):
                it = QTableWidgetItem(str(val))
                it.setData(Qt.UserRole, i)
                self.table.setItem(i, j, it)
        self._shown = rows
        self.table.setSortingEnabled(True)
        if not rows:
            self.detail.setPlainText("No findings." if self.rows else
                                     "No quality information in this fingerprint (scanned with an older "
                                     "version: re-scan to get quality checks and duplicates).")

    def show_row(self):
        items = self.table.selectedItems()
        if not items:
            return
        r = self._shown[items[0].data(Qt.UserRole)]
        kind, problem, cid, detail, other = r
        self.v1.show(self.fp, cid)
        has2 = self.v2.show(self.fp, other) if other else False
        self.v2.w.setVisible(bool(other))
        lines = [f"{kind}: {problem}", "", f"case: {cid}"]
        if detail:
            lines.append(f"detail: {detail}")
        case = next((c for c in self.fp.get("cases") or [] if c["case_id"] == cid), None)
        if case and case.get("quality"):
            ql = case["quality"]
            lines.append("")
            lines.append("quality measures: " + ", ".join(
                f"{k}={v:.3g}" for k, v in ql.items() if isinstance(v, float)))
        if kind == "Duplicate":
            lines += ["", "Duplicates inflate the dataset and, when they are in different splits "
                          "(train/test) or folders, leak information between them. Keep one copy."]
            if not has2 and other:
                lines.append("(second preview not available)")
        self.detail.setPlainText("\n".join(lines))
