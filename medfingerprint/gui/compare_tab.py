"""Compare tab: load two or more fingerprint JSONs and compare them side by side."""
from __future__ import annotations

import csv
import os

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPushButton, QSplitter,
    QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from ..core import compare as C
from . import help as H

COLORS = [(31, 119, 180), (214, 39, 40), (44, 160, 44), (255, 127, 14), (148, 103, 189),
          (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207)]
DIST_HELP = {
    "wasserstein": "Wasserstein-1 (earth mover's) distance in intensity units (HU for CT): the average "
                   "shift needed to turn one intensity distribution into the other. 0 = identical.",
    "ks": "Kolmogorov-Smirnov statistic: largest difference between the two cumulative distributions, "
          "0 (identical) to 1 (no overlap).",
    "js": "Jensen-Shannon divergence (base 2): 0 = identical, 1 = completely different. Symmetric, "
          "sensitive to differences in shape, not only shift.",
}


def fmt(v):
    if v is None:
        return "–"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        a = abs(v)
        if a == 0:
            return "0"
        if a >= 1000 or a < 0.01:
            return f"{v:.4g}"
        return f"{v:.3f}".rstrip("0").rstrip(".")
    return str(v)


def item(v, tip=None, color=None, align_right=False):
    it = QTableWidgetItem(fmt(v) if not isinstance(v, str) else v)
    if tip:
        it.setToolTip(tip)
    if color is not None:
        it.setForeground(QBrush(QColor(*color)))
    if align_right:
        it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
    return it


class CompareTab(QWidget):
    def __init__(self, current_path_fn=None):
        super().__init__()
        self.fps: list = []
        self.current_path_fn = current_path_fn
        v = QVBoxLayout(self)
        top = QHBoxLayout()
        for text, fn, tip in (
                ("Add fingerprints…", self.add_dialog, "Load one or more fingerprint JSON files"),
                ("Add currently open", self.add_current, "Add the fingerprint open in the other tabs"),
                ("Split open dataset…", self.split_current,
                 "Make one fingerprint per group of the open dataset (e.g. Fractured vs Non_fractured "
                 "folder, or a metadata column such as hand/leg) and compare the groups. Uses the scan "
                 "cache, no images are re-read."),
                ("Remove selected", self.remove_selected, ""), ("Clear", self.clear, "")):
            b = QPushButton(text)
            b.clicked.connect(fn)
            if tip:
                b.setToolTip(tip)
            top.addWidget(b)
        top.addSpacing(20)
        top.addWidget(QLabel("Region:"))
        self.region = QComboBox()
        self.region.setToolTip("all = every voxel; body = body found from intensities; "
                               "foreground = inside the masks (only datasets with labels)")
        self.region.currentTextChanged.connect(self.refresh)
        top.addWidget(self.region)
        top.addSpacing(12)
        top.addWidget(QLabel("Scale:"))
        self.scale = QComboBox()
        for key, label in (("original", "original values"), ("robust", "0–255 (p0.5–p99.5)"),
                           ("minmax", "0–255 (min–max)"), ("bits", "0–255 (bit depth)")):
            self.scale.addItem(label, key)
            self.scale.setItemData(self.scale.count() - 1, C.SCALES[key], Qt.ToolTipRole)
        self.scale.setToolTip(
            "Put datasets with different value ranges (e.g. 16-bit 0–65535 and 8-bit 0–255) on one\n"
            "0–255 scale. Each dataset is rescaled linearly with its own range:\n"
            "• p0.5–p99.5: recommended, ignores rare extreme pixels\n"
            "• min–max: the full range of the dataset\n"
            "• bit depth: like converting the files to 8 bit (12-bit data → 0–4095 maps to 0–255)\n"
            "Shape measures (skewness, kurtosis, entropy) do not change. Not for comparing CT HU values.")
        self.scale.currentIndexChanged.connect(self.on_scale)
        top.addWidget(self.scale)
        top.addStretch(1)
        self.export_btn = QPushButton("Export CSV…")
        self.export_btn.clicked.connect(self.export_csv)
        top.addWidget(self.export_btn)
        self.pdf_btn = QPushButton("Export PDF…")
        self.pdf_btn.setToolTip("PDF with the statistics table, histograms, distances, the current "
                                "bin set and labels")
        self.pdf_btn.clicked.connect(self.export_pdf)
        top.addWidget(self.pdf_btn)
        top.addWidget(H.help_button(self, "compare"))
        v.addLayout(top)
        self.scale_note = QLabel("")
        self.scale_note.setWordWrap(True)
        self.scale_note.setStyleSheet("color: #8a5a00")
        self.scale_note.setVisible(False)
        v.addWidget(self.scale_note)

        split = QSplitter(Qt.Horizontal)
        self.list = QListWidget()
        self.list.setToolTip("Loaded fingerprints. Their colour is used in the plots.")
        split.addWidget(self.list)
        self.sub = QTabWidget()
        split.addWidget(self.sub)
        split.setSizes([260, 1100])
        v.addWidget(split, 1)

        # statistics
        self.stats = QTableWidget()
        self.stats.setEditTriggers(QTableWidget.NoEditTriggers)
        self.stats.setAlternatingRowColors(True)
        self.sub.addTab(self.stats, "Statistics")
        # histograms
        w = QWidget()
        hv = QVBoxLayout(w)
        row = QHBoxLayout()
        self.logy = QCheckBox("log scale")
        self.logy.setChecked(True)
        self.logy.toggled.connect(self.draw_hist)
        self.cdf = QCheckBox("cumulative (CDF)")
        self.cdf.setToolTip("Cumulative distribution: the fraction of voxels below each intensity. "
                            "Shifts and spreads are easy to compare.")
        self.cdf.toggled.connect(self.draw_hist)
        row.addWidget(self.logy)
        row.addWidget(self.cdf)
        row.addWidget(QLabel("Each curve is normalised (area = 1), so datasets of different size are comparable."))
        row.addStretch(1)
        hv.addLayout(row)
        self.plot = pg.PlotWidget(background="w")
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.legend = self.plot.addLegend()
        hv.addWidget(self.plot, 1)
        self.sub.addTab(w, "Histograms")
        # distances
        w = QWidget()
        dv = QVBoxLayout(w)
        row = QHBoxLayout()
        row.addWidget(QLabel("Distance:"))
        self.metric = QComboBox()
        self.metric.addItems(C.DISTANCES)
        self.metric.currentTextChanged.connect(self.draw_dist)
        row.addWidget(self.metric)
        row.addStretch(1)
        dv.addLayout(row)
        self.dist_help = QLabel()
        self.dist_help.setWordWrap(True)
        dv.addWidget(self.dist_help)
        self.dist = QTableWidget()
        self.dist.setEditTriggers(QTableWidget.NoEditTriggers)
        dv.addWidget(self.dist, 1)
        self.sub.addTab(w, "Distances")
        # bins
        w = QWidget()
        bv = QVBoxLayout(w)
        row = QHBoxLayout()
        row.addWidget(QLabel("Bins:"))
        self.preset = QComboBox()
        self.preset.setEditable(False)
        self.preset.currentTextChanged.connect(self.draw_bins)
        self._user_preset = False
        self.preset.textActivated.connect(lambda _: setattr(self, "_user_preset", True))
        row.addWidget(self.preset)
        self.edges = QLineEdit()
        self.edges.setPlaceholderText("or custom edges, e.g. 0 120 180 216 (Enter)")
        self.edges.returnPressed.connect(self.custom_edges)
        row.addWidget(self.edges, 1)
        bv.addLayout(row)
        bsplit = QSplitter(Qt.Vertical)
        self.bins_table = QTableWidget()
        self.bins_table.setEditTriggers(QTableWidget.NoEditTriggers)
        bsplit.addWidget(self.bins_table)
        self.bar = pg.PlotWidget(background="w")
        self.bar.showGrid(y=True, alpha=0.2)
        self.bar.setLabel("left", "% of voxels")
        bsplit.addWidget(self.bar)
        bv.addWidget(bsplit, 1)
        self.sub.addTab(w, "Bins")
        # labels
        self.labels = QTableWidget()
        self.labels.setEditTriggers(QTableWidget.NoEditTriggers)
        self.labels.setAlternatingRowColors(True)
        self.sub.addTab(self.labels, "Labels")
        self._binset = None
        self.vfps: list = []
        self.update_presets()
        self.refresh()

    # ---------------------------------------------------------------- loading
    def add_paths(self, paths):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for p in paths:
                if any(fp["_path"] == p for fp in self.fps):
                    continue
                try:
                    self.fps.append(C.load_for_compare(p))
                except Exception as e:
                    QMessageBox.warning(self, "Compare", f"{p}\n\n{type(e).__name__}: {e}")
                QApplication.processEvents()
        finally:
            QApplication.restoreOverrideCursor()
        self.rebuild_list()

    def add_dialog(self):
        fs, _ = QFileDialog.getOpenFileNames(self, "Add fingerprints", os.path.expanduser("~/fingerprints"),
                                             "Fingerprint JSON (*.json)")
        if fs:
            self.add_paths(fs)

    def add_current(self):
        p = self.current_path_fn() if self.current_path_fn else None
        if not p:
            QMessageBox.information(self, "Compare", "No fingerprint is open. Use File → Open first, "
                                                     "or 'Add fingerprints…'.")
            return
        self.add_paths([p])

    def split_current(self):
        import json

        from PySide6.QtWidgets import QInputDialog

        from ..core.split import describe_choice, group_choices, save_split, split_fingerprint
        p = self.current_path_fn() if self.current_path_fn else None
        if not p:
            QMessageBox.information(self, "Split", "Open a fingerprint first (File → Open or a finished scan).")
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            with open(p) as fh:
                fp = json.load(fh)
            choices = [f"{g}  —  {describe_choice(fp, g, 4)}" for g in group_choices(fp)]
        finally:
            QApplication.restoreOverrideCursor()
        if not choices:
            QMessageBox.information(self, "Split", "No folder level or metadata column with 2–50 groups found.")
            return
        item, ok = QInputDialog.getItem(
            self, "Split open dataset", "Split by (level:N = N-th folder of the path, meta:… = metadata column):",
            choices, 0, True)
        if not ok or not item.strip():
            return
        group_by = item.split("  ")[0].strip()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            parts = split_fingerprint(fp, group_by)
            out_dir = os.path.splitext(p)[0] + "_split_" + group_by.replace(":", "_")
            paths = save_split(parts, out_dir)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Split", f"{type(e).__name__}: {e}")
            return
        QApplication.restoreOverrideCursor()
        if not paths:
            QMessageBox.information(self, "Split", "No group has at least 3 cases.")
            return
        self.add_paths(paths)
        QMessageBox.information(self, "Split", f"{len(paths)} group fingerprints saved in\n{out_dir}\n"
                                               "and added to the comparison.")

    def remove_selected(self):
        rows = sorted({i.row() for i in self.list.selectedIndexes()}, reverse=True)
        for r in rows:
            del self.fps[r]
        self.rebuild_list()

    def clear(self):
        self.fps = []
        self.rebuild_list()

    def names(self):
        names = [C.name_of(fp) for fp in self.fps]
        # disambiguate equal names
        return [n if names.count(n) == 1 else f"{n} ({os.path.basename(os.path.dirname(fp['_path']))})"
                for n, fp in zip(names, self.fps)]

    def rebuild_list(self):
        self.list.clear()
        for i, (n, fp) in enumerate(zip(self.names(), self.fps)):
            d = fp.get("dataset", {})
            it = QListWidgetItem(f"● {n}\n   {d.get('modality')}, {d.get('n_ok')} cases")
            it.setForeground(QBrush(QColor(*COLORS[i % len(COLORS)])))
            it.setToolTip(fp["_path"])
            self.list.addItem(it)
        regs = C.common_regions(self.fps) if self.fps else []
        cur = self.region.currentText()
        self.region.blockSignals(True)
        self.region.clear()
        self.region.addItems(regs)
        if cur in regs:
            self.region.setCurrentText(cur)
        elif "body" in regs:
            self.region.setCurrentText("body")
        self.region.blockSignals(False)
        self.update_presets()
        self.refresh()

    def update_presets(self):
        from ..core.bins import load_presets
        cur = self.preset.currentText()
        self.preset.blockSignals(True)
        self.preset.clear()
        ct = bool(self.vfps) and all(fp.get("dataset", {}).get("modality") == "CT" and
                                    fp.get("histogram_spec", {}).get("lo", 0) <= -1000 for fp in self.vfps)
        self.preset.addItems(["quartiles of first dataset", "deciles of first dataset"])
        if ct:
            self.preset.addItems(list(load_presets().keys()))
            self.preset.setCurrentText("ct_tissue")
        if self._user_preset and cur and self.preset.findText(cur) >= 0:
            self.preset.setCurrentText(cur)  # keep the user's choice
        self.preset.blockSignals(False)

    # ---------------------------------------------------------------- drawing
    def on_scale(self):
        self._user_preset = False
        self._binset = None
        self.refresh()

    def refresh(self):
        mode = self.scale.currentData() if hasattr(self, "scale") else "original"
        self.vfps = [C.rescale_fp(fp, mode) for fp in self.fps]
        self.update_presets()
        note = C.scale_note(self.vfps)
        self.scale_note.setText(note or "")
        self.scale_note.setVisible(bool(note))
        self.draw_stats()
        self.draw_hist()
        self.draw_dist()
        self.draw_bins()
        self.draw_labels()

    def _headers(self, table, first, extra_delta=True):
        names = self.names()
        cols = first + names
        two = len(self.vfps) == 2 and extra_delta
        if two:
            cols += [f"Δ ({names[1]} − {names[0]})", "Δ %"]
        table.clear()
        table.setColumnCount(len(cols))
        table.setHorizontalHeaderLabels(cols)
        for j in range(len(names)):
            table.horizontalHeaderItem(len(first) + j).setForeground(QBrush(QColor(*COLORS[j % len(COLORS)])))
        return two

    def draw_stats(self):
        t = self.stats
        if not self.vfps:
            t.clear()
            t.setRowCount(0)
            t.setColumnCount(1)
            t.setHorizontalHeaderLabels(["Add two or more fingerprint JSONs to compare them."])
            t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            return
        rows = C.stats_rows(self.vfps, self.region.currentText() or None)
        two = self._headers(t, ["section", "metric"])
        t.setRowCount(len(rows))
        for i, (sec, metric, vals, tip) in enumerate(rows):
            t.setItem(i, 0, item(sec))
            t.setItem(i, 1, item(metric, tip or None))
            for j, val in enumerate(vals):
                t.setItem(i, 2 + j, item(val, align_right=True))
            if two:
                d, rel = C.difference(vals[0], vals[1])
                big = rel is not None and abs(rel) >= 20
                t.setItem(i, 4, item(d, align_right=True))
                it = item(None if rel is None else f"{rel:+.1f} %", align_right=True)
                if big:
                    it.setForeground(QBrush(QColor(200, 30, 30)))
                    it.setToolTip("Differs by 20 % or more")
                t.setItem(i, 5, it)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        t.resizeColumnsToContents()
        t.setColumnWidth(0, min(t.columnWidth(0), 220))
        self._stats_rows = rows

    def draw_hist(self):
        self.plot.clear()
        self.legend.clear()
        reg = self.region.currentText()
        if not self.vfps or not reg:
            return
        ct = all(fp.get("dataset", {}).get("modality") == "CT" for fp in self.vfps)
        scaled = any(fp.get("_scale") for fp in self.vfps)
        self.plot.setLabel("bottom", "intensity on common 0–255 scale" if scaled else
                           "intensity (HU)" if ct else "intensity")
        if self.cdf.isChecked():
            x, F = C.cdfs_on_grid([C.region_hist(fp, reg) for fp in self.vfps])
            for i, (n, f) in enumerate(zip(self.names(), F)):
                self.plot.plot(x, f, pen=pg.mkPen(COLORS[i % len(COLORS)], width=2), name=n)
            self.plot.setLabel("left", "fraction of voxels below")
            self.plot.setLogMode(y=False)
            return
        lo, hi = np.inf, -np.inf
        for i, (n, fp) in enumerate(zip(self.names(), self.vfps)):
            e, d = C.density(fp, reg)
            nz = np.flatnonzero(d)
            if nz.size:
                lo, hi = min(lo, e[nz[0]]), max(hi, e[nz[-1] + 1])
            y = np.log10(d + 1e-12) if self.logy.isChecked() else d
            if self.logy.isChecked():
                y = np.where(d > 0, y, np.nan)
            self.plot.plot(e, y, stepMode="center", pen=pg.mkPen(COLORS[i % len(COLORS)], width=1.5),
                           name=n, connect="finite")
        self.plot.setLabel("left", "log10(density)" if self.logy.isChecked() else "density")
        if np.isfinite(lo):
            self.plot.setXRange(lo, hi, padding=0.02)
        self.plot.enableAutoRange(axis="y")

    def draw_dist(self):
        m = self.metric.currentText()
        self.dist_help.setText(DIST_HELP.get(m, ""))
        t = self.dist
        t.clear()
        reg = self.region.currentText()
        n = len(self.vfps)
        if n < 2 or not reg:
            t.setRowCount(0)
            t.setColumnCount(0)
            return
        D = C.distance_matrix(self.vfps, reg, m)
        names = self.names()
        t.setRowCount(n)
        t.setColumnCount(n)
        t.setHorizontalHeaderLabels(names)
        t.setVerticalHeaderLabels(names)
        mx = D.max() if D.max() > 0 else 1
        for i in range(n):
            for j in range(n):
                it = item(D[i, j], align_right=True)
                g = int(255 - 120 * D[i, j] / mx)
                it.setBackground(QBrush(QColor(255, g, g)))
                t.setItem(i, j, it)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def custom_edges(self):
        try:
            e = sorted(float(x) for x in self.edges.text().replace(",", " ").split())
        except ValueError:
            QMessageBox.warning(self, "Bins", "Edges must be numbers separated by spaces.")
            return
        if len(e) < 1:
            return
        from ..core.bins import BinSet
        self._binset = BinSet.from_edges(e, open_ends=True)
        self.draw_bins(custom=True)

    def current_binset(self):
        from ..core.bins import load_presets, quantile_bins
        name = self.preset.currentText()
        reg = self.region.currentText()
        if name.startswith(("quartiles", "deciles")):
            c, s = C.region_hist(self.vfps[0], reg)
            return quantile_bins(c, s, 4 if name.startswith("quartiles") else 10)
        return load_presets().get(name)

    def draw_bins(self, *_, custom=False):
        t = self.bins_table
        self.bar.clear()
        reg = self.region.currentText()
        if not self.vfps or not reg:
            t.clear()
            t.setRowCount(0)
            return
        bs = self._binset if custom else self.current_binset()
        if not custom:
            self._binset = None
        if bs is None:
            return
        Fr = 100 * C.bin_fractions(self.vfps, reg, bs)
        two = self._headers(t, ["bin", "range"])
        t.setRowCount(len(bs.bins))
        for i, b in enumerate(bs.bins):
            lo = "−∞" if b.lo is None else fmt(b.lo)
            hi = "∞" if b.hi is None else fmt(b.hi)
            t.setItem(i, 0, item(b.name))
            t.setItem(i, 1, item(f"[{lo}, {hi})"))
            for j in range(len(self.vfps)):
                t.setItem(i, 2 + j, item(f"{Fr[j, i]:.2f} %", align_right=True))
            if two:
                d = Fr[1, i] - Fr[0, i]
                t.setItem(i, 4, item(f"{d:+.2f} pp", "percentage points", align_right=True))
                rel = 100 * d / Fr[0, i] if Fr[0, i] > 0 else None
                t.setItem(i, 5, item(None if rel is None else f"{rel:+.1f} %", align_right=True))
        t.resizeColumnsToContents()
        n = len(self.vfps)
        w = 0.8 / n
        x = np.arange(len(bs.bins))
        for j in range(n):
            self.bar.addItem(pg.BarGraphItem(x=x - 0.4 + w * (j + 0.5), height=Fr[j], width=w,
                                             brush=pg.mkBrush(*COLORS[j % len(COLORS)], 200)))
        ax = self.bar.getAxis("bottom")
        ax.setTicks([[(i, b.name[:18]) for i, b in enumerate(bs.bins)]])
        self._bins = (bs, Fr)

    def draw_labels(self):
        t = self.labels
        rows = C.label_rows(self.vfps) if len(self.vfps) >= 2 else []
        if not rows:
            t.clear()
            t.setRowCount(0)
            t.setColumnCount(1)
            t.setHorizontalHeaderLabels(["Per-label comparison needs at least two fingerprints with the "
                                         "same label names (e.g. two TotalSegmentator-style datasets)."])
            t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            return
        two = self._headers(t, ["label", "metric"])
        t.setRowCount(len(rows))
        for i, (lab, metric, vals) in enumerate(rows):
            t.setItem(i, 0, item(lab))
            t.setItem(i, 1, item(metric))
            for j, val in enumerate(vals):
                t.setItem(i, 2 + j, item(val, align_right=True))
            if two:
                d, rel = C.difference(vals[0], vals[1])
                t.setItem(i, 4, item(d, align_right=True))
                t.setItem(i, 5, item(None if rel is None else f"{rel:+.1f} %", align_right=True))
        t.resizeColumnsToContents()

    # ---------------------------------------------------------------- export
    def export_pdf(self):
        if not self.vfps:
            return
        f, _ = QFileDialog.getSaveFileName(self, "Export comparison PDF",
                                           os.path.expanduser("~/fingerprints/comparison.pdf"), "PDF (*.pdf)")
        if not f:
            return
        f = f if f.lower().endswith(".pdf") else f + ".pdf"
        from ..report import compare_report
        bs = getattr(self, "_bins", None)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            compare_report(self.vfps, f, self.region.currentText() or None, binset=bs[0] if bs else None)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Export PDF", f"{type(e).__name__}: {e}")
            return
        QApplication.restoreOverrideCursor()
        QMessageBox.information(self, "Export PDF", f"Saved {f}")

    def export_csv(self):
        if not self.vfps:
            return
        f, _ = QFileDialog.getSaveFileName(self, "Export comparison", os.path.expanduser("~/fingerprints/comparison.csv"),
                                           "CSV (*.csv)")
        if not f:
            return
        names = self.names()
        with open(f, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["section", "metric"] + names)
            for sec, metric, vals, _ in C.stats_rows(self.vfps, self.region.currentText() or None):
                w.writerow([sec, metric] + ["" if v is None else v for v in vals])
            if getattr(self, "_bins", None):
                bs, Fr = self._bins
                for i, b in enumerate(bs.bins):
                    w.writerow([f"bins ({self.region.currentText()}) %", b.name] + list(Fr[:, i]))
            if len(self.vfps) >= 2:
                for m in C.DISTANCES:
                    D = C.distance_matrix(self.vfps, self.region.currentText(), m)
                    for i, n in enumerate(names):
                        w.writerow([f"distance {m}", n] + list(D[i]))
            for lab, metric, vals in C.label_rows(self.vfps) if len(self.vfps) >= 2 else []:
                w.writerow([f"label {lab}", metric] + ["" if v is None else v for v in vals])
        QMessageBox.information(self, "Export", f"Saved {f}")
