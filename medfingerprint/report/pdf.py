"""PDF reports: one fingerprint (``fingerprint_report``) or several (``compare_report``).

Pure matplotlib (Agg), no GUI needed:

    mfp report fp.json --out fp.pdf
    mfp compare a.json b.json --pdf comparison.pdf

Layout: A4 portrait pages filled top-down by a tiny flow layout (:class:`Doc`):
headings, wrapped paragraphs, tables that continue on the next page with their
header repeated, and figures.
"""
from __future__ import annotations

import datetime as _dt
import os
import textwrap
from typing import Callable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from ..core import explain as X  # noqa: E402

# Validated categorical order (blue, orange, aqua, yellow, magenta, green, violet, red);
# datasets keep their slot in every chart of a report.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
REGION_COLORS = {"all": SERIES[0], "body": SERIES[1], "foreground": SERIES[2]}
INK, INK2, MUTED, GRID, ZEBRA = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#f4f4f2"
SEQ_BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CRITICAL = "#c0392b"

W, H = 8.27, 11.69          # A4 inches
ML, MR, MT, MB = 0.6, 0.6, 0.65, 0.6
CW = W - ML - MR            # content width


def _style():
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "axes.edgecolor": MUTED,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 1.4,
        "pdf.fonttype": 42,
    })


def fmt(v, nd=3) -> str:
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return "–"
        a = abs(v)
        if a != 0 and (a >= 1e6 or a < 10 ** -nd):
            return f"{v:.{nd}g}"
        if a >= 1000:
            return f"{v:,.0f}"
        return f"{v:.{nd}g}" if a < 1 else f"{v:.{max(nd - 1, 1)}f}".rstrip("0").rstrip(".")
    return str(v)


# ====================================================================== flow layout
class Doc:
    def __init__(self, path: str, title: str, subtitle: str = ""):
        _style()
        self.pdf = PdfPages(path)
        self.title, self.subtitle = title, subtitle
        self.fig = None
        self.y = 0.0
        self.n = 0
        self.toc = []
        self.sec = 0

    # ---------------------------------------------------------------- pages
    def new_page(self):
        self._flush()
        self.fig = plt.figure(figsize=(W, H))
        self.n += 1
        self.fig.text(ML / W, 1 - 0.35 / H, self.title, fontsize=7, color=MUTED, va="center")
        self.fig.text(1 - MR / W, 1 - 0.35 / H, self.subtitle, fontsize=7, color=MUTED,
                      va="center", ha="right")
        self.fig.text(0.5, 0.3 / H, str(self.n), fontsize=7, color=MUTED, ha="center")
        self.y = H - MT

    def _flush(self):
        if self.fig is not None:
            self.pdf.savefig(self.fig)
            plt.close(self.fig)
            self.fig = None

    def ensure(self, h: float):
        if self.fig is None or self.y - h < MB:
            self.new_page()

    def space(self, h=0.12):
        self.y -= h

    def close(self):
        self._flush()
        d = self.pdf.infodict()
        d["Title"] = self.title
        d["Creator"] = "MedFingerPrint"
        d["CreationDate"] = _dt.datetime.now()
        self.pdf.close()

    # ---------------------------------------------------------------- text
    def _t(self, x, y, s, **kw):
        kw.setdefault("color", INK)
        kw.setdefault("va", "top")
        self.fig.text(x / W, y / H, s, **kw)

    def heading(self, text: str, level: int = 1):
        size = {0: 18, 1: 13, 2: 10.5}[level]
        h = size / 72 * 1.9
        self.ensure(h + (0.9 if level < 2 else 0.5))   # keep a heading with what follows
        if level == 1:
            self.sec += 1
            text = f"{self.sec}. {text}"
            self.toc.append((text, self.n))
        self.space(0.08 if level == 2 else 0.14)
        self._t(ML, self.y, text, fontsize=size, weight="bold")
        self.y -= h

    def para(self, text: str, size: float = 8.5, color: str = INK2, indent: float = 0.0):
        width = int((CW - indent) / (size / 72 * 0.5))
        lines = []
        for p in str(text).split("\n"):
            lines += textwrap.wrap(p, width) or [""]
        lh = size / 72 * 1.4
        for ln in lines:
            self.ensure(lh)
            self._t(ML + indent, self.y, ln, fontsize=size, color=color)
            self.y -= lh
        self.y -= 0.05

    def bullets(self, items, size=8.5, color=INK2):
        for it in items:
            self.para("• " + it, size=size, color=color, indent=0.1)

    def kv(self, pairs, size=8.5, key_w=2.0):
        lh = size / 72 * 1.5
        for k, v in pairs:
            self.ensure(lh)
            self._t(ML, self.y, str(k), fontsize=size, color=INK2)
            self._t(ML + key_w, self.y, str(v), fontsize=size)
            self.y -= lh
        self.y -= 0.05

    # ---------------------------------------------------------------- table
    def table(self, header, rows, widths=None, size=7.5, align=None, colors=None,
              section_col: Optional[int] = None):
        """``rows``: list of lists of str. ``colors``: optional per-row list of per-cell
        colours (None = default). Rows continue on new pages with the header repeated."""
        ncol = len(header)
        widths = widths or [CW / ncol] * ncol
        scale = CW / sum(widths)
        widths = [w * scale for w in widths]
        xs = np.concatenate([[ML], ML + np.cumsum(widths)])
        align = align or (["left"] + ["right"] * (ncol - 1))
        rh = size / 72 * 1.75
        cw = size / 72 * 0.52  # average char width

        def cell(i, txt, y, **kw):
            txt = str(txt)
            maxc = max(3, int((widths[i] - 0.08) / cw))
            if len(txt) > maxc:
                txt = txt[: maxc - 1] + "…"
            x = xs[i] + 0.04 if align[i] == "left" else xs[i + 1] - 0.04
            self._t(x, y - rh / 2, txt, fontsize=size, ha=align[i], va="center", **kw)

        def head():
            self.ensure(rh * 2)
            for i, h in enumerate(header):
                cell(i, h, self.y, weight="bold", color=INK2)
            self.fig.add_artist(plt.Line2D([ML / W, (ML + CW) / W], [(self.y - rh) / H] * 2,
                                           color=MUTED, lw=0.6))
            self.y -= rh

        head()
        last_sec = None
        for r, row in enumerate(rows):
            if self.y - rh < MB:
                self.new_page()
                head()
            if section_col is not None:
                if row[section_col] == last_sec:
                    row = list(row)
                    row[section_col] = ""
                else:
                    last_sec = row[section_col]
            if r % 2 == 1:
                self.fig.add_artist(plt.Rectangle((ML / W, (self.y - rh) / H), CW / W, rh / H,
                                                  color=ZEBRA, lw=0, zorder=-1))
            for i, v in enumerate(row):
                c = colors[r][i] if colors and colors[r] and colors[r][i] else INK
                cell(i, v, self.y, color=c)
            self.y -= rh
        self.y -= 0.1

    # ---------------------------------------------------------------- figure
    def figure(self, h: float, draw: Callable, w: float = CW):
        """Reserve ``h`` inches and call ``draw(fig, rect)`` with rect in figure fractions."""
        self.ensure(h)
        rect = [ML / W, (self.y - h) / H, w / W, h / H]
        draw(self.fig, rect)
        self.y -= h + 0.1

    def axes(self, rect, pad=(0.55, 0.4, 0.15, 0.25)):
        """Axes inside rect leaving room for tick labels (left, bottom, right, top inches)."""
        l, b, r, t = pad
        x, y, w, h = rect
        return self.fig.add_axes([x + l / W, y + b / H, w - (l + r) / W, h - (b + t) / H])


# ====================================================================== helpers
def short(s, n: int) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def short_left(s, n: int) -> str:
    s = str(s)
    return s if len(s) <= n else "…" + s[-(n - 1):]


def _g(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _counts_str(c, n=4):
    if not isinstance(c, dict) or not c:
        return "–"
    tot = sum(c.values())
    items = sorted(c.items(), key=lambda kv: -kv[1])
    s = ", ".join(f"{k} ({100 * v / tot:.0f}%)" for k, v in items[:n])
    return s + (f", +{len(items) - n} more" if len(items) > n else "")


def _axes_names(fp):
    dims = 3 if any(len(_g(fp, "geometry", "spacing") or []) == 3 for _ in [0]) else 2
    return ("z", "y", "x") if dims == 3 else ("height", "width")


def _occupied(counts, spec, q=(0.0005, 0.9995)):
    """Intensity range holding the central mass of a histogram (for x limits)."""
    c = np.asarray(counts, dtype=float)[1:-1]
    if c.sum() <= 0:
        return spec.lo, spec.hi
    cdf = np.cumsum(c) / c.sum()
    e = spec.edges()
    i0 = int(np.searchsorted(cdf, q[0]))
    i1 = int(np.searchsorted(cdf, q[1]))
    return float(e[i0]), float(e[min(i1 + 1, len(e) - 1)])


def _rebin(counts, spec, max_bins=600):
    """Merge fine bins for plotting: (edges, density per intensity unit)."""
    c = np.asarray(counts, dtype=float)[1:-1]
    k = max(1, int(np.ceil(c.size / max_bins)))
    n = c.size // k * k
    cc = c[:n].reshape(-1, k).sum(axis=1)
    e = spec.edges()[: n + 1: k]
    tot = np.asarray(counts, dtype=float).sum()
    return e, cc / (tot * spec.width * k) if tot > 0 else cc


def _unit(fp):
    if (fp.get("_scale") or {}).get("from"):
        return "intensity on common 0–255 scale"
    return "HU" if _g(fp, "config", "modality") == "CT" else "intensity"


def _load_preview(fp, cid):
    cf = (_g(fp, "local", "cache_files") or {}).get(cid)
    if not cf or not os.path.isfile(cf):
        return None
    try:
        with np.load(cf, allow_pickle=False) as z:
            for k in ("preview_coronal", "preview_axial"):
                if k in z.files:
                    a = z[k]
                    return a if k == "preview_axial" else a[::-1]   # coronal: head up
    except Exception:
        return None
    return None


def _show_gray(ax, img, ct=False):
    if ct:
        vmin, vmax = -1000, 1000
    else:
        f = img[np.isfinite(img)]
        vmin, vmax = (np.percentile(f, (1, 99)) if f.size else (0, 1))
        if vmax <= vmin:
            vmax = vmin + 1
    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest", aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


# ====================================================================== single report
def fingerprint_report(fp: dict, out: str, side: Optional[dict] = None,
                       outliers: Optional[dict] = None, max_outliers: int = 24) -> str:
    from ..core.config import HistogramSpec
    d = fp["dataset"]
    spec = HistogramSpec.from_dict(fp["histogram_spec"])
    doc = Doc(out, f"MedFingerPrint report: {d['name']}",
              _dt.datetime.now().strftime("%Y-%m-%d %H:%M"))
    doc.new_page()
    doc.heading(f"Dataset fingerprint: {d['name']}", 0)
    doc.para(f"Created {fp.get('created', '?')} with MedFingerPrint {_g(fp, 'software', 'version')} "
             f"(schema {fp.get('schema')}).", color=MUTED)
    _overview(doc, fp)
    _geometry(doc, fp)
    _intensity(doc, fp, spec)
    _bins(doc, fp)
    _nnunet(doc, fp)
    _plan(doc, fp)
    _labels(doc, fp)
    _annotations(doc, fp)
    _quality(doc, fp)
    _duplicates(doc, fp)
    _metadata(doc, fp)
    _outliers(doc, fp, outliers or fp.get("outliers"), max_outliers)
    _failed(doc, fp)
    doc.close()
    return out


def _overview(doc, fp):
    d, g, fl = fp["dataset"], fp["geometry"], fp.get("flags", {})
    doc.heading("Overview")
    pairs = [
        ("Modality", d.get("modality")),
        ("Layout", d.get("layout") or "–"),
        ("Cases", f"{fmt(d.get('n_ok'))} analysed, {fmt(d.get('n_failed'))} failed "
                  f"(of {fmt(d.get('n_cases'))} found)"),
        ("Masks / labels", "yes" if d.get("has_labels") else "no"),
        ("Annotations", "yes" if fp.get("annotations") else "no"),
        ("Original data type", _counts_str(g.get("original_dtype"))),
        ("Channels", _counts_str(g.get("channels"))),
        ("Orientation", _counts_str(g.get("orientation"))),
        ("Body region method", f"{fl.get('body_method')}  (median share of image "
                               f"{fmt(_g(fl, 'body_fraction', 'median'))})"),
        ("Anonymized", fmt(bool(d.get("anonymized")))),
        ("File list SHA-256", (d.get("file_list_sha256") or "")[:32] + "…"),
    ]
    doc.kv(pairs)
    warn = []
    if fl.get("not_hu_suspected"):
        warn.append(f"{len(fl['not_hu_suspected'])} case(s) do not look like Hounsfield units.")
    for w in (fp.get("discovery_warnings") or [])[:6]:
        warn.append(w)
    if warn:
        doc.heading("Warnings", 2)
        doc.bullets(warn, color=CRITICAL)


def _geometry(doc, fp):
    g = fp["geometry"]
    if not g.get("spacing"):
        return
    doc.heading("Geometry")
    axes = ("z (slices)", "y (rows)", "x (columns)") if len(g["spacing"]) == 3 else ("height", "width")
    rows = []
    for key, name, unit in (("spacing", "spacing", "mm"), ("shape", "size", "px"),
                            ("extent_mm", "extent", "mm")):
        for i, ax in enumerate(axes):
            s = g[key][i] if i < len(g.get(key) or []) else {}
            rows.append([f"{name} {ax} ({unit})", fmt(s.get("median")), fmt(s.get("p05")),
                         fmt(s.get("p95")), fmt(s.get("min")), fmt(s.get("max"))])
    an = g.get("anisotropy") or {}
    rows.append(["anisotropy (max/min spacing)", fmt(an.get("median")), fmt(an.get("p05")),
                 fmt(an.get("p95")), fmt(an.get("min")), fmt(an.get("max"))])
    doc.table(["", "median", "p05", "p95", "min", "max"], rows, widths=[2.6, 1, 1, 1, 1, 1])

    cases = [c for c in fp.get("cases") or [] if c.get("status") == "ok" and c.get("geometry")]
    if not cases:
        return
    sp = np.array([c["geometry"]["spacing"] for c in cases if len(c["geometry"]["spacing"]) == len(axes)])
    sh = np.array([c["geometry"]["shape"] for c in cases if len(c["geometry"]["shape"]) == len(axes)])

    def draw(fig, rect):
        x, y, w, h = rect
        n = len(axes)
        for row, (arr, lab) in enumerate(((sp, "spacing (mm)"), (sh, "size (px)"))):
            for i in range(n):
                r = [x + i * w / n, y + (1 - row) * h / 2, w / n, h / 2]
                ax = doc.axes(r, pad=(0.45, 0.42, 0.12, 0.22))
                v = arr[:, i] if arr.size else np.array([])
                v = v[np.isfinite(v)]
                if v.size:
                    bins = min(40, max(5, len(np.unique(v))))
                    ax.hist(v, bins=bins, color=SERIES[0], edgecolor="white", linewidth=0.5)
                ax.set_title(f"{lab}: {axes[i]}", loc="left", color=INK)
                ax.set_ylabel("cases")
    doc.figure(3.6, draw)


def _intensity(doc, fp, spec):
    doc.heading("Intensity")
    unit = _unit(fp)
    doc.para(f"Pooled statistics over all voxels/pixels of each region ({unit}). "
             "Regions: all = whole image; body = anatomy found from intensities (no masks needed); "
             "foreground = voxels inside masks (label > 0).")
    keys = ("count", "mean", "std", "p00_5", "p05", "p25", "p50", "p75", "p95", "p99_5", "entropy_bits")
    names = ("voxels", "mean", "std", "p0.5", "p5", "p25", "p50", "p75", "p95", "p99.5", "entropy")
    regs = list(fp["intensity"].keys())
    rows = [[r] + [fmt(_g(fp, "intensity", r, "pooled", k)) for k in keys] for r in regs]
    doc.table(["region", *names], rows, widths=[1.0, 1.2] + [0.75] * (len(keys) - 1), size=7)

    rows = []
    for r in regs:
        for k in ("mean", "std", "p50"):
            s = _g(fp, "intensity", r, "per_case", k) or {}
            rows.append([r, f"case {k}", fmt(s.get("median")), fmt(s.get("p25")), fmt(s.get("p75")),
                         fmt(s.get("min")), fmt(s.get("max"))])
    doc.heading("Variation between cases", 2)
    doc.para("Each case's own statistic, summarised over cases: a wide p25–p75 range means "
             "the cases look different from each other (scanners, protocols, exposure).")
    doc.table(["region", "per-case value", "median", "p25", "p75", "min", "max"], rows,
              widths=[1, 1.4, 1, 1, 1, 1, 1], section_col=0, align=["left", "left"] + ["right"] * 5)

    lo, hi = np.inf, -np.inf
    for r in regs:
        a, b = _occupied(fp["intensity"][r]["histogram"]["counts"], spec)
        lo, hi = min(lo, a), max(hi, b)

    def draw(fig, rect):
        ax = doc.axes(rect, pad=(0.6, 0.45, 0.15, 0.3))
        for i, r in enumerate(regs):
            e, dens = _rebin(fp["intensity"][r]["histogram"]["counts"], spec)
            ax.stairs(np.where(dens > 0, dens, np.nan), e, color=REGION_COLORS.get(r, SERIES[i % 8]),
                      lw=1.3, label=r)
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_xlabel(unit)
        ax.set_ylabel("density (log)")
        ax.set_title("Pooled intensity histogram per region", loc="left", color=INK)
        ax.legend(loc="upper right")
    doc.figure(3.0, draw)

    cases = [c for c in fp.get("cases") or [] if c.get("status") == "ok"]
    main = "body" if "body" in regs else "all"
    vals = [_g(c, "regions", main, "mean") for c in cases]
    vals = np.array([v for v in vals if isinstance(v, (int, float))], dtype=float)
    if vals.size >= 5:
        def draw2(fig, rect):
            ax = doc.axes(rect, pad=(0.6, 0.45, 0.15, 0.3))
            ax.hist(vals, bins=min(60, max(10, int(np.sqrt(vals.size)))), color=SERIES[0],
                    edgecolor="white", linewidth=0.4)
            ax.set_xlabel(f"mean {unit} of the {main} region, per case")
            ax.set_ylabel("cases")
            ax.set_title(f"How bright is each case? ({main} region)", loc="left", color=INK)
        doc.figure(2.4, draw2)


def _bins(doc, fp):
    bins = fp.get("bins") or []
    if not bins:
        return
    doc.heading("Intensity bins")
    for entry in bins:
        bs = entry["binset"]
        doc.heading(f"{bs.get('name')}", 2)
        if bs.get("description"):
            doc.para(bs["description"])
        regs = list(entry["regions"].keys())
        names = [r["name"] for r in entry["regions"][regs[0]]]
        rng = [f"{fmt(r.get('lo'))} … {fmt(r.get('hi'))}" for r in entry["regions"][regs[0]]]
        rows = []
        for j, n in enumerate(names):
            rows.append([n, rng[j]] + [f"{100 * entry['regions'][r][j]['fraction']:.2f} %" for r in regs])
        doc.table(["bin", "range", *regs], rows, widths=[2.4, 1.6] + [1] * len(regs))

        def draw(fig, rect, entry=entry, regs=regs, names=names):
            ax = doc.axes(rect, pad=(0.6, 0.85, 0.15, 0.25))
            n = len(regs)
            x = np.arange(len(names))
            bw = 0.8 / n
            for i, r in enumerate(regs):
                v = [100 * row["fraction"] for row in entry["regions"][r]]
                ax.bar(x + (i - (n - 1) / 2) * bw, v, width=bw * 0.92, color=REGION_COLORS.get(r, SERIES[i]),
                       label=r)
            ax.set_xticks(x, [short(s, 22) for s in names], rotation=30, ha="right")
            ax.set_ylabel("% of voxels")
            ax.legend(loc="upper right")
            ax.grid(axis="x", visible=False)
        doc.figure(2.6, draw)


def _nnunet(doc, fp):
    blocks = [(k, t) for k, t in (("nnunet_fingerprint", "nnU-Net fingerprint (labels > 0)"),
                                  ("nnunet_fingerprint_body", "nnU-Net-style fingerprint (body region)"))
              if fp.get(k)]
    if not blocks:
        return
    doc.heading("nnU-Net fingerprint")
    doc.para("Same quantities as nnU-Net's dataset_fingerprint.json: foreground intensity statistics "
             "from up to 10^8 sampled foreground voxels, spacings and shapes after cropping to the "
             "nonzero region. The body version uses the intensity body mask as foreground and is "
             "meant for comparing datasets without masks.")
    rows = []
    for k, t in blocks:
        p = _g(fp, k, "foreground_intensity_properties_per_channel", "0") or {}
        rows.append([t, fmt(p.get("mean")), fmt(p.get("std")), fmt(p.get("median")),
                     fmt(p.get("percentile_00_5")), fmt(p.get("percentile_99_5")),
                     fmt(_g(fp, k, "median_relative_size_after_cropping"))])
    doc.table(["", "mean", "std", "median", "p0.5", "p99.5", "rel. size after crop"], rows,
              widths=[2.6, 0.8, 0.8, 0.8, 0.8, 0.8, 1.3])


def _plan(doc, fp):
    plan = fp.get("nnunet_plan")
    if not plan:
        return
    doc.heading("What nnU-Net would do (predicted plan)")
    doc.para(plan.get("note", ""))
    rows = []
    for name, c in (plan.get("configurations") or {}).items():
        rows.append([name, " × ".join(fmt(v) for v in c.get("spacing", [])),
                     " × ".join(fmt(v) for v in c.get("median_shape", [])),
                     " × ".join(str(v) for v in c.get("patch_size", [])),
                     fmt(c.get("batch_size")), " / ".join(str(v) for v in c.get("pool_ops", []))])
    doc.table(["configuration", "target spacing (mm)", "median shape", "patch size", "batch",
               "poolings per axis"], rows, widths=[1.1, 1.6, 1.5, 1.4, 0.6, 1.2], size=7.2)
    pairs = [("Normalization", plan.get("normalization")), ("Foreground source", plan.get("foreground_source"))]
    if plan.get("normalization_params"):
        pairs.append(("Normalization values", ", ".join(f"{k}={fmt(v)}" for k, v in
                                                        plan["normalization_params"].items())))
    doc.kv(pairs)
    if plan.get("remarks"):
        doc.bullets(plan["remarks"])


def _labels(doc, fp):
    labels = fp.get("labels") or {}
    if not labels:
        return
    doc.heading("Labels (from masks)")
    from ..core.aggregate import sampling_note
    note = sampling_note(fp)
    if note:
        doc.para(note)
    is3d = any(_g(l, "volume_ml", "median") is not None for l in labels.values())
    rows = []
    for k, l in labels.items():
        pi = l.get("pooled_intensity") or {}
        size = _g(l, "volume_ml", "median") if is3d else _g(l, "voxels", "median")
        rows.append([k, f"{l.get('n_cases_present')} ({100 * (l.get('frac_cases_present') or 0):.0f}%)",
                     fmt(size), fmt(pi.get("mean")), fmt(pi.get("std")), fmt(pi.get("p50")),
                     fmt(pi.get("p00_5")), fmt(pi.get("p99_5"))])
    doc.table(["label", "cases", "median volume ml" if is3d else "median pixels", "mean", "std",
               "median", "p0.5", "p99.5"], rows, widths=[2.0, 0.9, 1.5, 0.75, 0.75, 0.75, 0.75, 0.75], size=7)


def _annotations(doc, fp):
    an = fp.get("annotations")
    if not an:
        return
    doc.heading("Annotations (objects)")
    doc.para(f"Source: {an.get('source', '?')}. Objects are polygons/boxes drawn by annotators; "
             "every object is measured separately (size, shape, position, contrast to its surroundings).")
    pairs = [("Images with ≥1 object", f"{an.get('n_images_with_objects')} of {an.get('n_images')}"),
             ("Objects in total", fmt(an.get("n_objects"))),
             ("Objects per annotated image", f"median {fmt(_g(an, 'objects_per_image', 'median'))}, "
                                             f"max {fmt(_g(an, 'objects_per_image', 'max'))}")]
    doc.kv(pairs)
    rows = []
    for cat, c in (an.get("categories") or {}).items():
        for key, name in (("area_px", "area (px)"), ("area_frac_image", "area / image"),
                          ("area_mm2", "area (mm²)"), ("bbox_w", "box width (px)"),
                          ("bbox_h", "box height (px)"), ("aspect", "box aspect w/h"),
                          ("n_vertices", "polygon points"), ("contrast", "contrast (inside − ring)"),
                          ("contrast_z", "contrast / image std")):
            s = c.get(key)
            if isinstance(s, dict) and s.get("n"):
                rows.append([cat, name, fmt(s.get("median")), fmt(s.get("p05")), fmt(s.get("p95")),
                             fmt(s.get("min")), fmt(s.get("max"))])
    if rows:
        doc.table(["category", "per object", "median", "p05", "p95", "min", "max"], rows,
                  widths=[1.3, 1.8, 0.9, 0.9, 0.9, 0.9, 0.9], section_col=0,
                  align=["left", "left"] + ["right"] * 5)
    for cat, c in (an.get("categories") or {}).items():
        hm = c.get("position_heatmap")
        if not hm:
            continue
        hm = np.asarray(hm, dtype=float)

        def draw(fig, rect, hm=hm, cat=cat, c=c):
            x, y, w, h = rect
            ax = doc.axes([x, y, w * 0.42, h], pad=(0.3, 0.3, 0.3, 0.25))
            from matplotlib.colors import LinearSegmentedColormap
            cmap = LinearSegmentedColormap.from_list("seq", ["#ffffff"] + SEQ_BLUES)
            im = ax.imshow(hm, cmap=cmap, extent=(0, 1, 1, 0), interpolation="nearest")
            ax.set_title(f"Where are '{cat}' objects? (centres)", loc="left", color=INK)
            ax.set_xlabel("image width →")
            ax.set_ylabel("← top    image height")
            ax.grid(False)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="objects")
            ar = c.get("_area_px_values")
            if ar:
                ax2 = doc.axes([x + w * 0.5, y, w * 0.5, h], pad=(0.55, 0.45, 0.1, 0.25))
                v = np.log10(np.asarray(ar, dtype=float).clip(1))
                ax2.hist(v, bins=40, color=SERIES[0], edgecolor="white", linewidth=0.4)
                ticks = [t for t in range(int(v.min()), int(v.max()) + 2)]
                ax2.set_xticks(ticks, [f"{10 ** t:,}" for t in ticks])
                ax2.set_xlabel("object area (px, log scale)")
                ax2.set_ylabel("objects")
                ax2.set_title("Object sizes", loc="left", color=INK)
        doc.figure(2.9, draw)
    if an.get("cooccurrence"):
        doc.heading("Categories appearing together", 2)
        rows = [[a, b, fmt(n)] for a, b, n in an["cooccurrence"][:20]]
        doc.table(["category", "with", "images"], rows, widths=[2, 2, 1])
    if an.get("unusual_objects"):
        doc.heading("Unusual objects", 2)
        doc.para("Objects whose size or shape is far from the other objects of the same category "
                 "(robust z-score of log area / log aspect ratio > 3.5).")
        rows = [[u["case_id"], u["category"], {"area_px": "area (px)", "aspect": "aspect w/h"}.get(u["feature"], u["feature"]),
                 fmt(u["value"]), f"{u['score']:+.1f}"] for u in an["unusual_objects"][:25]]
        doc.table(["case", "category", "feature", "value", "score"], rows, widths=[3.2, 1.2, 1.0, 0.8, 0.8],
                  align=["left", "left", "left", "right", "right"], size=7)
    chk = an.get("checks") or {}
    items = [f"{v['title']}: {v['count']}" + (f"  (e.g. {', '.join(map(str, v['examples'][:3]))})"
                                             if v.get("examples") else "")
             for v in chk.values() if v.get("count")]
    doc.heading("Consistency checks", 2)
    doc.bullets(items or ["No problems found."], color=CRITICAL if items else INK2)


def _quality(doc, fp):
    q = fp.get("quality")
    if not q:
        return
    doc.heading("Image quality checks")
    doc.para("Rule-based checks per image. Each finding should be confirmed by looking at the image; "
             "they are hints, not diagnoses.")
    rows = []
    colors = []
    for key, v in (q.get("problems") or {}).items():
        info = v.get("informational")
        rows.append([v["title"] + (" [info, not a problem]" if info else ""), fmt(v["count"]),
                     f"{100 * v['count'] / max(q.get('n_cases', 1), 1):.2f} %",
                     ", ".join(map(str, v.get("examples", [])[:2]))])
        colors.append([MUTED if info else None, CRITICAL if v["count"] and not info else None, None, MUTED])
    doc.table(["check", "images", "share", "examples"], rows, widths=[4.1, 0.55, 0.65, 1.8], colors=colors,
              align=["left", "right", "right", "left"], size=7)
    rows = []
    for key, s in (q.get("measures") or {}).items():
        if isinstance(s, dict) and s.get("n"):
            rows.append([X.FEATURES.get(key, (key,))[0], fmt(s.get("median")), fmt(s.get("p05")),
                         fmt(s.get("p95")), fmt(s.get("max"))])
    if rows:
        doc.heading("Quality measures (per image)", 2)
        doc.table(["measure", "median", "p05", "p95", "max"], rows, widths=[3, 1, 1, 1, 1])


def _duplicates(doc, fp):
    du = fp.get("duplicates")
    if not du:
        return
    doc.heading("Duplicates and near-duplicates")
    doc.para(du.get("method", ""))
    pairs = [("Exact duplicate groups", f"{du.get('n_exact_groups', 0)} groups, "
                                        f"{du.get('n_exact_cases', 0)} images"),
             ("Near-duplicate pairs", fmt(du.get("n_near_pairs", 0))),
             ("Pairs across different folders/groups", fmt(du.get("n_cross_group_pairs", 0))),
             ("Excluded (blank/flat images)", fmt(du.get("n_excluded_flat", 0)))]
    doc.kv(pairs, key_w=2.9)
    rows = []
    cross_exact = {tuple(g) for g in du.get("exact_cross_groups") or []}
    for gr in (du.get("exact_groups") or [])[:15]:
        rows.append(["exact", str(len(gr)) + (" ⚠" if tuple(gr) in cross_exact else ""),
                     "  |  ".join(gr[:3]) + (" …" if len(gr) > 3 else "")])
    for p in (du.get("near_pairs") or [])[:25]:
        rows.append([f"near (d={p.get('distance_fine', p['distance'])})", "2" if not p.get("cross") else "2 ⚠",
                     f"{p['a']}  |  {p['b']}"])
    if rows:
        doc.table(["kind", "images", "cases"], rows, widths=[1.1, 0.6, 5.3],
                  align=["left", "right", "left"], size=7)
    if du.get("n_cross_group_pairs"):
        doc.para("⚠ = the two images are in different groups (e.g. train/test folders or different "
                 "patients): possible data leakage.", color=CRITICAL)


def _metadata(doc, fp):
    md = fp.get("metadata") or {}
    if not md:
        return
    doc.heading("Metadata")
    rows = []
    for k, m in md.items():
        if m.get("type") == "numeric":
            rows.append([k, "numeric", f"median {fmt(m.get('median'))}  (p05 {fmt(m.get('p05'))} – "
                                       f"p95 {fmt(m.get('p95'))}), n={m.get('n')}"])
        else:
            rows.append([k, f"{len(m.get('counts') or {})} values", _counts_str(m.get("counts"), 5)])
    doc.table(["field", "type", "distribution"], rows, widths=[1.8, 0.9, 4.3],
              align=["left", "left", "left"], size=7)


def _outliers(doc, fp, res, max_n):
    if not res or not res.get("cases"):
        return
    doc.heading("Outliers")
    n = res.get("n_cases") or fp["dataset"].get("n_ok")
    multi = sum(1 for e in res["cases"] if e["n_features"] >= 2)
    doc.para(f"Method {res.get('method')}, threshold {res.get('threshold')}, at most "
             f"{100 * (res.get('max_frac') or 0):.1f} % of cases per feature and direction"
             + (f", compared within groups '{res['group_by']}'" if res.get("group_by") else "")
             + f". Flagged: {len(res['cases'])} of {n} cases; {multi} on two or more features "
             "(most likely real problems).")
    dims = 3 if len(_g(fp, "geometry", "spacing") or []) == 3 else 2
    by_case: dict = {}
    for f in res.get("flags") or []:
        by_case.setdefault(f["case_id"], []).append(f)
    rows = []
    for e in res["cases"][:60]:
        feats = sorted(by_case.get(e["case_id"], []), key=lambda f: -abs(f["score"]))
        rows.append([e["case_id"], str(e["n_features"]), f"{e['max_score']:.1f}",
                     "; ".join(f"{X.feature_title(f['feature'], dims)} {'↑' if f['score'] > 0 else '↓'}"
                               for f in feats[:3])])
    doc.table(["case", "# features", "max score", "most unusual"], rows, widths=[2.6, 0.7, 0.7, 3.0],
              align=["left", "right", "right", "left"], size=7)

    top = res["cases"][:max_n]
    previews = [(e, _load_preview(fp, e["case_id"])) for e in top]
    previews = [(e, p) for e, p in previews if p is not None]
    if not previews:
        doc.para("(No thumbnails: the scan cache is not available, e.g. anonymized fingerprint or "
                 "cache deleted.)", color=MUTED)
        return
    ct = _g(fp, "config", "modality") == "CT"
    per_row, rows_per_fig = 4, 3
    for start in range(0, len(previews), per_row * rows_per_fig):
        chunk = previews[start: start + per_row * rows_per_fig]
        nrow = int(np.ceil(len(chunk) / per_row))

        def draw(fig, rect, chunk=chunk, nrow=nrow):
            x, y, w, h = rect
            cw_, ch_ = w / per_row, h / nrow
            for k, (e, img) in enumerate(chunk):
                r, c = divmod(k, per_row)
                ax = fig.add_axes([x + c * cw_ + 0.004, y + (nrow - 1 - r) * ch_ + 0.028,
                                   cw_ - 0.008, ch_ - 0.045])
                _show_gray(ax, img, ct)
                ax.set_title(short_left(e["case_id"], 34), fontsize=6, color=INK, pad=2)
                ax.text(0.5, -0.04, f"{e['n_features']} features, max {e['max_score']:.0f}",
                        transform=ax.transAxes, fontsize=6, color=INK2, ha="center", va="top")
        doc.figure(min(2.3 * nrow, H - MT - MB - 0.4), draw)


def _failed(doc, fp):
    failed = fp.get("failed_cases") or []
    if not failed:
        return
    doc.heading("Failed cases")
    rows = [[f["case_id"], str(f.get("error", ""))] for f in failed[:80]]
    doc.table(["case", "error"], rows, widths=[2.5, 4.5], align=["left", "left"], size=7)


# ====================================================================== comparison report
def compare_report(fps: list, out: str, region: Optional[str] = None, binset=None) -> str:
    from ..core import compare as C
    from ..core.bins import load_presets
    names = [C.name_of(fp) for fp in fps]
    regs = C.common_regions(fps)
    region = region or ("body" if "body" in regs else (regs or ["all"])[0])
    doc = Doc(out, "MedFingerPrint comparison: " + " vs ".join(names),
              _dt.datetime.now().strftime("%Y-%m-%d %H:%M"))
    doc.new_page()
    doc.heading("Dataset comparison", 0)
    doc.para(f"Region for intensity comparison: {region}. Colours identify datasets in all charts.")
    note = C.scale_note(fps)
    if note:
        doc.para(note)
    doc.kv([(f"■ {short(n, 28)}", short_left(fp.get("_path", ""), 80)) for n, fp in zip(names, fps)])
    colors_kv = doc.fig.texts[-2 * len(fps):]
    for i, t in enumerate(colors_kv[0::2]):
        t.set_color(SERIES[i % 8])

    # ---- statistics table
    doc.heading("Statistics side by side")
    rows, colors = [], []
    two = len(fps) == 2
    for sec, metric, vals, _ in C.stats_rows(fps, region):
        r = [sec, metric] + [fmt(v) for v in vals]
        col = [None] * len(r)
        if two:
            dabs, rel = C.difference(vals[0], vals[1])
            r.append("" if rel is None else f"{rel:+.1f} %")
            col.append(CRITICAL if rel is not None and abs(rel) >= 20 else None)
        rows.append(r)
        colors.append(col)
    header = ["section", "metric"] + [short(n, 18) for n in names] + (["Δ %"] if two else [])
    nds = len(names)
    widths = [1.6, 1.9] + [3.5 / nds] * nds + ([0.7] if two else [])
    doc.table(header, rows, widths=widths, size=6.8, colors=colors, section_col=0,
              align=["left", "left"] + ["right"] * (len(header) - 2))
    if two:
        doc.para("Δ % = (second − first) / first; red when the difference is 20 % or more.", color=MUTED)

    # ---- histograms
    doc.heading("Intensity distributions")
    unit = _unit(fps[0])
    items = [C.region_hist(fp, region) for fp in fps]
    x, F = C.cdfs_on_grid(items)

    def draw(fig, rect):
        xx, yy, w, h = rect
        ax = doc.axes([xx, yy + h / 2, w, h / 2], pad=(0.6, 0.4, 0.15, 0.3))
        lo, hi = np.inf, -np.inf
        for i, (fp, (c, s)) in enumerate(zip(fps, items)):
            e, dens = _rebin(c, s)
            ax.stairs(np.where(dens > 0, dens, np.nan), e, color=SERIES[i % 8], lw=1.3, label=names[i])
            a, b = _occupied(c, s)
            lo, hi = min(lo, a), max(hi, b)
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylabel("density (log)")
        ax.set_title(f"Pooled histogram ({region})", loc="left", color=INK)
        ax.legend(loc="upper right")
        ax2 = doc.axes([xx, yy, w, h / 2], pad=(0.6, 0.45, 0.15, 0.3))
        for i, f in enumerate(F):
            ax2.plot(x, f, color=SERIES[i % 8], lw=1.4, label=names[i])
        ax2.set_xlim(lo, hi)
        ax2.set_ylim(0, 1)
        ax2.set_xlabel(unit)
        ax2.set_ylabel("share of voxels ≤ x")
        ax2.set_title("Cumulative distribution (vertical gaps = KS distance)", loc="left", color=INK)
    doc.figure(5.2, draw)

    # ---- distances
    if len(fps) >= 2:
        doc.heading("Distances between datasets")
        doc.para("wasserstein = average intensity shift needed to turn one distribution into the other "
                 f"(in {unit}); ks = largest gap between the cumulative curves (0–1); js = Jensen-Shannon "
                 "divergence (0 = identical, 1 = no overlap).")
        for m in C.DISTANCES:
            D = C.distance_matrix(fps, region, m)
            rows = [[names[i]] + [fmt(D[i, j]) for j in range(len(names))] for i in range(len(names))]
            doc.heading(m, 2)
            doc.table([""] + [short(n, 16) for n in names], rows,
                      widths=[2] + [5 / len(names)] * len(names))

    # ---- bins
    doc.heading("Intensity bins")
    is_ct = all(_g(fp, "config", "modality") == "CT" for fp in fps)
    if binset is not None:
        bs = binset
    elif is_ct:
        bs = load_presets()["ct_tissue"]
    else:
        from ..core.bins import quantile_bins
        c0, s0 = items[0]
        bs = quantile_bins(c0, s0, 4)
        bs.name = f"quartiles of {names[0]}"
    fr = C.bin_fractions(fps, region, bs)
    bnames = [b.name for b in bs.bins]
    rows = [[n] + [f"{100 * fr[i, j]:.2f} %" for i in range(len(fps))] for j, n in enumerate(bnames)]
    doc.para(f"Bin set: {bs.name}. {bs.description or ''}")
    doc.table(["bin"] + [short(n, 18) for n in names], rows,
              widths=[2.5] + [4.5 / len(names)] * len(names))

    def draw_b(fig, rect):
        ax = doc.axes(rect, pad=(0.6, 0.9, 0.15, 0.2))
        n = len(fps)
        xs = np.arange(len(bnames))
        bw = 0.8 / n
        for i in range(n):
            ax.bar(xs + (i - (n - 1) / 2) * bw, 100 * fr[i], width=bw * 0.92, color=SERIES[i % 8],
                   label=names[i])
        ax.set_xticks(xs, [short(s, 22) for s in bnames], rotation=30, ha="right")
        ax.set_ylabel("% of voxels")
        ax.legend(loc="upper right")
        ax.grid(axis="x", visible=False)
    doc.figure(2.8, draw_b)

    # ---- labels
    lr = C.label_rows(fps)
    if lr:
        doc.heading("Labels present in several datasets")
        rows = [[lab, m] + [fmt(v) for v in vals] for lab, m, vals in lr]
        doc.table(["label", "metric"] + [short(n, 16) for n in names], rows,
                  widths=[1.8, 1.6] + [3.6 / len(names)] * len(names), size=7, section_col=0,
                  align=["left", "left"] + ["right"] * len(names))
    doc.close()
    return out
