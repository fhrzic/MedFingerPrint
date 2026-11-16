"""Run case analysis over a dataset with multiprocessing and a per-case cache.

Each finished case is written to ``<cache_dir>/<key>.npz``. The key covers the
case spec, file sizes/mtimes and the config hash, so interrupted scans resume
and unchanged cases are never recomputed.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import traceback
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .case_analysis import analysis_channel
from .config import CT_HISTOGRAM, HistogramSpec, ScanConfig
from .discovery import CaseSpec, Discovery
from .io import load_image
from .schema import jsonify


class ScanCancelled(Exception):
    pass


@dataclass
class ScanOutput:
    discovery: Discovery
    config: ScanConfig
    outcomes: list = field(default_factory=list)   # per case: {case_id, status, cache|error}
    cache_dir: Optional[str] = None


# ------------------------------------------------------------------ cache helpers
def _stat_sig(path: str):
    try:
        st = os.stat(path)
    except OSError:
        return None
    if os.path.isdir(path):
        return [int(st.st_mtime), len(os.listdir(path))]
    return [st.st_size, int(st.st_mtime)]


def case_cache_key(spec: CaseSpec, cfg_hash: str) -> str:
    sig = {"spec": spec.to_dict(), "cfg": cfg_hash, "img": _stat_sig(spec.image)}
    if spec.mask:
        if spec.mask.kind == "binary_dir":
            sig["mask"] = {k: _stat_sig(p) for k, p in spec.mask.files.items()}
        elif spec.mask.kind == "objects":
            pass  # the objects themselves are part of the spec
        else:
            sig["mask"] = _stat_sig(spec.mask.path)
    return hashlib.sha1(json.dumps(sig, sort_keys=True, default=str).encode()).hexdigest()


def _samples_path(path: str) -> str:
    return path[:-4] + ".samples.npz"


def save_case_cache(path: str, result: dict, arrays: dict) -> None:
    """Histograms/previews go compressed into ``<key>.npz``; nnU-Net foreground
    samples (random floats, incompressible and up to 10^7 per case for small
    datasets) go uncompressed into ``<key>.samples.npz``. The main file is written
    last, so its presence marks a complete entry."""
    samples = {k: v for k, v in arrays.items() if k.endswith("_samples")}
    rest = {k: v for k, v in arrays.items() if not k.endswith("_samples")}
    if samples:
        tmp_s = path[:-4] + ".tmp.samples.npz"
        np.savez(tmp_s, **samples)
        os.replace(tmp_s, _samples_path(path))
    tmp = path[:-4] + ".tmp.npz"
    payload = np.frombuffer(json.dumps(jsonify(result)).encode(), dtype=np.uint8)
    np.savez_compressed(tmp, __result__=payload, **rest)
    os.replace(tmp, path)


def load_case_cache(path: str, arrays: bool = True, keys: Optional[tuple] = None,
                    skip_previews: bool = False):
    out = {}
    with np.load(path, allow_pickle=False) as z:
        result = json.loads(z["__result__"].tobytes().decode())
        if arrays:
            for k in z.files:
                if k == "__result__" or (skip_previews and k.startswith("preview")):
                    continue
                if keys is None or k in keys:
                    out[k] = z[k]
    sp = _samples_path(path)
    if arrays and os.path.isfile(sp):
        with np.load(sp, allow_pickle=False) as z:
            for k in z.files:
                if keys is None or k in keys:
                    out[k] = z[k]
    return result, out


# ------------------------------------------------------------------ histogram auto
def auto_histogram(cases: list, cfg: ScanConfig, n_probe: int = 20,
                   warnings: Optional[list] = None) -> HistogramSpec:
    """Pick a dataset-wide histogram spec (all cases must share one).

    For CT, the fixed 1-HU spec is used only if the probed images look like HU
    (air below -500 and values above 2). Otherwise -- e.g. normalised [0, 1]
    volumes -- a generic range is chosen, the HU body mask is disabled and a
    warning is recorded, because HU bins would collapse the data into one bin.
    """
    n_probe = 6 if cfg.is_ct else n_probe
    probe_idx = np.unique(np.linspace(0, len(cases) - 1, min(n_probe, len(cases))).astype(int))
    dtypes, vmin, vmax = set(), np.inf, -np.inf
    hu_votes = []
    for i in probe_idx:
        c = cases[i]
        try:
            img = load_image(c.image_kind, c.image, c.series_uid)
        except Exception:
            continue
        g = analysis_channel(img, cfg.gray_mode)
        g = g[np.isfinite(g)]
        dtypes.add(img.original_dtype)
        if g.size:
            gmin, gmax = float(g.min()), float(g.max())
            vmin, vmax = min(vmin, gmin), max(vmax, gmax)
            hu_votes.append(gmin < -500 and gmax > 2)
    if not np.isfinite(vmin):
        raise RuntimeError("Could not read any probe image to choose a histogram range")
    if cfg.is_ct:
        if sum(hu_votes) * 2 >= len(hu_votes):
            return CT_HISTOGRAM
        if cfg.resolved_body_method() == "hu":
            cfg.body_method = "otsu"
        msg = (f"CT selected but probed images are not in HU (range {vmin:g} to {vmax:g}): "
               "using an automatic histogram range, HU bins disabled, body mask by Otsu")
        if warnings is not None:
            warnings.append(msg)
    ints = all(np.issubdtype(np.dtype(d), np.integer) for d in dtypes)
    if ints and dtypes <= {"uint8", "bool"}:
        return HistogramSpec(0.0, 256.0, 1.0)
    if ints and dtypes <= {"int8"}:
        return HistogramSpec(-128.0, 128.0, 1.0)
    if ints and vmin >= 0:
        bits = int(np.ceil(np.log2(vmax + 1))) if vmax > 0 else 1
        bits = max(bits, 12)  # leave headroom: unprobed images may be brighter
        if vmax > 4095:
            bits = max(bits, 16)  # probes may miss brighter images: use the full dtype range
        width = float(2 ** max(0, bits - 12))
        return HistogramSpec(0.0, float(2 ** bits), width)
    # float or signed: pad the probed range by 10% and use 4096 bins
    pad = 0.1 * (vmax - vmin if vmax > vmin else 1.0)
    lo, hi = np.floor(vmin - pad), np.ceil(vmax + pad)
    if ints:
        width = max(1.0, float(np.ceil((hi - lo) / 4096)))
        return HistogramSpec(float(lo), float(lo + width * np.ceil((hi - lo) / width)), width)
    return HistogramSpec(float(lo), float(hi), float((hi - lo) / 4096))


TASKS_PER_WORKER_BATCH = 100  # cases per worker before its process is replaced

# Peak memory of one worker: measured on SAROS/TotalSegmentator CT with masks, about 24
# bytes per voxel (float volume, label maps, body mask, nnU-Net crop) plus the interpreter.
BYTES_PER_VOXEL = 24
WORKER_BASE_BYTES = 400 * 2 ** 20
MEMORY_RESERVE_BYTES = 3 * 2 ** 30    # left free for the GUI / browser / desktop
MEMORY_USE_FRACTION = 0.8


def available_memory() -> Optional[int]:
    """MemAvailable from /proc/meminfo (Linux); None elsewhere."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def estimate_case_bytes(spec: CaseSpec) -> int:
    """Rough peak memory of analysing one case, from the file headers/sizes (cheap)."""
    vox = 0
    try:
        if spec.image_kind == "dicom_series":
            with os.scandir(spec.image) as it:
                nbytes = sum(e.stat().st_size for e in it if e.is_file())
            vox = nbytes // 2                          # CT: 2 bytes per voxel
        elif spec.image_kind == "sitk":
            import SimpleITK as sitk
            r = sitk.ImageFileReader()
            r.SetFileName(spec.image)
            r.ReadImageInformation()
            vox = int(np.prod(r.GetSize())) * r.GetNumberOfComponents()
        else:                                          # 2D files are small
            vox = os.path.getsize(spec.image) * 4
    except Exception:
        vox = 0
    extra = 1.5 if spec.mask is not None and spec.mask.kind == "binary_dir" else 1.0
    return int(WORKER_BASE_BYTES + vox * BYTES_PER_VOXEL * extra)


def _memory_budget() -> Optional[int]:
    avail = available_memory()
    if avail is None:
        return None
    return max(int((avail - MEMORY_RESERVE_BYTES) * MEMORY_USE_FRACTION), 2 ** 30)


# ------------------------------------------------------------------ worker
def _worker(spec_dict: dict, cfg_dict: dict, cache_file: str) -> dict:
    import SimpleITK as sitk

    from .case_analysis import analyze_case
    # ITK prints warnings to stderr (e.g. slice spacing non-uniform by <1e-3 mm), which
    # floods the progress output on DICOM series. Real failures still raise.
    sitk.ProcessObject.SetGlobalWarningDisplay(False)
    spec = CaseSpec.from_dict(spec_dict)
    try:
        cfg = ScanConfig.from_dict(cfg_dict)
        result, arrays = analyze_case(spec, cfg)
        save_case_cache(cache_file, result, arrays)
        return {"case_id": spec.case_id, "status": "ok", "cache": cache_file}
    except Exception as e:  # one bad file must not kill the scan
        return {"case_id": spec.case_id, "status": "error",
                "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}


def _run_parallel(todo: list, cfg_dict: dict, workers: int, record: Callable,
                  cancel: Optional[Callable], notify: Optional[Callable]) -> None:
    """Process pool with a memory budget and crash recovery.

    Cases are started only while the summed memory estimate of the running ones fits in
    the available memory, so large volumes run fewer at a time. If a worker is killed
    anyway (the out-of-memory killer), Python marks every unfinished case of that pool as
    failed; those are re-queued instead: the pool restarts with half the workers and the
    cases that were running are retried one at a time at the end. Only a case that kills
    a worker while running alone is reported as failed.

    A fresh pool is used per batch instead of max_tasks_per_child: that option deadlocks in
    CPython's ProcessPoolExecutor once workers are recycled (seen on 3.13 after workers x
    max_tasks cases). Batches also free worker memory and bound the queue size."""
    from collections import deque
    from concurrent.futures import FIRST_COMPLETED, wait

    say = notify or (lambda m: None)
    ctx = mp.get_context("spawn")
    budget = _memory_budget()
    pending = deque(todo)
    solo: list = []                       # cases that were running when a worker died
    est_cache: dict = {}
    cur = max(1, workers)
    warned_budget = False

    def est(i, c):
        if i not in est_cache:
            est_cache[i] = estimate_case_bytes(c)
        return est_cache[i]

    def _pool(items, n_workers):
        """Run ``items`` in one pool; return the cases left unfinished by a crash."""
        nonlocal warned_budget
        q = deque(items)
        ex = ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx)
        inflight, used, submitted, crashed = {}, 0, 0, []
        limit = max(n_workers * TASKS_PER_WORKER_BATCH, 1)
        try:
            while q or inflight:
                while q and len(inflight) < n_workers and submitted < limit:
                    i, c, cf = q[0]
                    e = est(i, c)
                    if budget is not None and inflight and used + e > budget:
                        if not warned_budget and len(inflight) < n_workers:
                            warned_budget = True
                            say(f"Memory: large images, running {len(inflight)} at a time instead of "
                                f"{n_workers} (about {e / 2 ** 30:.1f} GB each, "
                                f"{budget / 2 ** 30:.0f} GB usable)")
                        break
                    q.popleft()
                    inflight[ex.submit(_worker, c.to_dict(), cfg_dict, cf)] = (i, c, cf, e)
                    used += e
                    submitted += 1
                if not inflight:
                    break                                   # batch done: new pool
                done_set, _ = wait(inflight, return_when=FIRST_COMPLETED)
                for f in done_set:
                    i, c, cf, e = inflight.pop(f)
                    used -= e
                    try:
                        out = f.result()
                    except BrokenProcessPool:
                        crashed.append((i, c, cf))
                        continue
                    record(i, out)
                if crashed:                                 # the pool is dead
                    crashed += [(i, c, cf) for i, c, cf, e in inflight.values()]
                    inflight.clear()
                    break
                if cancel and cancel():
                    ex.shutdown(wait=False, cancel_futures=True)
                    raise ScanCancelled()
        finally:
            ex.shutdown(wait=True, cancel_futures=True)
        return crashed, list(q)

    while pending:
        crashed, rest = _pool(list(pending), min(cur, len(pending)))
        pending = deque(rest)
        if crashed:
            solo += crashed
            new = max(1, cur // 2)
            say(f"A worker process was killed (probably out of memory). {len(crashed)} image(s) "
                f"that were running will be retried one at a time at the end; continuing with "
                f"{new} worker(s) instead of {cur}.")
            cur = new
    for i, c, cf in solo:
        if cancel and cancel():
            raise ScanCancelled()
        crashed, _ = _pool([(i, c, cf)], 1)
        if crashed:
            record(i, {"case_id": c.case_id, "status": "error",
                       "error": "worker process crashed even when run alone (out of memory? "
                                f"estimated {est(i, c) / 2 ** 30:.1f} GB needed)"})
    if solo:
        say(f"Retried {len(solo)} image(s) one at a time after the crash.")


def run_scan(discovery: Discovery, cfg: ScanConfig, cache_dir: str, workers: int = 4,
             progress: Optional[Callable] = None, cancel: Optional[Callable[[], bool]] = None,
             use_cache: bool = True, notify: Optional[Callable[[str], None]] = None) -> ScanOutput:
    """Analyse all cases. ``progress(done, total, case_id, status)`` is called per case;
    ``cancel()`` returning True aborts with :class:`ScanCancelled`; ``notify(message)``
    reports memory limits and crash recovery."""
    cases = discovery.cases
    if not cases:
        raise ValueError("No cases found")
    if discovery.label_names and not cfg.label_names:
        cfg.label_names = dict(discovery.label_names)
    cfg.nnunet_samples_per_case = int(cfg.nnunet_total_fg_samples // len(cases))
    if cfg.histogram is None:
        cfg.histogram = auto_histogram(cases, cfg, warnings=discovery.warnings)
    os.makedirs(cache_dir, exist_ok=True)
    cfg_hash = cfg.hash()
    cfg_dict = cfg.to_dict()

    outcomes: list = [None] * len(cases)
    todo = []
    for i, c in enumerate(cases):
        cf = os.path.join(cache_dir, case_cache_key(c, cfg_hash) + ".npz")
        if use_cache and os.path.isfile(cf):
            outcomes[i] = {"case_id": c.case_id, "status": "ok", "cache": cf, "cached": True}
        else:
            todo.append((i, c, cf))

    total = len(cases)
    done = total - len(todo)
    if progress:
        progress(done, total, None, "cached")

    def _record(i, out):
        nonlocal done
        outcomes[i] = out
        done += 1
        if progress:
            progress(done, total, out["case_id"], out["status"])

    if workers <= 1:
        for i, c, cf in todo:
            if cancel and cancel():
                raise ScanCancelled()
            _record(i, _worker(c.to_dict(), cfg_dict, cf))
    elif todo:
        _run_parallel(todo, cfg_dict, workers, _record, cancel, notify)
    return ScanOutput(discovery, cfg, outcomes, cache_dir)
