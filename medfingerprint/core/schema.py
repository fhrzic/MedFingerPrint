"""Fingerprint (de)serialisation: a JSON document plus an optional .npz sidecar
holding per-case histograms and per-label dataset histograms."""
from __future__ import annotations

import json
import math
import os
from typing import Optional

import numpy as np


def jsonify(obj):
    """Recursively convert numpy types / NaN / tuples to JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonify(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def sidecar_path(json_path: str) -> str:
    base = json_path[:-5] if json_path.endswith(".json") else json_path
    return base + ".arrays.npz"


def save_fingerprint(fp: dict, sidecar: Optional[dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if sidecar:
        sp = sidecar_path(path)
        fp["sidecar"] = os.path.basename(sp)
        np.savez_compressed(sp, **sidecar)
    tmp = path + ".tmp"
    doc = jsonify(fp)
    cases = doc.pop("cases", None)
    with open(tmp, "w") as fh:
        if cases is None:
            json.dump(doc, fh, indent=1)
        else:
            # dataset-level sections indented (readable); one compact line per case, which
            # keeps fingerprints of 200k+ images about a third smaller and quicker to load
            head = json.dumps(doc, indent=1)
            fh.write(head[:-2] + ',\n "cases": [\n')
            for k, c in enumerate(cases):
                fh.write("  " + json.dumps(c, separators=(",", ":")) + (",\n" if k < len(cases) - 1 else "\n"))
            fh.write(" ]\n}\n")
    os.replace(tmp, path)


def load_fingerprint(path: str, with_sidecar: bool = True):
    with open(path) as fh:
        fp = json.load(fh)
    side = None
    if with_sidecar:
        sp = os.path.join(os.path.dirname(os.path.abspath(path)), fp.get("sidecar") or "")
        if fp.get("sidecar") and os.path.isfile(sp):
            with np.load(sp, allow_pickle=False) as z:
                side = {k: z[k] for k in z.files}
    return fp, side
