"""Split one fingerprint into per-group fingerprints (e.g. fractured vs. non-fractured,
hand vs. leg, train vs. test) without re-reading any image.

Each group is rebuilt from the per-case scan cache with exactly the same aggregation
as a normal scan, so the group fingerprints can be compared like separate datasets:

    mfp split fracatlas.json --by level:0 --out-dir groups/
    mfp compare groups/*.json
"""
from __future__ import annotations

import os
from typing import Optional

from .config import ScanConfig
from .discovery import CaseSpec, Discovery
from .outliers import case_groups
from .runner import ScanOutput

MIN_SPLIT = 3


def group_choices(fp: dict, max_values: int = 50) -> list[str]:
    """Useful ``group_by`` keys: path levels with 2..max_values values, and metadata
    columns with 2..max_values distinct values."""
    cases = [c for c in fp.get("cases") or [] if c.get("status") == "ok"]
    out = []
    depth = max((str(c["case_id"]).count("/") for c in cases), default=0)
    for n in range(depth):
        vals = {g for g in case_groups(fp, cases, f"level:{n}")}
        if 2 <= len(vals) <= max_values:
            out.append(f"level:{n}")
    for k, m in (fp.get("metadata") or {}).items():
        cnt = m.get("counts") if isinstance(m, dict) else None
        if cnt and 2 <= len(cnt) <= max_values:
            out.append(f"meta:{k}")
    return out


def describe_choice(fp: dict, group_by: str, n: int = 4) -> str:
    cases = [c for c in fp.get("cases") or [] if c.get("status") == "ok"]
    from collections import Counter
    cnt = Counter(case_groups(fp, cases, group_by))
    return ", ".join(f"{k or '(empty)'} ({v})" for k, v in cnt.most_common(n)) + \
        (f", +{len(cnt) - n}" if len(cnt) > n else "")


def split_fingerprint(fp: dict, group_by: str, min_cases: int = MIN_SPLIT,
                      progress=None) -> list[tuple[str, dict, dict]]:
    """Return ``[(group value, fingerprint, sidecar)]`` for groups with >= min_cases cases."""
    from .aggregate import build_fingerprint
    local = fp.get("local") or {}
    cache_files = local.get("cache_files") or {}
    if not cache_files:
        raise ValueError("This fingerprint has no scan cache information (anonymized, or made by an "
                         "older version): re-scan the dataset to split it.")
    root = local.get("root")
    cases = [c for c in fp["cases"] if c.get("status") == "ok" and c["case_id"] in cache_files]
    missing = [c["case_id"] for c in cases if not os.path.isfile(cache_files[c["case_id"]])]
    if missing:
        raise ValueError(f"{len(missing)} cache file(s) are missing (e.g. {missing[0]}); the scan cache "
                         "was moved or deleted. Re-scan the dataset to split it.")
    groups = case_groups(fp, cases, group_by)
    cfg = ScanConfig.from_dict(fp["config"])
    by: dict = {}
    for c, g in zip(cases, groups):
        by.setdefault(g, []).append(c)
    out = []
    items = sorted(by.items(), key=lambda kv: -len(kv[1]))
    for i, (g, cs) in enumerate(items):
        if len(cs) < min_cases:
            continue
        specs = [CaseSpec(c["case_id"], os.path.join(root, c["image"]) if root and c.get("image")
                          else (c.get("image") or c["case_id"]), "raster", meta=c.get("meta") or {})
                 for c in cs]
        disc = Discovery(specs, label_names=cfg.label_names, root=root)
        outcomes = [{"case_id": c["case_id"], "status": "ok", "cache": cache_files[c["case_id"]]}
                    for c in cs]
        scan = ScanOutput(disc, cfg, outcomes, local.get("cache_dir"))
        kind, _, key = group_by.partition(":")
        label = f"folder {key}" if kind == "level" else key
        name = f"{fp['dataset']['name']} [{label}={g or '(empty)'}]"
        gfp, side, _ = build_fingerprint(scan, name=name)
        gfp["dataset"]["layout"] = fp["dataset"].get("layout")
        gfp["dataset"]["split_from"] = {"fingerprint": fp["dataset"]["name"], "group_by": group_by,
                                        "group": g}
        out.append((g, gfp, side))
        if progress:
            progress(i + 1, len(items), g)
    return out


def safe_name(s: str) -> str:
    keep = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(s))
    return keep.strip("_") or "empty"


def save_split(parts: list, out_dir: str, base: Optional[str] = None) -> list[str]:
    from .schema import save_fingerprint
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for g, gfp, side in parts:
        p = os.path.join(out_dir, f"{base + '_' if base else ''}{safe_name(g)}.json")
        save_fingerprint(gfp, side, p)
        paths.append(p)
    return paths
