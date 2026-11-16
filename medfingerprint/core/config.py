"""Scan configuration and histogram specification."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

MODALITIES = ("CT", "2D", "3D")


@dataclass
class HistogramSpec:
    """Fixed-width histogram over [lo, hi) plus one underflow and one overflow bin.

    Layout of a counts array: index 0 = values < lo, 1..nbins = regular bins,
    nbins+1 = values >= hi. Every case in a dataset uses the same spec so
    histograms can be summed exactly and re-binned afterwards.
    """

    lo: float
    hi: float
    width: float

    @property
    def nbins(self) -> int:
        return int(round((self.hi - self.lo) / self.width))

    @property
    def size(self) -> int:
        return self.nbins + 2

    def edges(self) -> np.ndarray:
        return self.lo + self.width * np.arange(self.nbins + 1)

    def index(self, x: np.ndarray) -> np.ndarray:
        """Map values to counts-array indices (int32). NaNs must be removed first."""
        idx = np.floor((x - self.lo) / self.width)
        np.clip(idx, -1, self.nbins, out=idx)
        return idx.astype(np.int32) + 1

    def to_dict(self) -> dict:
        return {"lo": float(self.lo), "hi": float(self.hi), "width": float(self.width)}

    @classmethod
    def from_dict(cls, d: dict) -> "HistogramSpec":
        return cls(float(d["lo"]), float(d["hi"]), float(d["width"]))


# 1-HU bins over the full 12-bit CT range; exact for integer-valued CT.
CT_HISTOGRAM = HistogramSpec(-1024.0, 3072.0, 1.0)


@dataclass
class ScanConfig:
    modality: str = "CT"                      # "CT" | "2D" | "3D"
    histogram: Optional[HistogramSpec] = None  # None -> chosen automatically
    gray_mode: str = "luminance"               # how to reduce RGB to one analysis channel
    # Body / anatomy region from intensities (works without masks):
    #   "auto" -> "hu" for CT, "otsu" otherwise;  "hu" -> threshold body_threshold_hu;
    #   "otsu" -> per-image Otsu threshold;  "none";  or a number as string ("120").
    body_method: str = "auto"
    body_threshold_hu: float = -500.0
    nnunet_body_block: bool = True             # nnU-Net-style stats with body as foreground
    compute_nnunet: bool = True                # only takes effect when masks exist
    nnunet_total_fg_samples: float = 10e7      # same constant as nnU-Net (= 1e8)
    nnunet_seed: int = 1234
    label_names: Optional[dict] = None         # {int value: name} for label maps
    nnunet_samples_per_case: int = 10000       # filled in by the runner
    chunk_voxels: int = 16_000_000             # memory bound for streaming passes
    compute_quality: bool = True               # quality measures + duplicate hashes per case
    analysis_version: int = 3                  # bump when per-case results change (invalidates caches)

    def __post_init__(self):
        if self.modality not in MODALITIES:
            raise ValueError(f"modality must be one of {MODALITIES}, got {self.modality!r}")
        if isinstance(self.histogram, dict):
            self.histogram = HistogramSpec.from_dict(self.histogram)
        self.resolved_body_method()
        if self.label_names:
            self.label_names = {int(k): str(v) for k, v in self.label_names.items()}

    @property
    def is_ct(self) -> bool:
        return self.modality == "CT"

    def resolved_body_method(self) -> str:
        m = str(self.body_method).lower()
        if m == "auto":
            return "hu" if self.is_ct else "otsu"
        if m not in ("hu", "otsu", "none"):
            float(m)  # raises for invalid values
        return m

    def to_dict(self) -> dict:
        d = asdict(self)
        d["histogram"] = self.histogram.to_dict() if self.histogram else None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ScanConfig":
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})

    def hash(self) -> str:
        """Hash of everything that changes per-case results (used for caching)."""
        d = self.to_dict()
        return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]
