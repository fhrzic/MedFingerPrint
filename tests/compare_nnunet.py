"""Compare the nnU-Net block of a MedFingerPrint JSON with nnU-Net's dataset_fingerprint.json.

    python tests/compare_nnunet.py ours.json dataset_fingerprint.json
Exit code 0 = identical. mean/std are compared at float32 precision (rtol 1e-5): nnU-Net
computes them with float32 accumulation, whose rounding differs between NumPy versions
(verified: identical samples give 315.56064 on NumPy 2.2.3 and 315.56046 on 2.5.4).
Everything else must match to 1e-9.
"""
import json
import sys

import numpy as np


def compare(ours: dict, ref: dict, rtol=1e-9) -> list:
    nn = ours["nnunet_fingerprint"]
    problems = []
    for key in ("spacings", "shapes_after_crop"):
        a, b = np.array(nn[key], float), np.array(ref[key], float)
        if a.shape != b.shape or not np.allclose(a, b, rtol=rtol, atol=0):
            problems.append(f"{key} differs")
    a, b = nn["median_relative_size_after_cropping"], ref["median_relative_size_after_cropping"]
    if not np.isclose(a, b, rtol=rtol, atol=0):
        problems.append(f"median_relative_size_after_cropping {a} vs {b}")
    pa = nn["foreground_intensity_properties_per_channel"]
    pb = ref["foreground_intensity_properties_per_channel"]
    for ch in pb:
        for k, v in pb[ch].items():
            tol = 1e-5 if k in ("mean", "std") else rtol
            if not np.isclose(pa[ch][k], v, rtol=tol, atol=0):
                problems.append(f"channel {ch} {k}: ours {pa[ch][k]!r} vs nnU-Net {v!r}")
    return problems


if __name__ == "__main__":
    ours = json.load(open(sys.argv[1]))
    ref = json.load(open(sys.argv[2]))
    p = compare(ours, ref)
    print("IDENTICAL to nnU-Net fingerprint" if not p else "\n".join(p))
    sys.exit(1 if p else 0)
