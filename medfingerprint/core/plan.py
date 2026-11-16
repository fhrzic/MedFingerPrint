"""Predict the nnU-Net v2 experiment plan (default ``nnUNetPlans`` / ExperimentPlanner)
from a fingerprint, without nnU-Net or PyTorch installed.

Follows nnU-Net v2's ExperimentPlanner:

* fullres target spacing = median spacing; if the data are anisotropic (worst axis
  spacing > 3x the others AND its voxel count < 1/3 of the others) the worst axis uses
  the 10th percentile of its spacings instead (``determine_fullres_target_spacing``);
* median shape after resampling to the target spacing;
* initial patch size with the aspect of 1/spacing and 256^3 (3D) / 2048^2 (2D) voxels,
  clipped to the median shape;
* pooling per axis until feature maps reach 4 voxels, only pooling axes whose spacing is
  within 2x of the finest (``get_pool_and_conv_props``); patch divisible by 2^pools;
* the patch is shrunk along the axis that is largest relative to the median shape until
  the estimated GPU memory fits the 8 GB reference; batch size from the remaining
  memory budget, capped so one batch covers at most 5 % of the dataset, minimum 2;
* 3d_lowres only when the median image is > 4x the fullres patch;
* normalisation: CT -> clip to foreground [p0.5, p99.5] and z-score with the foreground
  mean/std of the fingerprint; other modalities -> per-image z-score.

The memory estimate counts feature-map elements of a PlainConvUNet (32 features doubling
to 320 (3D) / 512 (2D), 2 convs per stage) and is calibrated to nnU-Net's reference
configurations (128^3 -> batch 2, 512^2 -> batch 12). Checked against nnU-Net v2's plans for
MSD Hippocampus, Heart and Liver; results are labelled as approximate.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Optional

import numpy as np

ANISO_THRESHOLD = 3
FEATUREMAP_MIN_EDGE = 4
REF_VAL_3D = 560_000_000
REF_VAL_2D = 85_000_000
REF_BS_3D, REF_BS_2D = 2, 12
MIN_BS = 2
MAX_DATASET_COVERED = 0.05
CALIB_3D = 1.0        # feature-map count of our simplified network -> nnU-Net's estimate


def fullres_target_spacing(spacings: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    target = np.percentile(spacings, 50, axis=0)
    target_size = np.percentile(sizes, 50, axis=0)
    worst = int(np.argmax(target))
    other = [i for i in range(len(target)) if i != worst]
    other_sp = [target[i] for i in other]
    other_sz = [target_size[i] for i in other]
    aniso_spacing = target[worst] > ANISO_THRESHOLD * max(other_sp)
    aniso_voxels = target_size[worst] * ANISO_THRESHOLD < min(other_sz)
    if aniso_spacing and aniso_voxels:
        t = np.percentile(spacings[:, worst], 10)
        if t < max(other_sp):
            t = max(max(other_sp), t) + 1e-5
        target[worst] = t
    return target


def pool_and_conv_props(spacing, patch_size, min_edge=FEATUREMAP_MIN_EDGE, max_numpool=999999):
    dim = len(spacing)
    cur_sp = list(spacing)
    cur_sz = list(patch_size)
    pool_ops = [[1] * dim]
    kernel = [1] * dim
    conv_kernels = []
    n_pool = [0] * dim
    while True:
        valid = [i for i in range(dim) if cur_sz[i] >= 2 * min_edge]
        if not valid:
            break
        mn = min(cur_sp[i] for i in valid)
        valid = [i for i in valid if cur_sp[i] / mn < 2 and n_pool[i] < max_numpool]
        if len(valid) == 1 and cur_sz[valid[0]] < 3 * min_edge:
            break
        if not valid:
            break
        for d in range(dim):
            if kernel[d] != 3 and cur_sp[d] / min(cur_sp) < 2:
                kernel[d] = 3
        p = [1] * dim
        for v in valid:
            p[v] = 2
            n_pool[v] += 1
            cur_sp[v] *= 2
            cur_sz[v] = np.ceil(cur_sz[v] / 2)
        pool_ops.append(p)
        conv_kernels.append(list(kernel))
    div = [2 ** n for n in n_pool]
    patch = [int(np.ceil(s / d) * d) for s, d in zip(patch_size, div)]
    conv_kernels.append([3] * dim)
    return n_pool, pool_ops, conv_kernels, patch, div


def _features(n_stages: int, dim: int) -> list:
    mx = 320 if dim == 3 else 512
    return [min(mx, 32 * 2 ** i) for i in range(n_stages)]


def estimate_memory(patch, pool_ops, n_classes: int = 2) -> float:
    """Feature-map elements of a PlainConvUNet for one sample (rough nnU-Net equivalent)."""
    dim = len(patch)
    feats = _features(len(pool_ops), dim)
    size = np.array(patch, dtype=float)
    vols = []
    for p in pool_ops:
        size = np.ceil(size / np.array(p))
        vols.append(float(np.prod(size)))
    enc = sum(2 * f * v for f, v in zip(feats, vols))
    dec = sum(3 * f * v + n_classes * v for f, v in zip(feats[:-1], vols[:-1]))
    return (enc + dec) * (CALIB_3D if dim == 3 else 1.0)


def plan_configuration(spacing, median_shape, n_cases: int, n_classes: int = 2,
                       n_vox_dataset: Optional[float] = None) -> dict:
    """``n_vox_dataset``: voxels of the whole dataset (nnU-Net uses the 3D median shape x cases,
    also for the 2d configuration)."""
    spacing = np.asarray(spacing, dtype=float)
    median_shape = np.asarray(median_shape, dtype=float)
    dim = len(spacing)
    tmp = 1 / spacing
    total = 256 ** 3 if dim == 3 else 2048 ** 2
    init = np.round(tmp * (total / np.prod(tmp)) ** (1 / dim)).astype(int)
    init = np.minimum(init, median_shape[:dim]).astype(int)
    n_pool, pool_ops, conv, patch, div = pool_and_conv_props(spacing, init)
    ref = REF_VAL_3D if dim == 3 else REF_VAL_2D
    ref_bs = REF_BS_3D if dim == 3 else REF_BS_2D
    est = estimate_memory(patch, pool_ops, n_classes)
    guard = 0
    while est > ref and guard < 1000:
        guard += 1
        ax = int(np.argsort([p / m for p, m in zip(patch, median_shape[:dim])])[-1])
        patch = list(patch)
        patch[ax] -= div[ax]
        n_pool, pool_ops, conv, patch, div = pool_and_conv_props(spacing, patch)
        est = estimate_memory(patch, pool_ops, n_classes)
    bs = round(ref / est * ref_bs)
    if n_vox_dataset is None:
        n_vox_dataset = float(np.prod(median_shape, dtype=np.float64) * n_cases)
    bs_5 = round(n_vox_dataset * MAX_DATASET_COVERED / np.prod(patch, dtype=np.float64))
    bs = max(min(bs, bs_5), MIN_BS)
    return {
        "spacing": [float(v) for v in spacing],
        "median_shape": [int(round(v)) for v in median_shape[:dim]],
        "patch_size": [int(v) for v in patch],
        "batch_size": int(bs),
        "pool_ops": [int(v) for v in n_pool],
        "n_stages": len(pool_ops),
        "features_per_stage": _features(len(pool_ops), dim),
        "estimated_feature_maps": float(est),
    }


def predict_plan(fp: dict) -> Optional[dict]:
    key = "nnunet_fingerprint" if fp.get("nnunet_fingerprint") else (
        "nnunet_fingerprint_body" if fp.get("nnunet_fingerprint_body") else None)
    if key is None:
        return None
    nn = fp[key]
    sp = np.asarray(nn.get("spacings") or [], dtype=float)
    sh = np.asarray(nn.get("shapes_after_crop") or [], dtype=float)
    if sp.ndim != 2 or len(sp) == 0 or sp.shape != sh.shape:
        return None
    n_cases = len(sp)
    n_classes = max(2, len(fp.get("labels") or {}) + 1)
    is_2d_data = sp.shape[1] == 3 and np.median(sh[:, 0]) == 1
    target = fullres_target_spacing(sp, sh)
    new_shapes = np.array([np.round(s * c / target) for s, c in zip(sp, sh)])
    median_shape = np.median(new_shapes, axis=0)
    configs = {}
    remarks = []
    if not is_2d_data:
        full = plan_configuration(target, median_shape, n_cases, n_classes)
        configs["3d_fullres"] = full
        # lowres
        if np.prod(median_shape) / np.prod(full["patch_size"]) >= 4:
            lowres = deepcopy(target)
            inc = 1.03
            cfg = full
            for _ in range(500):
                mx = max(lowres)
                if np.any(mx / lowres > 2):
                    lowres[(mx / lowres) > 2] *= inc
                else:
                    lowres *= inc
                shape_lr = median_shape * target / lowres
                cfg = plan_configuration(lowres, shape_lr, n_cases, n_classes)
                if np.prod(cfg["patch_size"]) / np.prod(shape_lr) >= 0.25:
                    break
            configs["3d_lowres"] = cfg
            remarks.append("3d_lowres + 3d_cascade_fullres are planned because the median image is more "
                           "than 4x the fullres patch.")
    two = plan_configuration(target[1:], median_shape[1:], n_cases, n_classes,
                             n_vox_dataset=float(np.prod(median_shape, dtype=np.float64) * n_cases))
    configs["2d"] = two
    is_ct = (fp.get("config") or {}).get("modality") == "CT"
    props = (nn.get("foreground_intensity_properties_per_channel") or {}).get("0", {})
    if is_ct:
        norm = "CTNormalization"
        params = {"clip_low (p0.5)": props.get("percentile_00_5"), "clip_high (p99.5)": props.get("percentile_99_5"),
                  "mean": props.get("mean"), "std": props.get("std")}
    else:
        norm = "ZScoreNormalization (per image)"
        params = {}
    if is_2d_data:
        remarks.append("2D images: nnU-Net only plans the 2d configuration (spacing 1×1 for PNG/JPG).")
    if key == "nnunet_fingerprint_body":
        remarks.append("No masks: the foreground used here is the intensity body region, so the CT "
                       "normalisation values are an approximation of what nnU-Net would compute.")
    if not is_2d_data and target[0] != np.median(sp[:, 0]):
        remarks.append(f"Anisotropic data: the target z spacing is the 10th percentile "
                       f"({target[0]:.3g} mm) instead of the median.")
    return {
        "planner": "ExperimentPlanner (nnUNetPlans, 8 GB), predicted",
        "note": ("Predicted with nnU-Net v2's planning rules (target spacing, patch size, pooling, batch "
                 "size); the GPU-memory model is an approximation, so patch and batch sizes may differ "
                 "slightly from nnUNetv2_plan_and_preprocess."),
        "foreground_source": "labels > 0" if key == "nnunet_fingerprint" else "intensity body region",
        "n_training_cases": n_cases,
        "fullres_target_spacing": [float(v) for v in target],
        "configurations": configs,
        "normalization": norm,
        "normalization_params": params,
        "remarks": remarks,
    }
