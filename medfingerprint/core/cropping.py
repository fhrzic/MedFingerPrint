"""nnU-Net's crop-to-nonzero, ported without the acvl_utils dependency.

Behaviour is identical to ``nnunetv2.preprocessing.cropping.cropping.crop_to_nonzero``:
the nonzero mask (hole-filled) defines a half-open bounding box; segmentation
voxels that are 0 and outside the nonzero mask are set to ``nonzero_label``.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_fill_holes


def create_nonzero_mask(data: np.ndarray) -> np.ndarray:
    assert data.ndim in (3, 4), "data must have shape (C, X, Y, Z) or (C, X, Y)"
    mask = data[0] != 0
    for c in range(1, data.shape[0]):
        mask |= data[c] != 0
    return binary_fill_holes(mask)


def get_bbox_from_mask(mask: np.ndarray) -> list[list[int]]:
    """Half-open bbox per axis. An empty mask yields the full extent (as nnU-Net)."""
    bbox = []
    for axis in range(mask.ndim):
        other = tuple(a for a in range(mask.ndim) if a != axis)
        proj = np.any(mask, axis=other)
        nz = np.flatnonzero(proj)
        if nz.size:
            bbox.append([int(nz[0]), int(nz[-1]) + 1])
        else:
            bbox.append([0, int(mask.shape[axis])])
    return bbox


def bounding_box_to_slice(bbox) -> tuple:
    return tuple(slice(*b) for b in bbox)


def crop_to_nonzero(data: np.ndarray, seg: np.ndarray | None = None, nonzero_label: int = -1):
    nonzero_mask = create_nonzero_mask(data)
    bbox = get_bbox_from_mask(nonzero_mask)
    slicer = bounding_box_to_slice(bbox)
    nonzero_mask = nonzero_mask[slicer][None]
    slicer = (slice(None),) + slicer
    data = data[slicer]
    if seg is not None:
        seg = seg[slicer]
        seg[(seg == 0) & (~nonzero_mask)] = nonzero_label
    else:
        seg = np.where(nonzero_mask, np.int8(0), np.int8(nonzero_label))
    return data, seg, bbox
