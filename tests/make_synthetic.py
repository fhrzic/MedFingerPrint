"""Generate small synthetic datasets for testing.

    python tests/make_synthetic.py OUT_DIR

Creates
  OUT_DIR/nnUNet_raw/Dataset990_MfpTest   nnU-Net raw layout (CT phantoms + label maps)
  OUT_DIR/ts                               TotalSegmentator layout (binary masks + meta.csv)
  OUT_DIR/xray8                            8-bit PNG/JPG images
  OUT_DIR/xray16                           16-bit PNG + a MONOCHROME1 DICOM
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import SimpleITK as sitk

LABELS = {1: "liver", 2: "spleen", 3: "vertebra"}


def ct_phantom(shape=(48, 112, 112), seed=0, zero_border=0, missing_intercept=False,
               circular_fov=False):
    rs = np.random.RandomState(seed)
    Z, Y, X = shape
    z, y, x = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing="ij")
    cy, cx = Y * 0.46, X / 2
    ry, rx = Y * 0.33 * (1 + 0.05 * rs.randn()), X * 0.40 * (1 + 0.05 * rs.randn())
    r = ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2
    vol = -1000 + 8 * rs.randn(*shape)
    lab = np.zeros(shape, np.uint8)
    body = r <= 1
    vol[body] = -100 + 20 * rs.randn(body.sum())                       # fat
    inner = r <= 0.80
    vol[inner] = 45 + 15 * rs.randn(inner.sum())                       # muscle / soft tissue
    upper = z >= Z // 2
    for sx in (-1, 1):                                                  # lungs
        lr = ((y - cy + 0.05 * Y) / (0.2 * Y)) ** 2 + ((x - cx - sx * 0.2 * X) / (0.14 * X)) ** 2
        m = (lr <= 1) & upper
        vol[m] = -850 + 40 * rs.randn(m.sum())
    lower = ~upper
    liv = (((y - cy + 0.02 * Y) / (0.17 * Y)) ** 2 + ((x - cx + 0.17 * X) / (0.14 * X)) ** 2 <= 1) & lower
    vol[liv] = 60 + 12 * rs.randn(liv.sum()); lab[liv] = 1
    spl = (((y - cy + 0.06 * Y) / (0.08 * Y)) ** 2 + ((x - cx - 0.22 * X) / (0.07 * X)) ** 2 <= 1) & lower
    vol[spl] = 50 + 10 * rs.randn(spl.sum()); lab[spl] = 2
    sp = ((y - cy - 0.22 * Y) / (0.07 * Y)) ** 2 + ((x - cx) / (0.07 * Y)) ** 2
    vert = sp <= 1
    vol[vert] = 350 + 80 * rs.randn(vert.sum())
    cort = (sp <= 1) & (sp > 0.55)
    vol[cort] = 1100 + 150 * rs.randn(cort.sum())
    lab[vert] = 3
    table = (y >= int(cy + ry) + 4) & (y <= int(cy + ry) + 6) & (np.abs(x - cx) < 0.45 * X)
    vol[table] = 150 + 20 * rs.randn(table.sum())                       # CT table (not body)
    if circular_fov:
        fov = ((y - Y / 2) ** 2 + (x - X / 2) ** 2) > (0.49 * min(X, Y)) ** 2
        vol[fov] = -2048
    vol = np.round(vol)
    if missing_intercept:
        vol = np.clip(vol + 1024, 0, None)   # raw stored values, intercept never applied
    if zero_border:
        b = zero_border
        out = np.zeros((Z, Y + 2 * b, X + 2 * b))
        out[:, b:-b, b:-b] = vol
        lo = np.zeros_like(out, dtype=np.uint8)
        lo[:, b:-b, b:-b] = lab
        vol, lab = out, lo
    return vol.astype(np.int16), lab


def _write(arr, spacing_xyz, path, dtype=None):
    img = sitk.GetImageFromArray(arr if dtype is None else arr.astype(dtype))
    img.SetSpacing(tuple(float(s) for s in spacing_xyz))
    sitk.WriteImage(img, path, useCompression=True)


CASES = [
    dict(seed=1, spacing=(0.8, 0.8, 2.5)),
    dict(seed=2, spacing=(0.75, 0.75, 2.5), shape=(52, 112, 112)),
    dict(seed=3, spacing=(0.8, 0.8, 2.5), circular_fov=True),
    dict(seed=4, spacing=(0.9, 0.9, 2.5), zero_border=6),
    dict(seed=5, spacing=(0.8, 0.8, 2.5), shape=(44, 120, 120)),
    dict(seed=6, spacing=(0.8, 0.8, 2.0)),
    dict(seed=7, spacing=(0.8, 0.8, 2.5), missing_intercept=True),   # outlier
    dict(seed=8, spacing=(0.8, 0.8, 5.0), shape=(24, 112, 112)),       # spacing outlier
    dict(seed=9, spacing=(0.85, 0.85, 2.5)),
    dict(seed=10, spacing=(0.8, 0.8, 2.5)),
]


def make_nnunet(out):
    d = os.path.join(out, "nnUNet_raw", "Dataset990_MfpTest")
    os.makedirs(os.path.join(d, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(d, "labelsTr"), exist_ok=True)
    for i, c in enumerate(CASES):
        c = dict(c)
        sp = c.pop("spacing")
        vol, lab = ct_phantom(**c)
        _write(vol, sp, os.path.join(d, "imagesTr", f"case_{i:03d}_0000.nii.gz"))
        _write(lab, sp, os.path.join(d, "labelsTr", f"case_{i:03d}.nii.gz"))
    with open(os.path.join(d, "dataset.json"), "w") as fh:
        json.dump({"channel_names": {"0": "CT"},
                   "labels": {"background": 0, **{v: k for k, v in LABELS.items()}},
                   "numTraining": len(CASES), "file_ending": ".nii.gz"}, fh, indent=1)
    return d


def make_ts(out):
    d = os.path.join(out, "ts")
    rows = []
    for i, c in enumerate(CASES[:6]):
        c = dict(c)
        sp = c.pop("spacing")
        vol, lab = ct_phantom(**c)
        cid = f"s{i:04d}"
        os.makedirs(os.path.join(d, cid, "segmentations"), exist_ok=True)
        _write(vol, sp, os.path.join(d, cid, "ct.nii.gz"))
        for v, name in LABELS.items():
            _write((lab == v).astype(np.uint8), sp, os.path.join(d, cid, "segmentations", f"{name}.nii.gz"))
        rows.append(f"{cid};{50 + i};{'mf'[i % 2]};{'AB'[i % 2]};ct abdomen;train;Siemens")
    with open(os.path.join(d, "meta.csv"), "w") as fh:
        fh.write("image_id;age;gender;institute;study_type;split;manufacturer\n" + "\n".join(rows) + "\n")
    return d


def make_xray(out):
    from PIL import Image
    d8 = os.path.join(out, "xray8")
    d16 = os.path.join(out, "xray16")
    os.makedirs(d8, exist_ok=True)
    os.makedirs(d16, exist_ok=True)
    rs = np.random.RandomState(0)
    yy, xx = np.mgrid[0:200, 0:160]
    for i in range(8):
        base = 60 + 90 * np.exp(-(((yy - 100) / 70) ** 2 + ((xx - 80) / 50) ** 2)) + 10 * rs.randn(200, 160)
        if i == 5:
            base = base * 0.4 + 140   # washed-out outlier
        img = np.clip(base, 0, 255).astype(np.uint8)
        if i == 6:
            img[:20] = 255            # saturated band
        Image.fromarray(img).save(os.path.join(d8, f"img_{i:02d}.png"))
    rgb = np.repeat(img[..., None], 3, axis=2)
    Image.fromarray(rgb).save(os.path.join(d8, "img_rgbgray.jpg"), quality=100)
    for i in range(5):
        base = 800 + 2500 * np.exp(-(((yy - 100) / 70) ** 2 + ((xx - 80) / 50) ** 2)) + 50 * rs.randn(200, 160)
        Image.fromarray(np.clip(base, 0, 4095).astype(np.uint16)).save(os.path.join(d16, f"img16_{i:02d}.png"))
    _write_mono1_dicom(np.clip(base, 0, 4095).astype(np.uint16), os.path.join(d16, "mono1.dcm"))
    return d8, d16


def _write_mono1_dicom(arr, path):
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1.1"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(path, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "DX"
    ds.PatientName = "Synthetic^Test"
    ds.Rows, ds.Columns = arr.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME1"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
    ds.ImagerPixelSpacing = [0.143, 0.143]
    ds.PixelData = (4095 - arr).astype(np.uint16).tobytes()   # stored inverted
    ds.save_as(path, enforce_file_format=True)


if __name__ == "__main__":
    out = sys.argv[1]
    print(make_nnunet(out))
    print(make_ts(out))
    print(*make_xray(out))
