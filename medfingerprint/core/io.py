"""Image and mask readers.

All readers return a :class:`LoadedImage` whose ``array`` is float32 with the
channel axis first: ``(C, Z, Y, X)`` for volumes and ``(C, Y, X)`` for 2D
images. Spacing is given in the same (array) axis order, which for SimpleITK
images means the reverse of ``GetSpacing()`` — the nnU-Net convention.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import SimpleITK as sitk

RASTER_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
SITK_EXTS = (".nii", ".nii.gz", ".mha", ".mhd", ".nrrd", ".nhdr")
DICOM_EXTS = (".dcm", ".dicom")

# Technical DICOM tags that are safe to publish (no patient-identifying data).
SAFE_DICOM_TAGS = {
    "0008|0060": "Modality",
    "0008|0070": "Manufacturer",
    "0008|1090": "ManufacturerModelName",
    "0018|0060": "KVP",
    "0018|1210": "ConvolutionKernel",
    "0018|0050": "SliceThickness",
    "0018|0015": "BodyPartExamined",
    "0028|0004": "PhotometricInterpretation",
    "0028|0101": "BitsStored",
    "0028|1052": "RescaleIntercept",
    "0028|1053": "RescaleSlope",
}
CONTRAST_TAG = "0018|0010"  # ContrastBolusAgent: only its presence is recorded


def split_ext(path: str) -> tuple[str, str]:
    """Split a filename into (stem, extension), treating .nii.gz as one extension."""
    name = os.path.basename(path)
    low = name.lower()
    for ext in (".nii.gz", ".nhdr.gz"):
        if low.endswith(ext):
            return name[: -len(ext)], name[-len(ext):]
    stem, ext = os.path.splitext(name)
    return stem, ext


@dataclass
class LoadedImage:
    array: np.ndarray                  # float32, (C, *spatial)
    spacing: tuple                     # mm, array axis order
    original_dtype: str
    orientation: Optional[str] = None  # e.g. "LPS" (3D only)
    photometric: Optional[str] = None
    meta: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    # SimpleITK physical geometry {"size", "spacing", "origin", "direction"} (x, y, z order),
    # used to place masks stored on another grid (e.g. SAROS 5 mm masks). None for 2D files.
    geom: Optional[dict] = None

    @property
    def spatial_shape(self) -> tuple:
        return tuple(int(s) for s in self.array.shape[1:])

    @property
    def ndim(self) -> int:
        return self.array.ndim - 1


def _sitk_orientation(img: sitk.Image) -> Optional[str]:
    if img.GetDimension() != 3:
        return None
    try:
        return sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(img.GetDirection())
    except Exception:
        return None


def _from_sitk(img: sitk.Image) -> LoadedImage:
    arr = sitk.GetArrayFromImage(img)
    original_dtype = str(arr.dtype)
    ncomp = img.GetNumberOfComponentsPerPixel()
    if ncomp > 1:
        arr = np.moveaxis(arr, -1, 0)
    else:
        arr = arr[None]
    spacing = tuple(abs(float(s)) for s in img.GetSpacing()[::-1])
    return LoadedImage(
        array=arr.astype(np.float32, copy=False),
        spacing=spacing,
        original_dtype=original_dtype,
        orientation=_sitk_orientation(img),
        geom={"size": list(img.GetSize()), "spacing": list(img.GetSpacing()),
              "origin": list(img.GetOrigin()), "direction": list(img.GetDirection())},
    )


def sitk_read(path: str, notes: Optional[list] = None) -> sitk.Image:
    """sitk.ReadImage that tolerates slightly non-orthonormal NIfTI sforms.

    Some datasets (e.g. TotalSegmentator v2+ oblique scans) store direction cosines that
    are orthonormal only to ~1e-4, which ITK rejects. nibabel/nnU-Net read them fine; we
    retry with ITK's permissive sform mode (voxel data is identical) and add a note.
    """
    try:
        return sitk.ReadImage(path)
    except RuntimeError as e:
        if "orthonormal" not in str(e):
            raise
    old = os.environ.get("ITK_NIFTI_SFORM_PERMISSIVE")
    os.environ["ITK_NIFTI_SFORM_PERMISSIVE"] = "1"
    warn = sitk.ProcessObject.GetGlobalWarningDisplay()
    sitk.ProcessObject.SetGlobalWarningDisplay(False)
    try:
        img = sitk.ReadImage(path)
    finally:
        sitk.ProcessObject.SetGlobalWarningDisplay(warn)
        if old is None:
            os.environ.pop("ITK_NIFTI_SFORM_PERMISSIVE", None)
        else:
            os.environ["ITK_NIFTI_SFORM_PERMISSIVE"] = old
    if notes is not None:
        notes.append("non-orthonormal sform (read permissively)")
    return img


def read_sitk_file(path: str) -> LoadedImage:
    notes: list = []
    li = _from_sitk(sitk_read(path, notes))
    li.notes.extend(notes)
    return li


def list_dicom_series(directory: str) -> list[str]:
    reader = sitk.ImageSeriesReader()
    return list(reader.GetGDCMSeriesIDs(directory) or [])


def read_dicom_series(directory: str, series_uid: Optional[str] = None) -> LoadedImage:
    reader = sitk.ImageSeriesReader()
    if series_uid is None:
        ids = list_dicom_series(directory)
        if not ids:
            raise FileNotFoundError(f"No DICOM series in {directory}")
        series_uid = ids[0]
    files = reader.GetGDCMSeriesFileNames(directory, series_uid)
    reader.SetFileNames(files)
    reader.MetaDataDictionaryArrayUpdateOn()
    img = reader.Execute()  # applies RescaleSlope/Intercept -> HU for CT
    out = _from_sitk(img)
    meta = {}
    for tag, name in SAFE_DICOM_TAGS.items():
        if reader.HasMetaDataKey(0, tag):
            meta[name] = reader.GetMetaData(0, tag).strip()
    meta["ContrastAgentPresent"] = bool(
        reader.HasMetaDataKey(0, CONTRAST_TAG) and reader.GetMetaData(0, CONTRAST_TAG).strip()
    )
    meta["NumberOfFiles"] = len(files)
    out.meta = meta
    out.photometric = meta.get("PhotometricInterpretation")
    return out


def read_dicom_2d(path: str) -> LoadedImage:
    """Read a single-file DICOM (X-ray, CR/DX, single CT slice, multi-frame)."""
    import pydicom
    from pydicom.pixels import apply_modality_lut

    ds = pydicom.dcmread(path)
    raw = ds.pixel_array
    original_dtype = str(raw.dtype)
    notes = []
    arr = apply_modality_lut(raw, ds).astype(np.float32)
    photometric = str(getattr(ds, "PhotometricInterpretation", "")) or None
    bits = int(getattr(ds, "BitsStored", 0) or 0)
    if photometric == "MONOCHROME1":
        # Inverted grayscale: high value = dark. Flip to MONOCHROME2 convention.
        has_rescale = float(getattr(ds, "RescaleSlope", 1) or 1) != 1 or float(
            getattr(ds, "RescaleIntercept", 0) or 0) != 0
        if bits and not has_rescale:
            arr = (2 ** bits - 1) - arr
        else:
            arr = (arr.max() + arr.min()) - arr
        notes.append("MONOCHROME1 inverted")
    spp = int(getattr(ds, "SamplesPerPixel", 1) or 1)
    if spp > 1:
        arr = np.moveaxis(arr, -1, 0)
    else:
        arr = arr[None]
    ps = getattr(ds, "PixelSpacing", None) or getattr(ds, "ImagerPixelSpacing", None)
    spacing = tuple(float(v) for v in ps) if ps else (1.0, 1.0)
    if arr.ndim == 4:  # multi-frame: (C, frames, y, x)
        st = float(getattr(ds, "SliceThickness", 1.0) or 1.0)
        spacing = (st,) + spacing
    meta = {}
    for name in SAFE_DICOM_TAGS.values():
        if name in ds:
            meta[name] = str(ds.data_element(name).value).strip()
    meta["ContrastAgentPresent"] = bool(str(getattr(ds, "ContrastBolusAgent", "") or "").strip())
    return LoadedImage(arr, spacing, original_dtype, photometric=photometric, meta=meta, notes=notes)


def read_raster(path: str) -> LoadedImage:
    """Read PNG/JPEG/BMP/TIFF. Alpha channels are dropped; palette images expanded."""
    ext = split_ext(path)[1].lower()
    notes = []
    if ext in (".tif", ".tiff"):
        import tifffile
        arr = np.asarray(tifffile.imread(path))
        if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
            arr = np.moveaxis(arr, 0, -1)  # planar RGB stored channel-first
    else:
        from PIL import Image
        with Image.open(path) as im:
            if im.mode == "P":
                im = im.convert("RGBA" if "transparency" in im.info else "RGB")
            if im.mode in ("LA",):
                im = im.convert("L")
            arr = np.asarray(im)
    original_dtype = str(arr.dtype)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
            notes.append("alpha dropped")
        arr = np.moveaxis(arr, -1, 0)
        if np.array_equal(arr[0], arr[1]) and np.array_equal(arr[0], arr[2]):
            arr = arr[:1]
            notes.append("RGB with identical channels -> grayscale")
    elif arr.ndim == 2:
        arr = arr[None]
    else:
        raise ValueError(f"Unsupported raster shape {arr.shape} in {path}")
    return LoadedImage(arr.astype(np.float32), (1.0, 1.0), original_dtype, notes=notes)


def load_image(kind: str, path: str, series_uid: Optional[str] = None) -> LoadedImage:
    if kind == "sitk":
        return read_sitk_file(path)
    if kind == "dicom_series":
        return read_dicom_series(path, series_uid)
    if kind == "dicom2d":
        return read_dicom_2d(path)
    if kind == "raster":
        return read_raster(path)
    raise ValueError(f"Unknown image kind {kind!r}")


def image_kind_for(path: str, modality: str) -> str:
    ext = split_ext(path)[1].lower()
    if os.path.isdir(path):
        return "dicom_series"
    if ext in SITK_EXTS:
        return "sitk"
    if ext in DICOM_EXTS:
        return "dicom2d"
    if ext in RASTER_EXTS:
        return "raster"
    raise ValueError(f"Unsupported file type: {path}")


RGB_LABEL_OFFSET = 1 << 24  # colour-coded masks: label value = offset + 0xRRGGBB


def read_label_array(path: str) -> np.ndarray:
    """Read a label map / binary mask as an integer array in array axis order."""
    ext = split_ext(path)[1].lower()
    if ext in RASTER_EXTS:
        arr = read_raster(path).array
        if arr.shape[0] > 1:
            arr = arr[:3] if arr.shape[0] >= 3 else arr[:1]       # drop alpha
            if all(np.array_equal(arr[0], arr[c]) for c in range(1, arr.shape[0])):
                arr = arr[:1]                                        # grey stored as RGB
            else:                                                    # colour-coded label map
                rgb = arr.astype(np.int64)
                code = (rgb[0] << 16) | (rgb[1] << 8) | rgb[2]   # 0xRRGGBB, black = background
                if np.unique(code).size > 256:
                    raise ValueError(f"Mask {path} has many colours; expected a label map "
                                     "(JPEG masks are lossy: use PNG)")
                return (code + RGB_LABEL_OFFSET * (code > 0)).astype(np.int64)
        return arr[0].astype(np.int32)
    img = sitk_read(path)
    arr = sitk.GetArrayFromImage(img)
    if img.GetNumberOfComponentsPerPixel() > 1:
        raise ValueError(f"Mask {path} is multi-component")
    return arr.astype(np.int32, copy=False)
