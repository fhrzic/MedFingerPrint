"""Tests for: PDF reports, metadata CSV, split, annotations (COCO/YOLO/VOC), 2D masks,
quality checks, duplicates and the predicted nnU-Net plan."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

HERE = os.path.dirname(__file__)


@pytest.fixture(scope="module")
def frac(tmp_path_factory):
    out = tmp_path_factory.mktemp("frac")
    subprocess.run([sys.executable, os.path.join(HERE, "make_fracatlas_like.py"), str(out)], check=True,
                   capture_output=True)
    # near-duplicate: image 40 re-saved with another JPEG quality, as new image 60. (These
    # synthetic images have large flat noisy areas, where a harsh re-save -- quality 70 and
    # 75 % size -- flips 18 of 256 hash bits, beyond the 14-bit limit calibrated on CheXpert.)
    im = Image.open(out / "images/Non_fractured/IMG0000040.jpg")
    im.save(out / "images/Non_fractured/IMG0000060.jpg", quality=90)
    return out


@pytest.fixture(scope="module")
def frac_fp(frac, tmp_path_factory):
    out = tmp_path_factory.mktemp("fp") / "frac.json"
    subprocess.run([sys.executable, "-m", "medfingerprint", "scan", "--root", str(frac / "images"),
                    "--ext", "jpg", "--modality", "2D", "--meta-csv", str(frac / "dataset.csv"),
                    "--annotations", str(frac / "Annotations/COCO JSON/COCO_fracture_masks.json"),
                    "--out", str(out), "--workers", "2"], check=True, capture_output=True)
    return out


def test_metadata_csv_and_annotation_checks(frac_fp):
    fp = json.load(open(frac_fp))
    assert fp["dataset"]["metadata_csv"]["id_col"] == "image_id"
    assert fp["dataset"]["metadata_csv"]["matched"] == 60
    an = fp["annotations"]
    assert an["n_images_with_objects"] == 24
    chk = {k: v["count"] for k, v in an["checks"].items()}
    assert chk["degenerate"] == 1 and chk["tiny"] >= 1 and chk["duplicate"] >= 1
    assert chk["size_mismatch"] == 1 and chk["count_fracture_count"] == 1 and chk["not_found"] == 1
    c = an["categories"]["fractured"]
    assert c["n_images"] == 24 and np.asarray(c["position_heatmap"]).sum() == c["n_objects"]
    assert c["contrast"]["median"] is not None
    assert "fractured" in fp["labels"]                         # objects -> label statistics
    assert fp["nnunet_fingerprint"]["foreground_intensity_properties_per_channel"]["0"]["mean"] > 100


def test_quality_and_duplicates(frac_fp):
    fp = json.load(open(frac_fp))
    pr = {k: v["examples"] for k, v in fp["quality"]["problems"].items()}
    assert pr["flat"] == ["Non_fractured/IMG0000050.jpg"]
    assert "Non_fractured/IMG0000051.jpg" in pr["bright_background"]      # inverted
    assert pr["colour"] == ["Non_fractured/IMG0000052.jpg"]
    du = fp["duplicates"]
    groups = [set(g) for g in du["exact_groups"]]
    assert {"Fractured/IMG0000005.jpg", "Non_fractured/IMG0000026.jpg"} in groups
    assert {"Non_fractured/IMG0000030.jpg", "Non_fractured/IMG0000053.jpg"} in groups
    assert du["n_cross_group_pairs"] >= 1                              # 5 / 26 across folders
    near = {(p["a"], p["b"]) for p in du["near_pairs"]}
    assert near == {("Non_fractured/IMG0000040.jpg", "Non_fractured/IMG0000060.jpg")}  # no false pairs


def test_yolo_and_voc_readers(frac):
    from medfingerprint.core.discovery import discover
    for sub in ("YOLO", "PASCAL VOC"):
        d = discover("files", "2D", root=str(frac / "images"), exts=["jpg"],
                     annotations=str(frac / "Annotations" / sub))
        assert d.annotations["n_cases_listed"] == 24
        assert sum(len(c.mask.objects) for c in d.cases) == d.annotations["n_objects_in_file"]


def test_split_and_reports(frac_fp, tmp_path):
    from medfingerprint.core.compare import load_for_compare
    from medfingerprint.core.split import group_choices, save_split, split_fingerprint
    from medfingerprint.report import compare_report, fingerprint_report
    fp = json.load(open(frac_fp))
    assert "level:0" in group_choices(fp) and "meta:hand" in group_choices(fp)
    parts = split_fingerprint(fp, "level:0")
    assert sorted(len(p[1]["cases"]) for p in parts) == [24, 37]
    paths = save_split(parts, str(tmp_path / "groups"))
    pdf = compare_report([load_for_compare(p) for p in paths], str(tmp_path / "cmp.pdf"))
    assert open(pdf, "rb").read(5) == b"%PDF-"
    pdf = fingerprint_report(fp, str(tmp_path / "r.pdf"))
    assert os.path.getsize(pdf) > 20_000


def test_paired_2d_masks_rgb_and_255(tmp_path):
    from medfingerprint.core.config import ScanConfig
    from medfingerprint.core.discovery import discover
    from medfingerprint.core.case_analysis import analyze_case
    (tmp_path / "img").mkdir()
    (tmp_path / "msk").mkdir()
    rs = np.random.RandomState(0)
    img = (rs.rand(64, 64) * 200).astype(np.uint8)
    Image.fromarray(img).save(tmp_path / "img/a.png")
    Image.fromarray(img).save(tmp_path / "img/b.png")
    m = np.zeros((64, 64), np.uint8)
    m[10:20, 10:30] = 255
    Image.fromarray(m).save(tmp_path / "msk/a_mask.png")                 # 0/255, "_mask" suffix
    rgb = np.zeros((64, 64, 3), np.uint8)
    rgb[5:15, 5:15] = (255, 0, 0)
    rgb[30:40, 30:50] = (0, 255, 0)
    Image.fromarray(rgb).save(tmp_path / "msk/b.png")                    # colour-coded labels
    d = discover("paired", "2D", images_dir=str(tmp_path / "img"), masks_dir=str(tmp_path / "msk"),
                 exts=["png"])
    assert all(c.mask is not None for c in d.cases)
    cfg = ScanConfig(modality="2D", nnunet_samples_per_case=100)
    from medfingerprint.core.config import HistogramSpec
    cfg.histogram = HistogramSpec(0, 256, 1)
    ra, _ = analyze_case(d.cases[0], cfg)
    rb, _ = analyze_case(d.cases[1], cfg)
    assert list(ra["labels"]) == ["255 (mask)"] and ra["labels"]["255 (mask)"]["voxels"] == 200
    assert sorted(rb["labels"]) == ["colour #00ff00", "colour #ff0000"]
    assert rb["labels"]["colour #ff0000"]["voxels"] == 100


@pytest.mark.parametrize("sp, sh, n, p3, b3, p2, b2", [
    ([1, 1, 1], [36, 50, 35], 260, [40, 56, 40], 9, [56, 40], 366),                  # MSD Hippocampus
    ([1, 0.7676, 0.7676], [432, 512, 512], 131, [128, 128, 128], 2, [512, 512], 12),  # MSD Liver
    ([1.37, 1.25, 1.25], [115, 320, 232], 20, [80, 192, 160], 2, [320, 256], None),  # MSD Heart
])
def test_plan_matches_nnunet_reference(sp, sh, n, p3, b3, p2, b2):
    from medfingerprint.core.plan import plan_configuration
    a = plan_configuration(sp, sh, n)
    b = plan_configuration(sp[1:], sh[1:], n, n_vox_dataset=float(np.prod(sh)) * n)
    assert a["patch_size"] == p3 and a["batch_size"] == b3
    assert b["patch_size"] == p2
    if b2:
        assert b["batch_size"] == b2


def test_plan_anisotropic_target_spacing():
    from medfingerprint.core.plan import fullres_target_spacing
    sp = np.array([[5.0, 0.7, 0.7]] * 8 + [[1.0, 0.7, 0.7]] * 2)
    sh = np.array([[40, 512, 512]] * 10, dtype=float)
    t = fullres_target_spacing(sp, sh)
    assert t[0] < 5.0 and t[0] == pytest.approx(1.0 + 1e-5, rel=0.5)   # 10th percentile rule


# ------------------------------------------------------------------ several folders (train/valid/test)
def _split_dataset(root, kind):
    """train/valid/test, 4 images each; the SAME file names in every split (as in many
    downloads); test/b.png is a copy of train/a.png (leakage)."""
    rs = np.random.RandomState(1)
    import yaml
    for si, sp in enumerate(("train", "valid", "test")):
        (root / sp / "images").mkdir(parents=True)
        coco = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "cell"}]}
        for i, n in enumerate("abcd"):
            img = (rs.rand(64, 64) * 200).astype(np.uint8)
            if sp == "test" and n == "b":
                img = np.asarray(Image.open(root / "train/images/a.png"))
            Image.fromarray(img).save(root / sp / "images" / f"{n}.png")
            k = si + 1 + i                       # objects per image differ between splits
            if kind == "paired":
                (root / sp / "masks").mkdir(exist_ok=True)
                m = np.zeros((64, 64), np.uint8)
                m[:k * 4, :8] = 1
                Image.fromarray(m).save(root / sp / "masks" / f"{n}.png")
            elif kind == "yolo":
                (root / sp / "labels").mkdir(exist_ok=True)
                with open(root / sp / "labels" / f"{n}.txt", "w") as fh:
                    for j in range(k):
                        fh.write(f"0 {0.1 + 0.08 * j:.3f} 0.5 0.05 0.05\n")
            else:
                coco["images"].append({"id": i, "file_name": f"{n}.png", "width": 64, "height": 64})
                for j in range(k):
                    coco["annotations"].append({"id": len(coco["annotations"]), "image_id": i,
                                                "category_id": 1, "bbox": [2 + 6 * j, 2, 4, 4]})
        if kind == "coco":
            json.dump(coco, open(root / sp / "_annotations.coco.json", "w"))
    if kind == "yolo":
        yaml.safe_dump({"names": ["cell"]}, open(root / "data.yaml", "w"))


@pytest.mark.parametrize("kind", ["coco", "yolo", "paired"])
def test_several_split_folders_as_one_dataset(tmp_path, kind):
    from medfingerprint.core.discovery import discover
    from medfingerprint.gui.app import detect_splits
    root = tmp_path / kind
    _split_dataset(root, kind)
    sp = detect_splits(str(root))
    assert sp["splits"] == ["test", "train", "valid"]
    if kind == "paired":
        assert len(sp["images"]) == 3 and len(sp["masks"]) == 3
        d = discover("paired", "2D", images_dir="; ".join(sp["images"]), masks_dir=sp["masks"], exts=["png"])
        ids = sorted(c.case_id for c in d.cases)
        assert ids[0] == "test/images/a.png" and len(ids) == 12
        for c in d.cases:   # every image got the mask of its own split
            assert c.mask.path.replace("/masks/", "/images/") == c.image
    else:
        assert len(sp["annotations"]) == 3
        d = discover("files", "2D", root=[str(root / s) for s in ("train", "valid", "test")], exts=["png"],
                     annotations="; ".join(sp["annotations"]))
        assert len(d.cases) == 12 and not d.annotations["ambiguous_names"]
        assert d.annotations["categories"] == ["cell"]
        for c in d.cases:   # same file names in all splits: objects come from the right split
            split, n = c.case_id.split("/")[0], c.case_id[-5]
            k = ("train", "valid", "test").index(split) + 1 + "abcd".index(n)
            assert len(c.mask.objects) == k, c.case_id
    # one parent folder gives the same case names; duplicates across splits are flagged
    if kind == "coco":
        out = tmp_path / "fp.json"
        subprocess.run([sys.executable, "-m", "medfingerprint", "scan", "--root", str(root / "train"),
                        str(root / "valid"), str(root / "test"), "--ext", "png", "--modality", "2D",
                        "--annotations", *sp["annotations"], "--out", str(out), "--workers", "2"],
                       check=True, capture_output=True)
        fp = json.load(open(out))
        assert fp["dataset"]["n_ok"] == 12 and fp["annotations"]["n_objects"] == (1 + 2 + 3) * 4 + (0 + 1 + 2 + 3) * 3
        g = [set(x) for x in fp["duplicates"]["exact_groups"]]
        assert {"train/images/a.png", "test/images/b.png"} in g and fp["duplicates"]["n_cross_group_pairs"] >= 1


# ------------------------------------------------------------------ common 0-255 scale in Compare
def test_compare_common_scale_8bit_vs_16bit(frac, frac_fp, tmp_path):
    """The same images stored as 8-bit JPEG and as 16-bit PNG (x257): far apart on the
    original scale, practically identical on the common 0-255 scale."""
    from medfingerprint.core import compare as C
    d16 = tmp_path / "img16"
    for f in sorted((frac / "images").rglob("*.jpg")):
        a = np.asarray(Image.open(f).convert("L"), dtype=np.uint16) * 257
        out = d16 / f.relative_to(frac / "images").with_suffix(".png")
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(a).save(out)
    fp16 = tmp_path / "fp16.json"
    subprocess.run([sys.executable, "-m", "medfingerprint", "scan", "--root", str(d16), "--ext", "png",
                    "--modality", "2D", "--out", str(fp16), "--workers", "2"], check=True, capture_output=True)
    a, b = C.load_for_compare(str(frac_fp)), C.load_for_compare(str(fp16))
    assert C.region_hist(b, "all")[1].hi > 60000
    w_orig = C.distance_matrix([a, b], "all")[0, 1]
    for mode in ("bits", "minmax", "robust"):
        sa, sb = C.rescale_fp(a, mode), C.rescale_fp(b, mode)
        w = C.distance_matrix([sa, sb], "all", "wasserstein")[0, 1]
        ks = C.distance_matrix([sa, sb], "all", "ks")[0, 1]
        pa, pb = sa["intensity"]["all"]["pooled"], sb["intensity"]["all"]["pooled"]
        # 0-255 units. KS stays a few % above 0 because the 16-bit histogram has 16-value bins
        # (8-bit: 1-value bins): a grey value holding many pixels (black background) falls on
        # slightly different bin edges -- a histogram resolution effect, not a scaling error.
        assert w < 1.0 and ks < 0.05, (mode, w, ks)
        assert abs(pa["mean"] - pb["mean"]) < 1.0 and abs(pa["std"] - pb["std"]) < 1.0, mode
        assert pa["skewness"] == pytest.approx(a["intensity"]["all"]["pooled"]["skewness"])
        rows = {m: v for _, m, v, _ in C.stats_rows([sa, sb], "all")}
        assert "intensity scale (original → 0–255)" in rows
    assert w_orig > 1000                                                 # original units: ~x257 apart
    s = C.rescale_fp(b, "bits")["_scale"]
    assert s["from"] == [0.0, 65535.0]
    assert C.rescale_fp(a, "original") is a


# ---------------------------------------------------------------- SAROS layout
def _write_dicom_series(vol, d, uid, spacing=(1.0, 1.0, 2.5)):
    import SimpleITK as sitk
    os.makedirs(d, exist_ok=True)
    img = sitk.GetImageFromArray(vol.astype(np.int16))
    img.SetSpacing(spacing)
    w = sitk.ImageFileWriter()
    w.KeepOriginalImageUIDOn()
    for i in range(img.GetDepth()):
        sl = img[:, :, i]
        for tag, val in {"0008|0060": "CT", "0020|000e": uid, "0020|000d": uid + ".1",
                         "0008|0018": f"{uid}.2.{i}", "0020|0013": str(i),
                         "0020|0032": "\\".join(map(str, img.TransformIndexToPhysicalPoint((0, 0, i)))),
                         "0020|0037": "1\\0\\0\\0\\1\\0", "0018|0050": str(spacing[2]),
                         "0028|1052": "0", "0028|1053": "1"}.items():
            sl.SetMetaData(tag, val)
        w.SetFileName(os.path.join(d, f"{i:03d}.dcm"))
        w.Execute(sl)


def test_saros_layout_sparse_masks(tmp_path):
    """SAROS: DICOM CT + 5 mm masks with every 5th slice annotated (255 elsewhere), linked by
    series UID. Labels must use only the annotated slices, placed by physical position."""
    import SimpleITK as sitk
    root = tmp_path / "SAROS"
    nz = 40                                             # 40 slices x 2.5 mm
    rows = []
    for k in range(2):
        uid = f"1.2.826.0.1.3680043.8.498.{k + 1}"
        vol = np.full((nz, 32, 32), 1000, np.int16)     # unannotated slices: 1000 (must be unused)
        for z in (0, 10, 20, 30):                       # image slices of annotated mask slices
            vol[z, :16], vol[z, 16:] = -100, 50
        _write_dicom_series(vol, str(root / "coll" / f"P{k}" / "series"), uid)
        m = np.full((nz // 2, 32, 32), 255, np.uint8)    # 5 mm grid, same origin
        for z in (0, 5, 10, 15):
            m[z, :16], m[z, 16:] = 1, 2
        mi = sitk.GetImageFromArray(m)
        mi.SetSpacing((1.0, 1.0, 5.0))
        os.makedirs(root / "nifti" / f"case_{k:03d}")
        sitk.WriteImage(mi, str(root / "nifti" / f"case_{k:03d}" / "body-regions.nii.gz"))
        rows.append(f"case_{k:03d},{uid},COLL,P{k},female,abdomen,FALSE,fold-{k + 1}")
    rows.append("case_002,1.2.3.missing,COLL,P9,male,thorax,FALSE,test")
    (root / "Segmentation-Info_x.csv").write_text(
        "id,tcia_series_instance_uid,tcia_collection,tcia_case_id,gender,anatomic_region,"
        "has_breast_implant,split\n" + "\n".join(rows) + "\n")

    from medfingerprint.core.discovery import discover
    disc = discover("saros", "CT", root=str(root))
    assert [c.case_id for c in disc.cases] == ["case_000", "case_001"]
    assert any("1 of 3 SAROS cases have no CT" in w for w in disc.warnings)

    out = tmp_path / "s.json"
    r = subprocess.run([sys.executable, "-m", "medfingerprint", "scan", "--layout", "saros", "--root",
                        str(root), "--modality", "CT", "--out", str(out), "--workers", "2"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    fp = json.load(open(out))
    assert set(fp["labels"]) == {"subcutaneous_tissue", "muscle"}
    fat, mus = fp["labels"]["subcutaneous_tissue"], fp["labels"]["muscle"]
    assert fat["pooled_intensity"]["mean"] == pytest.approx(-100)
    assert mus["pooled_intensity"]["mean"] == pytest.approx(50)
    # true volume: 16x32 voxels x 40 slices x 2.5 mm3 = 51.2 ml (estimated from 4 slices x 10)
    assert fat["volume_ml"]["median"] == pytest.approx(51.2)
    ls = fp["label_sampling"]
    assert ls["annotated_slices"]["median"] == 4 and ls["volume_factor"]["median"] == pytest.approx(10)
    assert fp["metadata"]["split"]["counts"] == {"fold-1": 1, "fold-2": 1}


def test_mura_study_labels(tmp_path):
    """MURA: study-level labels from *_labeled_studies.csv become per-image metadata."""
    root = tmp_path / "MURA-v1.1"
    rows = {"train": [], "valid": []}
    for split, bp, pat, study, lab in [("train", "XR_ELBOW", "patient00001", "study1_positive", 1),
                                       ("train", "XR_HAND", "patient00002", "study1_negative", 0),
                                       ("valid", "XR_ELBOW", "patient11185", "study2_negative", 1)]:
        d = root / split / bp / pat / study
        d.mkdir(parents=True)
        for k in (1, 2):
            Image.fromarray(np.full((40, 30), 50 * k, np.uint8)).save(d / f"image{k}.png")
        rows[split].append(f"MURA-v1.1/{split}/{bp}/{pat}/{study}/,{lab}")
    (root / "train" / "XR_HAND" / "patient00002" / "study1_negative" / "._image1.png").write_bytes(b"x")
    for split, r in rows.items():
        (root / f"{split}_labeled_studies.csv").write_text("\n".join(r) + "\n")
    from medfingerprint.core.discovery import discover
    for chosen in (str(root), f"{root / 'train'};{root / 'valid'}"):
        disc = discover("files", "2D", root=chosen, exts=["png"])
        assert len(disc.cases) == 6 and disc.metadata_csv["matched"] == 6
        meta = {c.case_id: c.meta for c in disc.cases}
        m = meta["train/XR_ELBOW/patient00001/study1_positive/image1.png"]
        assert m == {"split": "train", "body_part": "elbow", "patient_id": "patient00001",
                     "study": "patient00001/study1", "abnormal": "abnormal"}
        # the CSV wins over the folder name (deliberately inconsistent here)
        assert meta["valid/XR_ELBOW/patient11185/study2_negative/image2.png"]["abnormal"] == "abnormal"


def test_metadata_csv_never_uses_ambiguous_ids(tmp_path):
    """CheXpert: every study has view1_frontal.jpg and the CSV path starts higher up
    (CheXpert-v1.0-small/train/...). TCIA: one row per series, the patient id repeats. Each
    case must get its own row, never the first row sharing a file or folder name."""
    from medfingerprint.core.discovery import discover
    root = tmp_path / "train"
    rows = ["Path,Sex,Age"]
    for p in range(1, 4):
        for s in (1, 2):
            d = root / f"patient{p:05d}" / f"study{s}"
            d.mkdir(parents=True)
            Image.fromarray(np.full((20, 20), 10 * p + s, np.uint8)).save(d / "view1_frontal.jpg")
            rows.append(f"CheXpert-v1.0-small/train/patient{p:05d}/study{s}/view1_frontal.jpg,"
                        f"{'Male' if p % 2 else 'Female'},{20 * p + s}")
    (tmp_path / "train.csv").write_text("\n".join(rows) + "\n")
    disc = discover("files", "2D", root=str(root), exts=["jpg"], meta_csv=str(tmp_path / "train.csv"))
    assert disc.metadata_csv["id_col"] == "Path" and disc.metadata_csv["rows_used"] == 6
    ages = {c.case_id: c.meta["Age"] for c in disc.cases}
    assert ages["patient00002/study1/view1_frontal.jpg"] == "41"
    assert ages["patient00003/study2/view1_frontal.jpg"] == "62"

    # TCIA-style manifest: Subject ID repeats (CT + RTSTRUCT rows); File Location is unique
    ct = tmp_path / "tcia" / "Coll" / "SUBJ-1" / "study" / "2.000-CT-1"
    ct.mkdir(parents=True)
    Image.fromarray(np.zeros((8, 8), np.uint8)).save(ct / "slice.png")
    (tmp_path / "tcia" / "metadata.csv").write_text(
        "Subject ID,Modality,File Location\n"
        "SUBJ-1,RTSTRUCT,./Coll/SUBJ-1/study/1.000-RTSTRUCT-1\n"
        "SUBJ-1,CT,./Coll/SUBJ-1/study/2.000-CT-1\n")
    disc = discover("files", "2D", root=str(tmp_path / "tcia" / "Coll"), exts=["png"],
                    meta_csv=str(tmp_path / "tcia" / "metadata.csv"))
    assert disc.cases[0].meta.get("Modality") == "CT"


def test_several_metadata_csvs_one_per_folder(tmp_path):
    """CheXpert-style: train/ + valid/ scanned together, train.csv + valid.csv attached."""
    import numpy as np
    from PIL import Image
    from medfingerprint.core.aggregate import _csv_report
    from medfingerprint.core.discovery import discover
    for split, n in (("train", 4), ("valid", 2)):
        rows = ["Path,Sex,Age"]
        for i in range(n):
            d = tmp_path / split / f"patient{i}" / "study1"
            d.mkdir(parents=True)
            Image.fromarray(np.full((16, 16), 10 * i, np.uint8)).save(d / "view1_frontal.jpg")
            rows.append(f"CheXpert-v1.0-small/{split}/patient{i}/study1/view1_frontal.jpg,"
                        f"{'Male' if split == 'train' else 'Female'},{20 + i}")
        (tmp_path / f"{split}.csv").write_text("\n".join(rows) + "\n")
    disc = discover("files", "2D", root=f"{tmp_path / 'train'}; {tmp_path / 'valid'}", exts=["jpg"],
                    meta_csv=[str(tmp_path / "train.csv"), str(tmp_path / "valid.csv")])
    m = disc.metadata_csv
    assert m["matched"] == 6 and m["unmatched"] == 0 and len(m["files"]) == 2
    assert [f["matched"] for f in m["files"]] == [4, 2]
    assert not [w for w in disc.warnings if "Metadata" in w]
    sex = {c.case_id.split("/")[0]: c.meta["Sex"] for c in disc.cases}
    assert sex == {"train": "Male", "valid": "Female"}
    assert _csv_report(m)["path"] == "train.csv; valid.csv"


def test_case_filter_and_amos_modality(tmp_path):
    """AMOS (Decathlon-style dataset.json): ids >= 500 are MRI; --only keeps one modality."""
    import json
    import SimpleITK as sitk
    import numpy as np
    from medfingerprint.core.discovery import discover
    root = tmp_path / "amos22"
    for d in ("imagesTr", "labelsTr", "imagesVa", "labelsVa"):
        (root / d).mkdir(parents=True)
    tr, va = [], []
    for i, (sub, lst) in enumerate([("Tr", tr), ("Tr", tr), ("Va", va), ("Tr", tr)]):
        cid = f"amos_{(1, 4, 8, 507)[i]:04d}"
        img = sitk.GetImageFromArray(np.full((4, 8, 8), -500, np.int16))
        sitk.WriteImage(img, str(root / f"images{sub}" / f"{cid}.nii.gz"))
        sitk.WriteImage(sitk.GetImageFromArray(np.ones((4, 8, 8), np.uint8)),
                        str(root / f"labels{sub}" / f"{cid}.nii.gz"))
        lst.append({"image": f"./images{sub}/{cid}.nii.gz", "label": f"./labels{sub}/{cid}.nii.gz"})
    (root / "dataset.json").write_text(json.dumps({"name": "AMOS", "modality": {"0": "CT"},
                                                   "labels": {"0": "background", "1": "spleen"},
                                                   "training": tr, "validation": va}))
    disc = discover("nnunet", "3D", root=str(root))
    assert {c.case_id: c.meta["modality"] for c in disc.cases}["amos_0507"] == "MRI"
    ct = discover("nnunet", "3D", root=str(root), only="modality=CT")
    assert len(ct.cases) == 3 and ct.case_filter == {"rule": "modality=CT", "kept": 3, "removed": 1}
    va_ct = discover("nnunet", "3D", root=str(root), only="split=validation; modality!=mri")
    assert [c.case_id for c in va_ct.cases] == ["amos_0008"]
