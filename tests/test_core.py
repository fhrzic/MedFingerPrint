import json
import os
import subprocess
import sys

import numpy as np
import pytest
from scipy import stats as sstats

from medfingerprint.core.bins import BinSet, apply_binset, load_presets, quantile_bins
from medfingerprint.core.config import CT_HISTOGRAM, HistogramSpec
from medfingerprint.core.cropping import crop_to_nonzero
from medfingerprint.core.stats import Moments, hist_percentiles

HERE = os.path.dirname(__file__)
# Python of an environment with nnU-Net v2 installed, for the live comparison test (optional)
NNUNET_PY = os.environ.get("NNUNET_PY", "")


def test_moments_merge_matches_scipy():
    rs = np.random.RandomState(0)
    x = np.round(rs.gamma(2, 300, 200_000) - 1000)
    m = Moments.empty()
    for chunk in np.array_split(x, 7):
        m = m.merge(Moments.from_values(chunk))
    s = m.summary()
    assert np.isclose(s["mean"], x.mean(), rtol=1e-12)
    assert np.isclose(s["std"], x.std(), rtol=1e-12)
    assert np.isclose(s["skewness"], sstats.skew(x), rtol=1e-9)
    assert np.isclose(s["kurtosis"], sstats.kurtosis(x), rtol=1e-9)


def test_label_moments_match_per_label():
    rs = np.random.RandomState(1)
    x = rs.randn(50_000) * 100
    lab = rs.randint(0, 4, x.size)
    m = Moments.from_labels(x[:20000], lab[:20000], 4).merge(Moments.from_labels(x[20000:], lab[20000:], 4))
    s = m.summary()
    for k in range(4):
        assert np.isclose(s["mean"][k], x[lab == k].mean())
        assert np.isclose(s["std"][k], x[lab == k].std())


def test_hist_percentiles_exact_for_integer_data():
    rs = np.random.RandomState(2)
    x = np.round(rs.normal(40, 300, 100_001)).clip(-1500, 3500)  # includes under/overflow
    spec = CT_HISTOGRAM
    h = np.bincount(spec.index(x.astype(np.float32)), minlength=spec.size)
    p = hist_percentiles(h, spec, qs=(0.5, 5, 25, 50, 75, 95, 99.5), vmin=x.min(), vmax=x.max())
    ref = np.percentile(x, [0.5, 5, 25, 50, 75, 95, 99.5])
    assert np.allclose(list(p.values()), ref)


def test_bins_exact_and_open_ended():
    x = np.array([-2048, -1000, -100, 0, 50, 119, 120, 179, 180, 215, 216, 5000], dtype=np.float32)
    spec = CT_HISTOGRAM
    h = np.bincount(spec.index(x), minlength=spec.size)
    bs = BinSet.from_edges([0, 120, 180, 216], open_ends=True)
    got = apply_binset(h, spec, bs)
    assert np.allclose(got, [3, 3, 2, 2, 2])          # <0, [0,120), [120,180), [180,216), >=216
    assert np.isclose(got.sum(), x.size)
    assert not load_presets()["ct_tissue"].validate()  # preset has no gaps/overlaps


def test_quantile_bins_balanced():
    rs = np.random.RandomState(3)
    x = np.round(rs.normal(0, 200, 100_000)).astype(np.float32)
    h = np.bincount(CT_HISTOGRAM.index(x), minlength=CT_HISTOGRAM.size)
    c = apply_binset(h, CT_HISTOGRAM, quantile_bins(h, CT_HISTOGRAM, 4))
    assert np.all(np.abs(c / c.sum() - 0.25) < 0.01)


def test_crop_matches_nnunet_semantics():
    data = np.zeros((1, 6, 8, 8), np.float32)
    data[0, 1:5, 2:7, 1:6] = -1000
    seg = np.zeros((1, 6, 8, 8), np.int8)
    seg[0, 2, 3, 3] = 1
    d, s, bbox = crop_to_nonzero(data, seg)
    assert bbox == [[1, 5], [2, 7], [1, 6]]
    assert d.shape == (1, 4, 5, 5) and s[0, 1, 1, 2] == 1


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth")
    subprocess.run([sys.executable, os.path.join(HERE, "make_synthetic.py"), str(out)], check=True,
                   capture_output=True)
    return out


def _scan(args, out):
    subprocess.run([sys.executable, "-m", "medfingerprint", "scan", *args, "--out", str(out),
                    "--workers", "2"], check=True, capture_output=True)
    return json.load(open(out))


def test_pipeline_ct_nnunet_layout(synth, tmp_path):
    fp = _scan(["--layout", "nnunet", "--root", str(synth / "nnUNet_raw/Dataset990_MfpTest"),
                "--modality", "CT"], tmp_path / "fp.json")
    assert fp["dataset"]["n_ok"] == 10
    assert set(fp["intensity"]) == {"all", "body", "foreground"}
    assert set(fp["labels"]) == {"liver", "spleen", "vertebra"}
    assert fp["flags"]["not_hu_suspected"] == ["case_006"]
    assert fp["flags"]["padding_values"] == {"-2048.0": 1}
    ids = [e["case_id"] for e in fp["outliers"]["cases"]]
    assert ids[0] == "case_006"          # not in HU: unusual on many features
    assert "case_007" in ids             # 5 mm slices (rank depends on the other 9 cases)
    assert os.path.isfile(tmp_path / "fp.arrays.npz")


@pytest.mark.skipif(not NNUNET_PY or not os.path.isfile(NNUNET_PY), reason="set NNUNET_PY to a Python with nnU-Net v2")
def test_identical_to_nnunet(synth, tmp_path):
    env = dict(os.environ, nnUNet_raw=str(synth / "nnUNet_raw"),
               nnUNet_preprocessed=str(tmp_path / "pre"), nnUNet_results=str(tmp_path / "res"))
    code = ("from nnunetv2.experiment_planning.dataset_fingerprint.fingerprint_extractor import "
            "DatasetFingerprintExtractor as D\nif __name__=='__main__': D(990, 2, True).run(True)")
    subprocess.run([NNUNET_PY, "-c", code], env=env, check=True, capture_output=True)
    ref = tmp_path / "pre/Dataset990_MfpTest/dataset_fingerprint.json"
    _scan(["--layout", "nnunet", "--root", str(synth / "nnUNet_raw/Dataset990_MfpTest"),
           "--modality", "CT"], tmp_path / "fp.json")
    r = subprocess.run([sys.executable, os.path.join(HERE, "compare_nnunet.py"),
                        str(tmp_path / "fp.json"), str(ref)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout


def test_pipeline_totalsegmentator(synth, tmp_path):
    fp = _scan(["--layout", "totalsegmentator", "--root", str(synth / "ts"), "--modality", "CT"],
               tmp_path / "ts.json")
    assert fp["dataset"]["n_ok"] == 6
    assert fp["labels"]["liver"]["n_cases_present"] == 6
    assert fp["metadata"]["institute"]["counts"] == {"A": 3, "B": 3}
    assert abs(fp["labels"]["liver"]["pooled_intensity"]["mean"] - 60) < 1


def test_pipeline_2d(synth, tmp_path):
    fp = _scan(["--layout", "files", "--root", str(synth / "xray8"), "--ext", "png", "jpg",
                "--modality", "2D"], tmp_path / "x8.json")
    assert fp["histogram_spec"] == {"lo": 0.0, "hi": 256.0, "width": 1.0}
    flagged = {e["case_id"] for e in fp["outliers"]["cases"]}
    assert {"img_05.png", "img_06.png"} <= flagged
    fp16 = _scan(["--layout", "files", "--root", str(synth / "xray16"), "--ext", "png", "dicom",
                  "--modality", "2D"], tmp_path / "x16.json")
    means = {c["case_id"]: c["regions"]["all"]["mean"] for c in fp16["cases"]}
    assert abs(means["mono1.dcm"] - means["img16_04.png"]) < 1e-6   # MONOCHROME1 inverted
    assert "PatientName" not in fp16["cases"][-1]["meta"]


def test_images_only_ct_body_block(synth, tmp_path):
    """No masks: no real nnU-Net block, but a body-based nnU-Net-style block."""
    fp = _scan(["--layout", "files", "--root", str(synth / "nnUNet_raw/Dataset990_MfpTest/imagesTr"),
                "--modality", "CT"], tmp_path / "nomask.json")
    assert fp["nnunet_fingerprint"] is None and not fp["labels"]
    nb = fp["nnunet_fingerprint_body"]
    assert nb["_n_cases_used"] == 10 and "body mask" in nb["_foreground_source"]
    props = nb["foreground_intensity_properties_per_channel"]["0"]
    assert -1000 < props["percentile_00_5"] < -500     # lungs are inside the body
    assert props["percentile_99_5"] > 500                # bone
    assert set(fp["intensity"]) == {"all", "body"}


def test_body_block_also_present_with_labels(synth, tmp_path):
    fp = _scan(["--layout", "totalsegmentator", "--root", str(synth / "ts"), "--modality", "CT"],
               tmp_path / "ts.json")
    assert fp["nnunet_fingerprint"] and fp["nnunet_fingerprint_body"]


def test_2d_otsu_foreground(synth, tmp_path):
    fp = _scan(["--layout", "files", "--root", str(synth / "xray8"), "--ext", "png",
                "--modality", "2D"], tmp_path / "x8.json")
    assert "body" in fp["intensity"] and fp["flags"]["body_method"] == "otsu"
    assert 0.05 < fp["flags"]["body_fraction"]["median"] < 0.95
    # body excludes the dark background, so its mean is brighter than the whole image
    assert fp["intensity"]["body"]["pooled"]["mean"] > fp["intensity"]["all"]["pooled"]["mean"]
    fixed = _scan(["--layout", "files", "--root", str(synth / "xray8"), "--ext", "png",
                   "--modality", "2D", "--body", "100"], tmp_path / "x8b.json")
    assert fixed["flags"]["body_threshold"]["median"] == 100
    none = _scan(["--layout", "files", "--root", str(synth / "xray8"), "--ext", "png",
                  "--modality", "2D", "--body", "none"], tmp_path / "x8n.json")
    assert set(none["intensity"]) == {"all"} and none["nnunet_fingerprint_body"] is None


# ---------------------------------------------------------------- real-data regressions
def test_non_orthonormal_sform_is_read(tmp_path):
    """TotalSegmentator v3 oblique scans have sforms orthonormal only to ~2e-4 (ITK rejects)."""
    nib = pytest.importorskip("nibabel")
    from medfingerprint.core.io import read_label_array, read_sitk_file
    a = np.arange(4 * 5 * 6, dtype=np.int16).reshape(6, 5, 4)
    aff = np.diag([1.5, 1.5, 1.5, 1.0])
    aff[:3, :3] = aff[:3, :3] @ np.array([[0.9491, 0.3150, 0], [-0.3148, 0.9491, 0], [0, 0, 1]])
    p = str(tmp_path / "oblique.nii.gz")
    nib.save(nib.Nifti1Image(a, aff), p)
    li = read_sitk_file(p)
    assert li.array.shape == (1, 4, 5, 6) and np.array_equal(li.array[0], a.transpose(2, 1, 0))
    assert any("non-orthonormal" in n for n in li.notes)
    assert read_label_array(p).shape == (4, 5, 6)


def test_hidden_files_skipped(tmp_path):
    from PIL import Image
    from medfingerprint.core.discovery import discover
    Image.fromarray(np.zeros((8, 8), np.uint8)).save(tmp_path / "a.png")
    (tmp_path / "._a.png").write_bytes(b"\0" * 64)          # macOS AppleDouble fork
    (tmp_path / ".hidden").mkdir()
    Image.fromarray(np.zeros((8, 8), np.uint8)).save(tmp_path / ".hidden" / "b.png")
    d = discover("files", "2D", root=str(tmp_path), exts=["png"])
    assert [c.case_id for c in d.cases] == ["a.png"]


def test_many_batches_do_not_hang(synth, tmp_path, monkeypatch):
    """Regression: max_tasks_per_child deadlocked after workers x 25 cases."""
    from medfingerprint.core import runner
    from medfingerprint.core.config import ScanConfig
    from medfingerprint.core.discovery import discover
    monkeypatch.setattr(runner, "TASKS_PER_WORKER_BATCH", 1)
    d = discover("files", "2D", root=str(synth / "xray8"), exts=["png", "jpg"])
    out = runner.run_scan(d, ScanConfig(modality="2D", compute_nnunet=False), str(tmp_path / "c"),
                          workers=2)
    assert len(out.outcomes) >= 5 and all(o["status"] == "ok" for o in out.outcomes)


def test_outlier_scores_ties_fractions_groups():
    from medfingerprint.core.outliers import _scores, detect_outliers
    # >50% tied at 512, a large minority mode at ~400: not outliers; 2000 is.
    x = np.array([512.0] * 60 + list(np.linspace(380, 420, 39)) + [2000.0])
    s = _scores(x, "robust_z")
    assert np.sum(np.abs(s) > 3.5) == 1 and abs(s[-1]) > 3.5
    # float noise in spacing never scores high
    assert np.nanmax(np.abs(_scores(np.array([1.5] * 10 + [1.5000001] * 5), "robust_z"))) < 0.01
    rs = np.random.RandomState(0)
    frac = np.concatenate([rs.exponential(0.003, 199), [0.3]])   # one saturated image
    grp = ["A"] * 100 + ["B"] * 100
    mean = np.concatenate([rs.normal(50, 2, 100), rs.normal(150, 2, 100)])
    mean[5] = 150                                                  # normal for B, outlier in A
    fp = {"config": {"modality": "2D"}, "intensity": {"all": {}},
          "cases": [{"case_id": f"{g}/c{i}", "status": "ok", "geometry": {"dims": 2},
                     "flags": {"frac_at_image_max": f}, "regions": {"all": {"mean": m}}}
                    for i, (g, f, m) in enumerate(zip(grp, frac, mean))]}
    feats = ["flags.frac_at_image_max", "regions.all.mean"]
    r = detect_outliers(fp, features=feats)
    hits = {(f["case_id"], f["feature"]) for f in r["flags"]}
    assert ("B/c199", "flags.frac_at_image_max") in hits
    assert sum(1 for h in hits if h[1] == "flags.frac_at_image_max") <= 3   # log scale, high only
    assert ("A/c5", "regions.all.mean") not in hits                         # bimodal globally
    rg = detect_outliers(fp, features=feats, group_by="level:0")
    assert ("A/c5", "regions.all.mean") in {(f["case_id"], f["feature"]) for f in rg["flags"]}


def test_compare_distances_and_tables():
    from scipy import stats as st
    from medfingerprint.core import compare as C
    rs = np.random.RandomState(4)

    def fp(name, x, spec):
        h = np.bincount(spec.index(x.astype(np.float32)), minlength=spec.size)
        return {"schema": "medfingerprint/1.0", "dataset": {"name": name, "modality": "CT", "n_ok": 3},
                "histogram_spec": spec.to_dict(), "geometry": {"spacing": [{"median": 1.0}] * 3},
                "intensity": {"all": {"histogram": {"counts": h.tolist()},
                                      "pooled": {"mean": float(x.mean())}}},
                "metadata": {"ConvolutionKernel": {"type": "categorical", "counts": {"B30": 2}}}}
    xa = np.round(rs.normal(0, 100, 200_000))
    xb = np.round(rs.normal(40, 100, 200_000))
    a = fp("a", xa, CT_HISTOGRAM)
    b = fp("b", xb, HistogramSpec(-600, 700, 2.0))          # different histogram spec
    b["metadata"] = {"convolution_kernel": {"type": "categorical", "counts": {"B30": 1}}}
    w1 = C.distance_matrix([a, b], "all", "wasserstein")[0, 1]
    assert abs(w1 - st.wasserstein_distance(xa, xb)) < 1.5        # ~40 HU shift
    assert abs(C.distance_matrix([a, b], "all", "ks")[0, 1] - st.ks_2samp(xa, xb).statistic) < 0.01
    assert C.distance_matrix([a, a], "all", "js")[0, 1] < 1e-12
    rows = {(r[0], r[1]): r[2] for r in C.stats_rows([a, b], "all")}
    assert rows[("Metadata", "ConvolutionKernel")] == ["B30 100%", "B30 100%"]   # keys matched
    assert C.difference(10.0, 15.0) == (5.0, 50.0)
