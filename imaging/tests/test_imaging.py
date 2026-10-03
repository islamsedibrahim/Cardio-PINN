"""Stage 0 tests: DICOM (CT, cine MR) -> engines -> fusion -> USD -> CardioSolv recognises the heart."""

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "exts" / "cardiosolv.digitaltwin"))

from cardiosolv_imaging.dicom_io import dicom_to_nifti, scan_dicom  # noqa: E402
from cardiosolv_imaging.engines import nv_segment  # noqa: E402
from cardiosolv_imaging.pipeline import run  # noqa: E402
from synthetic_dicom import gt_label_volume, make_fake_bundle, write_ct_study, write_cine_mr_study  # noqa: E402


@pytest.fixture(scope="module")
def ct_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("ct")
    write_ct_study(str(d / "dicom"))
    return d


@pytest.fixture(scope="module")
def gt_path(tmp_path_factory):
    lab, aff, _ = gt_label_volume()
    p = tmp_path_factory.mktemp("gt") / "gt.nii.gz"
    nib.save(nib.Nifti1Image(lab, aff), str(p))
    return p


def test_dicom_series_selection_and_geometry(ct_dir, tmp_path):
    series = scan_dicom(ct_dir / "dicom")
    assert len(series) == 2  # CT + localiser
    info = dicom_to_nifti(ct_dir / "dicom", tmp_path / "ct.nii.gz")
    assert info["modality"] == "CT" and info["series"]["slice_positions"] > 100
    img = nib.load(str(tmp_path / "ct.nii.gz"))
    lab, aff, _ = gt_label_volume()
    data = np.asarray(img.dataobj)
    # LV blood pool (350 HU) sits where the ground truth says, in RAS
    lv_ras = (np.c_[np.argwhere(lab == 151), np.ones((lab == 151).sum())] @ aff.T)[:, :3].mean(0)
    ijk = np.round((np.r_[lv_ras, 1] @ np.linalg.inv(img.affine).T)[:3]).astype(int)
    assert data[tuple(ijk)] > 250
    # no PHI in the summary
    assert "PHANTOM" not in json.dumps(info)


def test_cine_mr_end_diastole(tmp_path):
    write_cine_mr_study(str(tmp_path / "mr"))
    ed = dicom_to_nifti(tmp_path / "mr", tmp_path / "ed.nii.gz", phase="ed")
    es = dicom_to_nifti(tmp_path / "mr", tmp_path / "es.nii.gz", phase="2")
    assert ed["modality"] == "MR" and ed["load"]["phases_available"] == 3
    assert ed["load"]["trigger_time_ms"] == 0.0 and es["load"]["trigger_time_ms"] == 600.0
    assert abs(ed["load"]["slice_spacing_mm"] - 8.0) < 1e-3
    bright_ed = (np.asarray(nib.load(str(tmp_path / "ed.nii.gz")).dataobj) > 600).sum()
    bright_es = (np.asarray(nib.load(str(tmp_path / "es.nii.gz")).dataobj) > 600).sum()
    assert bright_ed > bright_es  # blood pool is largest at end-diastole


def test_nv_segment_engine_contract_and_ct_pipeline(ct_dir, gt_path, tmp_path, monkeypatch):
    root = tmp_path / "bundle"
    fake_py = make_fake_bundle(root, gt_path)
    monkeypatch.setenv("NV_SEGMENT_CTMR_ROOT", str(root))
    orig = nv_segment.NVSegmentCTMR.segment
    monkeypatch.setattr(nv_segment.NVSegmentCTMR, "segment",
                        lambda self, *a, **k: orig(self, *a, python=fake_py, **k))
    rep = run(ct_dir / "dicom", tmp_path / "out", engine="nv-segment", log=lambda *a: None)
    s = rep["structures"]
    assert {"heart_myocardium", "heart_ventricle_left", "heart_ventricle_right", "heart_atrium_left",
            "heart_atrium_right", "aorta"} <= set(s)
    lab, aff, _ = gt_label_volume()
    gt_lv = (lab == 151).sum() / 1000.0
    assert abs(s["heart_ventricle_left"]["volume_ml"] - gt_lv) / gt_lv < 0.08
    assert all(v["surface"]["watertight"] for v in s.values())
    assert rep["qc"]["status"] == "READY_FOR_TWIN"
    assert rep["steps"]["engines"][0]["modality"] == "CT_BODY"
    _check_cardiosolv_recognises(rep["outputs"]["usd"])


def test_labelmap_engine_cine_mr_acdc(tmp_path):
    """Thick-slice cine MR + an existing ACDC-style segmentation -> isotropic smooth USD."""
    write_cine_mr_study(str(tmp_path / "mr"))
    info = dicom_to_nifti(tmp_path / "mr", tmp_path / "ed.nii.gz")
    img = nib.load(info["nifti"])
    lab, aff, _ = gt_label_volume()
    from nibabel.processing import resample_from_to

    gt_on_mr = np.asarray(resample_from_to(nib.Nifti1Image(lab, aff), img, order=0).dataobj).astype(np.int16)
    acdc = np.zeros_like(gt_on_mr)
    acdc[gt_on_mr == 152], acdc[gt_on_mr == 154], acdc[gt_on_mr == 151] = 1, 2, 3
    nib.save(nib.Nifti1Image(acdc, img.affine), str(tmp_path / "acdc.nii.gz"))
    rep = run(info["nifti"], tmp_path / "out", modality="MR", label_map=str(tmp_path / "acdc.nii.gz"),
              mapping="acdc", log=lambda *a: None)
    s = rep["structures"]
    assert set(s) == {"heart_myocardium", "heart_ventricle_left", "heart_ventricle_right"}
    iso = nib.load(rep["outputs"]["label_map"])
    assert np.allclose(np.sqrt((iso.affine[:3, :3] ** 2).sum(0)), 1.0, atol=0.15)  # 8 mm slices -> ~1 mm
    gt_lv = (lab == 151).sum() / 1000.0
    assert abs(s["heart_ventricle_left"]["volume_ml"] - gt_lv) / gt_lv < 0.25  # 8 mm slices lose the base
    _check_cardiosolv_recognises(rep["outputs"]["usd"])


def _check_cardiosolv_recognises(usd_path):
    """When the CardioSolv extension is next to this package, check Stage 1 recognises the heart."""
    try:
        import cardiosolv.digitaltwin.pipeline  # noqa: F401
    except ImportError:
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(usd_path)
        names = {p.GetName() for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)}
        assert {"heart_myocardium", "heart_ventricle_left"} <= names
        return
    from pxr import Usd

    from cardiosolv.digitaltwin.core import anatomy as A
    from cardiosolv.digitaltwin.pipeline import CardioSolvPipeline, PipelineConfig

    stage = Usd.Stage.Open(usd_path)
    pipe = CardioSolvPipeline(stage, "/World/Patient/Heart", PipelineConfig(), log=lambda *a: None)
    pipe.run_discover()
    assert pipe.scan.sim_scale == 1.0  # anatomical size, metres -> mm correct
    roles = {r: pipe.scan.parts[i].name for r, i in pipe.assignment.roles.items()}
    assert roles[A.MYOCARDIUM] == "heart_myocardium" and roles[A.LV] == "heart_ventricle_left"
    assert pipe.assignment.status(A.MYOCARDIUM) == "HIGH"
    pipe.run_geometry()
    assert pipe.geometry.myocardium_source == "part"
    assert 30 < pipe.geometry.metrics["myocardial_volume_ml"] < 160


def test_rest_service_roundtrip(ct_dir, gt_path, tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    import io
    import time
    import zipfile

    from fastapi.testclient import TestClient

    root = tmp_path / "bundle"
    monkeypatch.setenv("NV_SEGMENT_CTMR_ROOT", str(root))
    monkeypatch.setenv("NV_SEGMENT_CTMR_PYTHON", make_fake_bundle(root, gt_path))
    monkeypatch.setenv("CARDIOSOLV_IMAGING_WORKDIR", str(tmp_path / "work"))
    import importlib

    import cardiosolv_imaging.server as server

    importlib.reload(server)
    client = TestClient(server.app)
    assert client.get("/health").json()["engines"]["nv-segment"] == "ok"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for f in (ct_dir / "dicom").iterdir():
            zf.write(f, f"study/{f.name}")
    job = client.post("/segment", files={"file": ("study.zip", buf.getvalue(), "application/zip")},
                      data={"engine": "nv-segment"}).json()["job"]
    for _ in range(300):
        st = client.get(f"/jobs/{job}").json()
        if st["status"] in ("done", "failed"):
            break
        time.sleep(0.5)
    assert st["status"] == "done", st
    assert st["qc"]["status"] == "READY_FOR_TWIN"
    usd = client.get(f"/jobs/{job}/heart.usda")
    assert usd.status_code == 200 and b"heart_myocardium" in usd.content
    assert client.get(f"/jobs/{job}/../../etc/passwd").status_code == 404
    assert client.delete(f"/jobs/{job}").json()["deleted"]
