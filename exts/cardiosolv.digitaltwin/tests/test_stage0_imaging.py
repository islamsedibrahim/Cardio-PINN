"""Stage 0 -> Stage 1: DICOM -> imaging (local subprocess and REST service) -> heart referenced into a stage."""

import socket
import sys
import threading
import time
from pathlib import Path

import nibabel as nib
import pytest
from pxr import Usd, UsdGeom

IMAGING = Path(__file__).resolve().parents[3] / "imaging"
pytestmark = pytest.mark.skipif(not (IMAGING / "cardiosolv_imaging").is_dir(), reason="imaging package not present")
sys.path.insert(0, str(IMAGING))
sys.path.insert(0, str(IMAGING / "tests"))

from cardiosolv.digitaltwin.core import anatomy as A  # noqa: E402
from cardiosolv.digitaltwin.imaging_client import ImagingConfig, load_heart_into_stage, segment  # noqa: E402
from cardiosolv.digitaltwin.pipeline import CardioSolvPipeline, PipelineConfig  # noqa: E402


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    from synthetic_dicom import gt_label_volume, make_fake_bundle, write_ct_study

    d = tmp_path_factory.mktemp("study")
    write_ct_study(str(d / "dicom"))
    lab, aff, _ = gt_label_volume()
    nib.save(nib.Nifti1Image(lab, aff), str(d / "gt.nii.gz"))
    shim = make_fake_bundle(d / "bundle", d / "gt.nii.gz")
    return d, shim


def _check_stage(usd_path, tmp_path):
    # user scene in centimetres, Y-up (common in Omniverse)
    stage = Usd.Stage.CreateNew(str(tmp_path / "scene.usda"))
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    heart = load_heart_into_stage(stage, usd_path)
    pipe = CardioSolvPipeline(stage, heart, PipelineConfig(), log=lambda *a: None)
    pipe.run_discover()
    assert pipe.scan.sim_scale == 1.0
    assert pipe.assignment.status(A.MYOCARDIUM) == "HIGH"
    pipe.run_geometry()
    m = pipe.geometry.metrics
    assert 80 < m["myocardial_volume_ml"] < 110 and 45 < m["lv_cavity_volume_ml"] < 65
    # heart is upright in the Y-up scene: long axis apex->base points up (+Y)
    assert pipe.geometry.long_axis.direction[1] > 0.7


def test_stage0_local(study, tmp_path, monkeypatch):
    d, shim = study
    monkeypatch.setenv("NV_SEGMENT_CTMR_ROOT", str(d / "bundle"))
    monkeypatch.setenv("NV_SEGMENT_CTMR_PYTHON", shim)
    cfg = ImagingConfig(mode="local", engine="nv-segment", python=sys.executable, package_dir=str(IMAGING))
    rep = segment(d / "dicom", tmp_path / "out", cfg, log=lambda *a: None)
    assert rep["qc"]["status"] == "READY_FOR_TWIN"
    _check_stage(rep["outputs"]["usd"], tmp_path)


def test_stage0_service(study, tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    uvicorn = pytest.importorskip("uvicorn")
    d, shim = study
    monkeypatch.setenv("NV_SEGMENT_CTMR_ROOT", str(d / "bundle"))
    monkeypatch.setenv("NV_SEGMENT_CTMR_PYTHON", shim)
    monkeypatch.setenv("CARDIOSOLV_IMAGING_WORKDIR", str(tmp_path / "work"))
    import importlib

    import cardiosolv_imaging.server as server

    importlib.reload(server)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(server.app, host="127.0.0.1", port=port, log_level="error"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.1)
    try:
        cfg = ImagingConfig(mode="service", engine="nv-segment", service_url=f"http://127.0.0.1:{port}")
        rep = segment(d / "dicom", tmp_path / "out", cfg, log=lambda *a: None)
        assert Path(rep["outputs"]["usd"]).is_file()
        _check_stage(rep["outputs"]["usd"], tmp_path)
    finally:
        srv.should_exit = True
        th.join(timeout=5)
