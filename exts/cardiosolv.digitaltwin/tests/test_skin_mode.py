"""Skin-only hearts (visual / generated assets): auto-scale + assumed interior."""

import numpy as np
from pxr import Usd, UsdGeom

from cardiosolv.digitaltwin.pipeline import CardioSolvPipeline, PipelineConfig
from fixtures import write_skin_only_heart_usd


def test_skin_only_heart_pipeline(tmp_path):
    path = write_skin_only_heart_usd(str(tmp_path / "skin.usda"))
    stage = Usd.Stage.Open(path)
    pipe = CardioSolvPipeline(stage, "/root", PipelineConfig(element_size_mm=4.0), log=lambda *a: None)
    for name in ("discover", "geometry", "mesh", "ep"):
        getattr(pipe, f"run_{name}")()
    assert 0.08 < pipe.scan.sim_scale < 0.15  # ~1.5 m asset simulated at heart size
    gl = pipe.geometry
    assert pipe.skin_part == 0 and gl.myocardium_source == "skin_assumed"
    m = gl.metrics
    assert 40 < m["myocardial_volume_ml"] < 200
    assert 15 < m["lv_cavity_volume_ml"] < 120
    assert m["rv_cavity_volume_ml"] > 5
    # base (atria/vessels) is at local +z of the synthetic heart
    assert gl.long_axis.direction @ np.array([0, 0, 1.0]) > 0.8
    assert pipe.validation["status"] == "REQUIRES ANATOMICAL VALIDATION"
    assert any("ASSUMED" in w for w in pipe.validation["warnings"])
    skin = stage.GetPrimAtPath("/root/Generate_a_SimReady/Visuals/Generate_a_SimReady_001")
    names = {s.GetPrim().GetName() for s in UsdGeom.Subset.GetAllGeomSubsets(UsdGeom.Imageable(skin))}
    assert {"cardiosolv_LVEpicardium", "cardiosolv_AtriaGreatVessels"} <= names
    # twin painted on the user's skin, landmarks back at displayed (metre) scale
    pipe.write_twin("activation_time")
    col = UsdGeom.PrimvarsAPI(skin).GetPrimvar("displayColor").Get()
    assert len(col) == len(UsdGeom.Mesh(skin).GetPointsAttr().Get())
    assert stage.GetPrimAtPath("/CardioSolv/Debug/AssumedEndocardium")
    apex = np.array(stage.GetPrimAtPath("/CardioSolv/Landmarks/Apex").GetCustomDataByKey("cardiosolv:position_world"))
    pts = np.asarray(UsdGeom.Mesh(skin).GetPointsAttr().Get())
    # apex = centroid of the apical wall region (a few mm inside), scaled back to the metre-scale asset
    gap_sim_mm = np.linalg.norm(pts - apex, axis=1).min() * 1000.0 * pipe.scan.sim_scale
    assert gap_sim_mm < 6.0
