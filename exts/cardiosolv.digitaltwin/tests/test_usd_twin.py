"""Stage 2 + 7 USD authoring: semantic layer, GeomSubsets, painted/animated twin."""

import numpy as np
from pxr import Usd, UsdGeom

from cardiosolv.digitaltwin.usd.builder import SURFACE_FAMILY
from cardiosolv.digitaltwin.usd.layer import LAYER_TAG, RESULTS_TAG, find_layer


def test_semantic_layer_and_subsets(heart_pipeline):
    stage = heart_pipeline.stage
    root = stage.GetRootLayer()
    sem = find_layer(stage, LAYER_TAG)
    assert sem is not None
    # the user's own layer holds no CardioSolv opinions
    assert not root.GetPrimAtPath("/CardioSolv")
    assert stage.GetPrimAtPath("/CardioSolv/Geometry/Heart/Myocardium")
    lm = stage.GetPrimAtPath("/CardioSolv/Landmarks/Apex")
    assert lm.GetCustomDataByKey("cardiosolv:confidence") > 0.9
    myo = stage.GetPrimAtPath("/World/Patient/Heart/heart_myocardium")
    subsets = {s.GetPrim().GetName(): s for s in UsdGeom.Subset.GetGeomSubsets(UsdGeom.Imageable(myo), familyName=SURFACE_FAMILY)}
    assert {"cardiosolv_Endocardium", "cardiosolv_Epicardium", "cardiosolv_Base"} <= set(subsets)
    n_faces = len(UsdGeom.Mesh(myo).GetFaceVertexCountsAttr().Get())
    ids = np.concatenate([np.asarray(s.GetIndicesAttr().Get()) for s in subsets.values()])
    assert len(np.unique(ids)) == len(ids)  # non-overlapping
    assert len(ids) > 0.97 * n_faces


def test_twin_painted_and_animated_on_user_mesh(heart_pipeline):
    p = heart_pipeline
    stage = p.stage
    root_before = stage.GetRootLayer().ExportToString()
    codes, (lo, hi) = p.write_twin("activation_time")
    myo = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Patient/Heart/heart_myocardium"))
    col = UsdGeom.PrimvarsAPI(myo.GetPrim()).GetPrimvar("displayColor")
    assert col.GetInterpolation() == UsdGeom.Tokens.vertex
    assert len(col.Get()) == len(myo.GetPointsAttr().Get())
    raw = np.asarray(UsdGeom.PrimvarsAPI(myo.GetPrim()).GetPrimvar("cardiosolv:activation_time").Get())
    assert raw.min() >= -1 and raw.max() <= p.ep.activation_time.max() + 1
    # EP-only twin (no mechanics yet): painted, not moved
    assert myo.GetPointsAttr().GetNumTimeSamples() == 0
    # fake a short "cycle" to exercise the animation path on every part

    T = 3
    u = np.zeros((T, p.mesh.n_nodes, 3))
    u[1] = 2.0 * p.geometry.long_axis.direction  # 2 mm shift
    codes = p.writer.time_codes(np.array([0.0, 10.0, 20.0]))
    p.writer.write_animation(codes, u)
    pts0 = np.asarray(myo.GetPointsAttr().Get(Usd.TimeCode(codes[0])))
    pts1 = np.asarray(myo.GetPointsAttr().Get(Usd.TimeCode(codes[1])))
    shift_mm = np.linalg.norm((pts1 - pts0).mean(0)) * 10.0  # cm stage, rotation-invariant length
    assert abs(shift_mm - 2.0) < 0.2
    rv = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Patient/Heart/heart_ventricle_right"))
    assert rv.GetPointsAttr().GetNumTimeSamples() == T  # neighbours follow too
    assert stage.GetRootLayer().ExportToString().count("timeSamples") == root_before.count("timeSamples")
    assert find_layer(stage, RESULTS_TAG) is not None
