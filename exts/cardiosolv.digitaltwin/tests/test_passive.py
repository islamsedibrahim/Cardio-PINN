"""Passive personalisation: Klotz EDPVR, unloading of end-diastolic geometry, stiffness fit."""

import numpy as np
import pytest

from cardiosolv.digitaltwin.mechanics.passive import KLOTZ_AN, klotz_edpvr, klotz_v0

torch = pytest.importorskip("torch")


def test_klotz_relation():
    v, v0, v30 = klotz_edpvr(120.0, 12.0)
    assert v0 == pytest.approx(klotz_v0(120.0, 12.0)) == pytest.approx(120 * (0.6 - 0.072))
    assert np.interp(12.0, [0, 5, 10, 15, 20, 25, 30], v) == pytest.approx(120.0, rel=0.02)  # passes through (EDV, EDP)
    # Klotz normalisation: EDVn = (V - V0)/(V30 - V0) = (P / 27.78)^(1/2.76), so V(27.78 mmHg) = V30
    assert v[-1] == pytest.approx(v0 + (v30 - v0) * (30 / KLOTZ_AN) ** (1 / 2.76)) and np.all(np.diff(v) > 0)
    assert klotz_edpvr(120.0, 12.0, [KLOTZ_AN])[0][0] == pytest.approx(v30)


def test_unload_reproduces_image(heart_pipeline):
    from cardiosolv.digitaltwin.mechanics import MechanicsConfig, MechanicsModel
    from cardiosolv.digitaltwin.mechanics.passive import inflate, unload

    p = heart_pipeline
    model = MechanicsModel(p.mesh, p.coords, p.geometry.long_axis, MechanicsConfig(max_iter=120))
    v_img = model.V0s[0] / 1000
    info = unload(model, [10.0, 5.0], iters=5, steps=2, log=lambda *a: None)
    assert info["unloaded_volumes_ml"][0] < 0.9 * v_img  # the unloaded LV is smaller than the ED image
    assert abs(model.offset).max() > 0.5
    _, vols = inflate(model, [10.0, 5.0], steps=2)
    assert abs(vols[0] - v_img) / v_img < 0.03  # inflating the reference reproduces the image
    model.set_reference(p.mesh.points)  # restore


def test_geometry_state_from_imaging(heart_pipeline):
    p = heart_pipeline
    assert p.resolved_geometry_state() == "unloaded"  # default: the user declares an end-diastolic image
    p.cfg.geometry_state = "auto"
    assert p.resolved_geometry_state() == "unloaded"  # a plain USD heart
    prim = p.stage.GetPrimAtPath("/World/Patient")
    prim.SetCustomDataByKey("cardiosolv:imaging_usd", "heart.usda")  # as load_heart_into_stage marks it
    try:
        assert p.resolved_geometry_state() == "end_diastolic"
    finally:
        prim.ClearCustomDataByKey("cardiosolv:imaging_usd")
        p.cfg.geometry_state = "unloaded"
