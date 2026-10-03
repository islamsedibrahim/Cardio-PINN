"""Stages 3-6: mesh, fibres, electrophysiology, mechanics, surrogate."""

import dataclasses

import numpy as np
import pytest

from cardiosolv.digitaltwin.core.geometry_layer import ENDO
from cardiosolv.digitaltwin.ep import run_electrophysiology

torch = pytest.importorskip("torch")


def test_mesh_conforms_to_user_surface(heart_pipeline):
    md = heart_pipeline.mesh.metadata
    assert md["unlabelled_faces"] == 0
    gm = heart_pipeline.geometry.metrics
    # biventricular by default: the mesh holds the LV wall + septum and the RV free wall
    assert abs(md["volume_ml"] - gm["myocardial_volume_ml"] - gm.get("rv_wall_volume_ml", 0.0)) < 8
    assert md["snap_mean_distance_mm"] < 1.0
    assert (heart_pipeline.mesh.tet_volumes() > 0).all()


def test_fibre_helix_rotates_transmurally(heart_pipeline):
    vc, mesh = heart_pipeline.coords, heart_pipeline.mesh
    xt = vc.x_t[mesh.tets].mean(1)
    helix = np.degrees(np.arctan2(np.einsum("ij,ij->i", vc.fiber, vc.e_l), np.einsum("ij,ij->i", vc.fiber, vc.e_c)))
    assert helix[xt < 0.15].mean() > 40
    assert abs(helix[(xt > 0.4) & (xt < 0.6)].mean()) < 10
    assert helix[xt > 0.85].mean() < -40
    assert vc.x_t[mesh.node_set(ENDO)].max() < 1e-9


def test_ep_scenarios_are_physiological(heart_pipeline):
    p = heart_pipeline
    out = {}
    for proto in ("sinus", "lbbb", "crt"):
        cfg = dataclasses.replace(p.cfg.ep, protocol=proto)
        out[proto] = run_electrophysiology(p.mesh, p.coords, cfg, None, None).metrics
    assert 50 < out["sinus"]["qrs_duration_ms"] < 110
    assert out["lbbb"]["qrs_duration_ms"] > 130  # LBBB: QRS > 120 ms
    assert out["lbbb"]["septal_to_lateral_delay_ms"] > 60
    assert out["crt"]["septal_to_lateral_delay_ms"] < 0.5 * out["lbbb"]["septal_to_lateral_delay_ms"]


def test_vm_template_and_ecg(heart_pipeline):
    ep = heart_pipeline.ep
    assert ep.vm(-5.0).max() == pytest.approx(-85.0)
    v = ep.vm(float(np.median(ep.activation_time)) + 50.0)
    assert (v > 0).mean() > 0.4
    assert set(ep.ecg) == {"apex_base", "lateral"}


def test_mechanics_inflation_and_contraction(heart_pipeline):
    from cardiosolv.digitaltwin.mechanics import MechanicsConfig, MechanicsModel
    from cardiosolv.digitaltwin.mechanics.model import MMHG

    p = heart_pipeline
    model = MechanicsModel(p.mesh, p.coords, p.geometry.long_axis, MechanicsConfig(max_iter=150),
                           p.ep.activation_time, p.ep.apd)
    assert 45 < model.V0 / 1000 < 65
    u0 = np.zeros((p.mesh.n_nodes, 3))
    u, v_ed, _, _ = model.solve(u0, np.zeros(p.mesh.n_tets), "pressure", p_pa=8 * MMHG)
    assert v_ed > model.V0 * 1.15  # passive filling increases volume
    ta = model.active_tension(150.0)
    _, v_iso, p_iso, _ = model.solve(u, ta, "volume", alpha=200.0, v_star=v_ed + 8 * MMHG / 200.0)
    assert abs(v_iso - v_ed) / v_ed < 0.02  # isovolumetric
    assert p_iso / MMHG > 60  # active tension builds systolic pressure
    J = model.element_fields(u, ta)["jacobian"]
    assert abs(J.mean() - 1) < 0.02  # nearly incompressible myocardium
