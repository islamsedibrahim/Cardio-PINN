"""Biventricular twin: RV free wall, two transmural fields, per-ventricle Purkinje, LV + RV cavities."""

import dataclasses

import numpy as np
import pytest

from cardiosolv.digitaltwin.core.geometry_layer import ENDO, RV_SEPTUM, V_RV_MYO
from cardiosolv.digitaltwin.ep import run_electrophysiology

torch = pytest.importorskip("torch")


def test_geometry_and_mesh_include_rv_free_wall(heart_pipeline):
    g, mesh = heart_pipeline.geometry, heart_pipeline.mesh
    assert g.biventricular and (g.voxel_class == V_RV_MYO).any()
    assert 15 < g.metrics["rv_wall_volume_ml"] < 50  # 3.5 mm shell around the RV blood pool
    assert mesh.biventricular and 0.1 < (mesh.tet_region == 1).mean() < 0.5
    sets = mesh.node_sets()
    # nodes shared by RV endo and epi faces sit on the tricuspid / pulmonary rims (and the staircase
    # of a one-element-thick wall at the 4 mm test resolution)
    assert len(sets["rv_endo"]) > 50 and len(np.intersect1d(sets["rv_endo"], sets["epi"])) < 0.45 * len(sets["rv_endo"])
    assert mesh.metadata["unlabelled_faces"] == 0 and (mesh.tet_volumes() > 0).all()


def test_two_transmural_fields(heart_pipeline):
    vc, mesh = heart_pipeline.coords, heart_pipeline.mesh
    sets = mesh.node_sets()
    rv_fw_endo = sets["rv_endo"][vc.node_region[sets["rv_endo"]] == 1]
    assert vc.x_t[rv_fw_endo].max() < 1e-9  # RV free wall: 0 at the RV endocardium
    assert vc.x_t[mesh.node_set(ENDO)].max() < 1e-9  # LV: 0 at the LV endocardium
    sept_rv = sets["rv_endo"][vc.node_region[sets["rv_endo"]] == 0]
    assert vc.x_t[sept_rv].min() > 0.9  # septum: LV endo 0 -> RV side 1
    # fibres rotate across the RV free wall too
    rv = mesh.tet_region == 1
    helix = np.degrees(np.arctan2(np.einsum("ij,ij->i", vc.fiber, vc.e_l), np.einsum("ij,ij->i", vc.fiber, vc.e_c)))
    xt = vc.x_t[mesh.tets].mean(1)
    assert helix[rv & (xt < 0.3)].mean() > helix[rv & (xt > 0.7)].mean() + 40


def test_bundle_branch_blocks(heart_pipeline):
    p = heart_pipeline
    m = {proto: run_electrophysiology(p.mesh, p.coords, dataclasses.replace(p.cfg.ep, protocol=proto)).metrics
         for proto in ("sinus", "lbbb", "crt")}
    assert m["sinus"]["rv_total_activation_ms"] < 100 and abs(m["sinus"]["interventricular_delay_ms"]) < 25
    # LBBB: RV activated through the right bundle, LV free wall late
    assert m["lbbb"]["rv_total_activation_ms"] < 100 and m["lbbb"]["interventricular_delay_ms"] > 40
    assert m["crt"]["qrs_duration_ms"] < m["lbbb"]["qrs_duration_ms"] - 20


def test_two_cavity_mechanics(heart_pipeline):
    from cardiosolv.digitaltwin.mechanics import MechanicsConfig, MechanicsModel
    from cardiosolv.digitaltwin.mechanics.model import MMHG

    p = heart_pipeline
    model = MechanicsModel(p.mesh, p.coords, p.geometry.long_axis, MechanicsConfig(max_iter=150),
                           p.ep.activation_time, p.ep.apd)
    assert model.cavity_names == ["LV", "RV"]
    assert 40 < model.V0s[1] / 1000 < 90
    assert (p.mesh.face_labels == RV_SEPTUM).sum() > 50
    u0 = np.zeros((p.mesh.n_nodes, 3))
    u, _, _, inf = model.solve(u0, np.zeros(p.mesh.n_tets),
                               loads=[("pressure", 10 * MMHG, 0, 0), ("pressure", 5 * MMHG, 0, 0)])
    v_lv, v_rv = inf["volumes"]
    assert v_lv > model.V0s[0] * 1.15 and v_rv > model.V0s[1] * 1.1  # both ventricles fill
    ta = model.active_tension(150.0)
    iso = [("volume", 0.0, 200.0, v + p_ * MMHG / 200.0) for v, p_ in zip(inf["volumes"], (10, 5))]
    _, _, _, inf2 = model.solve(u, ta, loads=iso)
    p_lv, p_rv = (x / MMHG for x in inf2["pressures"])
    assert all(abs(a - b) / b < 0.03 for a, b in zip(inf2["volumes"], inf["volumes"]))
    assert p_lv > 60 and 10 < p_rv < p_lv  # thin RV builds a lower systolic pressure


def test_lv_only_switch(heart_usd):
    from pxr import Usd

    from cardiosolv.digitaltwin.pipeline import CardioSolvPipeline, PipelineConfig

    pipe = CardioSolvPipeline(Usd.Stage.Open(heart_usd), "/World/Patient/Heart",
                              PipelineConfig(biventricular=False, element_size_mm=4.0), log=lambda *a: None)
    pipe.run_discover()
    pipe.run_geometry()
    assert not pipe.geometry.biventricular and not (pipe.geometry.voxel_class == V_RV_MYO).any()
    pipe.run_mesh()
    assert not pipe.mesh.biventricular and pipe.coords.node_region is None
