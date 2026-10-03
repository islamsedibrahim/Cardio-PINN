"""Scar substrate (LGE / user-drawn meshes): labelling, EP block and slowing, channels, mechanics, CRT."""

import dataclasses

import numpy as np
import pytest

from cardiosolv.digitaltwin.core.scar import BORDER_ZONE, CORE, classify_scar_name
from cardiosolv.digitaltwin.ep import run_electrophysiology

torch = pytest.importorskip("torch")


def test_scar_names():
    assert classify_scar_name("scar_core") == CORE and classify_scar_name("myocardial_scar") == CORE
    assert classify_scar_name("scar_border_zone") == BORDER_ZONE and classify_scar_name("grey_zone") == BORDER_ZONE
    assert classify_scar_name("heart_myocardium") is None


def test_scar_mapped_not_anatomy(scar_pipeline):
    p = scar_pipeline
    assert {"/World/Patient/Scar/scar_core", "/World/Patient/Scar/scar_border_zone"} <= set(p.scar_paths)
    assert all("scar" not in part.name for part in p.scan.parts)
    m = p.scar.metrics
    assert abs(m["core_volume_ml"] - p.gt_scar["scar_core"]) / p.gt_scar["scar_core"] < 0.2
    assert abs(m["border_zone_volume_ml"] - p.gt_scar["scar_border_zone"]) / p.gt_scar["scar_border_zone"] < 0.25
    assert m["lateral_scar_pct"] > 20 and m["septal_scar_pct"] < 1 and m["max_transmurality_pct"] > 90
    assert m["conduction_channels"] >= 1  # the border-zone corridor through the core
    ch = m["channels"][0]
    assert ch["length_mm"] > 8 and 0.05 < ch["apparent_cv_m_s"] < 0.45  # slow conduction in the isthmus


def test_no_channel_without_corridor(tmp_path):
    from pxr import Usd

    from cardiosolv.digitaltwin.pipeline import CardioSolvPipeline, PipelineConfig
    from fixtures import add_synthetic_scar_usd, write_synthetic_heart_usd

    path = str(tmp_path / "heart.usda")
    write_synthetic_heart_usd(path)
    add_synthetic_scar_usd(path, channel=False)
    p = CardioSolvPipeline(Usd.Stage.Open(path), "/World/Patient/Heart", PipelineConfig(element_size_mm=3.0),
                           log=lambda *a: None)
    for name in ("discover", "geometry", "mesh"):
        getattr(p, f"run_{name}")()
    assert p.scar.metrics["conduction_channels"] == 0


def test_scar_blocks_and_slows_conduction(scar_pipeline):
    p = scar_pipeline
    healthy = run_electrophysiology(p.mesh, p.coords, p.cfg.ep).metrics
    scarred = p.ep.metrics
    assert scarred["unexcitable_nodes"] == int(p.scar.node_core.sum()) > 0
    assert scarred["qrs_duration_ms"] > healthy["qrs_duration_ms"] + 20
    # border zone repolarises later (longer APD)
    assert p.ep.apd[p.scar.node_bz].mean() > p.ep.apd[~p.scar.node_bz & ~p.scar.node_core].mean()
    # a lead in dense scar does not capture
    cfg = dataclasses.replace(p.cfg.ep, protocol="crt", lv_lead=(0.6, 0.5))
    r = run_electrophysiology(p.mesh, p.coords, cfg, scar=p.scar)
    assert not np.isin(r.stim_nodes, np.nonzero(p.scar.node_core)[0]).any()


def test_scar_mechanics(scar_pipeline):
    from cardiosolv.digitaltwin.mechanics import MechanicsConfig, MechanicsModel

    p = scar_pipeline
    model = MechanicsModel(p.mesh, p.coords, p.geometry.long_axis, MechanicsConfig(), p.ep.activation_time,
                           p.ep.apd, scar=p.scar)
    ta = model.active_tension(200.0)
    core, bz = p.scar.tet_label == CORE, p.scar.tet_label == BORDER_ZONE
    assert ta[core].max() == 0 and ta[bz].mean() < ta[~core & ~bz].mean()
    s = model.stiff_el.cpu().numpy()
    assert s[core].min() == pytest.approx(5.0) and s[~core & ~bz].max() == pytest.approx(1.0)


def test_crt_study_avoids_scar(scar_pipeline):
    r = scar_pipeline.run_crt_study()
    best = r["best_site"]
    assert best["capture"] and best["lead_scar_fraction"] <= 0.25
    assert best["lv_activation_time_ms"] < r["lbbb"]["lv_activation_time_ms"]
    assert r["response"]["predicted_response"] in ("likely", "possible", "unlikely")
    assert any(not s["capture"] or s["lead_scar_fraction"] > 0.5 for s in r["sites"])  # the scar is in the sweep
