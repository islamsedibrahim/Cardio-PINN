"""CRT planning on the twin: LV lead position sweep and response prediction.

For a patient in LBBB, every candidate LV epicardial lead site on the free wall
is simulated with the eikonal solver (fractions of a second each) together with
the RV septal lead. Sites in dense scar do not capture. The best site minimises
LV activation time. The electrical response is then graded with criteria from
the CRT literature:

* LV activation time (LVAT) shortening and QRS narrowing;
* scar burden > 33 % of the LV, or a lead over transmural scar, predicts
  non-response (Adelstein 2007; Bleeker 2006);
* optionally (``fe_beat``) the FE heartbeat of LBBB vs best-site CRT gives the
  acute haemodynamic response, change in LV dP/dt max (>= 10 % = responder) and EF.

Research tool: it ranks lead sites on this model; it does not replace clinical
assessment.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Optional

import numpy as np

from .electrophysiology import _site_nodes, run_electrophysiology

DEFAULT_XL = (0.35, 0.5, 0.65, 0.8)
DEFAULT_XC = tuple(np.round(np.arange(0.25, 0.76, 0.0625), 4))


def _lead_scar(mesh, coords, cfg, scar, xl, xc):
    nodes = _site_nodes(mesh, coords, mesh.node_sets(), "epi", xl, xc, cfg.site_radius_mm)
    if scar is None or not scar.any:
        return nodes, 0.0, 0.0
    return nodes, float(scar.node_core[nodes].mean()), float((scar.node_core | scar.node_bz)[nodes].mean())


def grade_response(lbbb, best, scar_metrics=None, lead_scar=0.0):
    lvat_red = 100.0 * (lbbb["lv_activation_time_ms"] - best["lv_activation_time_ms"]) / max(
        lbbb["lv_activation_time_ms"], 1e-9)
    qrs_red = lbbb["qrs_duration_ms"] - best["qrs_duration_ms"]
    reasons = []
    burden = (scar_metrics or {}).get("lv_scar_burden_pct", 0.0)
    if burden > 33:
        reasons.append(f"LV scar burden {burden:.0f} % > 33 %")
    if lead_scar > 0.5:
        reasons.append("best reachable LV lead site lies over scar")
    if (scar_metrics or {}).get("lateral_scar_pct", 0.0) > 50:
        reasons.append("transmural lateral-wall scar")
    if reasons:
        grade = "unlikely"
    elif lvat_red >= 20 and qrs_red >= 20:
        grade = "likely"
    elif lvat_red >= 10:
        grade = "possible"
    else:
        grade = "unlikely"
        reasons.append(f"LV activation shortens only {lvat_red:.0f} %")
    return {"predicted_response": grade, "lvat_reduction_pct": lvat_red, "qrs_reduction_ms": qrs_red,
            "reasons": reasons}


def crt_study(mesh, coords, ep_cfg, scar=None, long_axis=None, xl_grid=DEFAULT_XL, xc_grid=DEFAULT_XC,
              fe_beat: Optional[Callable] = None, log: Callable = print) -> dict:
    """Sweep LV lead sites; ``fe_beat(ep_result) -> cycle metrics`` adds the haemodynamic response."""
    base = dataclasses.replace(ep_cfg, solver="eikonal")
    lbbb = run_electrophysiology(mesh, coords, dataclasses.replace(base, protocol="lbbb"), log=log, scar=scar)
    sites = []
    for xl in xl_grid:
        for xc in xc_grid:
            _, core_frac, scar_frac = _lead_scar(mesh, coords, base, scar, xl, xc)
            row = {"x_l": float(xl), "x_c": float(xc), "lead_core_fraction": core_frac,
                   "lead_scar_fraction": scar_frac, "capture": core_frac < 0.999}
            if row["capture"]:
                r = run_electrophysiology(mesh, coords, dataclasses.replace(base, protocol="crt", lv_lead=(xl, xc)),
                                          log=lambda *a: None, scar=scar)
                row.update({k: r.metrics[k] for k in ("qrs_duration_ms", "lv_activation_time_ms",
                                                      "septal_to_lateral_delay_ms")})
            sites.append(row)
    ok = [s for s in sites if s["capture"]]
    if not ok:
        raise RuntimeError("No LV lead site captures (all candidate sites in dense scar).")
    # pacing over scar captures poorly and predicts non-response: use sites off scar when any exist
    off_scar = [s for s in ok if s["lead_scar_fraction"] <= 0.25]
    best = min(off_scar or ok, key=lambda s: s["lv_activation_time_ms"] + 30.0 * s["lead_scar_fraction"])
    default = min(ok, key=lambda s: abs(s["x_l"] - 0.55) + abs(s["x_c"] - 0.5))
    out = {
        "lbbb": {k: lbbb.metrics[k] for k in ("qrs_duration_ms", "lv_activation_time_ms", "septal_to_lateral_delay_ms")},
        "best_site": best, "standard_lateral_site": default, "sites": sites,
        "response": grade_response(lbbb.metrics, best, scar.metrics if scar is not None else None,
                                   best["lead_scar_fraction"]),
    }
    log(f"CRT: LBBB LVAT {lbbb.metrics['lv_activation_time_ms']:.0f} ms -> best lead (x_l {best['x_l']:.2f}, "
        f"x_c {best['x_c']:.2f}) {best['lv_activation_time_ms']:.0f} ms; predicted response "
        f"{out['response']['predicted_response']}")
    if fe_beat is not None:
        best_ep = run_electrophysiology(mesh, coords, dataclasses.replace(base, protocol="crt",
                                                                          lv_lead=(best["x_l"], best["x_c"])),
                                        long_axis, log=log, scar=scar)
        m0, m1 = fe_beat(lbbb), fe_beat(best_ep)
        out["haemodynamics"] = {
            "lbbb": m0, "crt": m1,
            "dpdt_max_change_pct": 100.0 * (m1["dpdt_max_mmhg_s"] - m0["dpdt_max_mmhg_s"]) / max(m0["dpdt_max_mmhg_s"], 1e-9),
            "ef_change_points": m1["ejection_fraction_pct"] - m0["ejection_fraction_pct"],
            "stroke_work_change_pct": 100.0 * (m1["stroke_work_mmhg_ml"] - m0["stroke_work_mmhg_ml"]) / max(
                abs(m0["stroke_work_mmhg_ml"]), 1e-9),
        }
        h = out["haemodynamics"]
        out["response"]["acute_haemodynamic_responder"] = bool(h["dpdt_max_change_pct"] >= 10.0)
        log(f"CRT: dP/dt max {h['dpdt_max_change_pct']:+.1f} %, EF {h['ef_change_points']:+.1f} points")
    return out
