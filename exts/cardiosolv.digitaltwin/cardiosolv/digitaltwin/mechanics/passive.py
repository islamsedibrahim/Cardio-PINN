"""Stage 6a': passive (diastolic) personalisation.

Two problems are solved here, with the same passive inflation as the beat:

1. **Unloaded reference.** A CT or cine-MR heart reconstructed at end-diastole is
   already inflated by the end-diastolic pressures. Treating it as stress-free
   overestimates EDV and underestimates stiffness. ``unload`` recovers the
   stress-free geometry with the backward-displacement fixed point of Sellier
   (2011): ``X_ref <- X_img - u(X_ref; EDP)`` until inflating ``X_ref`` to EDP
   reproduces the image.
2. **Stiffness.** One scale on the Holzapfel-Ogden stiffnesses
   (``MaterialParams.stiff_scale``) is fitted so the LV reproduces either

   * the Klotz end-diastolic pressure-volume relation (Klotz et al., AJP 2006),
     which predicts the unloaded volume ``V0 = EDV (0.6 - 0.006 EDP)`` and the
     whole EDPVR ``P = 27.78 ((V - V0)/(V30 - V0))^2.76`` from one (EDV, EDP)
     point, for end-diastolic images; or
   * a measured EDV at a measured EDP (echo / catheter) when the geometry is
     already unloaded (e.g. an artist asset).

The secant iterations work on ``log(stiff_scale)``; every evaluation warm-starts
from the previous solution so a calibration costs a few dozen passive solves.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .model import MMHG, MechanicsModel

KLOTZ_AN, KLOTZ_BN = 27.78, 2.76
EDPVR_PRESSURES = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0)


def klotz_v0(edv_ml: float, edp_mmhg: float) -> float:
    """Unloaded LV volume predicted from one end-diastolic point (Klotz 2006)."""
    return edv_ml * (0.6 - 0.006 * edp_mmhg)


def klotz_edpvr(edv_ml: float, edp_mmhg: float, pressures_mmhg=EDPVR_PRESSURES):
    """Volumes (mL) on the Klotz EDPVR through (EDV, EDP), plus V0 and V30."""
    v0 = klotz_v0(edv_ml, edp_mmhg)
    v30 = v0 + (edv_ml - v0) / (max(edp_mmhg, 0.5) / KLOTZ_AN) ** (1.0 / KLOTZ_BN)
    p = np.asarray(pressures_mmhg, float)
    return v0 + (v30 - v0) * (np.clip(p, 0, None) / KLOTZ_AN) ** (1.0 / KLOTZ_BN), v0, v30


def inflate(model: MechanicsModel, pressures_mmhg: Sequence[float], steps=4, u0=None):
    """Passive inflation of every cavity to ``pressures_mmhg``; returns (u, volumes mL)."""
    u = np.zeros((model.mesh.n_nodes, 3)) if u0 is None else np.asarray(u0, float)
    zero_ta = np.zeros(model.mesh.n_tets)
    ps = list(pressures_mmhg) + [0.0] * (model.n_cavities - len(pressures_mmhg))
    inf = {"volumes": model.V0s}
    for i in range(1, steps + 1):
        f = i / steps
        u, _, _, inf = model.solve(u, zero_ta, loads=[("pressure", p * MMHG * f, 0.0, 0.0) for p in ps])
    return u, [v / 1000.0 for v in inf["volumes"]]


def edpvr(model: MechanicsModel, pressures_mmhg=EDPVR_PRESSURES, rv_ratio=0.5):
    """Model LV EDPVR from the current reference (RV loaded at ``rv_ratio`` of the LV pressure)."""
    u = np.zeros((model.mesh.n_nodes, 3))
    vols = []
    prev = 0.0
    zero_ta = np.zeros(model.mesh.n_tets)
    for p in pressures_mmhg:
        if p <= 0:
            vols.append(model.V0s[0] / 1000.0)
            continue
        for f in np.linspace(0, 1, 3)[1:]:  # sub-steps between curve points
            q = prev + (p - prev) * f
            loads = [("pressure", q * MMHG, 0, 0), ("pressure", rv_ratio * q * MMHG, 0, 0)][: model.n_cavities]
            u, _, _, inf = model.solve(u, zero_ta, loads=loads)
        vols.append(inf["volumes"][0] / 1000.0)
        prev = p
    return np.array(vols)


def unload(model: MechanicsModel, edps_mmhg: Sequence[float], iters=8, tol_mm=None, steps=4,
           log: Callable = print) -> dict:
    """Recover the stress-free reference of an end-diastolic geometry (Sellier fixed point)."""
    X_img = model.mesh.points.copy()
    tol = tol_mm if tol_mm is not None else 0.1 * model.mesh.spacing  # on the mean node residual
    X = X_img.copy()
    model.set_reference(X)
    v_img = [v / 1000.0 for v in model.V0s]
    hist, hist_mean = [], []
    u_prev = None
    for it in range(iters):
        u, vols = inflate(model, edps_mmhg, steps, u0=None if u_prev is None else u_prev)
        res = np.linalg.norm(X + u - X_img, axis=1)
        hist.append(float(res.max()))
        hist_mean.append(float(res.mean()))
        if res.mean() < tol and abs(vols[0] - v_img[0]) / v_img[0] < 0.01:
            break
        step = 1.0
        while step > 0.1:
            try:
                model.set_reference(X + step * (X_img - u - X))
                break
            except RuntimeError:  # inverted elements: damp the update
                step *= 0.5
        else:
            log("passive: unloading stopped (reference would invert elements)")
            model.set_reference(X)
            break
        X = model.X.cpu().numpy()
        # previous solution mapped onto the new reference (same deformed state, new reference)
        u_prev = X_img - X
    else:
        u, vols = inflate(model, edps_mmhg, steps, u0=u_prev)
        res = np.linalg.norm(X + u - X_img, axis=1)
        hist.append(float(res.max()))
        hist_mean.append(float(res.mean()))
    vol_err = abs(vols[0] - v_img[0]) / v_img[0]
    return {"iterations": len(hist), "residual_mm": hist[-1], "mean_residual_mm": hist_mean[-1],
            "residual_history_mm": hist, "unloaded_volumes_ml": [v / 1000.0 for v in model.V0s],
            "image_volumes_ml": v_img, "loaded_volumes_ml": vols, "lv_volume_error_pct": 100 * vol_err,
            "converged": hist_mean[-1] < tol and vol_err < 0.02}


def _secant_log(f, s0, s1, tol=0.01, it=8, lo=0.05, hi=20.0, log=print):
    """Root of f(s) on log s; f increasing in s."""
    x0, x1 = np.log(s0), np.log(s1)
    f0, f1 = f(s0), f(s1)
    hist = [(s0, f0), (s1, f1)]
    for _ in range(it):
        if abs(f1) < tol:
            break
        if abs(f1 - f0) < 1e-9:
            break
        x2 = float(np.clip(x1 - f1 * (x1 - x0) / (f1 - f0), np.log(lo), np.log(hi)))
        x0, f0 = x1, f1
        x1, f1 = x2, f(float(np.exp(x2)))
        hist.append((float(np.exp(x1)), f1))
        log(f"passive: stiffness x{np.exp(x1):.3f} -> residual {f1:+.3f}")
    return float(np.exp(x1)), f1, hist


def calibrate_passive(model: MechanicsModel, edp_mmhg: float, rv_edp_mmhg: float = 5.0, mode: str = "klotz",
                      measured_edv_ml: Optional[float] = None, unload_iters=6, log: Callable = print) -> dict:
    """Personalise passive stiffness (and recover the unloaded reference for ``mode="klotz"``).

    ``mode="klotz"``: the mesh is the end-diastolic image; EDV = image (or ``measured_edv_ml``).
    Stiffness is chosen so the unloaded LV volume equals Klotz' V0, i.e. the imaged EDV sits on
    the Klotz EDPVR at EDP; the model is left at that unloaded reference.
    ``mode="measured"``: the mesh is unloaded; stiffness is chosen so inflation to EDP gives
    ``measured_edv_ml``.
    """
    mat = model.cfg.material
    s_init = mat.stiff_scale
    edps = [edp_mmhg, rv_edp_mmhg][: model.n_cavities]
    out: Dict[str, object] = {"mode": mode, "edp_mmhg": edp_mmhg, "initial_stiff_scale": s_init}

    if mode == "klotz":
        X_img = model.mesh.points.copy()
        model.set_reference(X_img)
        edv_img = model.V0s[0] / 1000.0
        edv = measured_edv_ml or edv_img
        target_ratio = klotz_v0(edv, edp_mmhg) / edv
        out.update({"image_edv_ml": edv_img, "target_edv_ml": edv, "klotz_v0_ml": klotz_v0(edv, edp_mmhg)})
        cache = {}

        def resid(s):
            mat.stiff_scale = s
            model.set_reference(X_img)
            info = unload(model, edps, iters=unload_iters, log=log)
            cache[s] = (model.X.cpu().numpy().copy(), info)
            return info["unloaded_volumes_ml"][0] / edv_img - target_ratio

        s, r, hist = _secant_log(resid, s_init, s_init * 1.6, tol=0.005, log=log)
        X_ref, info = cache[s]
        mat.stiff_scale = s
        model.set_reference(X_ref)
        out.update({"unloading": info, "unloaded_lv_ml": info["unloaded_volumes_ml"][0]})
    elif mode == "measured":
        if not measured_edv_ml:
            raise ValueError("mode='measured' needs measured_edv_ml")
        edv = measured_edv_ml
        state = {"u": None}

        def resid(s):
            mat.stiff_scale = s
            u, vols = inflate(model, edps, u0=state["u"])
            state["u"] = u
            return np.log(edv) - np.log(vols[0])  # stiffer -> smaller EDV -> residual increases

        s, r, hist = _secant_log(resid, s_init, s_init * 1.6, tol=0.005, log=log)
        mat.stiff_scale = s
        out.update({"target_edv_ml": edv})
    else:
        raise ValueError(f"unknown passive calibration mode {mode}")

    out.update({"stiff_scale": s, "residual": float(r), "history": [(float(a), float(b)) for a, b in hist],
                "a_iso_kpa": mat.a_iso * s / 1e3, "a_f_kpa": mat.a_f * s / 1e3})
    out["diastolic"] = diastolic_metrics(model, edp_mmhg, out.get("target_edv_ml"))
    log(f"passive: stiffness x{s:.3f} (a = {out['a_iso_kpa']:.2f} kPa), model EDV "
        f"{out['diastolic']['model_edv_ml']:.1f} mL at {edp_mmhg:.0f} mmHg")
    return out


def diastolic_metrics(model: MechanicsModel, edp_mmhg: float, edv_ref_ml: Optional[float] = None) -> dict:
    """Model EDPVR vs Klotz, V30, end-diastolic chamber stiffness."""
    p = np.array(EDPVR_PRESSURES)
    v = edpvr(model, p)
    edv = float(np.interp(edp_mmhg, p, v))
    k = int(np.clip(np.searchsorted(p, edp_mmhg), 1, len(p) - 1))
    stiffness = float((p[k] - p[k - 1]) / max(v[k] - v[k - 1], 1e-9))
    kv, v0k, v30k = klotz_edpvr(edv_ref_ml or edv, edp_mmhg, p)
    # Klotz shape: fit beta of P = alpha * (V - V0)^beta... reported as the exponent of the model curve
    sel = (p > 0) & (v > v[0] + 0.05)
    beta = float(np.polyfit(np.log(v[sel] - v[0]), np.log(p[sel]), 1)[0]) if sel.sum() >= 2 else float("nan")
    return {"edpvr_pressures_mmhg": p.tolist(), "edpvr_volumes_ml": v.tolist(), "klotz_volumes_ml": kv.tolist(),
            "model_edv_ml": edv, "model_v0_ml": float(v[0]), "model_v30_ml": float(v[-1]),
            "klotz_v0_ml": float(v0k), "klotz_v30_ml": float(v30k),
            "edpvr_rms_error_ml": float(np.sqrt(np.mean((v - kv) ** 2))),
            "ed_chamber_stiffness_mmhg_per_ml": stiffness, "edpvr_exponent": beta,
            "klotz_exponent": KLOTZ_BN}
