"""Stage 6b: one heartbeat coupled to Windkessel circulations (Cardio-PINN loop).

Each ventricle has its own valve state machine: passive filling to EDP ->
isovolumetric contraction -> ejection -> isovolumetric relaxation -> filling.
The LV ejects into a systemic and, on biventricular meshes, the RV into a
pulmonary 3-element Windkessel. Implicit Euler on a Windkessel makes cavity
pressure a linear function of cavity volume during ejection,
``p = alpha (V* - V)``, so every time step is a single energy minimisation
with one cavity term per ventricle:

* filling:            -p V           (prescribed atrial / filling pressure)
* isovolumetric:      alpha_iso/2 (V* - V)^2 with V* updated so V = V_ref
* ejection:           alpha_wk/2  (V* - V)^2 from the Windkessel state

Both ventricles are solved together, so septal interaction (ventricular
interdependence) comes out of the mechanics rather than a lumped coupling.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .model import MMHG, MechanicsModel

ALPHA_ISO = 200.0  # Pa / mm^3, isovolumetric penalty


@dataclass
class CirculationParams:
    r_periph: float = 1.0  # mmHg s / mL (systemic resistance)
    c_art: float = 1.5  # mL / mmHg (arterial compliance)
    z_char: float = 0.05  # mmHg s / mL (characteristic impedance)
    p_aortic_diastolic: float = 80.0  # mmHg, aortic pressure when the beat starts
    edp: float = 10.0  # mmHg end-diastolic LV pressure
    p_atrial: float = 8.0  # mmHg, mitral opening pressure
    cycle_ms: float = 800.0
    dt_ms: float = 10.0
    fill_steps: int = 6
    # pulmonary circulation (RV), biventricular meshes only
    rv_r_periph: float = 0.14  # mmHg s / mL (pulmonary vascular resistance ~ 2.3 Wood units incl. runoff)
    rv_c_art: float = 4.0  # mL / mmHg (pulmonary arterial compliance)
    rv_z_char: float = 0.02  # mmHg s / mL
    p_pulmonary_diastolic: float = 10.0  # mmHg
    rv_edp: float = 5.0  # mmHg end-diastolic RV pressure
    p_right_atrial: float = 4.0  # mmHg, tricuspid opening pressure

    def chambers(self, n=2):
        lv = dict(name="LV", r=self.r_periph, c=self.c_art, z=self.z_char, p_dia=self.p_aortic_diastolic,
                  edp=self.edp, p_open=self.p_atrial)
        rv = dict(name="RV", r=self.rv_r_periph, c=self.rv_c_art, z=self.rv_z_char,
                  p_dia=self.p_pulmonary_diastolic, edp=self.rv_edp, p_open=self.p_right_atrial)
        return [lv, rv][:n]


class Chamber:
    """Valve state machine + Windkessel of one ventricle (pressures in mmHg, volumes in mL here)."""

    def __init__(self, prm: dict, dt_ms: float, cycle_ms: float, v_ed_ml: float, min_reverse_steps=1,
                 min_ejection_ms=0.0):
        self.name = prm["name"]
        self.r, self.c, self.z = prm["r"], prm["c"], prm["z"]
        self.edp, self.p_open = prm["edp"], prm["p_open"]
        self.dt, self.cycle_ms = dt_ms, cycle_ms
        self.beta = 1.0 / (1.0 + dt_ms / 1000.0 / (self.r * self.c))
        self.phase = "ivc"
        self.p = self.edp  # cavity pressure
        self.p_c = prm["p_dia"]  # Windkessel capacitor (arterial) pressure
        self.v_prev = self.v_ref = v_ed_ml
        self.t_fill = self.p_fill = self.t_eject = None
        self.reverse_steps = 0
        self.min_reverse_steps, self.min_ejection_ms = min_reverse_steps, min_ejection_ms

    # ---- what the solver has to satisfy this step ----
    def ejection_line(self):
        """Ejection: p = a (v* - V), a in mmHg/mL, v* in mL."""
        a = self.beta / self.c + self.z / (self.dt / 1000.0)
        return a, self.v_prev + self.beta * self.p_c / a

    def fill_pressure(self, t):
        frac = min((t - self.t_fill) / max(self.cycle_ms - self.t_fill, self.dt), 1.0)
        return self.p_fill + (self.edp - self.p_fill) * frac

    def fe_load(self, t):
        """FE cavity load ``(mode, p_pa, alpha Pa/mm^3, v_star mm^3)``."""
        if self.phase in ("ivc", "ivr"):
            return ("volume", 0.0, ALPHA_ISO, self.v_ref * 1000.0 + self.p * MMHG / ALPHA_ISO)
        if self.phase == "ejection":
            a, v_star = self.ejection_line()
            return ("volume", 0.0, a * MMHG / 1000.0, v_star * 1000.0)
        return ("pressure", self.fill_pressure(t) * MMHG, 0.0, 0.0)

    # ---- valve events ----
    def transition(self, t, p, v):
        """Open / close a valve if (p, v) of this step says so; True means re-solve the step."""
        if self.phase == "ivc" and p >= self.beta * self.p_c:
            self.phase, self.t_eject, self.reverse_steps = "ejection", t, 0
            return True
        if self.phase == "ivr" and p <= self.p_open:
            self.phase, self.t_fill, self.p_fill = "filling", t, max(p, 1.0)
            return True
        if self.phase == "ejection":
            q = (self.v_prev - v) / (self.dt / 1000.0)
            if (q < 0 and self.reverse_steps + 1 >= self.min_reverse_steps
                    and t - self.t_eject >= self.min_ejection_ms):
                self.phase, self.v_ref = "ivr", self.v_prev
                return True
        return False

    def commit(self, p, v):
        """Accept the step; returns (cavity mmHg, volume mL, arterial mmHg, outflow mL/s, phase)."""
        q = 0.0
        if self.phase == "ejection":
            q = (self.v_prev - v) / (self.dt / 1000.0)
            self.reverse_steps = self.reverse_steps + 1 if q < 0 else 0
            q = max(q, 0.0)
            self.p_c = self.beta * (self.p_c + q * (self.dt / 1000.0) / self.c)
        else:
            self.p_c = self.beta * self.p_c
        self.p, self.v_prev = p, v
        return p, v, (p if self.phase == "ejection" else self.p_c), q, self.phase


@dataclass
class CycleResult:
    times: np.ndarray  # ms (filling steps negative)
    pressure_mmhg: np.ndarray
    volume_ml: np.ndarray
    aortic_mmhg: np.ndarray
    flow_ml_s: np.ndarray
    phase: List[str]
    displacements: np.ndarray  # (T,N,3) mm
    active_tension_kpa: np.ndarray  # (T,E)
    element_fields: Dict[str, np.ndarray]  # each (T,E)
    metrics: Dict[str, float]
    params: dict
    solver_info: List[dict] = field(default_factory=list)
    # per-ventricle traces {"RV": {"pressure_mmhg", "volume_ml", "arterial_mmhg", "flow_ml_s", "phase"}}
    chambers: Dict[str, dict] = field(default_factory=dict)

    def frame_count(self):
        return len(self.times)


def _ml(v_mm3):
    return v_mm3 / 1000.0


def simulate_cycle(model: MechanicsModel, circ: CirculationParams = None, t_scale=1.0, log=print,
                   progress=None) -> CycleResult:
    circ = circ or CirculationParams()
    dt = circ.dt_ms
    n = model.mesh.n_nodes
    K = model.n_cavities
    prm = circ.chambers(K)
    u = np.zeros((n, 3))

    times, U, TA, info = [], [], [], []
    tr = [{"pressure_mmhg": [], "volume_ml": [], "arterial_mmhg": [], "flow_ml_s": [], "phase": []} for _ in range(K)]
    fields: Dict[str, list] = {}

    def record(t, rows, u_, ta, inf):
        times.append(t)
        for k, (p, v, pa, q, ph) in enumerate(rows):
            for key, val in zip(("pressure_mmhg", "volume_ml", "arterial_mmhg", "flow_ml_s", "phase"), (p, v, pa, q, ph)):
                tr[k][key].append(val)
        U.append((u_ + model.offset).astype(np.float32))  # relative to the imaged mesh
        TA.append((ta / 1e3).astype(np.float32))
        for key, val in model.element_fields(u_, ta).items():
            fields.setdefault(key, []).append(val.astype(np.float32))
        info.append(inf)

    zero_ta = np.zeros(model.mesh.n_tets)
    record(-dt * (circ.fill_steps + 1), [(0.0, _ml(model.V0s[k]), prm[k]["p_dia"], 0.0, "unloaded") for k in range(K)],
           u, zero_ta, {})

    # ---------------- passive filling to EDP ----------------
    for i in range(1, circ.fill_steps + 1):
        f = i / circ.fill_steps
        loads = [("pressure", c["edp"] * MMHG * f, 0.0, 0.0) for c in prm]
        u, _, _, inf = model.solve(u, zero_ta, loads=loads)
        record(-dt * (circ.fill_steps + 1 - i),
               [(c["edp"] * f, _ml(inf["volumes"][k]), c["p_dia"], 0.0, "filling") for k, c in enumerate(prm)],
               u, zero_ta, inf)
    v_ed = [_ml(v) for v in inf["volumes"]]
    log("End-diastole: " + ", ".join(f"{c['name']}EDV {v:.1f} mL at {c['edp']:.0f} mmHg" for c, v in zip(prm, v_ed)))

    # ---------------- beat ----------------
    # a valve that just opened cannot close on the same step: right at opening the solver can return a
    # tiny spurious backflow (the RV opens at a few mmHg, where this is comparable to its flow)
    ch = [Chamber(c, dt, circ.cycle_ms, v, min_ejection_ms=2 * dt) for c, v in zip(prm, v_ed)]
    n_steps = int(round(circ.cycle_ms / dt))
    for step in range(1, n_steps + 1):
        t = step * dt
        ta = model.active_tension(t * t_scale)
        for attempt in range(2 * K + 2):
            u_new, _, _, inf = model.solve(u, ta, loads=[c.fe_load(t) for c in ch])
            ps = [p / MMHG for p in inf["pressures"]]
            vs = [_ml(v) for v in inf["volumes"]]
            if not any([c.transition(t, p, v) for c, p, v in zip(ch, ps, vs)]):
                break
        u = u_new
        rows = [c.commit(p, v) for c, p, v in zip(ch, ps, vs)]
        record(t, rows, u, ta, inf)
        if progress:
            progress(step / n_steps)
        if step % 10 == 0:
            log(f"t={t:4.0f} ms  " + "  ".join(f"{c.name} {c.phase:9s} p={p:6.1f} mmHg V={v:6.1f} mL"
                                               for c, p, v in zip(ch, ps, vs))
                + f"  Ta_max={ta.max() / 1e3:5.1f} kPa  it={inf.get('iterations', 0)}")

    P, V, PA, Q, phases = (tr[0][k] for k in ("pressure_mmhg", "volume_ml", "arterial_mmhg", "flow_ml_s", "phase"))
    v_ed = v_ed[0] * 1000.0
    V_arr, P_arr = np.array(V), np.array(P)
    beat = np.array(times) >= 0
    edv = float(_ml(v_ed))
    esv = float(V_arr[beat].min())
    sv = edv - esv
    metrics = {
        "edv_ml": edv,
        "esv_ml": esv,
        "stroke_volume_ml": sv,
        "ejection_fraction_pct": 100.0 * sv / edv if edv > 0 else 0.0,
        "peak_lv_pressure_mmhg": float(P_arr[beat].max()),
        "end_diastolic_pressure_mmhg": circ.edp,
        "cardiac_output_l_min": sv * 60.0 / circ.cycle_ms,
        "stroke_work_mmhg_ml": float(-np.trapezoid(P_arr[beat], V_arr[beat])) if hasattr(np, "trapezoid")
        else float(-np.trapz(P_arr[beat], V_arr[beat])),
        "peak_aortic_pressure_mmhg": float(np.max(PA)),
        "min_aortic_pressure_mmhg": float(np.min(np.array(PA)[beat])),
    }
    tb = np.array(times)[beat]
    # acute CRT response is classically judged by the change in LV dP/dt max (>= 10 %)
    metrics["dpdt_max_mmhg_s"] = float(np.max(np.diff(P_arr[beat]) / (np.diff(tb) / 1000.0))) if beat.sum() > 1 else 0.0
    ff = np.stack(fields["fiber_strain"])
    metrics["peak_mean_fiber_strain"] = float(ff[beat].mean(1).min())
    metrics["myocardial_volume_change_pct"] = float(100 * (np.stack(fields["jacobian"]).mean(1).min() - 1))
    chambers = {}
    for k in range(1, K):
        c = {key: (np.array(val) if key != "phase" else val) for key, val in tr[k].items()}
        chambers[prm[k]["name"]] = c
        metrics.update(chamber_metrics(np.array(times), c, prm[k]["name"].lower(), circ.cycle_ms, prm[k]["edp"]))
    if "RV" in chambers:
        metrics.update(ventricular_balance(metrics))
    return CycleResult(times=np.array(times), pressure_mmhg=P_arr, volume_ml=V_arr, aortic_mmhg=np.array(PA),
                       flow_ml_s=np.array(Q), phase=phases, displacements=np.stack(U), active_tension_kpa=np.stack(TA),
                       element_fields={k: np.stack(v) for k, v in fields.items()}, metrics=metrics,
                       params={"circulation": asdict(circ), "mechanics": model.config_dict()}, solver_info=info,
                       chambers=chambers)


def chamber_metrics(times, c, prefix, cycle_ms, edp):
    """EDV/ESV/EF/pressures of one ventricle from its traces (beat = t >= 0)."""
    beat = times >= 0
    vol, p, pa = np.asarray(c["volume_ml"]), np.asarray(c["pressure_mmhg"]), np.asarray(c["arterial_mmhg"])
    edv = float(vol[~beat][-1]) if (~beat).any() else float(vol[0])  # end of passive filling
    esv = float(vol[beat].min())
    sv = edv - esv
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return {
        f"{prefix}_edv_ml": edv, f"{prefix}_esv_ml": esv, f"{prefix}_stroke_volume_ml": sv,
        f"{prefix}_ejection_fraction_pct": 100.0 * sv / edv if edv > 0 else 0.0,
        f"peak_{prefix}_pressure_mmhg": float(p[beat].max()), f"{prefix}_end_diastolic_pressure_mmhg": edp,
        f"{prefix}_stroke_work_mmhg_ml": float(-trap(p[beat], vol[beat])),
        f"{prefix}_peak_arterial_pressure_mmhg": float(pa[beat].max()),
        f"{prefix}_min_arterial_pressure_mmhg": float(pa[beat].min()),
    }


def ventricular_balance(metrics):
    """RV/LV indices used clinically (pulmonary artery pressures, RV/LV volume ratio, stroke-volume mismatch)."""
    out = {
        "pa_systolic_mmhg": metrics["rv_peak_arterial_pressure_mmhg"],
        "pa_diastolic_mmhg": metrics["rv_min_arterial_pressure_mmhg"],
        "rv_lv_edv_ratio": metrics["rv_edv_ml"] / max(metrics["edv_ml"], 1e-9),
        # a single beat from separately prescribed EDPs need not balance; large mismatches mean the
        # RV/LV filling pressures or circulations are not consistent for this patient
        "rv_lv_stroke_volume_mismatch_ml": metrics["rv_stroke_volume_ml"] - metrics["stroke_volume_ml"],
    }
    return out
