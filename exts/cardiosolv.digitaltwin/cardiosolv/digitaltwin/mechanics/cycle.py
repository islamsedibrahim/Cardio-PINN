"""Stage 6b: one heartbeat coupled to a 3-element Windkessel (Cardio-PINN loop).

Phases: passive filling to EDP -> isovolumetric contraction -> ejection ->
isovolumetric relaxation -> filling. Implicit Euler on the Windkessel makes
LV pressure a linear function of cavity volume during ejection,
``p = alpha (V* - V)``, so every phase is a single energy minimisation:

* filling:            -p V           (prescribed atrial / filling pressure)
* isovolumetric:      alpha_iso/2 (V* - V)^2 with V* updated so V = V_ref
* ejection:           alpha_wk/2  (V* - V)^2 from the Windkessel state
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List

import numpy as np

from .model import MMHG, MechanicsModel


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

    def frame_count(self):
        return len(self.times)


def _ml(v_mm3):
    return v_mm3 / 1000.0


def simulate_cycle(model: MechanicsModel, circ: CirculationParams = None, t_scale=1.0, log=print,
                   progress=None) -> CycleResult:
    circ = circ or CirculationParams()
    dt = circ.dt_ms
    n = model.mesh.n_nodes
    u = np.zeros((n, 3))
    alpha_iso = 200.0  # Pa / mm^3

    times, P, V, PA, Q, phases, U, TA, info = [], [], [], [], [], [], [], [], []
    fields: Dict[str, list] = {}

    def record(t, p_pa, v, pao, q, ph, u_, ta, inf):
        times.append(t)
        P.append(p_pa / MMHG)
        V.append(_ml(v))
        PA.append(pao)
        Q.append(q)
        phases.append(ph)
        U.append(u_.astype(np.float32))
        TA.append((ta / 1e3).astype(np.float32))
        for k, val in model.element_fields(u_, ta).items():
            fields.setdefault(k, []).append(val.astype(np.float32))
        info.append(inf)

    zero_ta = np.zeros(model.mesh.n_tets)
    record(-dt * (circ.fill_steps + 1), 0.0, model.V0, circ.p_aortic_diastolic, 0.0, "unloaded", u, zero_ta, {})

    # ---------------- passive filling to EDP ----------------
    for k in range(1, circ.fill_steps + 1):
        p = circ.edp * MMHG * k / circ.fill_steps
        u, v, p_out, inf = model.solve(u, zero_ta, "pressure", p_pa=p)
        record(-dt * (circ.fill_steps + 1 - k), p, v, circ.p_aortic_diastolic, 0.0, "filling", u, zero_ta, inf)
    v_ed = v
    log(f"End-diastole: EDV {_ml(v_ed):.1f} mL at {circ.edp:.0f} mmHg")

    # ---------------- beat ----------------
    p_c = circ.p_aortic_diastolic  # Windkessel capacitor pressure (mmHg)
    phase = "ivc"
    v_ref = v_ed
    p_lv = circ.edp * MMHG
    v_prev = v_ed
    p_fill_start = None
    t_fill_start = None
    n_steps = int(round(circ.cycle_ms / dt))
    beta = 1.0 / (1.0 + dt / 1000.0 / (circ.r_periph * circ.c_art))
    for k in range(1, n_steps + 1):
        t = k * dt
        ta = model.active_tension(t * t_scale)
        for attempt in range(3):
            if phase in ("ivc", "ivr"):
                v_star = v_ref + p_lv / alpha_iso  # augmented-Lagrangian shift: V -> v_ref
                u_new, v, p_new, inf = model.solve(u, ta, "volume", alpha=alpha_iso, v_star=v_star)
                q = 0.0
                p_c_new = beta * p_c
                if phase == "ivc" and p_new / MMHG >= p_c_new:
                    phase = "ejection"
                    continue
                if phase == "ivr" and p_new / MMHG <= circ.p_atrial:
                    phase = "filling"
                    # mitral opening: early-diastolic LV pressure never drops below ~1 mmHg here
                    p_fill_start, t_fill_start = max(p_new, 1.0 * MMHG), t
                    continue
            elif phase == "ejection":
                # alpha, V* in Pa and mm^3 from the implicit 3-element Windkessel
                a_mmhg_ml = beta / circ.c_art + circ.z_char / (dt / 1000.0)
                alpha = a_mmhg_ml * MMHG / 1000.0  # Pa / mm^3
                v_star = v_prev + beta * p_c * MMHG / alpha
                u_new, v, p_new, inf = model.solve(u, ta, "volume", alpha=alpha, v_star=v_star)
                q = (v_prev - v) / 1000.0 / (dt / 1000.0)  # mL/s
                if q < 0 and k > 1:
                    phase = "ivr"
                    v_ref = v_prev
                    continue
                p_c_new = beta * (p_c + q * (dt / 1000.0) / circ.c_art)
            else:  # filling: pressure rises from the opening pressure to EDP by end of cycle
                frac = (t - t_fill_start) / max(circ.cycle_ms - t_fill_start, dt)
                p_target = (p_fill_start / MMHG + (circ.edp - p_fill_start / MMHG) * min(frac, 1.0)) * MMHG
                u_new, v, p_new, inf = model.solve(u, ta, "pressure", p_pa=p_target)
                q = 0.0
                p_c_new = beta * p_c
            break
        u, p_lv, p_c = u_new, p_new, p_c_new
        if phase in ("ivc", "ivr"):
            v_ref = v_ref  # unchanged
        record(t, p_lv, v, p_c if phase != "ejection" else p_lv / MMHG, q, phase, u, ta, inf)
        v_prev = v
        if progress:
            progress(k / n_steps)
        if k % 10 == 0:
            log(f"t={t:4.0f} ms  {phase:9s} p={p_lv / MMHG:6.1f} mmHg  V={_ml(v):6.1f} mL  "
                f"Ta_max={ta.max() / 1e3:5.1f} kPa  it={inf.get('iterations', 0)}")

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
    ff = np.stack(fields["fiber_strain"])
    metrics["peak_mean_fiber_strain"] = float(ff[beat].mean(1).min())
    metrics["myocardial_volume_change_pct"] = float(100 * (np.stack(fields["jacobian"]).mean(1).min() - 1))
    return CycleResult(times=np.array(times), pressure_mmhg=P_arr, volume_ml=V_arr, aortic_mmhg=np.array(PA),
                       flow_ml_s=np.array(Q), phase=phases, displacements=np.stack(U), active_tension_kpa=np.stack(TA),
                       element_fields={k: np.stack(v) for k, v in fields.items()}, metrics=metrics,
                       params={"circulation": asdict(circ), "mechanics": model.config_dict()}, solver_info=info)
