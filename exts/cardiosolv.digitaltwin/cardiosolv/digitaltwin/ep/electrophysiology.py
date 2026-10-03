"""Stage 5: cardiac electrophysiology on the user's myocardium.

Two solvers share one interface:

* ``eikonal`` (default, interactive): anisotropic shortest-path activation on
  the tetrahedral edge graph, with a fast endocardial layer standing in for
  the Purkinje network. Transmembrane potential is reconstructed from
  activation/repolarisation times and an action-potential template.
* ``monodomain``: Mitchell-Schaeffer reaction-diffusion with fibre-aligned
  conductivity (semi-implicit FE, factorised once). Needs fine meshes
  (<= 1 mm) for converged conduction velocity; use for research runs.

Both produce activation times that drive the active tension of the
biomechanics stage (electromechanical coupling). openCARP users can export
the same mesh/fibres/stimuli with ``io.opencarp``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from ..core.coordinates import VentricularCoordinates, lumped_mass, shape_gradients, stiffness_matrix
from ..core.volume_mesh import TetMesh

# Pacing protocols in ventricular coordinates: (node set, x_l, x_c, delay ms)
PACING_PROTOCOLS: Dict[str, dict] = {
    "sinus": {
        "description": "Normal sinus rhythm: LV septal + antero/postero-paraseptal Purkinje breakthroughs (Durrer 1970)",
        "sites": [("endo", 0.55, 0.00, 0.0), ("endo", 0.65, 0.15, 2.0), ("endo", 0.65, 0.85, 2.0),
                  ("rv_septum", 0.50, 0.00, 5.0)],
        "purkinje": {"lv": True, "rv": True},
    },
    "lbbb": {
        "description": "Left bundle branch block: activation enters from the RV septum, no LV Purkinje "
                       "(the right bundle and RV Purkinje still conduct)",
        "sites": [("rv_septum", 0.50, 0.00, 0.0)],
        "purkinje": {"lv": False, "rv": True},
    },
    "rv_apical_pacing": {
        "description": "RV apical pacing lead (myocardial capture; the intact right bundle conducts retrogradely)",
        "sites": [("rv_septum", 0.15, 0.00, 0.0)],
        "purkinje": {"lv": False, "rv": True},
    },
    "crt": {
        "description": "Cardiac resynchronisation: RV septal + LV lateral epicardial lead in LBBB",
        "sites": [("rv_septum", 0.45, 0.00, 0.0), ("epi", 0.55, 0.50, 0.0)],
        "purkinje": {"lv": False, "rv": True},
    },
}


@dataclass
class EPConfig:
    solver: str = "eikonal"  # "eikonal" | "monodomain"
    protocol: str = "sinus"
    cv_fiber: float = 0.60  # mm/ms
    cv_cross: float = 0.25  # mm/ms
    purkinje_speedup: float = 4.0  # fast endocardial layer factor
    endo_layer: float = 0.08  # x_t threshold of the fast layer
    apd_endo: float = 290.0  # ms
    apd_epi: float = 250.0  # ms
    apd_apex_base: float = -10.0  # ms difference base vs apex
    site_radius_mm: float = 4.0
    cycle_length: float = 800.0  # ms
    # monodomain
    dt: float = 0.05
    duration: float = 450.0
    output_dt: float = 5.0
    custom_sites: List[tuple] = field(default_factory=list)  # [(x, y, z, delay_ms)] world mm


@dataclass
class EPResult:
    activation_time: np.ndarray  # (N,) ms
    repolarization_time: np.ndarray  # (N,) ms
    apd: np.ndarray
    stim_nodes: np.ndarray
    metrics: Dict[str, float]
    config: dict
    vm_frames: Optional[np.ndarray] = None  # (T,N) mV, monodomain only
    frame_times: Optional[np.ndarray] = None
    ecg_times: Optional[np.ndarray] = None
    ecg: Optional[Dict[str, np.ndarray]] = None

    def vm(self, t: float) -> np.ndarray:
        """Transmembrane potential (mV) at time ``t`` (ms)."""
        if self.vm_frames is not None:
            i = int(np.clip(np.searchsorted(self.frame_times, t), 0, len(self.frame_times) - 1))
            return self.vm_frames[i]
        return ap_template(t - self.activation_time, self.apd)


def ap_template(tau, apd):
    """Ventricular action potential shape (mV) vs time since activation."""
    tau = np.asarray(tau, float)
    up = 1.0 - np.exp(-np.clip(tau, 0, None) / 1.2)
    plateau = 1.0 - 0.15 * np.clip(tau / np.maximum(apd, 1), 0, 1)
    repol = 1.0 / (1.0 + np.exp((tau - apd) / 14.0))
    v = np.where(tau >= 0, up * plateau * repol, 0.0)
    return -85.0 + 110.0 * v


def _site_nodes(mesh: TetMesh, coords: VentricularCoordinates, sets, set_name, x_l, x_c, radius):
    nodes = sets.get(set_name)
    if nodes is None or len(nodes) == 0:
        nodes = sets["endo"] if set_name != "epi" else sets["epi"]
    if set_name == "rv_septum" and coords.node_region is not None:
        septal = nodes[coords.node_region[nodes] == 0]  # RV endocardium on the septum, not the free wall
        nodes = septal if len(septal) else nodes
    dc = np.abs(coords.x_c[nodes] - x_c)
    dc = np.minimum(dc, 1 - dc)
    score = (coords.x_l[nodes] - x_l) ** 2 + dc**2
    centre = nodes[int(np.argmin(score))]
    d = np.linalg.norm(mesh.points[nodes] - mesh.points[centre], axis=1)
    return nodes[d <= radius]


def stimulus_sites(mesh, coords, cfg: EPConfig):
    sets = mesh.node_sets()
    proto = PACING_PROTOCOLS[cfg.protocol]
    nodes, delays = [], []
    for set_name, xl, xc, delay in proto["sites"]:
        sel = _site_nodes(mesh, coords, sets, set_name, xl, xc, cfg.site_radius_mm)
        nodes.append(sel)
        delays.append(np.full(len(sel), delay))
    if cfg.custom_sites:
        tree = cKDTree(mesh.points)
        for x, y, z, delay in cfg.custom_sites:
            sel = tree.query_ball_point([x, y, z], cfg.site_radius_mm) or [tree.query([x, y, z])[1]]
            nodes.append(np.asarray(sel))
            delays.append(np.full(len(sel), delay))
    nodes = np.concatenate(nodes)
    delays = np.concatenate(delays)
    order = np.argsort(delays)
    uniq, first = np.unique(nodes[order], return_index=True)
    return uniq, delays[order][first], proto["purkinje"]


def purkinje_nodes(coords: VentricularCoordinates, cfg: EPConfig, purkinje) -> np.ndarray:
    """Nodes of the fast subendocardial layer(s): ``purkinje`` is a bool or {"lv": bool, "rv": bool}."""
    if isinstance(purkinje, dict):
        lv, rv = bool(purkinje.get("lv")), bool(purkinje.get("rv"))
    else:
        lv = rv = bool(purkinje)
    fast = np.zeros(len(coords.x_t), bool)
    if lv:
        fast |= (coords.x_t if coords.x_t_lv is None else coords.x_t_lv) <= cfg.endo_layer
    if rv and coords.x_t_rv is not None:
        fast |= coords.x_t_rv <= cfg.endo_layer
    return fast


def _edges(tets):
    e = np.concatenate([tets[:, [i, j]] for i in range(4) for j in range(i + 1, 4)])
    e.sort(axis=1)
    return np.unique(e, axis=0)


def eikonal_activation(mesh: TetMesh, coords: VentricularCoordinates, cfg: EPConfig, stim, delays, purkinje):
    e = _edges(mesh.tets)
    d = mesh.points[e[:, 1]] - mesh.points[e[:, 0]]
    f = coords.fiber_nodes[e[:, 0]] + np.sign(np.einsum("ij,ij->i", coords.fiber_nodes[e[:, 0]],
                                                        coords.fiber_nodes[e[:, 1]]))[:, None] * coords.fiber_nodes[e[:, 1]]
    f /= np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)
    dl = np.einsum("ij,ij->i", d, f)
    dt2 = np.maximum(np.einsum("ij,ij->i", d, d) - dl**2, 0)
    w = np.sqrt(dl**2 / cfg.cv_fiber**2 + dt2 / cfg.cv_cross**2)
    fn = purkinje_nodes(coords, cfg, purkinje)
    if fn.any():
        fast = fn[e[:, 0]] & fn[e[:, 1]]
        w[fast] /= cfg.purkinje_speedup
    # Kuhn/structured graphs overestimate straight-line paths by ~6%
    w *= 0.94
    n = mesh.n_nodes
    src = n  # virtual source connected to the stimulus nodes
    rows = np.concatenate([e[:, 0], e[:, 1], np.full(len(stim), src)])
    cols = np.concatenate([e[:, 1], e[:, 0], stim])
    vals = np.concatenate([w, w, delays + 1e-6])
    G = sp.coo_matrix((vals, (rows, cols)), shape=(n + 1, n + 1)).tocsr()
    return dijkstra(G, directed=True, indices=src)[:n]


def apd_map(coords: VentricularCoordinates, cfg: EPConfig):
    return (cfg.apd_endo + (cfg.apd_epi - cfg.apd_endo) * coords.x_t + cfg.apd_apex_base * coords.x_l)


def conductivity_tensors(coords: VentricularCoordinates, d_f, d_t, endo_factor=1.0, fast_nodes=None, tets=None):
    f = coords.fiber
    D = d_t * np.eye(3)[None] + (d_f - d_t) * np.einsum("ei,ej->eij", f, f)
    if endo_factor != 1.0 and tets is not None and fast_nodes is not None:
        endo = fast_nodes[tets].mean(1) >= 0.5
        D[endo] *= endo_factor
    return D


def monodomain(mesh: TetMesh, coords: VentricularCoordinates, cfg: EPConfig, stim, delays, purkinje,
               log=None):
    """Mitchell-Schaeffer monodomain (normalised v in [0,1])."""
    # D from target CV: CV ~ sqrt(D / tau_in) * c for Mitchell-Schaeffer (c ~ 0.47 for v_gate 0.13)
    tau_in, tau_out, tau_open, tau_close, v_gate = 0.3, 6.0, 120.0, 150.0, 0.13
    c = 0.47
    d_f = (cfg.cv_fiber / c) ** 2 * tau_in
    d_t = (cfg.cv_cross / c) ** 2 * tau_in
    D = conductivity_tensors(coords, d_f, d_t, cfg.purkinje_speedup**2, purkinje_nodes(coords, cfg, purkinje),
                             mesh.tets)
    K = stiffness_matrix(mesh.points, mesh.tets, D)
    m = lumped_mass(mesh.points, mesh.tets)
    dt = cfg.dt
    from scipy.sparse.linalg import splu

    A = (sp.diags(m / dt) + 0.5 * K).tocsc()
    solver = splu(A)
    n = mesh.n_nodes
    v = np.zeros(n)
    h = np.ones(n)
    act = np.full(n, np.inf)
    rep = np.full(n, np.inf)
    n_steps = int(cfg.duration / dt)
    every = max(1, int(round(cfg.output_dt / dt)))
    frames, times = [], []
    for k in range(n_steps):
        t = k * dt
        stim_on = (t >= delays) & (t < delays + 2.0)
        jst = np.zeros(n)
        jst[stim[stim_on]] = 0.5
        jin = h * v * v * (1 - v) / tau_in
        jout = -v / tau_out
        v_star = v + dt * (jin + jout + jst)
        h = np.where(v < v_gate, h + dt * (1 - h) / tau_open, h - dt * h / tau_close)
        rhs = (m / dt) * v_star - 0.5 * (K @ v_star)
        v_new = solver.solve(rhs)
        up = (v < 0.5) & (v_new >= 0.5) & np.isinf(act)
        act[up] = t
        down = (v >= 0.5) & (v_new < 0.5) & np.isfinite(act) & np.isinf(rep) & (t - act > 20)
        rep[down] = t
        v = v_new
        if k % every == 0:
            frames.append(-85.0 + 110.0 * np.clip(v, 0, 1.05))
            times.append(t)
        if log and k % 2000 == 0:
            log(f"monodomain t={t:.0f} ms, activated {np.isfinite(act).mean() * 100:.0f}%")
    return act, rep, np.array(frames, np.float32), np.array(times)


def pseudo_ecg(mesh: TetMesh, coords: VentricularCoordinates, result: EPResult, electrodes: Dict[str, np.ndarray],
               dt=2.0):
    """Lead-field pseudo-ECG: phi(x_e) = sum_e V_e (D grad v) . grad(1/r)."""
    G, vol = shape_gradients(mesh.points, mesh.tets)
    D = conductivity_tensors(coords, 1.0, 0.25)
    cen = mesh.points[mesh.tets].mean(1)
    t_end = float(np.nanmax(result.repolarization_time[np.isfinite(result.repolarization_time)])) + 30.0
    times = np.arange(0.0, t_end, dt)
    out = {}
    lead = {}
    for name, x in electrodes.items():
        r = x - cen
        rn = np.linalg.norm(r, axis=1, keepdims=True)
        lead[name] = r / rn**3  # grad of 1/|x - y| w.r.t. y
    for name in electrodes:
        out[name] = np.zeros(len(times))
    for k, t in enumerate(times):
        v = result.vm(t)
        gv = np.einsum("ei,eid->ed", v[mesh.tets], G)
        flux = np.einsum("eij,ej->ei", D, gv) * vol[:, None]
        for name in electrodes:
            out[name][k] = float(np.einsum("ei,ei->", flux, lead[name]))
    return times, out


def run_electrophysiology(mesh: TetMesh, coords: VentricularCoordinates, cfg: EPConfig, long_axis=None,
                          septum_direction=None, log=print) -> EPResult:
    if cfg.protocol not in PACING_PROTOCOLS:
        raise ValueError(f"Unknown pacing protocol {cfg.protocol}; choose from {list(PACING_PROTOCOLS)}")
    stim, delays, purkinje = stimulus_sites(mesh, coords, cfg)
    apd = apd_map(coords, cfg)
    frames = times = None
    if cfg.solver == "eikonal":
        act = eikonal_activation(mesh, coords, cfg, stim, delays, purkinje)
        rep = act + apd
    elif cfg.solver == "monodomain":
        act, rep, frames, times = monodomain(mesh, coords, cfg, stim, delays, purkinje, log)
        fill = ~np.isfinite(rep) & np.isfinite(act)
        rep[fill] = act[fill] + apd[fill]
    else:
        raise ValueError(f"Unknown EP solver {cfg.solver}")
    finite = np.isfinite(act)
    if not finite.all():
        log(f"EP warning: {(~finite).sum()} nodes never activated")
        act[~finite] = np.nanmax(act[finite]) if finite.any() else 0.0
        rep[~np.isfinite(rep)] = act[~np.isfinite(rep)] + apd[~np.isfinite(rep)]
    lv = np.ones(len(act), bool) if coords.node_region is None else coords.node_region == 0
    lateral = lv & (np.abs(coords.x_c - 0.5) < 0.12) & (coords.x_l > 0.3) & (coords.x_l < 0.8)
    septal = lv & (np.minimum(coords.x_c, 1 - coords.x_c) < 0.08) & (coords.x_l > 0.3) & (coords.x_l < 0.8)
    metrics = {
        "total_activation_time_ms": float(act.max() - act.min()),
        "qrs_duration_ms": float(np.percentile(act, 98) - act.min()),
        "mean_activation_ms": float(act.mean()),
        "septal_to_lateral_delay_ms": float(act[lateral].mean() - act[septal].mean()) if lateral.any() and septal.any() else 0.0,
        "mean_apd_ms": float((rep - act).mean()),
        "stimulus_nodes": int(len(stim)),
    }
    if coords.node_region is not None and (coords.node_region == 1).any():
        rv = coords.node_region == 1
        metrics.update({
            "lv_total_activation_ms": float(act[lv].max() - act.min()),
            "rv_total_activation_ms": float(act[rv].max() - act.min()),
            # mean LV free-wall minus mean RV free-wall activation: >0 = LV late (LBBB), <0 = RV late (RBBB)
            "interventricular_delay_ms": float(act[lateral].mean() - act[rv].mean()) if lateral.any() else 0.0,
        })
    res = EPResult(activation_time=act, repolarization_time=rep, apd=rep - act, stim_nodes=stim, metrics=metrics,
                   config=asdict(cfg), vm_frames=frames, frame_times=times)
    if long_axis is not None:
        c = mesh.points.mean(0)
        L = 4.0 * long_axis.length_mm
        sdir = septum_direction if septum_direction is not None else np.cross(long_axis.direction, [0, 0, 1.0])
        electrodes = {
            "apex_base": c - long_axis.direction * L,  # lead looking from the apex (lead II-like)
            "lateral": c - sdir * L,  # lateral (V5/V6-like)
        }
        res.ecg_times, res.ecg = pseudo_ecg(mesh, coords, res, electrodes)
    return res
