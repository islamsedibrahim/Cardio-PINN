"""Myocardial scar substrate (from LGE MR, or drawn by the user) on the computational mesh.

Scar enters the twin as closed USD meshes next to (or under) the heart, named
like the outputs of the imaging package: ``scar_core`` / ``myocardial_scar`` /
``infarct`` for dense scar and ``scar_border_zone`` / ``grey_zone`` for the
border zone. Any closed mesh works, so a clinician can also draw a scar in
Omniverse. Every tet is labelled healthy (0), border zone (1) or core (2):

* EP: core is unexcitable (no conduction through it), border zone conducts
  slowly (``bz_cv_factor``) with a longer action potential;
* mechanics: core produces no active tension and is stiffer, border zone has
  reduced contractility and moderate stiffening;
* substrate metrics: scar burden, transmurality, and conduction channels
  (border-zone corridors through dense scar that connect two separate patches
  of healthy tissue, the substrate of scar-related re-entrant VT).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from .mesh import SurfaceMesh
from .voxel import VoxelGrid, voxelize

HEALTHY, BORDER_ZONE, CORE = 0, 1, 2
SCAR_NAMES = {BORDER_ZONE: "border_zone", CORE: "scar_core"}


def classify_scar_name(name: str) -> Optional[int]:
    """Scar class from a prim name, or None if the name is not a scar."""
    n = name.lower()
    if any(k in n for k in ("border_zone", "borderzone", "border zone", "grey_zone", "gray_zone", "greyzone",
                            "grayzone", "peri_infarct", "periinfarct")) or n.endswith("_bz"):
        return BORDER_ZONE
    if any(k in n for k in ("scar", "infarct", "fibrosis", "lge_core", "dense_core")):
        return CORE
    return None


@dataclass
class ScarParams:
    bz_cv_factor: float = 0.4  # border-zone conduction velocity / healthy
    bz_apd_factor: float = 1.15  # border-zone APD prolongation
    bz_contractility: float = 0.5  # border-zone active tension / healthy
    bz_stiffness: float = 2.0  # border-zone passive stiffness multiplier
    core_stiffness: float = 5.0  # dense scar passive stiffness multiplier
    min_channel_nodes: int = 3
    min_channel_length_mm: float = 8.0
    max_channel_half_width_mm: float = 5.0  # corridors up to ~10 mm wide
    min_detour_ratio: float = 1.5  # an isthmus must be a shortcut through the scar
    min_detour_mm: float = 10.0


@dataclass
class ScarModel:
    tet_label: np.ndarray  # (E,) HEALTHY / BORDER_ZONE / CORE
    node_core: np.ndarray  # (N,) inside dense scar (all incident tets core): unexcitable
    node_bz: np.ndarray  # (N,) slow-conducting (border zone, or the rim of the core)
    params: ScarParams = field(default_factory=ScarParams)
    sources: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    channels: List[dict] = field(default_factory=list)

    @property
    def any(self) -> bool:
        return bool((self.tet_label > 0).any())

    def contractility(self) -> np.ndarray:
        return np.select([self.tet_label == CORE, self.tet_label == BORDER_ZONE],
                         [0.0, self.params.bz_contractility], 1.0)

    def stiffness(self) -> np.ndarray:
        return np.select([self.tet_label == CORE, self.tet_label == BORDER_ZONE],
                         [self.params.core_stiffness, self.params.bz_stiffness], 1.0)

    def node_label(self) -> np.ndarray:
        lab = np.zeros(len(self.node_core), np.int8)
        lab[self.node_bz] = BORDER_ZONE
        lab[self.node_core] = CORE
        return lab

    def to_dict(self):
        return {"sources": self.sources, "params": self.params.__dict__, "metrics": self.metrics,
                "channels": self.channels}


def label_tets(mesh, surfaces: Dict[int, List[SurfaceMesh]], spacing: Optional[float] = None) -> np.ndarray:
    """Tet labels from closed scar surfaces (sim mm). A tet takes the most severe class that contains
    the majority of its 5 sample points (centroid + 4 points halfway to the vertices)."""
    pts = mesh.points[mesh.tets]  # (E,4,3)
    cen = pts.mean(1)
    samples = np.concatenate([cen[:, None], 0.5 * (pts + cen[:, None])], 1)  # (E,5,3)
    h = spacing or min(1.0, 0.5 * mesh.spacing)
    lab = np.zeros(mesh.n_tets, np.int8)
    for code in (BORDER_ZONE, CORE):
        for surf in surfaces.get(code, []):
            lo = np.maximum(surf.bbox_min, mesh.points.min(0)) - 2 * h
            hi = np.minimum(surf.bbox_max, mesh.points.max(0)) + 2 * h
            if np.any(hi <= lo):
                continue
            grid = VoxelGrid.around(lo, hi, h, padding_voxels=2)
            inside = voxelize(surf, grid)
            votes = grid.sample(inside.astype(np.int8), samples.reshape(-1, 3), fill=0).reshape(-1, 5).sum(1)
            lab[votes >= 3] = np.maximum(lab[votes >= 3], code)
    return lab


def node_graph(mesh):
    e = np.concatenate([mesh.tets[:, [i, j]] for i in range(4) for j in range(i + 1, 4)])
    e = np.unique(np.sort(e, axis=1), axis=0)
    n = mesh.n_nodes
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    return (A + A.T).tocsr(), e


def build_scar_model(mesh, coords, tet_label: np.ndarray, params: ScarParams = None, sources=None) -> ScarModel:
    params = params or ScarParams()
    n = mesh.n_nodes
    any_scar = np.zeros(n, bool)
    all_core = np.ones(n, bool)
    used = np.zeros(n, bool)
    for k in range(4):
        t = mesh.tets[:, k]
        used[t] = True
        np.logical_or.at(any_scar, t, tet_label > 0)
        np.logical_and.at(all_core, t, tet_label == CORE)
    node_core = all_core & used
    node_bz = any_scar & ~node_core
    model = ScarModel(tet_label=np.asarray(tet_label, np.int8), node_core=node_core, node_bz=node_bz, params=params,
                      sources=list(sources or []))
    model.metrics = scar_metrics(mesh, coords, model)
    model.channels = conduction_channels(mesh, model)
    model.metrics["conduction_channels"] = len(model.channels)
    return model


def scar_metrics(mesh, coords, scar: ScarModel) -> Dict[str, float]:
    vol = np.abs(mesh.tet_volumes())
    lv = np.ones(mesh.n_tets, bool) if mesh.tet_region is None else mesh.tet_region == 0
    lv_vol = vol[lv].sum()
    core, bz = scar.tet_label == CORE, scar.tet_label == BORDER_ZONE
    m = {
        "core_volume_ml": float(vol[core].sum() / 1000), "border_zone_volume_ml": float(vol[bz].sum() / 1000),
        "lv_scar_burden_pct": float(100 * vol[lv & (core | bz)].sum() / max(lv_vol, 1e-9)),
        "lv_core_burden_pct": float(100 * vol[lv & core].sum() / max(lv_vol, 1e-9)),
        "rv_scar_volume_ml": float(vol[~lv & (core | bz)].sum() / 1000),
    }
    # transmurality: per (longitudinal, circumferential) sector of the LV, fraction of the wall
    # thickness (x_t bins) containing core
    xl = coords.x_l[mesh.tets].mean(1)
    xc = coords.x_c[mesh.tets].mean(1)
    xt = coords.x_t[mesh.tets].mean(1)
    il = np.clip((xl * 6).astype(int), 0, 5)
    ic = np.clip((xc * 12).astype(int), 0, 11)
    it = np.clip((xt * 5).astype(int), 0, 4)
    sec = il * 12 + ic
    has = np.zeros((72, 5), bool)
    cov = np.zeros((72, 5), bool)
    np.logical_or.at(has, (sec[lv], it[lv]), True)
    np.logical_or.at(cov, (sec[lv & core], it[lv & core]), True)
    trans = cov.sum(1) / np.maximum(has.sum(1), 1)
    scarred = cov.any(1)
    m["max_transmurality_pct"] = float(100 * trans.max()) if scarred.any() else 0.0
    m["transmural_sectors"] = int((trans >= 0.75).sum())
    for name, lo, hi in (("basal", 2 / 3, 1.01), ("mid", 1 / 3, 2 / 3), ("apical", -0.01, 1 / 3)):
        sel = lv & (xl >= lo) & (xl < hi)
        m[f"{name}_scar_pct"] = float(100 * vol[sel & (core | bz)].sum() / max(vol[sel].sum(), 1e-9))
    septal = np.minimum(xc, 1 - xc) < 0.15
    m["septal_scar_pct"] = float(100 * vol[lv & septal & (core | bz)].sum() / max(vol[lv & septal].sum(), 1e-9))
    lateral = np.abs(xc - 0.5) < 0.2
    m["lateral_scar_pct"] = float(100 * vol[lv & lateral & (core | bz)].sum() / max(vol[lv & lateral].sum(), 1e-9))
    return m


def conduction_channels(mesh, scar: ScarModel) -> List[dict]:
    """Border-zone corridors through dense scar that open into healthy tissue at >= 2 separate places.

    Corridors are the gaps of the dense core: nodes that a morphological closing of the core (radius
    ``max_channel_half_width_mm``) fills but that are not core themselves. Gaps narrower than twice the
    radius are found, while the core's outer border is not expanded, so the border-zone rim around a
    scar is not mistaken for a channel. A candidate is a channel only if it is a functional shortcut
    (isthmus): without it, the shortest path through excitable tissue between its two openings is at
    least ``min_detour_ratio`` times (and ``min_detour_mm`` longer than) the path through it. Notches in
    a jagged core boundary fail this test."""
    from scipy.sparse.csgraph import dijkstra
    from scipy.spatial import cKDTree

    p = scar.params
    if not scar.node_core.any():
        return []
    A, _ = node_graph(mesh)
    X = mesh.points
    r = max(p.max_channel_half_width_mm, 0.75 * mesh.spacing)
    core_tree = cKDTree(X[scar.node_core])
    dilated = core_tree.query(X, distance_upper_bound=r)[0] <= r
    outside = ~dilated
    if outside.any():
        d_out = cKDTree(X[outside]).query(X, distance_upper_bound=r * 1.0001)[0]
        closed = dilated & (d_out > r)
    else:
        closed = dilated
    corridor = closed & ~scar.node_core
    if not corridor.any():
        return []
    sub = A[corridor][:, corridor]
    nc, comp = connected_components(sub, directed=False)
    cidx = np.nonzero(corridor)[0]
    _, edges = node_graph(mesh)
    elen = np.linalg.norm(X[edges[:, 0]] - X[edges[:, 1]], axis=1)
    n = mesh.n_nodes

    def path_len(allowed, src, dst):
        ok = allowed[edges[:, 0]] & allowed[edges[:, 1]]
        W = sp.coo_matrix((elen[ok], (edges[ok, 0], edges[ok, 1])), shape=(n, n)).tocsr()
        d = dijkstra(W, directed=False, indices=src, min_only=True)
        return float(d[dst].min())

    excitable = ~scar.node_core
    out = []
    for c in range(nc):
        nodes = cidx[comp == c]
        if len(nodes) < p.min_channel_nodes:
            continue
        contact = np.unique(A[nodes].indices)
        contact = contact[~corridor[contact] & ~scar.node_core[contact]]
        if len(contact) < 2:
            continue
        ncont, cc = connected_components(A[contact][:, contact], directed=False)
        patches = [contact[cc == k] for k in range(ncont)]
        patches = [q for q in patches if len(q) >= 2]
        if len(patches) < 2:
            continue
        cents = np.array([X[q].mean(0) for q in patches])
        d = np.linalg.norm(cents[:, None] - cents[None], axis=2)
        i, j = np.unravel_index(np.argmax(d), d.shape)
        if d[i, j] < p.min_channel_length_mm:
            continue
        through = path_len(excitable, patches[i], patches[j])
        blocked = excitable.copy()
        blocked[nodes] = False
        around = path_len(blocked, patches[i], patches[j])
        if not (around >= p.min_detour_ratio * through and around - through >= p.min_detour_mm):
            continue
        out.append({"nodes": nodes, "entrance_nodes": patches[i], "exit_nodes": patches[j],
                    "length_mm": float(d[i, j]), "n_openings": len(patches), "path_through_mm": through,
                    "path_around_mm": around if np.isfinite(around) else None,
                    "border_zone_fraction": float(scar.node_bz[nodes].mean()),
                    "centre_mm": X[nodes].mean(0).tolist()})
    return out


def channel_report(channels, activation=None):
    """JSON-friendly channel summary; with an activation map, the transit time through each corridor."""
    rep = []
    for ch in channels:
        r = {k: v for k, v in ch.items() if k not in ("nodes", "entrance_nodes", "exit_nodes")}
        r["n_nodes"] = int(len(ch["nodes"]))
        if activation is not None:
            a_in = float(np.mean(activation[ch["entrance_nodes"]]))
            a_out = float(np.mean(activation[ch["exit_nodes"]]))
            r["transit_time_ms"] = abs(a_out - a_in)
            r["apparent_cv_m_s"] = ch["length_mm"] / max(abs(a_out - a_in), 1e-6)
        rep.append(r)
    return rep
