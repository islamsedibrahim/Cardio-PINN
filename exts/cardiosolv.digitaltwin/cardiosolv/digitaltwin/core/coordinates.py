"""Stage 4b: ventricular coordinates and myofibre architecture.

Coordinates follow the Cardio-PINN parametrisation (Buoso et al. 2021):
``x_t`` transmural (0 endo -> 1 epi, Laplace solve), ``x_l`` longitudinal
(0 apex -> 1 base) and ``x_c`` circumferential (0..1 from the septum), with
local directions ``e_t, e_l, e_c``. Fibres use Cardio-PINN's
``GenerateFibers`` rule (helix angle linear from endo to epi, sheet angle
gamma), vectorised.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from .geometry_layer import GeometryLayer
from .mesh import normalize
from .volume_mesh import TetMesh


def shape_gradients(points, tets):
    """P1 shape-function gradients (E,4,3) and volumes (E,)."""
    p = points[tets]
    D = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]], axis=2)  # columns
    det = np.linalg.det(D)
    Dinv = np.linalg.inv(D)  # rows = gradients of barycentric 1..3
    G = np.empty((len(tets), 4, 3))
    G[:, 1:, :] = Dinv
    G[:, 0, :] = -Dinv.sum(axis=1)
    return G, np.abs(det) / 6.0


def stiffness_matrix(points, tets, conductivity=None):
    """P1 stiffness; ``conductivity`` optional (E,3,3) tensors."""
    G, vol = shape_gradients(points, tets)
    if conductivity is None:
        Ke = np.einsum("eid,ejd->eij", G, G) * vol[:, None, None]
    else:
        Ke = np.einsum("eid,edk,ejk->eij", G, conductivity, G) * vol[:, None, None]
    rows = np.repeat(tets, 4, axis=1).ravel()
    cols = np.tile(tets, (1, 4)).ravel()
    n = len(points)
    return sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(n, n)).tocsr()


def lumped_mass(points, tets):
    _, vol = shape_gradients(points, tets)
    m = np.zeros(len(points))
    np.add.at(m, tets.ravel(), np.repeat(vol / 4.0, 4))
    return m


def solve_laplace(K, n, fixed_idx, fixed_val):
    free = np.setdiff1d(np.arange(n), fixed_idx)
    u = np.zeros(n)
    u[fixed_idx] = fixed_val
    rhs = -K[free][:, fixed_idx] @ u[fixed_idx]
    u[free] = spsolve(K[free][:, free].tocsc(), rhs)
    return u


@dataclass
class VentricularCoordinates:
    x_t: np.ndarray  # nodal transmural 0 endo .. 1 epi
    x_l: np.ndarray  # nodal longitudinal 0 apex .. 1 base
    x_c: np.ndarray  # nodal circumferential 0..1 (0 = septum centre)
    e_t: np.ndarray  # element directions (E,3)
    e_l: np.ndarray
    e_c: np.ndarray
    fiber: np.ndarray  # element fibre f0 (E,3)
    sheet: np.ndarray  # s0
    normal: np.ndarray  # n0
    fiber_nodes: np.ndarray  # nodal fibres (N,3) for visualisation / openCARP
    params: dict = field(default_factory=dict)


def generate_fibers(e_t, e_l, e_c, x_t, endo_angle=60.0, epi_angle=-60.0, gamma_angle=-65.0):
    """Vectorised Cardio-PINN ``GenerateFibers``."""
    e_c = normalize(e_c)
    e_t = normalize(np.cross(e_c, e_l))
    e_l = normalize(e_l - np.einsum("ij,ij->i", e_c, e_l)[:, None] * e_c)
    alpha = np.deg2rad(epi_angle * x_t + endo_angle * (1.0 - x_t))[:, None]
    gamma = np.deg2rad(gamma_angle)
    f = normalize(np.cos(alpha) * e_c + np.sin(alpha) * e_l)
    s = -np.cos(gamma) * e_t + np.sin(gamma) * e_l
    s = normalize(s - np.einsum("ij,ij->i", f, s)[:, None] * f)
    n = normalize(np.cross(f, s))
    return f, s, n


def compute_coordinates(mesh: TetMesh, gl: GeometryLayer, endo_angle=60.0, epi_angle=-60.0,
                        gamma_angle=-65.0) -> VentricularCoordinates:
    n = mesh.n_nodes
    sets = mesh.node_sets()
    endo, epi = sets["endo"], np.setdiff1d(sets["epi"], sets["endo"])
    if len(endo) == 0 or len(epi) == 0:
        raise RuntimeError("Endocardial or epicardial boundary missing on the computational mesh.")
    K = stiffness_matrix(mesh.points, mesh.tets)
    x_t = solve_laplace(K, n, np.concatenate([endo, epi]), np.concatenate([np.zeros(len(endo)), np.ones(len(epi))]))
    x_t = np.clip(x_t, 0, 1)

    ax = gl.long_axis
    d = ax.direction
    x_l = np.clip(ax.project(mesh.points), 0.0, 1.0)
    e1 = gl.septum_direction
    e2 = np.cross(d, e1)
    r = mesh.points - ax.apex
    ang = np.arctan2(r @ e2, r @ e1)
    x_c = (ang / (2 * np.pi)) % 1.0

    G, vol = shape_gradients(mesh.points, mesh.tets)
    grad_t = np.einsum("ei,eid->ed", x_t[mesh.tets], G)
    # smooth element gradients through the nodes (robust near the apex / base rim)
    node_g = np.zeros((n, 3))
    np.add.at(node_g, mesh.tets.ravel(), np.repeat(grad_t * vol[:, None], 4, axis=0))
    e_t = normalize(node_g[mesh.tets].mean(1))
    degenerate = np.linalg.norm(node_g[mesh.tets].mean(1), axis=1) < 1e-12
    centroid = mesh.points[mesh.tets].mean(1)
    radial = centroid - ax.apex - np.outer((centroid - ax.apex) @ d, d)
    e_t[degenerate] = normalize(radial[degenerate])
    e_l = normalize(d[None, :] - (e_t @ d)[:, None] * e_t)
    e_c = normalize(np.cross(e_l, e_t))
    xt_el = x_t[mesh.tets].mean(1)
    f, s, nn = generate_fibers(e_t, e_l, e_c, xt_el, endo_angle, epi_angle, gamma_angle)

    fn = np.zeros((n, 3))
    # sign-consistent nodal averaging of an axial field
    ref = f.copy()
    np.add.at(fn, mesh.tets.ravel(), np.repeat(ref * vol[:, None], 4, axis=0))
    fn = normalize(fn)
    return VentricularCoordinates(x_t=x_t, x_l=x_l, x_c=x_c, e_t=e_t, e_l=e_l, e_c=e_c, fiber=f, sheet=s,
                                  normal=nn, fiber_nodes=fn,
                                  params={"endo_helix_deg": endo_angle, "epi_helix_deg": epi_angle,
                                          "sheet_gamma_deg": gamma_angle})
