"""Stage 8a: transfer simulation fields from the computational mesh back to
the user's own USD meshes (the twin is painted on *their* model)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .volume_mesh import TetMesh


@dataclass
class PointMap:
    """Sparse interpolation from tet-mesh nodes to target points.

    ``value_at_targets = sum_k weights[:, k] * nodal[indices[:, k]]``;
    ``falloff`` (0..1) attenuates displacements for points away from the
    myocardium (e.g. atria, great vessels) so neighbouring anatomy follows the
    ventricle smoothly instead of rigidly.
    """

    indices: np.ndarray  # (P,4)
    weights: np.ndarray  # (P,4)
    distance: np.ndarray  # (P,) distance to the myocardium (0 inside)
    falloff: np.ndarray  # (P,)

    def apply(self, nodal: np.ndarray) -> np.ndarray:
        nodal = np.asarray(nodal)
        if nodal.ndim == 1:
            return (nodal[self.indices] * self.weights).sum(1)
        return np.einsum("pk,pkd->pd", self.weights, nodal[self.indices])

    def apply_displacement(self, nodal_u: np.ndarray) -> np.ndarray:
        return self.apply(nodal_u) * self.falloff[:, None]


def build_point_map(mesh: TetMesh, targets: np.ndarray, falloff_mm=20.0, k_candidates=12) -> PointMap:
    pts = mesh.points
    tets = mesh.tets
    cen = pts[tets].mean(1)
    tree = cKDTree(cen)
    targets = np.asarray(targets, float)
    P = len(targets)
    idx = np.zeros((P, 4), np.int64)
    w = np.zeros((P, 4))
    found = np.zeros(P, bool)
    _, cand = tree.query(targets, k=min(k_candidates, len(tets)))
    cand = np.atleast_2d(cand)
    # barycentric search among candidate tets
    a = pts[tets[:, 0]]
    T = np.stack([pts[tets[:, 1]] - a, pts[tets[:, 2]] - a, pts[tets[:, 3]] - a], axis=2)
    Tinv = np.linalg.inv(T)
    for j in range(cand.shape[1]):
        c = cand[:, j]
        todo = ~found
        if not todo.any():
            break
        lam = np.einsum("pij,pj->pi", Tinv[c[todo]], targets[todo] - a[c[todo]])
        bary = np.c_[1 - lam.sum(1), lam]
        inside = np.all(bary >= -1e-6, axis=1)
        sel = np.nonzero(todo)[0][inside]
        idx[sel] = tets[c[todo][inside]]
        w[sel] = bary[inside]
        found[sel] = True
    dist = np.zeros(P)
    if (~found).any():
        ntree = cKDTree(pts)
        d, nn = ntree.query(targets[~found], k=4)
        inv = 1.0 / np.maximum(d, 1e-6) ** 2
        idx[~found] = nn
        w[~found] = inv / inv.sum(1, keepdims=True)
        dist[~found] = d[:, 0]
    fall = np.exp(-((np.maximum(dist - 0.5 * mesh.spacing, 0) / max(falloff_mm, 1e-6)) ** 2))
    return PointMap(indices=idx, weights=w, distance=dist, falloff=fall)


def element_to_nodal(mesh: TetMesh, values: np.ndarray) -> np.ndarray:
    """Volume-weighted average of element values at nodes. ``values`` (..., E)."""
    vol = np.abs(mesh.tet_volumes())
    num = np.zeros(values.shape[:-1] + (mesh.n_nodes,))
    den = np.zeros(mesh.n_nodes)
    np.add.at(den, mesh.tets.ravel(), np.repeat(vol, 4))
    flat = values.reshape(-1, values.shape[-1])
    out = num.reshape(-1, mesh.n_nodes)
    for i in range(flat.shape[0]):
        np.add.at(out[i], mesh.tets.ravel(), np.repeat(flat[i] * vol, 4))
    return (out / np.maximum(den, 1e-30)).reshape(num.shape)
