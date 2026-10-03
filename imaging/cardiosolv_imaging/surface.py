"""Watertight surfaces from the fused label map (Gaussian-smoothed marching cubes + Taubin)."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
import scipy.sparse as sp


def _adjacency(faces, n):
    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2], faces[:, 1], faces[:, 2], faces[:, 0]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0], faces[:, 0], faces[:, 1], faces[:, 2]])
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    return A


def taubin(points, faces, iterations=15, lam=0.5, mu=-0.53):
    A = _adjacency(faces, len(points))
    deg = np.maximum(np.asarray(A.sum(1)).ravel(), 1)
    p = points.copy()
    for it in range(2 * iterations):
        f = lam if it % 2 == 0 else mu
        p = p + f * ((A @ p) / deg[:, None] - p)
    return p


def signed_volume(points, faces):
    a, b, c = points[faces[:, 0]], points[faces[:, 1]], points[faces[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def mask_to_surface(mask: np.ndarray, affine: np.ndarray, sigma_mm=0.8, smooth_iter=15, step=1):
    """Return (points in RAS mm, triangle faces) for a binary mask on a NIfTI grid."""
    from skimage.measure import marching_cubes

    spacing = np.sqrt((affine[:3, :3] ** 2).sum(0))
    vol = ndimage.gaussian_filter(np.pad(mask, 2).astype(np.float32), sigma=sigma_mm / spacing)
    if vol.max() < 0.5:
        return None, None
    verts, faces, _, _ = marching_cubes(vol, 0.5, step_size=step)
    verts -= 2.0  # padding
    pts = (np.c_[verts, np.ones(len(verts))] @ affine.T)[:, :3]
    faces = faces.astype(np.int64)
    if smooth_iter:
        pts = taubin(pts, faces, smooth_iter)
    if signed_volume(pts, faces) < 0:  # outward normals
        faces = faces[:, ::-1].copy()
    return pts, faces


def topology(faces):
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e.sort(axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    return {"boundary_edges": int((cnt == 1).sum()), "non_manifold_edges": int((cnt > 2).sum()),
            "watertight": bool((cnt == 2).all())}
