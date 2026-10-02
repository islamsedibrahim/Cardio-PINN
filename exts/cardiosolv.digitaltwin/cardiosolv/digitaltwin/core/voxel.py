"""Voxel occupancy of arbitrary (possibly imperfect) USD surfaces.

The geometry layer reasons about anatomy in a common voxel grid because
real-world heart assets are rarely clean: parts overlap, meshes are not
watertight, and chambers are modelled as separate shells. Scan-line parity
voxelisation along three axes with a majority vote tolerates small holes and
non-manifold edges.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .mesh import SurfaceMesh


@dataclass
class VoxelGrid:
    origin: np.ndarray  # world position of the corner of voxel (0,0,0), mm
    spacing: float  # isotropic voxel size, mm
    shape: tuple  # (nx, ny, nz)

    @classmethod
    def around(cls, bbox_min, bbox_max, spacing, padding_voxels=3):
        bbox_min = np.asarray(bbox_min, float) - padding_voxels * spacing
        bbox_max = np.asarray(bbox_max, float) + padding_voxels * spacing
        shape = tuple(int(s) for s in np.ceil((bbox_max - bbox_min) / spacing).astype(int) + 1)
        return cls(origin=bbox_min, spacing=float(spacing), shape=shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    def centers(self, idx: np.ndarray) -> np.ndarray:
        """World centres of voxel indices (K,3)."""
        return self.origin + (np.asarray(idx, float) + 0.5) * self.spacing

    def corner(self, idx: np.ndarray) -> np.ndarray:
        """World position of voxel corner lattice points (K,3)."""
        return self.origin + np.asarray(idx, float) * self.spacing

    def index_of(self, points: np.ndarray) -> np.ndarray:
        return np.floor((np.asarray(points, float) - self.origin) / self.spacing).astype(np.int64)

    def in_bounds(self, idx: np.ndarray) -> np.ndarray:
        idx = np.asarray(idx)
        return np.all((idx >= 0) & (idx < np.asarray(self.shape)), axis=-1)

    def sample(self, volume: np.ndarray, points: np.ndarray, fill=0):
        idx = self.index_of(points)
        ok = self.in_bounds(idx)
        out = np.full(len(idx), fill, dtype=volume.dtype)
        out[ok] = volume[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
        return out


def _parity_along_axis(mesh: SurfaceMesh, grid: VoxelGrid, axis: int, chunk=2_000_000):
    """Inside/outside by ray parity along ``axis`` through every voxel column."""
    perm = [axis, (axis + 1) % 3, (axis + 2) % 3]
    pts = mesh.points[:, perm]
    origin = grid.origin[perm]
    shape = tuple(np.asarray(grid.shape)[perm])
    h = grid.spacing
    # Jitter column positions so rays never pass exactly through vertices/edges.
    jitter = np.array([0.0, 1.234567e-4, 2.718281e-4]) * h

    tri = pts[mesh.faces]  # (M,3,3)
    ymin, ymax = tri[:, :, 1].min(1), tri[:, :, 1].max(1)
    zmin, zmax = tri[:, :, 2].min(1), tri[:, :, 2].max(1)
    j0 = np.ceil((ymin - origin[1] - jitter[1]) / h - 0.5).astype(np.int64)
    j1 = np.floor((ymax - origin[1] - jitter[1]) / h - 0.5).astype(np.int64)
    k0 = np.ceil((zmin - origin[2] - jitter[2]) / h - 0.5).astype(np.int64)
    k1 = np.floor((zmax - origin[2] - jitter[2]) / h - 0.5).astype(np.int64)
    j0, k0 = np.maximum(j0, 0), np.maximum(k0, 0)
    j1, k1 = np.minimum(j1, shape[1] - 1), np.minimum(k1, shape[2] - 1)
    nj = np.clip(j1 - j0 + 1, 0, None)
    nk = np.clip(k1 - k0 + 1, 0, None)
    count = nj * nk

    toggle = np.zeros(shape, dtype=np.int32)
    tri_ids = np.nonzero(count)[0]
    # process in chunks of triangles to bound memory
    csum = np.cumsum(count[tri_ids])
    start = 0
    while start < len(tri_ids):
        base = csum[start - 1] if start > 0 else 0
        stop = int(np.searchsorted(csum, base + chunk, side="right"))
        stop = max(stop, start + 1)
        ids = tri_ids[start:stop]
        cnt = count[ids]
        t = np.repeat(ids, cnt)
        offs = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        local = np.arange(cnt.sum()) - np.repeat(offs, cnt)
        jj = j0[t] + local // nk[t]
        kk = k0[t] + local % nk[t]
        py = origin[1] + (jj + 0.5) * h + jitter[1]
        pz = origin[2] + (kk + 0.5) * h + jitter[2]

        a, b, c = tri[t, 0], tri[t, 1], tri[t, 2]
        # 2D barycentric coordinates in the (y,z) plane
        d = (b[:, 1] - a[:, 1]) * (c[:, 2] - a[:, 2]) - (c[:, 1] - a[:, 1]) * (b[:, 2] - a[:, 2])
        nz_ = np.abs(d) > 1e-18
        d = np.where(nz_, d, 1.0)
        u = ((b[:, 1] - py) * (c[:, 2] - pz) - (c[:, 1] - py) * (b[:, 2] - pz)) / d
        v = ((c[:, 1] - py) * (a[:, 2] - pz) - (a[:, 1] - py) * (c[:, 2] - pz)) / d
        w = 1.0 - u - v
        hit = nz_ & (u >= 0) & (v >= 0) & (w >= 0)
        x = u * a[:, 0] + v * b[:, 0] + w * c[:, 0]
        ii = np.ceil((x - origin[0]) / h - 0.5).astype(np.int64)
        ii = np.maximum(ii, 0)
        keep = hit & (ii < shape[0])
        np.add.at(toggle, (ii[keep], jj[keep], kk[keep]), 1)
        start = stop

    inside = (np.cumsum(toggle, axis=0) % 2).astype(bool)
    # undo permutation
    inv = np.argsort(perm)
    return np.transpose(inside, inv)


def voxelize(mesh: SurfaceMesh, grid: VoxelGrid, robust=True) -> np.ndarray:
    """Boolean occupancy of ``mesh`` in ``grid``.

    ``robust=True`` votes over three ray directions (2 of 3), which recovers
    the interior of meshes with small holes or cracks.
    """
    if mesh.n_faces == 0:
        return np.zeros(grid.shape, dtype=bool)
    if not robust:
        return _parity_along_axis(mesh, grid, 0)
    votes = sum(_parity_along_axis(mesh, grid, a).astype(np.int8) for a in range(3))
    return votes >= 2


def ball(radius_voxels: float) -> np.ndarray:
    r = int(np.ceil(radius_voxels))
    g = np.mgrid[-r : r + 1, -r : r + 1, -r : r + 1]
    return (g**2).sum(0) <= radius_voxels**2 + 1e-9


def dilate_mm(mask: np.ndarray, radius_mm: float, spacing: float) -> np.ndarray:
    if radius_mm <= 0:
        return mask.copy()
    dist = ndimage.distance_transform_edt(~mask) * spacing
    return dist <= radius_mm


def largest_component(mask: np.ndarray, connectivity=1) -> np.ndarray:
    structure = ndimage.generate_binary_structure(3, connectivity)
    lab, n = ndimage.label(mask, structure=structure)
    if n <= 1:
        return mask.copy()
    sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def fibonacci_directions(n=26) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5**0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


def enclosure_ratio(solid: np.ndarray, query_idx: np.ndarray, max_steps: int, n_dirs=26, chunk=20000):
    """Fraction of rays from each query voxel that hit ``solid`` within ``max_steps``.

    Cavity (blood-pool) voxels inside a myocardial cup are hit from almost
    every direction; voxels outside the heart escape in most directions.
    """
    dirs = fibonacci_directions(n_dirs)
    shape = np.asarray(solid.shape)
    steps = np.arange(1, max_steps + 1, dtype=float)
    out = np.zeros(len(query_idx))
    for s in range(0, len(query_idx), chunk):
        q = query_idx[s : s + chunk].astype(float) + 0.5
        hits = np.zeros(len(q))
        for d in dirs:
            p = np.floor(q[:, None, :] + steps[None, :, None] * d[None, None, :]).astype(np.int64)
            ok = np.all((p >= 0) & (p < shape), axis=2)
            pc = np.where(ok[..., None], p, 0)
            val = solid[pc[..., 0], pc[..., 1], pc[..., 2]] & ok
            hits += val.any(axis=1)
        out[s : s + chunk] = hits / len(dirs)
    return out
