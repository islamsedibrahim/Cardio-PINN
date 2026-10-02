"""Stage 4a: computational tetrahedral mesh of the user's myocardium.

A conforming Kuhn (6-tet) subdivision of the myocardial voxels is built at
simulation resolution, its boundary nodes are snapped onto the user's own
myocardium surface, and every boundary face inherits the endo / epi / base /
septum classification of the geometry layer. When gmsh is available the
caller can request it instead (``mesher="gmsh"``) for graded meshes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from scipy.spatial import cKDTree

from .geometry_layer import BASE, ENDO, EPI, RV_SEPTUM, UNKNOWN, V_MYO, GeometryLayer, label_surface_faces
from .mesh import SurfaceMesh
from .voxel import VoxelGrid, largest_component

# Kuhn subdivision of the unit cube: 6 tets around the main diagonal
_CORNERS = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
_KUHN = []
for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
    v = [np.zeros(3, int)]
    for a in perm:
        nxt = v[-1].copy()
        nxt[a] = 1
        v.append(nxt)
    _KUHN.append([int(c[0] * 4 + c[1] * 2 + c[2]) for c in v])
_KUHN = np.array(_KUHN)
TET_FACES = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])  # outward for positive tets


@dataclass
class TetMesh:
    points: np.ndarray  # (N,3) mm
    tets: np.ndarray  # (E,4)
    boundary_faces: np.ndarray  # (F,3) outward
    face_labels: np.ndarray  # (F,) ENDO/EPI/BASE/RV_SEPTUM
    spacing: float
    metadata: dict = field(default_factory=dict)

    @property
    def n_nodes(self):
        return len(self.points)

    @property
    def n_tets(self):
        return len(self.tets)

    def tet_volumes(self, points=None):
        p = self.points if points is None else points
        t = self.tets
        a, b, c, d = p[t[:, 0]], p[t[:, 1]], p[t[:, 2]], p[t[:, 3]]
        return np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a)) / 6.0

    def node_set(self, *codes) -> np.ndarray:
        sel = np.isin(self.face_labels, codes)
        return np.unique(self.boundary_faces[sel])

    def node_sets(self) -> Dict[str, np.ndarray]:
        return {"endo": self.node_set(ENDO), "epi": self.node_set(EPI, RV_SEPTUM), "base": self.node_set(BASE),
                "rv_septum": self.node_set(RV_SEPTUM), "surface": np.unique(self.boundary_faces)}

    def boundary_surface(self) -> SurfaceMesh:
        return SurfaceMesh("myocardium_sim_surface", self.points, self.boundary_faces)


def boundary_faces_of(tets: np.ndarray) -> np.ndarray:
    faces = tets[:, TET_FACES].reshape(-1, 3)
    key = np.sort(faces, axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    return faces[cnt[inv.ravel()] == 1]


def voxel_tets(mask: np.ndarray, grid: VoxelGrid):
    vox = np.argwhere(mask)
    corners = vox[:, None, :] + _CORNERS[None, :, :]  # (V,8,3)
    lattice = np.asarray(mask.shape) + 1
    flat = np.ravel_multi_index(corners.reshape(-1, 3).T, lattice).reshape(-1, 8)
    tets = flat[:, _KUHN].reshape(-1, 4)
    used, inv = np.unique(tets, return_inverse=True)
    tets = inv.reshape(-1, 4)
    pts = grid.corner(np.stack(np.unravel_index(used, lattice), axis=1))
    # positive orientation
    a, b, c, d = pts[tets[:, 0]], pts[tets[:, 1]], pts[tets[:, 2]], pts[tets[:, 3]]
    neg = np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a)) < 0
    tets[neg] = tets[neg][:, [0, 2, 1, 3]]
    return pts, tets


def _surface_samples(surface: SurfaceMesh, density_mm: float):
    """Dense point samples on the user's surface (vertices + barycentric fill)."""
    pts = [surface.points]
    a, b, c = surface.face_corners()
    pts.append((a + b + c) / 3.0)
    area = surface.face_areas()
    big = area > (0.5 * density_mm) ** 2
    if big.any():
        n = np.minimum(np.ceil(area[big] / (0.5 * density_mm) ** 2).astype(int), 200)
        idx = np.repeat(np.nonzero(big)[0], n)
        r = np.random.default_rng(0).random((len(idx), 2))
        flip = r.sum(1) > 1
        r[flip] = 1 - r[flip]
        pts.append(a[idx] + r[:, :1] * (b[idx] - a[idx]) + r[:, 1:] * (c[idx] - a[idx]))
    return np.vstack(pts)


def snap_to_surface(mesh: TetMesh, surface: SurfaceMesh, max_move_factor=0.9, iterations=4, labels=None):
    """Move boundary nodes (optionally only those on faces with ``labels``) onto the user's surface
    without inverting tets."""
    faces = mesh.boundary_faces if labels is None else mesh.boundary_faces[np.isin(mesh.face_labels, labels)]
    nodes = np.unique(faces)
    samples = _surface_samples(surface, mesh.spacing)
    tree = cKDTree(samples)
    d, j = tree.query(mesh.points[nodes])
    target = samples[j]
    move = target - mesh.points[nodes]
    lim = max_move_factor * mesh.spacing
    scale = np.minimum(1.0, lim / np.maximum(d, 1e-12))
    new = mesh.points.copy()
    new[nodes] += move * scale[:, None]
    v0 = mesh.tet_volumes()
    for _ in range(iterations):
        bad = mesh.tet_volumes(new) < 0.15 * v0
        if not bad.any():
            break
        bad_nodes = np.intersect1d(np.unique(mesh.tets[bad]), nodes)
        new[bad_nodes] = 0.5 * (new[bad_nodes] + mesh.points[bad_nodes])
    bad = mesh.tet_volumes(new) <= 0
    if bad.any():
        bn = np.unique(mesh.tets[bad])
        new[bn] = mesh.points[bn]
    mesh.metadata["snap_mean_distance_mm"] = float(np.linalg.norm(new[nodes] - target, axis=1).mean())
    mesh.points = new
    return mesh


def taubin_smooth_boundary(mesh: TetMesh, iterations=10, lam=0.5, mu=-0.53, fixed=None):
    faces = mesh.boundary_faces
    nodes = np.unique(faces)
    if fixed is not None:
        nodes = np.setdiff1d(nodes, fixed)
    import scipy.sparse as sp

    n = mesh.n_nodes
    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2], faces[:, 1], faces[:, 2], faces[:, 0]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0], faces[:, 0], faces[:, 1], faces[:, 2]])
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    p = mesh.points.copy()
    v0 = mesh.tet_volumes()
    for it in range(2 * iterations):
        f = lam if it % 2 == 0 else mu
        avg = (A @ p) / np.maximum(deg, 1)[:, None]
        q = p.copy()
        q[nodes] += f * (avg[nodes] - p[nodes])
        if (mesh.tet_volumes(q) < 0.15 * v0).any():
            break
        p = q
    mesh.points = p
    return mesh


def build_tet_mesh(gl: GeometryLayer, myo_surface: Optional[SurfaceMesh], spacing_mm=2.5,
                   mesher="voxel", skin_surface: Optional[SurfaceMesh] = None) -> TetMesh:
    """``skin_surface``: skin-only hearts - only the epicardium is snapped to the user's skin,
    the derived inner surfaces are smoothed."""
    if mesher == "gmsh":
        from ..io.external_tools import gmsh_tet_mesh

        return gmsh_tet_mesh(gl, myo_surface, spacing_mm)
    lo = gl.grid.centers(np.argwhere(gl.myo)).min(0) - gl.grid.spacing
    hi = gl.grid.centers(np.argwhere(gl.myo)).max(0) + gl.grid.spacing
    grid = VoxelGrid.around(lo, hi, spacing_mm, padding_voxels=1)
    idx = np.argwhere(np.ones(grid.shape, bool))
    centers = grid.centers(idx)
    # majority of 8 sub-samples per coarse voxel for a faithful resampling
    off = (np.array(_CORNERS, float) - 0.5) * 0.5 * spacing_mm
    votes = sum((gl.grid.sample(gl.voxel_class, centers + o) == V_MYO).astype(int) for o in off)
    mask = np.zeros(grid.shape, bool)
    mask[tuple(idx.T)] = votes >= 4
    mask = largest_component(mask, connectivity=1)
    if mask.sum() < 20:
        raise RuntimeError(f"Myocardium too thin for {spacing_mm} mm elements; reduce the element size.")
    pts, tets = voxel_tets(mask, grid)
    bfaces = boundary_faces_of(tets)
    mesh = TetMesh(points=pts, tets=tets, boundary_faces=bfaces, face_labels=np.zeros(len(bfaces), np.int8),
                   spacing=spacing_mm, metadata={"mesher": "voxel_kuhn", "voxels": int(mask.sum())})
    surf = SurfaceMesh("vox", pts, bfaces)
    mesh.face_labels = label_surface_faces(surf, gl.grid, gl.voxel_class, gl.long_axis)
    if myo_surface is not None:
        snap_to_surface(mesh, myo_surface)
    elif skin_surface is not None:
        taubin_smooth_boundary(mesh, fixed=mesh.node_set(EPI))
        snap_to_surface(mesh, skin_surface, labels=(EPI,))
    else:
        taubin_smooth_boundary(mesh)
    vols = mesh.tet_volumes()
    mesh.metadata.update({
        "nodes": mesh.n_nodes, "tets": mesh.n_tets, "volume_ml": float(vols.sum() / 1000.0),
        "min_tet_volume_ratio": float(vols.min() / (spacing_mm**3 / 6.0)),
        "unlabelled_faces": int((mesh.face_labels == UNKNOWN).sum()),
    })
    return mesh
