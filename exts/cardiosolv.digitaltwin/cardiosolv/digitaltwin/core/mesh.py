"""Triangle surface meshes in CardioSolv world space (millimetres).

Everything in ``cardiosolv.digitaltwin.core`` is pure NumPy/SciPy so it can be
unit-tested outside Isaac Sim. The USD adapter (``usd/scene_scanner.py``)
converts the user's USD meshes into :class:`SurfaceMesh` objects, keeping the
mapping back to the original prim so results can be painted onto it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def triangulate_polygons(face_vertex_counts, face_vertex_indices):
    """Fan-triangulate USD polygons.

    Returns ``(triangles, tri_to_face)`` where ``tri_to_face[i]`` is the index of
    the source polygon, so per-triangle labels can be folded back into
    per-face USD data (GeomSubsets).
    """
    counts = np.asarray(face_vertex_counts, dtype=np.int64)
    indices = np.asarray(face_vertex_indices, dtype=np.int64)
    if counts.size == 0:
        return np.empty((0, 3), np.int64), np.empty(0, np.int64)

    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])
    n_tris = np.clip(counts - 2, 0, None)
    tri_to_face = np.repeat(np.arange(counts.size), n_tris)
    # local fan index k = 1..count-2 for each triangle
    starts = np.concatenate([[0], np.cumsum(n_tris)[:-1]])
    k = np.arange(n_tris.sum()) - np.repeat(starts, n_tris) + 1
    base = offsets[tri_to_face]
    tris = np.stack(
        [indices[base], indices[base + k], indices[base + k + 1]], axis=1
    )
    return tris, tri_to_face


@dataclass
class SurfaceMesh:
    """A triangulated surface in world millimetres."""

    name: str
    points: np.ndarray  # (N, 3) float64, mm, world space
    faces: np.ndarray  # (M, 3) int64
    source_path: str = ""
    tri_to_face: Optional[np.ndarray] = None  # triangle -> original USD face
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=np.float64)
        self.faces = np.asarray(self.faces, dtype=np.int64).reshape(-1, 3)
        if self.tri_to_face is None:
            self.tri_to_face = np.arange(len(self.faces))

    # ------------------------------------------------------------------
    # Basic geometry
    # ------------------------------------------------------------------
    @property
    def n_points(self) -> int:
        return len(self.points)

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    @property
    def bbox_min(self) -> np.ndarray:
        return self.points.min(axis=0)

    @property
    def bbox_max(self) -> np.ndarray:
        return self.points.max(axis=0)

    @property
    def dimensions(self) -> np.ndarray:
        return self.bbox_max - self.bbox_min

    def face_corners(self):
        p = self.points
        f = self.faces
        return p[f[:, 0]], p[f[:, 1]], p[f[:, 2]]

    def face_area_vectors(self) -> np.ndarray:
        a, b, c = self.face_corners()
        return 0.5 * np.cross(b - a, c - a)

    def face_areas(self) -> np.ndarray:
        return np.linalg.norm(self.face_area_vectors(), axis=1)

    def face_normals(self) -> np.ndarray:
        n = self.face_area_vectors()
        norm = np.linalg.norm(n, axis=1, keepdims=True)
        return n / np.maximum(norm, 1e-30)

    def face_centers(self) -> np.ndarray:
        a, b, c = self.face_corners()
        return (a + b + c) / 3.0

    @property
    def surface_area(self) -> float:
        return float(self.face_areas().sum())

    def signed_volume(self) -> float:
        """Divergence-theorem volume; positive for outward-oriented closed meshes."""
        if self.n_faces == 0:
            return 0.0
        a, b, c = self.face_corners()
        ref = self.points.mean(axis=0)
        return float(np.einsum("ij,ij->i", a - ref, np.cross(b - ref, c - ref)).sum() / 6.0)

    @property
    def volume(self) -> float:
        return abs(self.signed_volume())

    @property
    def centroid(self) -> np.ndarray:
        """Area-weighted surface centroid (robust to uneven vertex density)."""
        areas = self.face_areas()
        if areas.sum() <= 0:
            return self.points.mean(axis=0)
        return (self.face_centers() * areas[:, None]).sum(axis=0) / areas.sum()

    # ------------------------------------------------------------------
    # Topology
    # ------------------------------------------------------------------
    def edge_counts(self):
        """Unique undirected edges and how many faces use each."""
        f = self.faces
        edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        edges.sort(axis=1)
        uniq, counts = np.unique(edges, axis=0, return_counts=True)
        return uniq, counts

    def topology(self) -> dict:
        uniq, counts = self.edge_counts()
        boundary = int((counts == 1).sum())
        nonmanifold = int((counts > 2).sum())
        used = np.unique(self.faces)
        euler = len(used) - len(uniq) + self.n_faces
        return {
            "boundary_edges": boundary,
            "non_manifold_edges": nonmanifold,
            "is_closed": boundary == 0,
            "is_manifold": nonmanifold == 0,
            "euler_characteristic": int(euler),
            "components": self.connected_component_count(),
        }

    def face_adjacency(self):
        """Sparse face-face adjacency (shared edge)."""
        import scipy.sparse as sp

        f = self.faces
        m = len(f)
        edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        edges.sort(axis=1)
        face_ids = np.tile(np.arange(m), 3)
        key = edges[:, 0] * (self.n_points + 1) + edges[:, 1]
        order = np.argsort(key, kind="stable")
        key_s = key[order]
        fid_s = face_ids[order]
        same = key_s[1:] == key_s[:-1]
        a = fid_s[:-1][same]
        b = fid_s[1:][same]
        adj = sp.coo_matrix((np.ones(len(a)), (a, b)), shape=(m, m))
        return (adj + adj.T).tocsr()

    def connected_components(self):
        import scipy.sparse as sp
        from scipy.sparse.csgraph import connected_components

        f = self.faces
        n = self.n_points
        rows = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
        cols = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
        g = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        _, labels = connected_components(g, directed=False)
        return labels

    def connected_component_count(self) -> int:
        if self.n_faces == 0:
            return 0
        labels = self.connected_components()
        return int(len(np.unique(labels[np.unique(self.faces)])))

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------
    def copy(self) -> "SurfaceMesh":
        return SurfaceMesh(
            name=self.name,
            points=self.points.copy(),
            faces=self.faces.copy(),
            source_path=self.source_path,
            tri_to_face=self.tri_to_face.copy(),
            metadata=dict(self.metadata),
        )

    def transformed(self, matrix4: np.ndarray) -> "SurfaceMesh":
        """Apply a 4x4 row-vector transform (USD convention: p' = p @ M)."""
        m = np.asarray(matrix4, dtype=np.float64)
        homo = np.c_[self.points, np.ones(self.n_points)]
        out = self.copy()
        out.points = (homo @ m)[:, :3]
        if np.linalg.det(m[:3, :3]) < 0:  # mirrored transform flips orientation
            out.faces = out.faces[:, ::-1].copy()
        return out

    def oriented_outward(self) -> "SurfaceMesh":
        """Return a copy whose faces point outward (positive signed volume)."""
        if self.signed_volume() < 0:
            out = self.copy()
            out.faces = out.faces[:, ::-1].copy()
            out.metadata["orientation_flipped"] = True
            return out
        return self


def merge_meshes(meshes, name="merged") -> SurfaceMesh:
    pts, faces, offset = [], [], 0
    for m in meshes:
        pts.append(m.points)
        faces.append(m.faces + offset)
        offset += m.n_points
    return SurfaceMesh(name=name, points=np.vstack(pts), faces=np.vstack(faces))


def pca_axes(points: np.ndarray):
    """Principal axes sorted by decreasing variance: (centroid, axes[3,3] rows, eigenvalues)."""
    centroid = points.mean(axis=0)
    cov = np.cov((points - centroid).T)
    w, v = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    return centroid, v[:, order].T, w[order]


def normalize(v, axis=-1, eps=1e-30):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, eps)
