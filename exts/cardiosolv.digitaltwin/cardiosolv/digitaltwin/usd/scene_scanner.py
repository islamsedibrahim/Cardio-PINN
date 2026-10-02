"""USD -> CardioSolv geometry: read the user's selected heart (no copies made).

Works with plain ``pxr`` (usd-core) so it runs in Isaac Sim, in Kit's
``python.sh`` and in headless tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from pxr import Usd, UsdGeom

from ..core.mesh import SurfaceMesh, triangulate_polygons

CARDIOSOLV_ROOT = "/CardioSolv"


@dataclass
class MeshSource:
    """Link from a CardioSolv part back to the USD geometry it came from."""

    prim_path: str
    subset_path: Optional[str] = None  # GeomSubset when a part is a face subset
    face_indices: Optional[np.ndarray] = None  # USD face ids of the subset
    point_indices: Optional[np.ndarray] = None  # USD point ids used by this part
    world_matrix: Optional[np.ndarray] = None  # 4x4, local -> world (USD units)
    mm_per_unit: float = 1000.0


@dataclass
class SceneScan:
    source_prim: str
    meters_per_unit: float
    up_axis: str
    parts: List[SurfaceMesh] = field(default_factory=list)
    sources: List[MeshSource] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    time_code: float = 0.0

    @property
    def mm_per_unit(self):
        return self.meters_per_unit * 1000.0


def _gf_matrix_to_np(m) -> np.ndarray:
    return np.array([[m[i][j] for j in range(4)] for i in range(4)], dtype=np.float64)


def _is_visible(prim, time) -> bool:
    img = UsdGeom.Imageable(prim)
    if not img:
        return True
    return img.ComputeVisibility(time) != UsdGeom.Tokens.invisible


def read_mesh(prim, time, xform_cache, mm_per_unit):
    mesh = UsdGeom.Mesh(prim)
    pts = mesh.GetPointsAttr().Get(time)
    counts = mesh.GetFaceVertexCountsAttr().Get(time)
    idx = mesh.GetFaceVertexIndicesAttr().Get(time)
    if pts is None or counts is None or idx is None or len(pts) == 0:
        raise RuntimeError("mesh has no points/topology")
    pts = np.asarray(pts, dtype=np.float64)
    tris, tri_to_face = triangulate_polygons(counts, idx)
    if UsdGeom.Mesh(prim).GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded:
        tris = tris[:, ::-1].copy()
    world = _gf_matrix_to_np(xform_cache.GetLocalToWorldTransform(prim))
    surf = SurfaceMesh(name=prim.GetName(), points=pts, faces=tris, source_path=str(prim.GetPath()),
                       tri_to_face=tri_to_face)
    surf = surf.transformed(world)
    surf.points *= mm_per_unit
    return surf, world, int(len(counts))


def _subset_parts(prim, surf: SurfaceMesh, n_faces: int):
    """Split a mesh into parts along its GeomSubsets (if they partition anatomy)."""
    subsets = UsdGeom.Subset.GetAllGeomSubsets(UsdGeom.Imageable(prim))
    parts = []
    for sub in subsets:
        fam = sub.GetFamilyNameAttr().Get() or ""
        if fam.startswith("cardiosolv"):
            continue  # our own annotations
        if sub.GetElementTypeAttr().Get() != UsdGeom.Tokens.face:
            continue
        ids = np.asarray(sub.GetIndicesAttr().Get() or [], dtype=np.int64)
        if ids.size == 0:
            continue
        mask = np.zeros(n_faces, bool)
        mask[ids[ids < n_faces]] = True
        tri_mask = mask[surf.tri_to_face]
        if tri_mask.sum() < 4:
            continue
        faces = surf.faces[tri_mask]
        used = np.unique(faces)
        remap = -np.ones(surf.n_points, np.int64)
        remap[used] = np.arange(len(used))
        part = SurfaceMesh(
            name=f"{prim.GetName()}/{sub.GetPrim().GetName()}", points=surf.points[used], faces=remap[faces],
            source_path=str(sub.GetPrim().GetPath()), tri_to_face=surf.tri_to_face[tri_mask],
        )
        parts.append((part, sub, ids, used))
    return parts


def scan_heart(stage: Usd.Stage, source_path: str, time_code=None, split_subsets=True,
               include_invisible=False) -> SceneScan:
    """Collect every Mesh under ``source_path`` in world millimetres."""
    if source_path.startswith(CARDIOSOLV_ROOT):
        raise RuntimeError("CardioSolv cannot analyse its own generated hierarchy.")
    root = stage.GetPrimAtPath(source_path)
    if not root or not root.IsValid():
        raise RuntimeError(f"Invalid source prim: {source_path}")

    if time_code is None:
        time_code = stage.GetStartTimeCode() if stage.HasAuthoredTimeCodeRange() else Usd.TimeCode.Default()
    time = Usd.TimeCode(time_code) if not isinstance(time_code, Usd.TimeCode) else time_code
    mpu = UsdGeom.GetStageMetersPerUnit(stage) or 0.01
    scan = SceneScan(source_prim=source_path, meters_per_unit=mpu, up_axis=str(UsdGeom.GetStageUpAxis(stage)),
                     time_code=float(time.GetValue()) if not time.IsDefault() else 0.0)
    cache = UsdGeom.XformCache(time)
    mm = mpu * 1000.0

    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if not include_invisible and not _is_visible(prim, time):
            scan.skipped.append(f"{prim.GetPath()} (invisible)")
            continue
        try:
            surf, world, n_faces = read_mesh(prim, time, cache, mm)
        except Exception as exc:  # malformed meshes must not stop the analysis
            scan.skipped.append(f"{prim.GetPath()} ({exc})")
            continue
        subs = _subset_parts(prim, surf, n_faces) if split_subsets else []
        if len(subs) >= 2:
            for part, sub, ids, used in subs:
                scan.parts.append(part)
                scan.sources.append(MeshSource(str(prim.GetPath()), str(sub.GetPrim().GetPath()), ids, used, world, mm))
        else:
            scan.parts.append(surf)
            scan.sources.append(MeshSource(str(prim.GetPath()), None, None, None, world, mm))
    if not scan.parts:
        raise RuntimeError(f"No USD meshes found under {source_path}")
    return scan
