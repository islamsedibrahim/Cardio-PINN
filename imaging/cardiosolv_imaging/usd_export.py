"""Write the reconstructed heart as an Isaac Sim-ready USD (metres, Z-up = patient superior).

Layout (part names follow TotalSegmentator, which CardioSolv recognises)::

    /World                       (defaultPrim)
      /Patient                   Xform, translate = -heart centre (heart at the origin)
        /Heart                   Xform  <- select this prim in the CardioSolv panel
          /heart_myocardium      Mesh
          /heart_ventricle_left  Mesh ...
        /Scar                    Xform (LGE scar, optional)
          /scar_core             Mesh  dense scar
          /scar_border_zone      Mesh  border / grey zone
"""

from __future__ import annotations

import numpy as np
from pxr import Gf, Usd, UsdGeom, Vt

from . import labels as L


SCAR_COLORS = {"scar_core": (0.95, 0.95, 0.95), "scar_border_zone": (0.95, 0.75, 0.3)}


def write_heart_usd(path, surfaces: dict, meta: dict, center_heart=True, scar_surfaces: dict = None):
    """``surfaces``: {structure name: (points_ras_mm, faces)}; ``scar_surfaces``: {scar_core|scar_border_zone: ...}."""
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    patient = UsdGeom.Xform.Define(stage, "/World/Patient")
    heart = UsdGeom.Xform.Define(stage, "/World/Patient/Heart")
    allpts = np.vstack([p for p, _ in surfaces.values()]) if surfaces else np.zeros((1, 3))
    centre = allpts.mean(0) / 1000.0 if center_heart else np.zeros(3)
    patient.AddTranslateOp().Set(Gf.Vec3d(*(-centre).tolist()))
    hp = heart.GetPrim()
    hp.SetCustomDataByKey("cardiosolv:source", "cardiosolv_imaging")
    hp.SetCustomDataByKey("cardiosolv:coordinate_system", "RAS (patient), metres; Patient xform recentres the heart")
    hp.SetCustomDataByKey("cardiosolv:patient_offset_m", Gf.Vec3d(*centre.tolist()))
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            hp.SetCustomDataByKey(f"cardiosolv:{k}", v)
    order = [s.name for s in L.STRUCTURES]
    for name in sorted(surfaces, key=order.index):
        pts, faces = surfaces[name]
        s = L.BY_NAME[name]
        mesh = UsdGeom.Mesh.Define(stage, f"/World/Patient/Heart/{name}")
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy((pts / 1000.0).astype(np.float32)))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(faces.reshape(-1).astype(int).tolist()))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        lo, hi = pts.min(0) / 1000.0, pts.max(0) / 1000.0
        mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*lo.tolist()), Gf.Vec3f(*hi.tolist())]))
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*s.color)])
        if name == "heart":
            mesh.CreateDisplayOpacityAttr([0.25])
        p = mesh.GetPrim()
        p.SetCustomDataByKey("cardiosolv:structure", name)
        p.SetCustomDataByKey("cardiosolv:title", s.title)
        p.SetCustomDataByKey("cardiosolv:label", s.label)
    if scar_surfaces:
        scar = UsdGeom.Xform.Define(stage, "/World/Patient/Scar")
        scar.GetPrim().SetCustomDataByKey("cardiosolv:source", "LGE")
        for name, (pts, faces) in scar_surfaces.items():
            mesh = UsdGeom.Mesh.Define(stage, f"/World/Patient/Scar/{name}")
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy((pts / 1000.0).astype(np.float32)))
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(faces.reshape(-1).astype(int).tolist()))
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.CreateDisplayColorAttr([Gf.Vec3f(*SCAR_COLORS.get(name, (1, 1, 1)))])
            mesh.GetPrim().SetCustomDataByKey("cardiosolv:structure", name)
    stage.GetRootLayer().Save()
    return str(path)
