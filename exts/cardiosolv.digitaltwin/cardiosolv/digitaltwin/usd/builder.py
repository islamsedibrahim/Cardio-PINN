"""/CardioSolv semantic USD layer (Phase 3 design).

* ``/CardioSolv/Geometry/Heart/*`` are ``Scope`` prims describing what each
  structure means, pointing at the user's geometry (no duplicate meshes).
* Endocardium / epicardium / base / RV-septum are authored as
  ``UsdGeom.Subset`` face sets directly on the user's myocardium mesh.
* Landmarks and the long axis are ``Xform`` prims with small debug glyphs
  tagged ``cardiosolv:debug_visualization``.
"""

from __future__ import annotations

import re

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, Vt

from ..core.canonical import CanonicalHeartGeometry

ROOT = "/CardioSolv"
SURFACE_FAMILY = "cardiosolv:surface"
SURFACE_COLORS = {
    "Endocardium": (0.85, 0.20, 0.25),
    "Epicardium": (0.95, 0.75, 0.55),
    "Base": (0.30, 0.55, 0.95),
    "RVSeptum": (0.55, 0.30, 0.85),
}
LANDMARK_COLORS = {"Apex": (1.0, 0.85, 0.1), "Base": (0.1, 0.8, 1.0)}


def safe_name(text: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_]", "_", text.strip("/")) or "root"
    return name if not name[0].isdigit() else "_" + name


def _vec(v):
    return Gf.Vec3d(*[float(x) for x in v])


class CardioSolvUSDBuilder:
    def __init__(self, stage: Usd.Stage, mm_per_unit: float):
        self.stage = stage
        self.mm_per_unit = float(mm_per_unit)

    def to_stage(self, p_mm):
        return np.asarray(p_mm, float) / self.mm_per_unit

    # ------------------------------------------------------------------
    def create_structure(self):
        for path in (ROOT, f"{ROOT}/Geometry", f"{ROOT}/Geometry/Heart", f"{ROOT}/Geometry/Source",
                     f"{ROOT}/Landmarks", f"{ROOT}/CoordinateSystem", f"{ROOT}/Analysis",
                     f"{ROOT}/Analysis/Candidates", f"{ROOT}/Metadata", f"{ROOT}/Debug"):
            if not self.stage.GetPrimAtPath(path):
                self.stage.DefinePrim(path, "Scope")

    def clear(self):
        if self.stage.GetPrimAtPath(ROOT):
            self.stage.RemovePrim(ROOT)

    def register_source(self, source_path: str, roles: dict):
        prim = self.stage.GetPrimAtPath(f"{ROOT}/Geometry/Source")
        prim.SetCustomDataByKey("cardiosolv:source_prim", source_path)
        prim.SetCustomDataByKey("cardiosolv:generated", False)
        for role, path in roles.items():
            prim.SetCustomDataByKey(f"cardiosolv:role:{role}", path)
        rel = prim.CreateRelationship("cardiosolv:source", custom=True)
        rel.SetTargets([Sdf.Path(source_path)])

    def register_metadata(self, can: CanonicalHeartGeometry, report: dict):
        prim = self.stage.GetPrimAtPath(f"{ROOT}/Metadata")
        prim.SetCustomDataByKey("cardiosolv:schema_version", can.schema_version)
        prim.SetCustomDataByKey("cardiosolv:source_prim", can.source_prim)
        prim.SetCustomDataByKey("cardiosolv:generated_geometry", False)
        prim.SetCustomDataByKey("cardiosolv:mm_per_stage_unit", self.mm_per_unit)
        for k, v in can.metadata.get("metrics", {}).items():
            prim.SetCustomDataByKey(f"cardiosolv:metric:{k}", float(v))
        prim.SetCustomDataByKey("cardiosolv:validation_status", report.get("status", ""))
        prim.SetCustomDataByKey("cardiosolv:warnings", Vt.StringArray(report.get("warnings", [])))
        prim.SetCustomDataByKey("cardiosolv:errors", Vt.StringArray(report.get("errors", [])))

    def register_surface(self, comp):
        prim = self.stage.DefinePrim(f"{ROOT}/Geometry/Heart/{comp.name}", "Scope")
        prim.SetCustomDataByKey("cardiosolv:component_type", comp.component_type)
        prim.SetCustomDataByKey("cardiosolv:source_path", comp.source_path)
        prim.SetCustomDataByKey("cardiosolv:confidence", float(comp.confidence))
        prim.SetCustomDataByKey("cardiosolv:face_count", int(len(comp.face_indices)))
        prim.SetCustomDataByKey("cardiosolv:area_cm2", float(comp.area_cm2))
        prim.SetCustomDataByKey("cardiosolv:generated", False)
        prim.CreateRelationship("cardiosolv:source", custom=True).SetTargets([Sdf.Path(comp.source_path)])
        return prim

    def register_region(self, region):
        prim = self.stage.DefinePrim(f"{ROOT}/Geometry/Heart/{region.name}", "Scope")
        prim.SetCustomDataByKey("cardiosolv:region_type", region.region_type)
        prim.SetCustomDataByKey("cardiosolv:confidence", float(region.confidence))
        prim.SetCustomDataByKey("cardiosolv:generated", False)
        for key in ("outer_boundary", "inner_boundary", "source_path"):
            val = getattr(region, key)
            if val:
                prim.SetCustomDataByKey(f"cardiosolv:{key}", val)
        for k, v in region.metadata.items():
            if isinstance(v, (str, int, float, bool)):
                prim.SetCustomDataByKey(f"cardiosolv:{k}", v)
        return prim

    def write_surface_subsets(self, can: CanonicalHeartGeometry, preview_colors=True):
        """GeomSubsets (endo/epi/base/septum) on the user's myocardium mesh."""
        by_mesh = {}
        for comp in can.surfaces.values():
            by_mesh.setdefault(comp.source_path, []).append(comp)
        for mesh_path, comps in by_mesh.items():
            prim = self.stage.GetPrimAtPath(mesh_path)
            if not prim:
                continue
            mesh = UsdGeom.Mesh(prim)
            img = UsdGeom.Imageable(prim)
            n_faces = len(mesh.GetFaceVertexCountsAttr().Get() or [])
            label = np.zeros(n_faces, np.int32)
            colors = np.tile(np.array([0.6, 0.6, 0.6]), (n_faces, 1))
            for k, comp in enumerate(comps, start=1):
                ids = np.asarray(comp.face_indices, int)
                UsdGeom.Subset.CreateGeomSubset(
                    img, f"cardiosolv_{comp.name}", UsdGeom.Tokens.face, Vt.IntArray(ids.tolist()),
                    familyName=SURFACE_FAMILY, familyType=UsdGeom.Tokens.nonOverlapping)
                label[ids] = k
                colors[ids] = SURFACE_COLORS.get(comp.name, (0.6, 0.6, 0.6))
            pv = UsdGeom.PrimvarsAPI(prim)
            lab_pv = pv.CreatePrimvar("cardiosolv:surfaceLabel", Sdf.ValueTypeNames.IntArray, UsdGeom.Tokens.uniform)
            lab_pv.Set(Vt.IntArray(label.tolist()))
            prim.SetCustomDataByKey("cardiosolv:surfaceLabelNames", Vt.StringArray([""] + [c.name for c in comps]))
            if preview_colors:
                col = pv.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.uniform)
                col.Set(Vt.Vec3fArray.FromNumpy(colors.astype(np.float32)))

    def create_landmark(self, lm, radius_mm=2.5):
        path = f"{ROOT}/Landmarks/{safe_name(lm.name)}"
        xf = UsdGeom.Xform.Define(self.stage, path)
        xf.ClearXformOpOrder()
        pos = self.to_stage(lm.position)
        xf.AddTranslateOp().Set(_vec(pos))
        prim = xf.GetPrim()
        prim.SetCustomDataByKey("cardiosolv:landmark_type", lm.landmark_type)
        prim.SetCustomDataByKey("cardiosolv:position_world", Gf.Vec3d(*pos.tolist()))
        prim.SetCustomDataByKey("cardiosolv:confidence", float(lm.confidence))
        prim.SetCustomDataByKey("cardiosolv:method", str(lm.metadata.get("method", "")))
        sph = UsdGeom.Sphere.Define(self.stage, f"{path}/Glyph")
        sph.GetRadiusAttr().Set(radius_mm / self.mm_per_unit)
        sph.GetDisplayColorAttr().Set([Gf.Vec3f(*LANDMARK_COLORS.get(lm.name, (0.2, 1.0, 0.4)))])
        sph.GetPrim().SetCustomDataByKey("cardiosolv:debug_visualization", True)
        return prim

    def create_axis(self, axis, length_mm=None, color=(1.0, 0.3, 0.1), width_mm=0.8):
        path = f"{ROOT}/CoordinateSystem/{safe_name(axis.name)}"
        xf = UsdGeom.Xform.Define(self.stage, path)
        prim = xf.GetPrim()
        d = np.asarray(axis.direction, float)
        d = d / max(np.linalg.norm(d), 1e-12)
        origin = self.to_stage(axis.origin)
        prim.SetCustomDataByKey("cardiosolv:axis_type", axis.name)
        prim.SetCustomDataByKey("cardiosolv:origin_world", Gf.Vec3d(*origin.tolist()))
        prim.SetCustomDataByKey("cardiosolv:direction_world", Gf.Vec3d(*d.tolist()))
        prim.SetCustomDataByKey("cardiosolv:confidence", float(axis.confidence))
        L = (length_mm or axis.metadata.get("length_mm", 40.0)) / self.mm_per_unit
        curve = UsdGeom.BasisCurves.Define(self.stage, f"{path}/Glyph")
        curve.CreateTypeAttr(UsdGeom.Tokens.linear)
        curve.CreateCurveVertexCountsAttr([2])
        p0, p1 = origin - 0.1 * L * d, origin + 1.15 * L * d
        curve.CreatePointsAttr([Gf.Vec3f(*p0), Gf.Vec3f(*p1)])
        curve.CreateWidthsAttr([width_mm / self.mm_per_unit])
        curve.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
        curve.GetPrim().SetCustomDataByKey("cardiosolv:debug_visualization", True)
        return prim

    def register_analysis(self, asg):
        for f in asg.features:
            prim = self.stage.DefinePrim(f"{ROOT}/Analysis/Candidates/{safe_name(f.path)}", "Scope")
            prim.SetCustomDataByKey("cardiosolv:source_path", f.path)
            cands = asg.candidates[f.index]
            for k, c in enumerate(cands[:5]):
                prim.SetCustomDataByKey(f"cardiosolv:candidate_{k + 1}", c.label)
                prim.SetCustomDataByKey(f"cardiosolv:score_{k + 1}", float(c.score))
            from ..core.anatomy import ambiguity_margin, confidence_class

            margin = ambiguity_margin(cands)
            status = "AMBIGUOUS" if margin < 0.10 else confidence_class(cands[0].score) + "_CONFIDENCE"
            prim.SetCustomDataByKey("cardiosolv:classification_status", status)
            prim.SetCustomDataByKey("cardiosolv:ambiguity_margin", float(margin))
            assigned = [r for r, i in asg.roles.items() if i == f.index]
            prim.SetCustomDataByKey("cardiosolv:assigned_role", assigned[0] if assigned else "Unknown")
            for k, v in f.summary().items():
                if isinstance(v, (bool, str)):
                    prim.SetCustomDataByKey(f"cardiosolv:{k}", v)
                else:
                    prim.SetCustomDataByKey(f"cardiosolv:{k}", float(v))

    # ------------------------------------------------------------------
    def build(self, can: CanonicalHeartGeometry, asg, report: dict, preview_colors=True):
        self.create_structure()
        self.register_source(can.source_prim, can.roles)
        self.register_metadata(can, report)
        for comp in can.surfaces.values():
            self.register_surface(comp)
        for region in can.regions.values():
            self.register_region(region)
        self.write_surface_subsets(can, preview_colors=preview_colors)
        for lm in can.landmarks.values():
            self.create_landmark(lm)
        self.create_axis(can.axes["LongAxis"])
        sep = can.axes.get("SeptalReference")
        if sep is not None:
            self.create_axis(sep, length_mm=25.0, color=(0.6, 0.3, 0.9))
        self.register_analysis(asg)
