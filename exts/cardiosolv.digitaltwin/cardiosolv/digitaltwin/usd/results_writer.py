"""Stage 8b: animate and paint the simulation onto the user's own USD heart.

* every Mesh prim under the selected heart gets time-sampled ``points``
  (the myocardium follows the FE solution exactly, neighbouring parts follow
  with a distance fall-off),
* the selected result field is painted as time-sampled vertex
  ``displayColor`` on the myocardium (and optionally every part),
* raw field values are kept as ``primvars:cardiosolv:<field>``,
* fibres are drawn as debug ``BasisCurves``.

Everything goes to the ``*_cardiosolv_twin.usdc`` sublayer.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, Vt

from ..core.colormap import colorize
from ..core.mapping import build_point_map
from ..core.volume_mesh import TetMesh
from .layer import LAYER_TAG, RESULTS_TAG, _resolve, find_layer, get_or_create_layer
from .scene_scanner import SceneScan

FIELD_STYLE = {
    "activation_time": ("ms", "turbo"),
    "transmembrane_potential": ("mV", "inferno"),
    "active_tension": ("kPa", "inferno"),
    "fiber_strain": ("-", "coolwarm"),
    "fiber_stress": ("kPa", "turbo"),
    "displacement": ("mm", "viridis"),
    "transmural": ("-", "viridis"),
}


class TwinResultsWriter:
    def __init__(self, stage: Usd.Stage, scan: SceneScan, mesh: TetMesh, myocardium_prim: Optional[str],
                 falloff_mm=20.0, paint_prims=None, paint_falloff_mm=3.0):
        """Fields are computed in the simulation frame (``scan.to_sim``) and mapped back to world.

        ``paint_prims``: prims painted besides the myocardium (e.g. the skin of a skin-only heart);
        their colours fade to neutral grey where they are farther than ``paint_falloff_mm`` from
        the simulated myocardium (RV free wall, atria)."""
        self.stage = stage
        self.scan = scan
        self.mesh = mesh
        self.myo_prim = myocardium_prim
        self.paint_prims = set(paint_prims or [])
        self.paint_falloff_mm = paint_falloff_mm
        self.layer = get_or_create_layer(stage, tag=RESULTS_TAG, ext="usdc")
        self._ensure_strongest()
        self.prims: Dict[str, dict] = {}
        mm = scan.mm_per_unit
        seen = set()
        for src in scan.sources:
            if src.prim_path in seen:
                continue
            seen.add(src.prim_path)
            prim = stage.GetPrimAtPath(src.prim_path)
            mesh_geom = UsdGeom.Mesh(prim)
            local = np.asarray(mesh_geom.GetPointsAttr().Get(Usd.TimeCode(scan.time_code)), dtype=np.float64)
            M = np.asarray(src.world_matrix, float)
            world_mm = (np.c_[local, np.ones(len(local))] @ M)[:, :3] * mm
            is_myo = src.prim_path == myocardium_prim
            pmap = build_point_map(mesh, scan.to_sim(world_mm), falloff_mm=1e9 if is_myo else falloff_mm)
            # Sublayers are weaker than the root layer: if the root layer itself authors this mesh's
            # points, a sublayer cannot animate it, so the animation goes to the session layer
            # (live view) and export_twin_stage() moves it into the twin layer for saving.
            root_owned = bool(stage.GetRootLayer().GetAttributeAtPath(mesh_geom.GetPointsAttr().GetPath()))
            self.prims[src.prim_path] = {"geom": mesh_geom, "M_inv": np.linalg.inv(M), "world_mm": world_mm,
                                         "map": pmap, "is_myo": is_myo, "n": len(local), "root_owned": root_owned}

    def _ensure_strongest(self):
        root = self.stage.GetRootLayer()
        subs = list(root.subLayerPaths)
        sem = find_layer(self.stage, LAYER_TAG)
        ident = [p for p in subs if _resolve(root, p) == self.layer]
        if ident and sem is not None and subs.index(ident[0]) != 0:
            subs.remove(ident[0])
            subs.insert(0, ident[0])
            root.subLayerPaths = subs

    # ------------------------------------------------------------------
    def time_codes(self, times_ms, slow_motion=4.0, start=None):
        tcps = self.stage.GetTimeCodesPerSecond() or 60.0
        t0 = float(times_ms[0])
        start = self.stage.GetStartTimeCode() if start is None else start
        return start + (np.asarray(times_ms, float) - t0) / 1000.0 * slow_motion * tcps

    def set_time_range(self, codes):
        session = self.stage.GetSessionLayer()
        session.startTimeCode = float(codes[0])
        session.endTimeCode = float(codes[-1])
        # (the stage range lives on root/session layers; export_twin_stage() persists it)
        self.layer.startTimeCode = float(codes[0])
        self.layer.endTimeCode = float(codes[-1])

    def write_animation(self, codes, nodal_u: np.ndarray, animate_neighbours=True):
        """``nodal_u`` (T,N,3) mm on the computational mesh."""
        mm = self.scan.mm_per_unit
        for path, info in self.prims.items():
            if not info["is_myo"] and not animate_neighbours:
                continue
            target = self.stage.GetSessionLayer() if info["root_owned"] else self.layer
            with Usd.EditContext(self.stage, target):
                pts_attr = info["geom"].GetPointsAttr()
                ext_attr = info["geom"].GetExtentAttr()
                for code, u in zip(codes, nodal_u):
                    world = info["world_mm"] + info["map"].apply_displacement(u) / self.scan.sim_scale
                    local = (np.c_[world / mm, np.ones(len(world))] @ info["M_inv"])[:, :3].astype(np.float32)
                    pts_attr.Set(Vt.Vec3fArray.FromNumpy(local), Usd.TimeCode(float(code)))
                    lo, hi = local.min(0), local.max(0)
                    ext_attr.Set(Vt.Vec3fArray([Gf.Vec3f(*lo.tolist()), Gf.Vec3f(*hi.tolist())]), Usd.TimeCode(float(code)))
        self.set_time_range(codes)

    def root_owned_paths(self):
        return [p for p, i in self.prims.items() if i["root_owned"]]

    def write_field(self, name, codes, nodal_values, vmin=None, vmax=None, cmap=None, all_parts=False):
        """Paint a nodal field (N,) static or (T,N) animated onto the user's meshes."""
        vals = np.asarray(nodal_values, float)
        static = vals.ndim == 1
        finite = vals[np.isfinite(vals)]
        vmin = float(np.percentile(finite, 1)) if vmin is None else vmin
        vmax = float(np.percentile(finite, 99)) if vmax is None else vmax
        cmap = cmap or FIELD_STYLE.get(name, ("", "turbo"))[1]
        with Usd.EditContext(self.stage, self.layer):
            for path, info in self.prims.items():
                if not info["is_myo"] and not all_parts and path not in self.paint_prims:
                    continue
                prim = info["geom"].GetPrim()
                # fade to grey away from the simulated wall (sim-frame mm)
                d = info["map"].distance
                wgt = np.clip(1.0 - np.maximum(d - 0.5 * self.mesh.spacing, 0) / self.paint_falloff_mm, 0, 1)
                wgt = wgt[:, None].astype(np.float32)
                grey = np.float32(0.55)
                paint = lambda v: colorize(v, vmin, vmax, cmap) * wgt + grey * (1 - wgt)  # noqa: E731
                pv_api = UsdGeom.PrimvarsAPI(prim)
                col = pv_api.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
                col.SetInterpolation(UsdGeom.Tokens.vertex)
                raw = pv_api.CreatePrimvar(f"cardiosolv:{name}", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex)
                col.GetAttr().Clear()
                if static:
                    v = info["map"].apply(vals)
                    col.Set(Vt.Vec3fArray.FromNumpy(paint(v)))
                    raw.Set(Vt.FloatArray.FromNumpy(v.astype(np.float32)))
                else:
                    for code, fv in zip(codes, vals):
                        v = info["map"].apply(fv)
                        col.Set(Vt.Vec3fArray.FromNumpy(paint(v)), Usd.TimeCode(float(code)))
                        raw.Set(Vt.FloatArray.FromNumpy(v.astype(np.float32)), Usd.TimeCode(float(code)))
                prim.SetCustomDataByKey("cardiosolv:displayed_field", name)
                prim.SetCustomDataByKey("cardiosolv:displayed_range", Gf.Vec2d(vmin, vmax))
                prim.SetCustomDataByKey("cardiosolv:displayed_units", FIELD_STYLE.get(name, ("",))[0])
        return vmin, vmax

    def write_fibers(self, coords, max_fibers=2500, length_factor=0.9, path="/CardioSolv/Debug/Fibers"):
        mm = self.scan.mm_per_unit
        E = self.mesh.n_tets
        sel = np.random.default_rng(0).choice(E, size=min(max_fibers, E), replace=False)
        c = self.mesh.points[self.mesh.tets[sel]].mean(1)
        f = coords.fiber[sel]
        half = 0.5 * length_factor * self.mesh.spacing
        p0, p1 = self.scan.to_world(c - f * half) / mm, self.scan.to_world(c + f * half) / mm
        pts = np.stack([p0, p1], 1).reshape(-1, 3).astype(np.float32)
        helix = np.degrees(np.arctan2(np.einsum("ij,ij->i", f, coords.e_l[sel]), np.einsum("ij,ij->i", f, coords.e_c[sel])))
        helix = (helix + 90) % 180 - 90
        colors = colorize(helix, -70, 70, "coolwarm")
        with Usd.EditContext(self.stage, self.layer):
            curves = UsdGeom.BasisCurves.Define(self.stage, path)
            curves.CreateTypeAttr(UsdGeom.Tokens.linear)
            curves.CreateCurveVertexCountsAttr(Vt.IntArray([2] * len(sel)))
            curves.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts))
            curves.CreateWidthsAttr(Vt.FloatArray([0.25 / self.scan.sim_scale / mm]))
            curves.GetWidthsAttr().SetMetadata("interpolation", UsdGeom.Tokens.constant)
            pv = UsdGeom.PrimvarsAPI(curves.GetPrim()).CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray,
                                                                     UsdGeom.Tokens.uniform)
            pv.Set(Vt.Vec3fArray.FromNumpy(colors))
            curves.GetPrim().SetCustomDataByKey("cardiosolv:debug_visualization", True)
            UsdGeom.Imageable(curves.GetPrim()).CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited)
        return path

    def write_assumed_endocardium(self, path="/CardioSolv/Debug/AssumedEndocardium"):
        """Debug mesh of the derived (not imaged) LV endocardium - generated geometry, flagged as such."""
        from ..core.geometry_layer import ENDO

        faces = self.mesh.boundary_faces[self.mesh.face_labels == ENDO]
        if len(faces) == 0:
            return None
        used, inv = np.unique(faces, return_inverse=True)
        pts = self.scan.to_world(self.mesh.points[used]) / self.scan.mm_per_unit
        with Usd.EditContext(self.stage, self.layer):
            m = UsdGeom.Mesh.Define(self.stage, path)
            m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
            m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
            m.CreateFaceVertexIndicesAttr(Vt.IntArray(inv.reshape(-1).astype(int).tolist()))
            m.CreateDisplayColorAttr([Gf.Vec3f(0.95, 0.45, 0.45)])
            m.CreateDisplayOpacityAttr([0.6])
            m.GetPrim().SetCustomDataByKey("cardiosolv:generated", True)
            m.GetPrim().SetCustomDataByKey("cardiosolv:assumed_anatomy", True)
            m.GetPrim().SetCustomDataByKey("cardiosolv:debug_visualization", True)
        return path

    def clear(self):
        """Remove all CardioSolv result opinions (restores the original look/motion)."""
        session = self.stage.GetSessionLayer()
        for path in self.prims:
            spec = self.layer.GetPrimAtPath(path)
            if spec:
                parent = spec.nameParent
                del parent.nameChildren[spec.name]
            for attr in ("points", "extent"):
                aspec = session.GetAttributeAtPath(Sdf.Path(path).AppendProperty(attr))
                if aspec:
                    aspec.owner.RemoveProperty(aspec)


def export_twin_stage(stage: Usd.Stage, out_path: str, writer: "TwinResultsWriter" = None) -> str:
    """Save a thin stage ``[twin results, CardioSolv semantics, original root]``.

    The original file stays untouched and becomes the weakest sublayer, so the twin's
    animation wins even for meshes defined directly in it.
    """
    import os

    root = stage.GetRootLayer()
    results = find_layer(stage, RESULTS_TAG)
    semantic = find_layer(stage, LAYER_TAG)
    if results is None:
        raise RuntimeError("No CardioSolv twin layer on this stage; build the twin first.")
    session = stage.GetSessionLayer()
    if writer is not None:
        for path in writer.root_owned_paths():
            for attr in ("points", "extent"):
                src = Sdf.Path(path).AppendProperty(attr)
                if session.GetAttributeAtPath(src):
                    Sdf.CreatePrimInLayer(results, path)
                    Sdf.CopySpec(session, src, results, src)
    results.startTimeCode = session.startTimeCode if session.HasStartTimeCode() else results.startTimeCode
    results.endTimeCode = session.endTimeCode if session.HasEndTimeCode() else results.endTimeCode
    for layer in (results, semantic):
        if layer is not None and not layer.anonymous:
            layer.Save()
    out_dir = os.path.dirname(os.path.abspath(out_path))
    twin = Sdf.Layer.CreateNew(out_path)
    rel = lambda layer: os.path.relpath(layer.realPath, out_dir) if layer.realPath else layer.identifier  # noqa: E731
    twin.subLayerPaths = [rel(l) for l in (results, semantic) if l is not None] + [rel(root)]
    twin.startTimeCode, twin.endTimeCode = results.startTimeCode, results.endTimeCode
    twin.timeCodesPerSecond = stage.GetTimeCodesPerSecond()
    twin.framesPerSecond = stage.GetFramesPerSecond()
    if root.defaultPrim:
        twin.defaultPrim = root.defaultPrim
    up = UsdGeom.GetStageUpAxis(stage)
    twin.Save()
    twin_stage = Usd.Stage.Open(out_path)
    UsdGeom.SetStageUpAxis(twin_stage, up)
    UsdGeom.SetStageMetersPerUnit(twin_stage, UsdGeom.GetStageMetersPerUnit(stage))
    twin_stage.GetRootLayer().Save()
    return out_path
