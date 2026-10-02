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
                 falloff_mm=20.0):
        self.stage = stage
        self.scan = scan
        self.mesh = mesh
        self.myo_prim = myocardium_prim
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
            pmap = build_point_map(mesh, world_mm, falloff_mm=1e9 if is_myo else falloff_mm)
            self.prims[src.prim_path] = {"geom": mesh_geom, "M_inv": np.linalg.inv(M), "world_mm": world_mm,
                                         "map": pmap, "is_myo": is_myo, "n": len(local)}

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
        # Stage time range is only read from the root/session layers: author it on the root layer
        # when the user's stage has none, so the twin still plays after saving and reopening.
        root = self.stage.GetRootLayer()
        if not root.HasStartTimeCode() or root.startTimeCode == root.endTimeCode:
            root.startTimeCode = float(codes[0])
            root.endTimeCode = float(codes[-1])
        self.layer.startTimeCode = float(codes[0])
        self.layer.endTimeCode = float(codes[-1])

    def write_animation(self, codes, nodal_u: np.ndarray, animate_neighbours=True):
        """``nodal_u`` (T,N,3) mm on the computational mesh."""
        mm = self.scan.mm_per_unit
        with Usd.EditContext(self.stage, self.layer):
            for path, info in self.prims.items():
                if not info["is_myo"] and not animate_neighbours:
                    continue
                pts_attr = info["geom"].GetPointsAttr()
                ext_attr = info["geom"].GetExtentAttr()
                for code, u in zip(codes, nodal_u):
                    world = info["world_mm"] + info["map"].apply_displacement(u)
                    local = (np.c_[world / mm, np.ones(len(world))] @ info["M_inv"])[:, :3].astype(np.float32)
                    pts_attr.Set(Vt.Vec3fArray.FromNumpy(local), Usd.TimeCode(float(code)))
                    lo, hi = local.min(0), local.max(0)
                    ext_attr.Set(Vt.Vec3fArray([Gf.Vec3f(*lo.tolist()), Gf.Vec3f(*hi.tolist())]), Usd.TimeCode(float(code)))
        self.set_time_range(codes)

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
                if not info["is_myo"] and not all_parts:
                    continue
                prim = info["geom"].GetPrim()
                pv_api = UsdGeom.PrimvarsAPI(prim)
                col = pv_api.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
                col.SetInterpolation(UsdGeom.Tokens.vertex)
                raw = pv_api.CreatePrimvar(f"cardiosolv:{name}", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex)
                col.GetAttr().Clear()
                if static:
                    v = info["map"].apply(vals)
                    col.Set(Vt.Vec3fArray.FromNumpy(colorize(v, vmin, vmax, cmap)))
                    raw.Set(Vt.FloatArray.FromNumpy(v.astype(np.float32)))
                else:
                    for code, fv in zip(codes, vals):
                        v = info["map"].apply(fv)
                        col.Set(Vt.Vec3fArray.FromNumpy(colorize(v, vmin, vmax, cmap)), Usd.TimeCode(float(code)))
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
        p0, p1 = (c - f * half) / mm, (c + f * half) / mm
        pts = np.stack([p0, p1], 1).reshape(-1, 3).astype(np.float32)
        helix = np.degrees(np.arctan2(np.einsum("ij,ij->i", f, coords.e_l[sel]), np.einsum("ij,ij->i", f, coords.e_c[sel])))
        helix = (helix + 90) % 180 - 90
        colors = colorize(helix, -70, 70, "coolwarm")
        with Usd.EditContext(self.stage, self.layer):
            curves = UsdGeom.BasisCurves.Define(self.stage, path)
            curves.CreateTypeAttr(UsdGeom.Tokens.linear)
            curves.CreateCurveVertexCountsAttr(Vt.IntArray([2] * len(sel)))
            curves.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts))
            curves.CreateWidthsAttr(Vt.FloatArray([0.25 / mm]))
            curves.GetWidthsAttr().SetMetadata("interpolation", UsdGeom.Tokens.constant)
            pv = UsdGeom.PrimvarsAPI(curves.GetPrim()).CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray,
                                                                     UsdGeom.Tokens.uniform)
            pv.Set(Vt.Vec3fArray.FromNumpy(colors))
            curves.GetPrim().SetCustomDataByKey("cardiosolv:debug_visualization", True)
            UsdGeom.Imageable(curves.GetPrim()).CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited)
        return path

    def clear(self):
        """Remove all CardioSolv result opinions (restores the original look/motion)."""
        for path in self.prims:
            spec = self.layer.GetPrimAtPath(path)
            if spec:
                parent = spec.nameParent
                del parent.nameChildren[spec.name]
