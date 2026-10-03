"""Geometry layer for skin-only hearts (visual / generated / SimReady assets).

Such assets are a single closed outer surface: no chambers, septum or wall
thickness exist in the data. Skin mode treats the surface as the epicardium
and derives the ventricular interior under it from anatomical rules. Every
derived quantity is flagged ``ASSUMED``:

1. **Heart axis** - PCA of the enclosed volume; the base is the end whose
   cross-sections split into several bodies (great vessels), the apex the
   single tapering tip; the stage up-axis is a weak prior.
2. **AV plane** - the waist (local minimum of cross-sectional area) between
   ventricles and atria; ventricles lie apex-side of it.
3. **LV / RV** - the apex is formed by the LV, so the LV lies on the side of
   the ventricular cross-section towards which the apex is offset. A septal
   plane parallel to the long axis splits LV (~62 % of the width) from RV.
4. **Walls** - literature thicknesses: LV free wall 10 mm, septum 10 mm,
   RV free wall 4 mm (Lang et al., JASE 2015 reference ranges).

The simulated domain is the biventricular wall (LV wall, septum and RV free
wall, ``biventricular=True``) or the LV wall + septum only; the atria and
great vessels are carried along visually.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage

from .geometry_layer import V_BASE_STRUCT, V_OUT, VoxelGrid, finalize_geometry_layer
from .mesh import SurfaceMesh, normalize, pca_axes
from .voxel import largest_component, voxelize

UP_VECTORS = {"X": np.array([1.0, 0, 0]), "Y": np.array([0, 1.0, 0]), "Z": np.array([0, 0, 1.0])}

# skin face classes (GeomSubsets on the user's mesh)
SKIN_LV_EPI, SKIN_RV_FREE, SKIN_BASE = 1, 2, 3
SKIN_NAMES = {SKIN_LV_EPI: "LVEpicardium", SKIN_RV_FREE: "RVFreeWall", SKIN_BASE: "AtriaGreatVessels"}


@dataclass
class SkinModeParams:
    spacing_mm: float = 1.0
    lv_wall_mm: float = 10.0
    septum_mm: float = 10.0
    rv_wall_mm: float = 4.0
    lv_width_fraction: float = 0.62  # LV share of the biventricular width at mid-ventricle
    av_plane_fraction: Optional[float] = None  # override the detected AV plane (0 apex .. 1 top)
    flip_base: bool = False  # user override if the apex/base sign is wrong
    flip_lv_side: bool = False  # user override if LV/RV sides are swapped
    biventricular: bool = True  # simulate the RV free wall and RV cavity too


def _slab_components(mask, proj, lo, hi):
    sel = mask & (proj >= lo) & (proj < hi)
    if not sel.any():
        return 0
    _, n = ndimage.label(sel, structure=np.ones((3, 3, 3)))
    return n


def build_skin_geometry_layer(skin: SurfaceMesh, up_axis: str = "Z", params: SkinModeParams = None):
    prm = params or SkinModeParams()
    h = prm.spacing_mm
    grid = VoxelGrid.around(skin.bbox_min, skin.bbox_max, h, padding_voxels=4)
    S = ndimage.binary_fill_holes(largest_component(voxelize(skin, grid)))
    idx = np.argwhere(S)
    pts = grid.centers(idx)
    dist_in = ndimage.distance_transform_edt(S) * h  # distance to the skin from inside
    warnings = ["Skin-only heart: myocardium, cavities and septum are ASSUMED from anatomical rules "
                "(no internal anatomy in the source mesh)."]

    # ---------------- 1. heart axis and apex/base sign ----------------
    c, ax, _ = pca_axes(pts)
    a = ax[0]
    proj_all = np.full(grid.shape, np.nan)
    proj_all[tuple(idx.T)] = (pts - c) @ a
    pmin, pmax = np.nanmin(proj_all), np.nanmax(proj_all)
    L = pmax - pmin
    projS = np.where(S, proj_all, np.inf)
    end = 0.22 * L
    comp_lo = np.mean([_slab_components(S, projS, pmin + k * h * 3, pmin + (k + 1) * h * 3) for k in range(int(end / (3 * h)))])
    comp_hi = np.mean([_slab_components(S, projS, pmax - (k + 1) * h * 3, pmax - k * h * 3) for k in range(int(end / (3 * h)))])
    votes = []
    if abs(comp_hi - comp_lo) > 0.15:
        votes.append(("vessel_branching", np.sign(comp_hi - comp_lo), 1.0,
                      f"{comp_lo:.2f} vs {comp_hi:.2f} separate bodies per slice (- end / + end)"))
    tip_lo = (S & (projS <= pmin + 0.05 * L)).sum()
    tip_hi = (S & (projS >= pmax - 0.05 * L)).sum()
    votes.append(("tapering_tip", np.sign(tip_hi - tip_lo) or 1.0, 0.5,
                  f"{tip_lo} vs {tip_hi} voxels in the last 5 % (- end / + end)"))
    up = UP_VECTORS.get(str(up_axis).upper(), UP_VECTORS["Z"])
    votes.append(("stage_up_axis", np.sign(a @ up) or 1.0, 0.3, f"axis . up = {a @ up:+.2f}"))
    agree = sum(s * w for _, s, w, _ in votes)
    sign = 1.0 if agree >= 0 else -1.0
    if prm.flip_base:
        sign = -sign
        warnings.append("Apex/base orientation flipped by user override.")
    a = a * sign
    axis_conf = abs(agree) / sum(w for _, _, w, _ in votes)
    proj = (pts - c) @ a
    apex_pt = pts[proj <= np.percentile(proj, 0.5)].mean(0)
    xl = (proj - proj.min()) / (proj.max() - proj.min())  # 0 apex .. 1 top of vessels

    # ---------------- 2. AV plane (waist of the area profile) ----------------
    bins = np.linspace(0, 1, 51)
    area = np.histogram(xl, bins)[0].astype(float)
    area_s = ndimage.uniform_filter1d(area, 3)
    centers = 0.5 * (bins[1:] + bins[:-1])
    if prm.av_plane_fraction is not None:
        f_av, av_method = prm.av_plane_fraction, "user"
    else:
        win = (centers >= 0.40) & (centers <= 0.80)
        cand = np.nonzero(win)[0]
        k = cand[np.argmin(area_s[cand])]
        left, right = area_s[: k].max(), area_s[k + 1:].max() if k + 1 < len(area_s) else 0
        if area_s[k] < 0.92 * min(left, right):
            f_av, av_method = float(centers[k]), "area-profile waist"
        else:
            f_av, av_method = 0.62, "default fraction (no clear waist)"
            warnings.append("No clear atrioventricular waist found; AV plane set at 62 % of the heart length.")

    vent = np.zeros(grid.shape, bool)
    vent[tuple(idx[xl < f_av].T)] = True
    base_struct = S & ~vent

    # ---------------- 3. LV side and septal plane ----------------
    vidx = np.argwhere(vent)
    vpts = grid.centers(vidx)
    vproj = (vpts - apex_pt) @ a
    vlen = vproj.max()
    mid = (vproj > 0.25 * vlen) & (vproj < 0.75 * vlen)
    inplane = vpts - np.outer((vpts - apex_pt) @ a, a)
    cm = inplane[mid].mean(0)
    q = inplane[mid] - cm
    w, v = np.linalg.eigh(q.T @ q)
    d_maj = normalize(v[:, -1] - (v[:, -1] @ a) * a)
    apex_off = (apex_pt - (apex_pt @ a) * a - (cm - (cm @ a) * a)) @ d_maj
    lv_sign = np.sign(apex_off) or 1.0
    lv_conf = float(np.clip(abs(apex_off) / 6.0, 0.1, 0.8))
    if prm.flip_lv_side:
        lv_sign = -lv_sign
        warnings.append("LV/RV sides flipped by user override.")
    d_lv = d_maj * lv_sign
    e = q @ d_lv
    e_lv, e_rv = np.percentile(e, 98), np.percentile(e, 2)
    s_pos = e_lv - prm.lv_width_fraction * (e_lv - e_rv)
    p_sep = cm + s_pos * d_lv
    if lv_conf < 0.4:
        warnings.append(f"LV/RV side is uncertain (apex offset {apex_off:.1f} mm from the ventricular centre); "
                        f"use 'flip LV side' if the septum is on the wrong side.")

    # ---------------- 4. walls and cavities ----------------
    allc = grid.centers(np.argwhere(np.ones(grid.shape, bool))).reshape(grid.shape + (3,))
    sd = (allc - p_sep) @ d_lv  # + towards the LV
    lvc = vent & (dist_in > prm.lv_wall_mm) & (sd > prm.septum_mm / 2)
    rvc = vent & (dist_in > prm.rv_wall_mm) & (sd < -prm.septum_mm / 2)
    lvc = largest_component(lvc) if lvc.any() else lvc
    rvc = largest_component(rvc) if rvc.any() else rvc
    if lvc.sum() * h**3 < 5000:
        raise RuntimeError("Skin mode: the ventricular region is too small for an LV cavity with the assumed "
                           f"{prm.lv_wall_mm} mm wall (LV cavity {lvc.sum() * h ** 3 / 1000:.1f} mL).")
    myo = vent & ~lvc & ~rvc & (sd > -prm.septum_mm / 2)
    myo = largest_component(myo)
    rv_free = vent & ~rvc & ~myo & ~lvc
    rv_myo = None
    if prm.biventricular and rvc.any():
        rv_myo = rv_free & largest_component(myo | rv_free)
        if rv_myo.sum() < 50:
            rv_myo = None
            warnings.append("RV free wall is not attached to the LV wall: simulating the LV only.")

    gl = finalize_geometry_layer(
        grid, myo, lvc, rvc, base_struct, rv_free if rv_myo is None else rv_free & ~rv_myo, np.zeros_like(S),
        rv_myo=rv_myo, source="skin_assumed",
        cav_method="assumed", myo_surface=None, myo_part=None, myo_score=0.3,
        named_base={"MitralAnnulus": base_struct}, wall_thickness_mm=prm.lv_wall_mm, warnings=warnings)
    gl.metrics.update({"heart_length_mm": float(L), "av_plane_fraction": float(f_av),
                       "rv_cavity_volume_ml": float(rvc.sum() * h**3 / 1000), "skin_volume_ml": float(S.sum() * h**3 / 1000)})
    gl.confidence.update({"heart_axis_sign": float(axis_conf), "lv_side": lv_conf, "myocardium": 0.3})
    gl.skin = {"axis_votes": [dict(name=n, sign=float(s), weight=w, detail=d) for n, s, w, d in votes],
               "av_method": av_method, "lv_side_offset_mm": float(apex_off), "params": prm.__dict__.copy()}
    return gl


def label_skin_faces(skin: SurfaceMesh, gl) -> np.ndarray:
    """Which part of the derived anatomy lies under each face of the user's skin."""
    from .geometry_layer import V_LVC, V_MYO, V_OTHER, V_RV_MYO, V_RVC

    g, vc = gl.grid, gl.voxel_class
    c, n = skin.face_centers(), skin.face_normals()
    out = np.full(skin.n_faces, SKIN_RV_FREE, np.int8)
    found = np.zeros(skin.n_faces, bool)
    for dist in (0.6, 1.5, 3.0, 5.0):
        for sgn in (-1.0, 1.0):  # inward side is -n for outward faces; test both for robustness
            cls = g.sample(vc, c + sgn * dist * g.spacing * n, fill=V_OUT)
            hit = ~found & (cls != V_OUT)
            out[hit & ((cls == V_MYO) | (cls == V_LVC))] = SKIN_LV_EPI
            out[hit & np.isin(cls, (V_RVC, V_OTHER, V_RV_MYO))] = SKIN_RV_FREE
            out[hit & (cls == V_BASE_STRUCT)] = SKIN_BASE
            found |= hit
    return out
