"""Geometry layer: find the myocardium inside the user's mesh, separate
endocardium / epicardium / base, and estimate the long axis.

Input is the user's own anatomy (parts + role assignment); nothing is
synthesised. If the heart has no explicit myocardium part, the LV wall is
derived from the LV blood pool and its neighbours and flagged as such.

Surface classes (shared by every later stage):
    1 ENDO        LV endocardium (faces the LV blood pool)
    2 EPI         epicardium
    3 BASE        basal / valve plane
    4 RV_SEPTUM   surface facing the RV blood pool: the septum's RV side in LV-only
                  meshes, the whole RV endocardium (septal + free wall) in biventricular ones
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy import ndimage

from . import anatomy as A
from .long_axis import LongAxis, estimate_long_axis
from .mesh import SurfaceMesh, normalize
from .voxel import VoxelGrid, dilate_mm, largest_component, voxelize

UNKNOWN, ENDO, EPI, BASE, RV_SEPTUM = 0, 1, 2, 3, 4
SURFACE_NAMES = {ENDO: "Endocardium", EPI: "Epicardium", BASE: "Base", RV_SEPTUM: "RVSeptum"}

# voxel classes
V_OUT, V_MYO, V_LVC, V_RVC, V_BASE_STRUCT, V_OTHER, V_RV_MYO = 0, 1, 2, 3, 4, 5, 6
WALL_CLASSES = (V_MYO, V_RV_MYO)


@dataclass
class Landmark:
    name: str
    position: np.ndarray
    confidence: float
    method: str

    def to_dict(self):
        return {"name": self.name, "position": self.position.tolist(), "confidence": self.confidence,
                "method": self.method}


@dataclass
class GeometryLayer:
    grid: VoxelGrid
    voxel_class: np.ndarray = field(repr=False)
    myocardium_source: str  # "part" | "derived"
    myocardium_part: Optional[int]
    long_axis: LongAxis
    septum_direction: np.ndarray
    septum_confidence: float
    landmarks: Dict[str, Landmark]
    surface_labels: Optional[np.ndarray] = None  # per triangle of the myocardium part
    metrics: Dict[str, float] = field(default_factory=dict)
    confidence: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    biventricular: bool = False  # RV free wall (V_RV_MYO) is part of the simulated wall
    skin: Optional[dict] = None  # skin-mode details (assumed anatomy)

    @property
    def myo(self):
        return self.voxel_class == V_MYO

    def to_dict(self):
        return {
            "grid": {"origin": self.grid.origin.tolist(), "spacing": self.grid.spacing, "shape": list(self.grid.shape)},
            "myocardium_source": self.myocardium_source,
            "biventricular": self.biventricular,
            "long_axis": self.long_axis.to_dict(),
            "septum_direction": self.septum_direction.tolist(),
            "septum_confidence": self.septum_confidence,
            "landmarks": {k: v.to_dict() for k, v in self.landmarks.items()},
            "metrics": self.metrics,
            "confidence": self.confidence,
            "warnings": self.warnings,
            "skin_mode": self.skin,
            "surface_label_counts": (
                {SURFACE_NAMES.get(int(k), "Unknown"): int(v) for k, v in zip(*np.unique(self.surface_labels, return_counts=True))}
                if self.surface_labels is not None else {}
            ),
        }


def auto_spacing(meshes: List[SurfaceMesh], target_voxels=400_000, lo=0.6, hi=3.0) -> float:
    bmin = np.min([m.bbox_min for m in meshes], axis=0)
    bmax = np.max([m.bbox_max for m in meshes], axis=0)
    vol = float(np.prod(np.maximum(bmax - bmin, 1e-3)))
    return float(np.clip((vol / target_voxels) ** (1 / 3), lo, hi))


def _centers(grid, mask):
    return grid.centers(np.argwhere(mask))


def label_surface_faces(mesh: SurfaceMesh, grid: VoxelGrid, vclass: np.ndarray, axis: LongAxis,
                        smooth_iterations=3, wall_classes=(V_MYO,)) -> np.ndarray:
    """Classify each triangle of the user's myocardium surface.

    The tissue on the outward side of each face decides its class: LV blood
    pool -> ENDO, RV blood pool -> RV_SEPTUM, atria/great vessels -> BASE,
    otherwise EPI. Faces on the flat basal cut (normal along the long axis at
    the top of the ventricle) are BASE.
    """
    h = grid.spacing
    c = mesh.face_centers()
    n = mesh.face_normals()
    off = 0.75 * h
    plus, minus = c + n * off, c - n * off
    cls_p = grid.sample(vclass, plus, fill=V_OUT)
    cls_m = grid.sample(vclass, minus, fill=V_OUT)
    # outward side = the side that is not myocardium (robust to flipped normals)
    wall = list(wall_classes)
    use_minus = np.isin(cls_p, wall) & ~np.isin(cls_m, wall)
    outward_n = np.where(use_minus[:, None], -n, n)
    # probe outward up to ~3.5 mm: segmentation parts often leave thin gaps
    # between the wall and the neighbouring blood pools
    outward_cls = np.full(mesh.n_faces, V_OUT, np.int8)
    pending = np.ones(mesh.n_faces, bool)
    for dist in (off, 1.5 * h, max(2.5 * h, 2.0), max(3.5 * h, 3.5)):
        cls = grid.sample(vclass, c[pending] + outward_n[pending] * dist, fill=V_OUT)
        hit = (cls != V_OUT) & ~np.isin(cls, wall)
        idx = np.nonzero(pending)[0]
        outward_cls[idx[hit]] = cls[hit]
        pending[idx[hit]] = False

    labels = np.full(mesh.n_faces, UNKNOWN, np.int8)
    labels[outward_cls == V_LVC] = ENDO
    labels[outward_cls == V_RVC] = RV_SEPTUM
    labels[outward_cls == V_BASE_STRUCT] = BASE
    labels[np.isin(outward_cls, (V_OUT, V_OTHER, V_RV_MYO, V_MYO))] = EPI

    # flat basal cut: normal along +axis near the top of the ventricle
    xl = axis.project(c)
    top = np.percentile(xl, 99.5)
    basal = (outward_n @ axis.direction > 0.75) & (xl > top - 0.12)
    labels[basal & (labels != ENDO)] = BASE
    labels[basal & (labels == ENDO) & (outward_n @ axis.direction > 0.95) & (xl > top - 0.04)] = BASE

    # majority smoothing over face neighbours
    if smooth_iterations and mesh.n_faces > 10:
        adj = mesh.face_adjacency()
        for _ in range(smooth_iterations):
            votes = np.zeros((mesh.n_faces, 5))
            for k in range(5):
                votes[:, k] = adj @ (labels == k).astype(float)
            votes[:, UNKNOWN] = 0
            votes[np.arange(mesh.n_faces), labels] += 0.9  # self-vote, avoids eroding thin bands
            best = votes.argmax(1).astype(np.int8)
            has = votes.max(1) > 0
            labels = np.where(has, best, labels)
    return labels


def derive_rv_wall(rv_occ, myo, lvc, exclude, spacing, rv_wall_mm=3.5, epi_occ=None):
    """RV free wall as a shell around the RV blood pool (RV myocardium is rarely segmented)."""
    rv_myo = dilate_mm(rv_occ, rv_wall_mm, spacing) & ~rv_occ & ~myo & ~lvc & ~exclude
    if epi_occ is not None and epi_occ.any():
        rv_myo &= epi_occ
    return rv_myo


def build_geometry_layer(parts: List[SurfaceMesh], asg: A.AnatomyAssignment, spacing: Optional[float] = None,
                         wall_thickness_mm: float = 10.0, biventricular: bool = True,
                         rv_wall_mm: float = 3.5) -> GeometryLayer:
    warnings: List[str] = []
    role = asg.part
    myo_i, lv_i, rv_i = role(A.MYOCARDIUM), role(A.LV), role(A.RV)
    if myo_i is None and lv_i is None:
        raise RuntimeError("Cannot locate the LV: no myocardium or LV blood pool part was identified. "
                           "Assign the 'Myocardium' role manually in the CardioSolv panel.")

    core = [parts[i] for i in (myo_i, lv_i, rv_i) if i is not None]
    if spacing is None:
        spacing = auto_spacing(core)
    lo = np.min([m.bbox_min for m in core], axis=0)
    hi = np.max([m.bbox_max for m in core], axis=0)
    margin = 0.25 * float(np.max(hi - lo))
    grid = VoxelGrid.around(lo - margin, hi + margin, spacing, padding_voxels=2)

    def occ(r):
        i = role(r)
        return voxelize(parts[i], grid) if i is not None else np.zeros(grid.shape, bool)

    lv_occ, rv_occ = occ(A.LV), occ(A.RV)
    base_occ = occ(A.LA) | occ(A.AORTA) | occ(A.PA)
    ra_occ = occ(A.RA)
    rvm_occ = occ(A.RV_MYOCARDIUM)
    other_occ = ra_occ | (rvm_occ if not biventricular else np.zeros_like(ra_occ))
    epi_occ = occ(A.EPICARDIUM)

    # ---------------- myocardium ----------------
    if myo_i is not None:
        myo = largest_component(voxelize(parts[myo_i], grid))
        source = "part"
        if myo.sum() < 50:
            raise RuntimeError(f"Myocardium part {parts[myo_i].source_path} encloses almost no volume "
                               f"({myo.sum()} voxels at {spacing:.2f} mm); is it an open sheet?")
    else:
        myo = dilate_mm(lv_occ, wall_thickness_mm, spacing) & ~lv_occ & ~rv_occ & ~base_occ & ~other_occ
        if epi_occ.any():
            myo &= epi_occ
        myo = largest_component(myo)
        source = "derived"
        warnings.append(f"Myocardium derived from LV blood pool with {wall_thickness_mm:.0f} mm wall "
                        f"(no myocardium part found): review before simulation.")

    # ---------------- LV cavity ----------------
    derived_cav = A.derive_cavity(myo, spacing)
    if lv_i is not None:
        lvc = lv_occ & ~myo
        # close small gaps between wall and blood-pool surfaces
        lvc |= derived_cav & ndimage.binary_dilation(lv_occ, iterations=2)
        cav_method = "part"
    else:
        lvc = derived_cav & ~rv_occ & ~base_occ
        lvc = largest_component(lvc) if lvc.any() else lvc
        cav_method = "derived_enclosure"
        if lvc.sum() < 20:
            warnings.append("Could not find an LV cavity enclosed by the myocardium.")

    if source == "derived":
        # trim the dilation cap above the LV base
        axis_tmp = estimate_long_axis(_centers(grid, myo), _centers(grid, lvc),
                                      _centers(grid, base_occ) if base_occ.any() else None, spacing)
        lim = (axis_tmp.base_center - axis_tmp.apex) @ axis_tmp.direction
        proj = (_centers(grid, myo) - axis_tmp.apex) @ axis_tmp.direction
        idx = np.argwhere(myo)
        cut = idx[proj > lim + 0.5 * spacing]
        myo[tuple(cut.T)] = False
        myo = largest_component(myo)

    # ---------------- RV free wall (biventricular) ----------------
    rv_myo = None
    if biventricular and rv_occ.any():
        if rvm_occ.any():
            rv_myo = rvm_occ & ~myo & ~lvc & ~rv_occ
        else:
            rv_myo = derive_rv_wall(rv_occ, myo, lvc, base_occ | ra_occ, spacing, rv_wall_mm, epi_occ)
            warnings.append(f"RV free wall derived as a {rv_wall_mm:.1f} mm shell around the RV blood pool.")
        both = largest_component(myo | rv_myo)
        rv_myo &= both
        if rv_myo.sum() < 50:
            warnings.append("RV free wall could not be attached to the LV: simulating the LV only.")
            rv_myo = None

    return finalize_geometry_layer(
        grid, myo, lvc, rv_occ, base_occ, other_occ, epi_occ, source=source, cav_method=cav_method, rv_myo=rv_myo,
        myo_surface=parts[myo_i] if myo_i is not None else None, myo_part=myo_i,
        myo_score=asg.scores.get(A.MYOCARDIUM, 0.45 if source == "derived" else 0.0),
        named_base={name: occ(r) for name, r in (("MitralAnnulus", A.LA), ("AorticAnnulus", A.AORTA))
                    if role(r) is not None},
        wall_thickness_mm=wall_thickness_mm, warnings=warnings + list(asg.warnings))


def finalize_geometry_layer(grid, myo, lvc, rv_occ, base_occ, other_occ, epi_occ, *, source, cav_method,
                            myo_surface=None, myo_part=None, myo_score=0.0, named_base=None,
                            wall_thickness_mm=10.0, warnings=None, rv_myo=None) -> GeometryLayer:
    """Common tail of every geometry-layer mode: classes, long axis, septum, labels, landmarks, QC."""
    warnings = list(warnings or [])
    spacing = grid.spacing
    # ---------------- voxel classes ----------------
    vclass = np.zeros(grid.shape, np.int8)
    vclass[other_occ | (epi_occ & ~myo)] = V_OTHER
    vclass[base_occ] = V_BASE_STRUCT
    vclass[rv_occ] = V_RVC
    vclass[lvc] = V_LVC
    if rv_myo is not None:
        vclass[rv_myo & ~myo] = V_RV_MYO
    vclass[myo] = V_MYO

    # ---------------- long axis ----------------
    myo_pts, cav_pts = _centers(grid, myo), _centers(grid, lvc)
    base_pts = _centers(grid, base_occ) if base_occ.any() else None
    axis = estimate_long_axis(myo_pts, cav_pts, base_pts, spacing)

    # ---------------- septum direction ----------------
    border = ndimage.binary_dilation(myo, iterations=2) & ~myo
    rv_contact = border & rv_occ
    sep_conf = 0.0
    if rv_contact.sum() > 5:
        sep_pt = _centers(grid, rv_contact).mean(0)
        sep_conf = 0.9
    elif rv_occ.any():
        sep_pt = _centers(grid, rv_occ).mean(0)
        sep_conf = 0.6
    else:
        from .mesh import pca_axes

        _, ax, _ = pca_axes(myo_pts)
        sep_pt = myo_pts.mean(0) + ax[1] * 10.0
        sep_conf = 0.1
        warnings.append("No RV found: circumferential reference (septum) is arbitrary.")
    radial = sep_pt - axis.apex
    radial -= (radial @ axis.direction) * axis.direction
    septum_dir = normalize(radial)

    # ---------------- surface labels on the user's myocardium ----------------
    surface_labels = None
    if myo_surface is not None:
        surface_labels = label_surface_faces(myo_surface, grid, vclass, axis)
        counts = np.bincount(surface_labels, minlength=5)
        if counts[ENDO] == 0:
            warnings.append("No endocardial faces found on the myocardium part.")

    # ---------------- landmarks ----------------
    lms: Dict[str, Landmark] = {
        "Apex": Landmark("Apex", axis.apex, axis.confidence, "epicardial extreme along long axis"),
        "Base": Landmark("Base", axis.base_center, axis.confidence, "centre of basal LV cavity opening"),
    }
    if len(cav_pts):
        pc = cav_pts @ axis.direction
        lms["EndocardialApex"] = Landmark("EndocardialApex", cav_pts[pc <= np.percentile(pc, 1)].mean(0),
                                          axis.confidence, "LV cavity extreme")
    lv_border = ndimage.binary_dilation(lvc | myo, iterations=2)
    for name, o in (named_base or {}).items():
        contact = o & lv_border
        if contact.sum() > 3:
            lms[name] = Landmark(name, _centers(grid, contact).mean(0), 0.8, "LV contact with neighbouring part")
    if "MitralAnnulus" not in lms:
        lms["MitralAnnulus"] = Landmark("MitralAnnulus", axis.base_center, 0.4, "basal opening centre (no LA part)")
    if rv_contact.sum() > 5:
        lms["SeptumMid"] = Landmark("SeptumMid", _centers(grid, rv_contact).mean(0), 0.8, "LV wall contact with RV")

    # ---------------- metrics & plausibility ----------------
    h3 = spacing**3
    myo_ml = myo.sum() * h3 / 1000.0
    cav_ml = lvc.sum() * h3 / 1000.0
    metrics = {
        "voxel_spacing_mm": spacing,
        "myocardial_volume_ml": myo_ml,
        "myocardial_mass_g": myo_ml * 1.055,
        "lv_cavity_volume_ml": cav_ml,
        "long_axis_length_mm": axis.length_mm,
    }
    if rv_myo is not None:
        metrics["rv_wall_volume_ml"] = float((vclass == V_RV_MYO).sum() * h3 / 1000.0)
        metrics["rv_cavity_volume_ml"] = float((vclass == V_RVC).sum() * h3 / 1000.0)
    if myo_surface is not None:
        a = myo_surface.face_areas()
        lab = surface_labels
        metrics["endo_area_cm2"] = float(a[lab == ENDO].sum() / 100)
        metrics["epi_area_cm2"] = float(a[(lab == EPI) | (lab == RV_SEPTUM)].sum() / 100)
        mean_area = 0.5 * (metrics["endo_area_cm2"] + metrics["epi_area_cm2"]) * 100
        metrics["mean_wall_thickness_mm"] = float(myo_ml * 1000 / max(mean_area, 1e-9))
    else:
        metrics["mean_wall_thickness_mm"] = wall_thickness_mm

    checks = [("myocardial_mass_g", 40, 400), ("lv_cavity_volume_ml", 20, 400), ("long_axis_length_mm", 40, 160),
              ("mean_wall_thickness_mm", 3, 25)]
    for key, lo_v, hi_v in checks:
        v = metrics.get(key)
        if v is not None and not lo_v <= v <= hi_v:
            warnings.append(f"{key} = {v:.1f} is outside the physiological range [{lo_v}, {hi_v}]; "
                            f"check stage units (metersPerUnit) and part roles.")

    conf = {
        "myocardium": float(myo_score if source == "part" else min(myo_score, 0.45)),
        "endocardium": {"part": 0.9, "derived_enclosure": 0.7}.get(cav_method, 0.3),
        "long_axis": axis.confidence,
        "septum": sep_conf,
    }
    return GeometryLayer(grid=grid, voxel_class=vclass, myocardium_source=source, myocardium_part=myo_part,
                         long_axis=axis, septum_direction=septum_dir, septum_confidence=sep_conf, landmarks=lms,
                         surface_labels=surface_labels, metrics=metrics, confidence=conf, warnings=warnings,
                         biventricular=rv_myo is not None)
