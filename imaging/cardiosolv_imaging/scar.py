"""Myocardial scar from late gadolinium enhancement (LGE) MR.

The myocardium of the anatomical segmentation (CT or cine MR of the same study)
is resampled onto the LGE grid, which shares the scanner's patient coordinates,
and the LGE intensities inside it are classified:

* ``nsd`` (default): the remote (healthy) myocardium is estimated by iterative
  2-SD clipping of the myocardial intensity histogram; scar = intensity >
  remote mean + ``n_sd`` x SD; within the scar, the dense core is >= 50 % of the
  maximum scar intensity (full width at half maximum) and the rest is border /
  grey zone (Schmidt et al., Circulation 2007).
* ``fwhm``: scar >= ``fwhm_bz`` (35 %) of the maximum myocardial intensity, core >=
  50 % (Roes et al., Circ CV Imaging 2009).
* an existing scar label map (``labels``): 1 = border zone, 2 = core.

Endocardial partial-volume rims (bright blood bleeding into the first
myocardial voxel) are removed: scar components lying almost entirely in the
one-voxel endocardial border are dropped, true subendocardial scar extends
deeper and is kept. Components smaller than ``min_component_ml`` are removed.

The class map is carried to the isotropic heart grid by signed-distance
interpolation (thick LGE slices), clipped to the myocardium, and written as
``/World/Patient/Scar/{scar_core, scar_border_zone}`` in heart.usda, next to the
heart, where the CardioSolv extension picks it up.

No registration is performed: LGE and anatomy must come from the same
examination (breath-hold differences shift the scar by a few millimetres).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import ndimage

BZ, CORE = 1, 2


def remote_statistics(values: np.ndarray, clip_sd=2.0, iters=8):
    """Mean / SD of healthy myocardium by iterative sigma clipping (scar is the bright tail)."""
    v = np.asarray(values, float)
    m = np.ones(len(v), bool)
    mu, sd = float(v.mean()), float(v.std())
    for _ in range(iters):
        mu, sd = float(v[m].mean()), float(v[m].std())
        new = v < mu + clip_sd * sd
        if new.sum() < 20 or np.array_equal(new, m):
            break
        m = new
    return mu, max(sd, 1e-6)


def classify_lge(lge: np.ndarray, myo: np.ndarray, blood: Optional[np.ndarray] = None, method="nsd", n_sd=3.0,
                 fwhm_core=0.5, fwhm_bz=0.35, voxel_ml=0.001, min_component_ml=0.1):
    """Scar classes (0/1/2) on the LGE grid inside ``myo``; returns (labels, details)."""
    vals = lge[myo]
    if vals.size < 50:
        raise RuntimeError("Too few myocardial voxels on the LGE grid (is the LGE from the same study?)")
    mu, sd = remote_statistics(vals)
    vmax = float(np.percentile(vals, 99.5))
    if method == "nsd":
        scar = myo & (lge > mu + n_sd * sd)
        smax = float(np.percentile(lge[scar], 99.5)) if scar.any() else vmax
        core = scar & (lge >= fwhm_core * smax)
        thr = {"remote_mean": mu, "remote_sd": sd, "scar_threshold": mu + n_sd * sd, "core_threshold": fwhm_core * smax}
    elif method == "fwhm":
        scar = myo & (lge >= fwhm_bz * vmax)
        core = scar & (lge >= fwhm_core * vmax)
        thr = {"remote_mean": mu, "remote_sd": sd, "scar_threshold": fwhm_bz * vmax, "core_threshold": fwhm_core * vmax}
    else:
        raise ValueError(f"unknown scar method {method}")
    # endocardial partial volume: drop components that live in the 1-voxel rim next to the blood pool
    rim = np.zeros_like(myo)
    if blood is not None and blood.any():
        st = np.zeros((3, 3, 3), bool)
        st[:, :, 1] = ndimage.generate_binary_structure(2, 1)  # in-plane neighbours only
        rim = myo & ndimage.binary_dilation(blood, structure=st)
    lab_cc, n = ndimage.label(scar, structure=np.ones((3, 3, 3)))
    removed = {"small": 0, "endocardial_rim": 0}
    if n:
        idx = np.arange(1, n + 1)
        size = ndimage.sum(np.ones_like(lab_cc), lab_cc, idx)
        in_rim = ndimage.sum(rim, lab_cc, idx)
        drop = np.zeros(n + 1, bool)
        small = size * voxel_ml < min_component_ml
        rimmed = in_rim / np.maximum(size, 1) > 0.8
        drop[1:] = small | rimmed
        removed = {"small": int(small.sum()), "endocardial_rim": int((rimmed & ~small).sum())}
        scar &= ~drop[lab_cc]
        core &= scar
    out = np.zeros(lge.shape, np.int8)
    out[scar] = BZ
    out[core] = CORE
    det = {"method": method, "n_sd": n_sd if method == "nsd" else None, "thresholds": thr,
           "removed_components": removed, "myocardial_voxels": int(myo.sum())}
    return out, det


def to_iso(lab: np.ndarray, aff_src: np.ndarray, shape_dst, aff_dst: np.ndarray, myo_dst: np.ndarray):
    """Carry scar classes to another grid by signed-distance interpolation (thick slices -> iso)."""
    spacing = np.sqrt((aff_src[:3, :3] ** 2).sum(0))
    idx = np.argwhere(np.ones(shape_dst, bool))
    ras = np.c_[idx, np.ones(len(idx))] @ aff_dst.T
    ijk = (ras @ np.linalg.inv(aff_src).T)[:, :3].T
    out = np.zeros(shape_dst, np.int8)
    for code, m in ((BZ, lab >= BZ), (CORE, lab == CORE)):
        if not m.any():
            continue
        sdf = ndimage.distance_transform_edt(~m, sampling=spacing) - ndimage.distance_transform_edt(m, sampling=spacing)
        v = ndimage.map_coordinates(sdf.astype(np.float32), ijk, order=1, mode="nearest").reshape(shape_dst)
        out[(v < 0) & myo_dst] = code
    return out


def segment_scar(lge_path, myo_iso: np.ndarray, aff_iso: np.ndarray, blood_iso: Optional[np.ndarray] = None,
                 method="nsd", n_sd=3.0, labels_path: Optional[str] = None, min_component_ml=0.1):
    """LGE NIfTI (+ anatomical myocardium on the iso grid) -> scar classes on the iso grid + report."""
    import nibabel as nib
    from nibabel.processing import resample_from_to

    img = nib.load(str(lge_path))
    sp_lge = np.sqrt((img.affine[:3, :3] ** 2).sum(0))
    voxel_ml = float(np.prod(sp_lge)) / 1000.0
    myo_img = nib.Nifti1Image(myo_iso.astype(np.int16), aff_iso)
    myo = np.asarray(resample_from_to(myo_img, (img.shape[:3], img.affine), order=0).dataobj) > 0
    blood = None
    if blood_iso is not None:
        blood = np.asarray(resample_from_to(nib.Nifti1Image(blood_iso.astype(np.int16), aff_iso),
                                            (img.shape[:3], img.affine), order=0).dataobj) > 0
    if labels_path:
        ext = nib.load(str(labels_path))
        lab = np.asarray(resample_from_to(ext, (img.shape[:3], img.affine), order=0).dataobj).astype(np.int8)
        lab[~myo] = 0
        det = {"method": "labels", "source": str(labels_path)}
    else:
        data = np.asarray(img.dataobj, np.float32)
        if data.ndim > 3:
            data = data[..., 0]
        lab, det = classify_lge(data, myo, blood, method, n_sd, voxel_ml=voxel_ml, min_component_ml=min_component_ml)
    iso = to_iso(lab, img.affine, myo_iso.shape, aff_iso, myo_iso)
    vml = float(np.prod(np.sqrt((aff_iso[:3, :3] ** 2).sum(0)))) / 1000.0
    myo_ml = float(myo_iso.sum() * vml)
    core_ml, bz_ml = float((iso == CORE).sum() * vml), float((iso == BZ).sum() * vml)
    det.update({"lge_spacing_mm": sp_lge.round(3).tolist(), "core_volume_ml": round(core_ml, 2),
                "border_zone_volume_ml": round(bz_ml, 2),
                "scar_burden_pct": round(100 * (core_ml + bz_ml) / max(myo_ml, 1e-9), 2),
                "core_burden_pct": round(100 * core_ml / max(myo_ml, 1e-9), 2),
                "registration": "none (same examination, scanner coordinates)"})
    return iso, det
