"""Fuse engine label maps into one canonical cardiac label map and clean it."""

from __future__ import annotations

from typing import List

import numpy as np
from scipy import ndimage

from . import labels as L
from .engines.base import EngineResult

# structures that legitimately have several components
MULTI_COMPONENT = {"pulmonary_vein": 4, "superior_vena_cava": 1, "inferior_vena_cava": 1}


def resample_to(img, ref):
    """Nearest-neighbour resample of a label image onto ``ref``'s grid."""
    import nibabel as nib
    from nibabel.processing import resample_from_to

    if img.shape[:3] == ref.shape[:3] and np.allclose(img.affine, ref.affine, atol=1e-3):
        return np.asarray(img.dataobj).astype(np.int32)
    out = resample_from_to(nib.Nifti1Image(np.asarray(img.dataobj).astype(np.float32), img.affine),
                           (ref.shape[:3], ref.affine), order=0)
    return np.asarray(out.dataobj).astype(np.int32)


def fuse(results: List[EngineResult], ref_img) -> np.ndarray:
    """Earlier engines win; within an engine, ``labels.PRIORITY`` decides overlaps.

    The whole-heart envelope never overrides a chamber/wall label."""
    import nibabel as nib

    fused = np.zeros(ref_img.shape[:3], np.int16)
    for res in results:
        lab = resample_to(nib.load(res.label_map), ref_img)
        for name in L.PRIORITY:
            values = [v for v, n in res.mapping.items() if n == name]
            if not values:
                continue
            mask = np.isin(lab, values)
            target = L.BY_NAME[name].label
            free = (fused == 0) | ((fused == L.BY_NAME["heart"].label) & (name != "heart"))
            fused[mask & free] = target
    return fused


def clean(fused: np.ndarray, spacing, min_ml=0.3) -> dict:
    """Largest component(s), hole filling, speck removal per structure. Returns report."""
    voxel_ml = float(np.prod(spacing)) / 1000.0
    report = {}
    for s in L.STRUCTURES:
        m = fused == s.label
        if not m.any():
            continue
        lab, n = ndimage.label(m, structure=np.ones((3, 3, 3)))
        keep_n = MULTI_COMPONENT.get(s.name, 1)
        if n > keep_n:
            sizes = ndimage.sum(m, lab, np.arange(1, n + 1))
            keep = np.argsort(sizes)[::-1][:keep_n] + 1
            removed = int(m.sum() - sizes[keep - 1].sum())
            m2 = np.isin(lab, keep)
            fused[m & ~m2] = 0
            m = m2
        else:
            removed = 0
        if s.name != "heart":
            filled = ndimage.binary_fill_holes(m) & (fused == 0)
            fused[filled] = s.label
            m = fused == s.label
        if m.sum() * voxel_ml < min_ml:
            fused[m] = 0
            continue
        report[s.name] = {"components_removed_voxels": removed, "volume_ml": round(float(m.sum() * voxel_ml), 2)}
    return report


def resample_isotropic(fused: np.ndarray, affine, target_mm=1.0):
    """Shape-preserving upsampling of thick-slice data (cine MR 8-10 mm slices):
    each structure's signed distance is interpolated linearly, then re-thresholded."""
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(0))
    zoom = spacing / target_mm
    if np.all(np.abs(zoom - 1) < 0.15):
        return fused, affine
    shape = np.maximum(np.round(np.array(fused.shape) * zoom).astype(int), 1)
    out = np.zeros(shape, np.int16)
    best = np.full(shape, np.inf, np.float32)
    for s in L.STRUCTURES:
        m = fused == s.label
        if not m.any():
            continue
        sdf = (ndimage.distance_transform_edt(~m, sampling=spacing) - ndimage.distance_transform_edt(m, sampling=spacing))
        up = ndimage.zoom(sdf.astype(np.float32), shape / np.array(fused.shape), order=1, grid_mode=True,
                          mode="nearest")
        inside = (up < 0) & (up < best)
        out[inside] = s.label
        best = np.where(inside, up, best)
    new_aff = affine.copy()
    new_aff[:3, :3] = affine[:3, :3] / zoom[None, :]
    # keep the centre of voxel (0,0,0) consistent with the original first voxel corner
    new_aff[:3, 3] = affine[:3, 3] + affine[:3, :3] @ (0.5 / zoom - 0.5)
    return out, new_aff
