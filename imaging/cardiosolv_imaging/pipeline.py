"""Stage 0: DICOM (CT / MR) -> cardiac segmentation -> 3D heart USD for CardioSolv."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from . import labels as L
from .engines import get_engine
from .fusion import clean, fuse, resample_isotropic
from .qc import assess
from .surface import mask_to_surface, topology
from .usd_export import write_heart_usd


def _is_nifti(p: Path):
    return p.is_file() and (p.name.endswith(".nii") or p.name.endswith(".nii.gz"))


def choose_engines(engine: str, modality: str) -> List[str]:
    """``auto``: NV-Segment-CTMR always; TotalSegmentator heartchambers_highres first on CT when licensed."""
    if engine != "auto":
        return {"both": ["totalsegmentator", "nv-segment"]}.get(engine, [engine])
    order = []
    if modality == "CT":
        try:
            from .engines.totalseg import TotalSegmentatorEngine

            if TotalSegmentatorEngine.license():
                order.append("totalsegmentator")
        except Exception:
            pass
    order.append("nv-segment")
    return order


def run(input_path, out_dir, modality: str = "auto", engine: str = "auto", label_map: Optional[str] = None,
        mapping=None, phase: str = "ed", iso_mm: float = 1.0, device: str = "gpu", log=print) -> dict:
    import nibabel as nib

    t0 = time.time()
    inp, out = Path(input_path), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {"cardiosolv_imaging": "0.1.0", "steps": {}}

    # 1. DICOM -> NIfTI ---------------------------------------------------
    if _is_nifti(inp):
        image = inp
        mod = modality.upper() if modality != "auto" else "CT"
        report["steps"]["input"] = {"nifti": inp.name, "modality": mod}
        if modality == "auto":
            data = np.asarray(nib.load(str(inp)).dataobj)
            mod = "CT" if data.min() < -500 else "MR"  # HU air = -1000
            report["steps"]["input"]["modality_inferred_from_intensities"] = True
    else:
        from .dicom_io import dicom_to_nifti

        log("[imaging] scanning DICOM series ...")
        info = dicom_to_nifti(inp, out / "image.nii.gz", None if modality == "auto" else modality, phase)
        image, mod = Path(info["nifti"]), info["modality"]
        report["steps"]["dicom"] = info
        log(f"[imaging] {mod} series: {info['series']['slice_positions']} slices, "
            f"{info['series']['pixel_spacing_mm']} mm, phase={phase}")
    mod = "CT" if mod.upper().startswith("CT") else "MR"
    report["modality"] = mod

    # 2. segmentation engines ---------------------------------------------
    engines = ["labelmap"] if label_map else choose_engines(engine, mod)
    results = []
    for name in engines:
        eng = get_engine(name)
        log(f"[imaging] segmenting with {name} ...")
        seg_dir = out / f"seg_{name}"
        if name == "labelmap":
            res = eng.segment(str(image), mod, str(seg_dir), label_map=label_map, mapping=mapping)
        elif name == "totalsegmentator":
            res = eng.segment(str(image), mod, str(seg_dir), device=device)
        else:
            res = eng.segment(str(image), mod, str(seg_dir))
        results.append(res)
        report["steps"].setdefault("engines", []).append({"engine": res.engine, **res.details,
                                                          "labels": sorted(set(res.mapping.values()))})

    # 3. fusion + cleanup + isotropic resampling ---------------------------
    ref = nib.load(str(image))
    fused = fuse(results, ref)
    spacing = np.sqrt((ref.affine[:3, :3] ** 2).sum(0))
    cleanup = clean(fused, spacing)
    fused_iso, aff_iso = resample_isotropic(fused, ref.affine, iso_mm)
    nib.save(nib.Nifti1Image(fused_iso.astype(np.int16), aff_iso), str(out / "heart_labels.nii.gz"))
    (out / "heart_labels.json").write_text(json.dumps({s.label: s.name for s in L.STRUCTURES}, indent=1))

    # 4. surfaces + USD ------------------------------------------------------
    iso_sp = np.sqrt((aff_iso[:3, :3] ** 2).sum(0))
    vox_ml = float(np.prod(iso_sp)) / 1000.0
    surfaces, structures = {}, {}
    for s in L.STRUCTURES:
        m = fused_iso == s.label
        if not m.any():
            continue
        pts, faces = mask_to_surface(m, aff_iso)
        if pts is None:
            continue
        surfaces[s.name] = (pts, faces)
        structures[s.name] = {"label": s.label, "volume_ml": round(float(m.sum() * vox_ml), 2),
                              "vertices": int(len(pts)), "faces": int(len(faces)), "surface": topology(faces),
                              **({"cleanup": cleanup[s.name]} if s.name in cleanup else {})}
    if not surfaces:
        raise RuntimeError("Segmentation contains no cardiac structures.")
    usd_path = write_heart_usd(out / "heart.usda", surfaces,
                               {"modality": mod, "engines": "+".join(r.engine for r in results), "phase": phase})
    report["structures"] = structures
    report["qc"] = assess(structures, mod)
    report["outputs"] = {"usd": usd_path, "label_map": str(out / "heart_labels.nii.gz"), "image": str(image),
                         "heart_prim": "/World/Patient/Heart"}
    report["runtime_s"] = round(time.time() - t0, 1)
    (out / "imaging_report.json").write_text(json.dumps(report, indent=2, default=str))
    log(f"[imaging] {len(surfaces)} structures -> {usd_path} ({report['qc']['status']})")
    return report
