"""DICOM -> NIfTI for cardiac CT and MR.

Follows the NVIDIA medical-AI-skills ``dicom-series-preflight`` /
``dicom-series-to-volume`` approach (sort by ImagePositionPatient along the
slice normal, apply RescaleSlope/Intercept, affine from
ImageOrientationPatient + PixelSpacing + slice spacing), and adds what cardiac
studies need:

* folders holding many series (localisers, cine stacks, reconstructions) are
  grouped by SeriesInstanceUID and the most suitable 3D series is selected;
* cine MR (several phases per slice position) is reduced to one cardiac
  phase - end-diastole by default (first trigger time after the R wave);
* enhanced multi-frame / compressed DICOM falls back to SimpleITK if present.

Patient identifiers are never copied into outputs: summaries contain only
geometry, modality and timing information.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])


@dataclass
class SeriesInfo:
    uid: str
    modality: str
    files: List[Path]
    rows: int
    cols: int
    pixel_spacing: tuple
    orientation: tuple
    n_positions: int
    n_phases: int
    multiframe: bool
    image_type: str
    description: str = ""
    score: float = 0.0
    notes: List[str] = field(default_factory=list)

    def summary(self, include_description=False):
        d = {"modality": self.modality, "files": len(self.files), "rows": self.rows, "cols": self.cols,
             "pixel_spacing_mm": list(self.pixel_spacing), "slice_positions": self.n_positions,
             "cardiac_phases": self.n_phases, "multiframe": self.multiframe, "score": round(self.score, 2),
             "notes": self.notes}
        if include_description:
            d["description"] = self.description
        return d


def _read_header(path: Path):
    import pydicom

    try:
        return pydicom.dcmread(str(path), stop_before_pixels=True, force=False)
    except Exception:
        return None


def scan_dicom(folder) -> List[SeriesInfo]:
    """Group every DICOM image under ``folder`` by series (headers only)."""
    groups: Dict[str, list] = defaultdict(list)
    for p in sorted(Path(folder).rglob("*")):
        if not p.is_file() or p.name.startswith("."):
            continue
        ds = _read_header(p)
        if ds is None or not hasattr(ds, "SOPClassUID"):
            continue
        if "Rows" not in ds:
            continue
        groups[str(getattr(ds, "SeriesInstanceUID", "unknown"))].append((p, ds))
    series = []
    for uid, items in groups.items():
        ds0 = items[0][1]
        mod = str(getattr(ds0, "Modality", "")).upper()
        nframes = int(getattr(ds0, "NumberOfFrames", 1) or 1)
        positions = {tuple(np.round(np.asarray(getattr(d, "ImagePositionPatient", [0, 0, i]), float), 2))
                     for i, (_, d) in enumerate(items)}
        n_pos = len(positions)
        n_phases = max(1, len(items) // max(n_pos, 1))
        s = SeriesInfo(
            uid=uid, modality=mod, files=[p for p, _ in items], rows=int(ds0.Rows), cols=int(ds0.Columns),
            pixel_spacing=tuple(float(x) for x in getattr(ds0, "PixelSpacing", [1.0, 1.0])),
            orientation=tuple(float(x) for x in getattr(ds0, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0])),
            n_positions=n_pos if nframes == 1 else nframes, n_phases=n_phases, multiframe=nframes > 1,
            image_type="\\".join(getattr(ds0, "ImageType", [])), description=str(getattr(ds0, "SeriesDescription", "")),
        )
        series.append(s)
    return series


def score_series(s: SeriesInfo, prefer: Optional[str] = None) -> float:
    """Heuristic suitability of a series for 3D cardiac segmentation."""
    score = 0.0
    if "LOCALIZER" in s.image_type.upper() or s.n_positions < 8:
        s.notes.append("localiser / too few slices")
        return -1.0
    if prefer and not s.modality.startswith(prefer[:2]):
        score -= 5.0
    if s.modality in ("CT", "MR"):
        score += 2.0
    score += min(s.n_positions, 400) / 100.0  # more coverage
    score += 1.0 / max(min(s.pixel_spacing), 0.3)  # finer in-plane resolution
    if s.modality == "MR" and s.n_phases > 1:
        score += 0.5  # cine stack: segmentable at end-diastole
        s.notes.append(f"cine: {s.n_phases} phases")
    if "DERIVED" in s.image_type.upper() and "SECONDARY" in s.image_type.upper():
        score -= 1.0
    s.score = score
    return score


def select_series(series: List[SeriesInfo], prefer: Optional[str] = None) -> SeriesInfo:
    if not series:
        raise RuntimeError("No DICOM image series found in the folder.")
    ranked = sorted(series, key=lambda s: score_series(s, prefer), reverse=True)
    if ranked[0].score < 0:
        raise RuntimeError("Only localiser / single-slice DICOM series were found.")
    return ranked[0]


def _sitk_fallback(series: SeriesInfo):
    import SimpleITK as sitk

    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(str(series.files[0].parent), series.uid)
    reader.SetFileNames(files)
    img = reader.Execute()
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # z, y, x
    data = np.transpose(arr, (2, 1, 0))
    direction = np.asarray(img.GetDirection()).reshape(3, 3)
    aff = np.eye(4)
    aff[:3, :3] = direction * np.asarray(img.GetSpacing())[None, :]
    aff[:3, 3] = img.GetOrigin()
    return data, LPS_TO_RAS @ aff, {"reader": "SimpleITK"}


def load_series(series: SeriesInfo, phase: str = "ed"):
    """Return (data[i=col, j=row, k=slice] float32, RAS affine 4x4, info)."""
    import pydicom

    if series.multiframe:
        try:
            return _sitk_fallback(series)
        except ImportError as exc:
            raise RuntimeError("Enhanced multi-frame DICOM needs SimpleITK (pip install SimpleITK).") from exc
    dss = [pydicom.dcmread(str(p)) for p in series.files]
    iop = np.asarray(dss[0].ImageOrientationPatient, float)
    r, c = iop[:3], iop[3:]
    n = np.cross(r, c)
    # group by slice position (cine: several phases per position)
    by_pos: Dict[float, list] = defaultdict(list)
    for ds in dss:
        d = float(np.asarray(ds.ImagePositionPatient, float) @ n)
        by_pos[round(d, 2)].append(ds)
    chosen, trig = [], []
    for d in sorted(by_pos):
        group = sorted(by_pos[d], key=lambda x: float(getattr(x, "TriggerTime", 0.0) or 0.0))
        if phase == "ed":
            pick = group[0]
        elif phase == "es":
            pick = group[len(group) // 3]  # ~end-systole for a typical cine (refined later by volume)
        else:
            pick = group[int(phase) % len(group)]
        chosen.append(pick)
        if "TriggerTime" in pick:
            trig.append(float(pick.TriggerTime))
    if len(chosen) < 2:
        raise RuntimeError("Series has fewer than two slice positions.")
    slices = []
    for ds in chosen:
        try:
            arr = ds.pixel_array.astype(np.float32)
        except Exception as exc:  # compressed transfer syntax without a decoder
            try:
                return _sitk_fallback(series)
            except ImportError:
                raise RuntimeError(f"Cannot decode pixel data ({exc}); install SimpleITK or pylibjpeg.") from exc
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        inter = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        slices.append(arr * slope + inter)
    vol = np.stack(slices, axis=-1)  # rows, cols, slices
    data = np.transpose(vol, (1, 0, 2))  # cols(i), rows(j), slices(k)
    pos = np.array([np.asarray(ds.ImagePositionPatient, float) for ds in chosen])
    proj = pos @ n
    steps = np.diff(proj)
    dz = float(np.median(steps))
    row_sp, col_sp = (float(x) for x in chosen[0].PixelSpacing)
    aff = np.eye(4)
    aff[:3, 0] = r * col_sp
    aff[:3, 1] = c * row_sp
    aff[:3, 2] = n * dz
    aff[:3, 3] = pos[0]
    info = {"reader": "pydicom", "slices": len(chosen), "slice_spacing_mm": abs(dz),
            "pixel_spacing_mm": [row_sp, col_sp], "phase": phase,
            "trigger_time_ms": float(np.median(trig)) if trig else None,
            "phases_available": max(len(g) for g in by_pos.values())}
    if steps.size and (np.max(np.abs(steps - dz)) > 0.1 * abs(dz) + 1e-3):
        info["warning"] = "non-uniform slice spacing (gaps); volume resampled on a uniform grid assumption"
    return data, LPS_TO_RAS @ aff, info


def dicom_to_nifti(folder, out_path, modality: Optional[str] = None, phase="ed"):
    """Select the best series in ``folder`` and write it as NIfTI. Returns a JSON-able summary."""
    import nibabel as nib

    series = scan_dicom(folder)
    best = select_series(series, modality)
    data, aff, info = load_series(best, phase)
    img = nib.Nifti1Image(data.astype(np.float32), aff)
    img.header.set_xyzt_units("mm")
    nib.save(img, str(out_path))
    mod = best.modality if best.modality in ("CT", "MR") else (modality or "CT")
    return {"nifti": str(out_path), "modality": mod, "series": best.summary(), "load": info,
            "candidates": [s.summary() for s in sorted(series, key=lambda s: s.score, reverse=True)[:6]],
            "axcodes": "".join(nib.aff2axcodes(aff))}
