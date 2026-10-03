"""Synthetic cardiac DICOM studies (CT and cine MR) with known ground truth, for tests only."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _heart_phantom import synthetic_heart_masks  # noqa: E402

# synthetic structure -> NV-Segment-CTMR label id
NV_IDS = {"myo": 154, "lv": 151, "rv": 152, "la": 149, "ra": 153, "ao": 6}


def gt_label_volume():
    """Ground-truth label volume on the fixture grid (1 mm, origin -70 mm, RAS)."""
    masks, origin, h = synthetic_heart_masks()
    lab = np.zeros(masks["myo"].shape, np.int16)
    for key in ("ao", "la", "ra", "rv", "lv", "myo"):
        lab[masks[key]] = NV_IDS[key]
    aff = np.diag([h, h, h, 1.0])
    aff[:3, 3] = origin
    return lab, aff, masks


def _ds(modality, uid_series, uid_study, k, rows, cols, iop, ipp, spacing, thickness, pixels, slope=1.0,
        intercept=0.0, trigger=None, series_desc="", image_type=("ORIGINAL", "PRIMARY", "AXIAL")):
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2" if modality == "CT" else "1.2.840.10008.5.1.4.1.1.4"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = modality
    ds.PatientName = "SYNTHETIC^PHANTOM"
    ds.PatientID = "PHANTOM-0001"
    ds.StudyInstanceUID = uid_study
    ds.SeriesInstanceUID = uid_series
    ds.SeriesDescription = series_desc
    ds.ImageType = list(image_type)
    ds.InstanceNumber = k + 1
    ds.Rows, ds.Columns = rows, cols
    ds.ImageOrientationPatient = [float(x) for x in iop]
    ds.ImagePositionPatient = [float(x) for x in ipp]
    ds.PixelSpacing = [float(spacing[0]), float(spacing[1])]
    ds.SliceThickness = float(thickness)
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
    ds.RescaleSlope, ds.RescaleIntercept = slope, intercept
    if trigger is not None:
        ds.TriggerTime = float(trigger)
    ds.PixelData = np.asarray(pixels, np.int16).tobytes()
    return ds


def _sample(volume, aff, pts_ras, order=1):
    from scipy.ndimage import map_coordinates

    ijk = (np.c_[pts_ras, np.ones(len(pts_ras))] @ np.linalg.inv(aff).T)[:, :3]
    return map_coordinates(volume, ijk.T, order=order, mode="constant", cval=0.0)


def write_ct_study(folder, with_localizer=True):
    """Contrast CT: blood 350 HU, myocardium 110, chest -100/-800; LPS axial slices, 1 mm."""
    import pydicom.uid as U

    lab, aff, masks = gt_label_volume()
    hu = np.full(lab.shape, -100.0, np.float32)
    hu[lab == 154] = 110
    for v in (151, 152, 149, 153, 6):
        hu[lab == v] = 350
    hu += np.random.default_rng(0).normal(0, 12, hu.shape)
    os.makedirs(folder, exist_ok=True)
    study, series = U.generate_uid(), U.generate_uid()
    n = lab.shape[2]
    rows = cols = lab.shape[0]
    # LPS axial: rows along +y(LPS)=posterior, cols along +x(LPS)=left
    iop = [1, 0, 0, 0, 1, 0]
    for k in range(0, n, 1):
        z = -70.0 + k
        ii, jj = np.meshgrid(np.arange(cols), np.arange(rows))
        x_lps, y_lps = -70.0 + ii * 1.0, -70.0 + jj * 1.0
        ras = np.stack([-x_lps.ravel(), -y_lps.ravel(), np.full(x_lps.size, z)], 1)
        vals = _sample(hu, aff, ras, order=0).reshape(rows, cols)
        px = np.round(vals + 1024).astype(np.int16)
        ds = _ds("CT", series, study, k, rows, cols, iop, [-70.0, -70.0, z], (1.0, 1.0), 1.0, px, 1.0, -1024.0,
                 series_desc="CTA heart 1mm")
        ds.save_as(os.path.join(folder, f"ct_{k:04d}.dcm"), enforce_file_format=True)
    if with_localizer:
        loc = U.generate_uid()
        for k in range(2):
            ds = _ds("CT", loc, study, k, 64, 64, [1, 0, 0, 0, 0, -1], [-160, 0, 160], (5.0, 5.0), 5.0,
                     np.zeros((64, 64)), image_type=("ORIGINAL", "PRIMARY", "LOCALIZER"), series_desc="scout")
            ds.save_as(os.path.join(folder, f"scout_{k}.dcm"), enforce_file_format=True)
    return folder, lab, aff


def write_cine_mr_study(folder, phases=3, slice_mm=8.0, inplane=1.5):
    """Short-axis cine (oblique), bright blood (SSFP-like): blood 900, myocardium 250, rest 120."""
    import pydicom.uid as U

    lab, aff, masks = gt_label_volume()
    os.makedirs(folder, exist_ok=True)
    study, series = U.generate_uid(), U.generate_uid()
    # short axis: normal = LV long axis (fixture +z), tilted by 20 deg about x for realism
    t = np.deg2rad(20)
    n_ax = np.array([0, np.sin(t), np.cos(t)])
    row_dir = np.array([1.0, 0, 0])
    col_dir = np.cross(n_ax, row_dir)
    rows = cols = 96
    k_count = int(80 / slice_mm) + 1
    from scipy.ndimage import binary_erosion

    for ph in range(phases):
        lvp = binary_erosion(masks["lv"], iterations=3 * ph) if ph else masks["lv"]  # systolic shrink
        sig = np.full(lab.shape, 120.0, np.float32)
        sig[lab == 154] = 250
        for v in (152, 149, 153, 6):
            sig[lab == v] = 900
        sig[lvp] = 900
        sig[masks["lv"] & ~lvp] = 250  # thickened wall
        for k in range(k_count):
            centre_ras = np.array([0.0, 0.0, -50.0]) + n_ax * (k * slice_mm)
            ii, jj = np.meshgrid(np.arange(cols), np.arange(rows))
            off = (ii - cols / 2)[..., None] * row_dir * inplane + (jj - rows / 2)[..., None] * col_dir * inplane
            ras = (centre_ras + off).reshape(-1, 3)
            vals = _sample(sig, aff, ras, order=1).reshape(rows, cols)
            px = np.round(vals).astype(np.int16)
            first_ras = centre_ras - (cols / 2) * row_dir * inplane - (rows / 2) * col_dir * inplane
            to_lps = np.array([-1, -1, 1.0])
            ds = _ds("MR", series, study, ph * k_count + k, rows, cols,
                     list(row_dir * to_lps) + list(col_dir * to_lps), list(first_ras * to_lps), (inplane, inplane),
                     slice_mm, px, trigger=ph * 300.0, series_desc="SA cine", image_type=("ORIGINAL", "PRIMARY", "M"))
            ds.save_as(os.path.join(folder, f"mr_{ph}_{k:03d}.dcm"), enforce_file_format=True)
    return folder, lab, aff


def write_lge_study(folder, slice_mm=8.0, inplane=1.5, noise_sd=10.0, seed=0):
    """Short-axis LGE (inversion recovery): nulled healthy myocardium 50, border zone 130, dense scar 330,
    blood 250, background 20, Gaussian noise; same geometry as the cine study."""
    import pydicom.uid as U

    from _heart_phantom import synthetic_scar_masks

    lab, aff, masks = gt_label_volume()
    scar, _, _ = synthetic_scar_masks()
    os.makedirs(folder, exist_ok=True)
    study, series = U.generate_uid(), U.generate_uid()
    t = np.deg2rad(20)
    n_ax = np.array([0, np.sin(t), np.cos(t)])
    row_dir = np.array([1.0, 0, 0])
    col_dir = np.cross(n_ax, row_dir)
    rows = cols = 96
    sig = np.full(lab.shape, 20.0, np.float32)
    sig[lab == 154] = 50
    for v in (151, 152, 149, 153, 6):
        sig[lab == v] = 250
    sig[scar["scar_border_zone"]] = 130
    sig[scar["scar_core"]] = 330
    rng = np.random.default_rng(seed)
    for k in range(int(80 / slice_mm) + 1):
        centre_ras = np.array([0.0, 0.0, -50.0]) + n_ax * (k * slice_mm)
        ii, jj = np.meshgrid(np.arange(cols), np.arange(rows))
        off = (ii - cols / 2)[..., None] * row_dir * inplane + (jj - rows / 2)[..., None] * col_dir * inplane
        vals = _sample(sig, aff, (centre_ras + off).reshape(-1, 3), order=1).reshape(rows, cols)
        px = np.round(np.clip(vals + rng.normal(0, noise_sd, vals.shape), 0, None)).astype(np.int16)
        first_ras = centre_ras - (cols / 2) * row_dir * inplane - (rows / 2) * col_dir * inplane
        to_lps = np.array([-1, -1, 1.0])
        ds = _ds("MR", series, study, k, rows, cols, list(row_dir * to_lps) + list(col_dir * to_lps),
                 list(first_ras * to_lps), (inplane, inplane), slice_mm, px, series_desc="LGE PSIR SA",
                 image_type=("ORIGINAL", "PRIMARY", "M"))
        ds.save_as(os.path.join(folder, f"lge_{k:03d}.dcm"), enforce_file_format=True)
    return folder, scar


FAKE_BUNDLE_PY = r'''
import ast, sys, os
import numpy as np, nibabel as nib
args = sys.argv[1:]
assert args[:3] == ["-m", "monai.bundle", "run"], args
kv = dict(zip(args[3::2], args[4::2]))
assert kv["--config_file"] == "configs/inference.json" and kv["--modality"] in ("CT_BODY", "MRI_BODY")
inp = ast.literal_eval(kv["--input_dict"])
img = nib.load(inp["image"])
gt = nib.load(os.environ["FAKE_GT"])
from nibabel.processing import resample_from_to
lab = np.asarray(resample_from_to(gt, img, order=0).dataobj).astype(np.int16)
lab[~np.isin(lab, inp["label_prompt"])] = 0
stem = os.path.basename(inp["image"]).replace(".nii.gz", "").replace(".nii", "")
out = os.path.join(kv["--output_dir"], stem); os.makedirs(out, exist_ok=True)
nib.save(nib.Nifti1Image(lab, img.affine), os.path.join(out, stem + "_trans.nii.gz"))
'''


def make_fake_bundle(root: Path, gt_path: Path):
    """A stand-in NV-Segment-CTMR bundle honouring the documented CLI contract (no GPU/weights)."""
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "models").mkdir(exist_ok=True)
    (root / "configs" / "inference.json").write_text("{}")
    (root / "models" / "model.pt").write_bytes(b"fake")
    shim = root / "fake_python.sh"
    script = root / "fake_bundle.py"
    script.write_text(FAKE_BUNDLE_PY)
    shim.write_text(f"#!/bin/sh\nexec {sys.executable} {script} \"$@\"\n")
    shim.chmod(0o755)
    os.environ["FAKE_GT"] = str(gt_path)
    return str(shim)
