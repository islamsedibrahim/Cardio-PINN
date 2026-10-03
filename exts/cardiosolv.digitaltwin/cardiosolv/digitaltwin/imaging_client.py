"""Stage 0 client: DICOM CT/MR -> segmentation -> heart USD loaded into the current stage.

Segmentation runs outside Isaac Sim's Python (it needs its own CUDA/MONAI/nnU-Net
environment), either

* **local**: ``<imaging python> -m cardiosolv_imaging.cli INPUT -o OUT --json`` (``imaging/`` package), or
* **service**: the ``cardiosolv_imaging.server`` REST API on a GPU host.

The resulting ``heart.usda`` is *referenced* into the stage (so CardioSolv's
sublayers can animate it) and its ``/World/Patient/Heart`` prim becomes the
source for Stage 1.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import time
import urllib.request
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


@dataclass
class ImagingConfig:
    mode: str = "local"  # "local" | "service"
    modality: str = "auto"  # auto | CT | MR
    engine: str = "auto"  # auto | nv-segment | totalsegmentator | both
    phase: str = "ed"
    iso_mm: float = 1.0
    python: str = "python"  # interpreter with cardiosolv_imaging + engines installed
    package_dir: Optional[str] = None  # path of the imaging/ folder if not pip-installed
    service_url: str = "http://localhost:8040"
    timeout_s: float = 3600.0


def run_local(input_path, out_dir, cfg: ImagingConfig, log: Callable = print) -> dict:
    cmd = [cfg.python, "-m", "cardiosolv_imaging.cli", str(input_path), "-o", str(out_dir), "--json",
           "--modality", cfg.modality, "--engine", cfg.engine, "--phase", cfg.phase, "--iso", str(cfg.iso_mm)]
    env = dict(os.environ)
    if cfg.package_dir:
        env["PYTHONPATH"] = os.pathsep.join([cfg.package_dir, env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    log(f"[imaging] local: {' '.join(cmd[:4])} ...")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=cfg.timeout_s)
    for line in proc.stderr.splitlines()[-20:]:
        if line.startswith("[imaging]"):
            log(line)
    if proc.returncode != 0:
        raise RuntimeError(f"segmentation failed: {proc.stderr[-1500:]}")
    report = json.loads(proc.stdout.strip().splitlines()[-1])
    return report


def _multipart(fields: dict, file_field: str, filename: str, payload: bytes):
    boundary = uuid.uuid4().hex
    out = io.BytesIO()
    for k, v in fields.items():
        out.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    out.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{file_field}\"; filename=\"{filename}\"\r\n"
              f"Content-Type: application/octet-stream\r\n\r\n".encode())
    out.write(payload)
    out.write(f"\r\n--{boundary}--\r\n".encode())
    return out.getvalue(), f"multipart/form-data; boundary={boundary}"


def _zip_input(input_path: Path):
    if input_path.is_file():
        return input_path.name, input_path.read_bytes()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in input_path.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(input_path))
    return "study.zip", buf.getvalue()


def run_service(input_path, out_dir, cfg: ImagingConfig, log: Callable = print, poll_s=2.0) -> dict:
    base = cfg.service_url.rstrip("/")
    name, payload = _zip_input(Path(input_path))
    body, ctype = _multipart({"modality": cfg.modality, "engine": cfg.engine, "phase": cfg.phase,
                              "iso_mm": cfg.iso_mm}, "file", name, payload)
    log(f"[imaging] uploading {len(payload) / 1e6:.1f} MB to {base}")
    req = urllib.request.Request(f"{base}/segment", data=body, method="POST", headers={"Content-Type": ctype})
    with urllib.request.urlopen(req, timeout=300) as r:
        job = json.loads(r.read())["job"]
    t0, seen = time.time(), 0
    while True:
        with urllib.request.urlopen(f"{base}/jobs/{job}", timeout=30) as r:
            st = json.loads(r.read())
        for line in st.get("log", [])[seen:]:
            log(line)
        seen = len(st.get("log", []))
        if st["status"] in ("done", "failed"):
            break
        if time.time() - t0 > cfg.timeout_s:
            raise TimeoutError("segmentation service timed out")
        time.sleep(poll_s)
    if st["status"] != "done":
        raise RuntimeError(f"segmentation failed on the service: {st.get('error')}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"{base}/jobs/{job}/bundle.zip", timeout=300) as r:
        zipfile.ZipFile(io.BytesIO(r.read())).extractall(out)
    report = json.loads((out / "imaging_report.json").read_text())
    report["outputs"]["usd"] = str(out / "heart.usda")
    report["outputs"]["label_map"] = str(out / "heart_labels.nii.gz")
    return report


def segment(input_path, out_dir, cfg: ImagingConfig, log: Callable = print) -> dict:
    return (run_service if cfg.mode == "service" else run_local)(input_path, out_dir, cfg, log)


def load_heart_into_stage(stage, usd_path, prim_path="/World/CardioSolvPatient") -> str:
    """Reference the reconstructed heart and return the heart prim path for Stage 1.

    heart.usda is authored in metres; the referencing prim is scaled to the stage's units."""
    from pxr import Gf, Sdf, UsdGeom

    if not stage.GetPrimAtPath("/World"):
        UsdGeom.Xform.Define(stage, "/World")
    xf = UsdGeom.Xform.Define(stage, prim_path)
    prim = xf.GetPrim()
    prim.GetReferences().ClearReferences()
    prim.GetReferences().AddReference(str(usd_path))
    mpu = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
    xf.ClearXformOpOrder()
    if abs(mpu - 1.0) > 1e-9:
        xf.AddScaleOp().Set(Gf.Vec3f(*(3 * [1.0 / mpu])))
    if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y:  # heart.usda is Z-up (patient superior)
        xf.AddRotateXOp().Set(-90.0)
    prim.SetCustomDataByKey("cardiosolv:imaging_usd", str(usd_path))
    heart = Sdf.Path(prim_path).AppendPath("Patient/Heart")
    if not stage.GetPrimAtPath(heart):
        raise RuntimeError(f"{usd_path} has no /World/Patient/Heart prim")
    return str(heart)
