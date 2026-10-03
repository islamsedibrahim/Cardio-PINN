"""REST service for a GPU host (the Isaac Sim extension calls it).

    uvicorn cardiosolv_imaging.server:app --host 0.0.0.0 --port 8040

POST /segment          multipart: file=<zip of a DICOM folder | .nii/.nii.gz>, modality, engine, phase
GET  /jobs/{id}        status + QC report
GET  /jobs/{id}/heart.usda | heart_labels.nii.gz | imaging_report.json | bundle.zip
GET  /health           engines available on this host

Uploaded studies stay on this host under CARDIOSOLV_IMAGING_WORKDIR and can be
deleted with DELETE /jobs/{id}. Run it inside your hospital network only.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import uuid
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .engines import available_engines
from .pipeline import run

WORKDIR = Path(os.environ.get("CARDIOSOLV_IMAGING_WORKDIR", tempfile.gettempdir())) / "cardiosolv_jobs"
WORKDIR.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="CardioSolv imaging", version="0.1.0")
JOBS: dict = {}
_LOCK = threading.Lock()  # one GPU job at a time


def _safe_extract(zf: zipfile.ZipFile, dest: Path):
    for m in zf.infolist():
        target = (dest / m.filename).resolve()
        if not str(target).startswith(str(dest.resolve())):
            raise HTTPException(400, "unsafe path in zip")
    zf.extractall(dest)


def _worker(job_id, inp, params):
    job = JOBS[job_id]
    try:
        with _LOCK:
            job["status"] = "running"
            rep = run(inp, job["dir"] / "out", log=lambda m: job["log"].append(str(m)), **params)
        job.update(status="done", report=rep)
    except Exception as exc:  # reported to the client
        job.update(status="failed", error=str(exc))


@app.get("/health")
def health():
    return {"status": "ok", "engines": available_engines()}


@app.post("/segment")
async def segment(file: UploadFile = File(...), modality: str = Form("auto"), engine: str = Form("auto"),
                  phase: str = Form("ed"), iso_mm: float = Form(1.0)):
    job_id = uuid.uuid4().hex[:12]
    jdir = WORKDIR / job_id
    (jdir / "input").mkdir(parents=True)
    name = Path(file.filename or "upload").name
    dst = jdir / name
    with open(dst, "wb") as f:
        shutil.copyfileobj(file.file, f)
    if name.endswith(".zip"):
        with zipfile.ZipFile(dst) as zf:
            _safe_extract(zf, jdir / "input")
        inp = jdir / "input"
    else:
        inp = dst
    JOBS[job_id] = {"status": "queued", "dir": jdir, "log": []}
    params = dict(modality=modality, engine=engine, phase=phase, iso_mm=iso_mm)
    threading.Thread(target=_worker, args=(job_id, str(inp), params), daemon=True).start()
    return {"job": job_id}


@app.get("/jobs/{job_id}")
def job(job_id: str):
    j = JOBS.get(job_id) or HTTPException(404, "unknown job")
    if isinstance(j, HTTPException):
        raise j
    out = {k: v for k, v in j.items() if k in ("status", "error", "log")}
    if j.get("report"):
        out["qc"] = j["report"]["qc"]
        out["structures"] = {k: v["volume_ml"] for k, v in j["report"]["structures"].items()}
    return out


@app.get("/jobs/{job_id}/{name}")
def job_file(job_id: str, name: str):
    j = JOBS.get(job_id)
    if not j or j.get("status") != "done":
        raise HTTPException(404, "job not finished")
    out = j["dir"] / "out"
    if name == "bundle.zip":
        z = j["dir"] / "bundle.zip"
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in ("heart.usda", "heart_labels.nii.gz", "heart_labels.json", "imaging_report.json"):
                if (out / f).exists():
                    zf.write(out / f, f)
        return FileResponse(z)
    if name not in ("heart.usda", "heart_labels.nii.gz", "heart_labels.json", "imaging_report.json"):
        raise HTTPException(404, "unknown file")
    return FileResponse(out / name)


@app.delete("/jobs/{job_id}")
def delete(job_id: str):
    j = JOBS.pop(job_id, None)
    if j:
        shutil.rmtree(j["dir"], ignore_errors=True)
    return {"deleted": bool(j)}
