"""NVIDIA NV-Segment-CTMR (VISTA3D architecture, CT + MRI, 345+ classes).

Runs the upstream MONAI bundle through its documented entry point, exactly as
the NVIDIA medical-AI-skills ``nv-segment-ctmr`` wrapper does::

    python -m monai.bundle run --config_file configs/inference.json \\
        --input_dict "{'image': ..., 'label_prompt': [...]}" --output_dir OUT --modality CT_BODY|MRI_BODY

``NV_SEGMENT_CTMR_ROOT`` must point at ``<NV-Segment-CTMR checkout>/NV-Segment-CTMR`` with
``models/model.pt`` (see ``scripts/setup_models.sh``). Weights: NVIDIA non-commercial licence.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from .. import labels as L
from .base import EngineResult

DEFAULT_ROOTS = [
    "~/.cache/nvidia-skills/upstreams/NV-Segment-CTMR-cb921f5/NV-Segment-CTMR",
    "/opt/NV-Segment-CTMR/NV-Segment-CTMR",
    "./NV-Segment-CTMR/NV-Segment-CTMR",
]


def resolve_root(root=None) -> Path:
    cands = [root, os.environ.get("NV_SEGMENT_CTMR_ROOT")] + DEFAULT_ROOTS
    for c in cands:
        if not c:
            continue
        p = Path(c).expanduser()
        if (p / "configs" / "inference.json").is_file():
            return p
    raise RuntimeError("NV-Segment-CTMR bundle not found: set NV_SEGMENT_CTMR_ROOT to "
                       "<NV-Segment-CTMR checkout>/NV-Segment-CTMR (scripts/setup_models.sh)")


def _strip(path: Path):
    name = path.name
    for suf in (".nii.gz", ".nii"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return path.stem


class NVSegmentCTMR:
    name = "nv-segment-ctmr"

    def check(self, root=None):
        r = resolve_root(root)
        if not (r / "models" / "model.pt").is_file():
            return f"bundle at {r} but models/model.pt missing (run scripts/setup_models.sh)"
        import monai  # noqa: F401

        return "ok"

    def segment(self, image: str, modality: str, out_dir: str, root=None, prompts=None, timeout=3600,
                python=None) -> EngineResult:
        root = resolve_root(root)
        if not (root / "models" / "model.pt").is_file():
            raise RuntimeError(f"NV-Segment-CTMR weights missing: {root / 'models' / 'model.pt'}")
        mod = "CT_BODY" if modality.upper().startswith("CT") else "MRI_BODY"
        prompts = prompts or L.NV_SEGMENT_CTMR_PROMPTS["CT" if mod == "CT_BODY" else "MR"]
        out_dir = Path(out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        image = str(Path(image).resolve())
        input_dict = {"image": image, "label_prompt": list(prompts)}
        cmd = [python or os.environ.get("NV_SEGMENT_CTMR_PYTHON") or sys.executable, "-m", "monai.bundle", "run", "--config_file", "configs/inference.json",
               "--input_dict", repr(input_dict), "--output_dir", str(out_dir), "--modality", mod]
        env = dict(os.environ)
        env.setdefault("MONAI_DATA_DIRECTORY", str(out_dir / "_monai_data"))
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,expandable_segments:True")
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(root), env=env, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError(f"NV-Segment-CTMR failed (rc {proc.returncode}): {proc.stderr[-2000:]}")
        stem = _strip(Path(image))
        cands = [out_dir / stem / f"{stem}_trans.nii.gz", out_dir / f"{stem}_trans.nii.gz"]
        found = next((c for c in cands if c.is_file()), None)
        if found is None:
            newest = sorted((p for p in out_dir.rglob("*.nii*") if p.stat().st_mtime >= t0 - 1),
                            key=lambda p: p.stat().st_mtime, reverse=True)
            found = newest[0] if newest else None
        if found is None:
            raise RuntimeError("NV-Segment-CTMR produced no label map")
        mapping = {k: v for k, v in L.NV_SEGMENT_CTMR.items() if k in prompts}
        return EngineResult(self.name, str(found), mapping,
                            {"modality": mod, "label_prompts": list(prompts), "runtime_s": round(time.time() - t0, 1),
                             "bundle_root": str(root), "command": " ".join(cmd[:6]) + " ..."})
