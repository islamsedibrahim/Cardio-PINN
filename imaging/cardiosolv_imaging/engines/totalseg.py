"""TotalSegmentator (nnU-Net) via its Python API.

* CT with a licence key: ``heartchambers_highres`` (myocardium, 4 chambers, aorta,
  pulmonary artery at sub-millimetre resolution; free academic licence:
  https://backend.totalsegmentator.com/license-academic/).
* Always (open): ``total`` (CT) / ``total_mr`` (MR) restricted to cardiac and great-vessel classes.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from .. import labels as L
from .base import EngineResult

CT_ROI = ["heart", "aorta", "pulmonary_vein", "superior_vena_cava", "inferior_vena_cava", "atrial_appendage_left"]
MR_ROI = ["heart", "aorta", "inferior_vena_cava"]


class TotalSegmentatorEngine:
    name = "totalsegmentator"

    def check(self):
        import totalsegmentator  # noqa: F401

        return "ok" + (" (heartchambers_highres licensed)" if self.license() else " (open tasks only)")

    @staticmethod
    def license():
        if os.environ.get("TOTALSEG_LICENSE"):
            return os.environ["TOTALSEG_LICENSE"]
        try:  # honours TOTALSEG_HOME_DIR, as set by totalseg_set_license
            from totalsegmentator.config import get_license_number

            return get_license_number() or None
        except Exception:
            return None

    def segment(self, image: str, modality: str, out_dir: str, device="gpu", fast=False, task=None) -> EngineResult:
        from totalsegmentator.python_api import totalsegmentator

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ct = modality.upper().startswith("CT")
        lic = self.license()
        if task is None:
            task = "heartchambers_highres" if (ct and lic) else ("total" if ct else "total_mr")
        out = out_dir / f"totalseg_{task}.nii.gz"
        t0 = time.time()
        kwargs = dict(ml=True, task=task, device=device, quiet=True)
        if task == "heartchambers_highres":
            kwargs["license_number"] = lic
        else:
            kwargs["roi_subset"] = CT_ROI if ct else MR_ROI
            kwargs["fast"] = fast
        totalsegmentator(image, str(out), **kwargs)
        from totalsegmentator.map_to_binary import class_map

        cmap = class_map[task]
        names = L.TOTALSEG[task]
        mapping = {int(k): names[v] for k, v in cmap.items() if v in names}
        return EngineResult(self.name, str(out), mapping,
                            {"task": task, "licensed": bool(lic), "runtime_s": round(time.time() - t0, 1)})
