"""Engineering QC of the reconstruction (not a clinical measurement)."""

from __future__ import annotations

# broad adult ranges (end-diastole); outside -> warning, never an error
RANGES_ML = {
    "heart_ventricle_left": (40, 350),
    "heart_ventricle_right": (40, 400),
    "heart_atrium_left": (15, 250),
    "heart_atrium_right": (15, 250),
    "heart_myocardium": (45, 350),  # volume; mass = 1.055 g/mL
}
REQUIRED_FOR_TWIN = ["heart_myocardium", "heart_ventricle_left"]


def assess(structures: dict, modality: str) -> dict:
    warnings, missing = [], [n for n in REQUIRED_FOR_TWIN if n not in structures]
    for n in missing:
        warnings.append(f"{n} not segmented: CardioSolv will derive it (lower confidence) or use skin mode.")
    for name, (lo, hi) in RANGES_ML.items():
        v = structures.get(name, {}).get("volume_ml")
        if v is not None and not lo <= v <= hi:
            warnings.append(f"{name} volume {v:.0f} mL outside the broad adult range [{lo}, {hi}] mL: "
                            f"check the series/phase or segmentation.")
    for name, info in structures.items():
        topo = info.get("surface", {})
        if topo and not topo.get("watertight", True):
            warnings.append(f"{name} surface is not watertight ({topo.get('boundary_edges')} open edges).")
    lv = structures.get("heart_ventricle_left", {}).get("volume_ml")
    myo = structures.get("heart_myocardium", {}).get("volume_ml")
    metrics = {}
    if lv:
        metrics["lv_cavity_ml"] = lv
    if myo:
        metrics["lv_mass_g"] = round(myo * 1.055, 1)
    status = "READY_FOR_TWIN" if not missing else ("PARTIAL" if structures else "FAILED")
    return {"status": status, "missing_for_twin": missing, "warnings": warnings, "metrics": metrics,
            "disclaimer": "Engineering reconstruction for simulation; not for diagnosis."}
