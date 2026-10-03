"""Canonical cardiovascular label set and engine label mappings.

Prim names follow TotalSegmentator conventions (``heart_myocardium``,
``heart_ventricle_left`` ...) so CardioSolv's anatomy engine recognises each
part with HIGH confidence in Stage 1.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Structure:
    label: int  # canonical label value in the fused label map
    name: str  # USD prim name
    title: str
    color: tuple  # display colour (linear RGB)
    cardiac: bool = True


STRUCTURES = [
    Structure(1, "heart_myocardium", "LV myocardium", (0.80, 0.25, 0.25)),
    Structure(2, "heart_ventricle_left", "LV blood pool", (0.95, 0.55, 0.55)),
    Structure(3, "heart_ventricle_right", "RV blood pool", (0.45, 0.60, 0.95)),
    Structure(4, "heart_atrium_left", "LA blood pool", (0.95, 0.70, 0.45)),
    Structure(5, "heart_atrium_right", "RA blood pool", (0.55, 0.80, 0.95)),
    Structure(6, "aorta", "Aorta", (0.90, 0.30, 0.35)),
    Structure(7, "pulmonary_artery", "Pulmonary artery", (0.35, 0.45, 0.90)),
    Structure(8, "superior_vena_cava", "Superior vena cava", (0.40, 0.55, 0.85), cardiac=False),
    Structure(9, "inferior_vena_cava", "Inferior vena cava", (0.40, 0.55, 0.85), cardiac=False),
    Structure(10, "pulmonary_vein", "Pulmonary veins", (0.90, 0.50, 0.50), cardiac=False),
    Structure(11, "atrial_appendage_left", "LA appendage", (0.95, 0.65, 0.40)),
    Structure(12, "heart", "Whole heart (envelope)", (0.85, 0.60, 0.55)),
]
BY_LABEL = {s.label: s for s in STRUCTURES}
BY_NAME = {s.name: s for s in STRUCTURES}

# NV-Segment-CTMR (configs/label_dict.json) -> canonical
NV_SEGMENT_CTMR = {
    154: "heart_myocardium",  # left ventricle myocardium (CT AutoPETAtlas, MR CMRMotion)
    151: "heart_ventricle_left",
    152: "heart_ventricle_right",
    149: "heart_atrium_left",
    153: "heart_atrium_right",
    6: "aorta",
    171: "pulmonary_artery",
    125: "superior_vena_cava",
    7: "inferior_vena_cava",
    119: "pulmonary_vein",
    108: "atrial_appendage_left",
    115: "heart",
}
# classes each modality was trained on (label_dict.json "datasets")
NV_SEGMENT_CTMR_PROMPTS = {
    "CT": [154, 151, 152, 149, 153, 6, 171, 125, 7, 119, 108, 115],
    "MR": [154, 151, 152, 149, 6, 7, 115],
}

# TotalSegmentator task -> {class name: canonical}
TOTALSEG = {
    "heartchambers_highres": {  # licensed (free academic) CT task, sub-millimetre
        "heart_myocardium": "heart_myocardium",
        "heart_atrium_left": "heart_atrium_left",
        "heart_ventricle_left": "heart_ventricle_left",
        "heart_atrium_right": "heart_atrium_right",
        "heart_ventricle_right": "heart_ventricle_right",
        "aorta": "aorta",
        "pulmonary_artery": "pulmonary_artery",
    },
    "total": {  # open CT task: vessels + heart envelope
        "heart": "heart",
        "aorta": "aorta",
        "pulmonary_vein": "pulmonary_vein",
        "superior_vena_cava": "superior_vena_cava",
        "inferior_vena_cava": "inferior_vena_cava",
        "atrial_appendage_left": "atrial_appendage_left",
    },
    "total_mr": {
        "heart": "heart",
        "aorta": "aorta",
        "inferior_vena_cava": "inferior_vena_cava",
    },
}

# Common names in user-supplied label maps (3D Slicer, MM-WHS, ACDC, ...)
ALIASES = {
    "myocardium": "heart_myocardium", "lv_myocardium": "heart_myocardium", "myo": "heart_myocardium",
    "left ventricle myocardium": "heart_myocardium", "lv myocardium": "heart_myocardium",
    "lv": "heart_ventricle_left", "left ventricle": "heart_ventricle_left", "lv_blood_pool": "heart_ventricle_left",
    "rv": "heart_ventricle_right", "right ventricle": "heart_ventricle_right",
    "la": "heart_atrium_left", "left atrium": "heart_atrium_left",
    "ra": "heart_atrium_right", "right atrium": "heart_atrium_right",
    "ao": "aorta", "ascending aorta": "aorta", "pa": "pulmonary_artery", "pulmonary trunk": "pulmonary_artery",
    "svc": "superior_vena_cava", "ivc": "inferior_vena_cava", "laa": "atrial_appendage_left",
    "left atrial appendage": "atrial_appendage_left", "pulmonary veins": "pulmonary_vein",
}
# ACDC / M&Ms cine-MR convention (1 RV, 2 myocardium, 3 LV) and MM-WHS
PRESETS = {
    "acdc": {1: "heart_ventricle_right", 2: "heart_myocardium", 3: "heart_ventricle_left"},
    "mmwhs": {500: "heart_ventricle_left", 600: "heart_ventricle_right", 420: "heart_atrium_left",
              550: "heart_atrium_right", 205: "heart_myocardium", 820: "aorta", 850: "pulmonary_artery"},
    "totalseg_heartchambers": {1: "heart_myocardium", 2: "heart_atrium_left", 3: "heart_ventricle_left",
                               4: "heart_atrium_right", 5: "heart_ventricle_right", 6: "aorta",
                               7: "pulmonary_artery"},
    "cardiosolv": {s.label: s.name for s in STRUCTURES},
}

# fusion priority when several engines label the same voxel (lower = stronger)
PRIORITY = ["heart_myocardium", "heart_ventricle_left", "heart_ventricle_right", "heart_atrium_left",
            "heart_atrium_right", "aorta", "pulmonary_artery", "atrial_appendage_left", "pulmonary_vein",
            "superior_vena_cava", "inferior_vena_cava", "heart"]


def canonical(name: str):
    key = str(name).strip().lower()
    if key in BY_NAME:
        return key
    return ALIASES.get(key.replace("_", " "), ALIASES.get(key))
