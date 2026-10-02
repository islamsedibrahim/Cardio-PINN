"""CardioSolv canonical geometry (Phase 3): the semantic meaning of the
user's heart, independent of its USD naming. It references the source
geometry; it never contains a replacement heart."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from . import anatomy as A
from .geometry_layer import BASE, ENDO, EPI, RV_SEPTUM, GeometryLayer

SCHEMA_VERSION = "0.2.0"


@dataclass
class SurfaceComponent:
    name: str
    source_path: str
    component_type: str
    confidence: float
    face_indices: Optional[np.ndarray] = None  # USD face ids on source_path (subset of the user's mesh)
    area_cm2: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class Region:
    name: str
    region_type: str
    confidence: float
    source_path: Optional[str] = None
    outer_boundary: Optional[str] = None
    inner_boundary: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Landmark:
    name: str
    position: np.ndarray
    confidence: float
    landmark_type: str = "anatomical"
    metadata: dict = field(default_factory=dict)


@dataclass
class Axis:
    name: str
    origin: np.ndarray
    direction: np.ndarray
    confidence: float
    metadata: dict = field(default_factory=dict)


@dataclass
class CanonicalHeartGeometry:
    source_prim: str
    surfaces: Dict[str, SurfaceComponent] = field(default_factory=dict)
    regions: Dict[str, Region] = field(default_factory=dict)
    landmarks: Dict[str, Landmark] = field(default_factory=dict)
    axes: Dict[str, Axis] = field(default_factory=dict)
    roles: Dict[str, str] = field(default_factory=dict)  # role -> USD path
    metadata: dict = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> dict:
        errors: List[str] = []
        warnings: List[str] = []
        if not self.source_prim:
            errors.append("No source heart prim.")
        if "Myocardium" not in self.regions:
            errors.append("Myocardium region missing.")
        for name in ("Epicardium", "Endocardium"):
            if name not in self.surfaces:
                warnings.append(f"{name} was not identified as a surface on the source mesh.")
        if "Apex" not in self.landmarks:
            errors.append("Apex landmark missing.")
        if "LongAxis" not in self.axes:
            errors.append("Cardiac long axis missing.")
        for name, s in self.surfaces.items():
            if s.confidence < 0.70:
                warnings.append(f"{name} confidence is {s.confidence:.2f}")
        for name, r in self.regions.items():
            if r.confidence < 0.70:
                warnings.append(f"{name} region confidence is {r.confidence:.2f}")
        warnings.extend(self.metadata.get("warnings", []))
        return {"valid": not errors, "errors": errors, "warnings": warnings,
                "status": "VALID" if not errors and not warnings else
                ("REQUIRES ANATOMICAL VALIDATION" if not errors else "INVALID")}


def faces_from_triangles(tri_labels: np.ndarray, tri_to_face: np.ndarray, code: int) -> np.ndarray:
    """USD polygon ids whose triangles are mostly labelled ``code``."""
    n = int(tri_to_face.max()) + 1 if len(tri_to_face) else 0
    hits = np.bincount(tri_to_face, weights=(tri_labels == code).astype(float), minlength=n)
    tot = np.bincount(tri_to_face, minlength=n)
    return np.nonzero((tot > 0) & (hits / np.maximum(tot, 1) > 0.5))[0]


def build_canonical(source_prim, parts, sources, asg: A.AnatomyAssignment, gl: GeometryLayer) -> CanonicalHeartGeometry:
    can = CanonicalHeartGeometry(source_prim=source_prim)
    can.roles = {r: parts[i].source_path for r, i in asg.roles.items()}
    myo_i = gl.myocardium_part
    myo_conf = gl.confidence["myocardium"]
    if myo_i is not None:
        part, src = parts[myo_i], sources[myo_i]
        areas = part.face_areas()
        for code, name, ctype, conf in ((EPI, "Epicardium", "external_surface", myo_conf),
                                        (ENDO, "Endocardium", "internal_surface", gl.confidence["endocardium"]),
                                        (BASE, "Base", "basal_surface", gl.long_axis.confidence),
                                        (RV_SEPTUM, "RVSeptum", "septal_surface", gl.septum_confidence)):
            tri_sel = gl.surface_labels == code
            if not tri_sel.any():
                continue
            can.surfaces[name] = SurfaceComponent(
                name=name, source_path=src.prim_path, component_type=ctype, confidence=float(min(conf, 1.0)),
                face_indices=faces_from_triangles(gl.surface_labels, part.tri_to_face, code),
                area_cm2=float(areas[tri_sel].sum() / 100.0),
            )
        can.regions["Myocardium"] = Region(
            "Myocardium", "volume_region", myo_conf, source_path=src.prim_path,
            outer_boundary="/CardioSolv/Geometry/Heart/Epicardium",
            inner_boundary="/CardioSolv/Geometry/Heart/Endocardium",
            metadata={"source": gl.myocardium_source},
        )
    else:
        can.regions["Myocardium"] = Region(
            "Myocardium", "volume_region", myo_conf, source_path=None,
            metadata={"source": "derived", "derived_from": can.roles.get(A.LV)},
        )
    for role, key in ((A.LV, "LVBloodPool"), (A.RV, "RVBloodPool")):
        if role in asg.roles:
            can.regions[key] = Region(key, "blood_volume", asg.scores.get(role, 0.0),
                                      source_path=parts[asg.roles[role]].source_path)

    for name, lm in gl.landmarks.items():
        can.landmarks[name] = Landmark(name, np.asarray(lm.position), lm.confidence, metadata={"method": lm.method})
    ax = gl.long_axis
    can.axes["LongAxis"] = Axis("LongAxis", ax.apex, ax.direction, ax.confidence,
                                metadata={"length_mm": ax.length_mm, "votes": [v.__dict__ for v in ax.votes]})
    can.axes["SeptalReference"] = Axis("SeptalReference", ax.apex, gl.septum_direction, gl.septum_confidence)
    can.metadata = {"metrics": gl.metrics, "confidence": gl.confidence, "warnings": gl.warnings,
                    "myocardium_source": gl.myocardium_source}
    return can
