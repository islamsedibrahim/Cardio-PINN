"""Anatomical confidence engine (CardioSolv Phase 3.3/3.4).

Every mesh part found under the selected heart is scored against the
CardioSolv controlled vocabulary using explainable evidence:

* naming evidence (TotalSegmentator / VISTA-3D / MM-WHS / artist conventions),
* topology evidence (closed, manifold),
* geometric evidence computed in a shared voxel grid: hollowness (does the part
  wrap a cavity, like a myocardial wall?), containment (is the part inside
  another part's cavity?), contact with other parts and elongation.

Labels are never asserted silently: each candidate keeps its evidence,
contradictions and a confidence class, and the UI lets the user override.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy import ndimage

from .mesh import SurfaceMesh, pca_axes
from .voxel import VoxelGrid, enclosure_ratio, voxelize

# ----------------------------------------------------------------------------
# Controlled vocabulary
# ----------------------------------------------------------------------------
MYOCARDIUM = "Myocardium"  # LV myocardial wall (volume region)
LV = "LeftVentricle"  # LV blood pool / cavity
RV = "RightVentricle"
LA = "LeftAtrium"
RA = "RightAtrium"
AORTA = "Aorta"
PA = "PulmonaryArtery"
EPICARDIUM = "Epicardium"  # whole-heart outer surface
RV_MYOCARDIUM = "RightVentricularMyocardium"
OTHER = "Unknown"

CARDIAC_STRUCTURES = [MYOCARDIUM, LV, RV, LA, RA, AORTA, PA, EPICARDIUM, RV_MYOCARDIUM]

# keyword phrases; single short tokens must match whole tokens
NAME_KEYWORDS: Dict[str, List[str]] = {
    MYOCARDIUM: ["myocardium", "myocard", "myo", "lv myo", "lvm", "lv wall", "left ventricular wall",
                 "heart myocardium", "muscle", "lv muscle", "ventricular wall"],
    LV: ["left ventricle", "ventricle left", "lv", "lv blood", "lv cavity", "lv bp", "lv pool",
         "leftventricle", "lv endo", "left ventricular cavity"],
    RV: ["right ventricle", "ventricle right", "rv", "rv blood", "rv cavity", "rightventricle"],
    LA: ["left atrium", "atrium left", "la", "leftatrium", "left atrial", "laa"],
    RA: ["right atrium", "atrium right", "ra", "rightatrium", "right atrial"],
    AORTA: ["aorta", "aortic", "ao", "aa", "ascending aorta", "aortic arch"],
    PA: ["pulmonary artery", "pulmonary trunk", "pa", "pulmonaryartery", "pat"],
    EPICARDIUM: ["heart", "epicardium", "epicard", "whole heart", "cardiac surface", "pericardium",
                 "heart surface", "outer"],
    RV_MYOCARDIUM: ["rv myo", "rv wall", "right ventricular wall", "rv myocardium"],
}
# phrases that contradict a role when present in the name
NEGATIVE_KEYWORDS: Dict[str, List[str]] = {
    MYOCARDIUM: ["blood", "cavity", "pool", "rv myo", "rv wall", "right"],
    LV: ["myo", "myocardium", "wall", "muscle"],
    RV: ["myo", "myocardium", "wall"],
    EPICARDIUM: ["myo", "ventricle", "atrium", "lv", "rv", "la", "ra", "valve", "lung", "bronch"],
}


def tokenize(path: str) -> List[str]:
    """Split a USD path/name into lowercase tokens (handles camelCase, _ - / . digits)."""
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", path)
    name = re.sub(r"[^A-Za-z]+", " ", name).lower()
    return [t for t in name.split() if t]


def keyword_score(path: str, phrases: List[str]) -> float:
    tokens = tokenize(path)
    text = " " + " ".join(tokens) + " "
    compact = "".join(tokens)
    best = 0.0
    for phrase in phrases:
        if " " in phrase:
            if f" {phrase} " in text:
                best = max(best, 1.0)
        elif len(phrase) <= 3:
            if phrase in tokens:
                best = max(best, 0.9)
        elif phrase in text or phrase in compact:
            best = max(best, 1.0 if phrase in tokens else 0.8)
    return best


def name_scores(path: str) -> Dict[str, float]:
    """Name evidence per role; uses the leaf name first, parents as weaker hints."""
    leaf = path.rstrip("/").split("/")[-1]
    out = {}
    for role, phrases in NAME_KEYWORDS.items():
        s = keyword_score(leaf, phrases)
        if s == 0.0:
            s = 0.5 * keyword_score(path, phrases)
        neg = NEGATIVE_KEYWORDS.get(role)
        if neg and keyword_score(leaf, neg) > 0 and s < 1.0:
            s *= 0.25
        out[role] = s
    return out


# ----------------------------------------------------------------------------
# Explainable candidates (from the Phase 3.4 design)
# ----------------------------------------------------------------------------
@dataclass
class Evidence:
    name: str
    value: float
    weight: float
    contribution: float
    explanation: str


@dataclass
class AnatomicalCandidate:
    label: str
    source_path: str
    part_index: int
    score: float = 0.0
    evidence: List[Evidence] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    method: str = "rule_based_v2"

    def add_evidence(self, name, value, weight, explanation):
        value = float(np.clip(value, 0.0, 1.0))
        self.evidence.append(Evidence(name, value, weight, value * weight, explanation))

    def calculate_score(self):
        total_w = sum(e.weight for e in self.evidence)
        total = sum(e.contribution for e in self.evidence)
        score = total / total_w if total_w > 0 else 0.0
        score *= 0.6 ** len(self.contradictions)
        self.score = float(np.clip(score, 0.0, 1.0))
        return self.score

    def to_dict(self):
        return {
            "label": self.label,
            "source_path": self.source_path,
            "score": round(self.score, 4),
            "confidence": confidence_class(self.score),
            "evidence": [e.__dict__ for e in self.evidence],
            "contradictions": list(self.contradictions),
            "method": self.method,
        }


def confidence_class(score: float) -> str:
    if score >= 0.85:
        return "HIGH"
    if score >= 0.65:
        return "MEDIUM"
    if score >= 0.45:
        return "LOW"
    return "REJECT"


def ambiguity_margin(candidates: List[AnatomicalCandidate]) -> float:
    if len(candidates) < 2:
        return 1.0
    s = sorted((c.score for c in candidates), reverse=True)
    return s[0] - s[1]


# ----------------------------------------------------------------------------
# Geometric features
# ----------------------------------------------------------------------------
@dataclass
class PartFeatures:
    index: int
    path: str
    volume_mm3: float  # voxel volume of the enclosed region
    surface_area_mm2: float
    is_closed: bool
    is_manifold: bool
    aspect: float  # sqrt(lambda1/lambda2) of voxel PCA
    centroid: np.ndarray
    hollow_ratio: float  # derived cavity volume / own volume
    thickness_mm: float = 0.0  # local thickness (2 x 95th pct of the inner distance transform)
    extent_mm: float = 1.0  # longest principal extent
    occupancy: np.ndarray = field(repr=False, default=None)
    cavity: np.ndarray = field(repr=False, default=None)
    contact: Dict[int, float] = field(default_factory=dict)  # boundary fraction touching part j
    inside_cavity_of: Dict[int, float] = field(default_factory=dict)  # own volume fraction inside j's cavity
    inside_of: Dict[int, float] = field(default_factory=dict)  # own volume fraction inside j's occupancy

    def summary(self):
        return {
            "path": self.path,
            "volume_ml": round(self.volume_mm3 / 1000.0, 2),
            "surface_area_cm2": round(self.surface_area_mm2 / 100.0, 2),
            "closed": self.is_closed,
            "manifold": self.is_manifold,
            "aspect": round(self.aspect, 2),
            "hollow_ratio": round(self.hollow_ratio, 3),
            "local_thickness_mm": round(self.thickness_mm, 1),
            "thickness_ratio": round(self.thickness_mm / max(self.extent_mm, 1e-9), 3),
        }


def derive_cavity(solid: np.ndarray, spacing: float, max_ray_mm=45.0, seed=0.62, grow=0.4,
                  min_fraction=0.02) -> np.ndarray:
    """Blood-pool voxels enclosed by ``solid`` (e.g. the LV cavity of a myocardial cup).

    Hysteresis on the ray-enclosure ratio: voxels hit from >= ``seed`` of all
    directions start a cavity which grows through voxels with >= ``grow``.
    """
    if not solid.any():
        return np.zeros_like(solid)
    sl = ndimage.find_objects(solid.astype(np.int8))[0]
    pad = 2
    sl = tuple(slice(max(s.start - pad, 0), min(s.stop + pad, n)) for s, n in zip(sl, solid.shape))
    sub = solid[sl]
    empty_idx = np.argwhere(~sub)
    if len(empty_idx) == 0:
        return np.zeros_like(solid)
    steps = max(4, int(round(max_ray_mm / spacing)))
    ratio = np.zeros(sub.shape)
    ratio[tuple(empty_idx.T)] = enclosure_ratio(sub, empty_idx, steps)
    seeds = ratio >= seed
    candidates = ratio >= grow
    lab, n = ndimage.label(candidates)
    if n == 0:
        return np.zeros_like(solid)
    keep = np.unique(lab[seeds & (lab > 0)])
    cav = np.isin(lab, keep) & candidates
    # drop specks
    lab2, n2 = ndimage.label(cav)
    if n2:
        sizes = ndimage.sum(cav, lab2, np.arange(1, n2 + 1))
        big = np.nonzero(sizes >= max(8, min_fraction * sub.sum()))[0] + 1
        cav = np.isin(lab2, big)
    out = np.zeros_like(solid)
    out[sl] = cav
    return out


def compute_part_features(meshes: List[SurfaceMesh], grid: VoxelGrid,
                          occupancies: Optional[List[np.ndarray]] = None) -> List[PartFeatures]:
    h = grid.spacing
    occs = occupancies or [voxelize(m, grid) for m in meshes]
    feats: List[PartFeatures] = []
    for i, (m, occ) in enumerate(zip(meshes, occs)):
        vol = float(occ.sum()) * h**3
        if occ.sum() >= 4:
            idx = np.argwhere(occ)
            cpts = grid.centers(idx)
            _, axes, w = pca_axes(cpts)
            aspect = float(np.sqrt(w[0] / max(w[1], 1e-9)))
            centroid = cpts.mean(0)
            pr = (cpts - centroid) @ axes[0]
            extent = float(pr.max() - pr.min())
            edt = ndimage.distance_transform_edt(occ) * h
            thickness = float(2 * np.percentile(edt[occ], 95))
        else:
            aspect, centroid, extent, thickness = 1.0, m.centroid, 1.0, 0.0
        topo = m.topology()
        cav = derive_cavity(occ, h) if vol > 0 else np.zeros_like(occ)
        feats.append(PartFeatures(
            index=i, path=m.source_path or m.name, volume_mm3=vol, surface_area_mm2=m.surface_area,
            is_closed=topo["is_closed"], is_manifold=topo["is_manifold"], aspect=aspect,
            centroid=centroid, hollow_ratio=float(cav.sum()) / max(occ.sum(), 1),
            thickness_mm=thickness, extent_mm=extent,
            occupancy=occ, cavity=cav,
        ))

    shell = [ndimage.binary_dilation(f.occupancy) & ~f.occupancy for f in feats]
    for f in feats:
        own = max(f.occupancy.sum(), 1)
        border = shell[f.index]
        nb = max(border.sum(), 1)
        for g in feats:
            if g.index == f.index:
                continue
            f.contact[g.index] = float((border & g.occupancy).sum()) / nb
            f.inside_cavity_of[g.index] = float((f.occupancy & ndimage.binary_dilation(g.cavity)).sum()) / own
            f.inside_of[g.index] = float((f.occupancy & g.occupancy).sum()) / own
    return feats


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
def _max_other(d: Dict[int, float]) -> float:
    return max(d.values()) if d else 0.0


def score_part(f: PartFeatures, feats: List[PartFeatures]) -> List[AnatomicalCandidate]:
    names = name_scores(f.path)
    total_vol = sum(g.volume_mm3 for g in feats) or 1.0
    rel_vol = f.volume_mm3 / total_vol
    encloses_other = max((g.inside_cavity_of.get(f.index, 0.0) for g in feats if g.index != f.index), default=0.0)
    contains_other = max((g.inside_of.get(f.index, 0.0) for g in feats if g.index != f.index), default=0.0)
    in_cavity = _max_other(f.inside_cavity_of)
    touches = _max_other(f.contact)
    hollow = float(np.clip(f.hollow_ratio / 0.6, 0, 1))
    # the most hollow part is the likely myocardial wall; its cavity holds the LV
    wall = max(feats, key=lambda g: g.hollow_ratio)
    wall = wall if wall.hollow_ratio > 0.3 and wall.index != f.index else None
    septal = wall.contact.get(f.index, 0.0) if wall is not None else 0.0  # share of wall border it covers
    lv_like = max(feats, key=lambda g: g.inside_cavity_of.get(wall.index, 0.0)) if wall is not None else None
    touches_lv = f.contact.get(lv_like.index, 0.0) if lv_like is not None and lv_like.index != f.index else 0.0
    # anonymous scenes ("Mesh_001"...) are scored on geometry alone, capped below HIGH
    named_scene = any(keyword_score(g.path.rstrip('/').split('/')[-1], kw) > 0
                      for g in feats for kw in NAME_KEYWORDS.values())
    out = []

    class _NoName(AnatomicalCandidate):
        def add_evidence(self, name, value, weight, explanation):
            if name != "name":
                super().add_evidence(name, value, weight, explanation)

        def calculate_score(self):
            super().calculate_score()
            self.score *= 0.8
            return self.score

    def cand(label):
        cls = AnatomicalCandidate if named_scene else _NoName
        c = cls(label=label, source_path=f.path, part_index=f.index,
                method="rule_based_v2" if named_scene else "geometry_only_v2")
        out.append(c)
        return c

    c = cand(MYOCARDIUM)
    c.add_evidence("name", names[MYOCARDIUM], 0.40, "USD name uses myocardium terminology")
    c.add_evidence("hollow", hollow, 0.30, f"Wraps an enclosed cavity (cavity/wall = {f.hollow_ratio:.2f})")
    c.add_evidence("encloses_part", encloses_other, 0.20, "Another part lies inside its cavity (blood pool)")
    c.add_evidence("closed", float(f.is_closed), 0.10, "Closed surface")
    if f.hollow_ratio < 0.05 and names[MYOCARDIUM] < 0.5:
        c.contradictions.append("No enclosed cavity: solid body, not a wall")
    # parts are in anatomical mm (auto-scaled): a myocardial wall is ~8-16 mm, hypertrophy rarely > 25 mm
    thick_ratio = f.thickness_mm / max(f.extent_mm, 1e-9)
    c.add_evidence("thin_wall", float(np.clip((26.0 - f.thickness_mm) / 10.0, 0, 1)), 0.20,
                   f"Local wall thickness {f.thickness_mm:.0f} mm ({100 * thick_ratio:.0f} % of its length)")
    if f.thickness_mm > 25.0 and thick_ratio > 0.15 and names[MYOCARDIUM] < 1.0:
        c.contradictions.append(f"Solid body {f.thickness_mm:.0f} mm thick: concavities, not a wall around a cavity")

    c = cand(LV)
    c.add_evidence("name", names[LV], 0.45, "USD name uses left-ventricle terminology")
    c.add_evidence("in_myocardial_cavity", in_cavity, 0.35, "Lies inside the cavity of a hollow part")
    c.add_evidence("solid", 1.0 - hollow, 0.10, "Solid blood-pool body (no own cavity)")
    c.add_evidence("closed", float(f.is_closed), 0.10, "Closed surface")
    if hollow > 0.5 and names[LV] < 1.0:
        c.contradictions.append("Part is hollow; a blood pool should be solid")

    c = cand(RV)
    c.add_evidence("name", names[RV], 0.55, "USD name uses right-ventricle terminology")
    c.add_evidence("septal_contact", float(np.clip(septal / 0.08, 0, 1)) * (1.0 - in_cavity), 0.30,
                   f"Covers {100 * septal:.0f}% of the myocardial wall border without lying in its cavity")
    if wall is None or septal < 0.02:
        c.contradictions.append("No septal contact with the myocardial wall")
    c.add_evidence("solid", 1.0 - hollow, 0.10, "Solid blood-pool body")
    c.add_evidence("closed", float(f.is_closed), 0.10, "Closed surface")

    for role in (LA, RA, PA, RV_MYOCARDIUM):
        if not named_scene and role != LA:
            continue  # no geometric signature without names: needs naming or user assignment
        c = cand(role)
        c.add_evidence("name", names[role], 0.75, f"USD name uses {role} terminology")
        c.add_evidence("closed", float(f.is_closed), 0.10, "Closed surface")
        if role == LA:
            c.add_evidence("mitral_contact", float(np.clip(touches_lv / 0.05, 0, 1)) * (septal < 0.08), 0.15,
                           "Touches the LV blood pool but not the septal wall (mitral inflow)")
        else:
            c.add_evidence("touches", float(np.clip(touches / 0.1, 0, 1)), 0.15, "Contacts other cardiac parts")

    c = cand(AORTA)
    c.add_evidence("name", names[AORTA], 0.65, "USD name uses aortic terminology")
    c.add_evidence("elongated", float(np.clip((f.aspect - 1.5) / 2.0, 0, 1)), 0.25, "Tubular / elongated body")
    c.add_evidence("closed", float(f.is_closed), 0.10, "Closed surface")

    c = cand(EPICARDIUM)
    c.add_evidence("name", names[EPICARDIUM], 0.40, "USD name uses whole-heart terminology")
    c.add_evidence("contains_parts", contains_other, 0.35, "Other cardiac parts lie inside it")
    c.add_evidence("largest", float(np.clip(rel_vol / 0.5, 0, 1)), 0.15, "Largest structure")
    c.add_evidence("closed", float(f.is_closed), 0.10, "Closed surface")
    if len(feats) > 1 and contains_other < 0.3 and names[EPICARDIUM] < 0.5:
        c.contradictions.append("Does not contain the other parts")

    for c in out:
        c.calculate_score()
    return sorted(out, key=lambda c: c.score, reverse=True)


@dataclass
class AnatomyAssignment:
    roles: Dict[str, int]  # role -> part index
    scores: Dict[str, float]
    candidates: Dict[int, List[AnatomicalCandidate]]
    features: List[PartFeatures]
    overrides: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def part(self, role) -> Optional[int]:
        return self.roles.get(role)

    def status(self, role) -> str:
        if role in self.overrides:
            return "USER_CONFIRMED"
        if role not in self.roles:
            return "MISSING"
        idx = self.roles[role]
        margin = ambiguity_margin(self.candidates[idx])
        top = self.candidates[idx][0]
        if top.label == role and margin < 0.10:
            return "AMBIGUOUS"
        return confidence_class(self.scores[role])

    def to_dict(self):
        return {
            "roles": {r: self.features[i].path for r, i in self.roles.items()},
            "scores": {r: round(s, 4) for r, s in self.scores.items()},
            "status": {r: self.status(r) for r in self.roles},
            "parts": [
                {**f.summary(), "candidates": [c.to_dict() for c in self.candidates[f.index][:4]],
                 "ambiguity_margin": round(ambiguity_margin(self.candidates[f.index]), 3)}
                for f in self.features
            ],
            "warnings": list(self.warnings),
        }


def assign_roles(feats: List[PartFeatures], overrides: Optional[Dict[str, int]] = None,
                 min_score=0.40) -> AnatomyAssignment:
    """Greedy one-to-one role assignment by descending score, honouring user overrides."""
    overrides = dict(overrides or {})
    cands = {f.index: score_part(f, feats) for f in feats}
    roles: Dict[str, int] = {}
    scores: Dict[str, float] = {}
    used = set()
    for role, idx in overrides.items():
        roles[role] = idx
        scores[role] = 1.0
        used.add(idx)
    pool = sorted((c for lst in cands.values() for c in lst), key=lambda c: c.score, reverse=True)
    for c in pool:
        if c.score < min_score or c.label in roles or c.part_index in used:
            continue
        roles[c.label] = c.part_index
        scores[c.label] = c.score
        used.add(c.part_index)
    asg = AnatomyAssignment(roles=roles, scores=scores, candidates=cands, features=feats, overrides=overrides)
    if MYOCARDIUM not in roles:
        if LV in roles or EPICARDIUM in roles:
            asg.warnings.append("No myocardium part found: the LV wall will be derived from the LV blood pool "
                                "and surrounding structures (low confidence).")
        else:
            asg.warnings.append("No myocardium or LV blood pool found; assign roles manually.")
    for role in roles:
        if asg.status(role) in ("LOW", "AMBIGUOUS"):
            asg.warnings.append(f"{role} assignment is {asg.status(role)}: please review.")
    return asg
