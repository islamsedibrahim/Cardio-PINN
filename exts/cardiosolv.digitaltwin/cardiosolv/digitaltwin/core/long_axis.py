"""Cardiac long-axis estimation with explicit apex/base sign resolution.

PCA gives a geometric axis whose sign is ambiguous (Phase 3.2 caveat). The
sign is resolved from anatomical evidence, each vote kept for the report:

1. base structures (LA / aorta / pulmonary artery) lie on the base side;
2. the LV cavity opens towards the base while the apex keeps a myocardial cap;
3. the ventricle is wider at the base than at the apex.

The axis is then refined iteratively as apex -> centre of the basal cavity
opening, which is how clinicians define the LV long axis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .mesh import normalize, pca_axes


@dataclass
class AxisVote:
    name: str
    sign: float  # +1 keeps PCA direction, -1 flips
    weight: float
    detail: str


@dataclass
class LongAxis:
    apex: np.ndarray  # epicardial apex (world mm)
    base_center: np.ndarray  # centre of the basal (valve-plane) opening
    direction: np.ndarray  # unit vector apex -> base
    length_mm: float
    confidence: float
    method: str = "pca+anatomical_sign+apex_base_refinement"
    votes: List[AxisVote] = field(default_factory=list)
    pca_direction: Optional[np.ndarray] = None

    def project(self, pts):
        """Normalised longitudinal coordinate: 0 at apex, 1 at base."""
        return (np.asarray(pts) - self.apex) @ self.direction / max(self.length_mm, 1e-9)

    def to_dict(self):
        return {
            "apex": self.apex.tolist(),
            "base_center": self.base_center.tolist(),
            "direction": self.direction.tolist(),
            "length_mm": self.length_mm,
            "confidence": self.confidence,
            "method": self.method,
            "votes": [v.__dict__ for v in self.votes],
        }


def _extreme_mean(points, proj, low=True, fraction=0.02, min_count=5):
    n = max(min_count, int(len(points) * fraction))
    order = np.argsort(proj)
    sel = order[:n] if low else order[-n:]
    return points[sel].mean(0)


def estimate_long_axis(myo_pts: np.ndarray, cavity_pts: np.ndarray, base_structure_pts: Optional[np.ndarray] = None,
                       spacing: float = 1.0, iterations=4) -> LongAxis:
    """``*_pts`` are voxel centres (world mm) of myocardium, LV cavity and base structures."""
    body = np.vstack([myo_pts, cavity_pts]) if len(cavity_pts) else myo_pts
    centroid, axes, _ = pca_axes(body)
    a0 = axes[0]
    votes: List[AxisVote] = []

    # 1. base structures
    if base_structure_pts is not None and len(base_structure_pts) > 10:
        d = (base_structure_pts.mean(0) - centroid) @ a0
        votes.append(AxisVote("base_structures", float(np.sign(d)) or 1.0, 1.0,
                              f"atria/great vessels lie at {d:+.1f} mm along PCA axis"))

    # 2. cavity opening vs apical myocardial cap
    if len(cavity_pts) > 10:
        pm, pc = myo_pts @ a0, cavity_pts @ a0
        gap_hi = pm.max() - pc.max()
        gap_lo = pc.min() - pm.min()
        diff = gap_lo - gap_hi
        w = float(np.clip(abs(diff) / max(3 * spacing, 1e-6), 0.0, 1.0)) * 0.8
        votes.append(AxisVote("cavity_opening", float(np.sign(diff)) or 1.0, w,
                              f"myocardial cap {gap_lo:.1f} mm at - end vs {gap_hi:.1f} mm at + end"))

    # 3. cross-sectional width profile
    p = body @ a0
    lo, hi = np.percentile(p, [15, 85])
    radial = body - np.outer(body @ a0, a0)
    r_mean = radial.mean(0)

    def width(mask):
        return np.linalg.norm(radial[mask] - r_mean, axis=1).mean() if mask.any() else 0.0

    w_lo, w_hi = width(p <= lo), width(p >= hi)
    rel = (w_hi - w_lo) / max(w_hi + w_lo, 1e-9)
    votes.append(AxisVote("width_profile", float(np.sign(rel)) or 1.0, float(np.clip(abs(rel) * 4, 0, 0.5)),
                          f"mean radius {w_lo:.1f} mm (- end) vs {w_hi:.1f} mm (+ end)"))

    total = sum(v.weight for v in votes)
    agree = sum(v.sign * v.weight for v in votes)
    sign = 1.0 if agree >= 0 else -1.0
    direction = a0 * sign
    sign_conf = abs(agree) / total if total > 0 else 0.0

    # iterative apex -> basal-opening refinement
    cav = cavity_pts if len(cavity_pts) > 10 else body
    for _ in range(iterations):
        pm = myo_pts @ direction
        apex = _extreme_mean(myo_pts, pm, low=True)
        pc = cav @ direction
        top = pc >= np.percentile(pc, 92)
        base_center = cav[top].mean(0)
        new_dir = normalize(base_center - apex)
        if new_dir @ direction < 0:  # never flip during refinement
            break
        direction = new_dir
    pm = myo_pts @ direction
    apex = _extreme_mean(myo_pts, pm, low=True)
    length = float((myo_pts @ direction).max() - apex @ direction)
    base_center = apex + direction * (np.percentile(cav @ direction, 96) - apex @ direction)
    pca_agreement = abs(float(direction @ a0))
    confidence = float(np.clip(0.5 * sign_conf + 0.3 * pca_agreement + 0.2 * (len(votes) >= 2), 0, 1))
    return LongAxis(apex=apex, base_center=base_center, direction=direction, length_mm=length,
                    confidence=confidence, votes=votes, pca_direction=a0 * sign)
