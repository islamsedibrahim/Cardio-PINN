from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class EngineResult:
    engine: str
    label_map: str  # NIfTI path
    mapping: Dict[int, str]  # label value -> canonical structure name
    details: dict = field(default_factory=dict)


def get_engine(name: str):
    if name in ("nv-segment", "nv-segment-ctmr", "vista3d"):
        from .nv_segment import NVSegmentCTMR

        return NVSegmentCTMR()
    if name in ("totalsegmentator", "totalseg"):
        from .totalseg import TotalSegmentatorEngine

        return TotalSegmentatorEngine()
    if name in ("labelmap", "existing"):
        from .labelmap import LabelMapEngine

        return LabelMapEngine()
    raise ValueError(f"Unknown engine '{name}'")


def available_engines() -> Dict[str, str]:
    """Which engines can run in this environment (and why not)."""
    out = {}
    for name in ("nv-segment", "totalsegmentator"):
        try:
            out[name] = get_engine(name).check()
        except Exception as exc:  # pragma: no cover - environment dependent
            out[name] = f"unavailable: {exc}"
    out["labelmap"] = "ok"
    return out
