"""Use a segmentation you already have (3D Slicer, ITK-SNAP, MM-WHS, ACDC, a radiologist's contours ...).

The mapping comes from a preset (``acdc``, ``mmwhs``, ``totalseg_heartchambers``,
``cardiosolv``), a JSON file ``{"1": "myocardium", ...}`` or a 3D Slicer
``.seg.nrrd``-style colour table; names are normalised with ``labels.canonical``.
"""

from __future__ import annotations

import json
from pathlib import Path

from .. import labels as L
from .base import EngineResult


def load_mapping(spec):
    if spec is None:
        return dict(L.PRESETS["cardiosolv"])
    if isinstance(spec, dict):
        raw = spec
    elif str(spec).lower() in L.PRESETS:
        return dict(L.PRESETS[str(spec).lower()])
    else:
        raw = json.loads(Path(spec).read_text())
    out = {}
    for k, v in raw.items():
        name = L.canonical(v)
        if name is None:
            raise ValueError(f"Label '{v}' (value {k}) is not a known cardiac structure; "
                             f"use one of {sorted(L.BY_NAME)}")
        out[int(k)] = name
    return out


class LabelMapEngine:
    name = "labelmap"

    def check(self):
        return "ok"

    def segment(self, image: str, modality: str, out_dir: str, label_map=None, mapping=None) -> EngineResult:
        if not label_map:
            raise ValueError("labelmap engine needs --label-map PATH")
        return EngineResult(self.name, str(label_map), load_mapping(mapping), {"source": Path(label_map).name})
