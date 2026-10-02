"""Perceptual colour maps for painting results onto USD meshes (no matplotlib)."""

from __future__ import annotations

import numpy as np

# control points sampled from matplotlib colormaps (public domain data)
_MAPS = {
    "viridis": [(0.267, 0.005, 0.329), (0.283, 0.141, 0.458), (0.254, 0.265, 0.530), (0.207, 0.372, 0.553),
                (0.164, 0.471, 0.558), (0.128, 0.567, 0.551), (0.135, 0.659, 0.518), (0.267, 0.749, 0.441),
                (0.478, 0.821, 0.318), (0.741, 0.873, 0.150), (0.993, 0.906, 0.144)],
    "turbo": [(0.190, 0.072, 0.232), (0.275, 0.400, 0.865), (0.155, 0.677, 0.977), (0.093, 0.877, 0.773),
              (0.392, 0.993, 0.426), (0.758, 0.966, 0.207), (0.979, 0.758, 0.208), (0.982, 0.480, 0.118),
              (0.835, 0.212, 0.031), (0.480, 0.016, 0.011)],
    "coolwarm": [(0.230, 0.299, 0.754), (0.403, 0.535, 0.934), (0.600, 0.729, 0.998), (0.788, 0.846, 0.939),
                 (0.932, 0.875, 0.820), (0.968, 0.721, 0.612), (0.906, 0.495, 0.384), (0.706, 0.016, 0.150)],
    "inferno": [(0.001, 0.000, 0.014), (0.160, 0.042, 0.346), (0.416, 0.090, 0.432), (0.645, 0.198, 0.329),
                (0.865, 0.317, 0.226), (0.987, 0.535, 0.038), (0.988, 0.998, 0.645)],
}


def colorize(values, vmin, vmax, cmap="turbo") -> np.ndarray:
    pts = np.asarray(_MAPS[cmap])
    x = np.clip((np.asarray(values, float) - vmin) / max(vmax - vmin, 1e-12), 0, 1)
    pos = x * (len(pts) - 1)
    i = np.clip(np.floor(pos).astype(int), 0, len(pts) - 2)
    f = (pos - i)[..., None]
    return (pts[i] * (1 - f) + pts[i + 1] * f).astype(np.float32)


def available():
    return list(_MAPS)
