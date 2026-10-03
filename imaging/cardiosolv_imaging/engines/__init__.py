"""Segmentation engines. Each returns a label map NIfTI plus {value: canonical structure name}."""

from .base import EngineResult, available_engines, get_engine

__all__ = ["EngineResult", "available_engines", "get_engine"]
