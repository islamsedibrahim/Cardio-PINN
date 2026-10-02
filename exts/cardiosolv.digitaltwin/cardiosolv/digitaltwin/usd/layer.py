"""CardioSolv authoring layer.

All CardioSolv opinions (semantic hierarchy, GeomSubsets on the user's mesh,
simulation colours and animation) are written into dedicated sublayers of the
root layer. The user's own asset files are never edited, and removing those
sublayers restores the original heart exactly.

USD rule to keep in mind: sublayers are weaker than the root layer itself. For
hearts brought in by reference (the usual Isaac Sim case) the sublayers win;
for meshes defined directly in the opened file the animation is authored in
the session layer and ``results_writer.export_twin_stage`` saves a thin stage
whose weakest sublayer is the original file.
"""

from __future__ import annotations

import os

from pxr import Sdf, Usd

LAYER_TAG = "cardiosolv"


RESULTS_TAG = "cardiosolv_twin"


def _matches(layer, tag):
    name = os.path.splitext(os.path.basename(layer.identifier))[0].lower()
    return name.endswith(tag) or name.startswith(tag)


def _resolve(root: Sdf.Layer, path: str):
    """Sublayer paths are relative to the root layer, not to the working directory."""
    layer = Sdf.Layer.Find(path)
    if layer is None and not root.anonymous:
        layer = Sdf.Layer.Find(Sdf.ComputeAssetPathRelativeToLayer(root, path)) or \
            Sdf.Layer.FindOrOpen(Sdf.ComputeAssetPathRelativeToLayer(root, path))
    return layer or Sdf.Layer.FindOrOpen(path)


def find_layer(stage: Usd.Stage, tag: str = LAYER_TAG):
    root = stage.GetRootLayer()
    for path in root.subLayerPaths:
        layer = _resolve(root, path)
        if layer and _matches(layer, tag):
            return layer
    return None


def get_or_create_layer(stage: Usd.Stage, file_path: str = None, tag: str = LAYER_TAG, ext: str = "usda") -> Sdf.Layer:
    """Return a CardioSolv sublayer (semantics: ``cardiosolv``; results: ``cardiosolv_twin``)."""
    layer = find_layer(stage, tag)
    if layer is not None:
        return layer
    root = stage.GetRootLayer()
    if file_path is None and not root.anonymous and root.realPath:
        stem, _ = os.path.splitext(root.realPath)
        file_path = f"{stem}_{tag}.{ext}"
    if file_path:
        layer = Sdf.Layer.FindOrOpen(file_path) or Sdf.Layer.CreateNew(file_path)
        ident = os.path.relpath(layer.realPath, os.path.dirname(root.realPath)) if root.realPath else layer.identifier
    else:
        layer = Sdf.Layer.CreateAnonymous(f"{tag}.{ext}")
        ident = layer.identifier
    layer.comment = "CardioSolv digital-twin annotations and results (safe to delete)."
    root.subLayerPaths.insert(0, ident)
    return layer


def edit_context(stage: Usd.Stage, layer=None):
    return Usd.EditContext(stage, layer or get_or_create_layer(stage))


def save_layers(stage: Usd.Stage):
    saved = []
    for tag in (LAYER_TAG, RESULTS_TAG):
        layer = find_layer(stage, tag)
        if layer is not None and not layer.anonymous:
            layer.Save()
            saved.append(layer.realPath)
    return saved
