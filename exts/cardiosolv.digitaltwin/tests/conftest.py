import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from fixtures import add_synthetic_scar_usd, write_lv_mean_usd, write_synthetic_heart_usd  # noqa: E402


@pytest.fixture(scope="session")
def heart_usd(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("heart") / "heart.usda")
    write_synthetic_heart_usd(path)
    return path


@pytest.fixture(scope="session")
def generic_heart_usd(tmp_path_factory):
    names = {k: f"Mesh_{i:03d}" for i, k in enumerate(["ao", "ra", "myo", "rv", "la", "lv"])}
    path = str(tmp_path_factory.mktemp("generic") / "heart.usda")
    write_synthetic_heart_usd(path, names=names)
    return path, names


@pytest.fixture(scope="session")
def lv_mean_usd(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("lvmean") / "lv.usda")
    return write_lv_mean_usd(path)


@pytest.fixture(scope="session")
def heart_pipeline(heart_usd):
    """Pipeline on the named synthetic heart, through stage 4 (EP)."""
    from pxr import Usd

    from cardiosolv.digitaltwin.pipeline import CardioSolvPipeline, PipelineConfig

    stage = Usd.Stage.Open(heart_usd)
    pipe = CardioSolvPipeline(stage, "/World/Patient/Heart", PipelineConfig(element_size_mm=4.0), log=lambda *a: None)
    for name in ("discover", "geometry", "mesh", "ep"):
        getattr(pipe, f"run_{name}")()
    return pipe


def world_direction(stage, prim_path, local_dir):
    from pxr import Gf, UsdGeom

    m = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(prim_path))
    d = m.TransformDir(Gf.Vec3d(*local_dir))
    d = np.array([d[0], d[1], d[2]])
    return d / np.linalg.norm(d)


@pytest.fixture(scope="session")
def scar_pipeline(tmp_path_factory):
    """Synthetic heart with a lateral infarct (core, border-zone rim, corridor), through EP at 3 mm."""
    from pxr import Usd

    from cardiosolv.digitaltwin.pipeline import CardioSolvPipeline, PipelineConfig

    path = str(tmp_path_factory.mktemp("scar") / "heart.usda")
    write_synthetic_heart_usd(path)
    _, masks = add_synthetic_scar_usd(path)
    pipe = CardioSolvPipeline(Usd.Stage.Open(path), "/World/Patient/Heart", PipelineConfig(element_size_mm=3.0),
                              log=lambda *a: None)
    for name in ("discover", "geometry", "mesh", "ep"):
        getattr(pipe, f"run_{name}")()
    pipe.gt_scar = {k: v.sum() / 1000.0 for k, v in masks.items()}
    return pipe
