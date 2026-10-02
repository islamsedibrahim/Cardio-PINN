"""Stages 1-3: anatomy discovery and the geometry layer on imported meshes."""

import numpy as np
from pxr import Usd

from cardiosolv.digitaltwin.core import anatomy as A
from cardiosolv.digitaltwin.core.geometry_layer import BASE, ENDO, EPI, RV_SEPTUM, auto_spacing, build_geometry_layer
from cardiosolv.digitaltwin.core.mesh import SurfaceMesh, triangulate_polygons
from cardiosolv.digitaltwin.core.voxel import VoxelGrid, voxelize
from cardiosolv.digitaltwin.usd.scene_scanner import scan_heart
from conftest import world_direction


def _assign(scan):
    parts = scan.parts
    lo = np.min([m.bbox_min for m in parts], axis=0)
    hi = np.max([m.bbox_max for m in parts], axis=0)
    feats = A.compute_part_features(parts, VoxelGrid.around(lo, hi, auto_spacing(parts, 150_000)))
    return A.assign_roles(feats)


def test_triangulation_keeps_face_ids():
    tris, owner = triangulate_polygons([4, 3, 5], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    assert tris.shape == (2 + 1 + 3, 3)
    assert owner.tolist() == [0, 0, 1, 2, 2, 2]


def test_voxelize_sphere_robust_to_holes():
    from skimage.measure import marching_cubes

    g = np.mgrid[-30:31, -30:31, -30:31].astype(float)
    v, f, _, _ = marching_cubes(np.sqrt((g**2).sum(0)) - 25, 0)
    sphere = SurfaceMesh("s", v - 30, f)
    grid = VoxelGrid.around(sphere.bbox_min, sphere.bbox_max, 1.5)
    exact = 4 / 3 * np.pi * 25**3
    assert abs(voxelize(sphere, grid).sum() * 1.5**3 - exact) / exact < 0.01
    holey = SurfaceMesh("h", sphere.points, sphere.faces[50:])
    assert abs(voxelize(holey, grid).sum() * 1.5**3 - exact) / exact < 0.01


def test_scan_converts_units_to_mm(heart_usd):
    scan = scan_heart(Usd.Stage.Open(heart_usd), "/World/Patient/Heart")
    myo = [p for p in scan.parts if p.name == "heart_myocardium"][0]
    assert scan.mm_per_unit == 10.0  # cm stage
    assert 85 < myo.volume / 1000 < 105  # ~96 mL myocardial wall


def test_named_parts_are_classified(heart_usd):
    scan = scan_heart(Usd.Stage.Open(heart_usd), "/World/Patient/Heart")
    asg = _assign(scan)
    expected = {A.MYOCARDIUM: "heart_myocardium", A.LV: "heart_ventricle_left", A.RV: "heart_ventricle_right",
                A.LA: "heart_atrium_left", A.RA: "heart_atrium_right", A.AORTA: "aorta"}
    for role, name in expected.items():
        assert scan.parts[asg.roles[role]].name == name, role
    assert asg.status(A.MYOCARDIUM) == "HIGH"


def test_anonymous_parts_found_by_geometry(generic_heart_usd):
    path, names = generic_heart_usd
    scan = scan_heart(Usd.Stage.Open(path), "/World/Patient/Heart")
    asg = _assign(scan)
    assert scan.parts[asg.roles[A.MYOCARDIUM]].name == names["myo"]
    assert scan.parts[asg.roles[A.LV]].name == names["lv"]
    assert scan.parts[asg.roles[A.RV]].name == names["rv"]
    # geometry-only evidence is never reported as HIGH confidence
    assert all(asg.status(r) != "HIGH" for r in asg.roles)


def test_geometry_layer_whole_heart(heart_usd):
    stage = Usd.Stage.Open(heart_usd)
    scan = scan_heart(stage, "/World/Patient/Heart")
    gl = build_geometry_layer(scan.parts, _assign(scan))
    m = gl.metrics
    assert 85 < m["myocardial_volume_ml"] < 105
    assert 48 < m["lv_cavity_volume_ml"] < 62
    assert 8 < m["mean_wall_thickness_mm"] < 13
    counts = np.bincount(gl.surface_labels, minlength=5)
    assert counts[ENDO] > 0 and counts[EPI] > 0 and counts[BASE] > 0 and counts[RV_SEPTUM] > 0
    # long axis: local +z (apex -> base) of the heart transform
    expected = world_direction(stage, "/World/Patient/Heart", (0, 0, 1))
    assert gl.long_axis.direction @ expected > 0.98
    assert gl.long_axis.confidence > 0.9
    # septum points towards the RV (local -x)
    sep = world_direction(stage, "/World/Patient/Heart", (-1, 0, 0))
    assert gl.septum_direction @ sep > 0.9


def test_endo_epi_match_cardiopinn_labels(lv_mean_usd):
    """Buoso's LV_mean.vtk: label 2 = inner (endocardial) surface, 1 = outer.

    (The Cardio-PINN README lists them the other way round, but the code loads
    the pressure surface from label 2 and label 2 is the smaller inner shell.)
    """
    path, gt = lv_mean_usd
    stage = Usd.Stage.Open(path)
    scan = scan_heart(stage, "/World/Imported")
    asg = _assign(scan)
    assert asg.roles[A.MYOCARDIUM] == 0
    gl = build_geometry_layer(scan.parts, asg)
    assert gl.myocardium_source == "part"
    m = scan.parts[0]
    votes = np.zeros((m.n_points, 5))
    for k in range(3):
        np.add.at(votes, (m.faces[:, k], gl.surface_labels), 1)
    vlab = votes.argmax(1)
    assert (vlab[gt == 2] == ENDO).mean() > 0.97
    assert (vlab[gt == 1] == EPI).mean() > 0.97
    # ground-truth long axis: epicardial apex -> centre of the mitral rings (labels 3, 5)
    from fixtures import lv_mean_surface

    orig, _, _ = lv_mean_surface()
    world = m.points  # same vertex order, world mm
    apex = world[np.argmin(orig[:, 2])]
    base = world[np.isin(gt, (3, 5))].mean(0)
    expected = (base - apex) / np.linalg.norm(base - apex)
    assert gl.long_axis.direction @ expected > 0.98
    # apex = centroid of the extreme 2% region (Phase 3.2), so a few mm inside the single extreme vertex
    assert np.linalg.norm(gl.long_axis.apex - apex) < 6.0
    assert 70 < gl.long_axis.length_mm < 85
