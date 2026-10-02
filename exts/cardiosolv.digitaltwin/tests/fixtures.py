"""Test anatomies.

These fixtures only exist to exercise the pipeline in CI. In Isaac Sim the
pipeline always runs on the heart the user selected in the stage.
"""

from __future__ import annotations

import os

import numpy as np
from pxr import Gf, Usd, UsdGeom
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
LV_MEAN_VTK = os.path.join(REPO, "Shape_model", "LV_mean.vtk")

TOTALSEG_NAMES = {
    "myo": "heart_myocardium",
    "lv": "heart_ventricle_left",
    "rv": "heart_ventricle_right",
    "la": "heart_atrium_left",
    "ra": "heart_atrium_right",
    "ao": "aorta",
}


def _surface(mask, origin, h, sigma=1.0):
    vol = gaussian_filter(mask.astype(float), sigma)
    v, f, _, _ = marching_cubes(np.pad(vol, 1), 0.5)
    v = (v - 1) * h + origin
    return v, f[:, ::-1]  # outward orientation for skimage's convention


def synthetic_heart_masks(h=1.0):
    ax = np.arange(-70, 100 + h, h)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    base = 15.0
    epi = (X**2 + Y**2) / 32**2 + (Z - base) ** 2 / 70**2 <= 1
    endo = (X**2 + Y**2) / 21**2 + (Z - base) ** 2 / 59**2 <= 1
    below = Z <= base
    myo = epi & ~endo & below
    lv = endo & below
    epi_pad = (X**2 + Y**2) / 32.5**2 + (Z - base) ** 2 / 70.5**2 <= 1
    rv = ((X + 30) ** 2 / 26**2 + Y**2 / 34**2 + (Z - 12) ** 2 / 55**2 <= 1) & (Z <= 12) & ~epi_pad
    la = (X - 6) ** 2 + Y**2 + (Z - 33) ** 2 <= 18**2
    ra = (X + 34) ** 2 + Y**2 + (Z - 31) ** 2 <= 15**2
    ao = ((X - 12) ** 2 + (Y - 14) ** 2 <= 11**2) & (Z >= 8) & (Z <= 90) & ~la
    origin = np.array([ax[0]] * 3)
    return {"myo": myo, "lv": lv, "rv": rv, "la": la, "ra": ra, "ao": ao}, origin, h


def write_synthetic_heart_usd(path, names=None, meters_per_unit=0.01, up_axis="Y",
                              rotate_xyz=(30.0, -20.0, 75.0), translate=(12.0, 105.0, -4.0)):
    """Multi-part heart in a rotated, cm-unit stage under /World/Patient/Heart."""
    names = names or TOTALSEG_NAMES
    masks, origin, h = synthetic_heart_masks()
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y if up_axis == "Y" else UsdGeom.Tokens.z)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Xform.Define(stage, "/World/Patient")
    heart = UsdGeom.Xform.Define(stage, "/World/Patient/Heart")
    heart.AddTranslateOp().Set(Gf.Vec3d(*translate))
    heart.AddRotateXYZOp().Set(Gf.Vec3f(*rotate_xyz))
    unit = 0.001 / meters_per_unit  # mm -> stage units
    for key, mask in masks.items():
        v, f = _surface(mask, origin, h)
        m = UsdGeom.Mesh.Define(stage, f"/World/Patient/Heart/{names[key]}")
        m.CreatePointsAttr([Gf.Vec3f(*p) for p in (v * unit)])
        m.CreateFaceVertexCountsAttr([3] * len(f))
        m.CreateFaceVertexIndicesAttr(f.reshape(-1).tolist())
        m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    stage.GetRootLayer().Save()
    return path, masks


def lv_mean_surface():
    """Boundary surface of Buoso's LV_mean tetrahedral mesh (metres) + vertex labels."""
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    r = vtk.vtkUnstructuredGridReader()
    r.SetFileName(LV_MEAN_VTK)
    r.ReadAllScalarsOn()
    r.Update()
    d = r.GetOutput()
    pts = vtk_to_numpy(d.GetPoints().GetData()).astype(float)
    labels = vtk_to_numpy(d.GetPointData().GetArray("labels")).astype(int)
    cells = vtk_to_numpy(d.GetCells().GetData()).reshape(-1, 5)[:, 1:]
    faces = np.concatenate([cells[:, [0, 2, 1]], cells[:, [0, 1, 3]], cells[:, [1, 2, 3]], cells[:, [2, 0, 3]]])
    key = np.sort(faces, axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    bfaces = faces[cnt[inv.ravel()] == 1]
    used = np.unique(bfaces)
    remap = -np.ones(len(pts), int)
    remap[used] = np.arange(len(used))
    return pts[used], remap[bfaces], labels[used]


def write_lv_mean_usd(path, prim_name="Mesh_017"):
    pts, faces, labels = lv_mean_surface()
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.Xform.Define(stage, "/World")
    g = UsdGeom.Xform.Define(stage, "/World/Imported")
    g.AddRotateXYZOp().Set(Gf.Vec3f(-90.0, 0.0, 25.0))
    m = UsdGeom.Mesh.Define(stage, f"/World/Imported/{prim_name}")
    m.CreatePointsAttr([Gf.Vec3f(*p) for p in pts])
    m.CreateFaceVertexCountsAttr([3] * len(faces))
    m.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
    stage.GetRootLayer().Save()
    return path, labels


def write_skin_only_heart_usd(path, scale=9.0):
    """Single closed outer skin of the synthetic heart, ~1 m tall, metres (like generated SimReady assets)."""
    masks, origin, h = synthetic_heart_masks()
    union = np.zeros_like(masks["myo"])
    for m in masks.values():
        union |= m
    # real hearts carry several great vessels at the base: add pulmonary trunk and SVC
    ax = np.arange(-70, 100 + h, h)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    union |= ((X + 14) ** 2 + (Y + 16) ** 2 <= 10**2) & (Z >= 5) & (Z <= 85)  # pulmonary trunk
    union |= ((X + 36) ** 2 + (Y - 6) ** 2 <= 8**2) & (Z >= 30) & (Z <= 80)  # superior vena cava
    from scipy.ndimage import binary_closing, binary_fill_holes

    union = binary_fill_holes(binary_closing(union, iterations=2))
    v, f = _surface(union, origin, h, sigma=1.5)
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/root")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Xform.Define(stage, "/root/Generate_a_SimReady")
    UsdGeom.Xform.Define(stage, "/root/Generate_a_SimReady/Visuals")
    m = UsdGeom.Mesh.Define(stage, "/root/Generate_a_SimReady/Visuals/Generate_a_SimReady_001")
    m.CreatePointsAttr([Gf.Vec3f(*p) for p in (v * 0.001 * scale)])
    m.CreateFaceVertexCountsAttr([3] * len(f))
    m.CreateFaceVertexIndicesAttr(f.reshape(-1).tolist())
    stage.GetRootLayer().Save()
    return path
