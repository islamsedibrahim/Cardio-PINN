"""Optional adapters for external tools (used only when installed).

* VMTK: centrelines / surface remeshing of vessels (aorta, pulmonary artery)
  for future CFD stages, following the vmtk Python API.
* gmsh: graded tetrahedral meshing of the myocardium surface.

Neither is required: CardioSolv's built-in voxel mesher and solvers always run.
"""

from __future__ import annotations

import os

import numpy as np

from ..core.mesh import SurfaceMesh


def write_stl(path, mesh: SurfaceMesh):
    a, b, c = mesh.face_corners()
    n = mesh.face_normals()
    with open(path, "w") as f:
        f.write(f"solid {mesh.name}\n")
        for i in range(mesh.n_faces):
            f.write(f" facet normal {n[i, 0]:.6e} {n[i, 1]:.6e} {n[i, 2]:.6e}\n  outer loop\n")
            for v in (a[i], b[i], c[i]):
                f.write(f"   vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}\n")
            f.write("  endloop\n endfacet\n")
        f.write(f"endsolid {mesh.name}\n")
    return path


def vmtk_available() -> bool:
    try:
        import vmtk  # noqa: F401

        return True
    except Exception:
        return False


def vmtk_centerlines(surface: SurfaceMesh, source_point, target_points):
    """Centrelines of a vessel part with vmtkcenterlines (requires vmtk)."""
    from vmtk import vmtkscripts
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy

    poly = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(surface.points))
    poly.SetPoints(pts)
    cells = np.c_[np.full(surface.n_faces, 3), surface.faces].ravel()
    ca = vtk.vtkCellArray()
    ca.SetCells(surface.n_faces, numpy_to_vtkIdTypeArray(cells.astype(np.int64)))
    poly.SetPolys(ca)
    cl = vmtkscripts.vmtkCenterlines()
    cl.Surface = poly
    cl.SeedSelectorName = "pointlist"
    cl.SourcePoints = list(np.asarray(source_point, float))
    cl.TargetPoints = list(np.asarray(target_points, float).ravel())
    cl.Execute()
    return vtk_to_numpy(cl.Centerlines.GetPoints().GetData())


def gmsh_available() -> bool:
    try:
        import gmsh  # noqa: F401

        return True
    except Exception:
        return False


def gmsh_tet_mesh(gl, myo_surface: SurfaceMesh, spacing_mm, workdir=None):
    """Tetrahedralise the user's (closed) myocardium surface with gmsh."""
    if myo_surface is None:
        raise RuntimeError("gmsh meshing needs an explicit myocardium surface part.")
    import gmsh
    import tempfile

    from ..core.geometry_layer import label_surface_faces
    from ..core.volume_mesh import TetMesh, boundary_faces_of

    workdir = workdir or tempfile.mkdtemp(prefix="cardiosolv_gmsh_")
    stl = write_stl(os.path.join(workdir, "myocardium.stl"), myo_surface)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.merge(stl)
        gmsh.model.mesh.classifySurfaces(np.pi / 4, True, True, np.pi)
        gmsh.model.mesh.createGeometry()
        s = gmsh.model.getEntities(2)
        loop = gmsh.model.geo.addSurfaceLoop([e[1] for e in s])
        gmsh.model.geo.addVolume([loop])
        gmsh.model.geo.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMax", spacing_mm)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5 * spacing_mm)
        gmsh.model.mesh.generate(3)
        tags, coords, _ = gmsh.model.mesh.getNodes()
        pts = coords.reshape(-1, 3)
        remap = {int(t): i for i, t in enumerate(tags)}
        _, _, elem_nodes = gmsh.model.mesh.getElements(3)
        tets = np.array([remap[int(t)] for t in elem_nodes[0]]).reshape(-1, 4)
    finally:
        gmsh.finalize()
    used, inv = np.unique(tets, return_inverse=True)
    pts, tets = pts[used], inv.reshape(-1, 4)
    a, b, c, d = (pts[tets[:, i]] for i in range(4))
    neg = np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a)) < 0
    tets[neg] = tets[neg][:, [0, 2, 1, 3]]
    bf = boundary_faces_of(tets)
    mesh = TetMesh(pts, tets, bf, np.zeros(len(bf), np.int8), spacing_mm, {"mesher": "gmsh"})
    mesh.face_labels = label_surface_faces(SurfaceMesh("gmsh", pts, bf), gl.grid, gl.voxel_class, gl.long_axis)
    mesh.metadata.update({"nodes": len(pts), "tets": len(tets)})
    return mesh
