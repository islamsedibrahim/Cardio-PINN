"""CardioSolv digital-twin pipeline: the stages run one by one on the heart
the user selected in the stage.

    1 discover    scan selected prim -> parts in world mm, anatomy confidence engine
    2 geometry    myocardium, endo/epi/base/septum, long axis, landmarks, /CardioSolv layer
    3 mesh        tetrahedral myocardium, ventricular coordinates, fibres
    4 ep          electrophysiology (eikonal / monodomain), pacing scenario, pseudo-ECG
    5 mechanics   Holzapfel-Ogden + EP-driven active tension + Windkessel heartbeat
    6 surrogate   Cardio-PINN surrogate (PhysicsNeMo backbone if present), EF calibration
    7 twin        paint + animate results on the user's meshes, exports, report

Each ``run_<stage>`` splits into pure computation and USD authoring so the Kit
UI can compute off the main thread. Works headless with plain ``pxr``.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from .core import anatomy as A
from .core.canonical import build_canonical
from .core.coordinates import compute_coordinates
from .core.geometry_layer import auto_spacing, build_geometry_layer
from .core.mapping import element_to_nodal
from .core.volume_mesh import build_tet_mesh
from .core.voxel import VoxelGrid
from .ep import EPConfig, run_electrophysiology

STAGES = ["discover", "geometry", "mesh", "ep", "mechanics", "surrogate", "twin"]
STAGE_TITLES = {
    "discover": "1  Anatomy discovery",
    "geometry": "2  Geometry layer (myocardium, endo/epi, long axis)",
    "mesh": "3  Computational mesh + fibres",
    "ep": "4  Electrophysiology",
    "mechanics": "5  Biomechanics + circulation",
    "surrogate": "6  Cardio-PINN surrogate",
    "twin": "7  Digital twin on your mesh",
}


@dataclass
class PipelineConfig:
    classification_voxels: int = 150_000
    geometry_spacing_mm: Optional[float] = None  # auto
    derived_wall_thickness_mm: float = 10.0
    element_size_mm: float = 3.0
    mesher: str = "voxel"  # "voxel" | "gmsh"
    endo_helix_deg: float = 60.0
    epi_helix_deg: float = -60.0
    sheet_gamma_deg: float = -65.0
    ep: EPConfig = field(default_factory=EPConfig)
    mechanics_dt_ms: float = 10.0
    cycle_ms: float = 800.0
    t_max_kpa: float = 85.0
    edp_mmhg: float = 10.0
    r_periph: float = 1.0
    c_art: float = 1.5
    p_aortic_diastolic: float = 80.0
    surrogate_epochs: int = 1500
    target_ef_pct: Optional[float] = None  # personalise contractility (e.g. echo LVEF)
    display_field: str = "transmembrane_potential"
    slow_motion: float = 4.0
    animate_neighbours: bool = True
    preview_surface_colors: bool = True
    output_dir: Optional[str] = None
    dashboard_url: Optional[str] = None  # Echocardiology dashboard, e.g. http://localhost:3000
    device: str = "auto"  # "auto" | "cpu" | "cuda"


class CardioSolvPipeline:
    def __init__(self, stage, source_path: str, config: PipelineConfig = None,
                 log: Callable[[str], None] = print, progress: Callable[[str, float], None] = None):
        self.stage = stage
        self.source_path = source_path
        self.cfg = config or PipelineConfig()
        self.log = log
        self.progress = progress or (lambda stage, frac: None)
        self.overrides: Dict[str, int] = {}
        self.timings: Dict[str, float] = {}
        self.done: List[str] = []
        # state
        self.scan = self.assignment = self.geometry = self.canonical = self.validation = None
        self.mesh = self.coords = self.ep = self.mech_model = self.cycle = None
        self.surrogate = self.surrogate_cycle = None
        self.calibration = None
        self.writer = None
        self.report: dict = {}

    # ------------------------------------------------------------------
    @property
    def device(self):
        if self.cfg.device != "auto":
            return self.cfg.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    @property
    def output_dir(self):
        if self.cfg.output_dir:
            d = self.cfg.output_dir
        else:
            root = self.stage.GetRootLayer()
            base = os.path.dirname(root.realPath) if root.realPath else os.getcwd()
            d = os.path.join(base, "cardiosolv_output")
        os.makedirs(d, exist_ok=True)
        return d

    def _timed(self, name, fn):
        t = time.time()
        self.progress(name, 0.0)
        out = fn()
        self.timings[name] = time.time() - t
        if name not in self.done:
            self.done.append(name)
        self.progress(name, 1.0)
        self.log(f"[CardioSolv] {STAGE_TITLES[name]}: done in {self.timings[name]:.1f} s")
        return out

    def invalidate_after(self, name):
        idx = STAGES.index(name)
        self.done = [s for s in self.done if STAGES.index(s) <= idx]

    # ------------------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------------------
    def run_discover(self):
        from .usd.scene_scanner import scan_heart

        def go():
            self.scan = scan_heart(self.stage, self.source_path)
            self.log(f"[CardioSolv] {len(self.scan.parts)} mesh parts under {self.source_path} "
                     f"(metersPerUnit={self.scan.meters_per_unit})")
            self.compute_assignment()
            return self.assignment

        self.invalidate_after("discover")
        return self._timed("discover", go)

    def compute_assignment(self):
        parts = self.scan.parts
        lo = np.min([m.bbox_min for m in parts], axis=0)
        hi = np.max([m.bbox_max for m in parts], axis=0)
        sp = auto_spacing(parts, self.cfg.classification_voxels)
        feats = A.compute_part_features(parts, VoxelGrid.around(lo, hi, sp))
        self.assignment = A.assign_roles(feats, self.overrides)
        for role, i in self.assignment.roles.items():
            self.log(f"    {role:28s} <- {parts[i].source_path}  "
                     f"({self.assignment.scores[role]:.2f}, {self.assignment.status(role)})")
        return self.assignment

    def set_role(self, role: str, part_index: Optional[int]):
        """User override from the UI (None clears)."""
        if part_index is None:
            self.overrides.pop(role, None)
        else:
            self.overrides = {r: i for r, i in self.overrides.items() if i != part_index}
            self.overrides[role] = part_index
        if self.scan is not None:
            self.compute_assignment()
        self.invalidate_after("discover")

    # ------------------------------------------------------------------
    # Stage 2
    # ------------------------------------------------------------------
    def compute_geometry(self):
        self.geometry = build_geometry_layer(self.scan.parts, self.assignment, self.cfg.geometry_spacing_mm,
                                             self.cfg.derived_wall_thickness_mm)
        self.canonical = build_canonical(self.source_path, self.scan.parts, self.scan.sources, self.assignment,
                                         self.geometry)
        self.validation = self.canonical.validate()
        for w in self.validation["warnings"]:
            self.log(f"    warning: {w}")
        m = self.geometry.metrics
        self.log(f"    myocardium {m['myocardial_volume_ml']:.1f} mL ({self.geometry.myocardium_source}), "
                 f"LV cavity {m['lv_cavity_volume_ml']:.1f} mL, long axis {m['long_axis_length_mm']:.1f} mm "
                 f"(conf {self.geometry.long_axis.confidence:.2f})")

    def write_geometry(self):
        from .usd.builder import CardioSolvUSDBuilder
        from .usd.layer import edit_context

        with edit_context(self.stage):
            b = CardioSolvUSDBuilder(self.stage, self.scan.mm_per_unit)
            b.clear()
            b.build(self.canonical, self.assignment, self.validation, self.cfg.preview_surface_colors)

    def run_geometry(self):
        self._require("discover")

        def go():
            self.compute_geometry()
            self.write_geometry()

        self.invalidate_after("geometry")
        return self._timed("geometry", go)

    # ------------------------------------------------------------------
    # Stage 3
    # ------------------------------------------------------------------
    def run_mesh(self):
        self._require("geometry")

        def go():
            gl = self.geometry
            myo_surface = self.scan.parts[gl.myocardium_part] if gl.myocardium_part is not None else None
            self.mesh = build_tet_mesh(gl, myo_surface, self.cfg.element_size_mm, self.cfg.mesher)
            self.coords = compute_coordinates(self.mesh, gl, self.cfg.endo_helix_deg, self.cfg.epi_helix_deg,
                                              self.cfg.sheet_gamma_deg)
            md = self.mesh.metadata
            self.log(f"    {md['nodes']} nodes, {md['tets']} tets, {md['volume_ml']:.1f} mL, "
                     f"surface snap {md.get('snap_mean_distance_mm', 0):.2f} mm")

        self.invalidate_after("mesh")
        return self._timed("mesh", go)

    # ------------------------------------------------------------------
    # Stage 4
    # ------------------------------------------------------------------
    def run_ep(self):
        self._require("mesh")

        def go():
            self.ep = run_electrophysiology(self.mesh, self.coords, self.cfg.ep, self.geometry.long_axis,
                                            self.geometry.septum_direction, log=self.log)
            m = self.ep.metrics
            self.log(f"    {self.cfg.ep.protocol}: QRS {m['qrs_duration_ms']:.0f} ms, total activation "
                     f"{m['total_activation_time_ms']:.0f} ms, septal->lateral delay {m['septal_to_lateral_delay_ms']:.0f} ms")

        self.invalidate_after("ep")
        return self._timed("ep", go)

    # ------------------------------------------------------------------
    # Stage 5
    # ------------------------------------------------------------------
    def _circulation(self):
        from .mechanics import CirculationParams

        return CirculationParams(r_periph=self.cfg.r_periph, c_art=self.cfg.c_art, edp=self.cfg.edp_mmhg,
                                 p_aortic_diastolic=self.cfg.p_aortic_diastolic, cycle_ms=self.cfg.cycle_ms,
                                 dt_ms=self.cfg.mechanics_dt_ms)

    def run_mechanics(self):
        self._require("ep")
        from .mechanics import MechanicsConfig, MechanicsModel, simulate_cycle

        def go():
            mcfg = MechanicsConfig(t_max_kpa=self.cfg.t_max_kpa, device=self.device)
            self.mech_model = MechanicsModel(self.mesh, self.coords, self.geometry.long_axis, mcfg,
                                             self.ep.activation_time, self.ep.apd)
            self.cycle = simulate_cycle(self.mech_model, self._circulation(), log=self.log,
                                        progress=lambda f: self.progress("mechanics", f))
            m = self.cycle.metrics
            self.log(f"    EDV {m['edv_ml']:.1f} mL, ESV {m['esv_ml']:.1f} mL, EF {m['ejection_fraction_pct']:.1f} %, "
                     f"LVP max {m['peak_lv_pressure_mmhg']:.0f} mmHg")

        self.invalidate_after("mechanics")
        return self._timed("mechanics", go)

    # ------------------------------------------------------------------
    # Stage 6
    # ------------------------------------------------------------------
    def run_surrogate(self):
        self._require("mechanics")
        from .surrogate import CardioPINNSurrogate, SurrogateConfig, calibrate_contractility, simulate_cycle_surrogate

        def go():
            self.surrogate = CardioPINNSurrogate(self.mech_model, self.cycle,
                                                 SurrogateConfig(epochs=self.cfg.surrogate_epochs))
            self.surrogate.train(log=self.log, progress=lambda f: self.progress("surrogate", f))
            circ = self._circulation()
            self.surrogate_cycle = simulate_cycle_surrogate(self.surrogate, circ)
            fe, su = self.cycle.metrics, self.surrogate_cycle["metrics"]
            self.log(f"    surrogate ({self.surrogate.backbone}): EF {su['ejection_fraction_pct']:.1f} % "
                     f"vs FE {fe['ejection_fraction_pct']:.1f} %")
            if not self.cfg.target_ef_pct and self.cfg.dashboard_url:
                try:
                    from .io.dashboard_bridge import fetch_echo_lvef

                    lvef = fetch_echo_lvef(self.cfg.dashboard_url)
                    if lvef:
                        self.cfg.target_ef_pct = float(lvef)
                        self.log(f"    echo LVEF {lvef} % fetched from the dashboard")
                except Exception as exc:
                    self.log(f"    dashboard not reachable ({exc}); no EF personalisation")
            if self.cfg.target_ef_pct:
                s, ef, ok = calibrate_contractility(self.surrogate, self.cfg.target_ef_pct, circ)
                self.calibration = {"target_ef_pct": self.cfg.target_ef_pct, "contractility_scale": s,
                                    "achieved_ef_pct": ef, "within_range": ok,
                                    "t_max_kpa_personalised": s * self.cfg.t_max_kpa}
                self.log(f"    calibrated contractility x{s:.2f} -> EF {ef:.1f} % (target {self.cfg.target_ef_pct})")

        self.invalidate_after("surrogate")
        return self._timed("surrogate", go)

    # ------------------------------------------------------------------
    # Stage 7
    # ------------------------------------------------------------------
    def nodal_field(self, name):
        """(T,N) or (N,) nodal values of a result field on the computational mesh."""
        cyc = self.cycle
        if name == "activation_time":
            return self.ep.activation_time
        if name == "transmural":
            return self.coords.x_t
        if name == "transmembrane_potential":
            if cyc is None:
                times = np.arange(0, 500, 10.0)
                return np.stack([self.ep.vm(t) for t in times])
            return np.stack([self.ep.vm(t) if t >= 0 else np.full(self.mesh.n_nodes, -85.0) for t in cyc.times])
        if name == "displacement":
            return np.linalg.norm(cyc.displacements, axis=2)
        if name == "active_tension":
            return element_to_nodal(self.mesh, cyc.active_tension_kpa)
        if name == "fiber_strain":
            return element_to_nodal(self.mesh, cyc.element_fields["fiber_strain"])
        if name == "fiber_stress":
            return element_to_nodal(self.mesh, cyc.element_fields["fiber_stress_kpa"])
        raise KeyError(name)

    def frame_times(self):
        if self.cycle is not None:
            return self.cycle.times
        return np.arange(0, 500, 10.0)

    def write_twin(self, field_name=None):
        from .usd.results_writer import TwinResultsWriter

        field_name = field_name or self.cfg.display_field
        myo_prim = None
        if self.geometry.myocardium_part is not None:
            myo_prim = self.scan.sources[self.geometry.myocardium_part].prim_path
        if self.writer is None:
            self.writer = TwinResultsWriter(self.stage, self.scan, self.mesh, myo_prim)
        times = self.frame_times()
        codes = self.writer.time_codes(times, self.cfg.slow_motion)
        if self.cycle is not None:
            self.writer.write_animation(codes, self.cycle.displacements, self.cfg.animate_neighbours)
        else:
            self.writer.set_time_range(codes)
        rng = self.writer.write_field(field_name, codes, self.nodal_field(field_name))
        self.writer.write_fibers(self.coords)
        return codes, rng

    def repaint(self, field_name):
        """Switch the field painted on the user's mesh (UI dropdown)."""
        self._require("twin")
        times = self.frame_times()
        codes = self.writer.time_codes(times, self.cfg.slow_motion)
        return self.writer.write_field(field_name, codes, self.nodal_field(field_name))

    def run_twin(self, export=True):
        self._require("ep")

        def go():
            self.write_twin()
            self.build_report()
            if export:
                self.export()

        return self._timed("twin", go)

    # ------------------------------------------------------------------
    def build_report(self):
        rep = {
            "cardiosolv_version": "0.3.0",
            "source_prim": self.source_path,
            "stage_file": self.stage.GetRootLayer().realPath or "",
            "timings_s": self.timings,
            "anatomy": self.assignment.to_dict() if self.assignment else None,
            "geometry": self.geometry.to_dict() if self.geometry else None,
            "validation": self.validation,
            "mesh": self.mesh.metadata if self.mesh else None,
            "fibers": self.coords.params if self.coords else None,
        }
        if self.ep is not None:
            rep["electrophysiology"] = {"protocol": self.cfg.ep.protocol, "solver": self.cfg.ep.solver,
                                        "metrics": self.ep.metrics}
            if self.ep.ecg is not None:
                rep["electrophysiology"]["pseudo_ecg"] = {"t_ms": self.ep.ecg_times,
                                                          **{k: v for k, v in self.ep.ecg.items()}}
        if self.cycle is not None:
            c = self.cycle
            rep["hemodynamics"] = {"metrics": c.metrics, "pv_loop": {"t_ms": c.times, "pressure_mmhg": c.pressure_mmhg,
                                                                      "volume_ml": c.volume_ml, "aortic_mmhg": c.aortic_mmhg,
                                                                      "phase": c.phase},
                                   "parameters": c.params}
        if self.surrogate_cycle is not None:
            s = self.surrogate_cycle
            rep["surrogate"] = {"backbone": self.surrogate.backbone, "pod_energy": self.surrogate.pod_energy,
                                "metrics": s["metrics"], "calibration": self.calibration,
                                "pv_loop": {"t_ms": s["times"], "pressure_mmhg": s["pressure_mmhg"],
                                            "volume_ml": s["volume_ml"]},
                                "training_history": self.surrogate.history}
        rep["config"] = _cfg_dict(self.cfg)
        self.report = rep
        return rep

    def export(self, vtk_every=2):
        from .io import export_opencarp, export_vtk_series, write_report
        from .usd.layer import save_layers

        out = self.output_dir
        files = {"report": write_report(os.path.join(out, "cardiosolv_report.json"), self.report or self.build_report())}
        if self.mesh is not None:
            files["opencarp"] = export_opencarp(os.path.join(out, "opencarp"), self.mesh, self.coords, self.ep)
            files["vtk"] = export_vtk_series(os.path.join(out, "vtk"), self.mesh, self.coords, self.ep, self.cycle,
                                             every=vtk_every)[:3] + ["..."]
        files["usd_layers"] = save_layers(self.stage)
        if self.cfg.dashboard_url:
            try:
                from .io.dashboard_bridge import post_report

                files["dashboard"] = post_report(self.cfg.dashboard_url, files["report"])
                self.log(f"[CardioSolv] report sent to dashboard {self.cfg.dashboard_url}")
            except Exception as exc:
                self.log(f"[CardioSolv] dashboard not reachable: {exc}")
        self.log(f"[CardioSolv] outputs written to {out}")
        return files

    # ------------------------------------------------------------------
    def _require(self, name):
        if name not in self.done:
            raise RuntimeError(f"Run '{STAGE_TITLES[name]}' first.")

    def run_all(self, until="twin", skip=()):
        for name in STAGES:
            if name in skip:
                continue
            getattr(self, f"run_{name}")()
            if name == until:
                break
        return self.report


def _cfg_dict(cfg):
    d = asdict(cfg)
    return d
