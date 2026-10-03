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
    "imaging": "0  Imaging: DICOM CT/MR -> 3D heart",
    "discover": "1  Anatomy discovery",
    "geometry": "2  Geometry layer (myocardium, endo/epi, long axis)",
    "mesh": "3  Computational mesh + fibres",
    "ep": "4  Electrophysiology",
    "mechanics": "5  Biomechanics + circulation",
    "surrogate": "6  Cardio-PINN surrogate",
    "twin": "7  Digital twin on your mesh",
    "crt": "CRT lead study",
}


@dataclass
class PipelineConfig:
    classification_voxels: int = 150_000
    auto_scale: bool = True  # simulate non-anatomical assets (e.g. a 1.1 m heart) at heart size
    anatomical_length_mm: float = 120.0
    skin_mode: str = "auto"  # "auto" | "on" | "off": derive the interior of skin-only hearts
    skin_lv_wall_mm: float = 10.0
    skin_septum_mm: float = 10.0
    skin_rv_wall_mm: float = 4.0
    skin_flip_base: bool = False
    skin_flip_lv_side: bool = False
    biventricular: bool = True  # simulate the RV free wall + RV cavity with a pulmonary circulation
    rv_wall_mm: float = 3.5  # RV free wall thickness when it has to be derived from the RV blood pool
    geometry_spacing_mm: Optional[float] = None  # auto
    scar: str = "auto"  # "auto": use scar / border-zone meshes found next to the heart | "off"
    scar_bz_cv_factor: float = 0.4
    scar_bz_contractility: float = 0.5
    scar_core_stiffness: float = 5.0
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
    rv_edp_mmhg: float = 5.0
    # passive personalisation: the imaged geometry state and how stiffness is fitted
    # set by the user: "end_diastolic" when the heart is an end-diastolic image (with edp_mmhg / rv_edp_mmhg
    # and optionally measured_edv_ml); "auto" infers it from Stage 0 provenance
    geometry_state: str = "unloaded"  # "unloaded" | "end_diastolic" | "auto"
    passive_calibration: str = "auto"  # "auto" | "off" | "klotz" | "measured"
    measured_edv_ml: Optional[float] = None  # e.g. echo / MR LVEDV at edp_mmhg
    pvr: float = 0.14  # mmHg s / mL, pulmonary resistance of the RV Windkessel
    c_pulmonary: float = 4.0  # mL / mmHg
    p_pulmonary_diastolic: float = 10.0
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
        self.passive = None
        self.scar = None
        self.scar_surfaces, self.scar_paths = {}, []
        self.crt = None
        self.writer = None
        self.skin_part = None
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
            self.extract_scar_surfaces()
            self.log(f"[CardioSolv] {len(self.scan.parts)} mesh parts under {self.source_path} "
                     f"(metersPerUnit={self.scan.meters_per_unit})")
            self.prepare_scan()
            self.compute_assignment()
            return self.assignment

        self.invalidate_after("discover")
        return self._timed("discover", go)

    def extract_scar_surfaces(self):
        """Scar / border-zone meshes (LGE reconstruction or drawn by the user) are not anatomy:
        take them out of the scan and keep them (world mm) for the scar substrate."""
        from pxr import Usd, UsdGeom

        from .core.scar import classify_scar_name
        from .usd.scene_scanner import read_mesh

        self.scar_surfaces, self.scar_paths = {}, []
        if self.cfg.scar == "off":
            return
        keep_p, keep_s = [], []
        for part, src in zip(self.scan.parts, self.scan.sources):
            code = classify_scar_name(src.prim_path.rsplit("/", 1)[-1])
            if code is None:
                keep_p.append(part)
                keep_s.append(src)
            else:
                self.scar_surfaces.setdefault(code, []).append(part)
                self.scar_paths.append(src.prim_path)
        self.scan.parts, self.scan.sources = keep_p, keep_s
        if not self.scan.parts:
            raise RuntimeError("Only scar meshes were found under the selected prim; select the heart.")
        # siblings of the heart, e.g. /World/Patient/Scar/scar_core from the imaging package
        root = self.stage.GetPrimAtPath(self.source_path)
        parent = root.GetParent()
        if parent and parent.IsValid() and str(parent.GetPath()) != "/":
            cache = UsdGeom.XformCache(Usd.TimeCode(self.scan.time_code))
            for prim in Usd.PrimRange(parent):
                path = str(prim.GetPath())
                if path == self.source_path or path.startswith(self.source_path + "/") or not prim.IsA(UsdGeom.Mesh):
                    continue
                names = [prim.GetName()] + [q.GetName() for q in (prim.GetParent(),) if q]
                code = next((c for c in (classify_scar_name(n) for n in names) if c is not None), None)
                if code is None:
                    continue
                try:
                    surf, _, _ = read_mesh(prim, Usd.TimeCode(self.scan.time_code), cache, self.scan.mm_per_unit)
                except Exception as exc:
                    self.log(f"    scar mesh {path} skipped ({exc})")
                    continue
                self.scar_surfaces.setdefault(code, []).append(surf)
                self.scar_paths.append(path)
        if self.scar_paths:
            self.log(f"    scar meshes: {', '.join(self.scar_paths)}")

    def build_scar(self):
        """Label the computational mesh with the scar meshes (after Stage 3)."""
        from .core.scar import ScarParams, build_scar_model, label_tets

        self.scar = None
        if not getattr(self, "scar_surfaces", None):
            return None
        sim = {}
        for code, surfs in self.scar_surfaces.items():
            out = []
            for srf in surfs:
                q = srf.copy()
                q.points = self.scan.to_sim(q.points)
                out.append(q)
            sim[code] = out
        prm = ScarParams(bz_cv_factor=self.cfg.scar_bz_cv_factor, bz_contractility=self.cfg.scar_bz_contractility,
                         core_stiffness=self.cfg.scar_core_stiffness)
        self.scar = build_scar_model(self.mesh, self.coords, label_tets(self.mesh, sim), prm, self.scar_paths)
        m = self.scar.metrics
        self.log(f"    scar: core {m['core_volume_ml']:.1f} mL, border zone {m['border_zone_volume_ml']:.1f} mL, "
                 f"LV burden {m['lv_scar_burden_pct']:.1f} %, {m['conduction_channels']} conduction channel(s)")
        return self.scar

    def prepare_scan(self):
        if self.cfg.auto_scale:
            length = self.scan.heart_length_mm()
            k = self.scan.apply_anatomical_scale(self.cfg.anatomical_length_mm)
            if k != 1.0:
                self.log(f"    heart is {length:.0f} mm long: simulating at {self.cfg.anatomical_length_mm:.0f} mm "
                         f"(scale x{k:.4f}); results are mapped back at the displayed size")

    def use_skin_mode(self) -> Optional[int]:
        """Index of the skin part when the heart has no internal anatomy, else None."""
        if self.cfg.skin_mode == "off":
            return None
        roles = self.assignment.roles
        if self.cfg.skin_mode == "auto" and (A.MYOCARDIUM in roles or A.LV in roles):
            return None
        feats = self.assignment.features
        big = max(feats, key=lambda f: f.volume_mm3)
        total = sum(f.volume_mm3 for f in feats) or 1.0
        # no myocardium / LV part: the heart is a skin if one part carries most of its volume
        if self.cfg.skin_mode == "on" or big.volume_mm3 / total > 0.7:
            return big.index
        return None

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
        self.skin_part = self.use_skin_mode()
        if self.skin_part is not None:
            from .core.skin_mode import SkinModeParams, build_skin_geometry_layer

            skin = self.scan.parts[self.skin_part]
            self.log(f"    skin-only heart ({skin.source_path}): deriving an ASSUMED ventricular interior")
            prm = SkinModeParams(spacing_mm=self.cfg.geometry_spacing_mm or 1.0, lv_wall_mm=self.cfg.skin_lv_wall_mm,
                                 septum_mm=self.cfg.skin_septum_mm, rv_wall_mm=self.cfg.skin_rv_wall_mm,
                                 flip_base=self.cfg.skin_flip_base, flip_lv_side=self.cfg.skin_flip_lv_side,
                                 biventricular=self.cfg.biventricular)
            self.geometry = build_skin_geometry_layer(skin, self.scan.up_axis, prm)
        else:
            self.geometry = build_geometry_layer(self.scan.parts, self.assignment, self.cfg.geometry_spacing_mm,
                                                 self.cfg.derived_wall_thickness_mm, biventricular=self.cfg.biventricular,
                                                 rv_wall_mm=self.cfg.rv_wall_mm)
        self.canonical = build_canonical(self.source_path, self.scan.parts, self.scan.sources, self.assignment,
                                         self.geometry)
        if self.skin_part is not None:
            self._add_skin_surfaces()
        self.validation = self.canonical.validate()
        for w in self.validation["warnings"]:
            self.log(f"    warning: {w}")
        m = self.geometry.metrics
        self.log(f"    myocardium {m['myocardial_volume_ml']:.1f} mL ({self.geometry.myocardium_source}), "
                 f"LV cavity {m['lv_cavity_volume_ml']:.1f} mL, long axis {m['long_axis_length_mm']:.1f} mm "
                 f"(conf {self.geometry.long_axis.confidence:.2f})")

    def _add_skin_surfaces(self):
        from .core.canonical import SurfaceComponent, faces_from_triangles
        from .core.skin_mode import SKIN_NAMES, label_skin_faces

        skin = self.scan.parts[self.skin_part]
        src = self.scan.sources[self.skin_part]
        lab = label_skin_faces(skin, self.geometry)
        areas = skin.face_areas() / max(self.scan.sim_scale, 1e-12) ** 2
        for code, name in SKIN_NAMES.items():
            sel = lab == code
            if sel.any():
                self.canonical.surfaces[name] = SurfaceComponent(
                    name=name, source_path=src.prim_path, component_type="assumed_" + name,
                    confidence=0.3, face_indices=faces_from_triangles(lab, skin.tri_to_face, code),
                    area_cm2=float(areas[sel].sum() / 100.0), metadata={"assumed": True})
        self.canonical.metadata["skin_mode"] = self.geometry.skin
        self.validation = self.canonical.validate()

    def write_geometry(self):
        from .usd.builder import CardioSolvUSDBuilder
        from .usd.layer import edit_context

        with edit_context(self.stage):
            b = CardioSolvUSDBuilder(self.stage, self.scan.mm_per_unit, to_world=self.scan.to_world,
                                     length_scale=1.0 / self.scan.sim_scale)
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
            skin = self.scan.parts[self.skin_part] if self.skin_part is not None else None
            self.mesh = build_tet_mesh(gl, myo_surface, self.cfg.element_size_mm, self.cfg.mesher, skin_surface=skin)
            self.coords = compute_coordinates(self.mesh, gl, self.cfg.endo_helix_deg, self.cfg.epi_helix_deg,
                                              self.cfg.sheet_gamma_deg)
            self.build_scar()
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
                                            self.geometry.septum_direction, log=self.log, scar=self.scar)
            if self.scar is not None and self.scar.channels:
                from .core.scar import channel_report

                self.scar.metrics["channels"] = channel_report(self.scar.channels, self.ep.activation_time)
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
                                 dt_ms=self.cfg.mechanics_dt_ms, rv_edp=self.cfg.rv_edp_mmhg,
                                 rv_r_periph=self.cfg.pvr, rv_c_art=self.cfg.c_pulmonary,
                                 p_pulmonary_diastolic=self.cfg.p_pulmonary_diastolic)

    def run_crt_study(self, fe_beat=False):
        """LV lead sweep for CRT in LBBB (eikonal), optionally with FE beats (LBBB vs best CRT)."""
        self._require("mesh")
        from .ep.crt import crt_study

        fe = None
        if fe_beat:
            from .mechanics import MechanicsConfig, MechanicsModel, simulate_cycle

            def fe(ep_res):
                model = MechanicsModel(self.mesh, self.coords, self.geometry.long_axis,
                                       MechanicsConfig(t_max_kpa=self.cfg.t_max_kpa, device=self.device),
                                       ep_res.activation_time, ep_res.apd, scar=self.scar)
                if self.mech_model is not None and self.passive:
                    model.cfg.material.stiff_scale = self.mech_model.cfg.material.stiff_scale
                    model.set_reference(self.mech_model.X.cpu().numpy())
                return simulate_cycle(model, self._circulation(), log=lambda *a: None).metrics

        def go():
            self.crt = crt_study(self.mesh, self.coords, self.cfg.ep, self.scar, self.geometry.long_axis,
                                 fe_beat=fe, log=self.log)
            return self.crt

        return self._timed("crt", go)

    def resolved_geometry_state(self) -> str:
        if self.cfg.geometry_state != "auto":
            return self.cfg.geometry_state
        # hearts reconstructed by Stage 0 are end-diastolic images (cine ED phase / CT)
        prim = self.stage.GetPrimAtPath(self.source_path)
        while prim and prim.IsValid():
            if prim.GetCustomDataByKey("cardiosolv:imaging_usd"):
                return "end_diastolic"
            prim = prim.GetParent()
        return "unloaded"

    def run_passive_calibration(self):
        """Unloaded reference + passive stiffness before the beat (see mechanics/passive.py)."""
        from .mechanics.passive import calibrate_passive

        state = self.resolved_geometry_state()
        mode = self.cfg.passive_calibration
        if mode == "auto":
            mode = "klotz" if state == "end_diastolic" else ("measured" if self.cfg.measured_edv_ml else "off")
        self.passive = None
        if mode == "off":
            return None
        self.log(f"    passive calibration: {mode} (geometry {state})")
        self.passive = calibrate_passive(self.mech_model, self.cfg.edp_mmhg, self.cfg.rv_edp_mmhg, mode,
                                         self.cfg.measured_edv_ml, log=self.log)
        self.passive["geometry_state"] = state
        d = self.passive["diastolic"]
        self.log(f"    stiffness x{self.passive['stiff_scale']:.2f}; unloaded LV {d['model_v0_ml']:.1f} mL "
                 f"(Klotz {d['klotz_v0_ml']:.1f}), EDPVR RMS error {d['edpvr_rms_error_ml']:.1f} mL")
        return self.passive

    def run_mechanics(self):
        self._require("ep")
        from .mechanics import MechanicsConfig, MechanicsModel, simulate_cycle

        def go():
            mcfg = MechanicsConfig(t_max_kpa=self.cfg.t_max_kpa, device=self.device)
            self.mech_model = MechanicsModel(self.mesh, self.coords, self.geometry.long_axis, mcfg,
                                             self.ep.activation_time, self.ep.apd, scar=self.scar)
            self.run_passive_calibration()
            self.cycle = simulate_cycle(self.mech_model, self._circulation(), log=self.log,
                                        progress=lambda f: self.progress("mechanics", f))
            m = self.cycle.metrics
            self.log(f"    EDV {m['edv_ml']:.1f} mL, ESV {m['esv_ml']:.1f} mL, EF {m['ejection_fraction_pct']:.1f} %, "
                     f"LVP max {m['peak_lv_pressure_mmhg']:.0f} mmHg")
            if "rv_edv_ml" in m:
                self.log(f"    RVEDV {m['rv_edv_ml']:.1f} mL, RVESV {m['rv_esv_ml']:.1f} mL, "
                         f"RVEF {m['rv_ejection_fraction_pct']:.1f} %, RVP max {m['peak_rv_pressure_mmhg']:.0f} mmHg, "
                         f"PA {m['pa_systolic_mmhg']:.0f}/{m['pa_diastolic_mmhg']:.0f} mmHg")

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
        if name == "scar":
            return (self.scar.node_label() if self.scar is not None else np.zeros(self.mesh.n_nodes)).astype(float)
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
            paint = [self.scan.sources[self.skin_part].prim_path] if getattr(self, "skin_part", None) is not None else []
            rv_i = self.assignment.part(A.RV) if self.assignment else None
            if self.mesh.biventricular:
                # the RV wall is simulated: paint the user's RV (blood pool / myocardium) parts too
                for i in (rv_i, self.assignment.part(A.RV_MYOCARDIUM)):
                    if i is not None:
                        paint.append(self.scan.sources[i].prim_path)
            self.writer = TwinResultsWriter(self.stage, self.scan, self.mesh, myo_prim, paint_prims=paint)
        times = self.frame_times()
        codes = self.writer.time_codes(times, self.cfg.slow_motion)
        if self.cycle is not None:
            self.writer.write_animation(codes, self.cycle.displacements, self.cfg.animate_neighbours)
        else:
            self.writer.set_time_range(codes)
        rng = self.writer.write_field(field_name, codes, self.nodal_field(field_name))
        self.writer.write_fibers(self.coords)
        if self.geometry.myocardium_source != "part":
            self.writer.write_assumed_endocardium()
        return codes, rng

    def repaint(self, field_name):
        """Switch the field painted on the user's mesh (UI dropdown)."""
        self._require("twin")
        times = self.frame_times()
        codes = self.writer.time_codes(times, self.cfg.slow_motion)
        return self.writer.write_field(field_name, codes, self.nodal_field(field_name))

    def save_twin_stage(self, out_path=None):
        """Write ``<stage>_twin.usda`` = [twin results, semantics, original stage] (original untouched)."""
        from .usd.results_writer import export_twin_stage

        if out_path is None:
            root = self.stage.GetRootLayer()
            stem = os.path.splitext(root.realPath)[0] if root.realPath else os.path.join(self.output_dir, "heart")
            out_path = f"{stem}_twin.usda"
        path = export_twin_stage(self.stage, out_path, self.writer)
        self.log(f"[CardioSolv] twin stage saved: {path}")
        return path

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
            "cardiosolv_version": "0.6.0",
            "biventricular": bool(self.mech_model is not None and self.mech_model.biventricular)
            or bool(self.mesh is not None and self.mesh.biventricular),
            "simulation_scale": self.scan.sim_scale if self.scan else 1.0,
            "skin_mode": bool(getattr(self, "skin_part", None) is not None),
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
            for name, tr in c.chambers.items():
                rep["hemodynamics"][f"{name.lower()}_pv_loop"] = {
                    "t_ms": c.times, "pressure_mmhg": tr["pressure_mmhg"], "volume_ml": tr["volume_ml"],
                    "pulmonary_mmhg" if name == "RV" else "arterial_mmhg": tr["arterial_mmhg"], "phase": tr["phase"]}
        if self.scar is not None:
            rep["scar"] = {"sources": self.scar.sources, "params": self.scar.params.__dict__,
                           "metrics": self.scar.metrics}
        if self.crt is not None:
            rep["crt_study"] = self.crt
        if getattr(self, "passive", None):
            rep["diastolic_calibration"] = self.passive
        if self.surrogate_cycle is not None:
            s = self.surrogate_cycle
            rep["surrogate"] = {"backbone": self.surrogate.backbone, "pod_energy": self.surrogate.pod_energy,
                                "metrics": s["metrics"], "calibration": self.calibration,
                                "pv_loop": {"t_ms": s["times"], "pressure_mmhg": s["pressure_mmhg"],
                                            "volume_ml": s["volume_ml"]},
                                **{f"{n.lower()}_pv_loop": {"t_ms": s["times"], "pressure_mmhg": tr["pressure_mmhg"],
                                                            "volume_ml": tr["volume_ml"]}
                                   for n, tr in s.get("chambers", {}).items()},
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
