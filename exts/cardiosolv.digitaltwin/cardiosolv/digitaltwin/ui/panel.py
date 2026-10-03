"""CardioSolv panel: walk the digital-twin stages one by one on the selected heart."""

from __future__ import annotations

import asyncio
import traceback
from functools import partial

import carb
import carb.settings
import numpy as np
import omni.kit.app
import omni.timeline
import omni.ui as ui
import omni.usd

from ..core import anatomy as A
from ..ep import PACING_PROTOCOLS
from ..imaging_client import ImagingConfig, load_heart_into_stage, segment
from ..pipeline import STAGE_TITLES, STAGES, CardioSolvPipeline, PipelineConfig
from ..usd.results_writer import FIELD_STYLE

SETTINGS = "/exts/cardiosolv.digitaltwin"
ROLE_CHOICES = ["(auto)", "Unknown"] + A.CARDIAC_STRUCTURES
FIELDS = list(FIELD_STYLE)
EP_SOLVERS = ["eikonal", "monodomain"]
IMG_MODES = ["local", "service"]
IMG_MODALITIES = ["auto", "CT", "MR"]
IMG_ENGINES = ["auto", "nv-segment", "totalsegmentator", "both"]
OK, WARN, ERR = 0xFF66CC66, 0xFF33BBFF, 0xFF5555FF


def _combo_value(model):
    return model.get_item_value_model().as_int


class CardioSolvPanel:
    def __init__(self, ext_id=""):
        self.window = None
        self.pipe: CardioSolvPipeline = None
        self.cfg = PipelineConfig()
        s = carb.settings.get_settings()
        self.cfg.element_size_mm = s.get(f"{SETTINGS}/element_size_mm") or self.cfg.element_size_mm
        self.cfg.slow_motion = s.get(f"{SETTINGS}/slow_motion") or self.cfg.slow_motion
        self.cfg.device = s.get(f"{SETTINGS}/device") or self.cfg.device
        self.icfg = ImagingConfig(
            mode=s.get(f"{SETTINGS}/imaging_mode") or "local",
            python=s.get(f"{SETTINGS}/imaging_python") or "python",
            package_dir=s.get(f"{SETTINGS}/imaging_package_dir") or None,
            service_url=s.get(f"{SETTINGS}/imaging_service_url") or "http://localhost:8040")
        self.imaging_input = ""
        self.source_path = None
        self._busy = False
        self._log_lines = []
        self._labels = {}
        self._frames = {}
        self._progress = {}
        self._parts_frame = None
        self._plots_frame = {}
        self._pending_progress = {}
        self._log_dirty = False
        self._update_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
            self._on_update, name="cardiosolv.ui")

    # ------------------------------------------------------------------ UI
    def show(self, *_):
        if self.window:
            self.window.visible = True
            return
        self.window = ui.Window("CardioSolv Digital Twin", width=520, height=900)
        with self.window.frame:
            with ui.ScrollingFrame():
                with ui.VStack(spacing=6, height=0):
                    ui.Label("CardioSolv  -  Cardiac Digital Twin", height=28, style={"font_size": 20})
                    ui.Label("Select your heart root prim in the Stage, then run the stages in order. "
                             "Nothing replaces your mesh: results are painted and animated on it via sublayers.",
                             word_wrap=True, height=0, style={"color": 0xFFAAAAAA})
                    with ui.HStack(height=26, spacing=6):
                        ui.Button("Use Selected Heart", clicked_fn=self._use_selection, width=150)
                        self._labels["source"] = ui.Label("Source: none", elided_text=True)
                    with ui.HStack(height=26, spacing=6):
                        ui.Button("Run All Stages", clicked_fn=lambda: self._spawn(self._run_all()))
                        ui.Button("Clear Results", clicked_fn=self._clear_results, width=120)
                    self._build_stage_imaging()
                    self._build_stage_discover()
                    self._build_stage_geometry()
                    self._build_stage_mesh()
                    self._build_stage_ep()
                    self._build_stage_mechanics()
                    self._build_stage_surrogate()
                    self._build_stage_twin()
                    with ui.CollapsableFrame("Log", collapsed=False, height=0):
                        self._labels["log"] = ui.Label("", word_wrap=True, height=0, style={"font_size": 13})

    def _stage_header(self, name):
        frame = ui.CollapsableFrame(STAGE_TITLES[name], collapsed=name not in ("imaging", "discover"), height=0)
        self._frames[name] = frame
        return frame

    def _status_row(self, name):
        with ui.HStack(height=20, spacing=4):
            self._labels[f"{name}_status"] = ui.Label("not run", width=0)
            self._progress[name] = ui.ProgressBar(height=16)

    def _float_field(self, label, value, key, width=90):
        with ui.HStack(height=22):
            ui.Label(label, width=220)
            f = ui.FloatField(width=width)
            f.model.set_value(float(value))
            f.model.add_value_changed_fn(lambda m, k=key: self._set_cfg(k, m.as_float))
        return f

    def _set_cfg(self, key, value):
        obj = self.cfg
        parts = key.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], value)

    def _string_field(self, label, value, setter):
        with ui.HStack(height=22):
            ui.Label(label, width=220)
            f = ui.StringField()
            f.model.set_value(value or "")
            f.model.add_value_changed_fn(lambda m: setter(m.as_string))
        return f

    def _combo(self, label, options, current, setter):
        with ui.HStack(height=22):
            ui.Label(label, width=220)
            c = ui.ComboBox(options.index(current) if current in options else 0, *options)
            c.model.add_item_changed_fn(lambda m, _: setter(options[_combo_value(m)]))
        return c

    def _build_stage_imaging(self):
        ic = self.icfg
        with self._stage_header("imaging"):
            with ui.VStack(spacing=4, height=0):
                ui.Label("Segment the heart and great vessels from a CT or MR DICOM folder (or NIfTI) with "
                         "NV-Segment-CTMR / TotalSegmentator, reconstruct it as USD and load it here. "
                         "Skip this stage if your heart is already in the stage.", word_wrap=True)
                self._string_field("DICOM folder / NIfTI / heart.usda", "", lambda v: setattr(self, "imaging_input", v))
                self._combo("Run segmentation", IMG_MODES, ic.mode, lambda v: setattr(ic, "mode", v))
                self._combo("Modality", IMG_MODALITIES, ic.modality, lambda v: setattr(ic, "modality", v))
                self._combo("Engine", IMG_ENGINES, ic.engine, lambda v: setattr(ic, "engine", v))
                self._string_field("Cine MR phase (ed / es / index)", ic.phase, lambda v: setattr(ic, "phase", v or "ed"))
                self._string_field("LGE MR (DICOM / NIfTI, optional)", "", lambda v: setattr(ic, "lge", v.strip() or None))
                self._combo("LGE scar method", ["nsd", "fwhm"], ic.scar_method, lambda v: setattr(ic, "scar_method", v))
                self._string_field("Local: imaging Python", ic.python, lambda v: setattr(ic, "python", v or "python"))
                self._string_field("Local: imaging package dir", ic.package_dir,
                                   lambda v: setattr(ic, "package_dir", v or None))
                self._string_field("Service URL", ic.service_url, lambda v: setattr(ic, "service_url", v))
                with ui.HStack(height=26, spacing=4):
                    ui.Button("Segment & Load Heart", clicked_fn=lambda: self._spawn(self._run_imaging()))
                    ui.Button("Load heart.usda", width=130, clicked_fn=self._load_existing_heart)
                self._status_row("imaging")
                self._labels["imaging_info"] = ui.Label("", word_wrap=True, height=0)

    async def _run_imaging(self):
        import os

        src = self.imaging_input.strip()
        if not src or not os.path.exists(src):
            self._log("Enter an existing DICOM folder or NIfTI file for Stage 0.")
            return
        self._busy = True
        self._set_status("imaging", "segmenting...", WARN)
        self._on_progress("imaging", 0.1)
        try:
            out = os.path.join(os.path.dirname(os.path.abspath(src.rstrip("/\\"))),
                               "cardiosolv_imaging_" + os.path.basename(src.rstrip("/\\")).split(".")[0])
            rep = await self._in_thread(segment, src, out, self.icfg, self._log)
            self._on_progress("imaging", 0.9)
            self._attach_heart(rep["outputs"]["usd"])
            qc = rep.get("qc", {})
            vols = ", ".join(f"{k.replace('heart_', '')} {v['volume_ml']:.0f} mL"
                             for k, v in rep.get("structures", {}).items())
            sc = rep.get("scar")
            self._labels["imaging_info"].text = (f"{rep.get('modality', '')}: {qc.get('status', '')}\n{vols}"
                                                 + (f"\nLGE scar: core {sc['core_volume_ml']} mL, border zone "
                                                    f"{sc['border_zone_volume_ml']} mL ({sc['scar_burden_pct']} % of LV)"
                                                    if sc else "")
                                                 + "".join(f"\n! {w}" for w in qc.get("warnings", [])[:5]))
            self._set_status("imaging", "done", OK)
        except Exception as exc:
            carb.log_error(traceback.format_exc())
            self._set_status("imaging", f"error: {exc}", ERR)
            self._log(f"ERROR in imaging: {exc}")
        finally:
            self._on_progress("imaging", 1.0)
            self._busy = False

    def _load_existing_heart(self):
        import os

        path = self.imaging_input.strip()
        if not path.lower().endswith((".usd", ".usda", ".usdc", ".usdz")) or not os.path.isfile(path):
            self._log("Enter the path of a heart.usda produced by cardiosolv-segment.")
            return
        try:
            self._attach_heart(path)
            self._set_status("imaging", "loaded", OK)
        except Exception as exc:
            self._set_status("imaging", f"error: {exc}", ERR)

    def _attach_heart(self, usd_path):
        ctx = omni.usd.get_context()
        heart = load_heart_into_stage(ctx.get_stage(), usd_path)
        ctx.get_selection().set_selected_prim_paths([heart], True)
        self._use_selection()
        self._log(f"Reconstructed heart loaded at {heart}; continue with Stage 1.")

    def _build_stage_discover(self):
        with self._stage_header("discover"):
            with ui.VStack(spacing=4, height=0):
                ui.Label("Scans every mesh under the selected prim and scores each part against the "
                         "cardiac vocabulary (names + hollowness + containment + contact).", word_wrap=True)
                ui.Button("Discover Anatomy", height=26, clicked_fn=lambda: self._spawn(self._run_stage("discover")))
                self._status_row("discover")
                self._parts_frame = ui.Frame(height=0)

    def _build_stage_geometry(self):
        with self._stage_header("geometry"):
            with ui.VStack(spacing=4, height=0):
                self._float_field("Derived wall thickness (mm)", self.cfg.derived_wall_thickness_mm,
                                  "derived_wall_thickness_mm")
                with ui.HStack(height=22):
                    ui.Label("Preview endo/epi colours on mesh", width=220)
                    cb = ui.CheckBox()
                    cb.model.set_value(self.cfg.preview_surface_colors)
                    cb.model.add_value_changed_fn(lambda m: self._set_cfg("preview_surface_colors", m.as_bool))
                with ui.HStack(height=22):
                    ui.Label("Skin-only heart: derive interior", width=220)
                    modes = ["auto", "on", "off"]
                    c = ui.ComboBox(0, *modes)
                    c.model.add_item_changed_fn(lambda m, _: self._set_cfg("skin_mode", modes[_combo_value(m)]))
                for label, key in (("Skin mode: flip apex/base", "skin_flip_base"),
                                   ("Skin mode: flip LV/RV side", "skin_flip_lv_side")):
                    with ui.HStack(height=22):
                        ui.Label(label, width=220)
                        cb = ui.CheckBox()
                        cb.model.add_value_changed_fn(lambda m, k=key: self._set_cfg(k, m.as_bool))
                with ui.HStack(height=22):
                    ui.Label("Biventricular (RV wall + RV cavity)", width=220)
                    cb = ui.CheckBox()
                    cb.model.set_value(self.cfg.biventricular)
                    cb.model.add_value_changed_fn(lambda m: self._set_cfg("biventricular", m.as_bool))
                self._float_field("Derived RV wall thickness (mm)", self.cfg.rv_wall_mm, "rv_wall_mm")
                ui.Button("Build Geometry Layer", height=26, clicked_fn=lambda: self._spawn(self._run_stage("geometry")))
                self._status_row("geometry")
                self._labels["geometry_info"] = ui.Label("", word_wrap=True, height=0)

    def _build_stage_mesh(self):
        with self._stage_header("mesh"):
            with ui.VStack(spacing=4, height=0):
                self._float_field("Element size (mm)", self.cfg.element_size_mm, "element_size_mm")
                self._float_field("Endocardial helix angle (deg)", self.cfg.endo_helix_deg, "endo_helix_deg")
                self._float_field("Epicardial helix angle (deg)", self.cfg.epi_helix_deg, "epi_helix_deg")
                self._float_field("Sheet angle gamma (deg)", self.cfg.sheet_gamma_deg, "sheet_gamma_deg")
                ui.Button("Build Mesh + Fibres", height=26, clicked_fn=lambda: self._spawn(self._run_stage("mesh")))
                self._status_row("mesh")
                self._labels["mesh_info"] = ui.Label("", word_wrap=True, height=0)

    def _build_stage_ep(self):
        with self._stage_header("ep"):
            with ui.VStack(spacing=4, height=0):
                protos = list(PACING_PROTOCOLS)
                with ui.HStack(height=22):
                    ui.Label("Scenario", width=220)
                    c = ui.ComboBox(protos.index(self.cfg.ep.protocol), *protos)
                    c.model.add_item_changed_fn(lambda m, _: self._set_cfg("ep.protocol", protos[_combo_value(m)]))
                with ui.HStack(height=22):
                    ui.Label("Solver", width=220)
                    c = ui.ComboBox(0, *EP_SOLVERS)
                    c.model.add_item_changed_fn(lambda m, _: self._set_cfg("ep.solver", EP_SOLVERS[_combo_value(m)]))
                self._float_field("Fibre conduction velocity (mm/ms)", self.cfg.ep.cv_fiber, "ep.cv_fiber")
                self._float_field("Cross-fibre CV (mm/ms)", self.cfg.ep.cv_cross, "ep.cv_cross")
                ui.Button("Run Electrophysiology", height=26, clicked_fn=lambda: self._spawn(self._run_stage("ep")))
                with ui.HStack(height=26, spacing=4):
                    ui.Button("CRT lead study", clicked_fn=lambda: self._spawn(self._run_crt()))
                    ui.Label("with FE beats", width=90)
                    cb = ui.CheckBox(width=20)
                    cb.model.add_value_changed_fn(lambda m: setattr(self, "_crt_fe", m.as_bool))
                self._labels["crt_info"] = ui.Label("", word_wrap=True, height=0)
                self._status_row("ep")
                self._labels["ep_info"] = ui.Label("", word_wrap=True, height=0)
                self._plots_frame["ep"] = ui.Frame(height=0)

    def _build_stage_mechanics(self):
        with self._stage_header("mechanics"):
            with ui.VStack(spacing=4, height=0):
                self._float_field("Peak active tension (kPa)", self.cfg.t_max_kpa, "t_max_kpa")
                self._float_field("End-diastolic pressure (mmHg)", self.cfg.edp_mmhg, "edp_mmhg")
                self._float_field("Peripheral resistance (mmHg s/mL)", self.cfg.r_periph, "r_periph")
                self._float_field("Arterial compliance (mL/mmHg)", self.cfg.c_art, "c_art")
                ui.Label("Diastole / passive personalisation", height=18)
                states = ["unloaded", "end_diastolic", "auto"]
                with ui.HStack(height=22):
                    ui.Label("Imaged geometry is", width=220)
                    c = ui.ComboBox(states.index(self.cfg.geometry_state), *states,
                                    tooltip="end_diastolic: your heart is an end-diastolic image at the LV / RV EDP "
                                            "entered below; the unloaded heart and the passive stiffness are fitted")
                    c.model.add_item_changed_fn(lambda m, _: self._set_cfg("geometry_state", states[_combo_value(m)]))
                calib = ["auto", "off", "klotz", "measured"]
                with ui.HStack(height=22):
                    ui.Label("Passive stiffness fit", width=220)
                    c = ui.ComboBox(0, *calib)
                    c.model.add_item_changed_fn(lambda m, _: self._set_cfg("passive_calibration", calib[_combo_value(m)]))
                with ui.HStack(height=22):
                    ui.Label("Measured LVEDV (mL, 0 = none)", width=220)
                    f = ui.FloatField(width=90)
                    f.model.add_value_changed_fn(lambda m: self._set_cfg("measured_edv_ml", m.as_float or None))
                ui.Label("Right heart (biventricular twins)", height=18)
                self._float_field("RV end-diastolic pressure (mmHg)", self.cfg.rv_edp_mmhg, "rv_edp_mmhg")
                self._float_field("Pulmonary resistance (mmHg s/mL)", self.cfg.pvr, "pvr")
                self._float_field("Pulmonary compliance (mL/mmHg)", self.cfg.c_pulmonary, "c_pulmonary")
                self._float_field("PA diastolic pressure (mmHg)", self.cfg.p_pulmonary_diastolic, "p_pulmonary_diastolic")
                self._float_field("Cycle length (ms)", self.cfg.cycle_ms, "cycle_ms")
                self._float_field("Time step (ms)", self.cfg.mechanics_dt_ms, "mechanics_dt_ms")
                ui.Button("Simulate Heartbeat", height=26, clicked_fn=lambda: self._spawn(self._run_stage("mechanics")))
                self._status_row("mechanics")
                self._labels["mechanics_info"] = ui.Label("", word_wrap=True, height=0)
                self._plots_frame["mechanics"] = ui.Frame(height=0)

    def _build_stage_surrogate(self):
        with self._stage_header("surrogate"):
            with ui.VStack(spacing=4, height=0):
                self._float_field("Training epochs", self.cfg.surrogate_epochs, "surrogate_epochs")
                with ui.HStack(height=22):
                    ui.Label("Target EF from echo (%, 0 = off)", width=220)
                    f = ui.FloatField(width=90)
                    f.model.set_value(0.0)
                    f.model.add_value_changed_fn(lambda m: self._set_cfg("target_ef_pct", m.as_float or None))
                with ui.HStack(height=22):
                    ui.Label("Echo dashboard URL (optional)", width=220)
                    sf = ui.StringField()
                    sf.model.set_value(self.cfg.dashboard_url or "")
                    sf.model.add_value_changed_fn(lambda m: self._set_cfg("dashboard_url", m.as_string or None))
                ui.Button("Train Cardio-PINN Surrogate", height=26,
                          clicked_fn=lambda: self._spawn(self._run_stage("surrogate")))
                self._status_row("surrogate")
                self._labels["surrogate_info"] = ui.Label("", word_wrap=True, height=0)
                with ui.HStack(height=22):
                    ui.Label("What-if contractility", width=220)
                    sl = ui.FloatSlider(min=0.4, max=1.6)
                    sl.model.set_value(1.0)
                    sl.model.add_end_edit_fn(lambda m: self._what_if(m.as_float))
                self._labels["whatif"] = ui.Label("", height=0)

    def _build_stage_twin(self):
        with self._stage_header("twin"):
            with ui.VStack(spacing=4, height=0):
                with ui.HStack(height=22):
                    ui.Label("Field painted on your mesh", width=220)
                    c = ui.ComboBox(FIELDS.index(self.cfg.display_field), *FIELDS)
                    c.model.add_item_changed_fn(lambda m, _: self._repaint(FIELDS[_combo_value(m)]))
                self._float_field("Slow motion factor", self.cfg.slow_motion, "slow_motion")
                with ui.HStack(height=22):
                    ui.Label("Animate neighbouring parts", width=220)
                    cb = ui.CheckBox()
                    cb.model.set_value(self.cfg.animate_neighbours)
                    cb.model.add_value_changed_fn(lambda m: self._set_cfg("animate_neighbours", m.as_bool))
                with ui.HStack(height=26, spacing=4):
                    ui.Button("Build Twin on My Mesh", clicked_fn=lambda: self._spawn(self._run_stage("twin")))
                    ui.Button("Play", width=60, clicked_fn=self._play)
                    ui.Button("Save Twin Stage", width=130, clicked_fn=self._save_twin_stage)
                self._status_row("twin")
                self._labels["twin_info"] = ui.Label("", word_wrap=True, height=0)

    # ------------------------------------------------------------ actions
    def _use_selection(self):
        ctx = omni.usd.get_context()
        paths = ctx.get_selection().get_selected_prim_paths()
        if len(paths) != 1:
            self._log("Select exactly one heart root prim in the Stage window.")
            return
        if paths[0].startswith("/CardioSolv"):
            self._log("CardioSolv cannot use its own generated hierarchy as source.")
            return
        self.source_path = paths[0]
        self.pipe = CardioSolvPipeline(ctx.get_stage(), self.source_path, self.cfg, log=self._log,
                                       progress=self._on_progress)
        self._labels["source"].text = f"Source: {self.source_path}"
        for name in STAGES:
            self._set_status(name, "not run")
        self._log(f"Source heart: {self.source_path}")

    def _spawn(self, coro):
        if self._busy:
            self._log("A stage is already running.")
            return
        asyncio.ensure_future(coro)

    async def _in_thread(self, fn, *args):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, partial(fn, *args))

    async def _run_all(self):
        for name in STAGES:
            ok = await self._run_stage(name, chained=True)
            if not ok:
                break

    async def _run_stage(self, name, chained=False):
        if self.pipe is None:
            self._use_selection()
            if self.pipe is None:
                return False
        if self._busy and not chained:
            return False
        self._busy = True
        self._set_status(name, "running...", WARN)
        await omni.kit.app.get_app().next_update_async()
        p = self.pipe
        try:
            if name == "discover":
                from ..usd.scene_scanner import scan_heart

                p.invalidate_after("discover")
                p.scan = scan_heart(p.stage, p.source_path)  # USD read on main thread
                p.extract_scar_surfaces()  # scar / border-zone meshes are not anatomy
                p.prepare_scan()
                await self._in_thread(p.compute_assignment)
                p.done.append("discover")
                self._show_parts()
            elif name == "geometry":
                p._require("discover")
                p.invalidate_after("geometry")
                await self._in_thread(p.compute_geometry)
                p.write_geometry()  # USD authoring on main thread
                p.done.append("geometry")
                self._show_geometry()
            elif name == "twin":
                p.write_twin()
                await self._in_thread(p.build_report)
                files = await self._in_thread(p.export)
                p.done.append("twin")
                self._labels["twin_info"].text = f"Painted '{self.cfg.display_field}' and animated your mesh.\n" \
                                                 f"Report: {files['report']}"
                self._play()
            else:
                await self._in_thread(getattr(p, f"run_{name}"))
                getattr(self, f"_show_{name}")()
            self._set_status(name, "done", OK)
            return True
        except Exception as exc:
            carb.log_error(traceback.format_exc())
            self._set_status(name, f"error: {exc}", ERR)
            self._log(f"ERROR in {name}: {exc}")
            return False
        finally:
            self._busy = False

    def _repaint(self, field):
        self.cfg.display_field = field
        if self.pipe and "twin" in self.pipe.done:
            try:
                lo, hi = self.pipe.repaint(field)
                units = FIELD_STYLE[field][0]
                self._labels["twin_info"].text = f"Showing {field}: {lo:.3g} .. {hi:.3g} {units}"
            except Exception as exc:
                self._log(f"repaint failed: {exc}")

    def _play(self):
        tl = omni.timeline.get_timeline_interface()
        stage = omni.usd.get_context().get_stage()
        tcps = stage.GetTimeCodesPerSecond() or 60.0
        tl.set_start_time(stage.GetStartTimeCode() / tcps)
        tl.set_end_time(stage.GetEndTimeCode() / tcps)
        tl.set_looping(True)
        tl.play()

    def _save_twin_stage(self):
        if not self.pipe or "twin" not in self.pipe.done:
            self._log("Build the twin first.")
            return
        try:
            path = self.pipe.save_twin_stage()
            self._labels["twin_info"].text += f"\nTwin stage: {path}"
        except Exception as exc:
            self._log(f"Saving twin stage failed: {exc}")

    def _clear_results(self):
        if self.pipe and self.pipe.writer:
            self.pipe.writer.clear()
        stage = omni.usd.get_context().get_stage()
        if stage and stage.GetPrimAtPath("/CardioSolv"):
            from ..usd.layer import edit_context

            with edit_context(stage):
                stage.RemovePrim("/CardioSolv")
        self._log("CardioSolv results removed from the stage (your mesh is untouched).")

    async def _run_crt(self):
        p = self.pipe
        if p is None or "mesh" not in p.done:
            self._log("Build the mesh (stage 3) first.")
            return
        self._busy = True
        try:
            r = await self._in_thread(p.run_crt_study, bool(getattr(self, "_crt_fe", False)))
            b, resp = r["best_site"], r["response"]
            txt = (f"LBBB LV activation {r['lbbb']['lv_activation_time_ms']:.0f} ms -> best LV lead "
                   f"(x_l {b['x_l']:.2f}, x_c {b['x_c']:.2f}) {b['lv_activation_time_ms']:.0f} ms, QRS "
                   f"{b['qrs_duration_ms']:.0f} ms\nPredicted response: {resp['predicted_response']}"
                   + "".join(f"; {x}" for x in resp["reasons"]))
            if "haemodynamics" in r:
                h = r["haemodynamics"]
                txt += f"\nFE: dP/dt max {h['dpdt_max_change_pct']:+.1f} %, EF {h['ef_change_points']:+.1f} points"
            self._labels["crt_info"].text = txt
        except Exception as exc:
            carb.log_error(traceback.format_exc())
            self._log(f"ERROR in CRT study: {exc}")
        finally:
            self._busy = False

    def _what_if(self, scale):
        p = self.pipe
        if not p or p.surrogate is None:
            return
        from ..surrogate import simulate_cycle_surrogate

        r = simulate_cycle_surrogate(p.surrogate, p._circulation(), scale)
        m = r["metrics"]
        self._labels["whatif"].text = (f"x{scale:.2f}: EF {m['ejection_fraction_pct']:.1f} %, SV "
                                       f"{m['stroke_volume_ml']:.1f} mL, LVP {m['peak_lv_pressure_mmhg']:.0f} mmHg"
                                       + (f" | RVEF {m['rv_ejection_fraction_pct']:.1f} %, RVP "
                                          f"{m['peak_rv_pressure_mmhg']:.0f} mmHg" if "rv_edv_ml" in m else ""))

    # ------------------------------------------------------------ displays
    def _show_parts(self):
        p = self.pipe
        asg = p.assignment
        role_of = {i: r for r, i in asg.roles.items()}
        with self._parts_frame:
            with ui.VStack(spacing=2, height=0):
                for k, part in enumerate(p.scan.parts):
                    role = role_of.get(k, "Unknown")
                    score = asg.scores.get(role, 0.0)
                    status = asg.status(role) if role in asg.roles else "-"
                    color = OK if status in ("HIGH", "USER_CONFIRMED") else WARN if status != "-" else 0xFF888888
                    with ui.HStack(height=22, spacing=4):
                        ui.Label(part.source_path.split("/")[-1], width=170, elided_text=True,
                                 tooltip=part.source_path)
                        idx = ROLE_CHOICES.index(role) if role in ROLE_CHOICES else 1
                        combo = ui.ComboBox(idx, *ROLE_CHOICES, width=170)
                        combo.model.add_item_changed_fn(lambda m, _, k=k: self._override(k, ROLE_CHOICES[_combo_value(m)]))
                        ui.Label(f"{score:.2f} {status}", style={"color": color})
                for w in asg.warnings:
                    ui.Label(f"! {w}", word_wrap=True, height=0, style={"color": WARN})

    def _override(self, part_index, role):
        if role in ("(auto)",):
            for r, i in list(self.pipe.overrides.items()):
                if i == part_index:
                    self.pipe.set_role(r, None)
        elif role != "Unknown":
            self.pipe.set_role(role, part_index)
        self._show_parts()
        for name in STAGES[1:]:
            self._set_status(name, "not run")

    def _show_geometry(self):
        g = self.pipe.geometry
        v = self.pipe.validation
        m = g.metrics
        self._labels["geometry_info"].text = (
            f"Myocardium: {m['myocardial_volume_ml']:.1f} mL ({g.myocardium_source}), mass {m['myocardial_mass_g']:.0f} g\n"
            f"LV cavity: {m['lv_cavity_volume_ml']:.1f} mL, wall {m['mean_wall_thickness_mm']:.1f} mm\n"
            + (f"RV free wall: {m['rv_wall_volume_ml']:.1f} mL, RV cavity {m['rv_cavity_volume_ml']:.1f} mL "
               f"(biventricular)\n" if g.biventricular else "") +
            f"Long axis: {m['long_axis_length_mm']:.1f} mm, confidence {g.long_axis.confidence:.2f} "
            f"({', '.join(f'{x.name}:{x.sign:+.0f}' for x in g.long_axis.votes)})\n"
            f"Surfaces: {g.to_dict()['surface_label_counts']}\nValidation: {v['status']}"
            + "".join(f"\n! {w}" for w in v["warnings"][:6]))

    def _show_mesh(self):
        md = self.pipe.mesh.metadata
        self._labels["mesh_info"].text = (f"{md['nodes']} nodes, {md['tets']} tets, {md['volume_ml']:.1f} mL; "
                                          f"snap to your surface {md.get('snap_mean_distance_mm', 0):.2f} mm"
                                          + (f"; RV free wall {md['rv_wall_volume_ml']:.1f} mL"
                                             if md.get("biventricular") else ""))

    def _show_ep(self):
        ep = self.pipe.ep
        m = ep.metrics
        self._labels["ep_info"].text = (f"QRS {m['qrs_duration_ms']:.0f} ms | total activation "
                                        f"{m['total_activation_time_ms']:.0f} ms | septal->lateral "
                                        f"{m['septal_to_lateral_delay_ms']:.0f} ms | APD {m['mean_apd_ms']:.0f} ms"
                                        + (f"\nScar: core {self.pipe.scar.metrics['core_volume_ml']:.1f} mL, BZ "
                                           f"{self.pipe.scar.metrics['border_zone_volume_ml']:.1f} mL, LV burden "
                                           f"{self.pipe.scar.metrics['lv_scar_burden_pct']:.0f} %, "
                                           f"{self.pipe.scar.metrics['conduction_channels']} conduction channel(s)"
                                           if self.pipe.scar is not None else "")
                                        + (f"\nRV activated by {m['rv_total_activation_ms']:.0f} ms | LV-RV free-wall "
                                           f"delay {m['interventricular_delay_ms']:.0f} ms"
                                           if "rv_total_activation_ms" in m else ""))
        if ep.ecg:
            with self._plots_frame["ep"]:
                with ui.VStack(height=0):
                    for name, sig in ep.ecg.items():
                        ui.Label(f"pseudo-ECG {name}", height=16)
                        ui.Plot(ui.Type.LINE, float(sig.min()), float(sig.max()), *sig.astype(float).tolist(),
                                height=60, style={"color": 0xFF55FF55})

    def _show_mechanics(self):
        c = self.pipe.cycle
        m = c.metrics
        self._labels["mechanics_info"].text = (
            f"EDV {m['edv_ml']:.1f} mL | ESV {m['esv_ml']:.1f} mL | SV {m['stroke_volume_ml']:.1f} mL | "
            f"EF {m['ejection_fraction_pct']:.1f} %\nLVP max {m['peak_lv_pressure_mmhg']:.0f} mmHg | aortic "
            f"{m['peak_aortic_pressure_mmhg']:.0f}/{m['min_aortic_pressure_mmhg']:.0f} mmHg | fibre strain "
            f"{100 * m['peak_mean_fiber_strain']:.1f} %"
            + (f"\nRVEDV {m['rv_edv_ml']:.1f} mL | RVESV {m['rv_esv_ml']:.1f} mL | RVEF "
               f"{m['rv_ejection_fraction_pct']:.1f} % | RVP max {m['peak_rv_pressure_mmhg']:.0f} mmHg | PA "
               f"{m['pa_systolic_mmhg']:.0f}/{m['pa_diastolic_mmhg']:.0f} mmHg" if "rv_edv_ml" in m else ""))
        ps = getattr(self.pipe, "passive", None)
        if ps:
            d = ps["diastolic"]
            self._labels["mechanics_info"].text += (
                f"\nPassive ({ps['mode']}): stiffness x{ps['stiff_scale']:.2f}, unloaded LV {d['model_v0_ml']:.1f} mL "
                f"(Klotz {d['klotz_v0_ml']:.1f}), V30 {d['model_v30_ml']:.0f} mL, ED stiffness "
                f"{d['ed_chamber_stiffness_mmhg_per_ml']:.2f} mmHg/mL")
        with self._plots_frame["mechanics"]:
            with ui.VStack(height=0):
                ui.Label("LV pressure (mmHg) over the beat", height=16)
                ui.Plot(ui.Type.LINE, 0.0, float(c.pressure_mmhg.max()), *c.pressure_mmhg.astype(float).tolist(),
                        height=70, style={"color": 0xFF5555FF})
                ui.Label("LV volume (mL) over the beat", height=16)
                ui.Plot(ui.Type.LINE, float(c.volume_ml.min()), float(c.volume_ml.max()),
                        *c.volume_ml.astype(float).tolist(), height=70, style={"color": 0xFFFFAA44})
                if "RV" in c.chambers:
                    rv = c.chambers["RV"]
                    ui.Label("RV pressure (mmHg) over the beat", height=16)
                    ui.Plot(ui.Type.LINE, 0.0, float(rv["pressure_mmhg"].max()),
                            *rv["pressure_mmhg"].astype(float).tolist(), height=60, style={"color": 0xFFFF7755})

    def _show_surrogate(self):
        p = self.pipe
        s = p.surrogate_cycle["metrics"]
        txt = (f"{p.surrogate.backbone}: EF {s['ejection_fraction_pct']:.1f} % (FE "
               f"{p.cycle.metrics['ejection_fraction_pct']:.1f} %), POD energy {p.surrogate.pod_energy:.5f}")
        if p.calibration:
            c = p.calibration
            txt += (f"\nPersonalised: contractility x{c['contractility_scale']:.2f} "
                    f"({c['t_max_kpa_personalised']:.0f} kPa) -> EF {c['achieved_ef_pct']:.1f} %")
        self._labels["surrogate_info"].text = txt

    def _show_twin(self):
        pass

    # ------------------------------------------------------------ helpers
    def _on_progress(self, stage, frac):
        # may be called from the worker thread: applied in _on_update on the main thread
        self._pending_progress[stage] = float(np.clip(frac, 0, 1))

    def _on_update(self, _event):
        if self._pending_progress:
            pending, self._pending_progress = self._pending_progress, {}
            for stage, frac in pending.items():
                bar = self._progress.get(stage)
                if bar is not None:
                    bar.model.set_value(frac)
        if self._log_dirty:
            self._log_dirty = False
            lab = self._labels.get("log")
            if lab:
                lab.text = "\n".join(self._log_lines)

    def _set_status(self, name, text, color=0xFFCCCCCC):
        lab = self._labels.get(f"{name}_status")
        if lab:
            lab.text = text
            lab.style = {"color": color}

    def _log(self, msg):
        carb.log_info(msg)
        print(msg)
        self._log_lines = (self._log_lines + [str(msg)])[-14:]
        self._log_dirty = True

    def destroy(self):
        self._update_sub = None
        if self.window:
            self.window.destroy()
            self.window = None
