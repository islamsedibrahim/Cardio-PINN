"""Headless CardioSolv run on an existing USD heart.

Isaac Sim:   ./python.sh exts/cardiosolv.digitaltwin/scripts/run_pipeline.py heart.usd /World/Heart
Plain USD:   python scripts/run_pipeline.py heart.usd /World/Heart --until ep

The selected prim's meshes are analysed, simulated and painted in place: the
results are authored into ``<stage>_cardiosolv.usda`` / ``<stage>_cardiosolv_twin.usdc``
sublayers next to the stage; ``<stage>_twin.usda`` opens the original (untouched)
with the twin on top.
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from pxr import Usd  # noqa: E402

from cardiosolv.digitaltwin.ep import PACING_PROTOCOLS  # noqa: E402
from cardiosolv.digitaltwin.pipeline import STAGES, CardioSolvPipeline, PipelineConfig  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("usd")
    ap.add_argument("prim", help="selected heart prim, e.g. /World/Heart")
    ap.add_argument("--until", default="twin", choices=STAGES)
    ap.add_argument("--element-size", type=float, default=3.0, help="mm")
    ap.add_argument("--protocol", default="sinus", choices=list(PACING_PROTOCOLS))
    ap.add_argument("--ep-solver", default="eikonal", choices=["eikonal", "monodomain"])
    ap.add_argument("--dt", type=float, default=10.0, help="mechanics time step (ms)")
    ap.add_argument("--target-ef", type=float, default=None, help="personalise contractility to this EF (%%)")
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--field", default="transmembrane_potential")
    ap.add_argument("--role", action="append", default=[], help="override, e.g. Myocardium=/World/Heart/LV_wall")
    ap.add_argument("--out", default=None)
    ap.add_argument("--end-diastolic", action="store_true",
                    help="the heart is an end-diastolic image: recover the unloaded heart and fit passive stiffness")
    ap.add_argument("--edp", type=float, default=10.0, help="LV end-diastolic pressure (mmHg)")
    ap.add_argument("--rv-edp", type=float, default=5.0, help="RV end-diastolic pressure (mmHg)")
    ap.add_argument("--measured-edv", type=float, default=None, help="measured LVEDV (mL) at --edp")
    ap.add_argument("--lv-only", action="store_true", help="do not simulate the RV")
    ap.add_argument("--crt-study", action="store_true", help="LV lead sweep for CRT (scar-aware)")
    ap.add_argument("--dashboard", default=None, help="Echocardiology dashboard URL (echo EF in, report out)")
    args = ap.parse_args(argv)

    stage = Usd.Stage.Open(args.usd)
    cfg = PipelineConfig(element_size_mm=args.element_size, mechanics_dt_ms=args.dt, target_ef_pct=args.target_ef,
                         surrogate_epochs=args.epochs, display_field=args.field, output_dir=args.out,
                         dashboard_url=args.dashboard, edp_mmhg=args.edp, rv_edp_mmhg=args.rv_edp,
                         measured_edv_ml=args.measured_edv, biventricular=not args.lv_only,
                         geometry_state="end_diastolic" if args.end_diastolic else "unloaded")
    cfg.ep.protocol = args.protocol
    cfg.ep.solver = args.ep_solver
    pipe = CardioSolvPipeline(stage, args.prim, cfg)
    if args.role:
        pipe.run_discover()
        paths = [p.source_path for p in pipe.scan.parts]
        for item in args.role:
            role, path = item.split("=", 1)
            pipe.set_role(role, paths.index(path))
    order = STAGES[: STAGES.index(args.until) + 1]
    for name in order:
        if name in ("mechanics", "surrogate") and args.until in ("discover", "geometry", "mesh", "ep"):
            continue
        if name == "discover" and "discover" in pipe.done:
            continue
        getattr(pipe, f"run_{name}")()
        if name == "ep" and args.crt_study:
            pipe.run_crt_study()
    if "twin" not in order:
        pipe.build_report()
        pipe.export()
    else:
        pipe.save_twin_stage()
    # the original stage only gained sublayer references to the CardioSolv layers
    return pipe


if __name__ == "__main__":
    main()
