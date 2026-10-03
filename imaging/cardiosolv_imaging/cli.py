"""cardiosolv-segment: DICOM/NIfTI CT or MR -> cardiac segmentation -> heart.usda

Examples::

    cardiosolv-segment /data/patient01/CT_DICOM -o out/p01                       # auto: NV-Segment-CTMR (+TotalSegmentator if licensed)
    cardiosolv-segment /data/cine_sa -o out/p02 --modality MR --phase ed           # cine MR, end-diastole
    cardiosolv-segment ct.nii.gz -o out/p03 --engine totalsegmentator
    cardiosolv-segment ct.nii.gz -o out/p04 --label-map seg.nii.gz --mapping acdc   # your own segmentation
    cardiosolv-segment /data/cine_sa -o out/p05 --lge /data/lge_sa                  # + scar from LGE MR
    cardiosolv-segment --check                                                      # which engines can run here
"""

from __future__ import annotations

import argparse
import json
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", help="DICOM folder or NIfTI volume")
    ap.add_argument("-o", "--out", default="cardiosolv_imaging_out")
    ap.add_argument("--modality", default="auto", choices=["auto", "CT", "MR"])
    ap.add_argument("--engine", default="auto", choices=["auto", "nv-segment", "totalsegmentator", "both", "labelmap"])
    ap.add_argument("--label-map", default=None, help="existing segmentation NIfTI (labelmap engine)")
    ap.add_argument("--mapping", default=None, help="acdc | mmwhs | totalseg_heartchambers | cardiosolv | mapping.json")
    ap.add_argument("--phase", default="ed", help="cine MR phase: ed | es | <index>")
    ap.add_argument("--iso", type=float, default=1.0, help="isotropic resolution for surfaces (mm)")
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--lge", default=None, help="LGE MR (DICOM folder or NIfTI) of the same examination: scar")
    ap.add_argument("--scar-method", default="nsd", choices=["nsd", "fwhm"])
    ap.add_argument("--scar-sd", type=float, default=3.0, help="n-SD scar threshold above remote myocardium")
    ap.add_argument("--scar-labels", default=None, help="existing scar label map (1 border zone, 2 core)")
    ap.add_argument("--check", action="store_true", help="report available engines and exit")
    ap.add_argument("--json", action="store_true", help="print the full report as JSON")
    args = ap.parse_args(argv)

    if args.check:
        from .engines import available_engines

        print(json.dumps(available_engines(), indent=2))
        return 0
    if not args.input:
        ap.error("input is required")
    from .pipeline import run

    log = (lambda *a: print(*a, file=sys.stderr)) if args.json else print
    rep = run(args.input, args.out, args.modality, args.engine, args.label_map, args.mapping, args.phase,
              args.iso, args.device, log=log, lge=args.lge, scar_method=args.scar_method,
              scar_n_sd=args.scar_sd, scar_labels=args.scar_labels)
    if args.json:
        print(json.dumps(rep, default=str))
    else:
        print(json.dumps({"usd": rep["outputs"]["usd"], "heart_prim": rep["outputs"]["heart_prim"],
                          "qc": rep["qc"]["status"], "warnings": rep["qc"]["warnings"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
