# Changelog

## 0.5.0
- Stage 0 imaging: DICOM CT / MR -> cardiovascular segmentation (NV-Segment-CTMR, TotalSegmentator
  heartchambers_highres, or an existing label map) -> watertight heart USD, loaded by reference
  into the stage (units and up-axis converted) and selected for Stage 1. Local subprocess or
  REST service on a GPU host (`imaging/` deployable package).

## 0.4.0
- Skin mode for skin-only hearts (visual/generated/SimReady assets): assumed LV/RV/septum under the
  user's surface, epicardium snapped to the skin, GeomSubsets and painting on the skin.
- Auto-scale for non-anatomical assets (simulation frame, results mapped back).
- Wall-thickness evidence in the anatomy engine (solid bodies are not mistaken for a myocardium).
- Thin `<stage>_twin.usda` export keeps the original file untouched.

## 0.3.0
- Full stage-by-stage digital twin on the selected heart: anatomy discovery, geometry layer,
  computational mesh + fibres, electrophysiology, biomechanics + Windkessel, Cardio-PINN
  surrogate, painted/animated twin, exports and Echocardiology dashboard bridge.
- Geometry layer refactored so imported multi-part hearts (TotalSegmentator/VISTA-3D/artist
  assets) plug into the same pipeline: myocardium detection, endo/epi/base/septum separation on the
  user's mesh, anatomically signed long axis.

## 0.1.0
- CardioSolv Phase 3 design: `/CardioSolv` semantic USD hierarchy, canonical geometry, confidence engine.
