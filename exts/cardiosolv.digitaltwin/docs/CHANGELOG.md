# Changelog

## 0.3.0
- Full stage-by-stage digital twin on the selected heart: anatomy discovery, geometry layer,
  computational mesh + fibres, electrophysiology, biomechanics + Windkessel, Cardio-PINN
  surrogate, painted/animated twin, exports and Echocardiology dashboard bridge.
- Geometry layer refactored so imported multi-part hearts (TotalSegmentator/VISTA-3D/artist
  assets) plug into the same pipeline: myocardium detection, endo/epi/base/septum separation on the
  user's mesh, anatomically signed long axis.

## 0.1.0
- CardioSolv Phase 3 design: `/CardioSolv` semantic USD hierarchy, canonical geometry, confidence engine.
