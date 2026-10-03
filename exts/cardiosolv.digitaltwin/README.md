# CardioSolv Digital Twin — Isaac Sim 6 extension

CardioSolv turns **the heart you select in the stage** into a biomechanical and
electrophysiological digital twin, and paints/animates the results **on your own
mesh**. It never generates a replacement heart: your geometry is the source of
truth, and every CardioSolv opinion lives in two sublayers that can be muted or
deleted at any time.

```
DICOM CT / MR  ── 0 Imaging ───────── NV-Segment-CTMR / TotalSegmentator -> heart.usda (imaging/ package)
   │
Selected heart prim (your USD)
   │  1 Anatomy discovery ── parts in world mm, explainable role scores (names + geometry)
   │  2 Geometry layer ───── myocardium · endocardium / epicardium / base / RV-septum · long axis · landmarks
   │  3 Mesh + fibres ────── conforming tets snapped to your surface · x_t, x_l, x_c · helix/sheet fibres
   │  4 Electrophysiology ── eikonal or Mitchell–Schaeffer monodomain · sinus / LBBB / RV pacing / CRT · pseudo-ECG
   │  5 Biomechanics ─────── Holzapfel–Ogden + EP-driven active tension + 3-element Windkessel heartbeat
   │  6 Cardio-PINN ──────── parametric PINN surrogate (PhysicsNeMo backbone if installed) · EF personalisation
   ▼  7 Twin on your mesh ── time-sampled points + colours on your prims · fibres · VTK / openCARP / JSON report
```

![CardioSolv results on the bundled test heart](images/cardiosolv_results.png)

## Stage 0: from DICOM

Install the segmentation package from `imaging/` on a GPU machine. Either install it locally
with `pip install ".[nvsegment,totalsegmentator]"` and run `scripts/setup_models.sh`, or start
it as a Docker service (see `imaging/README.md`). Then in the panel's **0 Imaging** section:

1. Enter the DICOM folder, or a NIfTI file.
2. Choose **local** (set *imaging Python* to that environment's interpreter) or **service**
   (set *Service URL*, e.g. `http://gpu-host:8040`).
3. Click **Segment & Load Heart**.

The heart is referenced at `/World/CardioSolvPatient/Patient/Heart` and selected for Stage 1.
Its parts are named `heart_myocardium`, `heart_ventricle_left`, ... so they are recognised with
HIGH confidence.

## Install (Isaac Sim 6)

The zip contains one folder, `cardiosolv.digitaltwin/`. Isaac Sim finds an extension only if its
**search path is the folder that contains** `cardiosolv.digitaltwin/`, not that folder itself.

**A. Extension Manager**
1. Unzip, e.g. to `C:/omni/exts/` (Windows) or `~/omni/exts/` (Linux), so that
   `C:/omni/exts/cardiosolv.digitaltwin/config/extension.toml` exists. If your unzip tool created
   `cardiosolv.digitaltwin-0.6.0/cardiosolv.digitaltwin/`, use the outer folder as the search path.
2. *Window → Extensions*, then **☰ (top-left) → Settings → Extension Search Paths → +**, and add
   `C:/omni/exts`.
3. Search **CardioSolv**. If it doesn't appear, clear the filters (*All*, not *Featured* or
   *Enabled*). Toggle **ENABLED** and tick **AUTOLOAD**.
4. The **CardioSolv Digital Twin** window docks on the right. A *CardioSolv* menu also appears in
   the menu bar (and *Window → CardioSolv Digital Twin*) to reopen it.

**B. Script Editor (no settings)**: open `scripts/open_cardiosolv.py`, set `EXT_FOLDER` to the
folder from step 1, paste it into *Window → Script Editor* and run it.

**C. Command line**: `isaac-sim.sh --ext-folder ~/omni/exts --enable cardiosolv.digitaltwin`
(on Windows, `isaac-sim.bat`).

Nothing appears? Open *Window → Console* and filter for `CardioSolv`. A load error is printed there.

Requirements: `numpy`, `scipy`, `torch` (all bundled with Isaac Sim). Optional: `physicsnemo`,
`gmsh`, `vmtk`, openCARP.

## Use it: entering your patient's data

The **Patient data (your inputs)** section at the top of the panel holds everything about the patient:

| Field | Used for |
|---|---|
| CT / MR DICOM folder, NIfTI or `heart.usda` (*Browse*) | Stage 0 segmentation (*Segment & Load Heart*) or loading an existing heart (*Load heart.usda*) |
| LGE MR folder / NIfTI (optional) | scar core / border zone → arrhythmia substrate, CRT planning |
| Heart rate (bpm) | cycle length |
| Aortic diastolic pressure | systemic Windkessel |
| LV / RV end-diastolic pressure | passive filling, end-diastolic unloading |
| Measured LVEDV | passive stiffness fit (with *Imaged geometry is* = `unloaded`) |
| Measured LVEF | contractility personalisation (surrogate) |
| Imaged geometry is | `end_diastolic` for a CT / cine-ED heart: unloaded reference + Klotz stiffness |
| Rhythm / pacing | sinus, LBBB, RV pacing, CRT |

Then press **Run All Stages with this data**, or run the stages one by one below. If your heart is
already in the stage, select its root prim and press **Use Selected Heart**. The same values also
appear in the stage sections and stay in sync.

Stage by stage:
1. **Stage 1** lists every mesh part with the role CardioSolv assigned
   (`Myocardium`, `LeftVentricle`, `RightVentricle`, `LeftAtrium`, `RightAtrium`, `Aorta`, …),
   its score and confidence class. Use the combo boxes to correct any role; later stages re-run.
2. **Stage 2** writes `/CardioSolv` and adds `cardiosolv_Endocardium / _Epicardium / _Base / _RVSeptum`
   GeomSubsets on your myocardium mesh.
3. **Stages 3–6** compute off the UI thread; *CRT lead study* is in the EP section.
4. **Stage 7** animates your heart and paints the chosen field (activation, Vm, active tension,
   fibre strain / stress, displacement, scar). Press *Play*.

Headless (Isaac Sim `python.sh`, or any Python with `usd-core`):

```bash
python scripts/run_pipeline.py heart.usd /World/Heart --protocol lbbb --target-ef 45 \
    --end-diastolic --edp 14 --rv-edp 7 --crt-study --dashboard http://localhost:3000
```

## How the geometry layer works on an imported multi-part heart

* **Units / transforms** — every `UsdGeom.Mesh` under the selected prim (instance proxies and
  face `GeomSubset` parts included) is triangulated and transformed to world millimetres using
  `metersPerUnit`.
* **Finding the myocardium** — parts are voxelised (3-axis parity vote: tolerant of holes and
  non-manifold edges). A myocardial wall is recognised by *hollowness*: rays cast from its empty
  interior hit the wall from most directions (enclosed cavity), and a blood-pool part sits inside
  that cavity. Names (TotalSegmentator, VISTA-3D, MM-WHS, artist conventions) add evidence; anonymous
  scenes (`Mesh_001…`) are scored on geometry only and never reported as HIGH confidence.
  Without a myocardium part the LV wall is derived from the LV blood pool (flagged LOW).
* **Endocardium vs epicardium** — each face of *your* myocardium is classified by the tissue on
  its outward side: LV blood pool → endocardium, RV blood pool → RV septum, atria/great vessels or
  the flat basal cut → base, otherwise epicardium; then majority-smoothed over neighbours.
  Validated on Buoso's `Shape_model/LV_mean.vtk` (single anonymous prim, no blood pool):
  **100 % of endocardial and epicardial ground-truth vertices** classified correctly.
* **Long axis** — PCA axis whose apex/base sign is resolved by explicit votes (atria/great vessels,
  cavity opening vs apical cap, base wider than apex), then refined as apex → centre of the basal
  cavity opening. Matches the ground-truth apex→mitral axis of `LV_mean` (cos > 0.98).

> Note: `LV_mean.vtk` stores the **endocardium as label 2** and the epicardium as label 1 (label 2
> is the smaller inner shell, and `LoadModelAnatomy` takes the pressure surface from label 2). The
> Cardio-PINN README lists them the other way round.

## Skin-only hearts and non-anatomical scale

Many Omniverse heart assets (artist / generated / SimReady, e.g. `isaac_human_heart.usd`) are a single
closed **outer skin**: rays through them cross the surface twice and the enclosed volume is solid, so
there is no myocardium, chamber or septum to find. CardioSolv detects this (no wall-like part: local
thickness > 25 mm at anatomical scale) and switches to **skin mode** (`skin_mode = auto | on | off`):

* the skin is the epicardium; the base is the end whose cross-sections split into several great
  vessels, the apex the single tapering tip (stage up-axis as a weak prior);
* the AV plane is the waist of the cross-section area profile (fallback 62 % of the heart length);
* the LV lies on the side the apex is offset to; a septal plane gives the LV ~62 % of the width;
* walls use reference thicknesses (LV 10 mm, septum 10 mm, RV 4 mm);
* the epicardial wall of the computational mesh is snapped to your skin, and your skin gets
  `cardiosolv_LVEpicardium / _RVFreeWall / _AtriaGreatVessels` GeomSubsets;
* results are painted on your skin (fading to grey over the RV free wall / atria, which follow the
  motion), and the derived endocardium is shown as `/CardioSolv/Debug/AssumedEndocardium`.

Everything derived is flagged **ASSUMED** and validation reports `REQUIRES ANATOMICAL VALIDATION`.
Flip the apex/base or LV/RV sides from the panel if the asset is unusual.

**Auto-scale**: assets outside 60–200 mm (the example heart is 1.1 m tall at `metersPerUnit = 1`)
are simulated in an anatomical frame (heart length 120 mm) and mapped back at the displayed size.

## Biventricular twin

By default (`biventricular = True`) the twin simulates both ventricles:

* **RV free wall**: taken from an RV myocardium part when the heart has one, otherwise derived as a
  `rv_wall_mm` (3.5 mm) shell around your RV blood pool, kept only where it is connected to the LV wall
  (flagged in the geometry warnings). Skin mode uses its assumed 4 mm RV wall.
* **Mesh**: one conforming tet mesh, region 0 = LV wall + septum, region 1 = RV free wall; RV
  endocardium = the `RVSeptum` surface class (septal + free-wall RV side). Your LV myocardium
  surface is snapped as before, the derived RV wall is smoothed.
* **Fibres**: two transmural fields (LV endo 0 -> epi / RV endo 1 for the LV and septum, RV endo 0 ->
  epi 1 for the RV free wall), each with its own helix rotation.
* **EP**: separate LV and RV Purkinje layers. Sinus uses both; LBBB keeps the right bundle (RV
  activated early, LV free wall late); RV pacing and CRT conduct retrogradely through the right bundle.
  Reports `rv_total_activation_ms` and `interventricular_delay_ms`.
* **Mechanics**: one energy with an LV and an RV cavity term, so the septum is loaded by both
  pressures (ventricular interdependence comes out of the mechanics). Each ventricle has its own valve
  state machine; the LV ejects into the systemic and the RV into a pulmonary 3-element Windkessel
  (`pvr`, `c_pulmonary`, `p_pulmonary_diastolic`, `rv_edp_mmhg`).
* **Surrogate**: inputs `(p_LV, p_RV, t, s)`; the surrogate beat solves both ventricles by Gauss–Seidel.
* **Outputs**: RVEDV/RVESV/RVEF, RV peak pressure, PA systolic/diastolic, RV/LV EDV ratio, RV PV loop
  in the report and the dashboard; the RV parts of your heart are painted too.

Set `biventricular = False` (panel: *Biventricular*) for the LV-only twin of earlier versions.

## Scar, arrhythmia substrate and CRT

**Scar input.** Closed meshes named like the imaging package's LGE output are recognised and kept out
of anatomy discovery: `scar_core` / `myocardial_scar` / `infarct` (dense scar) and
`scar_border_zone` / `grey_zone` (border zone). They can sit under the heart or next to it
(`/World/Patient/Scar/...`). Any closed mesh works, so you can draw a scar in Omniverse.
Stage 0 with an LGE series produces them automatically.

Every tet is labelled healthy / border zone / core:

| | EP | Mechanics |
|---|---|---|
| Dense core | unexcitable, blocks conduction, no capture of a lead placed in it | no active tension, passive stiffness ×5 |
| Border zone | CV ×0.4, APD ×1.15, no Purkinje | active tension ×0.5, stiffness ×2 |

**Substrate metrics.** Core and border-zone volume, LV scar burden, transmurality, distribution
(basal / mid / apical, septal / lateral), and **conduction channels**: border-zone corridors through
the core that open into healthy tissue at two places and are a real shortcut (removing them makes the
path between the openings ≥ 1.5× and ≥ 10 mm longer). Each channel reports its length, the
activation transit time and apparent conduction velocity: the isthmus candidates of scar-related VT.
Paint the `scar` field to see core (white) and border zone (amber) on your mesh.

**CRT lead study** (*CRT lead study* button, or `pipeline.run_crt_study(fe_beat=False)`). Starting from
LBBB, it sweeps 36 LV epicardial lead sites with the RV septal lead (≈ 5 s, eikonal). Sites over dense
scar do not capture, and sites over scar are avoided when another exists. It reports the best site,
LV activation time and QRS shortening, and a predicted response with reasons: scar burden > 33 %,
lead over scar, transmural lateral scar. With *FE beats* it also simulates LBBB vs best-site CRT and
reports the change in LV dP/dt max (≥ 10 % = acute responder), EF and stroke work.

## Passive personalisation (end-diastolic images)

Hearts reconstructed by Stage 0 are usually end-diastolic images (CT, cine ED phase), so they are
already inflated. You declare it: set *Imaged geometry is* = `end_diastolic` (`geometry_state`) and
enter the patient's LV and RV end-diastolic pressures (and, if measured, the LVEDV). The extension then:

1. recovers the **unloaded reference** with the backward-displacement fixed point of Sellier (2011):
   inflating the reference to LV/RV EDP reproduces your image;
2. fits the **passive stiffness** so the unloaded LV volume matches Klotz' prediction
   `V0 = EDV (0.6 − 0.006 EDP)`, i.e. the imaged EDV lies on the Klotz EDPVR;
3. reports the model EDPVR next to the Klotz curve, V0, V30 and the end-diastolic chamber stiffness.

`geometry_state = auto` infers `end_diastolic` for hearts loaded by Stage 0. The default is
`unloaded`, which simulates the mesh as stress-free, as in earlier versions. With an unloaded geometry
(artist assets) and a measured LVEDV at EDP (`measured_edv_ml`) the
stiffness is fitted to that point instead (`passive_calibration = measured`). The heartbeat then
starts from the unloaded heart. Displacements stay relative to your imaged mesh, so the twin is
painted and animated on your model and passes through it at end-diastole.

On the synthetic heart (taken as an ED image at 10 mmHg) the model EDPVR matches Klotz to
0.6 mL RMS from 0 to 30 mmHg; the unloaded LV is 28.9 mL vs Klotz' 28.9 mL, and the beat refills it to
53.2 mL at 10 mmHg (image 53.4 mL).

**Re-personalise contractility afterwards.** A fitted (usually softer) myocardium with the default
peak active tension ejects more. Enter the patient's EF (*Target EF*, `target_ef_pct`, or the echo
LVEF from the dashboard) so the surrogate re-scales contractility to it.

## Physics

| Stage | Model | Notes |
|---|---|---|
| Fibres | Cardio-PINN `GenerateFibers` (vectorised) | helix +60° endo → −60° epi, sheet γ = −65° |
| EP | anisotropic eikonal (fast endocardial layer ≈ Purkinje) or Mitchell–Schaeffer monodomain | sinus QRS ≈ 70 ms, LBBB ≈ 170 ms, CRT resynchronises |
| Passive | Holzapfel–Ogden, Sack et al. 2018 parameters (as Cardio-PINN), stiffness scale 0.75 | isochoric invariants + volumetric penalty |
| Active | `T_a(x,t) = T_max · twitch(t − t_act(x))` along fibres | EP → mechanics coupling per element |
| Circulation | 3-element Windkessel, implicit; isovolumetric phases by augmented Lagrangian | one energy minimisation per step |
| Surrogate | Cardio-PINN parametric PINN: `(p, t, s) → POD amplitudes`, energy loss + FE anchors | PhysicsNeMo `FullyConnected` when available |

On the bundled synthetic heart (biventricular, 4 mm elements): LV EDV 78 mL, ESV 30 mL, **EF 62 %**,
LV peak 131 mmHg, aortic 131/68 mmHg; RV EDV 94 mL, ESV 43 mL, **RVEF 55 %**, RV peak 28 mmHg,
PA 28/6 mmHg; LV/RV stroke volume 49/51 mL; myocardial volume preserved to 0.04 %. The surrogate
gives EF 64 % and RVEF 55 %. Its EDV is the same at every contractility, because the network sees
the activation level s·τ(t), not s. EF is 34 / 64 / 68 % at 0.6 / 1.0 / 1.4 × contractility.

**Transverse active stress.** Cardio-PINN adds `0.3·((I4s−1)+(I4n−1))` to the active energy.
With isochoric invariants this term behaves like an isotropic active stiffening that opposes wall
thickening, and on the same anatomy it lowers EF from 62 % to 40 %. The default here is therefore
`eta_transverse = 0`; set it to 0.3 to reproduce Cardio-PINN's formulation.

## Outputs

* `<stage>_cardiosolv.usda`: semantic layer, GeomSubsets, landmarks, axis.
* `<stage>_cardiosolv_twin.usdc`: time-sampled points/colours/primvars on your prims, fibres.
* `<stage>_twin.usda` (*Save Twin Stage* / CLI): a thin stage whose sublayers are
  `[twin results, semantics, your original file]`. Open it to replay the twin; your file is untouched.

USD layering note: sublayers are weaker than the root layer. When the heart is referenced into
the scene (the usual Isaac Sim case) the CardioSolv sublayers override it directly. When the meshes
are defined in the opened file itself, the live animation is authored in the session layer, and
*Save Twin Stage* moves it into the twin layer underneath the thin stage.
* `cardiosolv_output/cardiosolv_report.json`: anatomy, QC, EP, PV loop, surrogate, calibration.
* `cardiosolv_output/opencarp/`: `.pts/.elem/.lon`, stimulus `.vtx`, `.par` template.
* `cardiosolv_output/vtk/`: ParaView series (displacement, Vm, active tension, fibre strain/stress).

## Tests

```bash
cd exts/cardiosolv.digitaltwin && pip install numpy scipy usd-core torch scikit-image vtk pytest
python -m pytest
```

## Limitations (research prototype — not for clinical use)

* Quasi-static mechanics; lumped (0D) haemodynamics instead of 3D FSI; atria are not simulated
  (prescribed filling pressures), so a single beat need not balance LV and RV stroke volumes.
* A derived RV wall is a uniform 3.5 mm shell; at 3–4 mm elements it is one or two elements thick, so
  RV wall stresses are coarse. Use an RV myocardium segmentation and smaller elements for RV studies.
* P1 tetrahedra with a penalty volumetric term; refine elements for stress quantities.
* Monodomain conduction velocity needs ≤ 1 mm elements to converge; the eikonal solver is the
  interactive default.
* The surrogate matches FE ejection fraction within ~2 points but can show pressure overshoots in early
  ejection; use the FE beat for pressure waveforms.
* Scar from LGE is not registered to the anatomy (same examination assumed); thick LGE slices
  underestimate small cores. Channel detection needs elements ≤ 3 mm. Scar EP/mechanics factors are
  literature values, not patient-calibrated. VT inducibility (programmed stimulation with a
  monodomain model) is not simulated; channels are candidate isthmuses.
* Passive calibration fits one stiffness scale (fibre architecture and anisotropy ratios fixed) and
  takes ~5 min on CPU at 4 mm elements. EF above ~65 % may be out of the calibratable range (reported as such).
* If your heart asset is already animated (skinning/blend shapes), the twin's time samples override
  its points while the twin layer is active.
