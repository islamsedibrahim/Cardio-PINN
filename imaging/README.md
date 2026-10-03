# CardioSolv Imaging: Stage 0 (DICOM CT / MR → 3D heart USD)

Runs **before** the CardioSolv Isaac Sim extension. It segments the cardiovascular system
from your CT or MR DICOM, reconstructs watertight surfaces and writes `heart.usda`. The
extension then picks that file up for Stage 1 (anatomy discovery) and the rest of the twin.

```
DICOM folder (CT or cine/3D MR)
  │ series discovery + selection (localisers rejected), cine MR → end-diastole
  ▼ NIfTI (RAS, mm)                                 [NVIDIA medical-AI-skills dicom-series-to-volume approach]
  │ segmentation engines
  │   • NV-Segment-CTMR  (NVIDIA, VISTA3D arch.)   CT_BODY / MRI_BODY, cardiac label prompts
  │   • TotalSegmentator heartchambers_highres     CT, sub-mm (free academic licence)
  │   • your own label map                          3D Slicer / ACDC / MM-WHS / TotalSegmentator
  ▼ fusion → cleanup (largest components, holes) → shape-preserving isotropic resampling
  ▼ marching cubes + Taubin → watertight surfaces
  ▼ heart.usda  (/World/Patient/Heart/heart_myocardium, heart_ventricle_left, ...)  → CardioSolv Stage 1
```

| Structure (USD prim) | NV-Segment-CTMR id | CT | MR | TotalSegmentator |
|---|---|---|---|---|
| `heart_myocardium` (LV) | 154 | ✓ | ✓ cine | heartchambers_highres |
| `heart_ventricle_left` | 151 | ✓ | ✓ cine | heartchambers_highres |
| `heart_ventricle_right` | 152 | ✓ | ✓ cine | heartchambers_highres |
| `heart_atrium_left` | 149 | ✓ | ✓ | heartchambers_highres |
| `heart_atrium_right` | 153 | ✓ | – | heartchambers_highres |
| `aorta` | 6 | ✓ | ✓ | both / total(_mr) |
| `pulmonary_artery` | 171 | ✓ | – | heartchambers_highres |
| `superior_vena_cava`, `inferior_vena_cava`, `pulmonary_vein`, `atrial_appendage_left` | 125, 7, 119, 108 | ✓ | IVC | total |
| `heart` (envelope) | 115 | ✓ | ✓ | total(_mr) |

## Scar from LGE MR

Add the late-gadolinium-enhancement series of the same examination:

```bash
cardiosolv-segment /data/p02/cine_SA -o out/p02 --modality MR --lge /data/p02/LGE_SA          # n-SD (default)
cardiosolv-segment ct.nii.gz -o out/p06 --lge lge.nii.gz --scar-method fwhm
cardiosolv-segment ct.nii.gz -o out/p07 --lge lge.nii.gz --scar-labels my_scar.nii.gz         # 1 = BZ, 2 = core
```

How the scar is found:
- **Myocardium:** the anatomy's myocardium is resampled onto the LGE grid (scanner coordinates).
- **Remote myocardium:** estimated by iterative 2-SD clipping.
- **Scar:** remote mean + n·SD (default 3). Within it, the core is ≥ 50 % of the maximum scar intensity (FWHM); the rest is border zone. `fwhm` mode uses 35 % / 50 % of the maximum.
- **Cleanup:** endocardial partial-volume rims and specks under 0.1 mL are removed.
- **Output:** the result is interpolated to the isotropic grid by signed distance, clipped to the myocardium, and written to `scar_labels.nii.gz` and to `heart.usda` as `/World/Patient/Scar/{scar_core, scar_border_zone}`. The CardioSolv extension turns these into its scar substrate (EP block and slowing, contractility, conduction channels, CRT planning).

The REST service accepts the LGE as a second upload (`lge=`). On the synthetic study (8 mm LGE
slices, noise SD 10) the remote estimate is 48.9 ± 9.7 vs 50 ± 10 true, border zone 8.8 vs 8.9 mL,
core 7.8 vs 9.2 mL.

## Deploy on a GPU host

**Docker (recommended)**
```bash
cd imaging
TOTALSEG_LICENSE=aca_XXXXXXXX docker compose -f docker/docker-compose.yml up -d --build   # licence optional
curl http://GPU_HOST:8040/health
```
The image downloads NV-Segment-CTMR (bundle from `islamsedibrahim/NV-Segment-CTMR`, weights
from `huggingface.co/nvidia/NV-Segment-CTMR`) and TotalSegmentator (`islamsedibrahim/TotalSegmentator`).

**Bare metal / conda**
```bash
cd imaging
pip install ".[nvsegment,totalsegmentator,server,dicom-extra]"
scripts/setup_models.sh /opt/cardiosolv          # once; TOTALSEG_LICENSE=... to enable heartchambers_highres
source /opt/cardiosolv/env.sh
cardiosolv-segment --check
```

## Use

```bash
cardiosolv-segment /data/p01/CT_DICOM -o out/p01                        # auto (CT: TotalSegmentator if licensed + NV-Segment-CTMR)
cardiosolv-segment /data/p02/cine_SA   -o out/p02 --modality MR --phase ed
cardiosolv-segment ct.nii.gz -o out/p03 --engine both
cardiosolv-segment ct.nii.gz -o out/p04 --label-map seg.nii.gz --mapping acdc   # existing segmentation
uvicorn cardiosolv_imaging.server:app --host 0.0.0.0 --port 8040         # REST service for Isaac Sim
```
Outputs in `-o`: `heart.usda`, `heart_labels.nii.gz` (+ `.json` legend), `image.nii.gz`,
`imaging_report.json` (series chosen, engines, per-structure volume and watertightness, QC).

In Isaac Sim, open the CardioSolv panel, go to **0 Imaging** and choose **Local** (runs
`cardiosolv-segment` with the Python you configure) or **Service** (`http://GPU_HOST:8040`),
pick the DICOM folder and press **Segment & Load Heart**. The heart is referenced into the
stage at `/World/Patient/Heart` and selected for Stage 1.

## Notes and limits

* **Privacy:** patient identifiers are never copied into NIfTI, USD or reports. Run the
  service inside your hospital network; uploads stay under `CARDIOSOLV_IMAGING_WORKDIR`
  and are removed with `DELETE /jobs/{id}`.
* **Licences:** NV-Segment-CTMR weights are under the NVIDIA non-commercial licence;
  TotalSegmentator `heartchambers_highres` needs a licence (free for academic use).
* **MR:** NV-Segment-CTMR's cardiac MR classes come from cine data (CMRMotion). Thick slices
  (8–10 mm) are interpolated through signed distances; the base may be truncated. A 3D
  whole-heart MR or a CT gives the most complete anatomy.
* Enhanced multi-frame or compressed DICOM needs `SimpleITK` / `pylibjpeg` (`[dicom-extra]`).
* LGE is not registered to the anatomy: use series from the same examination (breath-hold shifts
  move the scar by a few mm). Thick LGE slices underestimate small cores.
* This is an engineering reconstruction for simulation. It is not for diagnosis.

## Tests
```bash
pip install pytest fastapi httpx && python -m pytest      # synthetic CT + cine MR DICOM, fake bundle contract, REST
```
