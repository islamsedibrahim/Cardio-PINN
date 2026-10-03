#!/usr/bin/env bash
# One-time setup of the segmentation engines (GPU host).
#   NV-Segment-CTMR : bundle from islamsedibrahim/NV-Segment-CTMR + weights from huggingface.co/nvidia/NV-Segment-CTMR
#   TotalSegmentator: islamsedibrahim/TotalSegmentator (heartchambers_highres needs a free academic licence)
# Usage: scripts/setup_models.sh [INSTALL_DIR]   (default /opt/cardiosolv)
set -euo pipefail
DIR="${1:-${CARDIOSOLV_MODELS_DIR:-/opt/cardiosolv}}"
NV_REPO="${NV_SEGMENT_CTMR_REPO:-https://github.com/islamsedibrahim/NV-Segment-CTMR.git}"
TS_REPO="${TOTALSEG_REPO:-https://github.com/islamsedibrahim/TotalSegmentator.git}"
HF_REVISION="${NV_SEGMENT_CTMR_HF_REVISION:-4fb8b4a6b2532be9f1c449a3726fe5440ab4213a}"
mkdir -p "$DIR"

echo "== NV-Segment-CTMR bundle"
if [ ! -d "$DIR/NV-Segment-CTMR/.git" ]; then
  git clone --depth 1 "$NV_REPO" "$DIR/NV-Segment-CTMR"
fi
BUNDLE="$DIR/NV-Segment-CTMR/NV-Segment-CTMR"
mkdir -p "$BUNDLE/models"
if [ ! -e "$BUNDLE/models/model.pt" ]; then
  python -m pip install -q "huggingface_hub[cli]"
  hf download nvidia/NV-Segment-CTMR --revision "$HF_REVISION" \
     --include "vista3d_pretrained_model/*" --local-dir "$BUNDLE/models/"
  mv "$BUNDLE/models/vista3d_pretrained_model/model.pt" "$BUNDLE/models/model.pt"
fi
echo "export NV_SEGMENT_CTMR_ROOT=$BUNDLE" > "$DIR/env.sh"

echo "== TotalSegmentator"
python -m pip install -q "git+$TS_REPO"
if [ -n "${TOTALSEG_LICENSE:-}" ]; then
  totalseg_set_license -l "$TOTALSEG_LICENSE"
  echo "export TOTALSEG_LICENSE=$TOTALSEG_LICENSE" >> "$DIR/env.sh"
fi
# pre-fetch the CT/MR weights used by CardioSolv so the first study is not slow
python - <<'PY'
from totalsegmentator.libs import download_pretrained_weights
for task_id in (291, 292, 293, 294, 295, 297, 298, 730, 731, 732, 733):  # total CT (+crop) and total_mr (+crop)
    try:
        download_pretrained_weights(task_id)
    except Exception as exc:
        print("skip", task_id, exc)
PY
echo "Done. Run:  source $DIR/env.sh && cardiosolv-segment --check"
