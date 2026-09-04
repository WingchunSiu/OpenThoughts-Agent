#!/bin/bash
set -o pipefail
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
source /leonardo_work/AIFAC_5C0_290/bfeuer00/miniforge3/etc/profile.d/conda.sh
conda activate otagent
cd /leonardo_work/AIFAC_5C0_290/bfeuer00/code/OpenThoughts-Agent
source hpc/dotenv/leonardo.env
source ~/secrets.env
export HF_HUB_ENABLE_HF_TRANSFER=1
LOG=/leonardo_work/AIFAC_5C0_290/bfeuer00/code/OpenThoughts-Agent/wave4_prep.log
echo "=== START $(date) ===" | tee -a $LOG

# checkpoint #13 already cached (skip dl); #14,#15,#16 need download
DL_REPOS=(
  "laion/delphi-9e19-p50m50-k0p20-lr83-a002"
  "laion/delphi-9e19-p67m33-k0p20-lr83-a002"
  "laion/delphi-2e20-p33m67-k0p20-lr83-a001"
)
for R in "${DL_REPOS[@]}"; do
  echo "=== DOWNLOAD $R $(date) ===" | tee -a $LOG
  for attempt in 1 2 3 4 5; do
    hf download "$R" --repo-type model >>$LOG 2>&1 && { echo "DL OK $R" | tee -a $LOG; break; }
    echo "DL retry $attempt failed for $R, sleeping 20s" | tee -a $LOG; sleep 20
  done
done

ALL_RUNIDS=(
  "delphi-9e19-p33m67-k0p20-lr83-a002"
  "delphi-9e19-p50m50-k0p20-lr83-a002"
  "delphi-9e19-p67m33-k0p20-lr83-a002"
  "delphi-2e20-p33m67-k0p20-lr83-a001"
)
for RID in "${ALL_RUNIDS[@]}"; do
  OUT=/leonardo_work/AIFAC_5C0_290/bfeuer00/data/delphi-tok-prep/$RID
  if [ -f "$OUT/model.safetensors.index.json" ] || ls "$OUT"/model*.safetensors >/dev/null 2>&1; then
    echo "TOKPREP SKIP (exists) $RID" | tee -a $LOG; continue
  fi
  SNAP=$(ls -d /leonardo_work/AIFAC_5C0_290/bfeuer00/data/hub/models--laion--$RID/snapshots/*/ 2>/dev/null | head -1)
  if [ -z "$SNAP" ]; then echo "TOKPREP FAIL: no snapshot for $RID" | tee -a $LOG; continue; fi
  echo "=== TOKPREP $RID  (snap=$SNAP) $(date) ===" | tee -a $LOG
  python sft/delphi/prepare_delphi_tokenizer.py --model "$SNAP" --output "$OUT" >>$LOG 2>&1 \
    && echo "TOKPREP OK $RID" | tee -a $LOG \
    || echo "TOKPREP FAIL $RID" | tee -a $LOG
done
echo "=== DONE $(date) ===" | tee -a $LOG
