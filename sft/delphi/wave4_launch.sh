#!/bin/bash
set -o pipefail
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
source /leonardo_work/AIFAC_5C0_290/bfeuer00/miniforge3/etc/profile.d/conda.sh
conda activate otagent
cd /leonardo_work/AIFAC_5C0_290/bfeuer00/code/OpenThoughts-Agent
source hpc/dotenv/leonardo.env
source ~/secrets.env
ROOT=/leonardo_work/AIFAC_5C0_290/bfeuer00/code/OpenThoughts-Agent
LOG=$ROOT/wave4_launch.log
JOBIDS=$ROOT/wave4_jobids.txt
: > $JOBIDS
echo "=== WAVE4 LAUNCH START $(date) ===" | tee -a $LOG

HEAD_DEP="afterany:45653808:45653836:45653914:45653964:45653980:45653983:45654028:45654029:45654076:45654079:45654080:45654132:45654147:45654148:45654411:45654412:45654413"

RUNIDS=(
  "delphi-9e19-p33m67-k0p20-lr83-a002"
  "delphi-9e19-p50m50-k0p20-lr83-a002"
  "delphi-9e19-p67m33-k0p20-lr83-a002"
  "delphi-2e20-p33m67-k0p20-lr83-a001"
)

PREV=""

launch_one () {
  local RID="$1" INSTR="$2" LABEL="$3"
  local BASE="laion/$RID"
  local HUBID="laion/${RID}-${LABEL}-sft"
  local JOBNAME="${RID}-${LABEL}"
  if [ -n "$PREV" ]; then DEP="afterany:$PREV"; else DEP="$HEAD_DEP"; fi
  echo "=== RUN base=$BASE instr=$INSTR label=$LABEL hub=$HUBID dep=$DEP ===" | tee -a $LOG

  OUT=$(DISABLE_VERSION_CHECK=1 python -m hpc.launch \
    --train_config_path sft/lf_configs/delphi/4k_sft.yaml \
    --time_limit 23:59:00 \
    --num_nodes 4 --gpus_per_node 4 \
    --model_path "$BASE" \
    --dataset_dir sft/delphi \
    --dataset "${INSTR},delphi_warmup" \
    --mix_strategy interleave_under --interleave_probs 0.9,0.1 \
    --learning_rate 1.0e-5 \
    --hub_model_id "$HUBID" \
    --job_name "$JOBNAME" \
    --push_to_hub true \
    --overwrite_output_dir true \
    --internet_node \
    --dependency "$DEP" 2>&1)
  echo "$OUT" | tee -a $LOG

  # extract submitted jobid from launcher output (Submitted batch job N / Writing logs ... _N.out)
  JID=$(echo "$OUT" | grep -oE "Submitted batch job [0-9]+" | grep -oE "[0-9]+" | tail -1)
  if [ -z "$JID" ]; then JID=$(echo "$OUT" | grep -oE "_[0-9]+\.out" | grep -oE "[0-9]+" | tail -1); fi
  if [ -z "$JID" ]; then echo "ABORT: no jobid parsed for $JOBNAME" | tee -a $LOG; return 1; fi

  # §5 sanity-check on the generated real sbatch (expect conda activate otagent + DCFT-based WORKDIR, no PWD/No-conda markers)
  SB=$(ls -t $ROOT/experiments/*${JOBNAME}*/sbatch/*_sft.sbatch 2>/dev/null | grep -v dryrun | head -1)
  echo "--- §5 sbatch check ($SB) ---" | tee -a $LOG
  grep -nE "conda activate otagent|No conda activation|WORKDIR=\"\\\$PWD\"" "$SB" 2>&1 | tee -a $LOG

  echo "${JID}|${HUBID}|dep=${DEP}|instr=${INSTR}|base=${BASE}" | tee -a $JOBIDS
  PREV="$JID"
}

for RID in "${RUNIDS[@]}"; do
  launch_one "$RID" "magpie"        "magpie_lr1e5" || { echo "STOP (magpie $RID)" | tee -a $LOG; break; }
  launch_one "$RID" "wildchat_386k" "wc386k_lr1e5" || { echo "STOP (wc386k $RID)" | tee -a $LOG; break; }
done
echo "=== WAVE4 LAUNCH DONE $(date) ===" | tee -a $LOG
echo "=== JOBIDS ==="; cat $JOBIDS
