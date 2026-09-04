#!/bin/bash
# WAVE 1 Delphi SFT chain launcher: 4 ckpts x {magpie, wc386k} = 8 runs.
# Internal afterany chain (one active at a time). Job1 blind.
set -o pipefail
source /leonardo_work/AIFAC_5C0_290/bfeuer00/miniforge3/etc/profile.d/conda.sh
conda activate otagent
cd /leonardo_work/AIFAC_5C0_290/bfeuer00/code/OpenThoughts-Agent
source hpc/dotenv/leonardo.env
source ~/secrets.env 2>/dev/null
export DISABLE_VERSION_CHECK=1

CKPTS=(
  delphi-3e18-p33m67-k0p20-lr83-a003
  delphi-3e18-p50m50-k0p20-lr83-a003
  delphi-3e18-p67m33-k0p20-lr83-a003
  delphi-9e18-p33m67-k0p20-lr83-a002
)
# chain order: ckpt1-magpie, ckpt1-wc386k, ckpt2-magpie, ckpt2-wc386k, ...
declare -a RECIPES=(magpie:magpie wc386k:wildchat_386k)

JOBFILE=/leonardo_work/AIFAC_5C0_290/bfeuer00/wave1_jobids.txt
: > $JOBFILE
PREV=""
POS=0
for ck in "${CKPTS[@]}"; do
  for rc in "${RECIPES[@]}"; do
    tag="${rc%%:*}"        # magpie | wc386k
    instr="${rc##*:}"      # magpie | wildchat_386k
    POS=$((POS+1))
    if [ "$tag" = "magpie" ]; then suffix="magpie_lr1e5"; else suffix="wc386k_lr1e5"; fi
    HUBID="laion/${ck}-${suffix}-sft"
    echo "===== POS $POS  ckpt=$ck  recipe=$tag  instr=$instr  hub=$HUBID ====="

    # 1. generate canonical sbatch (no submit)
    OUT=$(python gen_only_delphi.py \
      --train_config_path sft/lf_configs/delphi/4k_sft.yaml \
      --time_limit 23:59:00 \
      --num_nodes 4 --gpus_per_node 4 \
      --model_path laion/${ck} \
      --dataset_dir sft/delphi \
      --dataset ${instr},delphi_warmup \
      --mix_strategy interleave_under --interleave_probs 0.9,0.1 \
      --learning_rate 1.0e-5 \
      --hub_model_id ${HUBID} \
      --overwrite_output_dir true 2>&1)
    SB=$(echo "$OUT" | grep -oE "GENERATED_SBATCH=.*" | head -1 | sed "s/GENERATED_SBATCH=//")
    if [ -z "$SB" ] || [ ! -f "$SB" ]; then
      echo "FATAL: no sbatch generated for $ck/$tag"; echo "$OUT" | tail -25; exit 1
    fi
    echo "  sbatch: $SB"

    # 2. MANDATORY §5 post-patch: fix doubled \$DCFT//leonardo_work path (step 4)
    sed -i "s|\$DCFT//leonardo_work/AIFAC_5C0_290/bfeuer00/code/OpenThoughts-Agent/experiments|/leonardo_work/AIFAC_5C0_290/bfeuer00/code/OpenThoughts-Agent/experiments|g" "$SB"
    # verify patch
    if grep -q "DCFT//leonardo_work" "$SB"; then echo "FATAL: doubled-path patch failed"; exit 1; fi
    if ! grep -q "conda activate otagent" "$SB"; then echo "FATAL: no conda activate otagent"; exit 1; fi
    echo "  PATCH OK (no DCFT// doubling; conda activate otagent present)"

    # 3. submit with afterany dependency on previous
    if [ -z "$PREV" ]; then
      JID=$(sbatch --parsable "$SB")
      DEP="(blind/wave1-head)"
    else
      JID=$(sbatch --parsable --dependency=afterany:${PREV} "$SB")
      DEP="afterany:${PREV}"
    fi
    if [ -z "$JID" ]; then echo "FATAL: sbatch returned no jobid"; exit 1; fi
    echo "  SUBMITTED jobid=$JID  dep=$DEP"
    echo "${POS}|${ck}|${tag}|${HUBID}|${JID}|${DEP}" >> $JOBFILE
    PREV=$JID
  done
done
echo "===== WAVE1 CHAIN COMPLETE ====="
cat $JOBFILE
