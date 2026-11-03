#!/bin/bash
#SBATCH --job-name=v2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/data03/home/wkim/Music2Emotion/logs/v2_%A_%a.out
#SBATCH --nodelist=cn01
# ──────────────────────────────────────────────────────────────
# rerun-v2: one row of runs/run_matrix.tsv per array task.
#   train -> test with mood-selected ckpt (Jamendo metrics)
#         -> test with VA-selected ckpt   (MERGE metrics)
# Submit:  LAMBDA=0.05 sbatch --array=1-26%2 --export=ALL runs/run_one.sh
# Diagnostic: MATRIX=runs/run_matrix_diag.tsv LAMBDA=0.05 sbatch --array=1-6%2 --export=ALL runs/run_one.sh
# Re-running a task skips training if best_checkpoint.txt already exists.
# ──────────────────────────────────────────────────────────────
set -euo pipefail
source /data03/home/wkim/conda/etc/profile.d/conda.sh
conda activate /data03/home/wkim/conda_envs/music2emo
cd /data03/home/wkim/Music2Emotion

MATRIX=${MATRIX:-runs/run_matrix.tsv}
LINE=$(awk -v n="$SLURM_ARRAY_TASK_ID" 'NR == n + 1' "$MATRIX")
[ -n "$LINE" ] || { echo "no row $SLURM_ARRAY_TASK_ID"; exit 1; }
IFS=$'\t' read -r RUN CFG SEED DATASETS LYRICS RHYME TIER ROWEXTRA <<< "$LINE"
LAMBDA=${LAMBDA:-0.05}
# Smoke testing: RUN_SUFFIX=_smoke EXTRA="epochs=1 +trainer.limit_train_batches=30 +trainer.limit_val_batches=10"
RUN="${RUN}${RUN_SUFFIX:-}"
read -r -a EXTRA_ARGS <<< "${EXTRA:-}"
echo "=== $RUN | config=$CFG seed=$SEED datasets=$DATASETS lyrics=$LYRICS rhyme_mode=$RHYME tier=$TIER lambda=$LAMBDA extra=${ROWEXTRA:-} matrix=$MATRIX ==="
echo "start $(date)"

ARGS=(datasets="$DATASETS" model.classifier=linear-mt-attn-ck
      +model.va_loss_weight="$LAMBDA" seed="$SEED" run_name="$RUN")
if [ "$LYRICS" = "1" ]; then
  ARGS+=(+model.use_lyrics=true +model.use_rhyme=true +model.shared_dim=512
         +model.lyrics_proj_dim=256 +model.lyrics_dropout=0.1 model.rhyme_mode="$RHYME")
fi
# optional 8th column: extra Hydra overrides for this row (applied to train AND test)
if [ -n "${ROWEXTRA:-}" ]; then read -r -a ROW_ARGS <<< "$ROWEXTRA"; ARGS+=("${ROW_ARGS[@]}"); fi

RUN_DIR=tb_logs/train_audio_classification/$RUN
if [ -f "$RUN_DIR/best_checkpoint.txt" ]; then
  echo "training already done for $RUN, skipping to eval"
else
  python3 train.py "${ARGS[@]}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
fi

for SEL in mood va; do
  python3 test.py "${ARGS[@]}" checkpoint_latest=True multitask=True dataset_type=$SEL
done
echo "end $(date)"
