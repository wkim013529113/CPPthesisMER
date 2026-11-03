#!/bin/bash
#SBATCH --job-name=lambda_check
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/data03/home/wkim/Music2Emotion/logs/lambda_check_%j.out
#SBATCH --nodelist=cn01
# ──────────────────────────────────────────────────────────────
# Job 0: loss-balance check without GlobalMood_VA.
# C2 config (audio multitask), seed 0, 3 full epochs, lambda = 0.05.
# Uses training losses only (no test data). Decision rule in README.
# ──────────────────────────────────────────────────────────────
source /data03/home/wkim/conda/etc/profile.d/conda.sh
conda activate /data03/home/wkim/conda_envs/music2emo
cd /data03/home/wkim/Music2Emotion

python3 train.py \
    datasets=[jamendo,globalmood,merge] model.classifier=linear-mt-attn-ck \
    +model.va_loss_weight=0.05 seed=0 run_name=lambda_check_C2 epochs=3

python3 - <<'PY'
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
ea = EventAccumulator(glob.glob("tb_logs/train_audio_classification/lambda_check_C2/events*")[0]); ea.Reload()
for k in ["train_loss_mood", "train_loss_globalmood", "train_loss_va", "val_loss_mood", "val_loss_globalmood", "val_loss_va"]:
    if k in ea.Tags()["scalars"]:
        print(k, [round(e.value, 4) for e in ea.Scalars(k)])
PY
