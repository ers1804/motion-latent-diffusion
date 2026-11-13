#!/bin/bash -l
#SBATCH --job-name=rs1_regr
#SBATCH --output=/hnvme/workspace/v103fe12-ped_gen/outputs/rs1_regr_%j.txt
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=h100
#SBATCH --export=NONE

# Deterministic ego-to-motion regressor, the ego-aware baseline (research/review_action_plan_2026-11-13.md, C2).
# Trains on the scene-level training partition, keeps the epoch with the lowest validation MSE and scores
# validation and test once (FID and a K=1 root dump each). usage: regressor.sh <seed>
SEED=$1
source /hnvme/workspace/v103fe12-ped_gen/motion-latent-diffusion/slurm/resplit/_env.sh
extract_data || exit 1
cd "$REPO"
export EGOPED_NAS=$TMPDIR
mkdir -p "$MODELS/regressor"

python research/src/regressor_baseline.py --scene --seed "$SEED" \
    --ckpt "$MODELS/regressor/rs1_regressor_s$SEED.pt" --train --eval
