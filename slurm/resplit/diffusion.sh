#!/bin/bash -l
#SBATCH --job-name=rs1_diff
#SBATCH --output=/hnvme/workspace/v103fe12-ped_gen/outputs/rs1_diff_%j.txt
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=h100
#SBATCH --export=NONE

# usage: diffusion.sh <name> <config> <encoder name> <seed> <True|False interaction> [extra overrides]
NAME=$1; CFG=$2; ENCODER=$3; SEED=$4; INTERACTION=$5; shift 5
source /hnvme/workspace/v103fe12-ped_gen/motion-latent-diffusion/slurm/resplit/_env.sh
EGO_CKPT=$MODELS/ego_encoder/$ENCODER/checkpoints/best.pt
[[ -f "$VAE_CKPT" ]] || { echo "missing $VAE_CKPT"; exit 1; }
[[ -f "$EGO_CKPT" ]] || { echo "missing $EGO_CKPT"; exit 1; }
resume_or_finish "$NAME" 5000
extract_data || exit 1
cd "$REPO"
data_overrides

python -m train \
--cfg "$CFG" \
--nodebug \
--overrides \
    "${DATA_OVERRIDES[@]}" \
    "NAME=$NAME" \
    "SEED_VALUE=$SEED" \
    "model.latent_dim=[4, 256]" \
    "TRAIN.BATCH_SIZE=128" \
    "TRAIN.END_EPOCH=5000" \
    "TRAIN.PRETRAINED_VAE=$VAE_CKPT" \
    "TRAIN.PRETRAINED_EGO=$EGO_CKPT" \
    "LOGGER.WANDB.RESUME_ID=$NAME" \
    "DATASET.EGOMOTION.INTERACTION_CROP=$INTERACTION" \
    "DATASET.EGOMOTION.INTERACTION_WEIGHTED_SAMPLING=$INTERACTION" \
    "$@" \
    "${RESUME[@]}"
