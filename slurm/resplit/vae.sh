#!/bin/bash -l
#SBATCH --job-name=rs1_vae
#SBATCH --output=/hnvme/workspace/v103fe12-ped_gen/outputs/rs1_vae_%j.txt
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=h100
#SBATCH --export=NONE

source /hnvme/workspace/v103fe12-ped_gen/motion-latent-diffusion/slurm/resplit/_env.sh
resume_or_finish "$VAE_NAME" 6000
extract_data || exit 1
cd "$REPO"
data_overrides

python -m train \
--cfg configs/config_ego_motion_vae.yaml \
--nodebug \
--overrides \
    "${DATA_OVERRIDES[@]}" \
    "NAME=$VAE_NAME" \
    "SEED_VALUE=1234" \
    "model.latent_dim=[4, 256]" \
    "TRAIN.BATCH_SIZE=128" \
    "TRAIN.END_EPOCH=6000" \
    "LOGGER.WANDB.RESUME_ID=$VAE_NAME" \
    "DATASET.EGOMOTION.INTERACTION_CROP=True" \
    "DATASET.EGOMOTION.INTERACTION_WEIGHTED_SAMPLING=True" \
    "${RESUME[@]}"
