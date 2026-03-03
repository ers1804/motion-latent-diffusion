#!/bin/bash -l
#SBATCH --job-name=new_vae_stoch
#SBATCH --output=/home/slurm/outputs/vq-vae-train_%j.txt
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

source ~/miniconda3/etc/profile.d/conda.sh

conda activate mld

cd /home/slurm/motion-latent-diffusion

export OMP_NUM_THREADS=4
export WANDB_API_KEY="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"

python -m train \
--cfg configs/config_ego_motion_new_vae_stoch.yaml \
--nodebug
