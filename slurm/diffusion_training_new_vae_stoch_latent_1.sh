#!/bin/bash -l
#SBATCH --job-name=new_vae_stoch
#SBATCH --output=/mnt/md0/erik/outputs/diffusion_train_%j.txt
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:a6000:1
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

source ~/anaconda3/etc/profile.d/conda.sh

conda activate mld

cd /mnt/md0/erik/motion-latent-diffusion-1

export OMP_NUM_THREADS=12
export WANDB_API_KEY="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"

python -m train \
--cfg configs/config_ego_motion_new_vae_stoch_latent_1.yaml \
--nodebug
