#!/bin/bash -l
#SBATCH --job-name=pedgen_bank
#SBATCH --output=/hnvme/workspace/v103fe12-ped_gen/outputs/pedgen_bank_%j.txt
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=h100
#SBATCH --export=NONE

source /hnvme/workspace/v103fe12-ped_gen/motion-latent-diffusion/slurm/resplit/_env.sh
cd "$REPO"
python research/src/pedgen_baseline.py \
    --repo "$STORAGE_DIR/pedgen/PedGen" \
    --ckpt "$STORAGE_DIR/pedgen/pedgen_no_context.ckpt" \
    --out research/data/pedgen_no_context_bank.npz
