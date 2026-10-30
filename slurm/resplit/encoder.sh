#!/bin/bash -l
#SBATCH --job-name=rs1_enc
#SBATCH --output=/hnvme/workspace/v103fe12-ped_gen/outputs/rs1_enc_%j.txt
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=h100
#SBATCH --export=NONE

# usage: encoder.sh <name> <config> <on|off>
NAME=$1; CFG=$2; INTERACTION=$3
source /hnvme/workspace/v103fe12-ped_gen/motion-latent-diffusion/slurm/resplit/_env.sh
[[ -f "$VAE_CKPT" ]] || { echo "missing $VAE_CKPT"; exit 1; }
extract_data || exit 1
cd "$REPO"
OUTPUT_DIR=$MODELS/ego_encoder/$NAME
mkdir -p "$OUTPUT_DIR"

python pretrain_ego_encoder.py \
    --cfg "$CFG" \
    --epochs 200 \
    --batch_size 64 \
    --lr 1e-4 \
    --output_dir "$OUTPUT_DIR" \
    --pretrained_vae "$VAE_CKPT" \
    --scene_split \
    --interaction "$INTERACTION" \
    --mean_std_path "$SPLIT" \
    --data_roots \
        "$TMPDIR/data/diffusion/ava" \
        "$TMPDIR/data/diffusion/nuscenes" \
        "$TMPDIR/data/diffusion/waymo"
