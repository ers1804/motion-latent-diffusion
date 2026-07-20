#!/bin/bash -l
#SBATCH --job-name=rv_h2_seedB
#SBATCH --output=/hnvme/workspace/v103fe12-ped_gen/outputs/rv_h2_seedB_%j.txt
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=h100
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
module add cuda/12.6.2
module add python
conda activate mld
export WANDB_API_KEY="4d79c39eaee42c190a8e4b84553f61ad449ad09b"
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
export HTTP_PROXY=http://proxy.nhr.fau.de:80
export HTTPS_PROXY=http://proxy.nhr.fau.de:80
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=128

extract_until_success() {
    local src_dir="$1"; local dst_dir="$2"; local max_parallel=8; local sleep_sec=5
    local archives; mapfile -t archives < <(ls -1 "$src_dir")
    while true; do
        echo "Starting extraction pass for $src_dir"
        failed=$(mktemp); running=0
        for archive in "${archives[@]}"; do
            ( tar xzf "$src_dir/$archive" -C "$dst_dir" --overwrite || echo "$archive" >> "$failed" ) &
            ((running+=1)); if (( running >= max_parallel )); then wait -n; ((running-=1)); fi
        done
        wait
        if [[ ! -s "$failed" ]]; then echo "Extraction successful"; rm -f "$failed"; break; fi
        echo "Retrying failed archives:"; cat "$failed"; rm -f "$failed"; sleep "$sleep_sec"
    done
}

STORAGE_DIR=/hnvme/workspace/v103fe12-ped_gen
mkdir -p $TMPDIR/data
extract_until_success "$STORAGE_DIR/data" "$TMPDIR/data"
cd /hnvme/workspace/v103fe12-ped_gen/motion-latent-diffusion

python -m train \
--cfg configs/config_ego_motion_new_vae_stoch.yaml \
--nodebug \
--overrides \
    "NAME=ego_motion_diffusion_h2_seedB" \
    "SEED_VALUE=2345" \
    "TRAIN.BATCH_SIZE=128" \
    "TRAIN.PRETRAINED_VAE=/hnvme/workspace/v103fe12-ped_gen/models/vae/ego_motion_vae_latent_4_wo_traj_interaction_crop_weighted_sampling/checkpoints/epoch=5999.ckpt" \
    "TRAIN.PRETRAINED_EGO=/hnvme/workspace/v103fe12-ped_gen/models/ego_encoder/ego_encoder_interaction_crop_weighted/checkpoints/best.pt" \
    "LOGGER.WANDB.RESUME_ID=ego_motion_diffusion_h2_seedB" \
    "DATASET.EGOMOTION.ROOT=[$TMPDIR/data/diffusion/ava, $TMPDIR/data/diffusion/nuscenes, $TMPDIR/data/diffusion/waymo]" \
    "DATASET.EGOMOTION.MEAN_STD_PATH=$TMPDIR/data/vae/mean_std_txt/ava_nuscenes_waymo" \
    "DATASET.EGOMOTION.EGO_MEAN_STD_PATH=$TMPDIR/data/vae/mean_std_txt/ava_nuscenes_waymo" \
    "FOLDER=/hnvme/workspace/v103fe12-ped_gen/models" \
    "DATASET.EGOMOTION.INTERACTION_CROP=True" \
    "DATASET.EGOMOTION.INTERACTION_WEIGHTED_SAMPLING=True"
