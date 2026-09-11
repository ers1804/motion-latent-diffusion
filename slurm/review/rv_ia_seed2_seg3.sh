#!/bin/bash -l
# EgoPed-IA SECOND SEED (2345). Seed 1 (1234) gave the flagship FID 2.880 at ep1299.
# H4's seed spread is large (3.392 / 4.719 / 4.803), so the single-seed IA headline needs
# a replicate. Identical recipe; ONLY SEED_VALUE and NAME differ.
# 3 segments (24h/24h/10h) chained afterany: pipeline runs pace ~95-100 ep/h, so 5000
# epochs needs ~51h and a TIMEOUT never satisfies afterok.
#SBATCH --job-name=rv_ia_seed2_seg3
#SBATCH --output=/hnvme/workspace/v103fe12-ped_gen/outputs/rv_ia_seed2_seg3_%j.txt
#SBATCH --time=10:00:00
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
--cfg configs/config_ego_motion_new_vae_stoch_latent_4_trans_dec.yaml \
--nodebug \
--overrides \
    "NAME=ego_motion_diffusion_h4_pipeline_seed2" \
    "TRAIN.RESUME=/hnvme/workspace/v103fe12-ped_gen/models/mld/ego_motion_diffusion_h4_pipeline_seed2" \
    "SEED_VALUE=2345" \
    "TRAIN.PRETRAINED_VAE=/hnvme/workspace/v103fe12-ped_gen/models/vae/ego_motion_vae_latent_4_wo_traj_interaction_crop_weighted_sampling/checkpoints/epoch=5999.ckpt" \
    "TRAIN.PRETRAINED_EGO=/hnvme/workspace/v103fe12-ped_gen/models/ego_encoder/ego_encoder_h4_trans_dec/checkpoints/best.pt" \
    "LOGGER.WANDB.RESUME_ID=ego_motion_diffusion_h4_pipeline_seed2" \
    "DATASET.EGOMOTION.ROOT=[$TMPDIR/data/diffusion/ava, $TMPDIR/data/diffusion/nuscenes, $TMPDIR/data/diffusion/waymo]" \
    "DATASET.EGOMOTION.MEAN_STD_PATH=$TMPDIR/data/vae/mean_std_txt/ava_nuscenes_waymo" \
    "DATASET.EGOMOTION.EGO_MEAN_STD_PATH=$TMPDIR/data/vae/mean_std_txt/ava_nuscenes_waymo" \
    "FOLDER=/hnvme/workspace/v103fe12-ped_gen/models" \
    "DATASET.EGOMOTION.INTERACTION_CROP=True" \
    "DATASET.EGOMOTION.INTERACTION_WEIGHTED_SAMPLING=True"
