#!/bin/bash -l
#SBATCH --job-name=eval_h4_cfg15
#SBATCH --output=/hnvme/workspace/v103fe12-ped_gen/outputs/eval_h4_cfg15_%j.txt
#SBATCH --time=4:00:00
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
    local src_dir="$1"
    local dst_dir="$2"
    local max_parallel=8
    local sleep_sec=5

    local archives
    mapfile -t archives < <(ls -1 "$src_dir")

    while true; do
        echo "Starting extraction pass for $src_dir"

        failed=$(mktemp)

        running=0
        for archive in "${archives[@]}"; do
            (
                if ! tar xzf "$src_dir/$archive" -C "$dst_dir" --overwrite; then
                    echo "$archive" >> "$failed"
                fi
            ) &

            ((running+=1))
            if (( running >= max_parallel )); then
                wait -n
                ((running-=1))
            fi
        done

        wait

        if [[ ! -s "$failed" ]]; then
            echo "Extraction successful for $src_dir"
            rm -f "$failed"
            break
        fi

        echo "Retrying failed archives:"
        cat "$failed"

        mapfile -t archives < "$failed"
        rm -f "$failed"

        sleep "$sleep_sec"
    done
}

STORAGE_DIR=/hnvme/workspace/v103fe12-ped_gen

mkdir -p $TMPDIR/data
extract_until_success "$STORAGE_DIR/data" "$TMPDIR/data"

cd /hnvme/workspace/v103fe12-ped_gen/motion-latent-diffusion

# Evaluate H4 (trans_dec) at CFG=15 — definitive eval at best epoch
# Update EPOCH after segment-2 completes and best checkpoint is identified from wandb
# Goal: characterize H4 FID/R-prec at high guidance vs H2 (FID=7.400, R-prec=0.797)
EPOCH=4399
CHECKPOINT=/hnvme/workspace/v103fe12-ped_gen/models/mld/ego_motion_diffusion_h4_trans_dec/checkpoints/epoch=${EPOCH}.ckpt

python -m test \
--cfg configs/config_ego_motion_new_vae_stoch_latent_4_trans_dec.yaml \
--nodebug \
--overrides \
    "DATASET.EGOMOTION.ROOT=[$TMPDIR/data/diffusion/ava, $TMPDIR/data/diffusion/nuscenes, $TMPDIR/data/diffusion/waymo]" \
    "DATASET.EGOMOTION.MEAN_STD_PATH=$TMPDIR/data/vae/mean_std_txt/ava_nuscenes_waymo" \
    "DATASET.EGOMOTION.EGO_MEAN_STD_PATH=$TMPDIR/data/vae/mean_std_txt/ava_nuscenes_waymo" \
    "TEST.CHECKPOINTS=$CHECKPOINT" \
    "METRIC.TYPE=['EgoMotionMetrics']" \
    "TEST.REPLICATION_TIMES=3" \
    "TEST.SPLIT=val" \
    "model.guidance_scale=15"
