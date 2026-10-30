#!/bin/bash -l
#SBATCH --job-name=old_eval
#SBATCH --output=/hnvme/workspace/v103fe12-ped_gen/outputs/old_eval_%j.txt
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=h100
#SBATCH --export=NONE

# Old-split scoring of the Pooled models (plan step 3 and 5).
# usage: eval_old_split.sh <eval_ade_fde model> <model dir name> <epoch>
MODEL=$1; DIR=$2; EPOCH=$3
source /hnvme/workspace/v103fe12-ped_gen/motion-latent-diffusion/slurm/resplit/_env.sh
extract_data || exit 1
cd "$REPO"
export EGOPED_NAS=$TMPDIR
export EGOPED_MODELS=$STORAGE_DIR/models/mld
OLD=$TMPDIR/data/vae/mean_std_txt/ava_nuscenes_waymo

python research/src/eval_ade_fde.py --model "$MODEL" --k 5 --split val_test --save-roots

python -m test \
--cfg configs/config_ego_motion_new_vae_stoch.yaml \
--nodebug \
--overrides \
    "DATASET.EGOMOTION.ROOT=[$TMPDIR/data/diffusion/ava, $TMPDIR/data/diffusion/nuscenes, $TMPDIR/data/diffusion/waymo]" \
    "DATASET.EGOMOTION.MEAN_STD_PATH=$OLD" \
    "DATASET.EGOMOTION.EGO_MEAN_STD_PATH=$OLD" \
    "TEST.CHECKPOINTS=$EGOPED_MODELS/$DIR/checkpoints/epoch=$EPOCH.ckpt" \
    "METRIC.TYPE=['EgoMotionMetrics']" \
    "TEST.REPLICATION_TIMES=3" \
    "TEST.SPLIT=val" \
    "model.guidance_scale=5" \
    "NAME=old_eval_$MODEL" \
    "DATASET.EGOMOTION.INTERACTION_CROP=False" \
    "DATASET.EGOMOTION.INTERACTION_WEIGHTED_SAMPLING=False"
