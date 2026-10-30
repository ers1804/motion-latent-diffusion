#!/bin/bash -l
#SBATCH --job-name=rs1_eval
#SBATCH --output=/hnvme/workspace/v103fe12-ped_gen/outputs/rs1_eval_%j.txt
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=h100
#SBATCH --export=NONE

# Definitive FID and K=5 root dump on one partition. The test partition is for reporting only.
# usage: eval.sh <run name> <config> <epoch> <guidance> <val|test> [zero]
NAME=$1; CFG=$2; EPOCH=$3; GUIDANCE=$4; SPLIT_NAME=$5; ZERO=$6
source /hnvme/workspace/v103fe12-ped_gen/motion-latent-diffusion/slurm/resplit/_env.sh
extract_data || exit 1
cd "$REPO"
export EGOPED_NAS=$TMPDIR
CKPT=$MODELS/mld/$NAME/checkpoints/epoch=$EPOCH.ckpt
[[ -f "$CKPT" ]] || { echo "missing $CKPT"; exit 1; }
TAG=${NAME}_ep${EPOCH}_cfg${GUIDANCE}

ZERO_FLAG=()
ENTRY="-m test"
if [[ "$ZERO" == "zero" ]]; then
    ZERO_FLAG=(--zero-ego)
    ENTRY="baselines/eval_uncond.py"
fi

python research/src/eval_ade_fde.py --model "$TAG" --cfg "$CFG" --ckpt "$CKPT" --guidance "$GUIDANCE" \
    --k 5 --split "scene_$SPLIT_NAME" --save-roots "${ZERO_FLAG[@]}"

python $ENTRY \
--cfg "$CFG" \
--nodebug \
--overrides \
    "DATASET.EGOMOTION.ROOT=[$TMPDIR/data/diffusion/ava, $TMPDIR/data/diffusion/nuscenes, $TMPDIR/data/diffusion/waymo]" \
    "DATASET.EGOMOTION.MEAN_STD_PATH=$SPLIT" \
    "DATASET.EGOMOTION.EGO_MEAN_STD_PATH=$SPLIT" \
    "DATASET.EGOMOTION.SCENE_SPLIT=True" \
    "TEST.CHECKPOINTS=$CKPT" \
    "METRIC.TYPE=['EgoMotionMetrics']" \
    "TEST.REPLICATION_TIMES=3" \
    "TEST.SPLIT=$SPLIT_NAME" \
    "model.guidance_scale=$GUIDANCE" \
    "NAME=eval_${SPLIT_NAME}_$TAG" \
    "DATASET.EGOMOTION.INTERACTION_CROP=False" \
    "DATASET.EGOMOTION.INTERACTION_WEIGHTED_SAMPLING=False"
