#!/bin/bash -l
#SBATCH --job-name=rs1_selu
#SBATCH --output=/hnvme/workspace/v103fe12-ped_gen/outputs/rs1_selu_%j.txt
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=h100
#SBATCH --export=NONE

# Validation FID of every saved checkpoint of an unconditional run, with the ego zeroed. Training-time
# validation feeds these runs the real ego, so research/src/select_resplit.py picks from these results.
# One replication, as in training-time validation. usage: select_uncond.sh <run name> <config>
NAME=$1; CFG=$2
source /hnvme/workspace/v103fe12-ped_gen/motion-latent-diffusion/slurm/resplit/_env.sh
extract_data || exit 1
cd "$REPO"

for CKPT in $(ls "$MODELS/mld/$NAME/checkpoints/"epoch=*.ckpt | sort -t= -k2 -n); do
    EPOCH=$(basename "$CKPT" .ckpt); EPOCH=${EPOCH#epoch=}
    compgen -G "results/mld/selu_val_${NAME}_ep$EPOCH/metrics_*.json" > /dev/null && continue
    python baselines/eval_uncond.py \
    --cfg "$CFG" \
    --nodebug \
    --overrides \
        "DATASET.EGOMOTION.ROOT=[$TMPDIR/data/diffusion/ava, $TMPDIR/data/diffusion/nuscenes, $TMPDIR/data/diffusion/waymo]" \
        "DATASET.EGOMOTION.MEAN_STD_PATH=$SPLIT" \
        "DATASET.EGOMOTION.EGO_MEAN_STD_PATH=$SPLIT" \
        "DATASET.EGOMOTION.SCENE_SPLIT=True" \
        "TEST.CHECKPOINTS=$CKPT" \
        "METRIC.TYPE=['EgoMotionMetrics']" \
        "TEST.REPLICATION_TIMES=1" \
        "TEST.SPLIT=val" \
        "model.guidance_scale=1.0" \
        "NAME=selu_val_${NAME}_ep$EPOCH" \
        "DATASET.EGOMOTION.INTERACTION_CROP=False" \
        "DATASET.EGOMOTION.INTERACTION_WEIGHTED_SAMPLING=False"
done
