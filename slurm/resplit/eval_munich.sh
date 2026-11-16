#!/bin/bash -l
#SBATCH --job-name=rs1_munich
#SBATCH --output=/hnvme/workspace/v103fe12-ped_gen/outputs/rs1_munich_%j.txt
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=h100
#SBATCH --export=NONE

# A frozen model on the Munich external test set, both heading variants (research/src/munich_eval.py). K=5 root dumps
# only: the set has root paths and no body poses, so there is no FID. Nothing is selected on this set.
# The sequence files are not in git: copy research/data/munich_v1/sequences_*.npz next to the manifest first.
# usage: eval_munich.sh <run name> <config> <epoch> <guidance> [zero]
#        eval_munich.sh regressor <seed>
if [[ "$1" == regressor ]]; then NAME=rs1_regressor_s$2; else NAME=$1; CFG=$2; EPOCH=$3; GUIDANCE=$4; ZERO=$5; fi
source /hnvme/workspace/v103fe12-ped_gen/motion-latent-diffusion/slurm/resplit/_env.sh
cd "$REPO"
export EGOPED_MUNICH=$TMPDIR/munich
python research/src/munich_eval.py --materialize "$EGOPED_MUNICH" || exit 1

if [[ "$1" == regressor ]]; then
    python research/src/regressor_baseline.py --scene --seed "$2" --ckpt "$MODELS/regressor/$NAME.pt" \
        --eval --splits munich munich_motion
    exit
fi

CKPT=$MODELS/mld/$NAME/checkpoints/epoch=$EPOCH.ckpt
[[ -f "$CKPT" ]] || { echo "missing $CKPT"; exit 1; }
TAG=${NAME}_ep${EPOCH}_cfg${GUIDANCE}
ZERO_FLAG=(); [[ "$ZERO" == "zero" ]] && ZERO_FLAG=(--zero-ego)
for SPLIT_NAME in munich munich_motion; do
    python research/src/eval_ade_fde.py --model "$TAG" --cfg "$CFG" --ckpt "$CKPT" --guidance "$GUIDANCE" \
        --k 5 --split "$SPLIT_NAME" --save-roots "${ZERO_FLAG[@]}" || exit 1
done
