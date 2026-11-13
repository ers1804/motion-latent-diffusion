#!/bin/bash -l
#SBATCH --job-name=rs1_ctrl
#SBATCH --output=/hnvme/workspace/v103fe12-ped_gen/outputs/rs1_ctrl_%j.txt
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=h100
#SBATCH --export=NONE

# Ego controls for one selected checkpoint: K=5 root dumps with the ego zeroed, taken from another scene
# (wrong vehicle), or shown only for the first 0/1/2/4 s. They select nothing, so the test partition is fine.
# usage: eval_controls.sh <run name> <config> <epoch> <guidance> <val|test>
NAME=$1; CFG=$2; EPOCH=$3; GUIDANCE=$4; SPLIT_NAME=$5
source /hnvme/workspace/v103fe12-ped_gen/motion-latent-diffusion/slurm/resplit/_env.sh
extract_data || exit 1
cd "$REPO"
export EGOPED_NAS=$TMPDIR
CKPT=$MODELS/mld/$NAME/checkpoints/epoch=$EPOCH.ckpt
[[ -f "$CKPT" ]] || { echo "missing $CKPT"; exit 1; }
TAG=${NAME}_ep${EPOCH}_cfg${GUIDANCE}

run() {  # tag suffix, extra flags
    local suffix=$1; shift
    python research/src/eval_ade_fde.py --model "${TAG}_$suffix" --cfg "$CFG" --ckpt "$CKPT" --guidance "$GUIDANCE" \
        --k 5 --split "scene_$SPLIT_NAME" --save-roots "${WF_FLAG[@]}" "$@"
}
run zero --zero-ego
run swap --ego-swap
for h in 0 1 2 4; do run "hor$h" --ego-horizon "$h"; done
