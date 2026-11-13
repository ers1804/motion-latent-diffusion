#!/bin/bash -l
#SBATCH --job-name=rs1_rev
#SBATCH --output=/hnvme/workspace/v103fe12-ped_gen/outputs/rs1_rev_%j.txt
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=h100
#SBATCH --export=NONE

# Review of 2026-11-13 (research/review_action_plan_2026-11-13.md, C3 and C4), one selected checkpoint, K=5 root
# dumps on validation and test:
#   hor1brake2: the first second of the ego trajectory, then the same direction with braking at 2 m/s^2 until the
#               vehicle stands. Its pair is the hor1 dump of eval_controls.sh (first second, then constant velocity),
#               so the two differ only in the speed profile.
#   body:       the unedited generation again, with foot-skate ratio and articulation of every sample and of the labels.
# They select nothing, so the test partition is fine. usage: eval_review.sh <run name> <config> <epoch> <guidance>
NAME=$1; CFG=$2; EPOCH=$3; GUIDANCE=$4
source /hnvme/workspace/v103fe12-ped_gen/motion-latent-diffusion/slurm/resplit/_env.sh
extract_data || exit 1
cd "$REPO"
export EGOPED_NAS=$TMPDIR
CKPT=$MODELS/mld/$NAME/checkpoints/epoch=$EPOCH.ckpt
[[ -f "$CKPT" ]] || { echo "missing $CKPT"; exit 1; }
TAG=${NAME}_ep${EPOCH}_cfg${GUIDANCE}

run() {  # tag suffix, split, extra flags
    local suffix=$1 split=$2; shift 2
    python research/src/eval_ade_fde.py --model "${TAG}_$suffix" --cfg "$CFG" --ckpt "$CKPT" --guidance "$GUIDANCE" \
        --k 5 --split "scene_$split" --save-roots "$@"
}
for SPLIT_NAME in val test; do
    run hor1brake2 "$SPLIT_NAME" --ego-horizon 1 --ego-brake 2
    run body "$SPLIT_NAME" --save-skate
done
