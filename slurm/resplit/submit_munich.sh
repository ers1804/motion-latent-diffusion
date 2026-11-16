#!/bin/bash
# Submits slurm/resplit/eval_munich.sh for the models of the main table and the three regressors, at the checkpoints
# and guidance scales selected on validation. Run from the repository root on the cluster, once.
# usage: [NICE=<n>] submit_munich.sh <resplit_selection.json>
set -e -o pipefail
SELECTION=$1
[[ -f "$SELECTION" ]] || { echo "usage: submit_munich.sh <resplit_selection.json>"; exit 1; }
EGOPED=configs/config_ego_motion_new_vae_stoch_latent_4_trans_dec.yaml
POOLED=configs/config_ego_motion_new_vae_stoch.yaml

config_of() {
    case ${1#rs[0-9]_} in
        pooled*) echo "$POOLED" ;;
        *) echo "$EGOPED" ;;
    esac
}

ls research/data/munich_v1/sequences_label.npz research/data/munich_v1/sequences_motion.npz > /dev/null
python3 -c "import json, sys; [print(r, s['epoch'], s.get('guidance', '')) for r, s in sorted(json.load(open(sys.argv[1])).items())]" \
    "$SELECTION" | while read -r run epoch scale; do
    [[ $run == *unfrozen* ]] && continue  # an encoder ablation, not a model of the main table
    zero=; [[ $run == *uncond* ]] && zero=zero
    sbatch ${NICE:+--nice="$NICE"} slurm/resplit/eval_munich.sh "$run" "$(config_of "$run")" "$epoch" "$scale" $zero
done
for seed in 1234 2345 3456; do
    sbatch ${NICE:+--nice="$NICE"} slurm/resplit/eval_munich.sh regressor "$seed"
done
