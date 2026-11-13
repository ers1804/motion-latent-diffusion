#!/bin/bash
# Submits slurm/resplit/eval.sh for every run of a selection file. Run from the repository root on the cluster.
# usage: submit_evals.sh sweep <resplit_epochs.json>      validation, guidance 5/10/15 (unconditional: 1, ego zeroed)
#        submit_evals.sh test  <resplit_selection.json>   test partition, once, at the selected guidance
set -e
MODE=$1; SELECTION=$2
EGOPED=configs/config_ego_motion_new_vae_stoch_latent_4_trans_dec.yaml
POOLED=configs/config_ego_motion_new_vae_stoch.yaml
UNFROZEN=configs/config_ego_motion_new_vae_stoch_latent_4_h6_unfreeze_ego.yaml

config_of() {
    case ${1#rs[0-9]_} in  # run kind without the generation prefix (rs1_, rs2_)
        pooled*) echo "$POOLED" ;;
        unfrozen*) echo "$UNFROZEN" ;;
        *) echo "$EGOPED" ;;
    esac
}

python3 -c "import json, sys; [print(r, s['epoch'], s.get('guidance', '')) for r, s in sorted(json.load(open(sys.argv[1])).items())]" \
    "$SELECTION" | while read -r run epoch scale; do
    cfg=$(config_of "$run")
    zero=; [[ $run == *uncond* ]] && zero=zero
    if [[ $MODE == sweep ]]; then
        scales="5 10 15"; [[ -n $zero ]] && scales=1
        for g in $scales; do sbatch --job-name="${run%%_*}_sweep" slurm/resplit/eval.sh "$run" "$cfg" "$epoch" "$g" val $zero; done
    elif [[ $MODE == test ]]; then
        sbatch --job-name="${run%%_*}_test" slurm/resplit/eval.sh "$run" "$cfg" "$epoch" "$scale" test $zero
    else
        echo "unknown mode $MODE"; exit 1
    fi
done
