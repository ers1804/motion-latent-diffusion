#!/bin/bash
# Submits VAE -> two encoders -> Tier 1 diffusion runs. Run from the repository root on the cluster.
# TIMEOUT never satisfies afterok, so segments chain with afterany; each job checks its inputs.
set -e
EGOPED=configs/config_ego_motion_new_vae_stoch_latent_4_trans_dec.yaml
POOLED=configs/config_ego_motion_new_vae_stoch.yaml
UNFROZEN=configs/config_ego_motion_new_vae_stoch_latent_4_h6_unfreeze_ego.yaml
UNCOND=("model.guidance_uncondp=1.0" "model.guidance_scale=1.0")

submit() { sbatch --parsable "$@"; }

prev=$(submit slurm/resplit/vae.sh)
for _ in 1 2 3; do prev=$(submit --dependency=afterany:"$prev" slurm/resplit/vae.sh); done
vae=$prev
echo "vae last segment $vae"

enc_full=$(submit --dependency=afterany:"$vae" slurm/resplit/encoder.sh rs1_enc_fullseq "$EGOPED" off)
enc_pool=$(submit --dependency=afterany:"$vae" slurm/resplit/encoder.sh rs1_enc_pooled "$POOLED" off)
echo "encoders $enc_full $enc_pool"

run() {  # name config encoder seed interaction [extra overrides]
    local name=$1 cfg=$2 enc=$3 seed=$4 ia=$5; shift 5
    local dep=afterany:$enc_full:$enc_pool id
    for _ in 1 2 3; do
        id=$(submit --job-name="$name" --dependency="$dep" slurm/resplit/diffusion.sh "$name" "$cfg" "$enc" "$seed" "$ia" "$@")
        dep=afterany:$id
    done
    echo "$name last segment $id"
}

# Three seeds per cell of the 2x2 (conditioning x recipe); the other models are single-seed references.
for seed in 1234 2345 3456; do
    run "rs1_egoped_s$seed"    "$EGOPED" rs1_enc_fullseq "$seed" False
    run "rs1_egoped_ia_s$seed" "$EGOPED" rs1_enc_fullseq "$seed" True
    run "rs1_pooled_s$seed"    "$POOLED" rs1_enc_pooled  "$seed" False
    run "rs1_pooled_ia_s$seed" "$POOLED" rs1_enc_pooled  "$seed" True
done
run rs1_unfrozen_s1234  "$UNFROZEN" rs1_enc_fullseq 1234 False
run rs1_uncond_s1234    "$EGOPED"   rs1_enc_fullseq 1234 False "${UNCOND[@]}"
run rs1_uncond_ia_s1234 "$EGOPED"   rs1_enc_fullseq 1234 True  "${UNCOND[@]}"
