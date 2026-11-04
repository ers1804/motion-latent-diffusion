#!/bin/bash
# Queues segments for runs by name: new runs start, runs of submit_chain.sh that stopped early resume from the
# newest intact checkpoint, and complete runs exit at once. Run from the repository root on the cluster.
# usage: resume_runs.sh <segments> <run name>...
set -e
SEGMENTS=$1; shift
EGOPED=configs/config_ego_motion_new_vae_stoch_latent_4_trans_dec.yaml
POOLED=configs/config_ego_motion_new_vae_stoch.yaml
UNFROZEN=configs/config_ego_motion_new_vae_stoch_latent_4_h6_unfreeze_ego.yaml
UNCOND=("model.guidance_uncondp=1.0" "model.guidance_scale=1.0")

for name in "$@"; do
    seed=${name##*_s}
    extra=()
    case $name in
        rs1_egoped_ia_s*) args=("$EGOPED" rs1_enc_fullseq "$seed" True) ;;
        rs1_egoped_s*) args=("$EGOPED" rs1_enc_fullseq "$seed" False) ;;
        rs1_pooled_ia_s*) args=("$POOLED" rs1_enc_pooled "$seed" True) ;;
        # ablations of the method (Pooled): encoder trainable, and trainable without pretraining
        rs1_pooled_unfrozen_s*) args=("$POOLED" rs1_enc_pooled "$seed" False); extra=("TRAIN.FREEZE_EGO=False") ;;
        rs1_pooled_scratch_s*) args=("$POOLED" rs1_enc_pooled "$seed" False); extra=("TRAIN.FREEZE_EGO=False" "TRAIN.PRETRAINED_EGO=") ;;
        rs1_pooled_s*) args=("$POOLED" rs1_enc_pooled "$seed" False) ;;
        rs1_unfrozen_s*) args=("$UNFROZEN" rs1_enc_fullseq "$seed" False) ;;
        rs1_uncond_ia_s*) args=("$EGOPED" rs1_enc_fullseq "$seed" True); extra=("${UNCOND[@]}") ;;
        rs1_uncond_s*) args=("$EGOPED" rs1_enc_fullseq "$seed" False); extra=("${UNCOND[@]}") ;;
        *) echo "unknown run $name"; exit 1 ;;
    esac
    dep=()
    for _ in $(seq "$SEGMENTS"); do
        id=$(sbatch --parsable --job-name="$name" "${dep[@]}" slurm/resplit/diffusion.sh "$name" "${args[@]}" "${extra[@]}")
        dep=(--dependency=afterany:"$id")
    done
    echo "$name last segment $id"
done
