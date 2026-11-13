#!/bin/bash
# Queues segments for runs by name: new runs start, runs of submit_chain.sh that stopped early resume from the
# newest intact checkpoint, and complete runs exit at once. Run from the repository root on the cluster.
# usage: [AFTER=<job id>[:<job id>]] [NICE=<n>] resume_runs.sh <segments> <run name>...
# AFTER makes the first segment of every run wait until those jobs have finished without error (afterok, so a
# failed or timed-out encoder blocks the chain instead of leaving a partly trained encoder to train on). NICE
# lowers the priority of all segments. Later segments follow their predecessor whatever its state.
set -e
SEGMENTS=$1; shift
EGOPED=configs/config_ego_motion_new_vae_stoch_latent_4_trans_dec.yaml
POOLED=configs/config_ego_motion_new_vae_stoch.yaml
UNFROZEN=configs/config_ego_motion_new_vae_stoch_latent_4_h6_unfreeze_ego.yaml
UNCOND=("model.guidance_uncondp=1.0" "model.guidance_scale=1.0")

for name in "$@"; do
    seed=${name##*_s}
    gen=${name%%_*}  # rs1: ego in the frame of the track's first pose, rs2: in the frame of the window start
    extra=()
    case ${name#"$gen"_} in
        egoped_ia_s*) args=("$EGOPED" "${gen}_enc_fullseq" "$seed" True) ;;
        egoped_s*) args=("$EGOPED" "${gen}_enc_fullseq" "$seed" False) ;;
        pooled_ia_s*) args=("$POOLED" "${gen}_enc_pooled" "$seed" True) ;;
        # ablations of the method (Pooled): encoder trainable, and trainable without pretraining
        pooled_unfrozen_s*) args=("$POOLED" "${gen}_enc_pooled" "$seed" False); extra=("TRAIN.FREEZE_EGO=False") ;;
        pooled_scratch_s*) args=("$POOLED" "${gen}_enc_pooled" "$seed" False); extra=("TRAIN.FREEZE_EGO=False" "TRAIN.PRETRAINED_EGO=") ;;
        pooled_s*) args=("$POOLED" "${gen}_enc_pooled" "$seed" False) ;;
        unfrozen_s*) args=("$UNFROZEN" "${gen}_enc_fullseq" "$seed" False) ;;
        # ego-blind, so the frame does not matter: only rs1 priors exist
        uncond_ia_s*) args=("$EGOPED" rs1_enc_fullseq "$seed" True); extra=("${UNCOND[@]}") ;;
        uncond_s*) args=("$EGOPED" rs1_enc_fullseq "$seed" False); extra=("${UNCOND[@]}") ;;
        *) echo "unknown run $name"; exit 1 ;;
    esac
    dep=()
    [[ -n "$AFTER" ]] && dep=(--dependency=afterok:"$AFTER")
    for _ in $(seq "$SEGMENTS"); do
        id=$(sbatch --parsable --job-name="$name" ${NICE:+--nice="$NICE"} "${dep[@]}" slurm/resplit/diffusion.sh "$name" "${args[@]}" "${extra[@]}")
        dep=(--dependency=afterany:"$id")
    done
    echo "$name last segment $id"
done
