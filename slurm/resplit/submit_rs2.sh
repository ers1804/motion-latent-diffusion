#!/bin/bash
# Retraining with the ego in the frame of the window start (research/review_action_plan_2026-11-13.md): two ego
# encoders, then every conditional model of the paper. The motion VAE and the ego-blind priors of rs1 are kept.
# Run from the repository root on the cluster. NICE keeps these jobs behind everything already queued.
set -e
EGOPED=configs/config_ego_motion_new_vae_stoch_latent_4_trans_dec.yaml
POOLED=configs/config_ego_motion_new_vae_stoch.yaml
export NICE=${NICE:-100}

enc_full=$(sbatch --parsable --nice="$NICE" --job-name=rs2_enc_fullseq slurm/resplit/encoder.sh rs2_enc_fullseq "$EGOPED" off)
enc_pool=$(sbatch --parsable --nice="$NICE" --job-name=rs2_enc_pooled slurm/resplit/encoder.sh rs2_enc_pooled "$POOLED" off)
echo "encoders $enc_full $enc_pool"

runs=(rs2_unfrozen_s1234)
for seed in 1234 2345 3456; do
    runs+=("rs2_pooled_s$seed" "rs2_egoped_s$seed" "rs2_pooled_ia_s$seed" "rs2_egoped_ia_s$seed"
           "rs2_pooled_unfrozen_s$seed" "rs2_pooled_scratch_s$seed")
done
AFTER=$enc_full:$enc_pool bash slurm/resplit/resume_runs.sh 3 "${runs[@]}"
