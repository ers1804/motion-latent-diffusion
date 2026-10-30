#!/bin/bash -l
#SBATCH --job-name=rs1_smoke
#SBATCH --output=/hnvme/workspace/v103fe12-ped_gen/outputs/rs1_smoke_%j.txt
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=h100
#SBATCH --export=NONE

# One epoch of every recipe in the chain with validation on, before days of queue time are spent.
source /hnvme/workspace/v103fe12-ped_gen/motion-latent-diffusion/slurm/resplit/_env.sh
extract_data || exit 1
cd "$REPO"
export WANDB_MODE=disabled
export EGOPED_NAS=$TMPDIR
MODELS=$STORAGE_DIR/models_resplit_smoke
rm -rf "$MODELS"
mkdir -p "$MODELS"
data_overrides
RESUME=("TRAIN.RESUME=")
OLD_VAE=$STORAGE_DIR/models/vae/ego_motion_vae_latent_4_wo_traj_interaction_crop_weighted_sampling/checkpoints/epoch=5999.ckpt
EGOPED=configs/config_ego_motion_new_vae_stoch_latent_4_trans_dec.yaml
POOLED=configs/config_ego_motion_new_vae_stoch.yaml
UNFROZEN=configs/config_ego_motion_new_vae_stoch_latent_4_h6_unfreeze_ego.yaml
SHORT=("TRAIN.END_EPOCH=1" "LOGGER.VAL_EVERY_STEPS=1" "LOGGER.SACE_CHECKPOINT_EPOCH=1")

python research/src/test_scene_split.py || exit 1

python -m train --cfg configs/config_ego_motion_vae.yaml --nodebug --overrides \
    "${DATA_OVERRIDES[@]}" "${SHORT[@]}" "${RESUME[@]}" "NAME=smoke_vae" "model.latent_dim=[4, 256]" \
    "TRAIN.BATCH_SIZE=128" "DATASET.EGOMOTION.INTERACTION_CROP=True" "DATASET.EGOMOTION.INTERACTION_WEIGHTED_SAMPLING=True" || exit 1

for e in "fullseq $EGOPED" "pooled $POOLED"; do
    set -- $e
    python pretrain_ego_encoder.py --cfg "$2" --epochs 1 --batch_size 64 --lr 1e-4 \
        --output_dir "$MODELS/ego_encoder/$1" --pretrained_vae "$OLD_VAE" --scene_split --interaction off \
        --mean_std_path "$SPLIT" --no_tensorboard \
        --data_roots "$TMPDIR/data/diffusion/ava" "$TMPDIR/data/diffusion/nuscenes" "$TMPDIR/data/diffusion/waymo" || exit 1
done

run() {  # name config encoder interaction [extra overrides]
    local name=$1 cfg=$2 enc=$3 ia=$4; shift 4
    python -m train --cfg "$cfg" --nodebug --overrides \
        "${DATA_OVERRIDES[@]}" "${SHORT[@]}" "${RESUME[@]}" "NAME=smoke_$name" "model.latent_dim=[4, 256]" \
        "TRAIN.BATCH_SIZE=128" "TRAIN.PRETRAINED_VAE=$OLD_VAE" "TRAIN.PRETRAINED_EGO=$MODELS/ego_encoder/$enc/checkpoints/best.pt" \
        "DATASET.EGOMOTION.INTERACTION_CROP=$ia" "DATASET.EGOMOTION.INTERACTION_WEIGHTED_SAMPLING=$ia" "$@" || exit 1
}
run egoped_ia "$EGOPED" fullseq True
run pooled "$POOLED" pooled False
run unfrozen "$UNFROZEN" fullseq False
run uncond "$EGOPED" fullseq False "model.guidance_uncondp=1.0" "model.guidance_scale=1.0"

CK=$(ls -d "$MODELS"/mld/smoke_pooled/checkpoints/*.ckpt | head -1)
python research/src/eval_ade_fde.py --model smoke_pooled --cfg "$POOLED" --ckpt "$CK" --guidance 5 --k 1 --split scene_test || exit 1
echo SMOKE_OK
