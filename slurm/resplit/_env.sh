# Sourced by every job of the scene-split retraining (research/review_action_plan_2026-10-30.md, step 4).
unset SLURM_EXPORT_ENV
module add cuda/12.6.2
module add python
conda activate mld

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
export HTTP_PROXY=http://proxy.nhr.fau.de:80
export HTTPS_PROXY=http://proxy.nhr.fau.de:80
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=128

STORAGE_DIR=/hnvme/workspace/v103fe12-ped_gen
REPO=$STORAGE_DIR/motion-latent-diffusion
MODELS=$STORAGE_DIR/models_resplit
SPLIT=research/data/scene_split_v1
VAE_NAME=rs1_vae_latent4
VAE_CKPT=$MODELS/mld/$VAE_NAME/checkpoints/epoch=5999.ckpt

extract_data() {
    mkdir -p "$TMPDIR/data"
    local failed archive pass
    for pass in 1 2 3 4 5; do
        failed=$(mktemp)
        for archive in "$STORAGE_DIR"/data/*; do
            ( tar xzf "$archive" -C "$TMPDIR/data" --overwrite || echo "$archive" >> "$failed" ) &
            while (( $(jobs -r | wc -l) >= 8 )); do wait -n; done
        done
        wait
        if [[ ! -s "$failed" ]]; then rm -f "$failed"; echo "Extraction successful"; return 0; fi
        echo "Extraction pass $pass failed for:"; cat "$failed"; rm -f "$failed"; sleep 5
    done
    echo "Extraction failed"; return 1
}

data_overrides() {
    DATA_OVERRIDES=(
        "DATASET.EGOMOTION.ROOT=[$TMPDIR/data/diffusion/ava, $TMPDIR/data/diffusion/nuscenes, $TMPDIR/data/diffusion/waymo]"
        "DATASET.EGOMOTION.MEAN_STD_PATH=$SPLIT"
        "DATASET.EGOMOTION.EGO_MEAN_STD_PATH=$SPLIT"
        "DATASET.EGOMOTION.SCENE_SPLIT=True"
        "FOLDER=$MODELS"
    )
    DATA_OVERRIDES+=("${WF_OVERRIDE[@]}")
}

# rs2_* runs see the ego of a windowed track in the frame of the window start (EgoMotionDataset.ego_window_frame,
# research/review_action_plan_2026-11-13.md). Keyed on the run name, so that a resumed segment or a later
# evaluation cannot fall back to the frame of the rs1_* runs. Sets flags for the python entry points.
WF_OVERRIDE=(); WF_FLAG=(); WF_PRETRAIN=()
if [[ "$NAME" == rs2_* ]]; then
    WF_OVERRIDE=("DATASET.EGOMOTION.EGO_WINDOW_FRAME=True"); WF_FLAG=(--window-frame); WF_PRETRAIN=(--window_frame)
fi

# Sets RESUME when a checkpoint exists; exits when the last epoch is already there.
resume_or_finish() {
    local name=$1 end=$2
    local dir=$MODELS/mld/$name
    # A save that hit a full filesystem leaves an empty or truncated file; resuming would load it and crash.
    local max f
    max=$(stat -c %s "$dir"/checkpoints/epoch=*.ckpt 2>/dev/null | sort -n | tail -1)
    for f in "$dir"/checkpoints/epoch=*.ckpt; do
        [[ -f "$f" ]] || continue
        if (( $(stat -c %s "$f") < max * 99 / 100 )); then
            mkdir -p "$dir/failed_saves"; mv "$f" "$dir/failed_saves/"; echo "moved aside truncated $f"
        fi
    done
    if [[ -f "$dir/checkpoints/epoch=$((end - 1)).ckpt" ]]; then echo "$name already complete"; exit 0; fi
    # Always explicit: configs carry stale RESUME paths that would load an unrelated model.
    RESUME=("TRAIN.RESUME=")
    if compgen -G "$dir/checkpoints/epoch=*.ckpt" > /dev/null; then RESUME=("TRAIN.RESUME=$dir"); fi
}
