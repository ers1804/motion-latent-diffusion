# Reproducing the Paper Qualitatives (Fig. 4) — Weights, Configs, and the CARLA Port Contract

*Written 2026-08-03 for handoff. The figure shows the H4 model (epoch 3399, CFG = 10) on three
staged AVA validation scenes. Everything below is pinned; the "Traps" section is not optional
reading — three of the four have already bitten this project.*

## 1. Exact artifacts

| Artifact | Path |
|---|---|
| **Model weights** (H4, best ckpt) | NAS: `/home/erik/NAS/methods/diffusion_gen/models/helma_models/models/mld/ego_motion_diffusion_h4_trans_dec/checkpoints/epoch=3399.ckpt` |
| — same, on helma | `/hnvme/workspace/v103fe12-ped_gen/models/mld/ego_motion_diffusion_h4_trans_dec/checkpoints/epoch=3399.ckpt` |
| **Config (repo)** | `configs/config_ego_motion_new_vae_stoch_latent_4_trans_dec.yaml` |
| **Config (authoritative)** | `config_2026-04-02-11-57-36_train.yaml` inside the run dir next to the checkpoint — this is what the run *actually* used; the repo config has drifted (paths) |
| **Input samples** | `/home/erik/NAS/methods/diffusion_gen/data/diffusion/ava/val/{0002_26,0005_26,0008_37}.json` |
| **Normalization stats** | `/home/erik/NAS/methods/diffusion_gen/data/vae/mean_std_txt/ava_nuscenes_waymo/` (`Mean/Std.npy` for motion, `Ego_Mean/Ego_Std.npy` for ego) |
| Frozen VAE (already inside the ckpt; pretrain source) | `.../models/vae/ego_motion_vae_latent_4_wo_traj_interaction_crop_weighted_sampling/checkpoints/epoch=5999.ckpt` |
| Frozen ego encoder (already inside the ckpt; pretrain source) | `.../models/ego_encoder/ego_encoder_h4_trans_dec/checkpoints/best.pt` |

The checkpoint is **self-contained**: `state_dict` includes denoiser + frozen VAE + frozen ego
encoder. You only need the ckpt + config + normalization stats + an ego trajectory.

## 2. Regenerating the qualitatives

```bash
conda activate mld
cd motion-latent-diffusion

NAS=/home/erik/NAS/methods/diffusion_gen
for S in 0002_26 0005_26 0008_37; do
python demo_ego_motion.py \
  --config configs/config_ego_motion_new_vae_stoch_latent_4_trans_dec.yaml \
  --checkpoint $NAS/models/helma_models/models/mld/ego_motion_diffusion_h4_trans_dec/checkpoints/epoch=3399.ckpt \
  --ego_json $NAS/data/diffusion/ava/val/$S.json \
  --mean_std_path $NAS/data/vae/mean_std_txt/ava_nuscenes_waymo \
  --output_dir outputs/ego_demo/ego_motion_diffusion_h4_trans_dec \
  --trajectories
done
```

Outputs per sample: `*_features.npy` (263-D), `*_joints.npy` (T×22×3), `*_joints.mp4`,
`*_trajectories.png`. The paper figure is assembled by
`research/paper/figures/make_qualitatives.py` (it additionally reads skeleton frames extracted
from the mp4s into `/tmp/qual_frames/<sample>_<i>.png` — re-extract with ffmpeg if regenerating).

CFG scale: sampling guidance comes from the config (`model.guidance_scale: 10`) — do not confuse
it with the demo's `--guide_scale` flag, which controls an unrelated trajectory-guidance feature.

## 3. Model I/O contract (what the CARLA port must provide/consume)

**Input** — one ego trajectory per generation:
- `ego ∈ R^(T×2)`, `T = 196` frames at **20 fps** (~9.8 s), ground-plane **XZ** coordinates.
- Frame convention: **pedestrian-centric** — translation/yaw-normalized to the pedestrian
  (the dataset stores `ego_in_ped_frame`; in CARLA you must transform the vehicle trajectory
  into the frame of the pedestrian you want to animate: ped at origin, ped-heading-aligned).
- Normalize with `Ego_Mean/Ego_Std` from the stats dir above ((3,)-stats are projected to XZ
  by the loader). Shorter sequences: zero-pad to 196 and pass the true length.
- CFG dropout convention: the **all-zero trajectory is the null condition** — zero the ego input
  to sample the unconditional prior.

**Output**:
- `features ∈ R^(T×263)` (normalized HumanML3D). Denormalize with `Mean/Std`, then
  `recover_from_ric` (`mld/data/humanml/scripts/motion_process.py`) or
  `datamodule.feats2joints` → `joints ∈ R^(T×22×3)` (SMPL-topology skeleton, meters,
  pelvis = joint 0). For meshes in CARLA, fit/retarget SMPL to the 22 joints
  (`mld/transforms/joints2rots` has SMPLify utilities).
- Output is in the same pedestrian-centric frame — apply the inverse transform to place the
  motion back into the CARLA world frame.
- Sampling: DDIM, 50 steps, CFG = 10. Stochastic — sample K per condition for diversity.

## 4. Traps (all previously hit in this project — see research/review_tracker.md)

1. **`demo_ego_motion.py`'s default `--mean_std_path` is the WRONG stats dir**
   (`ava_human_nuscenes_waymo`, an old normalization). Always pass
   `ava_nuscenes_waymo` explicitly, as above. Mismatched stats produce silently degraded motion.
2. **Naming trap**: everything called `trans_dec` (run dir, config filename) actually contains a
   **trans_enc** (self-attention) model — a historical config-merge bug. The checkpoint and config
   are mutually consistent as-is; do NOT "fix" the config's arch before loading, and do not add
   `model.denoiser.params.arch=trans_dec` overrides — strict `load_state_dict` will fail.
3. **Config drift**: if a repo config disagrees with the dumped `config_*_train.yaml` in the run
   dir, the dumped one is the truth.
4. **OmegaConf CLI overrides YAML-parse their values**: bare `no`/`yes`/`off`/`on` become
   booleans. Quote them (`"key='no'"`).

## 5. Provenance chain (for the paper)
Figure 4 caption ↔ `make_qualitatives.py` ↔ `outputs/ego_demo/ego_motion_diffusion_h4_trans_dec/`
↔ `demo_ego_motion.py` with the artifacts of §1. The same checkpoint produced the main-table H4
row (FID 3.392) — eval side documented in `research/review_tracker.md`.
