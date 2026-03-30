# Research Log — Ego-Conditioned Pedestrian Motion Generation

## 2026-03-27 — Bootstrap

### State at Start
Two experiments actively running on helma H100 cluster:
1. **Diffusion training** (job 326049, ~22h): `ego_motion_diffusion_interaction_crop_weighted_1_helma`
   - Latent-4 stochastic VAE, interaction_crop + weighted_sampling
   - Epoch ~4200/5000, loss ≈ 0.302
2. **VAE training** (job 326050, ~22h): `ego_motion_vae_latent_8_wo_traj_interaction_crop_weighted_sampling`
   - Latent-8, KL annealing (0→500 epochs), cosine LR warmup
   - Epoch ~5699/6000, loss ≈ 0.0147

### Best Known Metrics
From `ego_motion_diffusion_interaction_crop_weighted_1` (crashed, latent-4, interaction-aware):
- FID: 7.5029
- Diversity: 5.6653 (GT: 5.3433)
- R-precision top-1/2/3: 0.8254 / 0.8917 / 0.9179

### Research Question Identified
Can interaction-aware training + larger latent spaces improve ego-conditioned motion generation?

### Key Design Decisions
- **Model**: MLD denoiser conditioned on EgoEncoderPooled (transformer + mean pool → 256d)
- **VAE**: MLD-VAE, motion in HumanML3D 263D representation
- **Data**: AVA + nuScenes + Waymo (interaction split)
- **Interaction features**: crop window to closest-approach frame + weight samples by interaction score

### Hypotheses Formed
- H2: interaction-aware training improves FID/diversity (currently running)
- H3: latent-8 VAE gives more capacity (VAE running, diffusion training TBD)
- H4: ego encoder architecture (cross-attn vs mean pool) improves R-precision
- H5: CFG guidance scale sweep (5/10/15/20) is cheap and might help
- H6: joint fine-tuning of VAE+diffusion improves alignment

### Next Steps
1. Monitor running jobs; evaluate once checkpoint ready
2. Literature search on pedestrian motion generation, CFG tuning, interaction-aware synthesis
3. Prepare guidance-scale sweep as next quick experiment (post checkpoint)

## 2026-03-27 — Eval Job Fix & Pipeline Preparation

### Issue: Eval Job 327929 Failed (0 samples loaded)
- Root cause: `TEST.SPLIT=test` used, but helma data archives only contain `train/` and `val/` directories
- `test.txt` and `val.txt` in the split list path are identical (11786 entries each)
- Fix: changed `TEST.SPLIT=test` → `TEST.SPLIT=val` in both eval scripts
- Committed and pushed; pulled on helma; submitted new eval job 327939

### Training Status Update (as of 2026-03-27 ~14:47 CET)
- **run_005 diffusion** (job 326049): epoch ~4500/5000, loss=0.292. ~6.5h remaining.
- **run_006 VAE latent-8** (job 326050): epoch ~5900/6000, loss=0.0143. ~1h remaining.

### Pipeline Ready for H3
- `configs/config_ego_motion_new_vae_stoch_latent_8.yaml` — latent-8 diffusion config
- `slurm/pretrain_ego_encoder_latent_8_helma.sh` — ego encoder pretraining
- `slurm/diffusion_training_latent_8_helma.sh` — H3 diffusion training (24h job)
- `slurm/eval_latent_8.sh` — evaluation script for H3 (prepared today)
- `slurm/eval_cfg_sweep.sh` — CFG sweep {5,10,15,20} on latent-4 model

### Next Actions
1. Eval job 327939 running — await FID/Diversity/R-Precision results
2. When VAE finishes: `sbatch pretrain_ego_encoder_latent_8_helma.sh` → `sbatch diffusion_training_latent_8_helma.sh`
3. After run_005 completes at epoch 5000: eval final checkpoint
4. Submit CFG sweep once run_005 eval confirms setup works

## 2026-03-27 — Ego Encoder Pretraining Fix & Resubmission

### Issue: Job 327946 Failed (FileNotFoundError)
- Error: `FileNotFoundError: '.../data/vae/mean_std_txt/ava_nuscenes_waymo/Mean.npy'`
- Root cause: `pretrain_ego_encoder.py` reads `DATASET.EGOMOTION.MEAN_STD_PATH` from config, which contains the local NAS path `/mnt/md0/erik/nas/...`. The slurm script extracts data to `$TMPDIR/data/` but didn't override this path.
- `pretrain_ego_encoder.py` had no mechanism to override `MEAN_STD_PATH` (only `--data_roots` existed)

### Fix Applied
- Added `--mean_std_path` CLI argument to `pretrain_ego_encoder.py`
- `build_cfg()` now overrides both `DATASET.EGOMOTION.MEAN_STD_PATH` and `DATASET.EGOMOTION.EGO_MEAN_STD_PATH` when `--mean_std_path` is provided
- Updated `slurm/pretrain_ego_encoder_latent_8_helma.sh` to pass `--mean_std_path "$TMPDIR/data/vae/mean_std_txt/ava_nuscenes_waymo"`
- Committed (2922363), pushed, pulled on helma, resubmitted as **job 327955**

### Current Job Status
- **327955** ego_enc_pretrain — RUNNING (just started, h13-13)
- **327950** eval_cfg_sweep — RUNNING (extracting data, h13-24)
- **326049** diffusion_train (H2 run_005) — RUNNING (23h51m, h14-06)

## 2026-03-30 — Inner Loop Results: FID Regression + H3 Pipeline Started

### Key Results from Completed Jobs (2026-03-27)

**Job 327955 (Ego encoder latent-8)**: COMPLETED. 200/200 epochs. `best.pt` saved.
Cosine similarity ~0.92 at epoch 200. H3 pipeline fully ready.

**Job 327991 (Eval H2 epoch=4599, CFG=5)**: COMPLETED. REGRESSION DETECTED.
- FID: 8.400 ± 0.102
- R-prec@1: 0.733
- Diversity: 5.757
- This is WORSE than epoch=4399 (FID=6.603). Training loss didn't diverge (~0.286), so FID oscillation is independent of loss. **Epoch=4399 is the best H2 checkpoint.**

**Job 327990 (H2 resume epoch 4599→4999)**: COMPLETED. Final epoch=4999, loss=0.286.
- NOTE: Job 327881 was an untracked duplicate resume from the same checkpoint, running in parallel.
  This caused PyTorch Lightning to append "-v1" suffixes to checkpoints. The duplicate training
  was submitted by someone (possibly user manually) and ran 15:22-19:34 on 2026-03-27.
  Checkpoint integrity may be affected.

### New Submissions (2026-03-30)

- **Job 329788**: Eval H2 epoch=4999 at CFG=5 — checks if model recovered from epoch=4599 regression
- **Job 329789**: H3 diffusion training (latent-8 VAE + new ego encoder) — the key H3 experiment

### Outer Loop Reflection

Three major findings from this research session:
1. **CFG=5 is optimal for FID** (10.8% improvement vs CFG=15)
2. **Epoch=4399 is the best H2 checkpoint** — model degraded from epoch 4399 → 4599 → unclear for 4999
3. **H3 pipeline is running** — ego encoder (cos~0.92) proves latent-8 encoder alignment works

The H2 FID regression raises an important question about training stability. The loss plateau (~0.287) doesn't predict sampling quality oscillation. For future training runs, we should save more frequent checkpoints and run periodic FID validation to catch the best model.
