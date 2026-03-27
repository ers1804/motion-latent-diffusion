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
