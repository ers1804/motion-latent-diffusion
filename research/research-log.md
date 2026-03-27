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
