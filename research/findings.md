# Findings — Ego-Conditioned Pedestrian Motion Generation

*Last updated: 2026-03-27*

## Current Understanding

We are training a Motion Latent Diffusion (MLD) model to generate realistic pedestrian body motion
conditioned on a vehicle's ego trajectory. The core hypothesis is that pedestrian motion is
highly influenced by the ego vehicle's proximity and speed, so conditioning on ego trajectory
should enable realistic, contextually-grounded motion synthesis.

**Architecture**: The ego trajectory (B, T, 2) is encoded by `EgoEncoderPooled` (transformer +
mean pooling → 256d) and injected into the MLD denoiser via cross-attention (same pathway as
text conditioning in the original MLD). The VAE compresses 263D HumanML3D motion to a
latent-4×256 space.

## Key Results So Far

| Run | Epoch | FID ↓ | Diversity | R-prec@1 | MM | Notes |
|-----|-------|-------|-----------|----------|----|-------|
| interaction_crop_weighted_1 (crashed) | ~mid | 7.503 | 5.665 | 0.825 | — | Latent-4, interaction crop+weighted; crashed |
| **interaction_crop_weighted_1_helma** | **4399** | **7.400 ±0.02** | **5.742** | **0.797** | **2.724** | **Continued run; FID improved; 500 epochs remaining** |
| GT reference | — | — | 5.330 | — | — | Ground truth motion diversity |

**Key finding (2026-03-27)**: The full H2 run at epoch 4399 achieves FID=7.40, improving on the crashed partial run's 7.50. R-precision @1 is 0.797 (slightly below crashed 0.825), suggesting model has room to improve on conditioning alignment in the remaining 500 epochs. Diversity at 5.74 is slightly above GT (5.33), indicating mild overgeneration but not mode collapse.

## Patterns and Insights

1. **Interaction-aware training helps**: Cropping to the interaction window and up-weighting
   interaction-rich samples focuses the model on the most ego-relevant motion. The FID=7.5
   result with this approach is the best we have, though the run crashed before completion.

2. **Latent dimensionality matters**: The project is actively exploring latent-4 vs latent-8.
   The latent-8 VAE uses KL annealing and cosine LR warmup — suggesting the team learned
   that larger latent spaces need more careful training to avoid posterior collapse.

3. **Guidance scale**: Currently at 15 (high). This was increased from 7.5 in earlier runs.
   Higher CFG generally improves quality at the cost of diversity — worth sweeping.

## Lessons and Constraints

- **Interaction crop + weighted sampling** must be enabled consistently between VAE and diffusion
  training (both currently use it)
- **Mean/Std path** must match between VAE training and diffusion training (data normalization
  is a known sharp edge in this codebase)
- **Latent dim** must match between VAE and diffusion model configs
- VAE loss around 0.015 indicates good convergence (latent-8 VAE at epoch 5700)
- Diffusion loss around 0.30 is typical mid-training for this setup

## Open Questions

1. ~~**Will the full run of H2 outperform crashed partial run's FID=7.5?**~~ → YES: epoch 4399 gives FID=7.40. Final eval at epoch 5000 pending.
2. **Does latent-8 VAE give better reconstruction quality?** → VAE done (loss=0.0142). Ego encoder pretraining running. Diffusion training TBD.
3. **What is the sensitivity to CFG guidance scale?** → Sweep {5,10,15,20} ready to submit (`slurm/eval_cfg_sweep.sh`).
4. **Can a cross-attention ego encoder improve R-precision?** → Untested (H4). Current R-prec@1=0.797 leaves room for improvement.
5. **Is there a meaningful gap vs retrieval baseline?** → ADE/FDE evaluation not yet set up.
6. **Will the H3 latent-8 diffusion outperform H2?** → VAE+encoder pipeline underway; results expected in ~2-3 days.

## Related Work (to be filled via literature search)

*See literature/ for paper summaries*
