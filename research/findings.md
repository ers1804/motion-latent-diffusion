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

| Run | FID ↓ | Diversity | R-prec@1 | Notes |
|-----|-------|-----------|----------|-------|
| ego_motion_diffusion_interaction_crop_weighted_1 (crashed) | **7.50** | 5.67 | 0.825 | Latent-4, interaction crop+weighted |

This is the only run with recorded metrics. The interaction-aware approach seems promising.

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

1. **Will the full run of H2 (interaction crop + weighted, latent-4) outperform the crashed
   partial run's FID=7.5?** (job 326049 running now)
2. **Does latent-8 VAE give better reconstruction quality that propagates to better generation?**
   (VAE job 326050 running, diffusion training will follow)
3. **What is the sensitivity to CFG guidance scale?** 15 seems high — sweep needed.
4. **Can a cross-attention ego encoder (vs mean pooling) improve R-precision** by giving the
   denoiser access to the full ego trajectory at each decoding step?
5. **Is there a meaningful gap between our model and retrieval baseline** on interaction-specific
   metrics like ADE/FDE w.r.t. ego trajectory?

## Related Work (to be filled via literature search)

*See literature/ for paper summaries*
