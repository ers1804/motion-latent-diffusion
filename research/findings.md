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

## Architecture Analysis

### Current Denoiser Architecture (trans_enc)
The MLD denoiser uses a transformer encoder (`trans_enc` arch) where the latent `z` sequence
and the conditioning tokens are concatenated along the sequence dimension:
```
xseq = cat(sample [L, B, D], emb_latent [2, B, D])  # L latent tokens + 2 condition tokens
```
Attention is then self-attention over `xseq` of length L+2.

With `EgoEncoderPooled`, the ego trajectory (196 timesteps) is compressed to a single token (B, 1, D)
before being added to `emb_latent`. So the denoiser only "sees" a single averaged ego representation.

### H4 Design: Cross-Attention Decoder (trans_dec)
For full temporal ego conditioning, the efficient approach is cross-attention:
- **Query**: motion latent z of length L (4 or 8)
- **Key/Value**: full ego sequence of length T=196
- **Complexity**: O(L × T) vs O((L+T)²) for self-attention with full sequence

This would use `arch=trans_dec` (already exists in the denoiser) with:
1. Change `EgoEncoderPooled` → `EgoEncoder` (return full sequence)
2. Use `trans_dec` arch so z queries the ego sequence via cross-attention
3. Expected benefit: better temporal alignment — R-precision should improve

**Practical constraint**: The pretrained ego encoder weights (from contrastive pretraining)
are for the `EgoEncoderPooled` arch. H4 would require re-pretraining the ego encoder +
re-training the diffusion model. Cost: ~2-3 days on H100.

### CFG Sweep Predictions (H5)
For guidance_scale sweep {5, 10, 15, 20}:
- **Diversity**: monotone decrease with higher CFG (generation concentrates on high-likelihood modes)
- **FID**: typically optimal at intermediate CFG (inverted U-curve; both extremes hurt FID)
- **R-precision**: monotone increase with higher CFG (guidance forces conditioning alignment)
- **Current**: CFG=15 gives FID=7.40, R-prec@1=0.797, Diversity=5.74
- **Prediction**: FID minimum around CFG=10-15; R-prec may be higher at CFG=20 but diversity drops

## Related Work (see literature/)

See `research/literature/survey_ego_motion_generation.md` — 28 papers across:
1. Ego/context-conditioned pedestrian motion (7 papers)
2. Interaction-aware human motion generation (5 papers)
3. Latent diffusion for human motion (8 papers)
4. Classifier-free guidance for motion generation (5 papers)
5. VAE latent space design (5 papers)

**Key finding**: This work is the first to combine full 3D body motion generation (HumanML3D 263D)
with vehicle ego odometry conditioning — previously, closest works either (a) generated 2D
trajectories from ego context, or (b) generated 3D body motion from text/action labels.

**Most related**: WoSAD (2024) — ego odometry → polyline encoder → diffusion denoiser with
cross-attention, but 2D trajectory output only. UniTraj (ECCV 2024) — multi-dataset AV training
with ego odometry, same AVA+nuScenes+Waymo datasets, but vehicle/agent trajectory prediction.
