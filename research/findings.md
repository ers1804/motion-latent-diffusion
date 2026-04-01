# Findings — Ego-Conditioned Pedestrian Motion Generation

*Last updated: 2026-03-30 (H2 epoch=4999 eval — FID regression confirmed)*

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

| Run | Epoch | CFG | FID ↓ | FID CI | Diversity | R-prec@1 | MM | Notes |
|-----|-------|-----|-------|--------|-----------|----------|----|-------|
| interaction_crop_weighted_1 (crashed) | ~mid | 15 | 7.503 | — | 5.665 | 0.825 | — | Latent-4, interaction crop+weighted; crashed |
| interaction_crop_weighted_1_helma | 4399 | 20 | 7.856 | ±0.041 | 5.759 | 0.815 | 2.573 | CFG sweep |
| interaction_crop_weighted_1_helma | 4399 | 15 | 7.400 | ±0.016 | 5.742 | 0.797 | 2.724 | Default CFG |
| interaction_crop_weighted_1_helma | 4399 | 10 | 6.963 | ±0.035 | 5.762 | 0.767 | 3.031 | CFG sweep |
| **interaction_crop_weighted_1_helma** | **4399** | **7** | **6.716** | **±0.068** | **5.771** | **0.724** | **3.291** | **Best FID/R-prec tradeoff** |
| **interaction_crop_weighted_1_helma** | **4399** | **5** | **6.603** | **±0.067** | **5.779** | **0.671** | **3.503** | **Best FID — H2 BASELINE** |
| interaction_crop_weighted_1_helma | 4599 | 5 | 8.400 | ±0.102 | 5.757 | 0.733 | 3.278 | REGRESSION after epoch 4399 |
| interaction_crop_weighted_1_helma | 4999 | 5 | 7.510 | ±0.084 | 5.816 | 0.671 | 3.613 | Partial recovery — NOT back to best |
| GT reference | — | — | — | — | 5.330 | — | — | Ground truth motion diversity |

**MAJOR FINDING (2026-03-27 — H5 CFG Sweep, completed 2026-03-30)**: CFG guidance scale has a large, monotonic effect on FID. The original default (CFG=15) was suboptimal — **CFG=5 gives FID=6.603, a 10.8% improvement** (7.40→6.60) at zero training cost. The full sweep including CFG=7:

| CFG | FID | R-prec@1 | Notes |
|-----|-----|----------|-------|
| 5 | **6.603** | 0.671 | Best FID |
| **7** | **6.716** | **0.724** | **Best FID/R-prec tradeoff** (+1.7% FID, +7.9% R-prec vs CFG=5) |
| 10 | 6.963 | 0.767 | — |
| 15 | 7.400 | 0.797 | Prior default |
| 20 | 7.856 | 0.815 | — |

**CFG=7 is the recommended operating point** for balanced evaluation — minimal FID cost (+1.7%) with substantially better conditioning alignment (+7.9% R-prec). CFG=5 is best when FID alone is the target metric.

**Implication**: Future evaluations should use **CFG=5** to report FID. **Best H2 checkpoint is epoch=4399** (FID=6.603).

**FINDING (2026-03-30 — FID Regression CONFIRMED)**: H2 epoch trajectory: 4399→FID=6.603 (best), 4599→FID=8.400 (regression), 4999→FID=7.510 (partial recovery, but NOT back to best). The model degraded and never fully recovered. Training loss was flat (~0.286–0.287) throughout — **sampling quality oscillates independently of training loss**. Early stopping with periodic FID validation is critical. **H2 final baseline: FID=6.603 at epoch=4399, CFG=5.**

### H3 Intermediate Eval (epoch=2299, job 331157)

| Run | Epoch | CFG | FID ↓ | FID CI | Diversity | R-prec@1 | MM | Notes |
|-----|-------|-----|-------|--------|-----------|----------|----|-------|
| H3 latent-8 (intermediate) | 2299 | 5 | 7.475 | ±0.048 | 5.718 | 0.576 | 3.631 | Segment-1 final; only 52% of H2 best epoch |

**H3 at epoch=2299 is undertrained** — this is expected. At epoch=2299, H3 has seen only 52% of the training that H2 needed to reach its best performance (epoch=4399). FID=7.475 is 13.2% worse than H2's best (6.603), but no strong conclusion can be drawn from this. The H3 definitive eval at epoch=4399 (submitted as jobs 333132/333133 on 2026-04-01) will provide the fair comparison.

**Compared to H2 at similar epochs**:
- H2 at epoch=4399 (segment-1 equivalent for H3): FID=6.603
- H3 at epoch=2299 (52% of H2's best): FID=7.475
- Trajectory suggests H3 may catch up or surpass H2 with more training.

## Patterns and Insights

1. **Interaction-aware training helps**: Cropping to the interaction window and up-weighting
   interaction-rich samples focuses the model on the most ego-relevant motion. The FID=7.5
   result with this approach is the best we have, though the run crashed before completion.

2. **Latent dimensionality matters**: The project is actively exploring latent-4 vs latent-8.
   The latent-8 VAE uses KL annealing and cosine LR warmup — suggesting the team learned
   that larger latent spaces need more careful training to avoid posterior collapse.

3. **Guidance scale (H5 — COMPLETED)**: CFG has a large monotonic effect on FID — **lower is better for FID**.
   CFG=5 achieves FID=6.603 vs FID=7.400 at CFG=15 (10.8% improvement). Diversity barely changes (5.74–5.78 across all scales).
   R-precision monotonically increases with CFG (0.671 at CFG=5 → 0.815 at CFG=20), so there is a genuine quality/conditioning tradeoff.
   This finding falsifies the prior assumption that FID would be optimal at intermediate CFG — the relationship is monotone in this regime.
   **Recommendation**: Use CFG=5 for FID-focused evaluation; CFG=10–15 if conditioning fidelity (R-prec) matters more.

## Lessons and Constraints

- **Interaction crop + weighted sampling** must be enabled consistently between VAE and diffusion
  training (both currently use it)
- **Mean/Std path** must match between VAE training and diffusion training (data normalization
  is a known sharp edge in this codebase)
- **Latent dim** must match between VAE and diffusion model configs
- VAE loss around 0.015 indicates good convergence (latent-8 VAE at epoch 5700)
- **FID does not monotonically improve with training epochs** — sampling quality oscillates independently of training loss. Early stopping / checkpoint selection is critical for final FID.
- Running two training jobs from the same checkpoint in parallel causes checkpoint name conflicts (PyTorch Lightning appends "-v1") — avoid duplicate job submissions.
- Diffusion loss around 0.30 is typical mid-training for this setup.
- **Latent-8 training is ~3x slower per epoch** than latent-4: H3 trains at ~51.6s/epoch vs H2's ~17.6s/epoch. At 24h wall time, H3 only reaches ~1674 epochs. Reaching H2's best epoch (4399) requires ~3× 24h job segments. Plan resume jobs proactively.
- **CFG=7 is the best operating point** for balanced FID/R-prec evaluation (FID=6.716, R-prec=0.724). CFG=5 for FID-only comparisons (FID=6.603).

## Open Questions

1. ~~**Will the full run of H2 outperform crashed partial run's FID=7.5?**~~ → **ANSWERED**: YES at epoch=4399 (FID=6.603), but model regressed afterward. Epoch trajectory: 4399→6.603 (best), 4599→8.400 (regression), 4999→7.510 (partial recovery). **H2 final baseline = FID=6.603 at epoch=4399, CFG=5.**
2. **Does latent-8 VAE give better reconstruction quality?** → VAE done (loss=0.0142). Ego encoder DONE (200 epochs, cos~0.92). **H3 diffusion training started (job 329789, 2026-03-30).**
3. ~~**What is the sensitivity to CFG guidance scale?**~~ → **ANSWERED**: FID monotonically decreases with lower CFG. CFG=5 is best for FID (6.603), CFG=20 worst (7.856). R-prec monotonically improves with higher CFG. Diversity is nearly constant. **Use CFG=5 for FID evaluation.**
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

### CFG Sweep Results (H5 — COMPLETED)

Evaluated H2 epoch=4399 checkpoint at CFG ∈ {5, 10, 15, 20}:

| CFG | FID | FID CI | Diversity | R-prec@1 | MultiModality |
|-----|-----|--------|-----------|----------|---------------|
| 5 | **6.603** | ±0.067 | 5.779 | 0.671 | **3.503** |
| 10 | 6.963 | ±0.035 | 5.762 | 0.767 | 3.031 |
| 15 | 7.400 | ±0.016 | 5.742 | 0.797 | 2.724 |
| 20 | 7.856 | ±0.041 | 5.759 | **0.815** | 2.573 |

**What we predicted vs what happened**:
- Diversity: predicted monotone decrease → **WRONG**: nearly constant (5.74–5.78 all scales)
- FID: predicted inverted-U (optimal at intermediate) → **WRONG**: monotone decrease, CFG=5 best
- R-precision: predicted monotone increase → **CORRECT**
- MultiModality: increases at lower CFG → consistent with less-constrained generation

**Mechanism hypothesis**: In this model, CFG guidance primarily constrains the output distribution toward conditioning-aligned samples, reducing FID by allowing more diverse motion generation rather than "quality polishing." The ego conditioning signal may be weak enough that high CFG causes over-constraint (mode collapse toward a few conditioning-aligned modes), hurting FID.

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
