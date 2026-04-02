# Findings — Ego-Conditioned Pedestrian Motion Generation

*Last updated: 2026-04-01 (H3 definitive eval — latent-8 rejected; H4 direction set)*

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

### H3 Definitive Results (epoch=4399, jobs 333132/333133)

**HYPOTHESIS REJECTED: latent-8 is significantly worse than latent-4.**

| Run | Epoch | CFG | FID ↓ | FID CI | Diversity | R-prec@1 | MM | Notes |
|-----|-------|-----|-------|--------|-----------|----------|----|-------|
| H3 latent-8 (intermediate) | 2299 | 5 | 7.475 | ±0.048 | 5.718 | 0.576 | 3.631 | 52% of H2 best epoch |
| **H3 latent-8 (definitive)** | **4399** | **5** | **7.563** | **±0.078** | **5.842** | **0.676** | **3.245** | **Same epoch as H2 best** |
| H3 latent-8 (definitive) | 4399 | 7 | 8.131 | ±0.103 | 5.830 | 0.733 | 2.989 | H2 comparison at CFG=7 |
| **H2 latent-4 (best baseline)** | **4399** | **5** | **6.603** | **±0.067** | **5.779** | **0.671** | **3.503** | **H2 best** |
| H2 latent-4 | 4399 | 7 | 6.716 | ±0.068 | 5.771 | 0.724 | 3.291 | H2 at CFG=7 |

**H3 vs H2 comparison at identical epoch=4399:**
- CFG=5: H3 FID=7.563 vs H2 FID=6.603 → **H3 is 14.5% WORSE** (significant, non-overlapping CIs)
- CFG=7: H3 FID=8.131 vs H2 FID=6.716 → **H3 is 21.1% WORSE**
- R-prec@1 (CFG=5): H3=0.676 vs H2=0.671 → essentially identical (+0.7%, within CI)
- Diversity: H3=5.842 vs H2=5.779 → slightly higher (+1.1%), consistent with larger latent space

**Why latent-8 failed**: Same-capacity diffusion model must now model an 8D latent distribution instead of 4D. The denoiser (fixed transformer architecture) appears insufficient for the harder task of modeling 8-dimensional latents. The marginal diversity improvement (+1.1%) does not compensate for the FID degradation. R-prec being essentially equal shows that the latent dimension does NOT affect conditioning quality — the bottleneck for R-prec is elsewhere (likely the ego encoder architecture, not the latent size).

## Patterns and Insights

1. **Interaction-aware training helps**: Cropping to the interaction window and up-weighting
   interaction-rich samples focuses the model on the most ego-relevant motion. The FID=7.5
   result with this approach is the best we have, though the run crashed before completion.

2. **Latent dimensionality does NOT help (H3 — REJECTED)**: Doubling the latent dimension (4→8) makes FID significantly WORSE (+14.5%) with the same denoiser capacity. The diffusion model's ability to model the latent distribution is the bottleneck, not the VAE's expressiveness. Diversity increases marginally (+1.1%) but doesn't compensate. **Lesson: changing latent dimension is a bad lever for FID improvement unless denoiser capacity scales too.**

3. **R-prec is decoupled from latent dimension**: H2 and H3 have essentially identical R-prec@1 (0.671 vs 0.676 at CFG=5). This reveals that the bottleneck for conditioning quality (R-prec) is the ego encoder architecture, not the latent space.  The current `EgoEncoderPooled` (mean-pool → single token) is the likely weak link.

4. **Guidance scale (H5 — COMPLETED)**: CFG has a large monotonic effect on FID — **lower is better for FID**.
   CFG=5 achieves FID=6.603 vs FID=7.400 at CFG=15 (10.8% improvement). Diversity barely changes (5.74–5.78 across all scales).
   R-precision monotonically increases with CFG (0.671 at CFG=5 → 0.815 at CFG=20), so there is a genuine quality/conditioning tradeoff.
   This finding falsifies the prior assumption that FID would be optimal at intermediate CFG — the relationship is monotone in this regime.
   **Recommendation**: Use CFG=5 for FID-focused evaluation; CFG=7 for balanced FID/R-prec.

5. **FID does not monotonically improve with training**: H2 epoch trajectory peaked at 4399 (FID=6.603), regressed at 4599 (FID=8.400), partially recovered at 4999 (FID=7.510). Training loss was flat — FID oscillates independently. **Critical lesson: checkpoint selection with periodic FID validation is required; the final checkpoint is NOT the best checkpoint.**

## Lessons and Constraints

- **Interaction crop + weighted sampling** must be enabled consistently between VAE and diffusion training (both currently use it)
- **Mean/Std path** must match between VAE training and diffusion training (data normalization is a known sharp edge in this codebase)
- **Latent dim** must match between VAE and diffusion model configs
- VAE loss around 0.015 indicates good convergence (latent-8 VAE at epoch 5700)
- **FID does not monotonically improve with training epochs** — sampling quality oscillates independently of training loss. Early stopping / checkpoint selection is critical.
- Running two training jobs from the same checkpoint in parallel causes checkpoint name conflicts (PyTorch Lightning appends "-v1") — avoid duplicate job submissions.
- Diffusion loss around 0.30 is typical mid-training for this setup.
- **Increasing latent dim degrades FID without increasing denoiser capacity** — H3 latent-8 is 14.5% worse FID than H2 latent-4 at the same epoch. Do not change latent dim as a primary lever.
- **CFG=7 is the best operating point** for balanced FID/R-prec evaluation (FID=6.716, R-prec=0.724). CFG=5 for FID-only comparisons (FID=6.603).
- **Helma submit order**: always `git push` locally BEFORE `ssh helma "... sbatch"` — helma pulls before the push if done wrong order (caused job 330924 to fail with wrong checkpoint).

## Open Questions

1. ~~**Will the full run of H2 outperform crashed partial run's FID=7.5?**~~ → **ANSWERED**: YES at epoch=4399 (FID=6.603), but model regressed afterward. **H2 final baseline = FID=6.603 at epoch=4399, CFG=5.**
2. ~~**Does latent-8 VAE give better reconstruction quality?**~~ → **ANSWERED**: Latent-8 diffusion is 14.5% WORSE FID than latent-4 at the same epoch (7.563 vs 6.603). Larger latent with same denoiser capacity is counterproductive. **H3 rejected.**
3. ~~**What is the sensitivity to CFG guidance scale?**~~ → **ANSWERED**: FID monotonically decreases with lower CFG. CFG=5 is best for FID (6.603). **Use CFG=5 for FID evaluation, CFG=7 for balanced.**
4. **Can a cross-attention ego encoder improve R-precision and/or FID?** → Untested (H4). H3 showed R-prec@1 barely changes with different latent dims (0.671 vs 0.676). The ego encoder architecture (mean-pooling → single token) is the suspected bottleneck. H4 would use `trans_dec` cross-attention with full T=196 ego sequence.
5. **What is the true best FID possible with this dataset and architecture?** → H2 FID=6.603 is current best. GT diversity=5.330 vs generated=5.779 — generated motion is slightly over-diverse, which could be a source of FID.
6. **Is there a meaningful gap vs retrieval baseline?** → ADE/FDE evaluation not yet set up.

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
