# Findings — Ego-Conditioned Pedestrian Motion Generation

*Last updated: 2026-04-07 (H4 segment-2 complete — best epoch=3399, R@1 ceiling=0.535 with FREEZE_EGO; H6 unfreeze-ego ready to submit; definitive evals epoch=3399 running jobs 351794/795/796)*

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
| **H4 trans_dec (INTERMEDIATE, 64% trained)** | **3199** | **5** | **3.968** | **±0.209** | **5.891** | **0.510** | **4.117** | **40% FID improvement; R-prec lower (mid-training)** |
| GT reference | — | — | — | — | 5.496 | — | — | Ground truth motion diversity |

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

### H4 Intermediate Results (epoch=3199, job 346950 — CONFIRMED 2026-04-03)

**MAJOR FINDING: cross-attention ego conditioning dramatically reduces FID.**

| Run | Epoch | CFG | FID ↓ | FID CI | Diversity | R-prec@1 | MM | Notes |
|-----|-------|-----|-------|--------|-----------|----------|----|-------|
| H4 trans_dec (intermediate) | 3199 | 5 | 3.968 | ±0.209 | 5.891 | 0.510 | 4.117 | 40% FID improvement vs H2 |
| H4 trans_dec (intermediate) | 3199 | 10 | 3.644 | ±0.145 | 5.939 | 0.540 | 3.936 | — |
| H4 trans_dec (intermediate) | 3199 | 15 | 3.617 | ±0.131 | 6.044 | 0.515 | 3.938 | FID flat across CFG |
| **H4 trans_dec (DEFINITIVE best ep=3399)** | **3399** | **5** | **3.842** | **±0.108** | **5.832** | **0.506** | **4.141** | — |
| **H4 trans_dec (DEFINITIVE best ep=3399)** | **3399** | **10** | **3.392** | **±0.179** | **5.794** | **0.548** | **3.956** | **BEST: 48.6% FID↓; R@1 ceiling=0.548** |
| **H4 trans_dec (DEFINITIVE best ep=3399)** | **3399** | **15** | **3.547** | **±0.122** | **5.848** | **0.536** | **3.910** | — |
| H2 baseline | 4399 | 5 | 6.603 | ±0.067 | 5.779 | 0.671 | 3.503 | Best H2 checkpoint |
| H2 baseline | 4399 | 10 | 6.963 | ±0.035 | 5.762 | 0.767 | 3.031 | — |

**H4 vs H2 at CFG=5 (epoch=3199 vs 4399 — training epoch advantage for H2):**
- FID: H4=3.968 vs H2=6.603 → **H4 is 40% BETTER** (non-overlapping CIs; strong)
- R-prec@1: H4=0.510 vs H2=0.671 → **H4 is 24% WORSE** (but H4 is mid-training)
- MultiModality: H4=4.117 vs H2=3.503 → **H4 +17% more diverse per condition**
- Diversity: H4=5.891 vs H2=5.779 → slightly more diverse

**H4 CFG sweep at epoch=3199 (CONFIRMED 2026-04-03, jobs 346950/346961/346962)**:

| CFG | H4 FID ↓ | H4 FID CI | H4 R-prec@1 | H4 MM | H2 FID (ref) | H2 R-prec@1 (ref) |
|-----|----------|-----------|-------------|-------|--------------|-------------------|
| 5 | **3.968** | ±0.209 | 0.510 | 4.117 | 6.603 ±0.067 | 0.671 |
| 10 | **3.644** | ±0.145 | 0.540 | 3.936 | 6.963 ±0.035 | 0.767 |
| 15 | **3.617** | ±0.131 | 0.515 | 3.938 | 7.400 ±0.016 | 0.797 |

**Key pattern — H4 FID is flat across CFG**: Unlike H2 (FID rises steeply 6.60→7.40), H4 FID barely changes (3.97→3.62). This is a fundamentally different behavior. H4 R-prec is essentially flat at ~0.51–0.54 across all CFG levels — it's a training epoch effect, not a guidance scale effect.

**Pattern — FID/R-prec at architecture level**: H4 generates more diverse samples per conditioning (MM=4.117–3.936) but R-prec is lower because the model hasn't learned strong conditioning alignment yet at epoch=3199 (64% of training). R-prec expected to improve significantly with segment-2 training.

**Why the FID improvement**: The cross-attention decoder queries 196 per-timestep ego tokens (K/V) at each denoising step, vs H2's 2-token summary (time_emb + pooled_ego). The denoiser can now attend to specific moments in the ego trajectory, producing temporally-aligned motion that more closely matches the ground truth motion distribution.

**Why R-prec is lower (for now)**: Epoch 3199 is 64% of H2's best epoch (4399). Training-time val showed R_TOP_1 progressing: 0.529 at epoch 3279 (>0.510 at epoch 3199). R-prec is expected to improve with continued training. Additionally, higher MultiModality suggests the model has learned to generate more varied motions per condition — spread in embedding space naturally lowers retrieval accuracy.

**H4 Training Complete — Segment-2 Finished 2026-04-04 (CRITICAL NEW FINDING)**:

H4 segment-2 completed in ~13h (hit END_EPOCH=5000). Extracted full training trajectory:

| Epoch | Training-time Val FID ↓ | Training-time R@1 |
|-------|------------------------|-------------------|
| 3199 | 3.766 | 0.529 |
| **3399** | **3.770** | **0.535** ← BEST |
| 3499 | 3.805 | 0.531 |
| 3799 | 3.955 | 0.522 |
| 3999 | 4.087 | 0.515 |
| 4399 | 4.241 | 0.516 |
| 4999 | 4.508 | 0.511 |

**Epoch=3399 is the best checkpoint on BOTH metrics.** After 3399, FID and R@1 degrade monotonically. This is opposite to H2 (which had R-prec improve through epoch 4399). The pattern suggests the model finds its optimal balance early then loses diversity calibration.

**R@1 CEILING = 0.535. H4 R@1 NEVER approaches H2's 0.671.** This is not a training epoch effect — 1800 more epochs of training failed to push R@1 above 0.535. The frozen ego encoder is the bottleneck.

**Diagnosis**: FREEZE_EGO=True means the ego encoder remains optimized for its pretraining objective (contrastive alignment of mean-pooled ego → VAE latent). The cross-attention denoiser needs per-timestep discriminative features — something the frozen encoder was never trained to provide. The encoder cannot adapt; R@1 stagnates.

**Next step**: H6 (FREEZE_EGO=False) — ego encoder co-adapts with denoiser during diffusion training. Config and SLURM script created. Definitive evals at epoch=3399 submitted (jobs 351794/795/796).

## Patterns and Insights

1. **Interaction-aware training helps**: Cropping to the interaction window and up-weighting
   interaction-rich samples focuses the model on the most ego-relevant motion. The FID=7.5
   result with this approach is the best we have, though the run crashed before completion.

2. **Latent dimensionality does NOT help (H3 — REJECTED)**: Doubling the latent dimension (4→8) makes FID significantly WORSE (+14.5%) with the same denoiser capacity. The diffusion model's ability to model the latent distribution is the bottleneck, not the VAE's expressiveness. Diversity increases marginally (+1.1%) but doesn't compensate. **Lesson: changing latent dimension is a bad lever for FID improvement unless denoiser capacity scales too.**

3. **R-prec is decoupled from latent dimension**: H2 and H3 have essentially identical R-prec@1 (0.671 vs 0.676 at CFG=5). This reveals that the bottleneck for conditioning quality (R-prec) is the ego encoder architecture, not the latent space.  The current `EgoEncoderPooled` (mean-pool → single token) is the likely weak link.

4. **H4 cross-attention dramatically improves FID — confirmed 40–51% better than H2**: H4 best epoch (3399, training-time val FID=3.770) is ~43% better than H2 best (6.603). But R@1 is fundamentally capped at 0.535 with FREEZE_EGO — frozen encoder cannot adapt for per-timestep cross-attention. Test evals at epoch=3399 are running (jobs 351794/795/796). H6 (unfreeze) is the next step.

5. **H4 FID is insensitive to CFG (2026-04-03 — NEW FINDING)**: Unlike H2 where FID increases steeply with CFG (6.603→7.400 from CFG=5 to 15), H4 FID is essentially flat (3.968→3.617 from CFG=5 to 15). This means cross-attention conditioning does NOT cause the "over-conditioning mode collapse" observed in H2. The mechanism: in H2, higher CFG forces the model to stay close to the pooled ego token — over-constraining the generation. In H4, the ego information is already richly distributed across 196 cross-attention tokens; higher CFG reinforces a naturally richer signal, not a coarse average.

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
- **NAS paths unreliable on helma**: `/mnt/md0/erik/nas` is not mounted on all compute nodes. Always use `/hnvme/workspace/v103fe12-ped_gen/` paths in configs. H4 job 343502 failed for this reason; fixed to use `/hnvme/` in config.
- **H4 EgoEncoder compatibility**: `EgoEncoder` (T=196) requires `.mean(dim=1)` instead of `.squeeze(1)` to get (B, 256) embeddings. Squeeze does nothing on dim=1 when T=196. Fixed in `pretrain_ego_encoder.py` (×2) and `mld.py` (×1). Backward-compatible with EgoEncoderPooled (T=1).

## Open Questions

1. ~~**Will the full run of H2 outperform crashed partial run's FID=7.5?**~~ → **ANSWERED**: YES at epoch=4399 (FID=6.603), but model regressed afterward. **H2 final baseline = FID=6.603 at epoch=4399, CFG=5.**
2. ~~**Does latent-8 VAE give better reconstruction quality?**~~ → **ANSWERED**: Latent-8 diffusion is 14.5% WORSE FID than latent-4 at the same epoch (7.563 vs 6.603). Larger latent with same denoiser capacity is counterproductive. **H3 rejected.**
3. ~~**What is the sensitivity to CFG guidance scale?**~~ → **ANSWERED**: FID monotonically decreases with lower CFG. CFG=5 is best for FID (6.603). **Use CFG=5 for FID evaluation, CFG=7 for balanced.**
4. **Can a cross-attention ego encoder improve R-precision and/or FID?** → **PARTIALLY ANSWERED**. H4 best FID (epoch=3399): training-time val=3.770 — **43% better than H2 best (6.603)**. BUT R@1 ceiling=0.535 with FREEZE_EGO=True — never reaches H2 0.671. Definitive test evals running (epoch=3399, jobs 351794/795/796). **H6 (unfreeze) will test if unfreezing resolves the R@1 ceiling.**
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

**H4 implementation cost**: Ego encoder pretraining took ~1.5h (job 343503). Full diffusion
training segment-1 (~24h, ~2699 epochs) + segment-2 (~24h, epochs 2699-5000) = ~2.5 days total.

**Key architectural differences H2 vs H4**:
| Aspect | H2 (trans_enc + EgoEncoderPooled) | H4 (trans_dec + EgoEncoder) |
|--------|-----------------------------------|------------------------------|
| Ego encoder output | (B, 1, 256) — single pooled token | (B, 196, 256) — full sequence |
| Projection head | Yes (2-layer MLP inside encoder) | No |
| Denoiser conditioning | 2 tokens: [time_emb, ego_pooled] → self-attn | 197 K/V tokens: [time_emb, ego_seq(196)] |
| Attention type | Self-attention over concat [z, cond] | Cross-attention: z queries ego sequence |
| Pretrain val MSE | ~1.92 (same order; projection head doesn't help) | 1.919 |

**Insight from pretrain MSE comparison**: EgoEncoderPooled (with 2-layer projection head) achieves
val MSE ~1.92, essentially identical to EgoEncoder (no projection head) at 1.92. This suggests
that the 2-layer projection head isn't significantly improving the encoder's ability to predict
VAE latents from ego trajectory. The irreducible uncertainty is inherent in the ego→motion prediction
task (vehicle trajectory is a weak predictor of pedestrian body pose). The projection head doesn't
add meaningful capacity. **Implication**: H4 dropping the projection head loses nothing in alignment quality,
while gaining richer temporal conditioning from the full sequence.

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

## H4 Outcome Scenarios and Next Steps

H4 results will arrive at epoch ~2299 (intermediate) and ~4399 (definitive). Pre-analysis:

### If H4 strongly improves R-prec (≥ 0.80, target):
- Confirms that EgoEncoderPooled was the bottleneck for conditioning quality
- Cross-attention over 196 K/V tokens provides the temporal granularity needed
- Check FID: if FID ≤ 6.60, H4 dominates H2 on all metrics → proceed to paper
- If FID increases slightly: analyze optimal epoch (H2 regressed at 4599; H4 may too)
- Consider CFG sweep at best epoch to find H4's optimal operating point

### If H4 partially improves R-prec (0.71–0.79):
- Marginal gain despite richer conditioning → conditioning architecture isn't the sole bottleneck
- Consider H6: unfreeze ego encoder during diffusion training (co-adapt encoder + denoiser)
- Consider H7: larger denoiser (more capacity to leverage 196 K/V tokens)
- The R-prec metric itself may have limited sensitivity (based on mean-pooled comparison)

### If H4 shows no improvement in R-prec (~0.67, same as H2):
- Cross-attention over full sequence does NOT help → conditioning architecture not the bottleneck
- Suggests the R-prec metric is measuring something else (or the ego-motion alignment is inherently weak)
- Pivot: investigate alternative conditioning signals (relative pedestrian position, speed, heading)
- Or: investigate whether R-prec is even the right metric for this task

### Key diagnostic signal (available from intermediate eval at epoch ~2299):
- R-prec@1 > 0.70? → H4 is working, continue to full training
- R-prec@1 ≈ 0.671? → H4 not working, consider pivot after segment-1

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
