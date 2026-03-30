# H4 Protocol: Cross-Attention Ego Encoder

**Status**: Planned (pending H3 results)
**Locked**: 2026-03-30
**Prerequisite**: H3 result — if H3 fails to beat H2, H4 should still be attempted as the R-prec issue exists independently

## Hypothesis

The current `EgoEncoderPooled` architecture compresses 196 ego timesteps into a **single pooled token** before
conditioning the denoiser. This loses all temporal structure in the ego trajectory. The denoiser's cross-attention
sees only a mean-aggregated ego representation, which may explain why R-precision (ego↔motion alignment) is low
(0.671 at CFG=5, 0.797 at CFG=15 — both worse than R-prec=0.825 seen from the crashed run_004).

**Prediction**: Using cross-attention over the full ego sequence (EgoEncoder without pooling → trans_dec arch)
will improve R-precision by preserving temporal correspondence between ego trajectory moments and motion phases.
Expected: R-prec@1 ≥ 0.80 with comparable or better FID vs H2 (FID ≤ 6.60 target).

## Why This Should Work

The MLD denoiser already supports `trans_dec` (cross-attention decoder) architecture. In `trans_dec`:
- **Query**: motion latent z of shape (L, B, D) where L=4 or L=8
- **Key/Value**: ego sequence of shape (T, B, D) where T=196

This is the standard temporal cross-attention pattern used in text-conditioned motion generation — the difference
is that text tokens are short (word-length), while ego trajectory tokens are long (T=196).

Complexity: O(L × T) vs the current O((L+2)²) self-attention, which is actually cheaper for long conditioning sequences.

## Architecture Change Required

1. **Ego encoder**: Replace `EgoEncoderPooled` (transformer + mean pool → single token) with `EgoEncoder`
   (transformer → full sequence of T tokens). The encoder backbone weights may be partially reusable if the
   transformer layers before pooling are the same — but contrastive pretraining was done with pooled output,
   so pretraining must be redone.

2. **Denoiser**: Change `arch: trans_enc` → `arch: trans_dec` in config. The `trans_dec` arch passes z
   as queries and conditioning tokens as key/value pairs.

3. **Config**: Create `config_ego_motion_new_vae_stoch_latent_4_trans_dec.yaml` (or latent-8 variant if
   H3 succeeds) with:
   - `model.denoiser.arch: trans_dec`
   - Update ego encoder class to `EgoEncoder` (no pooling)
   - Ego encoder output: (B, T, D) rather than (B, 1, D)

## Predicted Outcomes

| Metric | H2 baseline | H4 prediction | Confidence |
|--------|-------------|---------------|------------|
| FID @ CFG=5 | 6.603 | ≤ 6.60 (aim: ≤ 6.0) | Medium |
| R-prec@1 @ CFG=5 | 0.671 | ≥ 0.80 | High |
| R-prec@1 @ CFG=15 | 0.797 | ≥ 0.85 | Medium |
| Diversity | 5.779 | ~5.7–5.9 | High |

The key prediction to falsify: **R-precision improves significantly** (from ~0.67 to ≥ 0.80) when the
denoiser has access to the full temporal ego sequence rather than a pooled summary.

## What Falsifies This

- R-precision does NOT improve (< 0.75): Temporal cross-attention is not the bottleneck; pooling is not
  the reason for low R-prec. Alternative cause: ego conditioning signal is inherently noisy.
- FID degrades significantly (> 7.5): cross-attention introduces instability in this architecture.
- Training fails to converge (loss > 0.4 at epoch 2000): trans_dec architecture has stability issues.

## Practical Constraints

- **Cost**: ~2-3 days on H100 (ego encoder pretraining ~4h + diffusion training ~24h)
- **Risk**: Requires code changes to ego encoder (remove pooling) and slurm scripts
- **Dependency**: Should be done with latent-4 first; if H3 (latent-8) succeeds, can also try H4+H3 combined

## Implementation Steps

**Infrastructure check (done 2026-03-30):**
- `trans_dec` arch: EXISTS in `mld/models/architectures/mld_denoiser.py` (lines 130, 235)
- `EgoEncoder` (no pooling): EXISTS in `mld/models/architectures/ego_encoder.py:18`, returns `(B, T, D)`
- `EgoEncoderPooled` INHERITS from `EgoEncoder` and adds mean pooling (line 106)
- Infrastructure is 90% ready — no new classes needed, only config + pretraining changes

**Steps:**
1. Create config `config_ego_motion_new_vae_stoch_latent_4_trans_dec.yaml`:
   - Set `model.denoiser.arch: trans_dec`
   - Change ego encoder class from `EgoEncoderPooled` → `EgoEncoder`
2. Adapt ego encoder pretraining: contrastive loss needs to use `EgoEncoder` (sequence output),
   aggregate T tokens for loss computation (e.g., mean pool only during loss, not in encoder output)
3. Create slurm scripts: pretrain_ego_encoder_trans_dec.sh + diffusion_training_trans_dec.sh
4. Train (~4h pretrain + ~24h diffusion)
5. Eval at CFG=5 (and CFG=10 for R-prec comparison)
