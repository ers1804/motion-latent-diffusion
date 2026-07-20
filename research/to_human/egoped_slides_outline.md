# EgoPed — Presentation Outline (for Gamma / PPTX)

*13 slides · research progress deck for professor · all numbers verified from experiments.*
*Paste into Gamma (after refilling credits) or use to build a PPTX.*

---

## 1. Title
**EgoPed: Ego-Conditioned Pedestrian Body Motion Generation via Latent Diffusion**
Generating realistic 3D full-body pedestrian motion conditioned on vehicle ego trajectory, for autonomous-driving simulation.
Backbone: Motion Latent Diffusion (MLD) on 263-D HumanML3D. Data: AVA + nuScenes + Waymo.

---

## 2. Problem & Motivation
- AV simulation needs diverse, plausible pedestrian behaviors to stress-test perception & planning.
- Existing 3D motion generators are text/action conditioned → context-agnostic (a crossing looks identical whether a car approaches at 50 km/h or is stopped).
- Real pedestrian behavior is coupled to the ego vehicle: speed, proximity, trajectory shape crossing/waiting/accelerating/turning.
- Gap: ego-conditioned **full-body** synthesis is unexplored (prior work = 2D trajectories, or 3D bodies from text).
- Question: how should the ego trajectory be injected into a latent-diffusion denoiser?

---

## 3. Approach: Ego-Conditioned Latent Diffusion
- MLD backbone: diffuse in a frozen HumanML3D VAE latent (263-D → L=4 tokens × 256-D). Only the denoiser is trained.
- Ego encoder: transformer maps 196-step ego trajectory → conditioning tokens; pretrained to align with VAE latents, then **frozen**.
- Studied: **pooled** (1 mean-pooled ego token) vs **full-sequence** (all 196 per-timestep tokens).
- Full-sequence: 196 ego + timestep + 4 latent tokens jointly self-attended → per-timestep temporal resolution.
- CFG at inference; DDIM, 50 steps.

---

## 4. Dataset: A New Ego–Pedestrian Benchmark
- Three sources: **AVA** (our sensor vehicle), **nuScenes**, **Waymo**.
- Pipeline: MS3D annotation → per-camera 3D SMPL (adapted OmniRe) → stitch → 263-D HumanML3D → pair with ego odometry (pedestrian-centric frame).
- **11,954 samples** = 9,540 train / 2,414 val (scene-disjoint).
- Per-source (train/val): AVA 27/7, nuScenes 3,637/943, Waymo 5,876/1,464.
- Interaction-aware: score = travel × (%frames within 5 m) × relative-bearing change → weighted sampling + closest-approach crop.

---

## 5. Main Result: Full-Sequence Conditioning Wins (−49% FID)

| Method | FID ↓ | R@1 | MM | Diversity |
|---|---|---|---|---|
| H2 Pooled (baseline) | 6.603 ± 0.067 | 0.671 | 3.50 | 5.78 |
| **H4 Full-sequence (ours)** | **3.392 ± 0.179** | 0.548 | 3.96 | 5.79 |
| Ground truth (ref) | — | — | — | 5.50 |

- Non-overlapping CIs → significant. Best checkpoint occurs **earlier** (ep 3399 vs 4399) → not a compute effect.

---

## 6. Guidance-Scale Behavior

| CFG | H2 Pooled FID | H4 Full-seq FID |
|---|---|---|
| 5 | 6.603 | 3.842 |
| 10 | 6.963 | **3.392** |
| 15 | 7.400 | 3.547 |

- Full-sequence is nearly CFG-insensitive; pooled degrades steeply. Rich 196-token signal resists over-conditioning collapse.

---

## 7. Ablation 1 — Larger Latent (rejected)
- H3: latent dim 8 vs 4. Result: **14.5% worse** FID (7.563 vs 6.603); R@1 unchanged (0.676 vs 0.671).
- Lesson: bigger latent without a bigger denoiser is counterproductive; bottleneck is the ego representation, not latent size.

---

## 8. Ablation 2 — Frozen vs Unfrozen Ego Encoder (freeze wins)

| System | FID | R@1 | MM |
|---|---|---|---|
| **H4 (frozen ego encoder)** | 3.392 | 0.548 | 3.96 |
| H6 (unfrozen ego encoder) | 5.079 | 0.352 | 2.618 |

- Unfreezing collapses MultiModality (−34%): encoder over-specializes to discriminative per-condition features.
- Matches frozen-encoder paradigm (CLIP in Stable Diffusion; ControlNet / IP-Adapter adapters on frozen backbones).

---

## 9. Ablation 3 — Self-Attention vs Cross-Attention (comparable)
Both consume the full 196 ego tokens; differ only in attention. Matched training, held-out test:

| Denoiser | FID | R@1 | MM |
|---|---|---|---|
| **Self-attention over concat (ours)** | 3.338 ± 0.190 | 0.537 | 4.14 |
| Cross-attention (Q / K,V) | 3.490 ± 0.059 | 0.501 | 4.61 |

- Statistically comparable (overlapping CIs). **The win is full-sequence conditioning, not the attention mechanism.**

---

## 10. Held-Out Test & Reference Baselines
- Generalization: on a scene-disjoint held-out split, best system reproduces the headline (FID 3.38 vs 3.39) → not a selection artifact.
- Reference baselines (FID ↓): Retrieval 0.06 (memorization sanity), Unconditional prior 5.18 (→ 3.39 with ego, −34%), VAE interp 7.91, Trajectory+static body 48–50 (task needs articulated motion).
- Unconditional R@1 = 0.031 = chance (1/32) → protocol sanity check passes.

---

## 11. Design Recipe & Contributions
- **First** system: full 3D body motion conditioned on vehicle ego odometry (AVA+nuScenes+Waymo).
- Full-sequence temporal conditioning beats pooled by **49% FID** — robust to CFG and attention mechanism.
- Frozen ego encoder is critical — unfreezing collapses diversity.
- Recipe: contrastively pretrain ego encoder on frozen VAE latents → freeze → condition on full ego sequence. CFG ≈ 10; select checkpoints on val FID.

---

## 12. Limitations & Next Steps
- Estimated (not GT) 3D pose labels → noisy targets.
- Only 2D ego trajectory conditioning; richer geometry/velocity/map could help.
- R@1 gap (0.548 vs pooled 0.671) — conditioning-vs-diversity tradeoff; adapter tuning is a next step.
- No physics constraints; single benchmark; no perceptual study yet.
- Status: main results + 3 ablations + held-out validation complete; paper draft in progress.

---

## 13. Thank You / Discussion
- EgoPed = full-sequence ego conditioning + frozen encoder: a strong, validated recipe.
- 49% FID improvement, confirmed on held-out test.
- Discussion: richer conditioning signals · adapter-based R@1 · perceptual evaluation.
- *(Insert real figures from the paper: architecture diagram, H4/H6 training curves, qualitative renders.)*
