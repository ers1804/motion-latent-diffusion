# Review Tracker — EgoPed Paper Hardening

*Created 2026-07-22 from the full-project review. Living document — update as items progress.*
*Severity: 🔴 blocks top-venue acceptance · 🟠 reviewers will push · 🟡 disclosed/structural.*

## Already fixed (2026-07-20/22 review pass)

| Item | Fix | Commit |
|---|---|---|
| Fig 1 showed false cross-attention architecture | Redrawn (`make_architecture.py`), paper+deck rebuilt | `bf41bde` |
| Held-out split referenced but never defined | Defined in §Eval + generalization check (3.34 vs 3.39) | `bf41bde` |
| Wrong crop cross-ref (§Eval → §4.1) | Fixed | `bf41bde` |
| Ablation checkpoint-selection asymmetry undisclosed | Disclosed (conservative for our conclusion) | `bf41bde` |
| MM "higher is better" (uncond maximizes MM trivially) | Reworded — joint with R-Precision | (follow-up commit) |

## Open items

### 1. 🔴 No external baselines — IN PROGRESS
Every learned comparison is our own variant (H2/H3/H4/H6) + non-learned baselines. No published method in any table.
- **Decision (2026-07-22)**: build BOTH — (a) deterministic seq2seq regressor ego→motion and
  (b) MDM-style raw-space diffusion with our ego encoder.
- **Status**: (a) ✅ Regressor DONE 2026-07-21 — FID 35.96 / ADE 1.80 (K=1): quantifies the
  realism/accuracy trade-off; H4 minADE_5 (1.29) beats the L2-optimal predictor while staying
  realistic. (b) ⏳ MDM-style raw-space diffusion: the codebase's VAE_TYPE='no' diffusion_only
  path is the natural implementation — verify plumbing for condition=ego, then submit to helma.
  → 2026-07-21: plumbing verified (forward smoke test OK, 393-token self-attn); job 578124
  FAILED in 4m (YAML gotcha: bare `no` parsed as bool False → vae_type comparisons broke; fixed
  by quoting). Resubmit 578473 trained fine (loss 0.287 @ ep99) but crashed at first VALIDATION:
  R-Precision undefined for raw-space (263-D feats vs 256-D ego emb) → assertion. Fixed in
  ego_motion.py (skip R-prec on dim mismatch, report 0.0). RESUBMITTED as **580009**.

### 2. 🔴 Single training seed per configuration — IN PROGRESS
All conclusions rest on one training run each; 3 "replications" are sampling-only. H2's own FID
oscillation (6.60→8.40→7.51, flat loss) shows training noise is large. The 49% headline likely
survives; the self-vs-cross-attention tie (Δ=0.15) cannot be resolved at n=1 seeds.
- **Decision (2026-07-22)**: H2 + H4, 2 extra seeds each (4 runs), on **helma** (parallel SLURM).
  The self-vs-cross-attention tie stays single-seed (framed conservatively in the paper).
- **Status**: agent preparing SLURM submissions.

### 3. 🔴 Evaluator validity untested on this domain — DEFERRED (user decision 2026-07-22)
t2m evaluator is HumanML3D-trained; nothing validates it ranks systems as a human would on
driving-pedestrian motion.
- **Decision**: human study deferred entirely; stays in Limitations as future work.
- **Mitigation**: evaluator-independent ADE/FDE metrics (item 5) become the primary defense.

### 4. 🟠 Interaction pipeline described but never ablated — IN PROGRESS
§4.1 presents interaction scoring/weighted sampling/closest-approach crop as design choices, but no
experiment isolates their effect.
- **Plan**: 2 training runs on H4 recipe: (a) uniform sampling (no interaction weighting);
  (b) random crop (no closest-approach centering). Compare FID/R@1 vs H4.
- **Status**: queued for helma (decision 2026-07-22).

### 5. 🟠 No evaluator-independent conditioning metric — IN PROGRESS
R@1 uses each model's own encoder → H2-vs-H4 R@1 not apples-to-apples. Open TODO since April.
- **Status 2026-07-21**: ✅ COMPLETE. H4 beats H2 on all 4 trajectory metrics (3/4 significant,
  paired N=1,190) despite lower R@1 → R@1 cross-model gap shown to be an embedding artifact.
  Uncond sanity passes. Metric caveats documented (mean rewards collapse, min-of-K rewards
  diversity; H4 alone strong on both). Ready for paper §Eval + results table.

### 6. 🟠 Ego-encoder pretraining never ablated — IN PROGRESS
H6 tests unfreezing, but random-init+frozen was never run — so "pretrain contrastively then freeze"
has only half its claim ablated. (Adapter variant from Discussion also untried — stretch goal.)
- **Plan**: 1 training run: H4 recipe with random-init frozen ego encoder.
- **Status**: queued for helma (decision 2026-07-22).

### 7. 🟡 Pose pseudo-GT quality unquantified — ⚠️ **USER TODO**
The entire dataset is OmniRe-estimated poses; no MPJPE vs any reference, no failure-rate stats.
- **Why user**: needs a reference source (mocap subset, manually verified clips, or a labeled
  benchmark like JRDB-Pose overlap) that the agent cannot conjure from the repo.
- **Suggested minimal version**: manually rate N=50 random extracted sequences (good/usable/broken)
  and report the rate; or quantify against any scene where a second pose source exists.
- **Status**: ⚠️ waiting on Erik — decide reference source / do the manual rating.

### 8. 🟠 Statistical practice — IN PROGRESS
CI-overlap eyeballing; n=3 normality assumption; main table on full val (also the selection set)
while ablation is on held-out; H2/H3/H6 never rerun on val_test.
- **Status 2026-07-21**: (a) ✅ held-out table COMPLETE — H4 3.34, H6 5.00, H2 5.69, H3 6.79
  (ranking preserved; H2-vs-H4 gap Welch p<0.001). (b) ✅ stats tooling `research/src/stats_tests.py`
  (Welch + bootstrap from per-rep logs); formal n.s. verdict for the mechanism-ablation tie.
  (c) ⏳ paper table edit pending the eval-regime symmetric cell + helma 2×2.
- **MAJOR (eval-side confound resolved)**: H2's published 6.603 was CROP-eval; unified no-crop eval
  gives H2=5.62 full-val → **unified headline ≈40%, not 49%** (still significant). H4 crop-eval
  symmetric cell running.

## Structural / disclosed (no action beyond awareness)
- ✅ AVA = 34 samples framing — SIGNED OFF by user 2026-07-22; applied to abstract, §4.1 opening, and deck (staged high-interaction complement to nuScenes/Waymo scale).
- "First system" claim — qualified to full-body 3D + ego odometry; defend boundary vs WoSAD-style 2D work.
- **Reproducibility landmine**: `configs/*trans_dec.yaml` silently trains `trans_enc`
  (module-merge bug); run dirs named `h4_trans_dec` contain trans_enc models. MUST fix configs +
  naming before any code release.
- Paper `\TODO`s remaining (user): vehicle/sensor figure, example-scene figure, staged-scenario count.

## Compute ledger — SUBMITTED to helma 2026-07-22 (24h H100 each)
| Run | Item | SLURM job | Notes |
|---|---|---|---|
| rv_h2_seedB / rv_h2_seedC | 2 | 575102 / 575103 | original H2 recipe (pipeline ON, bs128, seeds 2345/3456) |
| rv_h4_seedB / rv_h4_seedC | 2 | 575106 / 575107 | original H4 recipe (pipeline OFF, seeds 2345/3456) |
| rv_h4_pipeline | 4 (2×2) | 575104 | H4 + interaction crop/weighting |
| rv_h2_nopipeline | 4 (2×2) | 575101 | H2 − interaction crop/weighting |
| rv_h4_randinit_ego | 6 | 575105 | H4 with random-init FROZEN ego encoder |
| Regressor baseline | 1 | (local, next) | deterministic ego→motion seq2seq |
| MDM-style baseline | 1 | (TBD) | raw-space diffusion + ego encoder |

**Protocol note (seeds)**: single 24h segments; per-seed best checkpoint selected by training-time
val FID (matches the paper's checkpoint-selection protocol) — NOT a fixed epoch, since seg-1 wall
time may end slightly before the original best epochs (H4: 3399, H2: 4399).

**⚠️ Item 4 redesign (2026-07-22, after integrity finding #2)**: H2 trained WITH the interaction
pipeline, H4 WITHOUT (silent config default; ~40% of sequences >196 frames → material). The two
runs above complete the 2×2 (existing corners: H2+pipe=6.603, H4−pipe=3.392). Also needed:
unified-eval re-run of H2 ep4399 under the no-crop eval config (local, eval-only).
