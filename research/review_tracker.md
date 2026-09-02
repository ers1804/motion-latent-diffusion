# Review Tracker — EgoPed Paper Hardening

*Created 2026-07-22 from the full-project review. Living document — update as items progress.*
*2026-07-23: PAPER REWRITTEN (commit da22972) — unified 40% headline, EgoPed-IA flagship (2.880),
seed/2×2/ADE-FDE/encoder sections in. Report 009 delivered. Open: H3-unified + H2−pipe finals
(evals in flight), item 9 probe, deck refresh, user items (7, figures, sign-off).*
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
  (two early failures fixed: YAML bare-`no` bool gotcha; R-prec dim assertion.) **580009 COMPLETED
  2026-07-22** (3,000 epochs): best train-time FID **26.2@ep299**, degrading to 33.5 — raw-space
  diffusion ~9× worse than latent (justifies backbone). Definitive eval: job 589986.

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
- **Status**: ✅ CLOSED 2026-07-27 — definitive 2×2 complete (see findings).

### 5. 🟠 No evaluator-independent conditioning metric — IN PROGRESS
R@1 uses each model's own encoder → H2-vs-H4 R@1 not apples-to-apples. Open TODO since April.
- **Status 2026-07-21**: ✅ COMPLETE. H4 beats H2 on all 4 trajectory metrics (3/4 significant,
  paired N=1,190) despite lower R@1 → R@1 cross-model gap shown to be an embedding artifact.
  Uncond sanity passes. Metric caveats documented (mean rewards collapse, min-of-K rewards
  diversity; H4 alone strong on both). Ready for paper §Eval + results table.

### 6. 🟠 Ego-encoder pretraining never ablated — PRELIMINARY VERDICT (2026-07-21)
- **Training-time val (rv_h4_randinit_ego, ep~3100)**: best FID 5.45, R@1 = 0.022 (chance).
  Random frozen encoder ≈ unconditional prior (5.18) → **contrastive pretraining is what injects
  usable ego information**. Definitive 3-rep eval after job completes.

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

### 9. 🟢 Behavioral-validity analysis (added 2026-07-21, user-approved)
Does the generated distribution match GT in INTENTION space — p(cross/stop/walk | ego)? Tests
behaviorally meaningful ego→behavior coupling, which FID/R@1/ADE all miss; gives the MM/diversity
story semantic teeth. (Full intention *prediction* from prefixes = future work / next arc, not this paper.)
- **Plan**: (i) proxy intention labeler from trajectories (crossing = ped/ego path intersection;
  stopping = speed <ε sustained; CPU, heuristics documented + spot-checked); (ii) validate label
  distributions on GT train/val; (iii) extend eval_ade_fde.py to dump root trajectories, label K
  generated samples per condition for H4/H2/H6, compare intention distributions vs GT (per-scene +
  marginal); (iv) latent-organization figure (PCA of VAE latents colored by label) as garnish.
  Predictions to test: H6 concentrates one intention/ego (collapse); H2 shifts less with ego than H4.
- **Order**: COMPLETE 2026-07-23 (probe run on the ADE/FDE sample dumps; paper subsection added).
- **GT validation (2026-07-21)**: val(all) n=2835: crossing 1.7%, stopping 36.5%, walking-only
  62.1%; val_test: crossing 1.8%, stopping 24.0%; train(1.5k): crossing 1.9%, stopping 35.3%.
  **Design consequences**: (i) strict path-intersection crossing is too rare (~2%) to be the
  primary category → lean on stopping-vs-walking (healthy 36/62 split); either broaden crossing
  (corridor-proximity/lateral-traversal definition) or report it as a rare-event annex.
  (ii) NOTE: val_test has less stopping than full val (24% vs 36.5%) — the scene-disjoint halves
  are behaviorally imbalanced; harmless for FID (held-out matched 3.34≈3.39) but report per-split
  base rates in the analysis. (iii) labeler stem-matching can collide across sources (n=1199 vs
  1190) — use source-prefix mapping in the final analysis.

### 10. 🟢 Unconditional + text-conditioned MLD baselines (added 2026-09-02, user request)

**(a) Properly-trained unconditional MLD** — closes the gap the repo's own config flagged
("approximate" ego-zeroed prior). ⚠️ `configs/config_ego_motion_train_uncond.yaml` is STALE
(latent-[1,256], VAE `vae_ava_nuscenes_waymo` ep6999, batch 32) — would NOT be comparable to
H4 (latent-[4,256], interaction-crop VAE ep5999). Same drift class as findings #1/#2.
`slurm/review/rv_uncond_trained.sh` therefore drives the **H4 config** with everything pinned via
CLI overrides + `guidance_uncondp=1.0`, CFG=1.0 at test, no interaction pipeline (matches H4).
Auto-submits when helma maintenance ends (18:00 2026-09-02).

**(b) Original text-conditioned MLD** (Chen et al. 2023) as an external baseline.
- ⚠️ PREREQUISITE: the official checkpoint `1222_mld_humanml3d_FID041.ckpt` is NOT on this machine
  or the NAS — `configs/config_mld_humanml3d.yaml:55` points at `/home/mohan/Documents/erik/...`
  (a colleague's machine). Must be obtained (Mohan / official MLD release) before this can run.
  CLIP (`deps/clip-vit-large-patch14`) is present; our JSONs have NO text field, so prompts are synthesized.
- DESIGN: a single fixed prompt = a CONSTANT condition → cannot adapt per scene, R@1 at chance.
  Proposed prompt ladder (inference-only, cheap) instead of one prompt:
  P1 naive (user's): "a pedestrian tries to cross the street and reacts to the ego vehicle"
     — out-of-distribution for HumanML3D captions; tests the practitioner's first instinct.
  P2 idiomatic: "a person walks forward and then stops" — in-distribution phrasing matching our
     dominant behavior (36% stopping / 62% walking); gives the baseline its BEST shot (fair version).
  P3 oracle-text (optional): per-sample prompt chosen by our GT stopping label — fairest per-scene
     text competitor; must be labeled an ORACLE since it uses GT-derived info.
- FRAMING CAVEAT: pretrained MLD is HumanML3D-trained (mocap, dance/exercise) evaluated on
  driving-scene pedestrians with estimated poses → a poor FID conflates DOMAIN GAP with the
  conditioning question. Honest claim = "off-the-shelf text-to-motion does not transfer; in-domain
  ego-conditioned training is required" — NOT "text conditioning is inherently worse".
  Controlled alternative (costs 1 training run): train OUR architecture with text conditioning on
  OUR data using synthesized prompts — removes the domain gap, isolates text-vs-ego conditioning.

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

**Pre-completion extraction (2026-07-21, ~22.4h, training-time val — NOT definitive):**
| Run | best FID @ ep | last | note |
|---|---|---|---|
| rv_h4_pipeline | **3.664 @ 1299** | 4.92 @ 2099 | pipeline HELPS H4 (orig no-pipe train-best 3.77) → headline likely conservative |
| rv_h2_nopipeline | 5.763 @ 1199 | 6.03 @ 1699 | too short (H2 best @4399); CFG-15 val — resume needed |
| rv_h2_seedB / C | 5.86/5.21 @ 99 | 12.3 @ 1099 | mid-run FID hump or instability; ~69s/epoch (4× slower than orig H2) — resumes needed |
| rv_h4_seedB | 4.658 @ 3099 | still ↓ | seed variance REAL (orig seed 3.77) — resume to let it bottom out |
| rv_h4_seedC | 4.973 @ 1799 | 5.27 plateau | seed variance REAL |
| rv_h4_randinit_ego | 5.453 @ 2699 | 5.56 | ≈ uncond prior; R@1 = chance → pretraining essential |

**Seg-1 COMPLETE (all TIMEOUT@24h, 2026-07-22). Final training-time-val bests:**
h4_pipeline 3.664@1299 | h4_randinit 5.034@3199 (R1 chance) | h4_seedB 4.658@3099 |
h4_seedC 4.973@1799 | h2_nopipeline 5.763@1199 | h2_seedB 5.86@99→11.5 | h2_seedC 5.21@99→13.7.
**Submitted 2026-07-22**: seg-2 resumes 581907-581911 (h2_seedB/C, h2_nopipeline, h4_pipeline,
h4_seedB); definitive 3-rep evals 581912-581915 (h4_pipeline@1299, h4_randinit@3199,
h4_seedB@3099, h4_seedC@1799 — all CFG10, unified eval). rv_mdm_style (580009) running @12h.
H4 SEED VARIANCE flag: train-time bests 3.77 / 4.66↓ / 4.97 across seeds — definitive evals will
quantify how much of FID=3.39 is seed luck; must be reported honestly (mean±std across seeds).

**⚠️ Item 4 redesign (2026-07-22, after integrity finding #2)**: H2 trained WITH the interaction
pipeline, H4 WITHOUT (silent config default; ~40% of sequences >196 frames → material). The two
runs above complete the 2×2 (existing corners: H2+pipe=6.603, H4−pipe=3.392). Also needed:
unified-eval re-run of H2 ep4399 under the no-crop eval config (local, eval-only).

## Weights backup (2026-08-03) — every paper-reported model on NAS
All review-run artifacts synced to `/home/erik/NAS/methods/diffusion_gen/models/helma_models/models/mld/`
(best checkpoint + ALL dumped `config_*_train.yaml` + training logs per run):
h4_pipeline ep1299 (EgoPed-IA) · h2_nopipeline ep1199 · h4_seedB ep3099 · h4_seedC ep1799 ·
h4_randinit_ego ep3199 · mdm_style ep299 · h4_real_trans_dec(+_bs128) ep2499 (from local) ·
h2_seedB/C configs+logs only (instability evidence). Verified 2026-08-03; original H4/H2/H3/H6
mirrors were already present. Paper numbers → weights+configs+logs now in two locations.
