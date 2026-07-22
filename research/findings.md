# Findings — Ego-Conditioned Pedestrian Motion Generation

## ⚠️ INTEGRITY FINDING #2 (2026-07-22): H2 and H4 trained under DIFFERENT data pipelines

While preparing the item-4 ablation (review_tracker.md), discovered from the DUMPED run configs
on the NAS mirror (authoritative):
- **H2** (`interaction_crop_weighted_1_helma`): `INTERACTION_CROP: true`, `INTERACTION_WEIGHTED_SAMPLING: true`
- **H4** (`h4_trans_dec`): keys ABSENT → code defaults **False** → H4 trained with UNIFORM sampling
  and RANDOM crops. Same for H6 and both new trans_dec runs (they reuse the H4 config).
- Cause: the H4 config was written fresh and never carried the `DATASET.EGOMOTION.INTERACTION_*`
  keys; `get_datasets()` defaults them to False (`EgoMotion.py:613`). Same silent-config-default
  failure class as the trans_enc/trans_dec bug.
- **Material**: ~40% of sequences exceed 196 frames (val 39.3%, train 41.0%) → the crop-mode
  difference is exercised constantly; weighted sampling affects ALL training batches.

**Blast radius:**
- H2-vs-H4 headline (49%) is CONFOUNDED: conditioning granularity AND data pipeline both differ.
  Direction: if the pipeline HELPS, the headline is conservative (H4 won while handicapped);
  if it HURTS, part of H4's edge may be pipeline removal. Unresolved until the 2×2 completes.
- H4-vs-H6 and self-vs-cross-attention ablations: CLEAN (all no-pipeline).
- H3-vs-H2: CLEAN (H3 config sets both True). H3-vs-H4 in the main table: mixed (not a claimed head-to-head).
- EVAL-side: H2's numbers were evaluated under its own config (interaction crop at eval) while
  H4's used no-crop eval → eval data differs for long sequences. Needs a unified-eval re-run of H2.
- Paper §4.1 as written claims weighted sampling is used in diffusion training — TRUE for H2/H3/VAE,
  FALSE for the main model H4. Text corrected 2026-07-22.

### Eval-side confound RESOLVED (2026-07-21, item 8) — headline refines 49% → ~40% under unified eval

Completed the held-out table + unified-eval H2 re-run locally (3 reps, unified NO-CROP eval pipeline):

| Model (ckpt, CFG) | val_test FID | full-val FID (no-crop eval) | published (own eval) | R@1 (val_test) |
|---|---|---|---|---|
| H4 ep3399 CFG10 | **3.338 ±0.190** | 3.392 (orig, no-crop) | 3.392 | 0.537 |
| H6 ep3399 CFG10 | 5.003 ±0.228 | — | 5.079 | 0.374 |
| H2 ep4399 CFG5  | 5.690 ±0.151 | **5.620 ±0.117** | 6.603 (CROP-eval) | 0.676 |
| H3 ep4399 CFG5  | 6.792 ±0.345 | — | 7.563 (CROP-eval) | 0.678 |

- **KEY**: H2's published 6.603 was measured under its own interaction-crop EVAL pipeline; under the
  unified no-crop eval it is **5.62** (full val) / 5.69 (val_test). The 49% headline mixed eval
  pipelines. **Unified apples-to-apples: 5.62→3.39 ≈ 40% (full val); 5.69→3.34 ≈ 41% (held-out).**
- Ranking PRESERVED everywhere (H4 < H6 < H2 < H3); held-out H2-vs-H4 gap formally significant
  (diff +2.35, Welch p<0.001, bootstrap CI [+2.11,+2.59]).
- H3's crop-eval 7.563 → similar shift direction expected; its no-crop val_test = 6.79.
- Symmetric cell DONE (2026-07-21): **H4 crop-eval FID = 4.077 ±0.231** (R@1 0.552). Eval-regime
  2×2 complete and CONSISTENT: crop-eval 6.603→4.077 = 38.3% improvement; no-crop 5.620→3.392 =
  39.6%. **The robust unified headline is ~38-40% under either regime**; 49% was purely the
  regime-mixing artifact. Both models find interaction-centered eval windows harder (sensible).
- **Paper edit pending**: headline + tables rewrite ONCE when (a) the symmetric cell and (b) the
  helma training 2×2 land. Until then the deck's caveat covers it.

### ADE/FDE — evaluator-independent conditioning metric (2026-07-21, item 5) — R@1 gap is an ARTIFACT

Root-trajectory (pelvis XZ) ADE/FDE, held-out val_test, K=5 samples/condition, N=1,190, meters:

| Model | ADE ↓ | FDE ↓ | minADE_5 ↓ | minFDE_5 ↓ | (R@1) | (MM) |
|---|---|---|---|---|---|---|
| **H4 full-seq** | **2.320** | **4.826** | **1.291** | **2.591** | 0.537 | 4.14 |
| H2 pooled | 2.434 | 4.990 | 1.482 | 2.979 | 0.676 | 3.83 |
| H6 unfrozen | 2.227 | 4.578 | 1.528 | 3.090 | 0.374 | 2.44 |
| H4 uncond (ego=0) | 2.880 | 5.876 | 1.228 | 2.454 | 0.031 | — |

- **KEY: H4 beats H2 on ALL FOUR trajectory metrics** (paired over 1,190 conditions: ADE −0.114
  sig., minADE −0.190 sig., FDE −0.164 n.s., minFDE −0.388 sig.) **despite R@1 0.537 vs 0.676** →
  the cross-model R@1 gap is an embedding-space artifact, NOT worse conditioning. Replaces the
  paper's hand-wavy "diversity explains R@1" paragraph with direct evidence.
- Metric behavior caveats (report both): ADE_mean rewards mode-concentration — H6 (MM-collapsed)
  has the best mean ADE but the worst minADE; minADE_K rewards diversity — uncond (max diversity)
  has the best minADE_5 but the worst mean ADE. H4 is the only model strong on BOTH.
- Uncond sanity: worst mean ADE/FDE (2.88/5.88) → conditioning demonstrably helps trajectory
  following (2.88→2.32 mean ADE).
- Data: research/data/ade_fde_*_val_test_k5.npz; script research/src/eval_ade_fde.py.

### Regressor baseline (2026-07-21, item 1a) — the realism/accuracy trade-off, quantified

Deterministic seq2seq ego→motion transformer (EgoEncoder backbone + MLP head, masked MSE, H4's
exact data recipe; val MSE plateaus at ep20 = regression-to-the-conditional-mean saturation).
Scored via the same baseline_utils t2m path + ADE/FDE:

| Metric | Regressor (K=1) | H4 diffusion | H2 | interpretation |
|---|---|---|---|---|
| FID (val_test) | **35.96** | 3.34 | 5.69 | regression mean is UNREALISTIC (10× worse FID) |
| Diversity | 3.16 | ~5.8 | ~5.8 | severe diversity collapse (GT 5.32) |
| ADE (mean) | **1.80** | 2.32 | 2.43 | L2-optimal by construction — "wins" mean ADE |
| minADE_5 | — (deterministic) | **1.29** | 1.48 | H4 with 5 samples BEATS the L2-optimal point predictor |

- **The classic trajectory-prediction insight, now quantified for full-body motion**: the
  conditional-mean predictor minimizes ADE but produces unrealistic, collapsed motion (FID 36 —
  sits between the diffusion family ~3-7 and the kinematic floor ~48). Diffusion trades ~0.5m mean
  ADE for realism + diversity — and with K=5 samples H4's minADE (1.29) beats the regressor's 1.80:
  **better coverage AND realism**. This is the justification-for-generative-modeling row the
  external-baseline table needed.
- Full-val numbers consistent (FID 35.47, ADE 1.86). Data: research/data/regressor_metrics_*.json.

### DEFINITIVE review-run evals, round 1 (2026-07-22) — pipeline makes H4 BETTER; pretraining essential; seed sensitivity real

3-rep definitive evals (full val, unified no-crop eval, CFG=10) of the settled seg-1 checkpoints:

| Run | FID | R@1 | MM | Verdict |
|---|---|---|---|---|
| **H4 + interaction pipeline (ep1299)** | **2.880 ±0.084** | 0.323 | 3.37 | **NEW BEST FID** — beats original H4 (3.392) by 15% at 1/3 the epochs. 2×2 H4-side: pipeline HELPS → the unified ~40% headline is CONSERVATIVE. R@1 low at this early epoch (seg-2 running may improve both). |
| H4 random-init frozen ego (ep3199) | 4.968 ±0.171 | 0.024 (chance) | 4.44 | **Item 6 CLOSED**: ≈ uncond prior (5.18); contrastive pretraining is what injects ego information. |
| H4 seed B (ep3099) | 4.719 ±0.164 | 0.441 | 4.00 | seed variance |
| H4 seed C (ep1799) | 4.803 ±0.158 | 0.650 | 3.69 | seed variance |

- **Seed sensitivity (interim)**: definitive best-checkpoint FID across seeds = 3.392 / 4.719 / 4.803
  → mean 4.30 ±0.79. CAVEATS before final claim: new seeds trained 1×24h segment (~3,300 epochs) vs
  the original's 2 segments (5,000; best@3399); seedB was still descending at cutoff and its seg-2
  resume is running. Final seed verdict after seg-2. Either way: per-seed checkpoint selection and
  seed error bars MUST appear in the paper; single-seed point estimates overstate precision.
- **Ops pitfall (recorded)**: eval jobs that don't override NAME write metrics JSONs to the same
  results dir; two jobs with identical start-timestamps overwrote each other (581913/581915) —
  recovered from stdout. Future eval scripts: override NAME per job.
- Remaining for the full 2×2: H2-side (h2_nopipeline seg-2 running; H2+pipe corner = 5.620 known).

### MDM-style raw-space baseline (2026-07-22, item 1b) — latent diffusion justified

Raw-space ego-conditioned diffusion (VAE_TYPE='no', diffusion_only path; same frozen ego encoder,
same data recipe, same denoiser width as H4) completed 3,000 epochs on helma.
**Training-time val trajectory: best FID 26.2 @ ep299**, hovers 26–28 through ~ep1600, degrades to
33.5 by ep2999 — never approaches latent-space territory (2.9–3.4). **~9× FID gap justifies the
latent-diffusion backbone**; sits between latent models (~3) and the regressor (36).
Honest framing for the paper: a same-capacity raw-space VARIANT of our model (controlled
comparison), not a faithful MDM reproduction (MDM uses larger models/longer training).
Definitive 3-rep eval of ep299 submitted (job 589986, NAME-collision-proofed).

### Item 2 (seeds) FINAL + item 1b definitive (2026-07-22)

- **h4_seedB seg-2 COMPLETED** (full 5,000 epochs): seg-2 best 4.937@ep3899 — never beat its seg-1
  best (4.658@ep3099, definitive 4.719 ±0.164). The "still descending" caveat is resolved: no
  further improvement. **FINAL H4 seed spread (definitive best-checkpoint FID): 3.392 / 4.719 /
  4.803 → mean 4.30 ± 0.79 (best seed 3.392).** Paper must report the spread; the flagship
  H4+pipeline number (2.880) is single-seed and must say so.
- **MDM-style definitive (3-rep, ep299): FID = 26.43 ±0.10, Diversity 3.55 (collapsed; GT 5.45),
  R@1 = 0.0 (undefined-space skip path ✓).** Item 1b CLOSED. Final external-baseline ladder
  (unified eval): latent diffusion 2.88–3.39 ≪ raw-space diffusion 26.4 < regressor 36.0 ≪
  kinematic floor 48–50.

**Resolution plan (submitted to helma as part of review items 2/4/6):**
2×2 completion — (a) H4 + interaction pipeline; (b) H2 − interaction pipeline. Existing corners:
H2+pipe (6.603), H4−pipe (3.392). Plus unified-eval re-run of H2 ep4399 under the no-crop eval
pipeline (eval-only, local). Seeds runs replicate each model's ORIGINAL recipe (H2 with pipeline,
H4 without).


*Last updated: 2026-07-09 (PAPER-STRENGTHENING arc: dataset-provenance audit + Dataset §4.1 TODO cleanup)*

## FINAL RESULT (2026-07-17): at matched batch, cross-attention ≈ self-attention-concat

Closed the batch confound by retraining trans_dec at effective batch 128 (micro 64 × grad-accum 2,
via env ACCUM_GRAD_BATCHES added to train.py). Trained to epoch 3775 (stopped), scanned + 3-rep
definitive on held-out val_test. Also ran trans_enc's own 3-rep on val_test for a same-split CI.

**Definitive 3-way comparison — held-out val_test, CFG=10, 3 replications:**

| Model | arch | batch | FID ↓ | R@1 | MM |
|---|---|---|---|---|---|
| **trans_enc** (accidental — paper's ACTUAL model) | self-attn concat | 128 | **3.338 ±0.190** | 0.537 | 4.14 |
| trans_dec (real cross-attention, best ep=2499) | cross-attn | 128 | 3.490 ±0.059 | 0.501 | 4.61 |
| trans_dec (real cross-attention, best ep=2499) | cross-attn | 64  | 3.725 ±0.085 | 0.509 | 4.56 |

**Verdict**: at MATCHED batch 128, trans_enc (3.338) and trans_dec (3.490) are **statistically
comparable** — 95% CIs overlap ([3.15,3.53] vs [3.43,3.55]); the 0.15 FID gap is not significant.
The batch size explained most of the bs64 gap (3.725→3.490). So the *real cross-attention is NOT
better* than the accidental self-attention-over-concatenation — if anything marginally worse and it
trains less stably (noisy FID trajectory 3.5–4.8 across epochs vs trans_enc's smooth curve).
trans_enc also has slightly better R@1 (0.537 vs 0.501); trans_dec has slightly higher MM (4.61 vs 4.14).

**What this means for the paper (the real contribution is intact, the mechanism story changes):**
- The 49% win over pooled H2 (6.6 → ~3.3–3.5) is REAL and is about **full-sequence temporal
  conditioning** (196 ego tokens) beating **pooled single-token** conditioning — this is
  architecture-agnostic (holds for both self-attn-concat AND cross-attention).
- The specific "cross-attention / trans_dec / O(L×T)-is-affordable" mechanism claim is FALSE and
  also unnecessary: the model actually uses self-attention over the concatenated sequence, and
  cross-attention gives no benefit.
- **Recommended paper fix**: (a) rewrite §Method / Fig 1 / Abstract to the true self-attention-concat
  mechanism; (b) reframe the contribution as full-sequence vs pooled conditioning; (c) ADD the
  trans_dec-vs-trans_enc comparison as an ablation ("cross-attention is not required; self-attention
  over the concatenated ego sequence is comparable"); (d) drop the O(L×T) affordability argument.
  This turns the config-merge bug into a rigorous ablation and keeps the headline result honest.

**APPLIED 2026-07-18** (commit 98544ac): paper corrected throughout — title, abstract, intro, §Method
Diffusion Denoiser (+equation), Fig 1 caption, CFG analysis, discussion, conclusion, appendix,
checklist all reframed to full-sequence self-attention-concat; dropped the false O(L×T) claim; added
ablation §"Conditioning Mechanism: Self-Attention vs Cross-Attention" (Table tab:xattn) with the
matched-batch comparison. Compiles clean. Draft for human review — figures/staged-count §4.1 items
(human-only) still outstanding.

---

## RESULT (2026-07-16): real cross-attention (trans_dec) is WORSE than the accidental trans_enc  [SUPERSEDED — was batch-confounded; see FINAL RESULT above]

Trained the REAL cross-attention denoiser (trans_dec) locally on the 4090 — same recipe as H4
(frozen VAE ep5999 + frozen ego encoder, data ava_nuscenes_waymo) EXCEPT batch=64 (H4 used 128;
4090 memory limit). Fixed the config-merge bug via CLI `model.denoiser.params.arch=trans_dec`
(verified builds decoder + cross-attention, 9.9M params vs trans_enc 8.0M). Trained to epoch 4999
(~18s/epoch, full trajectory saved every 100 epochs), then evaluated on the held-out `val_test`.

**Definitive comparison (held-out val_test, CFG=10, 3 replications unless noted):**

| Model | arch | batch | FID ↓ | R@1 | MM |
|---|---|---|---|---|---|
| **trans_enc** (accidental — the paper's ACTUAL reported H4) | self-attn concat | 128 | **3.377** (1 rep) | 0.541 | 3.96 |
| trans_dec (real cross-attention, best ep=2499) | cross-attn | 64 | 3.725 ±0.085 | 0.509 | 4.56 |
| trans_dec (ep=1999) | cross-attn | 64 | 4.163 ±0.126 | 0.505 | 4.52 |

**trans_dec 1-rep FID trajectory (val_test, CFG=10)** — note the instability:
1999→4.05, 2499→**3.64**, 2999→5.59, 3399→5.11, 3499→4.90, 3999→4.45, 4499→4.54, 4999→4.33.
Finer scan 2000–2700: 2099→4.92, 2199→4.32, 2299→4.36, 2399→4.70, 2599→5.00. The ep=2499 dip
is a sharp outlier among 4.3–5.0 neighbours → largely 1-rep noise; the honest trans_dec best is
~3.7 (3-rep). trans_enc's curve was smooth and bottomed at 3.38–3.39.

**Conclusion**: the *intended* cross-attention architecture is ~10% WORSE on FID and trains far
less stably than the *accidental* self-attention-over-concatenation that the paper's model actually
uses. The config bug was fortuitous. trans_dec does have higher MultiModality (4.56 vs 3.96) —
more diverse per condition — but worse distributional fidelity (FID). Both still crush pooled H2 (6.6).

**⚠️ CONFOUND (must close before claiming)**: trans_dec ran at batch=64, published trans_enc at
batch=128. Smaller batch could handicap trans_dec. The clean control is trans_enc @ batch=64 (same
recipe as this trans_dec run) → isolates architecture. NOT yet run. Until then the comparison is
suggestive, not airtight. [held-out test H4=3.377 and the trans_dec numbers are on the same val_test split.]

**Paper implication**: whichever way the batch-matched control lands, the paper's Method/Fig1/Abstract
still need correcting (reported model is trans_enc, not trans_dec). If trans_enc stays better even at
matched batch, the honest framing is "full-sequence conditioning via self-attention concatenation" +
this trans_dec comparison as an ablation (self-attn concat > cross-attention here). Deferred to human.

---

## ⚠️ CRITICAL FINDING (2026-07-14): paper misdescribes the core architecture

While setting up a held-out-test eval locally (RTX 4090), inspecting the actual checkpoints
revealed that **the paper's central "cross-attention / trans_dec" claim is FALSE**. Verified two
independent ways (checkpoint weights + denoiser code):

- **Checkpoint weights** (`epoch=3399.ckpt` H4, also H2 & H6): denoiser has **118 `encoder.*`
  keys, 0 `decoder.*` keys, 0 `multihead_attn` (cross-attention) keys, 36 `self_attn` keys**.
  → All three models are `trans_enc` (self-attention encoders). No cross-attention exists anywhere.
- **Code** (`mld_denoiser.py:192-224`, `arch=="trans_enc"` + `condition=='ego'`): the ego encoder
  output is concatenated with the latent — `xseq = cat([z (4 tok), time (1), ego_seq (196)])` = 201
  tokens — and run through **self-attention** (`self.encoder`). The first 4 tokens are taken as the
  denoised latent.
- **Why**: the H4 config sets `denoiser.params.arch: trans_dec`, but that override never took effect
  (silent config-merge bug); the runtime merged config shows `arch: trans_enc`. Training used
  trans_enc; the paper was written to the *intended* design, not the *actual* one.

**What is FALSIFIED in the paper** (Abstract, §Method "Diffusion Denoiser", Fig 1, Related Work):
  1. "trans_dec ... z_t forms queries and ego tokens form K/V" — FALSE (it's a self-attn encoder).
  2. "Cross-attention makes full-sequence conditioning affordable: O(L×T) vs O((L+T)²)" — FALSE and
     backwards; H4 actually pays the O((L+T)²) self-attention-over-concatenation cost it disclaims.
  3. The denoiser equation (TransformerDecoder) and Fig 1 cross-attention arrows.

**What SURVIVES (the real, still-valid contribution)** — reproduced on held-out test below:
  - Full per-timestep ego conditioning (196 tokens) via self-attention concat dramatically beats
    pooled single-token conditioning (H2): the H2→H4 delta is purely ego granularity (1 vs 196
    condition tokens fed to the SAME trans_enc), NOT enc-vs-dec.
  - H6 frozen-encoder ablation (also trans_enc, differs only by unfreezing) — unaffected.
  - All empirical numbers (FID 3.392, 49% improvement, H6 rejection) are real and reproduce.

**Recommended fix**: rewrite §Method (Diffusion Denoiser + Fig 1 + Abstract wording) to describe
the true mechanism — self-attention over the concatenated [latent, time, full-ego-sequence]
token set, with H2 vs H4 = pooled-token vs full-sequence conditioning. Drop the O(L×T) cross-
attention complexity argument (or reframe honestly: full-sequence conditioning costs more attention
than pooled, and is worth it). **Deferred to human — substantive scientific correction, not an
autonomous edit.**

### Held-out test set eval (2026-07-14) — headline result GENERALIZES
Built a scene-disjoint split of `val` (no reserve data exists): `val_sel` (1,140 samples / 522
scenes) + `val_test` (1,190 / 521), stats copied, run locally on the 4090.
- **H4 on val_test (CFG=10, 1 replication): FID = 3.377, R@1 = 0.541, Div = 5.729, gt_Div = 5.393.**
- Paper full-val H4: FID = 3.392, R@1 = 0.548. → Held-out ≈ full-val: the selection-bias concern is
  addressed; the result is not an artifact of evaluating on the checkpoint-selection set.
- TODO: run H2/H3/H6 on val_test (3 reps each) for a full held-out test table.

## Paper-Strengthening Arc (started 2026-07-09)

The research CONCLUDED 2026-04-15 with a drafted NeurIPS paper (`research/paper/main.tex`).
This arc targets the paper itself: resolve open `\TODO`s, add missing dataset facts/tables,
and preempt reviewer objections. Direction chosen by user: **strengthen the paper** (deepen H4,
do not open new hypotheses).

### Dataset Provenance Audit (2026-07-09) — VERIFIED, no leakage

Paper §4.1 had 8 unresolved `\TODO`s. Audited the real data pipeline
(`mld/data/EgoMotion.py`, H4 config, helma SLURM scripts) against the on-NAS data.

- **Data location**: `/home/erik/NAS/methods/diffusion_gen/data/diffusion/{ava,nuscenes,waymo}`,
  one JSON per sample (`{scene}_{obj}.json`) with fields `ego_in_ped_frame`, `ped_in_ped_frame`,
  `vectors_263`. Split-list root = `.../mean_std_txt/ava_nuscenes_waymo` (both train AND eval use
  this dir — verified in `diffusion_training_h4_trans_dec_helma.sh` and all `eval_h4_*` scripts).
- **⚠️ Investigated potential leakage**: `ava_nuscenes_waymo/train.txt` and `val.txt` are
  BYTE-IDENTICAL (11,786 names each). RESOLVED — **not leakage**. The real split is *physical by
  directory*: names resolve against `root/train/` vs `root/val/`, and those subdirs are DISJOINT
  (0 overlapping scenario files across all 3 sources). The identical name lists are just a shared
  master list; `full_dataset/` in ava (34 = 27+7) is their union. Train/val scenarios do not overlap.
- **Sample counts** (physical JSON files per split subdir): train = ava 27 + nusc 3,611 + waymo
  7,699 = **11,337**; val = ava 7 + nusc 903 + waymo 1,925 = **2,835**.
- **Loaded counts — CONFIRMED** (real `EgoMotionDataset` instantiated in conda env `mld`;
  authoritative `Loaded N samples` printout): **train = 9,540** (ava 27 / nusc 3,637 / waymo 5,876),
  **val = 2,414** (ava 7 / nusc 943 / waymo 1,464), **total = 11,954**. Val=2,414 is the eval set
  for ALL reported tables. The "not found" warnings (val 9,372 / train 2,246) are exactly the names
  living in the OTHER split's directory → re-confirms the disjoint physical split. These numbers are
  now IN THE PAPER (Table `tab:dataset`, §4.1). Standalone resolver replica matched runtime exactly.
- **Interaction score** (`_compute_interaction_weights`, verified):
  `s = ped_travel · (pct_within_5m/100) · (1 + heading_change/180)`. Used for `WeightedRandomSampler`
  (train only; floored 0.01, normalized to N) — NOT a hard filter. Paper's old "we only use samples
  that adhere to criteria" framing was INACCURATE; corrected to weighted-sampling description.
- **Interaction crop** (`_pad_or_crop_ego_motion`, verified): window centred on closest-approach
  frame $t^\star=\arg\min_t\lVert g_t-r_t\rVert$, ±25% jitter, clamped; shared index for ego+motion;
  zero-pad at end if shorter than 196.

**Paper edits made 2026-07-09**: §4.1 now has code-accurate interaction-score equation,
weighted-sampling paragraph, interaction-crop algorithm, and per-source dataset stats table
(`tab:dataset`, authoritative counts train 9,540 / val 2,414 / total 11,954).

**Paper edits made 2026-07-13**: added the "god script" data-extraction pipeline paragraph,
grounded in the verified per-sample JSON structure (`scene_id`, `object_id`,
`ego_in_ped_frame` T×3, `ped_in_ped_frame` T×22×3, `vectors_263` T×263), pedestrian-centric
frame, 20fps resample, variable T windowed to 196. Did NOT invent external-script internals
(the JSON-producing script lives in a separate data-prep repo, not this training repo).
**5 of 8 §4.1 `\TODO`s resolved.** Paper compiles CLEAN (pdflatex+bibtex, exit 0, no undefined
refs/citations). Progress report: `research/to_human/progress_report_007_paper_strengthening.html`.

**Remaining §4.1 `\TODO`s are HUMAN-ONLY** (flagged in report 007): vehicle/sensor-setup figure,
example interactive-scene figure, and the staged-scenario count — assets/numbers not in the repo.
Also flagged for human decision: AVA=34 samples framing (reviewer risk).

---

*Prior conclusion (2026-04-15): H6 definitive REJECTED — ep=3399 seg-2 eval FID=5.079/R@1=0.352/MM=2.618; H4 confirmed best; CONCLUDE decision made.*

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

**H6 Training Segment-1 Complete — 2026-04-15 (NEW FINDING: FID/R@1 TRADEOFF)**:

H6 seg-1 ran as job 365378, timed out after 24h at epoch ~3359. Last saved checkpoint: epoch=3299. Extracted full training trajectory from SLURM log (3260 metric lines, deduplicated to 33 checkpoint eval points):

| Epoch | Training-time Val FID ↓ | Training-time R@1 ↑ |
|-------|------------------------|----------------------|
| 99    | 9.835 | 0.036 |
| 499   | 8.373 | 0.038 |
| 999   | 6.011 | 0.065 |
| 1499  | 5.053 | 0.158 |
| **2099**  | **4.789** | 0.273 ← **BEST FID** |
| 2499  | 5.385 | 0.332 |
| 2999  | 5.889 | 0.373 |
| **3299**  | 5.273 | **0.398** ← **BEST R@1 so far (STILL CLIMBING)** |

> **CORRECTION (2026-06-12)**: An earlier version of this table contained wrong values at
> epochs 499/999/1499/2499/2999 (apparently smoothed/interpolated during transcription).
> The values above were re-extracted directly from the SLURM log
> `diffusion_h6_unfreeze_ego_365378.txt` and match research-state.yaml run_026.

**KEY FINDING: FID/R@1 TRADEOFF in H6**. As the ego encoder specializes its representations for cross-attention (via backprop from denoiser loss), R@1 improves steadily but FID degrades. The two optima do NOT coincide — best FID at epoch=2099, best R@1 at epoch=3299.

**H6 vs H4 at seg-1 cutoff (epoch ~3299)**:
- H6: FID=5.273, R@1=0.398 — WORSE than H4 best (FID=3.392, R@1=0.548 at epoch=3399)
- BUT H6 R@1 is strongly ascending (0.273→0.398 over 1200 epochs). H4's R@1 plateaued at 0.535.
- H6 R@1 may exceed H4's ceiling with more training — seg-2 submitted (job 368862, resume from ep=3299)

**Definitive evals completed** (3× replicated, CFG=10):

| Checkpoint | FID↓ | R@1↑ | Diversity | MM |
|---|---|---|---|---|
| ep=2099 (best train FID) | 4.952 ±0.088 | 0.263 ±0.009 | 5.957 | 2.455 |
| ep=3299 (best train R@1) | 5.311 ±0.060 | 0.348 ±0.004 | 6.034 | 2.581 |
| **H4 ep=3399 (CFG=10)** | **3.392 ±0.179** | **0.548 ±0.002** | **5.794** | **3.956** |

**H6 is currently WORSE than H4 on ALL metrics at both checkpoints.** Key pattern: MultiModality=2.5 vs H4 4.0 — the ego encoder's specialization is over-constraining generation, reducing output diversity. This is a more severe form of the discriminative-vs-generative tension than expected.

**Mechanism**: When FREEZE_EGO=False, the ego encoder's representations evolve to match the cross-attention decoder's needs. Early in training, the encoder produces general ego features (low R@1). As training progresses, the encoder becomes increasingly task-specialized — better at discriminating ego conditions for denoising (R@1 rises) — but this specialization pulls the generated motion distribution away from the natural motion distribution (FID rises). The MultiModality collapse (2.5) suggests the model is "collapsing" to near-deterministic outputs for each ego condition — the encoder encodes fine-grained discriminative information that constrains generation to a narrow region per condition.

**H6 seg-2 submitted** (job 368901, resumes from ep=3299). Seg-2 ran to END_EPOCH=5000.

> **CORRECTION (2026-06-12)**: The earlier claim that "R@1 peaked at ep=3299 and is declining"
> was based on only the first two seg-2 val points. The full seg-2 log
> (`diffusion_h6_seg2_368901.txt`) shows training-time R@1 *saturates*, oscillating in the
> 0.37–0.42 band through ep=4999 (max 0.417 at ep=4599), while FID oscillates between 5.1
> and 6.3. The H6 rejection is unchanged — both metrics stay far from H4 (FID=3.392, R@1=0.548)
> at every point in 5000 epochs.

| Epoch | Training-time Val FID ↓ | Training-time R@1 ↑ |
|-------|------------------------|----------------------|
| 3299  | 5.273 | 0.398 |
| 3399  | 5.141 | 0.380 |
| 3499  | 5.415 | 0.367 |
| 3999  | 6.221 | 0.407 |
| 4599  | 6.339 | 0.417 ← max train-time R@1 |
| 4999  | 5.561 | 0.408 |

**H6 ep=3399 seg-2 DEFINITIVE EVAL (3× replicated, CFG=10 — job 369437):**

| Metric | H6 ep=3399 | H4 ep=3399 | Delta |
|--------|------------|------------|-------|
| FID↓ | **5.079 ± 0.034** | 3.392 ± 0.179 | H6 is 50% WORSE |
| R@1↑ | **0.352 ± 0.009** | 0.548 ± 0.002 | H6 is 36% WORSE |
| MM | **2.618 ± 0.309** | 3.956 ± 0.406 | H6 is 34% less diverse |
| Diversity | 6.064 | 5.794 | H6 over-diverse |

**H6 is DEFINITIVELY REJECTED.** R@1 barely changed from ep=3299 to ep=3399 (0.348→0.352), confirming the model has plateaued far below H4's ceiling. Full unfreezing of the ego encoder causes systematic MultiModality collapse and worse results on all metrics. FREEZE_EGO=True (H4 architecture) is confirmed as the correct design choice.

**Summary of all H6 definitive evals:**

| Checkpoint | FID↓ | R@1↑ | Diversity | MM |
|---|---|---|---|---|
| ep=2099 (best train FID) | 4.952 ±0.088 | 0.263 ±0.009 | 5.957 | 2.455 |
| ep=3299 (best train R@1) | 5.311 ±0.060 | 0.348 ±0.004 | 6.034 | 2.581 |
| ep=3399 (seg-2) | 5.079 ±0.034 | 0.352 ±0.009 | 6.064 | 2.618 |
| **H4 ep=3399 (CFG=10)** | **3.392 ±0.179** | **0.548 ±0.002** | **5.794** | **3.956** |

**H6 is WORSE than H4 on ALL metrics at ALL evaluated checkpoints.** The research is concluded: H4 (FREEZE_EGO=True, cross-attention conditioning) is the correct architecture.

## Patterns and Insights

1. **Interaction-aware training helps**: Cropping to the interaction window and up-weighting
   interaction-rich samples focuses the model on the most ego-relevant motion. The FID=7.5
   result with this approach is the best we have, though the run crashed before completion.

2. **Latent dimensionality does NOT help (H3 — REJECTED)**: Doubling the latent dimension (4→8) makes FID significantly WORSE (+14.5%) with the same denoiser capacity. The diffusion model's ability to model the latent distribution is the bottleneck, not the VAE's expressiveness. Diversity increases marginally (+1.1%) but doesn't compensate. **Lesson: changing latent dimension is a bad lever for FID improvement unless denoiser capacity scales too.**

3. **R-prec is decoupled from latent dimension**: H2 and H3 have essentially identical R-prec@1 (0.671 vs 0.676 at CFG=5). This reveals that the bottleneck for conditioning quality (R-prec) is the ego encoder architecture, not the latent space.  The current `EgoEncoderPooled` (mean-pool → single token) is the likely weak link.

4. **H4 cross-attention dramatically improves FID — confirmed 49% better than H2 (DEFINITIVE)**: H4 epoch=3399, CFG=10: **FID=3.392, R@1=0.548** (3× replicated definitive eval). H2 best: FID=6.603 at CFG=5. H4 achieves **49% better FID** than H2. But R@1 is capped at 0.535–0.548 with FREEZE_EGO=True — frozen encoder cannot adapt for per-timestep cross-attention. H6 (unfreeze) seg-1 complete — FID/R@1 tradeoff observed; seg-2 running.

5. **H4 FID is insensitive to CFG (2026-04-03 — NEW FINDING)**: Unlike H2 where FID increases steeply with CFG (6.603→7.400 from CFG=5 to 15), H4 FID is essentially flat (3.968→3.617 from CFG=5 to 15). This means cross-attention conditioning does NOT cause the "over-conditioning mode collapse" observed in H2. The mechanism: in H2, higher CFG forces the model to stay close to the pooled ego token — over-constraining the generation. In H4, the ego information is already richly distributed across 196 cross-attention tokens; higher CFG reinforces a naturally richer signal, not a coarse average.

4. **Guidance scale (H5 — COMPLETED)**: CFG has a large monotonic effect on FID — **lower is better for FID**.
   CFG=5 achieves FID=6.603 vs FID=7.400 at CFG=15 (10.8% improvement). Diversity barely changes (5.74–5.78 across all scales).
   R-precision monotonically increases with CFG (0.671 at CFG=5 → 0.815 at CFG=20), so there is a genuine quality/conditioning tradeoff.
   This finding falsifies the prior assumption that FID would be optimal at intermediate CFG — the relationship is monotone in this regime.
   **Recommendation**: Use CFG=5 for FID-focused evaluation; CFG=7 for balanced FID/R-prec.

5. **FID does not monotonically improve with training**: H2 epoch trajectory peaked at 4399 (FID=6.603), regressed at 4599 (FID=8.400), partially recovered at 4999 (FID=7.510). Training loss was flat — FID oscillates independently. **Critical lesson: checkpoint selection with periodic FID validation is required; the final checkpoint is NOT the best checkpoint.**

6. **Unfreezing the ego encoder causes MultiModality collapse (H6 — DEFINITIVELY CONFIRMED)**: H6 MM=2.618 vs H4 MM=3.956 at same architecture. Evaluated across 3 checkpoints (ep=2099, 3299, 3399) — all show MM collapse (2.455–2.618) and worse FID (4.952–5.311 vs H4 3.392) and R@1 (0.263–0.352 vs H4 0.548). The ego encoder co-adapting with the denoiser learns highly discriminative per-condition representations — each ego trajectory maps to a narrow distribution of motions rather than diverse plausible ones ("mode collapse per condition"). H6 R@1 peaked at ep=3299 (training-time: 0.398, definitive: 0.348) and is now declining. **Full unfreezing is definitively the wrong approach.** FREEZE_EGO=True (H4) is the correct design, consistent with the literature (frozen CLIP in Stable Diffusion, frozen DINO in DreamBooth). IP-Adapter/ControlNet patterns (adapter on frozen backbone) would be the proper way to add conditioning flexibility — but H4 results are already strong enough for publication.

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
4. **Can a cross-attention ego encoder improve R-precision and/or FID?** → **ANSWERED**. H4 (FREEZE_EGO=True, CFG=10): FID=3.392 ±0.179, R@1=0.548 ±0.002 at epoch=3399 — **49% better FID than H2, R@1 ceiling=0.548**. H6 (FREEZE_EGO=False) definitively rejected: FID=5.079, R@1=0.352, MM=2.618 at ep=3399 (seg-2) — WORSE than H4 on all metrics at all checkpoints. R@1 ceiling with unfreezing is ~0.35 (far below H4's 0.548). FREEZE_EGO=True is the correct design choice. **H4 is the best architecture.**
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

## Research Conclusion (2026-04-15)

**CONCLUDE** decision made after H6 definitive evaluation. Evidence is sufficient for publication.

### Main Contribution
Cross-attention ego conditioning with frozen encoder (H4) achieves **49% FID improvement** over the best baseline (H2): FID=3.392 ±0.179 vs 6.603 ±0.067. This represents a significant, well-validated improvement with non-overlapping confidence intervals across 3 replications.

### Final Best System: H4 (epoch=3399, CFG=10)
| Metric | H4 Value | vs H2 Baseline |
|--------|----------|----------------|
| FID↓ | **3.392 ± 0.179** | **−49%** (6.603→3.392) |
| R@1↑ | 0.548 ± 0.002 | −18% (0.671→0.548) |
| MM | 3.956 ± 0.406 | +13% (3.503→3.956) |
| Diversity | 5.794 | +0.3% (5.779→5.794) |

### Complete Ablation Story
| System | Architecture | FID↓ | R@1↑ | MM | Status |
|--------|-------------|------|------|-----|--------|
| H2 (baseline) | trans_enc, EgoEncoderPooled, FREEZE=True, CFG=5 | 6.603 | 0.671 | 3.503 | BASELINE |
| H3 | trans_enc, latent-8, FREEZE=True | 7.563 | 0.676 | 3.245 | REJECTED (worse FID) |
| H4 | trans_dec, EgoEncoder×196, FREEZE=True, CFG=10 | **3.392** | **0.548** | **3.956** | **BEST** |
| H6 | trans_dec, EgoEncoder×196, FREEZE=False, CFG=10 | 5.079 | 0.352 | 2.618 | REJECTED (worse all) |

### Trivial Baselines (2026-06-12, RERUN under matched config — supersedes earlier same-day numbers)
Four non-learned / reference baselines establish the lower and upper bounds of the metric space.

> **CORRECTION / RERUN (2026-06-12 evening):** the first 2026-06-12 baseline numbers
> (retrieval 0.042, uncond 8.90, interp 12.18, kinematic 54–56) were computed under the
> WRONG pipeline: `MEAN_STD_PATH=ava_human_nuscenes_waymo` (old normalization AND old
> split lists — MEAN_STD_PATH doubles as split_list_root in EgoMotion.py), and the uncond
> baseline used the old H0 model (`ego_motion_diffusion_all_new_vae_stochastic` ep1599,
> latent-1, different VAE). gt_Diversity ≈5.7–5.9 instead of the main table's ≈5.50
> exposed the mismatch. Also, the recorded uncond FID=8.90/MM=5.25 did not match its own
> JSON (9.088/5.423). All four baselines rerun with `MEAN_STD_PATH=ava_nuscenes_waymo`
> (the directory the helma eval scripts override to for ALL main-table evals) and the
> uncond baseline now uses the actual H4 model (trans_dec, latent-4, ep=3399, ego zeroed).
> Sanity check: uncond gt_Diversity = 5.515 ± 0.081 — consistent with main-table 5.496. ✓

All rows now 3 replications (2026-06-12: non-learned baselines rerun with seeds 1234/2345/3456;
CI = 1.96·σ/√3, same formula as test.py; randomness comes from the dataset's stochastic
pad/crop + metric subsampling + interpolation draws):

| Baseline | What it does | FID↓ | Diversity | gt_Div | R@1 | MM |
|----------|--------------|------|-----------|--------|-----|-----|
| Retrieval (NN ego) | copy training motion of nearest-ego-trajectory neighbour | **0.062 ±0.004** | 5.34 ±0.04 | 5.34 ±0.06 | — | — |
| Unconditional H4 | H4 ep3399 with ego zeroed (CFG cancels exactly) | 5.18 ±0.21 | 5.96 ±0.13 | 5.51 ±0.08 | 0.031 (=chance 1/32) | 5.16 ±0.63 |
| VAE latent interp | decode random interpolation of two train latents | 7.91 ±0.03 | 5.82 ±0.01 | 5.33 ±0.06 | — | — |
| Traj+kinematic (oracle) | GT root features + mean body (zeros) | 47.67 ±0.08 | 0.72 ±0.00 | 5.36 ±0.06 | — | — |
| Traj+kinematic (mean) | training-mean motion (all zeros) | 49.60 ±0.09 | 0.46 ±0.01 | 5.36 ±0.06 | — | — |
| H2 (pooled) | — | 6.603 | 5.78 | — | 0.671 | 3.50 |
| **H4 (ours)** | — | **3.392** | 5.79 | — | 0.548 | 3.96 |

**What the baselines establish (REVISED with matched numbers):**
- **Retrieval FID≈0 is a sanity check, not a competitor**: copying real training motions trivially matches the GT distribution (FID 0.062), but retrieval is not generative and has no ego-conditioned R-precision. FID alone is gameable by memorization.
- **Unconditional H4 (FID=5.18) is the key reference**: it is exactly the H4 model with ego removed. Ego conditioning improves FID 5.18→3.39 (**34%**, not the previously claimed 62%). STRIKING: the H4 *unconditional* prior (5.18) already beats H2's *conditional* FID (6.60) — the trans_dec architecture is a better generative model even before conditioning. It also ≈matches H6's conditional FID (5.08).
- **Uncond R@1 = 0.03125 = exactly 1/32 = chance**: protocol sanity check — without ego info, retrieval is random.
- **VAE interp (7.91)** worse than uncond prior (5.18): arbitrary latent interpolation leaves the data manifold; the learned diffusion prior matters.
- **Traj+kinematic (48–50)** is the floor: following the trajectory with a static/mean body is catastrophic (near-zero Diversity 0.5–0.7) — the task genuinely requires articulated body motion, not just root translation.

**Implementation note (bug fixed 2026-06-12, morning):** the three non-model baseline scripts
(`eval_retrieval`, `eval_vae_interp`, `eval_traj_kinematic`) originally built
`EgoMotionDataModule` directly with `mean=None, std=None`, so motions were fed to the
t2m evaluator **unnormalized** → collapsed embeddings. Fixed via `get_datasets()` +
`pl.seed_everything`. The Feb-26 JSONs and the first Jun-12 JSONs are both stale;
current numbers (evening rerun, matched pipeline) above.

### Paper Narrative
The paper tells a clean 3-part story:
1. **Architecture**: Cross-attention over the full ego trajectory sequence (H4) dramatically outperforms single-token pooled conditioning (H2): 49% FID improvement.
2. **Encoder freeze ablation**: Unfreezing the ego encoder (H6) causes MultiModality collapse (2.618 vs 3.956) and worsens both FID and R@1 — consistent with the literature finding that frozen encoders are critical for generative diversity (CLIP in SD, DINO in DreamBooth).
3. **Design recommendation**: Frozen ego encoder + cross-attention denoiser is the correct architecture for ego-conditioned pedestrian motion generation.

The R@1 gap (H4: 0.548 vs H2: 0.671 at CFG=5) is real but expected — cross-attention generates more diverse motions per condition (MM: 3.956 vs 3.503), which naturally increases the retrieval difficulty. When evaluated at the same CFG (CFG=10), H4 R@1=0.548 vs H2 R@1=0.767 — there is a genuine conditioning-alignment gap that future work (adapter approaches) could address.

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
