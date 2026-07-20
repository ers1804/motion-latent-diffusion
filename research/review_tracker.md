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
- **Status**: agent implementing. Regressor first (cheap), MDM-style second.

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
- **Plan**: implement root-trajectory ADE/FDE (generated vs GT, via `recover_from_ric`), plus
  min-of-K variant for stochastic models; evaluate H2/H4/H6 + uncond. Pure eval work, no training.
- **Status**: agent implementing now (no decisions needed).

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
- **Plan**: (a) rerun H2/H3/H6 (+H4 3-rep already done) on val_test → complete held-out table;
  (b) implement Welch t-test + bootstrap on per-replication metrics; (c) add held-out column or
  appendix table to paper.
- **Status**: agent running now (evals + stats tooling; no decisions needed).

## Structural / disclosed (no action beyond awareness)
- AVA = 34 samples framing ("targeted high-interaction subset") — recommended, needs user sign-off.
- "First system" claim — qualified to full-body 3D + ego odometry; defend boundary vs WoSAD-style 2D work.
- **Reproducibility landmine**: `configs/*trans_dec.yaml` silently trains `trans_enc`
  (module-merge bug); run dirs named `h4_trans_dec` contain trans_enc models. MUST fix configs +
  naming before any code release.
- Paper `\TODO`s remaining (user): vehicle/sensor figure, example-scene figure, staged-scenario count.

## Compute ledger (planned training runs)
| Run | Item | Est. wall (4090 / H100) |
|---|---|---|
| H2 seed B, C | 2 | ~2×20h / ~2×12h |
| H4 seed B, C | 2 | ~2×17h / ~2×10h |
| H4 no-weighting | 4 | ~17h / ~10h |
| H4 random-crop | 4 | ~17h / ~10h |
| H4 random-init frozen ego | 6 | ~17h / ~10h |
| External baseline (TBD) | 1 | TBD |
