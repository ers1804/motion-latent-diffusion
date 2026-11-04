# Action plan: Fabian Flohr's Overleaf review (2026-10-28 to 2026-10-30)

Status: agreed with Erik on 2026-10-30. Source: 25 Overleaf comment threads on the Overleaf `main.tex`.

## Decisions taken

| # | Question | Decision |
|---|---|---|
| 0 | Source of truth | Overleaf `main.tex` is the master |
| 0 | Venue | IEEE Open Journal of Intelligent Transportation Systems (OJ-ITS); RA-L dropped |
| 1 | Re-split and retrain? | Yes: scene-level split, retrain everything on it |
| 2 | Title / body orientation | Test orientation on labels first; retitle to trajectory-level wording if the labels show nothing |
| 3 | If Pooled wins the matched-recipe test | Adapt the contribution then |

## Why the science comes first

Flohr's "Pooled without IA is missing" comment is more serious than a missing row. EgoPed's lead over
Pooled compares two different training recipes. With interaction sampling (IA) on both sides, Pooled wins:

| Pooled vs full-sequence, both with IA (held-out dumps, old split) | Pooled's advantage | p |
|---|---|---|
| Mean ADE, four EgoPed-IA checkpoints | 0.14 to 0.20 m | <0.001 |
| Stop/walk separation, same checkpoints | 0.06 to 0.16 | <=0.014 |

Pooled without IA has never been scored per condition. IA cost EgoPed 0.30 m, so Pooled without IA could
beat EgoPed (an extrapolation, not a measurement).

Flohr's leak comment also goes further than he has seen:

- Training-time validation used the full validation set, which includes the "held-out" half.
- The two validation halves share 143 of 430 real scenes and 102 mirrored pedestrian pairs. "521 scenes"
  counts mirrored scenes as separate scenes.
- `M*.json` files are left-right mirrored copies, split independently of their originals: 83% of held-out
  samples have their mirrored twin in training, and only 8 come from a scene with no training data.
- Twins and same-scene pedestrians are not independent, so the paired bootstrap over conditions is
  optimistic. Bootstrap over scenes from now on.

## Phases

### Phase 0: basis

1. Sync Overleaf into the repository as the master.
2. Move to the official OJ-ITS template (`ieeeoj.cls`, IEEE template selector, `ojits-latex-template.zip`).
   The journal page states no hard page limit and recommends at most 10 pages; Flohr says 10 to 12 for the
   main part. The draft is 18 pages, so cutting is a Phase 2 task.

### Phase 1: experiments, before any claim changes

3. Score Pooled without IA (checkpoint exists, FID 6.214) on ADE and separation; fill Tables 7 and 8.
4. New split by real scene, mirrored twins kept together: train, selection, test. Select checkpoints on
   the selection half only, report on the test half. Rebuild statistics, VAE, ego encoders, diffusion models.
5. Test the "baseline's best seed" claim: definitive evaluation of the other two Pooled seeds at epoch 99
   (training-time FID 5.86 and 5.21; checkpoints exist on the cluster). Otherwise drop the phrase.
6. Add PedGen as an external baseline if feasible. Confirm the checkpoints are public and whether its
   required scene context is available in our data.
7. Body orientation toward the vehicle (decision 2): test on labels before any model.

### Phase 2: claims and framing

8. Three contributions, each stated against prior work. Provisional wording, pending Phase 1:
   - Generation that responds to another moving agent, where prior work conditions on text, waypoints,
     goals or scene.
   - An evaluation protocol showing FID is blind to conditioning.
   - A released dataset with a pose-label audit, where prior work filters noisy labels without measuring them.
9. Resolve the FID conflict: abstract and contribution 2 lead with a 40% FID gain, then disqualify FID.
   Lead with per-condition results.
10. Soften "first" to "to our knowledge" after reading ControlPed in full.
11. Cite and position Trace and Pace (arXiv 2304.01893), PedGen (arXiv 2410.07500), ControlPed
    (arXiv 2610.06171, concurrent).
12. Frame Unfrozen as a trade-off. Old-split evidence, EgoPed vs Unfrozen: mean ADE 2.360 vs 2.262 (p=0.013);
    minADE 1.306 vs 1.535 (p<0.001); Brier 0.147 vs 0.178 (p<0.001); separation 0.334 vs 0.351 (n.s.).
13. Name the metric. Only separation is new. Suggested name: Ego-Coupling Score.
14. Robotics spill-over in the introduction, picked up in discussion and outlook. Label it as outlook:
    we have no robot data.
15. Cut to the OJ-ITS length.
16. Remove claims the abstract check found unsupported: "scene-disjoint", "both significant", "critical".

### Phase 3: figures and dataset

17. Teaser image on the first page.
18. More qualitative results, including Waymo, the self-recorded data and failure cases. Confirm which
    scenes the current figure uses and that they lie in the new test split.
19. Fix the EgoPed training-curve figure: top unreadable, best-checkpoint note on the x-axis unreadable.
20. Rename the dataset ("AVA" also clashes with Google's AVA dataset) and build the teaser page on
    iv.ee.hm.edu. Needs Erik and Flohr.

### Phase 4: text fixes

21. Remove the photo copyright line (Flohr confirmed).
22. Fix "Iits" and "breaked" (should be "braked").
23. Fix the stale ladder numbers (2.99, 2.32, 0.04 against 3.01, 2.36, about 0.05), cite the specific table,
    sweep the paper for similar values. All numbers change with the new split anyway.
24. Explain the dimensions: records store the ego trajectory in 3D, the model uses 2D.
25. Consistent reference abbreviations (assuming "first rev" means "first reference"; confirm).
26. Acknowledgments and funding: details needed from Erik or Flohr.
27. Abstract wording: "a benchmarked consisting" is a typo; the sentence says the audit uses one skeleton
    when the labels do.

## Comment-to-action map

| Flohr's comment | Action |
|---|---|
| Motivation / teaser image on first page | 17 |
| Summarize to max 3; differentiate from SoA | 8 |
| More qualitative results, also failures; also Waymo, AVA | 18 |
| Emphasize spillover to robotics (Discussion) | 14 |
| Acknowledgments / funding | 26 |
| Consistency, first "rev" has no abbreviations | 25 |
| AVA: nobody knows this; name it, teaser on iv.ee.hm.edu | 20 |
| Is there a name for this new metric? | 13 |
| Is this now good wrt 3? | 9 |
| + Spill-over to robotics (Introduction) | 14 |
| Can you claim to be the first? | 10 |
| Venue? RA-L? Page limits; correct template first | 2 |
| Photo copyright: can be removed | 21 |
| Pooled without IA missing in Tables 7 and 8 | 3, 4 |
| Closest prior work missing; PedGen baseline | 6, 11 |
| Unfrozen is rather a trade-off? | 12 |
| Best ckpt note and top of figure unreadable | 19 |
| Title vs body: test orientation; do labels show it? | 7 |
| Leak: pick checkpoints on one half, report on the other | 4 |
| Numbers differ from Table 5; cite the specific table | 23 |
| Dimensions: compare e in R^{196x2} | 24 |
| "Iits" typo; "breaked" | 22 |
| Do not put 40% FID first when disqualifying the metric | 9 |
| Teaser page please | 20 |

## Progress log

### Phase 0: done (2026-10-30)

Overleaf text taken as master (6096ae5). Paper moved to the official OJ-ITS template (5448f1a): compiles
with no errors, 21 pages before any cutting. Affiliations, funding and biographies are still red TODOs.

### Phase 1: in progress

**Step 4, scene split.** `research/data/scene_split_v1` is built by `research/src/make_scene_split.py`
from the same 11,786 names as before.

| Partition | Files | Scenes | nuScenes | Waymo | AVA |
|-----------|-------|--------|----------|-------|-----|
| train     | 9,410 | 796    | 3,594    | 5,794 | 22  |
| val (selection) | 1,194 | 107 | 472 | 716   | 6   |
| test (reporting) | 1,182 | 91 | 448 | 728  | 6   |

- No scene or mirrored pair crosses a partition; checked through the real dataset class by
  `research/src/test_scene_split.py`.
- Statistics are computed on train only. The same formula on the old list reproduces the old files
  (mean within 0.024, std within 4.8%). Ego x mean is now exactly 0, as expected when both mirror
  images of every sample stay together; the old value was 0.04.
- `SCENE_SPLIT=True` resolves names exactly and fails on a miss. Held-out partitions always use fixed
  uniform windows and never interaction cropping, so a training recipe cannot change what selection
  or testing sees.
- Old loader bug, now bypassed: a listed name missing from the split folder fell back to a same-named
  file of another source. This is why the old split loaded 9,540/2,414 samples against 9,456/2,330 files.
- Windows no longer depend on the machine, so cluster dumps stay aligned with the existing local dumps.

**Jobs.** `slurm/resplit/`: VAE (latent 4, 6,000 epochs) then full-sequence and pooled encoders, then
diffusion: three seeds (1234, 2345, 3456) for each of EgoPed, EgoPed-IA, Pooled, Pooled-IA, and one seed
each for Unfrozen, Uncond, Uncond-IA. Both encoders are trained without IA, so encoder recipe is not a
confound. Selection stays validation FID, with validation ADE reported at the selected checkpoint.
Outputs go to `models_resplit`; the old models are untouched. A one-epoch smoke job of every recipe
runs first.

**Step 3, Pooled without IA on the old split: full-sequence no longer wins on mean ADE (decision 3).**
`research/src/matched_recipe.py`, log in `research/data/matched_recipe.log`; bootstrap clustered by the 430
scenes. Deltas are EgoPed minus Pooled, positive when EgoPed is better; ** marks a 95% CI that excludes 0.

| Recipe | Model | FID | ADE | minADE | Separation | Brier |
|--------|-------|-----|-----|--------|------------|-------|
| No IA | EgoPed, 3 seeds | 3.39 / 4.72 / 4.80 | 2.360 / 2.375 / 2.258 | 1.306 / 1.321 / 1.421 | 0.334 / 0.325 / 0.311 | 0.147 / 0.167 / 0.163 |
| No IA | Pooled, 1 seed | 6.214 | 2.233 | 1.416 | 0.299 | 0.176 |
| No IA | delta by EgoPed seed | | -0.126** / -0.142** / -0.025 | +0.110** / +0.095** / -0.005 | +0.035 / +0.026 / +0.012 | +0.029** / +0.009 / +0.014** |
| IA | delta, best seed 1 checkpoint vs Pooled-IA | | -0.203** | -0.239** | -0.122** | -0.009 |

- Mean ADE favours Pooled at both recipes. minADE and Brier lean to EgoPed without IA, not for every seed.
  Separation is tied without IA and favours Pooled with IA.
- With IA, Pooled wins for all four EgoPed-IA checkpoints on ADE. So "full-sequence conditions better" does
  not hold as a general statement on this split. EgoPed's clear win is FID, which cannot see conditioning.
- Robust: either trajectory encoder beats the trained ego-blind prior (ADE +0.78 and +0.65, separation
  +0.25 and +0.29, all CIs exclude 0). The interaction recipe hurts mean ADE in both architectures
  (+0.22 and +0.30 without it).
- On the 202 twin-free samples the mean-ADE differences lose significance, while EgoPed's separation and
  Brier lead without IA is significant for seeds 1 and C.
- Only one Pooled seed exists without IA, on a leaky split. The contribution is rewritten after the retrained
  2x2 with three seeds per cell, not now.

**Step 5, "baseline's best seed".** Definitive FID at each seed's best epoch: Pooled-IA 5.620 (epoch 4399),
8.618 and 8.813 (epoch 99, CFG 5, gt_Diversity 5.55). The claim holds with a caveat: seeds B and C were
stopped near epoch 2,400 with FID still 10.9-12.1, so they did not get the same horizon. Their ADE is
actually low (2.261, 2.211), another sign that FID and per-condition quality rank differently.

**Step 6, PedGen.** Feasible but not like for like. Code and checkpoints are public
(`pedgen_no_context`, `pedgen_with_context`); the repository declares no license. It generates 60 frames
from an initial position, against our 196 frames with no initial pose, and it has no ego input. Full
context needs depth and segmentation from images. Decided: `pedgen_no_context` as an ego-blind comparison on
short-horizon root metrics only.

Result on the old held-out split, first 1.95 s (40 frames at 20 fps), 5 samples per condition, 15,000-sample
bank; `research/src/pedgen_baseline.py`, `research/src/short_horizon.py`, log in
`research/data/short_horizon_val_test.log`.

| Model | ADE | FDE | minADE | minFDE |
|-------|-----|-----|--------|--------|
| EgoPed, 3 seeds | 0.588 / 0.582 / 0.568 | 1.171 / 1.159 / 1.129 | 0.316 / 0.323 / 0.350 | 0.617 / 0.630 / 0.687 |
| Pooled | 0.550 | 1.090 | 0.333 | 0.655 |
| Trained prior, ours | 0.749 | 1.497 | 0.284 | 0.554 |
| PedGen no-context | 0.809 | 1.592 | 0.437 | 0.829 |

Every model here beats PedGen on ADE, FDE and minADE (95% CIs exclude 0). Treat this as a reference, not a
ranking: PedGen was trained on other data and walks more slowly (median 0.65 m/s against 1.28 m/s in our
labels), which alone explains part of the gap. The trained prior beating it shows the data matters, not only
the ego input. To be rerun on the new test split.

**Step 7, orientation: no evidence, so the wording changes (decision 2).** Result of
`research/src/orientation_test.py`, log in `research/data/orientation_test.log`.

- Training labels: facing and walking direction agree (cos 0.984), torso twist is 3.3 deg median, and the
  22-joint labels hold no head yaw. Standing within 15 m, facing the vehicle is 0.175 against 0.159 shuffled.
- Raw SMPL fits do hold head rotation and pass the sanity checks (body up 0.98 on both datasets). The link
  to the vehicle gets stronger with distance, the opposite of attention:

| Dataset | r(head turn, vehicle side), <15 m / >30 m | cos(head, vehicle) standing, <15 m / >30 m |
|---------|------------------------------------------|--------------------------------------------|
| nuScenes | +0.399 / +0.509 | +0.173 / +0.356 |
| Waymo    | +0.136 / +0.410 | +0.124 / +0.315 |

  This fits a camera-facing bias of the monocular fits, since the camera is on the vehicle. It does not
  rule out a small real effect nearby, but these data cannot show one, and the model could not generate
  head turns from the current labels anyway. The paper therefore claims trajectory-level behaviour only.
  The retitling happens in Phase 2, together with the claims rewrite.

### Decisions from Erik, 2026-10-30

Encoder recipe, three seeds per 2x2 cell, selection rule and PedGen scope as recorded above. Erik ports repo
changes to Overleaf himself. Affiliation and funding statement are in `main.tex`; biographies come later.

### Text fixes done (2026-10-30)

Actions 21 (photo copyright), 22 (typos), 24 (dimensions, now linking the 3D record to the 2D input), 25
(references in IEEE abbreviated form, extra arXiv notes removed from published entries), 26 (funding) and 27
(abstract wording) are done. Action 10: "first" now reads "to our knowledge" after reading ControlPed in full.
Action 11: new Related Work subsection on Trace and Pace, PedGen and ControlPed, references harvested from
Crossref and arXiv. Fabian's "first rev" is taken to mean the first reference (Erik does not know either).

Action 14: robotics spill-over added to the introduction and the discussion, labelled as untested outlook.
Action 19: legend and best-checkpoint note in the EgoPed training-curve figure no longer overlap the data or
the axis; the data in `make_h4_curve.py` still has to be replaced by the retrained run.

### Prepared for after training

- Selection rule, fixed now: lowest training-time validation FID among saved checkpoints
  (`research/src/select_resplit.py`, reproduces epochs 3099 and 1799 on the old seeds B and C). Unconditional
  runs are selected offline.
- `slurm/resplit/eval.sh`: definitive FID and K=5 root dump for one checkpoint, guidance and partition. Ran end to
  end on early checkpoints of three runs (val only, 5-7 min per job); the zero-ego model scores ECS 0.02, as an
  ego-blind model should.
- Unconditional selection: `slurm/resplit/select_uncond.sh` scores every saved checkpoint with the ego zeroed. Queued
  to start after each unconditional run's last segment (jobs 1199192, 1199193).
- Guidance: one value per cell, lowest validation FID averaged over its seeds, from {5, 10, 15}.

Steps once the training queue is empty (from the repository root on Helma; check every run reached epoch 4999 first,
and queue another segment for any that did not):

1. `python3 research/src/select_resplit.py --out research/data/resplit_epochs.json`
2. `bash slurm/resplit/submit_evals.sh sweep research/data/resplit_epochs.json`
3. `python3 research/src/select_resplit.py --guidance research/data/resplit_epochs.json --out research/data/resplit_selection.json`
4. `python research/src/compare_resplit.py --split scene_val --selection research/data/resplit_selection.json`
   (validation only; take this to Erik before the test pass)
5. `bash slurm/resplit/submit_evals.sh test research/data/resplit_selection.json`, then step 4 with `--split scene_test`.

### Chain status

Submitted 2026-10-30 (jobs 1187329-1187379, 51 jobs). VAE finished at epoch 5999; encoders finished with best val
MSE 1.915 (full-sequence) and 1.904 (pooled). All 15 diffusion runs started 2026-11-02 around 02:05 CET. Their
dumped configs differ only as intended, and gt_Diversity at the first validation is 5.29-5.95 on the new partition
(the smoke job gave 5.42), so normalization is correct.

Pace after the first validation: EgoPed 143 ep/h, EgoPed-IA 96, Pooled 78, Pooled-IA 52. Pooled ran with 4 data
workers against 8, and at 52 ep/h three segments would end near epoch 3,700. The config now sets 8 (117e79a), which
later segments read at start, so every run should reach epoch 5000 within its three segments. Expected end is
about 2026-11-04 evening, queue permitting.

### Comparison rule, fixed before any test result

`research/src/compare_resplit.py` scores the 2x2 from the K=5 dumps. Seeds of one cell are averaged and the
bootstrap resamples scenes. A model beats another only if the 95% CI excludes 0 and all seed pairs agree in sign,
because the bootstrap alone does not capture seed variance. The primary contrast is EgoPed vs Pooled without IA on
mean ADE; every other contrast is secondary. Any other outcome is reported as no measurable difference. What that
means for the contribution goes to Erik before the claims change.

### Validation result of the 2x2 (2026-11-04)

All 15 runs reached epoch 5000. Seven died on 2026-11-02 when /hnvme filled up and resumed from their last intact
checkpoint (5a38ad4). Selection per the rules above: `research/data/resplit_epochs.json` (checkpoints) and
`research/data/resplit_selection.json` (guidance: EgoPed 15, EgoPed-IA 10, Pooled 15, Pooled-IA 5, Unfrozen 5,
unconditional 1). Val N=1194 conditions, 107 scenes. Delta > 0 means the first cell is better; * meets the rule.

| Contrast | ADE (m) | minADE (m) | ECS |
|---|---|---|---|
| EgoPed vs Pooled | -0.370 [-0.494,-0.246] * | +0.049 (n.s.) | -0.065 [-0.103,-0.025] * |
| EgoPed-IA vs Pooled-IA | -0.012 (n.s.) | +0.048 (n.s.) | -0.069 [-0.136,-0.009] * |
| EgoPed vs EgoPed-IA | +0.043 (n.s.) | +0.395 * | +0.134 * |
| Pooled vs Pooled-IA | +0.401 * | +0.395 * | +0.129 * |
| each conditioned cell vs its unconditional prior | +0.25 to +0.80 * | prior better (diversity) | +0.15 to +0.34 * |

Cell means: Pooled ADE 2.27 / ECS 0.36; EgoPed 2.64 / 0.30; Pooled-IA 2.67 / 0.23; EgoPed-IA 2.68 / 0.16; unconditional
prior 3.07 / 0.02. Val FID (mean over seeds at the chosen guidance) runs the other way: EgoPed-IA 2.20, EgoPed 2.74,
Pooled-IA 3.87, Pooled 3.90. Pooled also wins at matched guidance 5 and 10 (ADE -0.22, -0.30; ECS -0.07, -0.08).
Unfrozen: R@1 at chance for the entire run and val FID 5.4-17, so the rule picks epoch 99; ECS 0.05.

Reading: conditioning on the ego trajectory works, the pooled summary is at least as good as full-sequence
attention on every per-condition metric and better on ECS in both recipes, and the IA recipe buys FID with ego
coupling. FID ranks the cells almost exactly opposite to the per-condition metrics. The test pass waits for Erik.

### Decisions from Erik, 2026-10-31

The metric is the Ego-Coupling Score (ECS), applied in `main.tex`. The dataset keeps its working name until Erik
gives one. The teaser is ours to propose and should show what the method does.

### Teaser (action 17): proposal in the paper, data to be replaced

`figures/teaser.pdf` sits on the first page, in the title block below the index terms, because a `figure*` cannot
float onto page 1. One model and one noise seed run under two ego trajectories: a vehicle driving past
(generated pedestrian stays, 0.2-1.0 m over 8 seeds) and a vehicle slowing to a stop (generated pedestrian walks,
11.8-17.8 m over 8 seeds). No street is drawn because the data has no map. Caption says the pair was chosen for
illustration.

The motions come from the old model (`h4`, old split, rows 137 and 333 of `val_test`). After the retrain, regenerate
from `rs1_egoped` on `scene_val`, so the test split stays untouched:

1. `teaser_generate.py --scan --split scene_val --dump <K=5 dump>`, then pick a pair with full length, ground truth
   agreeing, all 5 samples agreeing and the vehicle within about 8 m.
2. `teaser_generate.py --rows A B --seeds 8`; keep the pair only if all 8 seeds agree.
3. `render_teaser.py` once per row, then `make_teaser.py`. The footer overlap depends on title and abstract length,
   so check page 1 after the rewrite (the crop in `make_teaser.py` can relax if there is room).

No simple vehicle rule predicts stopping on the old split: generated stop rate against ego mean, start and end speed
correlates at r = -0.05, -0.09, -0.01 (ground truth -0.07, -0.11, -0.03). Do not write "the vehicle brakes, so the
pedestrian walks" anywhere; ECS is the supported claim.

### Still open

Waiting on results: actions 8, 9, 12, 15, 16, 18, 23, the title, and regenerating the teaser. Waiting on people: 20
(dataset name, teaser page), biographies.
