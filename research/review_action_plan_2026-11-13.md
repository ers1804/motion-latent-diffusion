# Action plan: pre-submission review of 2026-11-13

Working plan for Erik and Claude, not for circulation. Source: the AI-generated review Fabian Flohr sent on
2026-11-13 (`paper.skill`, "borderline accept"). Its header names IEEE IV 2027 by mistake. The target stays
IEEE RA-L, `research/paper_ral/`, 8 pages including references.

## Decisions (Erik, 2026-11-13)

| # | Question | Decision |
|---|---|---|
| 1 | Venue | RA-L. "IV 2027" in the review header was a mistake |
| 2 | Narrow the claim from "responds to the vehicle" to "follows the vehicle's motion context" | Yes |
| 3 | Two more seeds for the prior and the prior with IA | Yes, without touching the existing Helma queue |
| 4 | Ego-aware baseline | Our deterministic ego-to-motion regressor, retrained on the scene split. Not Trajectron++ |
| 5 | Staged recordings | The pedestrian is a co-author and consented |
| 5 | Root source | The root comes from the pose fit. Cameras are stitched through the annotated 3D box |
| 6 | Brier column of Table II (asked 2026-11-13, after the test results) | Replace it by the noise-corrected score |
| 7 | Retrain with the ego in the frame of the window start | Yes, queue it on Helma without touching the existing queue |

## Verdict on the review

| Point | Verdict | Action |
|---|---|---|
| Flaw 1: response to new ego plans untested | Valid, but the proposed mechanism (vehicle reacts to the pedestrian) is not the carrier | A1, B1, C3, text |
| Flaw 2a: no ego-aware baseline | Valid. Trajectron++ needs the pedestrian's past, which the task withholds | C2 |
| Flaw 2b: body quality not evaluated | Valid | C4 |
| Flaw 3a: single-seed priors | Valid. "Every model is trained with three seeds" is false as written | C1, text |
| Flaw 3b: correlation over 15 dependent runs | Valid | A2 |
| Unresolved citations | Not in our PDF. The reviewed copy had an outdated bibliography | Erik syncs `refs.bib` and `ral_ctl.bib` to Overleaf |
| 11,786 counts mirrored copies | Valid. 5,893 recordings, each with one mirrored copy in the same partition | text |
| Full-seq MM and Brier gaps in no table, gaps differ from Table IV | Valid. The gaps are right before rounding | MM column in Table II, text |
| Draft leftovers (placeholder figure, note to Mohan) | Known, waiting for Mohan | none |
| Consent for the staged recordings | Valid | one sentence |

Questions to the authors:

- Share of stops that stand throughout, ECS for those who move and stop: A1.
- The trainable Full-seq run stays a single seed, labelled as such. It is a side remark, not a contrast.
- Rankings under selection by validation ADE: not planned. The guidance section and the last-checkpoint rows
  already show the ranking under two other selections. Revisit only if a reviewer insists.
- Root accuracy: never checked against a reference. The text must stop calling the root "reliable" and say
  where it comes from. A check against the annotated box tracks needs the extraction repository (optional, D2).
- One frozen ego-motion evaluator for R-Precision: declined. ADE and ECS already score per condition, and an
  evaluator trained on these labels inherits their limits.

## What the validation checks showed (1,194 conditions, no test access)

Script: `research/src/coupling_breakdown.py`, constants fixed on validation on 2026-11-13. Output:
`research/data/coupling_breakdown_scene_val.txt`. Geometry is computed in the window frame (see the pipeline
facts below).

| Finding | Number |
|---|---|
| Stops that stand throughout (never leave a 1 m radius) | 161 of 272 (59%) |
| ECS on those who move and stop, EgoPed against prior | 0.18 [0.08, 0.32] against 0.02 |
| Pedestrian never within 2.5 m of where the vehicle is or later drives | 1,172 of 1,194 (98%) |
| ECS on those conditions, EgoPed against prior | 0.37 [0.25, 0.48] against 0.03 |
| Median closest approach | 10.9 m |
| Recorded stop rate, vehicle travelling within 10 degrees of the pedestrian's facing axis | 0.07 (506 conditions) |
| Recorded stop rate at 60 to 90 degrees | 0.46 (179 conditions) |
| ECS within angle bins | 0.29 [0.16, 0.45] of 0.36 |
| Rule "stops unless the vehicle travels within 10 degrees of the facing axis" | ECS 0.38, Brier 0.41, AUC 0.69, predicts stopping for 58% |
| EgoPed samples | ECS 0.36, Brier 0.17, AUC 0.75, stop rate 0.20 (recorded 0.23) |
| Angle-bin stopping rate, out of fold | ECS 0.10, Brier 0.17, AUC 0.66 |

## What the test partition showed (2026-11-13, 1,182 rows)

`research/data/coupling_breakdown_scene_test.txt`. Run once at 295937b. A second run at 8805df3 added two
derived lines (noise-corrected Brier score, metrics without replaced rows) and reproduced every earlier line.

| Finding | Test | Validation |
|---|---|---|
| Stops that stand throughout | 161 of 284 (57%) | 161 of 272 (59%) |
| EgoPed stop probability: stands / moves and stops / walks | 0.55 / 0.30 / 0.13 | 0.60 / 0.30 / 0.11 |
| ECS on moves and stops, EgoPed minus prior | +0.12 [-0.02, +0.23], rule not met | +0.17 [+0.06, +0.32], rule met |
| Never within 2.5 m of the vehicle's path | 98.7% | 98.2% |
| ECS there, EgoPed minus prior | +0.30 [+0.19, +0.39], rule met | +0.34 [+0.22, +0.46], rule met |
| ADE gain there, EgoPed minus prior | 0.68 m, rule met | 0.80 m, rule met |
| Stop rate at 0 to 10 degrees / 60 to 90 degrees | 0.09 / 0.40 | 0.07 / 0.46 |
| ECS within angle bins, of overall | 0.17 of 0.31 | 0.29 of 0.36 |
| Rule: ECS, Brier, share predicted to stop | 0.29, 0.47, 65% | 0.38, 0.41, 58% |
| AUC: EgoPed / angle-bin rate / prior | 0.70 / 0.65 / 0.52 | 0.75 / 0.66 / 0.54 |
| Brier without the K=5 sampling noise: EgoPed / Full-seq / prior | 0.180 / 0.162 / 0.179 | 0.156 / 0.157 / 0.171 |
| Brier of a constant stop rate | 0.183 | 0.176 |
| ECS on windows with a stale frame / on the others | 0.26 / 0.35 | 0.22 / 0.55 |

What is new against validation:

- Among pedestrians who move and then stop, EgoPed's lead over the prior does not meet the rule on test.
  The score is carried by pedestrians who stand.
- The angle explains more on test (0.31 falls to 0.17 within bins) than on validation.
- With the sampling noise removed, EgoPed's Brier score on test is level with the prior and with a constant
  stop rate. The Brier column of Table II is computed from five-sample frequencies, which penalises models
  whose samples spread (the prior most). Erik decided to replace the column (decision 6): Tables II and V
  now show the corrected score (`BrierC` in `compare_resplit.py`), and Section V says that the correction
  was adopted after the test pass and gives the uncorrected values of EgoPed and the prior. With it,
  Full-seq beats EgoPed on test under the rule (0.162 against 0.180, level on validation), which the
  encoder paragraph now states. Frozen against trainable encoder still meets the rule on both partitions.

Reading:

1. The vehicle reacting to the pedestrian cannot be what carries the score. Almost no pedestrian is ever in
   the vehicle's path.
2. For the same reason the data holds almost no interaction, so the score is not a measured reaction either.
   It is a statistical dependence on the vehicle's motion and on the geometry that motion reveals.
3. About a fifth of the score is the angle between the road direction, as shown by the vehicle, and the
   direction the pedestrian faces. Walkers face along the road, people who stand face anywhere.
4. ECS can be matched by a rule that over-predicts stopping. It must always be read with the Brier score or AUC.

## Two pipeline facts found while checking (2026-11-13)

Both were found by the code review of `coupling_breakdown.py` and confirmed in `mld/data/EgoMotion.py`.
Neither changes a reported number. Both change what the paper may say.

1. **Stale ego frame for windowed tracks.** A track of 196 frames or more is cut by index, and the ego
   trajectory is cut with it but never re-anchored. The model therefore sees the vehicle in the frame of the
   track's first pose, while the generated motion starts at the origin of the window. On validation this is
   523 of 1,194 conditions (44%), with a median offset of 2.5 m (90th percentile 10.9 m) and a median yaw
   offset of 5 degrees (90th percentile 40). ECS is 0.22 on these conditions and 0.55 on the others, where
   the frame is right. Part of that gap is composition (long tracks hold more standing pedestrians).
   Consequences: the method text must say it, the qualitative export and the teaser must be checked for
   windowed conditions (the exported `ego.npy` is in the stale frame), and re-anchoring the ego at the
   window start is a likely improvement that needs a full retrain.
2. **Too-short tracks are replaced by their neighbour.** `__getitem__` returns the next sample when a track
   has fewer than 40 frames. Listed against usable: train 9,410 / 9,038, validation 1,194 / 1,152, test
   1,182 / 1,146, total 11,786 / 11,336 (5,668 recordings and their mirrored copies). On test, 36 rows are
   therefore duplicates of a neighbouring row. Table I and the sequence counts in the text are too high by
   3.8%. Every model is scored on the same rows, so comparisons stand.

## Braking and body controls (C3, C4), 2026-11-13

`research/src/review_controls.py`, subsets and outputs fixed on validation and committed (72f2929) before the
test dumps were opened. Outputs in `research/data/review_controls_scene_{val,test}.txt`. Brakes minus keeps
going, EgoPed's three seeds, `*` = rule met.

| | Validation | Test |
|---|---|---|
| Stop probability, all conditions | 0.200 to 0.170, -0.030 [-0.076, +0.011] | 0.227 to 0.151, -0.076 [-0.126, -0.036]* |
| Vehicle faster than 2 m/s after 1 s (61%) | -0.042 [-0.126, +0.023] | -0.106 [-0.178, -0.046]* |
| Vehicle at most 2 m/s (inputs barely differ) | -0.012 [-0.026, -0.002] | -0.030 [-0.076, +0.006] |
| Fast vehicle, recorded pedestrian stops | -0.279 [-0.395, -0.153]* | -0.285 [-0.396, -0.150]* |
| Fast vehicle, recorded pedestrian walks | +0.020 [-0.046, +0.073] | -0.042 [-0.105, +0.010] |
| Distance covered by the generated root | -0.00 m [-0.96, +0.86] | +0.99 m [+0.42, +1.54]* |
| Foot-skate ratio, labels / generated (mean) | 0.903 / 0.858* | 0.879 / 0.851* |
| Foot-skate ratio, labels / generated (median) | 0.970 / 0.931 | 0.956 / 0.927 |
| Planted fraction of walking frames | 0.013 / 0.005* | 0.015 / 0.005* |
| Articulation, generated over labels | 0.74 | 0.72 |

- The expectation "little change" was wrong on test. The model responds to a plan that was never driven: a
  braking vehicle makes it generate fewer stops and longer paths. The overall shift meets the rule on test
  only, the shift among recorded stoppers with a fast vehicle on both partitions. Whether the response is
  right cannot be read from recordings, and the paper says so (Section VI-D, limitations, conclusion).
- Generated motion slides like the labels and is less articulated than they are. Written into the body-pose
  limitation. The medians were added to the script after the test read, to quote the audit's statistic
  (Section IV-C uses the median over sequences). The means and the paired differences were fixed before.

## Prior seeds (C1), plain prior, 2026-11-15

Seeds 2345 and 3456 of the ego-blind prior finished 2026-11-15 00:54. Checkpoints by the lowest zero-ego
validation FID (epoch 699 and 2,699, seed 1234 had 3,299), guidance 1, one test pass (jobs 1338001 and
1338002). Report in `research/data/resplit_test_report_priors3.txt`, selection in
`research/data/resplit_selection_priors2.json`.

| Test, plain prior | Seed 1234 | Seed 2345 | Seed 3456 | Mean |
|---|---|---|---|---|
| FID | 4.67 | 4.56 | 4.69 | 4.64 |
| Diversity / MM | 5.79 / 5.23 | 5.60 / 5.10 | 5.88 / 5.20 | 5.76 / 5.18 |
| ADE / FDE | 3.11 / 6.38 | 3.00 / 6.18 | 2.97 / 6.08 | 3.03 / 6.21 |
| minADE / minFDE | 1.31 / 2.64 | 1.33 / 2.70 | 1.28 / 2.59 | 1.31 / 2.64 |
| ECS / corrected Brier | 0.015 / 0.179 | 0.024 / 0.178 | 0.029 / 0.175 | 0.023 / 0.177 |

| Contrast on test | One prior seed (in the paper) | Three prior seeds |
|---|---|---|
| EgoPed vs prior, ADE | +0.68 [0.49, 0.86]* | +0.59 [0.42, 0.77]* |
| EgoPed vs prior, ECS | +0.29 [0.19, 0.39]* | +0.29 [0.18, 0.38]* |
| EgoPed vs prior, minADE | -0.34 [-0.46, -0.22]* | -0.34 [-0.46, -0.23]* |
| Full-seq vs prior, ADE | +0.42 [0.22, 0.63]* | +0.34 [0.14, 0.54], rule not met (one seed pair at -0.006) |
| Full-seq vs prior, ECS | +0.26 [0.16, 0.34]* | +0.25 [0.15, 0.34]* |

- The headline gain of EgoPed over the prior shrinks from 0.68 m to 0.59 m (prior ADE 3.03 instead of
  3.11 m). It still meets the rule with all nine seed pairs. Seed 1234 was the weakest of the three priors.
- Full-seq no longer beats the prior on mean ADE under the rule. On validation the same (+0.34, pair range
  -0.05 to +0.61).
- **Paper changed 2026-11-15 09:00 for the plain prior** (the IA prior follows tonight, Erik gets one list of
  changes for Overleaf then). The two new seeds are merged into `research/data/resplit_selection.json`, so
  that every analysis script reads three prior seeds, and `coupling_breakdown.py`, `ego_controls.py`,
  `review_controls.py`, `resplit_breakdown.py` and `short_horizon.py` were run again. In the paper: abstract
  (3.03 to 2.43 m), contribution 1 and conclusion (0.59 m), Section V (uncorrected Brier of the prior 0.211,
  seeds sentence), Table II prior row with seed ranges, Table III (EgoPed against the prior +0.59* [0.42,
  0.77], Full-seq +0.34 [0.14, 0.54] without star), Section VI-A (FDE gap 1.10 m, validation ECS of the prior
  0.03, per source -0.02 and 0.06), Section VI-D and Table V (3.03, 1.30, 0.177), Section VI-E (ADE gain 0.60
  m on the 99%, AUC 0.53), PedGen paragraph (prior 0.733 m, 0.06 m closer than PedGen).
- **Correction found on the way.** Section VI-E said the corrected Brier score separates EgoPed from the prior
  on validation (0.156 against 0.171). The contrast never met the rule (+0.015 [-0.021, +0.048] with one prior
  seed, +0.012 [-0.023, +0.046] with three). My wording from the Brier change on 2026-11-13. Now "does not
  separate under our rule (0.180 against 0.177 on test, 0.156 against 0.168 on validation)".
- **Fig. 3 and the correlation.** `research/paper/figures/make_fid_vs_conditioning.py` now reads every test
  report and prints the per-model correlation, which was computed by hand before. With three prior seeds it
  is unchanged: FID against mean ADE -0.75 (p = 0.02), against ECS +0.33 (p = 0.38), nine models. The figure
  shows 17 runs (per run -0.73 and +0.49), caption updated.

### Prior with IA, three seeds, 2026-11-15 20:00

Seeds 2345 and 3456 finished 18:55. Checkpoints by zero-ego validation FID (jobs 1345297 and 1345298): epoch
1,199 (2.301) and 2,899 (2.106), seed 1234 had 1,099. Validation at guidance 1 (jobs 1345752, 1345753): FID
2.485 and 2.215. Selection committed (4477bf4) before the single test pass (jobs 1345893, 1345894).

| Test, prior with IA | Seed 1234 | Seed 2345 | Seed 3456 | Mean |
|---|---|---|---|---|
| FID | 3.29 | 2.85 | 2.51 | 2.88 |
| Diversity / MM | 5.48 / 4.76 | 5.62 / 4.95 | 5.49 / 4.73 | 5.53 / 4.81 |
| ADE / minADE | 3.00 / 1.52 | 3.02 / 1.54 | 2.99 / 1.51 | 3.00 / 1.52 |
| ECS / corrected Brier | 0.008 / 0.205 | 0.007 / 0.205 | 0.009 / 0.206 | 0.008 / 0.205 |

- **FID.** The ego-blind prior with IA now has the best FID of all models (2.88), ahead of Full-seq + IA
  (3.24, seeds 2.73 to 4.14). The paper said "level with it (3.29)". Section VI-B now states the order.
  Erik, 2026-11-15 evening: change the abstract and the conclusion as recommended. Done, together with the
  same statement in the second contribution: "It ranks an ego-blind prior first" (abstract, 1,179 of 1,200
  characters), "An ego-blind prior has the best FID" (contribution), "ranks an ego-blind prior first"
  (conclusion). Erik also confirmed that rs2 continues as planned, so the six interaction-weighted rs2 runs
  finish and get their validation sweep.
- **Contrasts on test, unchanged in substance.** EgoPed + IA against its prior ADE +0.07 [-0.23, 0.35],
  minADE -0.52*, ECS +0.21*. Full-seq + IA ADE -0.00 [-0.29, 0.26], minADE -0.55*, ECS +0.09*.
- **Validation differs on mean ADE.** There both IA models beat their prior under the rule (+0.26* and
  +0.27*, and already with one prior seed, +0.25* and +0.26*). The paper said without qualification that the
  IA models are no better than their own prior on mean ADE. It now says "on test ..., although they are on
  validation".
- **Correlation per model** (nine models, seeds averaged): FID against mean ADE -0.73 (p = 0.02), against ECS
  +0.38 (p = 0.31). Before -0.75 and +0.33. Fig. 3 has 19 runs (per run -0.70 and +0.61).
- **In the paper:** Table II row, last two rows of Table III, Section VI-B, Fig. 3 and caption, seeds
  sentence. Eight pages, no overfull box, 39 references. Change list for Overleaf in
  `research/paper_ral/CHANGES_2026-11-15.md`.

## Regressor baseline (C2), 2026-11-13

Section 3 of `review_controls.py` ran on validation unchanged (committed at cbf1f0f), then on test once. Three
seeds, checkpoint with the lowest validation MSE (epoch 10 to 20 of 300, the regressor overfits afterwards).

| | Validation | Test |
|---|---|---|
| Mean ADE, regressor / EgoPed / prior | 1.79 / 2.27 / 3.07 | 1.97 / 2.43 / 3.11 |
| minADE (regressor has one sample) | 1.79 / 1.43 / 1.22 | 1.97 / 1.65 / 1.31 |
| ECS | 0.05 / 0.36 / 0.02 | 0.08 / 0.31 / 0.02 |
| AUC of the stop prediction | 0.53 / 0.75 / 0.54 | 0.54 / 0.70 / 0.52 |
| Corrected Brier | 0.223 / 0.156 / 0.171 | 0.224 / 0.180 / 0.179 |
| Share of conditions with a generated stop | 2.5% / 19.7% / 22.7% | 2.3% / 20.5% / 22.9% |
| FID of the regressor | 32.3 | 36.0 |
| Regressor minus EgoPed, ADE | +0.48 m better* | +0.47 m better* |
| Regressor minus EgoPed, ECS | -0.31* | -0.23* |
| Regressor minus EgoPed, corrected Brier | -0.066 worse* | -0.043 worse, rule not met |

The regressor wins mean ADE, as expected from the old split, by predicting the average: a slow walk that
almost never stops. It has no coupling, a worse best-of-five error and an FID of 36. In the paper as the last
row of Table II and a paragraph in Section VI-A, which now says that mean ADE alone rewards a model that
never decides. To hold 8 pages, five passages were tightened (R-Precision, the bootstrap remark, the
last-checkpoint sentence, PedGen, one limitation).

## Plan

### A. From existing results, CPU only

- A1. Apply `coupling_breakdown.py` once to the test partition after the script is committed. Report the
  stop kinds, the never-in-path subset and the geometry rule in the paper.
- A2. Report the FID correlation per model, not per run. Today, over the nine models of Tables II and IV,
  Spearman FID against mean ADE is -0.75 (p = 0.02), and FID against ECS is +0.33 (p = 0.38), so the ECS
  half of the claim goes. Recompute when the new prior seeds are in.
- A3. Optional. A small classifier from the ego trajectory to the stop label, trained on the training
  partition, as the ceiling of what the trajectory reveals.

### B. Text, no new results

- B1. Narrow the claim. The paper shows that generated motion depends on the vehicle input in the way the
  recordings do. It does not show a reaction to the vehicle. New limitation: response to plans that were not
  recorded with the pedestrian.
- B2. Seeds. State which models have three seeds and that contrasts against a single-seed model have three
  seed pairs. Simplify once C1 is in.
- B3. Data. 5,893 recordings with mirrored copies, consent sentence, root source, and no "reliable" for the root.
- B4. Numbers. MM column in Table II, gaps quoted before rounding, Brier gap out of the encoder paragraph.

### C. GPU on Helma

- C1. `rs1_uncond_s2345`, `rs1_uncond_s3456`, `rs1_uncond_ia_s2345`, `rs1_uncond_ia_s3456`. Submitted
  2026-11-13 as jobs 1319359 to 1319370, three chained 24 h segments each, same code and configuration as
  seed 1234 (Helma checkout 00da6a8, training code unchanged since). Then `select_uncond.sh` for the
  zero-ego validation pass, selection as before, one test pass for the four new runs, and Tables II and III
  recomputed with nine seed pairs.
- C2. Retrain `research/src/regressor_baseline.py` on the scene split and score it with the per-condition
  metrics. On the old split it had mean ADE 1.80 m against 2.32 m for the diffusion model and FID 36, so it
  will probably win mean ADE and lose realism and coverage. The paper has to say that plainly.
- C3. Yield against no-yield, fixed before any run. Both arms show the model the first second of the
  recorded ego trajectory and then continue in the same direction. "Keeps going" holds the speed (the
  existing `hor1` control), "brakes" decelerates at 2 m/s^2 until the vehicle stands (`--ego-brake 2`,
  tested in `test_ego_editor.py`). Only the speed profile differs. Reported: the shift in the stop
  probability, overall and for vehicles that move at more than 2 m/s after the first second, with the scene
  bootstrap. Expectation: little change, because the data holds few interactions. Either outcome is reported.
  Job: `slurm/resplit/eval_review.sh`.
- C4. Foot-skate ratio and articulation of generated motion against the labels of the same windows
  (`--save-skate`, same job), to put a number on "the feet slide as in the labels".

### D. Open

- D1. Mohan's qualitative figure and paragraph (Erik copies them in).
- D2. Optional root check against the annotated box tracks, in the extraction repository.
- D3. Neutral dataset name.

## Retraining with the ego in the window frame (`rs2`, decision 7)

- **Change.** `DATASET.EGOMOTION.EGO_WINDOW_FRAME=True` re-anchors the ego of a window cut from a longer
  track at the pedestrian's root pose at the window start (`EgoMotionDataset._ego_in_window_frame`). The
  default (off) is bit-identical to the old loader. Checked on validation by `research/src/test_window_frame.py`:
  motion and windows unchanged, the new ego equals the independent `load_geometry` to 0.01 mm, and the
  vehicle-pedestrian distance is preserved for random start frames.
- **Kept from rs1.** The split, the motion VAE (`rs1_vae_latent4`), the normalisation statistics of
  `scene_split_v1`, every recipe, seed, epoch count and selection rule. The ego-blind priors never see the
  ego, so the rs1 priors (three seeds each once C1 is in) stay valid.
- **Retrained.** Both ego encoders (`rs2_enc_pooled`, `rs2_enc_fullseq`) and all 19 conditional runs:
  EgoPed, Full-seq, EgoPed+IA, Full-seq+IA and the two encoder ablations with three seeds each, and the
  single trainable Full-seq run. `slurm/resplit/submit_rs2.sh`; the flag is keyed on the run name in
  `_env.sh`, so a resumed segment or a later evaluation cannot fall back to the old frame.
- **Not yet done for rs2.** The regressor baseline (`regressor.sh <seed> rs2`), the ego controls, and an
  `rs2` switch in `coupling_breakdown.py` (its section 0 assumes the old frame). Selection needs
  `select_resplit.py --pattern 'rs2_*'` and its own selection file, which also lists the rs1 priors.
  The analysis scripts now take `--gen rs2`, see the decision rule below.
- **Reviewed before submission.** An independent review of the change traced the flag from the job scripts
  into training, resumed segments, training-time validation, encoder pretraining and every evaluation
  script, and found no path that runs with the flag off. It found four later problems, all fixed in 3c3596c
  (config choice in `submit_evals.sh`, `cell_of`, the K=1 case of the corrected Brier score, no frame
  switch in `teaser_generate.py`). A CPU check of 400 training items with the interaction recipe gave the
  same windows with and without the flag, a changed ego in 112, all finite.
- **Submitted 2026-11-13, about 15:25.** Jobs 1320604 and 1320605 (encoders), 1320606 to 1320662 (19 runs,
  three chained 24 h segments each), all with `--nice=100`. The first segment of every run waits for both
  encoders with `afterok`. Log on Helma in `outputs/submit_rs2_2026-11-13.log`. The 32 jobs queued before
  were not touched. With nice 100 the rs2 jobs rank behind `r1pre` (nice 42) and `wymres` (nice 0) but
  ahead of the `mtr_rev` array (nice 4000), whose estimated start moved from 20:30 on 2026-11-13 to 09:52
  on 2026-11-14 around the submission. Erik decides whether rs2 should go behind it.
- **Code version of the running rs1 prior runs.** Their later segments start from checkout b443a6c. With
  the flag off the loader is identical to the one of seed 1234 (checked on validation and on training items).
- **Protocol.** The test partition has been read once for rs1. The rs2 models are selected by the same
  rules on validation and then scored on test once. The paper must say that the frame was corrected after
  the first test pass and that the selection rules were not changed. Whether rs2 replaces rs1 in the tables
  is decided when the validation results are in. If rs2 is not clearly better on validation, rs1 stays.

### Decision rule and steps for rs2, fixed 2026-11-13 before any rs2 run has a validation result

- **Rule.** Each family at its own selected checkpoints and guidance, on validation. rs2 replaces rs1 in the
  tables if its method cell (`pooled`) beats that of rs1 under the usual rule (interval excludes 0, all nine
  seed pairs agree) on mean ADE or on ECS, and loses under the rule on none of ADE, ECS and corrected Brier.
  Otherwise rs1 stays, and the paper gets one sentence with the rs2 validation numbers as a robustness check.
  FID plays no part. The seed-to-seed spread of rs1 `pooled` on validation is about 0.11 m in ADE and 0.03
  in ECS, so a real gain has to exceed that in every pair.
- **Secondary, reported either way.** The ECS on windows that start later against the others (rs1
  validation 0.22 against 0.55, `coupling_breakdown.py --gen rs2`). The fix should narrow that gap.
- **Steps, on Helma from the repository root.**
  1. `python research/src/select_resplit.py --pattern 'rs2_*' --out research/data/resplit_epochs_rs2.json`
  2. `bash slurm/resplit/submit_evals.sh sweep research/data/resplit_epochs_rs2.json`
  3. `python research/src/select_resplit.py --pattern 'rs2_*' --guidance research/data/resplit_epochs_rs2.json --out research/data/resplit_selection_rs2.json`
  4. `python research/src/compare_resplit.py --split scene_val --gen rs2 --against rs1 --selection research/data/resplit_selection.json research/data/resplit_ablation_selection.json research/data/resplit_selection_rs2.json`
  5. Only if the rule says rs2: `submit_evals.sh test` with the rs2 selection, `eval_controls.sh` and
     `eval_review.sh` for the three `rs2_pooled` seeds, `regressor.sh <seed> rs2`, then every table, with
     `--gen rs2` in `coupling_breakdown.py`, `ego_controls.py` and `review_controls.py`.
- The analysis scripts take `--gen` (default rs1). The ego-blind priors belong to both generations.
  Checked with stand-in dumps: the same three rs1 dumps under rs2 names give a difference of zero and no star.
- **Progress, 2026-11-15 06:20.** The rule concerns the `pooled` cell, so that cell goes first and the other
  cells follow in files of their own (selection files are merged by the analysis scripts, the rule is
  unchanged). All three `rs2_pooled` seeds reached epoch 4,999 by 06:02. Checkpoints by the lowest
  training-time validation FID, `research/data/resplit_epochs_rs2_pooled.json`: epoch 2,799 (seed 1234, FID
  4.29), 1,499 (seed 2345, 4.06), 1,099 (seed 3456, 3.82). Guidance sweep on validation queued, jobs 1339008
  to 1339016, nice 100 (`NICE=100 submit_evals.sh sweep`, the option is new in the script). Next:
  `select_resplit.py --pattern 'rs2_pooled_s[0-9]*' --guidance research/data/resplit_epochs_rs2_pooled.json
  --out research/data/resplit_selection_rs2_pooled.json`, then step 4 with that file. The pattern needs
  `s[0-9]*`, since `rs2_pooled_s*` also matches the `pooled_scratch` runs.

### rs2 decision, 2026-11-15 07:00: rs1 stays

Validation, 1,194 conditions, pooled cell at its own checkpoints and guidance 15 (the sweep picked 15, as for
rs1). Report in `research/data/resplit_val_report_rs2_pooled.txt`. Delta above 0 means rs2 is better.

| Validation, pooled | rs1 | rs2 | rs2 minus rs1 | Rule |
|---|---|---|---|---|
| Mean ADE (m) | 2.27 | 2.41 | -0.142 [-0.225, -0.051], seed pairs -0.301 to +0.005 | not met |
| minADE (m) | 1.43 | 1.51 | -0.076 [-0.142, -0.005], seed pairs disagree | not met |
| ECS | 0.36 | 0.33 | -0.035 [-0.089, +0.016] | not met |
| Corrected Brier | 0.156 | 0.171 | -0.015 [-0.029, -0.002]* | met, rs1 better |

- **Rule applied.** rs2 beats rs1 on neither mean ADE nor ECS and loses on the corrected Brier score. rs1
  stays in every table. No test pass, no controls and no regressor for rs2. The paper gets one sentence.
- **Secondary check, and my expectation was wrong.** The fix does not narrow the gap in coupling between
  windows that start later and the other conditions (`research/src/rs2_frame_check.py`): ECS 0.18 against 0.53
  for rs2, 0.22 against 0.55 for rs1, difference of the gaps +0.03 [-0.07, +0.12]. So the stale frame was not
  the cause of that gap. Windows that start later are the tracks of full length (9.8 s) with twice the stop
  rate (0.31 against 0.16) and twice the error (ADE 3.2 against 1.5 m), which is the plainer reading.
  Post hoc, on a remark of the code review: the offset between the two frames is the distance the pedestrian
  covered before the window, so it is near zero for everyone who stood until then. On the windows the fix
  really moves, rs2 gains nothing either: offset of at least 1 m (366 conditions) ADE 3.29 against 3.18,
  -0.11 [-0.31, +0.13]; at least 5 m (181 conditions) ADE 3.27 against 3.04, -0.23 [-0.51, +0.06]; ECS
  differences -0.01 and -0.02 with wide intervals.
- **Where rs2 is behind.** On the 671 conditions whose ego input is the same for both generations
  (ADE -0.151 [-0.217, -0.082]*), and not reliably on the 523 whose input changed (-0.129 [-0.295, +0.063]).
  So the deficit is not an effect of the frame of the evaluated input. Cause not established. Candidates are
  the changed inputs of the windowed training tracks and ordinary variation of seeds and checkpoints.
- **Against the prior, rs2 keeps the result.** rs2 pooled against the three prior seeds on validation: ADE
  +0.568 [0.351, 0.764]*, ECS +0.298 [0.163, 0.435]*.
- **Sentence for the paper** (Section IV-A, in place of or next to the sentence on the stale frame, to be
  written in the consolidated update): retraining with the vehicle in the frame of the window start does not
  improve the model on validation (mean ADE 2.41 against 2.27 m, ECS 0.33 against 0.36), so the reported
  models keep the original input.
- **Other rs2 cells.** For a robustness statement on the comparisons (Full-seq against EgoPed, the encoder
  ablations), the validation sweep of the nine other finished plain runs is queued, jobs 1339223 to 1339249,
  nice 100, `research/data/resplit_epochs_rs2_plain.json`. `rs2_pooled_scratch_s3456` follows when it reaches
  epoch 4,999, the six interaction-weighted runs at about 22:30. Validation only.
- **Other plain rs2 cells, 2026-11-15 08:00, validation.** `research/data/resplit_val_report_rs2_plain.txt`.
  No rs2 cell beats its rs1 counterpart on mean ADE or ECS. rs2 minus rs1: Full-seq ADE -0.04, ECS
  -0.07*; pooled_unfrozen ADE -0.01, ECS -0.00; pooled_scratch ADE -0.08*, minADE +0.11*, ECS -0.01; the single
  trainable Full-seq run ADE -0.58*. Within rs2 the paper's comparisons keep their direction: EgoPed against
  the prior ADE +0.57*, ECS +0.30*; Full-seq against EgoPed ECS -0.10* and ADE -0.27 [-0.42, -0.13] with one
  of nine seed pairs reversed by 0.04 m, so the rule is not met on ADE there (it is for rs1, -0.37*);
  freezing the encoder ECS +0.05* (rs1 +0.08*); pretraining no difference on ADE or ECS in either family.
- **Interaction-weighted rs2 cells, 2026-11-16 01:25, validation. rs2 is closed.**
  `research/data/resplit_val_report_rs2_ia.txt`. All 19 rs2 runs finished (the last at 01:09). Guidance by
  validation FID: 5 for `egoped_ia`, 15 for `pooled_ia` (rs1 had 10 and 5). rs2 minus rs1: `egoped_ia` ADE
  -0.08, ECS -0.04; `pooled_ia` ADE -0.56 [-0.70, -0.42], ECS -0.04, none under the rule. The gap of
  `pooled_ia` comes with the guidance FID selects (mean ADE 3.23 at 15 against 2.55 at 5).
  **Over all seven conditional cells rs2 beats rs1 nowhere on mean ADE or ECS, all fourteen differences are
  negative, three under the rule.** Within rs2 the IA recipe worsens mean ADE, minADE and ECS for both
  encoders (six of six in direction, four under the rule), as in rs1.
  The sentence in Section IV-A now covers all conditional models: "Retraining all conditional models with the
  vehicle in the window's own frame improved none on mean ADE or ECS on validation (EgoPed 2.41 against 2.27 m
  and 0.33 against 0.36)." Eight pages. Change list regenerated, entries 5 and 6 differ from the list of 20:07.
  No rs2 job is left on Helma. The rs2 checkpoints stay in `models_resplit/mld/rs2_*`.
- **Was open for Erik, settled 2026-11-15 evening (continue as planned).** The six interaction-weighted rs2 runs still hold six GPUs until about 22:30. With rs1
  staying, they only serve the robustness statement. They keep running unless Erik wants them stopped.

## The regressor in the abstract, 2026-11-16

The abstract led with EgoPed's gain over the prior on mean ADE, while the deterministic regressor has the
lowest mean ADE of all models (1.97 m on test, EgoPed 2.43 m). Erik, 2026-11-16: add the qualifier. The lead
sentence now ends "against 1.97 m and 0.08 for a deterministic regressor", so both of the regressor's numbers
stand next to EgoPed's 2.43 m and 0.31. Five words pay for it ("existing", "directly", "trajectory" before
"encoder", and "the two design choices it favours both raise" is now "both design choices it favours raise").
The abstract has 1,197 of 1,200 characters, the paper still 8 pages. The contributions and the conclusion
are unchanged, Section VI-A carries the regressor in the body.

## Style pass, 2026-11-16

Erik, 2026-11-16: the first contribution and the conclusion get the regressor as well, the paper is to use
American spelling, and it must hold no sentence fragments. Done in `main.tex`, listed for Overleaf in
`research/paper_ral/CHANGES_2026-11-16.md` (15 entries and a table of 21 words, to apply after the list of
2026-11-15).

- The contributions and the opening phrases of eight captions were the fragments. Run-in headings stay.
- The paper has the same length as before (8 pages, every page ends where it did). Page 2 is packed tightly.
  One more line before the last equation of Section III pushes that equation to page 3, and the breaks after
  it cost about ten lines, which is 9 pages. Measure any change to pages 1 and 2 with a build.
- Not done, since it was not asked for. IEEE style puts a comma before "and" in a list of three, the paper
  does not.

## Munich external test, 2026-11-16

Erik, 2026-11-16: the group's Munich drives are being labelled (`NAS2/car_data/dataset_annotations_ayadata`,
nuScenes format, 124 of 136 scenes labelled, tables `v2.2-mini`). He asked what they can give this paper and
decided for a second, external test of the frozen models on root paths, with the interactive encounters as a
subset. No pose fits, no retraining, nothing selected on this set.

What the check of the labels found, all before any model was run on them:

- **The vehicle heading in `ego_pose.json` is 90 degrees off the direction of travel in every scene**, and all
  global boxes inherit it (stored = t + Rz(-90 deg)(p - t)). Labelled parked cars move a median 49 m as stored
  and 0.8 m after the correction. Boxes project correctly into the cameras with the tables as they are. The
  cause is the group's converter, not the labels. The staged scenes in the paper do not show it.
- **The dataset's vehicle input and root path come from the labelled boxes, not from the pose fit.** Step 1 of the
  pose pipeline gives SMPL the box centre as translation and the box rotation as orientation, and expresses the
  vehicle in the frame of the pedestrian's box at its first frame. Rebuilding 40 nuScenes and 40 Waymo files
  from the boxes reproduces the vehicle path to a median 5 and 3 cm and the root path to 18 and 14 cm
  (`munich_eval.py --check-dataset 40`). The paper said "The root position comes from this fit" and "The root
  itself has no independent reference". Erik, 2026-11-16: both corrected in `main.tex`, entries 16 and 17 of
  `research/paper_ral/CHANGES_2026-11-16.md`. The build stays at 8 pages with about three lines free.
- Box centres jitter by about 0.1 m per keyframe. A pedestrian labelled standing reads a median 0.27 m/s, above
  the 0.2 m/s of the paper's stop rule.
- The labelled pedestrian heading lies within 30 degrees of the walking direction for 83% of walkers, against
  100% (nuScenes) and 89% (Waymo) in the files checked.

**Protocol, fixed with this commit** (`research/src/munich_eval.py`, `research/data/munich_v1/manifest.json`):

1. All labelled pedestrians not sitting or lying on half or more of their keyframes, heading corrected, box
   centres and vehicle poses resampled linearly to 20 fps.
2. One window of at most 196 frames per track, centred on the closest approach to the vehicle. Tracks under 40
   frames are dropped. 1,327 sequences in 114 scenes.
3. Frame of the pedestrian's box at the window start. The labelled heading is the primary variant, the heading
   taken from the first 2 s of motion (when they cover 1.5 m) is a sensitivity variant, run alongside.
4. Recorded stop is the label `pedestrian.standing` on three consecutive keyframes of the window (337 stops,
   25.4%). The speed rule on the box path is secondary (336 stops, agreement 85%).
5. Interactive subset from the window (60 sequences in 34 scenes). In the vehicle's path with the vehicle
   arriving within 5 s, while it moves (4) or while it waits (11), behind it within 5 s (16), or otherwise
   within 2.5 m of its path (29). The vehicle must drive at least 3 m in the window. Only 11 of the 60 are
   recorded stops, so ECS on the subset is weak by construction and ADE is the number to read there.
6. Models are those of the main table at their selected checkpoints and guidance scales
   (`research/data/resplit_selection.json`), three seeds each, plus the three regressors. K = 5.
7. Primary outcome. EgoPed against the prior on mean ADE and ECS, on all sequences and on the interactive
   subset, with the scene bootstrap and the seed-pair rule of the paper (`research/src/munich_report.py`).
   Everything else is secondary.
8. One pass. A rerun is allowed for a technical fault only, and gets recorded here. The result is reported
   whichever way it comes out.

The whole-track parse reported to Erik earlier the same day found 79 interacting tracks in 41 scenes. The
subset above has 60 because it is taken from the evaluated window and requires the vehicle to drive.

Code review before the freeze (2026-11-16): the rule for a waiting vehicle had no limit on the time gap,
against the text above. Fixed, which moved nine sequences from "waiting" to "within 2.5 m of its path" and left
the subset and the paths unchanged. The frame of a window that starts late is that of the window start, which the
rs1 models saw only for tracks of up to 196 frames, so this set does not copy the stale frame of the paper.

Checks without a model: unit tests (`research/src/test_munich_eval.py`), the real loader returns recorded roots
and vehicle inputs exactly as stored for both variants, and the vehicle input lies inside the training range
(mean 0.8 and 2.8 m against 0.0 and 5.5 m, standard deviation 12.1 and 16.3 m against 13.0 and 16.2 m).

The sequence files (`sequences_label.npz`, `sequences_motion.npz`) hold paths derived from the group's labels and
stay out of git. Their hashes are in the manifest and are checked whenever the set is loaded.

Submitted on Helma 2026-11-16 12:00, after the freeze (commit 26f7e4e): jobs 1353679 to 1353699 (`rs1_munich`),
18 models and 3 regressors, both heading variants each, default priority.

## Next steps, in order

1. Done. Regressor (C2), braking (C3) and body statistics (C4) are scored on test and in the paper.
2. When C1 finishes (about 2.5 days): `select_uncond.sh` for the four runs, selection with
   `select_resplit.py` as for seed 1234, one test pass (`eval.sh ... test zero`), then Tables II and III,
   the per-model correlation and the seeds sentence.
3. Done. Panel (b) of the teaser is rendered again with the vehicle in the frame of the joints.
4. When the rs2 runs finish (encoders, then about 2.5 days of training once they start): selection on
   validation as for rs1, compare rs2 with rs1 on validation, then decide which family the paper reports.

## Protocol

New subsets, baselines and edits are defined on validation, committed, and then applied to test once. The
test partition is never used to choose a threshold or a model.

## Status

| Item | State |
|---|---|
| C1 prior seeds | `rs1_uncond_s2345` and `rs1_uncond_s3456` finished 2026-11-15 00:54 (epoch 4,999, 50 checkpoints each). Selected, scored on validation and on test once, see the section on the prior seeds. `rs1_uncond_ia_s2345` and `rs1_uncond_ia_s3456` are at about epoch 3,300 to 3,400 in their second segment and finish at about 19:00 on 2026-11-15 |
| A1 breakdown | done on validation and test, written into the paper as Section VI-E |
| A2 correlation per model | done in the text (-0.75, p = 0.02, nine models). Recompute after C1 |
| B1 to B4 text | done. Paper at 8 pages, 39 references, abstract 1,199 characters |
| Qualitative export | regenerated with the vehicle in the joints' frame: 16 of 28 `ego.npy` files were wrong by up to 12.6 m and 111 degrees. Corrected copy on the NAS in `qualitatives/egoped_scene_test_2026-11-13_frame_fixed/`, warning note added to the old folder. Mohan has to place the vehicle again in any scene built from the 16 folders |
| Teaser, Fig. 1 | done. Panel (b) (validation row 523) rendered again with the vehicle in the frame of the joints, camera moved to keep all five vehicle positions in view |
| Brier column | done. Tables II and V carry the noise-corrected score, Section V says it was adopted after the test pass |
| rs2 retraining | running on Helma, jobs 1320604 to 1320662, nice 100. All 19 runs handed over to their second segment on 2026-11-14 between 17:05 and 19:10, each restored from its last checkpoint with the flag on. Plain runs finish in the morning of 2026-11-15, interaction-weighted ones at about 22:30. Both encoders completed 2026-11-13 17:03 (best validation MSE 1.908 full-sequence and 1.902 pooled, against 1.915 and 1.904 for rs1, so the frame barely changes pretraining). At 17:14 nine of the 19 runs were training and ten waited for a GPU. Checked in the dumped run configs of the first five: `EGO_WINDOW_FRAME: true`, the rs2 encoder of the right kind, no errors |
| C2 regressor | done on validation and test, in the paper (Table II, Section VI-A). The training jobs 1319608 to 1319610 hit the two-hour limit after the kept checkpoint (epoch 10 to 20), scoring jobs 1320734 to 1320736 completed |
| C3, C4 | done on validation and test, in the paper (Section VI-D, limitations, conclusion). See the section above |
| Everything else | open |
