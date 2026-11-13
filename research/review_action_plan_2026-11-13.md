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
  `cell_of` now drops the generation prefix and `compare_resplit.py` refuses a cell that mixes rs1 and rs2,
  but `coupling_breakdown.py`, `ego_controls.py` and `review_controls.py` take one selection file and must
  be given the rs2 one. `review_controls.py` looks for `rs1_regressor` dumps only.
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

## Next steps, in order

1. Regressor (C2): when jobs 1320734 to 1320736 have scored the three checkpoints, run section 3 of
   `research/src/review_controls.py` on validation, fix it there if needed, commit, then read its test
   dumps once and write the comparison into the paper. Sections 1 and 2 (braking, body) are done.
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
| C1 prior seeds | running on Helma since 2026-11-13 13:35 (jobs 1319359 to 1319370, three chained segments each) |
| A1 breakdown | done on validation and test, written into the paper as Section VI-E |
| A2 correlation per model | done in the text (-0.75, p = 0.02, nine models). Recompute after C1 |
| B1 to B4 text | done. Paper at 8 pages, 39 references, abstract 1,199 characters |
| Qualitative export | regenerated with the vehicle in the joints' frame: 16 of 28 `ego.npy` files were wrong by up to 12.6 m and 111 degrees. Corrected copy on the NAS in `qualitatives/egoped_scene_test_2026-11-13_frame_fixed/`, warning note added to the old folder. Mohan has to place the vehicle again in any scene built from the 16 folders |
| Teaser, Fig. 1 | done. Panel (b) (validation row 523) rendered again with the vehicle in the frame of the joints, camera moved to keep all five vehicle positions in view |
| Brier column | done. Tables II and V carry the noise-corrected score, Section V says it was adopted after the test pass |
| rs2 retraining | queued on Helma, jobs 1320604 to 1320662, nice 100 |
| C2 regressor | training on Helma, seeds 1234, 2345, 3456 (jobs 1319608 to 1319610). 300 epochs need a little over the two-hour limit, so the jobs end in a timeout before scoring. The kept checkpoint is from epoch 10 to 20 (validation MSE 1.30, rising to 1.45 by epoch 130), so nothing is lost. Scoring jobs 1320734 to 1320736 run after them (`regressor.sh <seed> rs1 eval`) |
| C3, C4 | done on validation and test, in the paper (Section VI-D, limitations, conclusion). See the section above |
| Everything else | open |
