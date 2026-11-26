# RA-L submission (IEEE Robotics and Automation Letters)

Converted from `research/paper/` (OJ-ITS, ieeeoj.cls) on 2026-11-09. Self-contained: upload this folder to
Overleaf as a new project. Build: `latexmk -pdf main.tex`.

## Rules applied (RA-L Information for Authors, checked 2026-11-09)

- 6 pages in two-column IEEE format **including figures and references**, up to 2 more at $175 each.
  Current build: **8 pages**, with only a few free lines at the end of the references.
- Initial submission uses the RAS `ieeeconf.cls` (from ras.papercept.net/conferences/support/files/ieeeconf.zip),
  US letter. The final version moves to `\documentclass[letterpaper, 10 pt, journal, twoside]{IEEEtran}`.
- **Double-anonymous review**: no names, affiliations, acknowledgment or funding, no lab logos, and no individual
  vehicle names. `\anonymoustrue` in main.tex hides authors and acknowledgment. The vehicle photo figure is gone
  (it shows the vehicle name, the university name, a QR code and the licence plate), "AVA" is now "staged" or "our
  research vehicle", and Munich is not named.
- No biographies.
- Abstract field in PaperCept: at most 1,200 characters, no TeX. The abstract below matches the paper (1,193, rewritten 2026-11-26).
- Keywords: 2-5, the first two from the RA-L list. Used: Human and Humanoid Motion Analysis and Synthesis;
  Intelligent Transportation Systems; Deep Learning Methods; Modeling and Simulating Humans; Performance
  Evaluation and Benchmarking.
- PDF at most 5 MB (current 4.4 MB with both of Mohan's figures), fonts embedded, no author in the PDF metadata.

## What changed against the OJ-ITS text

- Related work condensed; 39 references instead of 67 (40 since 2026-11-18); author lists of four or more names print as "et al."
  (`ral_ctl.bib`).
- The pose-audit, guidance and PedGen tables are folded into the text with their numbers.
- Shorter captions, discussion and limitations; the broader-impact paragraph is dropped.
- Fixed a broken cross-reference in the teaser caption (it printed "Section c").
- 2026-11-09 review pass (against Flohr's 25 Overleaf threads): no colons or semicolons in the prose, robotics
  in the introduction, three comparative claims reworded to what the pre-registered rule supports, ADE/FDE
  spelled out, orientation null result stated, two rounding errors fixed, bibliography venues and acronyms
  fixed. The guidance-sweep FIDs are now traceable in `research/data/resplit_val_guidance_fid.txt`.

- 2026-11-13, after the pre-submission review (`research/review_action_plan_2026-11-13.md`):
  - New Section VI-E "What the Coupling Rests On", from `research/src/coupling_breakdown.py` and
    `research/data/coupling_breakdown_scene_test.txt`. The claim is narrowed to an observational coupling,
    in the abstract, the limitations and the conclusion as well.
  - Sequence counts corrected to the tracks the loader really uses (11,336, of which 5,668 recorded), and
    the stale ego frame of windowed tracks is stated in Section IV-A.
  - FID correlation reported per model, MM column in Table II, seeds stated per model, consent sentence,
    root source, gaps quoted before rounding.
  - Brier column of Tables II and V replaced by the score without the sampling noise of the five samples
    (decided 2026-11-13, after the test pass, and said so in Section V).
  - To hold 8 pages: discussion and limitations merged, the robotics outlook folded into a limitation,
    shorter guidance, PedGen and encoder paragraphs, one row less in Table V, five references fewer.
  - Braking edit of the vehicle's plan in Section VI-D and body statistics of generated motion in the
    limitations (`research/src/review_controls.py`, `research/data/review_controls_scene_test.txt`).
  - Teaser panel (b) rendered again with the vehicle path in the frame of the joints.
  - Deterministic regressor as the ego-aware baseline, last row of Table II and a paragraph in Section VI-A.
    Paid for with shorter R-Precision, bootstrap, last-checkpoint, PedGen and metrics-limitation sentences.

- 2026-11-15, three seeds of the plain prior and the rs2 decision (`research/review_action_plan_2026-11-13.md`):
  - The prior is now the mean of three seeds (test ADE 3.03 m, seeds 3.11, 3.00, 2.97). EgoPed's gain is
    0.59 m instead of 0.68 m and still meets the rule. Changed in the abstract, contribution 1, Section V
    (uncorrected Brier of the prior, seeds sentence), Tables II, III and V, Sections VI-A, VI-D, VI-E, the
    PedGen paragraph and the conclusion. Full-seq against the prior on mean ADE loses its star in Table III.
  - Section VI-E said the corrected Brier score separates EgoPed from the prior on validation. It does not
    under the rule (not with one prior seed either). The sentence now says so for both partitions.
  - Retraining with the ego in the frame of the window start (`rs2`) does not beat the current models on
    validation, so they stay. One sentence in Section IV-A, paid for by dropping the clause on the frame from
    the limitations.
  - To hold 8 pages, seventeen sentences lost a few words each without a change of content.
  - The prior with IA has three seeds as well (test FID 2.88, seeds 3.29, 2.85, 2.51, ADE 3.00 m). It now
    has the best FID of all models, ahead of Full-seq + IA (3.24), and Section VI-B says so. Table II row,
    last two rows of Table III, the per-model correlation (-0.73, p = 0.02 and +0.38, p = 0.31), Fig. 3 with
    19 runs, the seeds sentence. The IA models are no better than their prior on mean ADE on test but are on
    validation, and the sentence now names both.
  - Erik, 2026-11-15: the abstract, the second contribution and the conclusion now say that FID ranks an
    ego-blind prior first, in place of "matches" and "level with" the best conditional model.
  - 2026-11-16: all retrained models are evaluated on validation. None of the seven conditional models
    improves on mean ADE or ECS, and the sentence in Section IV-A now says so for all of them.
  - Erik, 2026-11-16: the abstract names the deterministic regressor. It has the lowest mean ADE of all
    models, and the abstract led with EgoPed's gain over the prior without saying so. The lead sentence now
    ends "against 1.97 m and 0.08 for a deterministic regressor". Paid for with five words ("existing",
    "directly", "trajectory" before "encoder", and "the two ... both" is now "both"). 1,197 of 1,200
    characters, so the abstract is full as well.
  - Every change of these two days against the version of 2026-11-13 is listed in `CHANGES_2026-11-15.md`.

- 2026-11-16, style pass asked for by Erik (`CHANGES_2026-11-16.md`, to apply after the list above):
  - The first contribution and the conclusion name the deterministic regressor as the abstract does, and
    give the error as 3.03 to 2.43 m in place of a gain of 0.59 m.
  - American spelling throughout (23 words, among them "sidewalks" for "pavements" and "curb" for "kerb").
    Titles in the reference list stay as published.
  - No sentence fragments. The three contributions and the opening phrase of eight captions are full
    sentences now. Bold or italic run-in headings stay.
  - Same length as before. One word less in the second paragraph of the introduction pays for the line the
    regressor adds to the first contribution. Without it the last equation of Section III leaves page 2 and
    the paper runs to 9 pages, so that page is packed tightly.

- 2026-11-16, later, root source corrected (entries 16 and 17 of `CHANGES_2026-11-16.md`, against the version
  pasted into Overleaf that morning):
  - Section IV-A said the root position comes from the pose fit. The pipeline takes root position and heading
    from the annotated 3D box and only the body pose from the fit. Found while building the Munich test set
    (`research/src/munich_eval.py --check-dataset`).
  - Section IV-C said the root has no independent reference. It follows the annotated boxes, human labels in
    nuScenes and Waymo and automatic ones (MS3D) in the staged scenes.
  - Still 8 pages, with about three lines free on the last page.

- 2026-11-16, afternoon, external test and language pass (entries 18 to 67 of `CHANGES_2026-11-16.md`, against
  commit d945e11):
  - Section VI-E has a paragraph on an external test, the labeled everyday drives of the research vehicle
    (1,327 pedestrians in 114 scenes, root path only, the same checkpoints, scored once by the protocol frozen
    in commit 26f7e4e, report in `research/data/munich_report_label.txt`). On all pedestrians EgoPed keeps its
    lead over the prior (ADE 2.74 against 3.05 m, ECS 0.24 against 0.00, both under the rule) with a worse
    Brier score (0.246 against 0.193). On the 60 near the vehicle's path it does not beat the prior (ADE 4.18
    against 3.86 m, ECS 0.18 against 0.03, Brier 0.317 against 0.148). The limitation on observational
    coupling names this. The city is not named.
  - Erik, 2026-11-16: the release is stated exactly. The conclusion says that the paired dataset will be
    released, but not the everyday drives, whose pedestrians are members of the public.
  - Paid for by folding the Guidance subsection into Section VI-B (PedGen is now VI-F, Qualitative Results
    VI-G), by removing three sentences whose numbers are in the tables, and by shorter sentences in Sections
    VI-C and VI-F.
  - Language pass asked for by Erik, with an independent read of the whole text. Sixteen sentences of 40
    words or more are split, no fragment was found, and about thirty slips of grammar, reference and
    terminology are fixed. Section VI-C no longer says "all four per-condition metrics" while naming three.
  - 8 pages, no overfull box, 39 references, about two lines free on page 8. The abstract has 1,193
    characters.

- 2026-11-18, Mohan's figures and text integrated (`CHANGES_2026-11-18.md`, against the Overleaf version of
  that day):
  - Fig. 4 is Mohan's CARLA figure (`figures/carla_main_20261117/`), three test conditions rendered from the
    export of 2026-11-13, one of them a failure on a staged crossing. Fig. 5 is new, his MotionBricks
    realization of one sample (`figures/motionbricks_20261117/`), with a paragraph in Section VI-G and one
    reference more (40). `main.tex` reads the captions and the two paragraphs from files in those folders.
  - His version shortens Sections VI-C to VIII and the captions of Tables IV and V by about 40 lines to pay
    for the second figure. The shortening is kept. 25 sentences are corrected where it had made a statement
    wrong, contradictory or unclear, and three clauses of reasoning are restored.
  - The MotionBricks PDF did not embed its fonts. The paper uses a copy with the fonts embedded
    (`FINAL_MOTIONBRICKS_CARLA_embedded.pdf`, images untouched).
  - His captions and paragraphs follow the language rules now (no colon or semicolon, full sentences).
  - 8 pages, no overfull box, 40 references, all fonts embedded, 4.4 MB. One more line fits, a second does
    not. The abstract is unchanged (1,193 characters).

## Still to do

- The paper is full, so anything added needs a cut of the same length. Pages 6 to 8 are one flow and the
  references quantize its end. One more line fits at present, a second gives 9 pages.
- Overleaf needs `refs.bib` and `ral_ctl.bib` from this folder, both figure folders and the font-embedded
  MotionBricks PDF. `main.tex` there must keep the three `\input` lines for Mohan's files.
- To settle with Mohan before submission (details in `CHANGES_2026-11-18.md`):
  - His `model_provenance.json` marks the MotionBricks material `"publication_approved": false`, and his
    README says the figure is deliberately not in the live `main.tex`, yet it is. Confirm that it is meant to
    be in the paper and that the MotionBricks license allows it.
  - Which control sets the head pose in Fig. 5. The paragraph credits retiming and root projection.
  - The footer inside Fig. 5 has two semicolons and a sequence name, and the script should embed the fonts.
- The audit files in the two figure folders hold names and machine paths. They are not in the PDF and must
  not go into supplementary material unchanged.
- The dataset still has no name (Flohr's comment). It needs a neutral one that names neither the lab, the
  city nor the vehicle, and the web page goes up with the camera-ready only.

## Abstract (plain text, for the submission form)

Simulation testing of automated vehicles needs pedestrians whose full-body motion depends on the vehicle, yet motion generators condition on text, waypoints, goals or static scenes. We present EgoPed, a latent diffusion model that generates 3D full-body pedestrian motion from the ego vehicle's trajectory, encoded by a pretrained, frozen encoder into one token. We use 5,668 paired sequences from nuScenes, Waymo and staged recordings, mirrored and split by scene. EgoPed lowers the root trajectory error of a trained unconditional prior from 3.03 to 2.43 m and raises our Ego-Coupling Score from 0.02 to 0.31. The score is how much more often generated pedestrians stop if the recorded one stops than if it walks. A deterministic regressor errs less (1.97 m) but barely couples (0.08). The Fréchet distance is blind to this coupling, as it ranks an ego-blind prior first and favors two designs that raise mean trajectory error. We thus propose per-condition trajectory error and our score, with scene-clustered statistics across seeds. The coupling is observational, resting mostly on standing pedestrians and partly on scene geometry, and the body labels are a lightly articulated template.
