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
- Abstract field in PaperCept: at most 1,200 characters, no TeX. The abstract below matches the paper (1,199).
- Keywords: 2-5, the first two from the RA-L list. Used: Human and Humanoid Motion Analysis and Synthesis;
  Intelligent Transportation Systems; Deep Learning Methods; Modeling and Simulating Humans; Performance
  Evaluation and Benchmarking.
- PDF at most 5 MB (current 2.6 MB), fonts embedded, no author in the PDF metadata.

## What changed against the OJ-ITS text

- Related work condensed; 39 references instead of 67; author lists of four or more names print as "et al."
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

## Still to do

- A full retraining with the ego in the frame of the window start (`rs2`) is queued. If it replaces the
  current models, every table changes.
- Pending results that will change numbers in the text: two more seeds each for the prior and the prior with
  IA (Tables II and III, the per-model correlation, the seeds sentence in Section V). The paper is full,
  so anything added needs a cut of the same length.
- Overleaf needs `refs.bib` and `ral_ctl.bib` from this folder. The reviewed copy printed "[?]" for some
  citations, which this build does not.

- Copy Mohan's qualitative figure and CARLA paragraph from the OJ-ITS Overleaf (red TODOs in Section VI-G and
  the caption of Fig. 4). For the review version the figure must not show the vehicle's name, lab logos or
  faces, and the staged row must not be labelled "AVA". Keep it at most the height of the current placeholder,
  or the paper goes to 9 pages. No colons or semicolons in the caption, paragraph or figure text.
- The dataset still has no name (Flohr's comment). It needs a neutral one that names neither the lab, the
  city nor the vehicle, and the web page goes up with the camera-ready only.

## Abstract (plain text, for the submission form)

Simulation-based testing of automated vehicles needs pedestrians whose whole-body motion depends on the vehicle, yet existing motion generators condition on text, waypoints, goals or the static scene. We present EgoPed, a latent diffusion model that generates 3D full-body pedestrian motion directly from the ego vehicle's trajectory, encoded by a pretrained, frozen trajectory encoder into a single token. On 5,668 paired sequences from nuScenes, Waymo and our staged recordings, mirrored and split by scene, EgoPed lowers the root displacement error of a trained unconditional prior from 3.11 to 2.43 m and lifts the coupling between generated and recorded stopping decisions (Ego-Coupling Score) from 0.02 to 0.31. The Fréchet distance is blind to this coupling. An ego-blind prior matches the best conditional model, and the two design choices it favours both raise mean trajectory error. We therefore propose per-condition trajectory error and the Ego-Coupling Score, with scene-clustered statistics across seeds, to evaluate this task. The coupling is observational. It rests mostly on pedestrians who stand and partly on scene geometry, and the body labels are a lightly articulated template.
