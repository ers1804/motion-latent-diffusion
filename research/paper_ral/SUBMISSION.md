# RA-L submission (IEEE Robotics and Automation Letters)

Converted from `research/paper/` (OJ-ITS, ieeeoj.cls) on 2026-11-09. Self-contained: upload this folder to
Overleaf as a new project. Build: `latexmk -pdf main.tex`.

## Rules applied (RA-L Information for Authors, checked 2026-11-09)

- 6 pages in two-column IEEE format **including figures and references**, up to 2 more at $175 each.
  Current build: **8 pages**, about 12 free lines at the end of the references.
- Initial submission uses the RAS `ieeeconf.cls` (from ras.papercept.net/conferences/support/files/ieeeconf.zip),
  US letter. The final version moves to `\documentclass[letterpaper, 10 pt, journal, twoside]{IEEEtran}`.
- **Double-anonymous review**: no names, affiliations, acknowledgment or funding, no lab logos, and no individual
  vehicle names. `\anonymoustrue` in main.tex hides authors and acknowledgment. The vehicle photo figure is gone
  (it shows the vehicle name, the university name, a QR code and the licence plate), "AVA" is now "staged" or "our
  research vehicle", and Munich is not named.
- No biographies.
- Abstract field in PaperCept: at most 1,200 characters, no TeX. The abstract below matches the paper (1,192).
- Keywords: 2-5, the first two from the RA-L list. Used: Human and Humanoid Motion Analysis and Synthesis;
  Intelligent Transportation Systems; Deep Learning Methods; Modeling and Simulating Humans; Performance
  Evaluation and Benchmarking.
- PDF at most 5 MB (current 2.6 MB), fonts embedded, no author in the PDF metadata.

## What changed against the OJ-ITS text

- Related work condensed; 49 references instead of 67; author lists of four or more names print as "et al."
  (`ral_ctl.bib`).
- The pose-audit, guidance and PedGen tables are folded into the text with their numbers.
- Shorter captions, discussion and limitations; the broader-impact paragraph is dropped.
- Fixed a broken cross-reference in the teaser caption (it printed "Section c").

## Still to do

- Copy Mohan's qualitative figure and CARLA paragraph from the OJ-ITS Overleaf (red TODOs in Section VI-G and
  the caption of Fig. 4). For the review version the figure must not show the vehicle's name, lab logos or
  faces, and the staged row must not be labelled "AVA". Keep it at most the height of the current placeholder,
  or the paper goes to 9 pages.

## Abstract (plain text, for the submission form)

Simulation-based testing of automated vehicles needs pedestrians whose whole-body motion responds to the vehicle, yet existing motion generators condition on text, waypoints, goals or the static scene. We present EgoPed, a latent diffusion model that generates 3D full-body pedestrian motion directly from the ego vehicle's trajectory, encoded by a pretrained, frozen trajectory encoder into a single token. On 11,786 paired sequences from nuScenes, Waymo and our staged recordings, split by scene, EgoPed lowers the root displacement error of a separately trained unconditional prior from 3.11 to 2.43 m and lifts the coupling between generated and recorded stopping decisions (Ego-Coupling Score) from 0.02 to 0.31, across three seeds. The Fréchet distance is blind to this coupling: an ego-blind prior matches the best conditional model, and the two design choices it favours both lower per-condition accuracy. We therefore propose per-condition trajectory error and the Ego-Coupling Score, with scene-clustered statistics across seeds, as the evaluation for this task. An audit of the pose labels shows that root trajectories are reliable while the body is a lightly articulated template.
