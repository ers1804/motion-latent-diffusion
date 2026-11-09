> Review of `paper_ral/main.tex` as of ece68a0 (line numbers refer to that version). Applied in the next commit except where SUBMISSION.md lists an item as open.

# Pre-submission review of `research/paper_ral/main.tex` (RA-L version), 2026-11-09

Reviewed against the built PDF (8 pages, rebuilt from source in scratch with latexmk, zero overfull boxes, no undefined references or citations), the professor's 25 Overleaf threads, `review_action_plan_2026-10-30.md`, and the result files `resplit_test_report.txt`, `resplit_ablation_test_report.txt`, `resplit_ego_controls.txt`, `resplit_breakdown.txt`, `resplit_selection.json`, `resplit_ablation_selection.json`, the per-run `ade_fde_*_scene_test_k5.npz` dumps, `pose_quality_audit.npz`, `vae_deslid_pilot.npz` and `skate_decomposition.npz`. Note that the result files use the OLD naming (`rs1_egoped_*` is the paper's Full-seq, `rs1_pooled_*` is the paper's EgoPed). Line numbers refer to `main.tex` as of this review. The qualitative figure (Fig. 4) and the CARLA paragraph are red TODOs owned by a co-author and were not reviewed for content.

## 1. Summary verdict

The paper is in good shape for submission once the co-author's figure lands. Every number I could trace reconciles with the result files (Tables I to V, abstract, introduction, Sections VI-A to VI-F, conclusion); only the six validation FIDs of the guidance sweep (Sec. VI-E) have no source file in the repository, and two table entries differ by a rounding unit. Three statements use comparative language that does not meet the paper's own "beats" rule (prior vs PedGen on ADE, "lowers every per-condition metric", "most strongly coupled" / "weaken the coupling"), and all three have one-line fixes. Of the professor's 25 threads, 16 are addressed, 5 are moot or handled elsewhere (venue, photo, acknowledgment, training-curve figure, ladder numbers), 3 are partly addressed (qualitative figure pending, orientation test not mentioned, pose audit not positioned against PedGen's label filtering), and 1 is not addressed: the robotics spill-over is in the discussion but not in the introduction, which matters for RA-L. The dataset-naming request conflicts with double-anonymous review and should be handled with a neutral name and no URL. The prose contains 70 colons and semicolons; verified exact-match rewrites follow in Section 5, net +123 characters across the whole paper (roughly line-neutral), plus a class-level fix for the ".:" that ieeeconf prints after every run-in heading.

## 2. Professor's comments, thread by thread

Legend: **addressed** / **partly** / **not addressed** / **moot** (no longer applies, or handled outside the paper).

| # | Comment (shortened) | Status | Evidence in RA-L main.tex | Fix |
|---|---|---|---|---|
| 1 | Motivation / teaser image on first page | **addressed** | Lines 61 to 73 hook `\@maketitle` so Fig. 1 (`figures/teaser.pdf`) sits under the title block on page 1; render checked, two conditions, no leaks. | None. |
| 2 | Max 3 contributions; state what is new vs SoA | **addressed** | Lines 107 to 112, exactly three items; differentiation at lines 95 to 99 (event/path predictors, text-conditioned generators, waypoint/goal/scene control, ControlPed via 2D path and prompt) and line 109 "To our knowledge the first model". | Residual wording: contribution 2 says "lose on per-condition accuracy"; Full-seq wins minADE, so write "mean" (finding M3). |
| 3 | More qualitative results, also failures, Waymo and staged data | **partly (pending co-author)** | Fig. 4 caption placeholder, line 418, requires "test-partition conditions, including Waymo, a staged scene and a failure case, with no vehicle name or lab logo". | Requirements are sensible. Add two: the staged row must be labelled "staged", not by vehicle name, and the figure must not exceed the placeholder height (SUBMISSION.md already says so). The permanent part of the caption has three colons (C70). |
| 4 | Spill-over to robotics in discussion and outlook | **addressed** | Lines 433 to 434, paragraph "Outlook: spill-over to robotics", JRDB cited, labelled as untested. | Heading colon (C64). |
| 5 | Acknowledgments / funding | **moot for review, addressed for camera-ready** | Lines 455 to 458 inside `\ifanonymous\else ... \fi`; `\anonymoustrue` at line 41, so the build omits it. Flipping to `\anonymousfalse` restores authors (lines 48 to 50) and the acknowledgment. RA-L requires this omission in the review version. | None. |
| 6 | Reference consistency (first reference without abbreviations vs the rest) | **partly** | Journal and conference names are now IEEE-abbreviated throughout (rendered list checked, [1] "IEEE Trans. Intell. Transp. Syst."). Two residual inconsistencies: [5] and [42] print no venue because the entries are `@inbook`; title capitalisation is mixed (28 cited titles double-braced and kept in Title Case, 21 single-braced and sentence-cased by IEEEtran.bst, producing "3d", "Omnire", "Ms3d"). | Finding m10. |
| 7 | "AVA": nobody knows it; name the dataset and tease it on the lab page | **not addressed, and must change form** | The paper never names the dataset ("our staged recordings" line 80, "Staged (ours)" line 219, "our research vehicle" line 111/190); line 452 promises release. A lab URL or lab-specific name would break double-anonymous review (RA-L rules, also SUBMISSION.md). | Give the dataset a neutral name that does not reference the lab, city or vehicle (one word in line 190 and the Table I caption), say "will be released on acceptance" (line 452 already nearly does), and put the web page up with the camera-ready only. Do not reuse "AVA" (clash with Google's AVA, and it identifies the vehicle). Length-neutral. |
| 8 | Is there a name for this new metric? | **addressed** | "Ego-Coupling Score (ECS)" defined at line 257, used in abstract, tables and text. | None. |
| 9 | Is contribution 2 (full-seq FID gain) still good given contribution 3 disqualifies FID? | **addressed** | The FID gain is no longer claimed; full-sequence attention is an ablation (lines 109 to 110, 325, 337 to 338), and FID is used only for checkpoint selection (line 325). | None. |
| 10 | Spill-over to robotics in the introduction | **not addressed** | `grep -n robot main.tex` hits only lines 433 to 434. The introduction (lines 92 to 112) is AV-only. | Finding M4: one sentence in the introduction, with compensating cuts. |
| 11 | Can you claim to be the first? | **addressed** | Line 109 "To our knowledge the first model that generates 3D pedestrian body motion directly from the ego-vehicle trajectory"; ControlPed positioned as concurrent and indirect (lines 99, 129). | None. |
| 12 | Venue? RA-L? Page limit 8 for RA-L; get the correct template | **moot / addressed** | RA-L with `ieeeconf.cls` (line 1), 8 pages confirmed by rebuild, header comment notes the move to IEEEtran for the final version. | None. |
| 13 | Photo copyright | **moot** | The vehicle photo figure is gone (no `.jpg` in `figures/`, no `fig:ava`). | None. |
| 14 | Pooled without IA missing in Tables 7 and 8; is the benefit architecture or recipe? | **addressed** | Full 2x2 with three seeds per cell in Tables II and III (lines 268 to 312); the recipe question is answered at line 325 (IA hurts both encoders) and line 338 (architecture contrast). | None. |
| 15 | Closest prior work missing (Trace and Pace, PedGen, ControlPed); PedGen as external baseline | **addressed, one clause missing** | Cited and positioned at lines 99 and 127 to 129; PedGen baseline in Sec. VI-F (lines 405 to 408). The professor's point that PedGen and Trace and Pace address the label-noise / foot-skate problem is not reflected: the pose audit (Sec. IV-C) is not positioned against PedGen's label filtering. | Add one clause in Sec. II-C or II-D, e.g. after line 129 "PedGen filters noisy pseudo-labels; we measure ours" (written without the semicolon: "PedGen filters noisy pseudo-labels, whereas we measure how noisy ours are."). +1 line. |
| 16 | Unfrozen is rather a trade-off (follows tighter on average, frozen covers more and is calibrated)? | **addressed with new evidence** | The three-seed pooled ablation (Table IV, lines 340 to 346) shows the frozen encoder better on all four per-condition metrics under the rule, so it is not a trade-off for the pooled encoder; the concentration of samples (MM 2.0 vs 3.10) is described, and the single-seed full-seq trainable run is framed as a failure mode (line 346, Table II). | Numeric nit m3 ("2.0--2.1" should be "2.0--2.2"). |
| 17 | Training-curve figure unreadable | **moot** | Figure removed. | None. |
| 18 | Title vs body: conclusions rest on the path; either retitle or test body/head orientation toward the vehicle; would GT labels show it? | **partly** | The paper keeps "Body Motion Generation" in the title and says plainly that the conditioning evidence is root-level (lines 242, 442). The orientation test on labels (action plan step 7: facing and walking direction agree, head yaw link to the vehicle grows with distance, i.e. estimator bias, so no evidence) is not mentioned anywhere. | Add one sentence to Sec. IV-C after line 242 or to Limitations line 442, e.g. "The labels also show no orientation of head or torso toward the vehicle beyond an estimator bias that grows with distance, so we claim trajectory-level behaviour only." (+2 lines; cut line 239 to pay for it). The title is defensible because the model does generate body motion and the paper states what the evidence covers. |
| 19 | Leak: checkpoints selected on the held-out set | **addressed** | Scene-level split (lines 203 to 205), selection on validation and single test pass (lines 205, 260 to 261), unconditional selection offline with ego zeroed (line 260), last-checkpoint ablation row pre-declared (line 345). | Tense fix m6 at line 205. |
| 20 | Numbers differ from Table 5; always cite the specific table | **moot** | The ladder is gone. Every number in the text now has a table or figure reference (checked in Section 4). | None. |
| 21 | Dimensions: SMPL R^{T x 22 x 3} vs e in R^{196 x 2} | **addressed** | Line 149 defines e as the ground-plane position over T = 196 frames; lines 195 to 196 give the SMPL to HumanML3D conversion. | None. |
| 22 | Typo "Iits" | **moot** | Sentence removed. | None. |
| 23 | "breaked" | **addressed** | Line 192 "brakes". | None. |
| 24 | Do not put the 40% FID gain first when disqualifying the metric | **addressed** | Abstract leads with ADE and ECS, then says FID is blind (line 80). | None. |
| 25 | Teaser page please | **addressed** | Same as thread 1. | None. |

## 3. Findings ranked by severity

No critical (submission-blocking) finding apart from the two red TODOs owned by the co-author. Length impact is stated for every fix; "neutral" means the same number of lines in the current layout.

### Major

**M1. "Our own trained prior is closer than PedGen as well" fails the rule (line 408).** `resplit_test_report.txt`, section "Paired against PedGen": `rs1_uncond_s1234` ADE +0.043 [−0.021, +0.103], FDE +0.056 [−0.067, +0.179]; both intervals include zero. Only minADE (+0.136 [+0.091, +0.183]) and minFDE exclude it. The sentence makes a comparative claim the paper's own rule does not support, and the argument that follows ("so part of the gap is the domain") still works with a weaker statement. Fix (also removes the colon, supersedes C61), −6 characters:

```
OLD: Our own trained prior is closer than PedGen as well, so part of the gap is the domain: PedGen's samples walk at a median 0.65\,m/s against 1.26\,m/s in our data.
NEW: Even our ego-blind prior is level with PedGen (0.750 against 0.794\,m, interval including zero), so part of the gap is the domain. PedGen's samples walk at a median 0.65\,m/s against 1.26\,m/s in our data.
```

**M2. "it lowers every per-condition metric for both encoders" (line 325) is wrong in three ways.** (a) "lowers" is the wrong direction word for ADE, where lower is better. (b) Full-seq vs Full-seq+IA on ADE is +0.31 [0.11, 0.55] with seed-pair range (−0.163, +0.612), so one seed pair disagrees and the rule is not met (Table III row 4 has no asterisk). (c) Brier is a per-condition metric by Table II's own caption, and EgoPed vs EgoPed+IA on Brier is +0.004 [−0.021, +0.031]. Fix, +18 characters, same line count likely:

```
OLD: and at the same time it lowers every per-condition metric for both encoders (Table~\ref{tab:contrasts}, rows 3--4); the IA models are no better than their own prior on mean ADE.
NEW: and at the same time it worsens mean ADE, minADE and ECS for both encoders (Table~\ref{tab:contrasts}, rows 3--4, five of six under our rule), and the IA models are no better than their own prior on mean ADE.
```
(This replaces C42.)

**M3. Coupling comparisons between EgoPed and Full-seq without IA do not meet the rule, but two sentences treat them as settled.** Table III row 1: ECS +0.04 [−0.00, 0.08], no asterisk. Line 325 calls EgoPed "the most strongly coupled"; line 452 says FID "favours the design choices that weaken the coupling" (true for IA, +0.09* and +0.17*, not established for Full-seq on test). The abstract (line 80) and contribution 2 (line 110) say "lower per-condition accuracy", which is supported (ADE +0.26* and +0.50*) but should say "mean" because Full-seq wins minADE (−0.11*). Fixes, all length-neutral:

```
OLD: while being the most accurate on mean ADE and the most strongly coupled.
NEW: while being the most accurate on mean ADE and no weaker in coupling.
```
```
OLD: it favours the design choices that weaken the coupling and places an ego-blind prior level with the best conditional model.
NEW: it favours the design choices that raise per-condition error and places an ego-blind prior level with the best conditional model.
```
```
OLD: and the two design choices it favours both lower per-condition accuracy.
NEW: and the two design choices it favours both raise mean trajectory error.
```
(The abstract edit keeps the PaperCept text under 1,200 characters. It is 1,192 now and 1,191 after C4 and this change, recounted. Update the copy in SUBMISSION.md.)
```
OLD: and both design choices FID prefers lose on per-condition accuracy.
NEW: and both design choices FID prefers lose on mean trajectory error.
```

**M4. The robotics spill-over is missing from the introduction (Thread 10, RA-L fit).** The introduction speaks only to AV simulation. RA-L reviewers will ask why this is a robotics letter. One sentence after line 95 ("...not an articulated body.") makes the link and sets up Sec. VII-c:

```
OLD: The models built on them, however, predict a discrete event or a ground-plane path~\cite{alahi2016sociallstm,salzmann2020trajectron,gu2022mid}, not an articulated body.
NEW: The models built on them, however, predict a discrete event or a ground-plane path~\cite{alahi2016sociallstm,salzmann2020trajectron,gu2022mid}, not an articulated body. Mobile robots that share pavements with people face the same gap, and their simulators need pedestrians that react to the robot~\cite{martinmartin2023jrdb}.
```
Adds about 2 lines. Pay for it with m14 and m15 (−2 lines together), or with m19 (−2 lines).

**M5. Dataset naming under double-anonymous review (Thread 7).** See the table above. A neutral name (not lab, city or vehicle), "released on acceptance", no URL. Also keep "Staged (ours)" in Table I, which is fine. Length-neutral.

**M6. ADE and FDE are never expanded.** First use is line 253 ("ADE and FDE average over the samples"); line 121 only says "displacement metrics". Fix at line 253, about +6 words (same paragraph line count likely):

```
OLD: ADE and FDE average over the samples, minADE and minFDE take the best of five.
NEW: The average and final displacement errors (ADE, FDE) average over the samples, minADE and minFDE take the best of five.
```
(Apply after C32, which only changes the preceding colon.)

**M7. 13,958 audited sequences vs 11,786 dataset sequences is unexplained (line 235 vs Table I).** The audit ran on every extracted sequence (14,172 extracted, 13,958 scorable, per `pose_quality_audit.npz`), the dataset holds 11,786. A reviewer will ask where 2,172 sequences went. One clause is enough, but the author must supply the reason (filtering before the split, missing ego pairing, or length). Suggested form, +4 words: "on all 13{,}958 extracted sequences long enough to score, a superset of the released 11{,}786".

**M8. "the combination with the weakest coupling to the vehicle among our conditional models" (line 428).** The single-seed Full-seq with trainable encoder (ECS 0.04, Table II) is a conditional model with weaker coupling than Full-seq+IA (0.10). Fix, −4 characters:

```
OLD: the combination with the weakest coupling to the vehicle among our conditional models.
NEW: the combination with the weakest coupling to the vehicle among our four main models.
```

**M9. Every run-in heading prints ".:" (eleven times, e.g. "a) Pretraining.:" on page 2).** `ieeeconf.cls` line 3857 defines `\def\@IEEEsectpunct{:\ \,}` and appends it after `\paragraph` titles, which in main.tex already end in a period (lines 163, 229, 249, 252, 255, 259, 337, 340, 427, 430, 433). Two options, both length-neutral: (a) keep the periods and switch the class punctuation off, which also removes the only remaining colons in the paper, by adding to the preamble after line 29

```
\makeatletter\def\@IEEEsectpunct{\ \,}\makeatother % ieeeconf prints ":" after run-in headings; the titles end in a period
```
or (b) delete the trailing period inside each `\paragraph{...}` and accept the class colon ("a) Pretraining:"). Option (a) matches the author's style rule. IEEEtran, used for the final version, has the same macro name, so the line carries over.

### Minor

**m1. Validation FIDs in Sec. VI-E are not reproducible from the repository (line 403).** "4.61 to 4.07 and 3.90" (EgoPed) and "3.29, 2.80, 2.74" (Full-seq) have no source in `research/data`; only the two cfg-15 values match the action plan (Pooled 3.90, EgoPed 2.74). The ADE, minADE and ECS values in the same sentence are confirmed by `resplit_breakdown.txt`. Export the `eval_val` sweep metrics (the `select_resplit.py --guidance` step reads them) or re-check the six numbers.

**m2. Two table entries are off by a rounding unit if the per-seed values in the reports are exact.** Table II, EgoPed+IA Brier 0.195: per-seed 0.197, 0.186, 0.200 average 0.1943. Table IV, last checkpoint, pretrained trainable MM 1.65: per-seed 1.635, 1.532, 1.767 average 1.6447. Check against unrounded values; otherwise 0.194 and 1.64.

**m3. "MM 2.0--2.1 against 3.10" (line 344) vs Table IV 2.15 and 2.03.** Write "2.0--2.2".

**m4. "+0.04 [−0.00, 0.08]" (line 302).** A signed zero looks like an error. Print `\ci{0.00}{0.08}` (the unrounded bound is +0.001).

**m5. "one \method model and one noise seed" (line 412) vs Fig. 1 caption "8 noise seeds" (line 70).** Fig. 1 renders one seed and draws the other seven root paths. Write "one \method model under the same noise and two vehicle trajectories" or "one model, one of eight noise seeds rendered".

**m6. Tense (line 205).** "We select models on the validation partition and evaluated the test partition once" → "We selected models on the validation partition and evaluated the test partition once".

**m7. British/American spelling.** The paper is British (behaviour, modelling, colour, centres, kerb, favours, initialised, manoeuvre, travelled). American forms left: "normalized" (lines 155, 197), "localization" (line 191), "localizing" (line 195). "Modeling" in the Index Terms (line 85) is RA-L's official keyword and must stay.

**m8. Misplaced citation (line 121).** Rudenko et al. [23] is a survey of *human* trajectory prediction but is attached to "MTR does the same for vehicles". Move it: "Trajectory predictors treat pedestrians as points on the ground plane~\cite{rudenko2020survey}, from social pooling..." and end the sentence "...does the same for vehicles." Neutral.

**m9. "scene-~\cite{...} and partner-conditioned" renders as "scene- [32], [33] and partner-conditioned [34]" (page 2).** Write "scene-conditioned~\cite{hassan2021samp,jiang2024trumans} and partner-conditioned~\cite{liang2024intergen} models". +11 characters.

**m10. Bibliography.** (a) `salzmann2020trajectron` and `rempe2020contact` are `@inbook`, so IEEEtran prints the title in italics and drops the ECCV venue ([5], [42]) and bibtex warns "missing publisher"; change both to `@inproceedings`. (b) `rudenko2020survey` has no volume or pages (Int. J. Robot. Res., vol. 39, no. 8, pp. 895–935, 2020). (c) Title case is mixed: 28 cited titles are double-braced and stay in Title Case, 21 are single-braced and sentence-cased by IEEEtran.bst, which lowercases unprotected acronyms ("3d" in [10], [24], [44], [45]; "Omnire" [47]; "Ms3d" [44], [45]). Pick one convention; the cheapest is to keep IEEEtran's sentence case, remove the outer braces from the 28 entries and protect acronyms as {3D}, {OmniRe}, {MS3D}, {MS3D++}, {LSTM}, {GAN}, {PIE}, {JAAD}, {nuScenes}, {FID}, {GANs}, {ImageNet}, {MMD}, {DDPM}... (d) `ho2022cfg` is an arXiv entry, acceptable. (e) 31 uncited entries in refs.bib are harmless. The reference list has about 12 free lines, so (a) and (b) fit.

**m11. Tables as actors (lines 341, 373).** "Table~\ref{tab:design} trains \method's encoder together with the denoiser" and "Table~\ref{tab:controls} feeds \method's three seeds manipulated ego input". Suggested: "Table~\ref{tab:design} reports \method with its encoder trained together with the denoiser" (+8 characters) and "Table~\ref{tab:controls} gives \method's three seeds manipulated ego input" (−1). Optional.

**m12. Five names for one model.** "separately trained unconditional prior" (line 80), "ego-blind prior" (lines 80, 110, 325, 452), "trained prior (no ego)" (Table II), "prior" (Table III), "a prior trained on the same data without the ego trajectory" (line 318). Define once at line 318 and then use "the trained prior" in text, keeping "ego-blind" as an adjective. Neutral.

**m13. "Fréchet distance (FID)" (line 104).** FID conventionally stands for Fréchet Inception Distance; motion papers keep the acronym but say "Fréchet inception distance". Either write "Fréchet inception distance (FID)" or keep it and accept a reviewer remark. Neutral.

**m14. Sensor list (line 191) is long and identifying.** "seven 128-layer LiDARs, eight RGB cameras facing forward, sideways and back, and a dual-antenna GNSS receiver" is a distinctive configuration. "The vehicle's LiDARs and surround cameras supply the 3D boxes, a dual-antenna GNSS receiver the ego trajectory." saves about one line and lowers identifiability.

**m15. Line 103 is informal and long.** "Showing that such a model \emph{uses} the vehicle is harder than it looks, and most of this paper is about how to tell." → "Showing that such a model \emph{uses} the vehicle is the harder part." Saves about one line.

**m16. Table II caption says bold marks the best per column, but the Div. column has no bold.** Either bold the value closest to the ground-truth range (EgoPed 5.63) or write "Bold marks the best of the four main models per column except Diversity".

**m17. "Ground-truth Diversity is 5.26--5.40" (Table II caption, line 250).** The two values come from the two evaluator passes (5.259 for the Full-seq cells, 5.402 for the EgoPed cells). A reviewer may wonder why the ground truth has a range; "(range over evaluation passes)" explains it, +4 words. Optional.

**m18. Fig. 2 smallest labels** ("gradient reaches the ego encoder only", tensor shapes, legend) are about 5 pt at print size. Readable on screen, marginal on paper. Increase the font in `figures/architecture.tex` without changing the figure height. No page impact.

**m19. "In an earlier version of this pipeline" (line 239)** gives no provenance for the 0.93--0.94 round-trip numbers (they are the old-split models, `skate_decomposition.npz`: VAE round-trip 0.939, generated 0.931--0.944). Either say "with models trained on the earlier file-level split" or cut the sentence (−2 lines); the pilot sentence that follows carries the argument.

**m20. Spearman with ECS (line 325).** ρ = +0.48 over 15 runs has p = 0.07 (ρ = −0.74 with ADE has p = 0.002, both reproduced from the test report). The text states it as fact; "(the latter not significant at 5\%)" would pre-empt a reviewer. Optional, +1 line.

**m21. Fig. 4 placeholder requirements** (line 418) are sensible; see thread 3 for two additions. The TODO bracket on line 413 also contains a colon ("[Mohan: ...]"); it disappears with the TODO.

**m22. PaperCept abstract.** 1,192 of 1,200 characters, identical to the main.tex abstract. C4 is length-neutral and the M3 abstract edit is −1 (1,191 characters after both), so it stays under the limit, but any other abstract edit must be recounted and SUBMISSION.md updated.

**m23. Anonymity, verified clean.** Text: no author, lab, city or vehicle name outside the `\else` branch (grep for AVA, Munich, Hochschule, hm.edu, author surnames, STADT, BMWK: only lines 48 to 50 and 457, both compiled out). Figures: teaser, architecture, FID plot and the CARLA placeholder rendered and inspected, no names or logos; panel IDs "0002_26" etc. are harmless. PDF metadata: Author and Title empty. Bib: no self-citation. The Camara et al. author list contains a "Weber, Florian", unrelated to the removed photo credit.

**m24. Layout, verified.** Floats are within one page of first mention (Fig. 2 on p. 3 for p. 2 mention; Tables II/III and Fig. 3 on p. 5 for p. 4 mention; Tables IV/V on p. 6; Fig. 4 on p. 7 for p. 6 mention). No overfull boxes. Page 8 has about 15 free lines in the right column. Sec. VI-G is two sentences plus a TODO; the co-author's paragraph must stay under about 8 lines or the paper goes to 9 pages.

## 4. Number checks

Every number in the abstract, introduction, results text, tables and conclusion was traced. Paper names: EgoPed = `rs1_pooled`, Full-seq = `rs1_egoped`, prior = `rs1_uncond`, +IA = `_ia`. "OK" means the rounded value matches the source.

| Where (line) | Paper value | Source and value | Status |
|---|---|---|---|
| Abstract 80, Intro 109, VI-A 318, Concl. 452 | prior ADE 3.11 → EgoPed 2.43, gap 0.68 | test report: uncond 3.110; pooled 2.382/2.476/2.445 mean 2.434; contrast +0.676* [0.487, 0.858], seed pairs (+0.634, +0.728) | OK, rule met |
| same | ECS 0.02 → 0.31, +0.29 | uncond 0.015; pooled 0.335/0.299/0.290 mean 0.308; contrast +0.293* | OK, rule met |
| Abstract 80, VI-B 325 | best FID 3.24 (Full-seq+IA), prior+IA 3.29 | egoped_ia FID 2.727/2.867/4.137 mean 3.244; uncond_ia 3.287 | OK |
| Abstract 80, Table I | 11,786 sequences | Table I sums: 9,410 + 1,194 + 1,182; rows 34 + 4,514 + 7,238; action plan split table | OK |
| Table I | all cells, 994 scenes | action plan "Step 4" table, 796 + 107 + 91 | OK |
| IV-A 192 | 17 staged scenarios | Table I staged total 34 with mirrored copies = 17 originals | OK (consistent) |
| IV-A 199, V-c 257 | about 40% windowed; stopping 24%; crossing about 2% | OJ-ITS text identical; test report "stopping rate=0.240" | 40% and 2% not independently verifiable; 24% OK |
| IV-B 204 | 83% mirrored twin in training (file split) | action plan, memory note | OK |
| IV-C 235 to 240 | 13,958 scorable; 10^-5 bone length; skate 0.97; 94.6%; 1.29x, 0.44x, 0.26x, 0.37x; 0.93--0.94; skate 0.15; 1,418 | `pose_quality_audit.npz` 13,958 rows; audit memo 6.3e-6, 0.97, 94.6% of 10,900 locomotion seqs, 1.29/0.44/0.26/0.37; `skate_decomposition.npz` 0.939/0.944/0.931; `vae_deslid_pilot.npz` recon_finetuned 0.15, n_sequences 1418 | OK (see M7 for 13,958 vs 11,786) |
| V-a 250, Table II caption | GT Diversity 5.26--5.40 | test report gt_Diversity 5.259 and 5.402 | OK |
| V-d 260 | 50 checkpoints every 100 epochs; guidance from {5, 10, 15} | 5,000 epochs; `resplit_selection.json` guidance 15/15/10/5/5/1 | OK |
| Table II, EgoPed | 5.67 [5.36, 6.26]; 5.63; 2.43 [2.38, 2.48]; 5.11; 1.65 [1.59, 1.74]; 3.40; 0.31 [0.29, 0.34]; 0.191 | FID 6.256/5.397/5.364; Div 5.412/5.587/5.906 (5.635); npz FDE 5.107, minFDE 3.404; Brier 0.185/0.195/0.193 | OK |
| Table II, Full-seq | 4.16 [2.96, 4.91]; 5.78; 2.69 [2.52, 2.98]; 5.67; 1.54 [1.49, 1.57]; 3.18; 0.27 [0.24, 0.29]; 0.178 | FID 4.596/2.965/4.914; Div 5.778; ADE 2.576/2.979/2.524; npz FDE 5.674, minFDE 3.176; ECS 0.287/0.241/0.284; Brier 0.1777 | OK |
| Table II, EgoPed+IA | 4.84 [4.69, 5.04]; 5.83; 2.93 [2.86, 3.03]; 6.23; 2.04 [1.98, 2.13]; 4.29; 0.21 [0.18, 0.25]; 0.195 | FID 4.789/4.694/5.041; Div 5.831; ADE 3.032/2.897/2.860; FDE 6.233; minADE 2.126/1.982/2.007; minFDE 4.292; ECS 0.179/0.248/0.216; Brier 0.197/0.186/0.200 = 0.1943 | Brier 0.195 vs 0.194 (m2); rest OK |
| Table II, Full-seq+IA | 3.24 [2.73, 4.14]; 5.89; 3.01 [2.82, 3.14]; 6.50; 2.07 [1.72, 2.31]; 4.42; 0.10 [0.06, 0.12]; 0.214 | FID 2.727/2.867/4.137; Div 5.893; ADE 3.136/3.066/2.816; FDE 6.504; minADE 2.307/2.186/1.720; minFDE 4.421; ECS 0.065/0.115/0.123; Brier 0.214 | OK |
| Table II, Full-seq trainable | 6.54; 5.32; 2.67; 5.55; 1.39; 2.79; 0.04; 0.209 | unfrozen 6.544; 5.318; 2.671; npz 5.552; 1.387; 2.793; 0.040; 0.209 | OK |
| Table II, prior | 4.67; 5.79; 3.11; 6.38; 1.31; 2.64; 0.02; 0.214 | uncond 4.667; 5.791; 3.110; npz 6.375; 1.307; 2.635; 0.015; 0.214 | OK |
| Table II, prior+IA | 3.29; 5.47; 3.00; 6.30; 1.52; 3.09; 0.01; 0.220 | uncond_ia 3.287; 5.475; 2.998; 6.300; 1.517; 3.086; 0.008; 0.220 | OK |
| Table III, all 8 rows x 3 columns | as printed | test report contrasts, signs flipped where the paper reverses the order (egoped vs pooled → EgoPed vs Full-seq); asterisks match "rule met" markers, including the missing asterisks on rows 1 (ECS), 2 (ADE, minADE), 4 (ADE), 7 (ADE), 8 (ADE) | OK, 24 of 24 |
| VI-A 318 | FDE 1.27 m | npz 6.375 − 5.107 = 1.268 | OK |
| VI-A 320 | validation ECS 0.36 vs 0.02 | ego controls scene_val: full 0.363, prior 0.024 | OK |
| VI-A 320 | nuScenes/Waymo ECS 0.32, 0.26; prior −0.01, 0.04 | `resplit_breakdown.txt` scene_test: pooled +0.32/+0.26, uncond −0.01/+0.04 | OK |
| VI-B 325 | FID 5.67 worst of four; Spearman −0.74 (ADE), +0.48 (ECS); 15 runs | recomputed from the 15 test-report rows: −0.739 (p = 0.002), +0.482 (p = 0.069) | OK (m20) |
| VI-B 325 | "IA models no better than their own prior on mean ADE" | rows 7, 8: +0.07 [−0.23, 0.34], −0.01 [−0.29, 0.25] | OK |
| VI-B 325 | "lowers every per-condition metric for both encoders" | row 4 ADE +0.31, one seed pair negative; Brier EgoPed +0.004 n.s. | NOT supported (M2) |
| VI-B 325 | "most strongly coupled" | row 1 ECS +0.04 [−0.00, 0.08] | NOT under the rule (M3) |
| VI-C-a 338 | 0.26 m; validation 0.37 m; MM 3.92 vs 3.10; minADE 0.11; ECS 0.27 vs 0.31 below rule on test, met on validation; IA 0.21 vs 0.10 rule met | +0.259*; plan validation −0.370*; MM 3.683/4.29/3.786 = 3.920 and 2.632/3.299/3.367 = 3.099; +0.112*; ECS +0.038 n.s. test, −0.065* validation; +0.114* | OK |
| VI-C-b 342 | ADE 0.09, minADE 0.25, ECS 0.07, Brier 0.025, all rule met; validation all but ADE; FID 5.67 vs 5.65 | ablation report: +0.093*, +0.253*, +0.073*, +0.025*; plan validation ADE +0.048 n.s., others *; FID 5.649 | OK |
| VI-C-b 343 | pretrained vs scratch same on every metric, both partitions | ablation report: all contrasts n.s. under the rule (val minADE −0.071 [−0.128, −0.008] but seed pairs disagree) | OK under the rule |
| VI-C-b 344 | MM 2.0--2.1 against 3.10 | Table IV 2.15, 2.03 | "2.1" should be "2.2" (m3) |
| VI-C-b 345 | 4 of 6 trainable runs selected at epochs 999--1099; last-checkpoint ADE gap 0.07 no longer meets rule | `resplit_ablation_selection.json` 999/3199/999, 1099/2499/999; last: +0.073 [−0.016, 0.160] | OK |
| VI-C-b 346 | ECS 0.04 vs 0.27 | Table II | OK |
| Table IV, selected rows | 5.65/2.15/2.53/1.90/0.23; 5.77/2.03/2.50/1.87/0.24 | ablation FID means 5.649, 5.770; MM 2.148, 2.030; ADE 2.527, 2.501; minADE 1.902, 1.869; ECS 0.235, 0.244 | OK |
| Table IV, last rows | 6.78/1.65/2.51/2.01/0.23; 7.03/1.74/2.50/1.97/0.23 | 6.781, 1.645 (→1.64, m2), 2.507, 2.012, 0.228; 7.034, 1.739, 2.496, 1.972, 0.227 | MM 1.65 vs 1.64; rest OK |
| Table IV caption | guidance 15, 15, 10 | selection JSONs | OK |
| Table V, all rows | as printed | `resplit_ego_controls.txt` scene_test: 2.434/1.649/0.308/0.191; 2.495/1.719/0.322/0.191; 2.522/1.738/0.331/0.194; 2.603/1.814/0.307/0.205; 3.039/2.065/0.056/0.239; 3.090/2.314/−0.003/0.312; 3.027/1.310/0.025/0.210; 3.110/1.307/0.015/0.214 | OK, 32 of 32 |
| VI-D 374 to 375 | ADE 3.09 vs 3.11; ECS 0.00; Brier 0.31; stationary ECS 0.06; 1 s ECS 0.31, val 0.30 vs 0.36 n.s.; 2.60 between 2.43 and 3.04; 1 s ADE cost meets rule on both partitions; stationary/foreign/zeroed meet rule | controls: 1 s ADE +0.169* test, +0.316* val; ECS +0.001 test, +0.060 [−0.010, 0.134] val; all three large controls * on ADE and ECS on both partitions | OK |
| VI-E 403 | FID 4.61, 4.07, 3.90; minADE 1.22, 1.36, 1.43; Full-seq FID 3.29, 2.80, 2.74; ADE 2.46, 2.55, 2.64; ECS 0.36--0.37, 0.30; w=5 ADE 0.03 and minADE 0.21 lower | breakdown "guidance on validation": pooled minADE 1.223/1.358/1.431, ECS 0.365/0.374/0.363, ADE 2.237/2.249/2.270 (diff 0.033, 0.208); egoped ADE 2.455/2.550/2.639, ECS 0.296/0.297/0.298; FIDs: only 3.90 and 2.74 confirmed by the action plan | FIDs 4.61, 4.07, 3.29, 2.80 unverified (m1); rest OK |
| VI-F 408 | ADE 0.575, 0.632, 0.750, 0.794; 0.22 m closer, interval excludes zero every seed; minADE order prior, Full-seq, EgoPed, PedGen (0.286 to 0.423); PedGen 0.65 m/s vs 1.26 m/s | short_horizon: pooled 0.561/0.589/0.576 = 0.5753; egoped 0.610/0.697/0.590 = 0.6323; uncond 0.750; pedgen 0.794; per-seed CIs [0.162, 0.299], [0.143, 0.260], [0.155, 0.275]; minADE 0.286 / 0.341 / 0.370 / 0.423; GT speed median 1.26; PedGen 0.65 from action plan | OK |
| VI-F 408 | "Our own trained prior is closer than PedGen as well" | uncond vs PedGen ADE +0.043 [−0.021, +0.103] | NOT supported (M1) |
| VII-b 431 | 6 tokens vs 201 | 4 + 1 + 1; 4 + 1 + 196 | OK |
| Limitations 443, Table V | 9.8 s | 196 / 20 fps | OK |
| Fig. 1 caption | 0.2--0.6 m and 16.4--17.7 m over 8 seeds | action plan teaser section | OK |
| Fig. 3 | 15 points, Spearman labels | matches Table II runs | OK |

## 5. Colon and semicolon rewrites

70 instances in prose (running text, abstract, captions, list items, a heading). Excluded as LaTeX syntax: `\label{...}`, `\ref{...}`, `\bstctlcite`, `_{1:L}`, "STADT:up" inside the compiled-out acknowledgment, and the "[Mohan: ...]" TODO. Each OLD string was checked by script to occur exactly once in main.tex; apply them in any order, except that M1 supersedes C61, M2 supersedes C42, and M6 should be applied after C32. Net change over all 70 is +123 characters, which is about one line of two-column text spread over the paper; the shortest ones (C6, C21, C23, C62, C68) each save part of a line.

Formatting note, separate from the 70: ieeeconf.cls prints a colon after every `\paragraph` heading (`\@IEEEsectpunct`, see M9). Fix M9(a) removes those eleven class-generated colons as well. If instead M9(b) is chosen, the class colons remain as formatting.

**C1** (main.tex line 70, Fig. 1 caption, +5 chars)

```
OLD: Panels (a) and (b) share model and noise; only the input trajectory differs.
NEW: Panels (a) and (b) share model and noise and differ only in the input trajectory.
```

**C2** (main.tex line 70, Fig. 1 caption, +4 chars)

```
OLD: Titles give the distance travelled across 8 noise seeds; time stamps pair vehicle and pedestrian; the grid spacing is 5\,m.
NEW: Titles give the distance travelled across 8 noise seeds, time stamps pair vehicle and pedestrian, and the grid spacing is 5\,m.
```

**C3** (main.tex line 70, Fig. 1 caption, 0 chars)

```
OLD: Both validation conditions were chosen for illustration; Section~\ref{sec:main} measures the coupling over the whole test set.
NEW: Both validation conditions were chosen for illustration. Section~\ref{sec:main} measures the coupling over the whole test set.
```

**C4** (main.tex line 80, Abstract (also update the plain-text abstract in SUBMISSION.md), 0 chars)

```
OLD: The Fréchet distance is blind to this coupling: an ego-blind prior matches the best conditional model, and the two design choices it favours both lower per-condition accuracy.
NEW: The Fréchet distance is blind to this coupling. An ego-blind prior matches the best conditional model, and the two design choices it favours both lower per-condition accuracy.
```

**C5** (main.tex line 94, Intro, +2 chars)

```
OLD: That behaviour is coupled to the vehicle: its speed, path and proximity influence whether a pedestrian crosses, waits, accelerates or turns
NEW: That behaviour is coupled to the vehicle, whose speed, path and proximity influence whether a pedestrian crosses, waits, accelerates or turns
```

**C6** (main.tex line 98, Intro, -8 chars)

```
OLD: These generators are blind to the vehicle in exactly the way a simulator cannot afford: a person crossing a street is generated the same whether a car approaches at 50\,km/h or stands still.
NEW: These generators are blind to the vehicle in the way a simulator cannot afford. A person crossing a street is generated the same whether a car approaches at 50\,km/h or stands still.
```

**C7** (main.tex line 107, Intro, list lead-in, +6 chars)

```
OLD: Our contributions are:
NEW: We make three contributions.
```

**C8** (main.tex line 110, Contribution 2, 0 chars)

```
OLD: With them we show that FID cannot see the conditioning: an ego-blind prior matches the best conditional FID, and both design choices FID prefers lose on per-condition accuracy.
NEW: With them we show that FID cannot see the conditioning. An ego-blind prior matches the best conditional FID, and both design choices FID prefers lose on per-condition accuracy.
```

**C9** (main.tex line 121, Sec. II-A (see also finding on the misplaced Rudenko citation), +4 chars)

```
OLD: and diffusion~\cite{gu2022mid,mao2023led}; MTR~\cite{shi2022mtr} does the same for vehicles~\cite{rudenko2020survey}.
NEW: and diffusion~\cite{gu2022mid,mao2023led}, and MTR~\cite{shi2022mtr} does the same for vehicles~\cite{rudenko2020survey}.
```

**C10** (main.tex line 125, Sec. II-B, 0 chars)

```
OLD: Diffusion~\cite{ho2020ddpm,song2021ddim} with classifier-free guidance~\cite{ho2022cfg} now dominates: MDM~\cite{tevet2023mdm}
NEW: Diffusion~\cite{ho2020ddpm,song2021ddim} with classifier-free guidance~\cite{ho2022cfg} now dominates. MDM~\cite{tevet2023mdm}
```

**C11** (main.tex line 125, Sec. II-B, +4 chars)

```
OLD: constrains where the generated person's own body should be; scene-~\cite{hassan2021samp,jiang2024trumans} and partner-conditioned~\cite{liang2024intergen} models condition on static geometry or a second human.
NEW: constrains where the generated person's own body should be, and scene-~\cite{hassan2021samp,jiang2024trumans} and partner-conditioned~\cite{liang2024intergen} models condition on static geometry or a second human.
```

**C12** (main.tex line 125, Sec. II-B, 0 chars)

```
OLD: Our condition differs in kind: it is the trajectory of a different, independently moving agent that only shapes a distribution over plausible reactions.
NEW: Our condition differs in kind. It is the trajectory of a different, independently moving agent that only shapes a distribution over plausible reactions.
```

**C13** (main.tex line 129, Sec. II-C, 0 chars)

```
OLD: \method generates the body directly from the vehicle trajectory; Section~\ref{sec:pedgen} compares it with PedGen.
NEW: \method generates the body directly from the vehicle trajectory. Section~\ref{sec:pedgen} compares it with PedGen.
```

**C14** (main.tex line 133, Sec. II-D, +8 chars)

```
OLD: Their known failure modes concern how faithfully they measure the \emph{marginal}: estimator bias~\cite{chong2020unbiasedfid}, preprocessing~\cite{parmar2022cleanfid} and the extractor's own class structure~\cite{kynkaanniemi2023imagenetfid}, with kernel~\cite{binkowski2018mmdgan} and precision--recall~\cite{kynkaanniemi2019precisionrecall} alternatives.
NEW: Their known failure modes concern how faithfully they measure the \emph{marginal}, such as estimator bias~\cite{chong2020unbiasedfid}, preprocessing~\cite{parmar2022cleanfid} and the extractor's own class structure~\cite{kynkaanniemi2023imagenetfid}, with kernel~\cite{binkowski2018mmdgan} and precision--recall~\cite{kynkaanniemi2019precisionrecall} alternatives.
```

**C15** (main.tex line 133, Sec. II-D, 0 chars)

```
OLD: The problem we document is orthogonal: a score can measure the marginal well and say nothing about whether the condition was used.
NEW: The problem we document is orthogonal. A score can measure the marginal well and say nothing about whether the condition was used.
```

**C16** (main.tex line 133, Sec. II-D, 0 chars)

```
OLD: Estimated body motion is also known to be physically imperfect~\cite{rempe2020contact,yuan2023physdiff}; we measure how imperfect our labels are, because it limits what a pose-level score can certify.
NEW: Estimated body motion is also known to be physically imperfect~\cite{rempe2020contact,yuan2023physdiff}. We measure how imperfect our labels are, because it limits what a pose-level score can certify.
```

**C17** (main.tex line 143, Fig. 2 caption, 0 chars)

```
OLD: The ego encoder is \emph{frozen} and produces the single pooled token $\bar{\mathbf{c}}$; for classifier-free guidance the ego input is zeroed for 10\% of training samples.
NEW: The ego encoder is \emph{frozen} and produces the single pooled token $\bar{\mathbf{c}}$. For classifier-free guidance the ego input is zeroed for 10\% of training samples.
```

**C18** (main.tex line 160, Sec. III-C, 0 chars)

```
OLD: A lightweight transformer encodes the ego trajectory: a linear projection from 2 to 256 dimensions, sinusoidal positional encoding and four encoder layers with four heads give
NEW: A lightweight transformer encodes the ego trajectory. A linear projection from 2 to 256 dimensions, sinusoidal positional encoding and four encoder layers with four heads give
```

**C19** (main.tex line 164, Sec. III-C, leads into Eq. (1), +12 chars)

```
OLD: Before diffusion training, the encoder is trained to regress the paired motion's latent: with $\bar{\mathbf{z}}$ the mean-pooled VAE posterior mean,
NEW: Before diffusion training, the encoder is trained to regress the paired motion's latent. With $\bar{\mathbf{z}}$ the mean-pooled VAE posterior mean, the loss is
```

**C20** (main.tex line 173, Sec. III-D, leads into Eq. (2), 0 chars)

```
OLD: over the concatenation of the noisy latent tokens, the timestep embedding and the condition; the first $L$ output tokens predict the noise:
NEW: over the concatenation of the noisy latent tokens, the timestep embedding and the condition. The first $L$ output tokens predict the noise,
```

**C21** (main.tex line 190, Sec. IV-A, -13 chars)

```
OLD: so we built one from two complementary sources: the public driving datasets nuScenes~\cite{caesar2020nuscenes} and Waymo~\cite{sun2020waymo}, which supply scale and behavioural diversity, and a small set of deliberately \emph{staged} high-interaction scenes recorded with our research vehicle, which target close-range encounters that are rare in public data.
NEW: so we built one from two complementary sources. The public driving datasets nuScenes~\cite{caesar2020nuscenes} and Waymo~\cite{sun2020waymo} supply scale and behavioural diversity, and a small set of deliberately \emph{staged} high-interaction scenes recorded with our research vehicle targets close-range encounters that are rare in public data.
```

**C22** (main.tex line 203, Sec. IV-B, +3 chars)

```
OLD: All partitions are split by scene: a scene, every pedestrian in it and every mirrored copy stay on one side (Table~\ref{tab:dataset}).
NEW: All partitions are split by scene, so a scene, every pedestrian in it and every mirrored copy stay on one side (Table~\ref{tab:dataset}).
```

**C23** (main.tex line 204, Sec. IV-B, -14 chars)

```
OLD: This matters: in a file-level split of the same data, the mirrored twin of 83\% of the held-out sequences was in the training set.
NEW: In a file-level split of the same data, the mirrored twin of 83\% of the held-out sequences was in the training set.
```

**C24** (main.tex line 210, Table I caption, +4 chars)

```
OLD: \caption{Dataset composition: paired ego trajectory ($196\times2$) and pedestrian body motion ($196\times263$) per sequence, counted with mirrored copies. Partitions share no scene.}
NEW: \caption{Dataset composition, counted with mirrored copies. Each sequence pairs an ego trajectory ($196\times2$) with pedestrian body motion ($196\times263$). Partitions share no scene.}
```

**C25** (main.tex line 236, Sec. IV-C, 0 chars)

```
OLD: First, \textbf{body shape was never estimated}: every sequence shares one skeleton
NEW: First, \textbf{body shape was never estimated}. Every sequence shares one skeleton
```

**C26** (main.tex line 237, Sec. IV-C, 0 chars)

```
OLD: Second, \textbf{the feet do not plant}: during locomotion one foot should be stationary
NEW: Second, \textbf{the feet do not plant}. During locomotion one foot should be stationary
```

**C27** (main.tex line 238, Sec. IV-C, 0 chars)

```
OLD: Third, \textbf{articulation is damped}: against HumanML3D's mocap statistics, root translation varies more
NEW: Third, \textbf{articulation is damped}. Against HumanML3D's mocap statistics, root translation varies more
```

**C28** (main.tex line 238, Sec. IV-C, 0 chars)

```
OLD: ($0.44\times$, $0.26\times$, $0.37\times$); part of that gap is content, since our data is people walking near roads.
NEW: ($0.44\times$, $0.26\times$, $0.37\times$). Part of that gap is content, since our data is people walking near roads.
```

**C29** (main.tex line 242, Sec. IV-C, 0 chars)

```
OLD: Root trajectories are well tracked; the body is a lightly articulated canonical template sliding along them.
NEW: Root trajectories are well tracked. The body is a lightly articulated canonical template sliding along them.
```

**C30** (main.tex line 250, Sec. V-a, 0 chars)

```
OLD: with its pretrained motion encoders as feature extractor: FID compares generated and real feature distributions,
NEW: with its pretrained motion encoders as feature extractor. FID compares generated and real feature distributions,
```

**C31** (main.tex line 250, Sec. V-a, 0 chars)

```
OLD: We do not report R-Precision: it retrieves with each model's own ego encoder,
NEW: We do not report R-Precision. It retrieves with each model's own ego encoder,
```

**C32** (main.tex line 253, Sec. V-b (see also finding on expanding ADE/FDE), 0 chars)

```
OLD: compare it with the recorded pedestrian over the true sequence length: ADE and FDE average over the samples, minADE and minFDE take the best of five.
NEW: compare it with the recorded pedestrian over the true sequence length. ADE and FDE average over the samples, minADE and minFDE take the best of five.
```

**C33** (main.tex line 257, Sec. V-c, +9 chars)

```
OLD: minus its mean over conditions whose pedestrian walks: 0 for a model that ignores the vehicle, 1 for perfect coupling.
NEW: minus its mean over conditions whose pedestrian walks. It is 0 for a model that ignores the vehicle and 1 for perfect coupling.
```

**C34** (main.tex line 261, Sec. V-d, 0 chars)

```
OLD: Before the test partition was used, we fixed one rule: a model beats another only if
NEW: Before the test partition was used, we fixed one rule. A model beats another only if
```

**C35** (main.tex line 270, Table II caption, +16 chars)

```
OLD: \caption{Test partition (1{,}182 conditions, 91 scenes): mean over three seeds, seed range in brackets. FID and Diversity use the HumanML3D evaluator; ADE to Brier are per-condition, from $K{=}5$ samples. Ground-truth Diversity is 5.26--5.40. Bold: best of the four main models per column.}
NEW: \caption{Test partition (1{,}182 conditions, 91 scenes), mean over three seeds with the seed range in brackets. FID and Diversity use the HumanML3D evaluator, ADE to Brier are per-condition from $K{=}5$ samples. Ground-truth Diversity is 5.26--5.40. Bold marks the best of the four main models per column.}
```

**C36** (main.tex line 293, Table III caption, +6 chars)

```
OLD: \caption{Contrasts on the test partition: difference of seed means, positive when the first model is better,
NEW: \caption{Contrasts on the test partition as the difference of seed means, positive when the first model is better,
```

**C37** (main.tex line 319, Sec. VI-A, -1 chars)

```
OLD: The prior is better on minADE, as any sampler that spreads its five samples widely must be; minADE alone is no test of conditioning, and the prior's ECS of 0.02 shows it ignores the vehicle, as it was built to.
NEW: The prior is better on minADE, as any sampler that spreads its five samples widely must be, so minADE alone is no test of conditioning. The prior's ECS of 0.02 shows it ignores the vehicle, as it was built to.
```

**C38** (main.tex line 320, Sec. VI-A, 0 chars)

```
OLD: and holds on both large sources: on the nuScenes and Waymo test sequences \method reaches ECS 0.32 and 0.26, the prior $-$0.01 and 0.04.
NEW: and holds on both large sources. On the nuScenes and Waymo test sequences \method reaches ECS 0.32 and 0.26, the prior $-$0.01 and 0.04.
```

**C39** (main.tex line 320, Sec. VI-A, 0 chars)

```
OLD: It is specific to the vehicle in the scene: given another scene's vehicle, \method falls to the prior's level
NEW: It is specific to the vehicle in the scene. Given another scene's vehicle, \method falls to the prior's level
```

**C40** (main.tex line 325, Sec. VI-B, +3 chars)

```
OLD: the rank correlation of FID with mean ADE is $-0.74$ and with ECS $+0.48$: the lower a run's FID, the larger its trajectory error
NEW: the rank correlation of FID with mean ADE is $-0.74$ and with ECS $+0.48$, so the lower a run's FID, the larger its trajectory error
```

**C41** (main.tex line 325, Sec. VI-B, 0 chars)

```
OLD: Interaction-weighted data illustrates the mechanism: it lowers FID for the conditional model and the prior alike,
NEW: Interaction-weighted data illustrates the mechanism. It lowers FID for the conditional model and the prior alike,
```

**C42** (main.tex line 325, Sec. VI-B (see also finding on 'lowers every per-condition metric'), +4 chars)

```
OLD: for both encoders (Table~\ref{tab:contrasts}, rows 3--4); the IA models are no better than their own prior on mean ADE.
NEW: for both encoders (Table~\ref{tab:contrasts}, rows 3--4), and the IA models are no better than their own prior on mean ADE.
```

**C43** (main.tex line 330, Fig. 3 caption, +6 chars)

```
OLD: one point per training run (15 runs; colour: model, marker: data recipe).
NEW: one point per training run (15 runs, colour for model, marker for data recipe).
```

**C44** (main.tex line 330, Fig. 3 caption, +4 chars)

```
OLD: Lower FID goes with \emph{larger} trajectory error (a) and \emph{weaker} coupling (b); the ego-blind prior on IA data sits among the best FIDs.
NEW: Lower FID goes with \emph{larger} trajectory error (a) and \emph{weaker} coupling (b), and the ego-blind prior on IA data sits among the best FIDs.
```

**C45** (main.tex line 338, Sec. VI-C-a, 0 chars)

```
OLD: makes per-condition predictions worse on average: \fullseq trails \method by 0.26\,m in mean ADE
NEW: makes per-condition predictions worse on average. \fullseq trails \method by 0.26\,m in mean ADE
```

**C46** (main.tex line 338, Sec. VI-C-a, 0 chars)

```
OLD: The trade-off is spread: \fullseq's samples cover more outcomes
NEW: The trade-off is spread. \fullseq's samples cover more outcomes
```

**C47** (main.tex line 338, Sec. VI-C-a, +2 chars)

```
OLD: We attribute part of the gap to pretraining: the alignment objective supervises only the pooled token,
NEW: We attribute part of the gap to pretraining, whose alignment objective supervises only the pooled token,
```

**C48** (main.tex line 342, Sec. VI-C-b, 0 chars)

```
OLD: Freezing matters: the frozen encoder beats the trainable pretrained one
NEW: Freezing matters. The frozen encoder beats the trainable pretrained one
```

**C49** (main.tex line 342, Sec. VI-C-b, +6 chars)

```
OLD: and on validation on all but ADE; FID does not separate the two (5.67 vs.\ 5.65).
NEW: and on validation on all but ADE, while FID does not separate the two (5.67 vs.\ 5.65).
```

**C50** (main.tex line 343, Sec. VI-C-b, 0 chars)

```
OLD: Pretraining by itself does not help: once the encoder is trainable, pretrained and scratch initialisation give the same result
NEW: Pretraining by itself does not help. Once the encoder is trainable, pretrained and scratch initialisation give the same result
```

**C51** (main.tex line 345, Sec. VI-C-b, 0 chars)

```
OLD: so we decided before the test pass to also report the last checkpoint; it shows the same picture with worse FID,
NEW: so we decided before the test pass to also report the last checkpoint. It shows the same picture with worse FID,
```

**C52** (main.tex line 346, Sec. VI-C-b, +8 chars)

```
OLD: The pooled encoders learn the coupling even when trainable; a single-seed \fullseq run with a trainable encoder did not: its alignment between encoder output and motion latent stays at chance
NEW: The pooled encoders learn the coupling even when trainable, whereas a single-seed \fullseq run with a trainable encoder did not. Its alignment between encoder output and motion latent stays at chance
```

**C53** (main.tex line 350, Table IV caption, +10 chars)

```
OLD: Upper rows: checkpoint and guidance selected on validation like every other model (guidance 15, 15, 10). Lower rows: the last checkpoint of the same runs at the same guidance, reported as decided before the test pass.
NEW: Upper rows use the checkpoint and guidance selected on validation like every other model (guidance 15, 15, 10). Lower rows use the last checkpoint of the same runs at the same guidance, reported as decided before the test pass.
```

**C54** (main.tex line 374, Sec. VI-D, 0 chars)

```
OLD: and its coupling vanishes (ECS 0.00); its Brier score (0.31) is far worse than the prior's,
NEW: and its coupling vanishes (ECS 0.00). Its Brier score (0.31) is far worse than the prior's,
```

**C55** (main.tex line 375, Sec. VI-D, +8 chars)

```
OLD: keeps the coupling (ECS 0.31; 0.30 against 0.36 on validation, not a difference under our rule)
NEW: keeps the coupling (ECS 0.31 on test, 0.30 against 0.36 on validation, not a difference under our rule)
```

**C56** (main.tex line 375, Sec. VI-D, 0 chars)

```
OLD: and its later trajectory refines the path: the ADE cost of the 1\,s horizon meets our rule on both partitions.
NEW: and its later trajectory refines the path. The ADE cost of the 1\,s horizon meets our rule on both partitions.
```

**C57** (main.tex line 403, Sec. VI-E, +3 chars)

```
OLD: but raises its minADE from 1.22 to 1.36 and 1.43\,m; \fullseq behaves alike (FID 3.29, 2.80, 2.74; ADE 2.46, 2.55, 2.64\,m), and ECS stays flat
NEW: but raises its minADE from 1.22 to 1.36 and 1.43\,m. \fullseq behaves alike (FID 3.29, 2.80, 2.74 and ADE 2.46, 2.55, 2.64\,m), and ECS stays flat
```

**C58** (main.tex line 403, Sec. VI-E, 0 chars)

```
OLD: therefore picks the least accurate of the three settings; at $w{=}5$, \method's validation ADE would be
NEW: therefore picks the least accurate of the three settings. At $w{=}5$, \method's validation ADE would be
```

**C59** (main.tex line 408, Sec. VI-F, 0 chars)

```
OLD: so we use it as an ego-blind reference on the one quantity both models share: the root path over its 2-second horizon,
NEW: so we use it as an ego-blind reference on the one quantity both models share, the root path over its 2-second horizon,
```

**C60** (main.tex line 408, Sec. VI-F, 0 chars)

```
OLD: and 0.794\,m for PedGen's context-free model; \method is 0.22\,m closer than PedGen,
NEW: and 0.794\,m for PedGen's context-free model. \method is 0.22\,m closer than PedGen,
```

**C61** (main.tex line 408, Sec. VI-F (superseded by the MAJOR finding on the prior-vs-PedGen claim if that fix is applied), 0 chars)

```
OLD: Our own trained prior is closer than PedGen as well, so part of the gap is the domain: PedGen's samples walk at a median 0.65\,m/s against 1.26\,m/s in our data.
NEW: Our own trained prior is closer than PedGen as well, so part of the gap is the domain. PedGen's samples walk at a median 0.65\,m/s against 1.26\,m/s in our data.
```

**C62** (main.tex line 428, Sec. VII-a, -15 chars)

```
OLD: We suggest the same minimum for conditional motion generation in general: report a separately trained unconditional prior on the same data recipe, and report at least one per-condition metric next to FID.
NEW: We suggest the same minimum for conditional motion generation in general, a separately trained unconditional prior on the same data recipe and at least one per-condition metric next to FID.
```

**C63** (main.tex line 431, Sec. VII-b, +8 chars)

```
OLD: may prefer \fullseq's coverage despite its higher mean error; one that must reproduce typical reactions to a given manoeuvre should use \method,
NEW: may prefer \fullseq's coverage despite its higher mean error, whereas one that must reproduce typical reactions to a given manoeuvre should use \method,
```

**C64** (main.tex line 433, Sec. VII-c heading, +1 chars)

```
OLD: \paragraph{Outlook: spill-over to robotics.}
NEW: \paragraph{Spill-over to robotics (outlook).}
```

**C65** (main.tex line 434, Sec. VII-c, 0 chars)

```
OLD: Nothing in the model is specific to cars: its input is the 2D trajectory of a moving agent in the pedestrian's frame.
NEW: Nothing in the model is specific to cars. Its input is the 2D trajectory of a moving agent in the pedestrian's frame.
```

**C66** (main.tex line 442, Limitations, +4 chars)

```
OLD: The labels are estimated: one canonical skeleton, feet that do not plant and damped articulation (Section~\ref{sec:posequality}).
NEW: The labels are estimates with one canonical skeleton, feet that do not plant and damped articulation (Section~\ref{sec:posequality}).
```

**C67** (main.tex line 443, Limitations, +4 chars)

```
OLD: Its stop decisions survive a 1\,s horizon (Section~\ref{sec:controls}); a model trained on past ego motion only is untested.
NEW: Its stop decisions survive a 1\,s horizon (Section~\ref{sec:controls}), but a model trained on past ego motion only is untested.
```

**C68** (main.tex line 445, Limitations, -1 chars)

```
OLD: ECS covers one decision, stopping; crossing is too rare in our windows to score.
NEW: ECS covers only stopping, because crossing is too rare in our windows to score.
```

**C69** (main.tex line 452, Conclusion (see also finding on 'weaken the coupling'), 0 chars)

```
OLD: The same experiments show that FID cannot certify such a model: it favours the design choices that weaken the coupling
NEW: The same experiments show that FID cannot certify such a model. It favours the design choices that weaken the coupling
```

**C70** (main.tex line 418, Fig. 4 caption, permanent part (co-author's figure, lowest priority), +21 chars)

```
OLD: \textbf{Left:} view from the ego vehicle. \textbf{Middle:} third-person view over time with the ego trajectory (red), the shown sample's root path (dark blue) and the recorded pedestrian (orange). \textbf{Right:} bird's-eye view of five samples for the same ego trajectory.
NEW: The left column shows the view from the ego vehicle, the middle column a third-person view over time with the ego trajectory (red), the shown sample's root path (dark blue) and the recorded pedestrian (orange), and the right column a bird's-eye view of five samples for the same ego trajectory.
```

## 6. Disguised colons (lower priority)

**Dashes.** None. Every `--` in the body is a compound (ego--pedestrian, precision--recall, encoder--decoder) or a numeric range, and the single hyphens in math are minus signs. No em-dash is used as a colon.

**Parentheses that gloss or restate.** Two candidates, both acceptable as they stand, with optional rewrites:

- Line 237: "moves at a median $0.97$ of root speed ($0$ = clean footfalls, $1$ = the body sliding as a rigid unit)". The parenthesis defines a scale rather than restating the clause, so it is a legitimate gloss. Optional, +9 characters:
  ```
  OLD: moves at a median $0.97$ of root speed ($0$ = clean footfalls, $1$ = the body sliding as a rigid unit), and
  NEW: moves at a median $0.97$ of root speed, where $0$ means clean footfalls and $1$ a body sliding as a rigid unit, and
  ```
- Line 431: "which is also cheaper (6 tokens per denoising step instead of 201)". A supporting figure, not a restatement. Leave.

**Comma appositions introduced by the rewrites.** C59 ("the one quantity both models share, the root path over its 2-second horizon") and C66 ("estimates with one canonical skeleton, feet that do not plant and damped articulation") use an appositive or a prepositional list in place of the colon. They read as ordinary English and do not move the problem into another mark.

**Lead-in phrases.** No "as follows", "the following", "namely", "i.e." or "e.g." introduces a list anywhere in the body (the `\ie` and `\eg` macros at lines 34 and 35 are defined but unused and can be deleted).

## Appendix: verification notes

- Build: `latexmk -pdf -interaction=nonstopmode -outdir=<scratch>/build main.tex` from `paper_ral/`, exit 0, 8 pages US letter, 0 overfull boxes, no undefined references or citations, two bibtex warnings ("missing publisher" for the two `@inbook` entries, see m10).
- Renders inspected: all 8 pages at 110 dpi, `teaser.pdf`, `fid_vs_conditioning.pdf`, `FINAL_setA_compressed.pdf` at 130 dpi, plus `pdftotext` of each figure for hidden strings.
- Professor's CSV: 37 data rows, 25 threads, parsed with `csv` and `field_size_limit`.
- Scripts kept in the scratchpad: `colon_rewrites.py` (uniqueness check and section 5 generator).
