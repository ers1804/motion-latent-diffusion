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

