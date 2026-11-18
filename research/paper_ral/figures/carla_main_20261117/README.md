# Main CARLA figure, 2026-11-17

This is Mohan's replacement for the earlier-model placeholder at `fig:qualitatives`.
It preserves the approved three-column layout, typefaces and colors. The old
A/B/C/v3 files are retained for history and should not be used in the submission.

## Preserve these changes during the next paper sync

The live paper uses this single hook in `main.tex`:

```tex
\input{figures/carla_main_20261117/figure_block.tex}
```

Before pushing a paper copy from the model repository, pull the latest Overleaf
commit, copy this entire directory into `research/paper_ral/figures/`, and retain
that hook in `research/paper_ral/main.tex`. The companion patch applies the same
changes to Erik's later paper revision without replacing his surrounding text.
Do not overwrite the live Overleaf tree from an older local copy or force-push.
The change note records the base revisions and the precise scope.

## Contents

- `figure.pdf` is the vector figure used by the paper, with original CARLA raster panels.
- `figure_block.tex` holds the caption and the existing figure label.
- `figure_source.tex` and `panels/` preserve the editable TikZ source and six images.
- `manifest.json`, `coordinate_audit.json`, `source_audit.json`, and `verification.json`
  record the conditions, chosen samples, transformations, source hashes and checks.

The include retains the previous placeholder's maximum height. The figure has
been checked both at full resolution and in the compiled paper.

## Scientific provenance

Source export `egoped_scene_test_2026-11-13_frame_fixed`, eight generated samples
per held-out condition, pooled model `rs1_pooled_s1234`, checkpoint epoch 1399,
CFG 15 and DDIM 50. No model training or new motion generation is part of this
revision. The frame correction updates visualization coordinates, not the model's
original conditioning input. All pedestrian joint arrays remain unchanged.

Rows are Waymo M441_65, nuScenes M005_1, and staged M0008_37. Their rendered root-path
medoids are samples 7, 2 and 7. Rows one and three are reflected jointly with all
vehicle and pedestrian paths. These source clips also appeared in set A. They were
chosen after reviewing all four sets because they provide a stop, a clear walk,
and an interpretable failure, with one row per source. Every camera panel was
rendered again and the two problematic placements were moved as complete scenes.

## Corrections

- The BEV is a metric orthographic ground-plane view with the elevated camera's
  heading and handedness. All samples, the recording, the vehicle and time dots
  use the same transform. No path is reflected independently.
- The waiting pedestrian has clear curb space. The staged crossing and its
  recorded path avoid the former bin and street-furniture conflict.
- Exposure selection omits clipped or overlapping bodies and vehicles. The BEV
  retains all eight complete pedestrian paths. Clipped vehicle approaches are
  intentional, so the person remains legible at paper size.
- Semantic actor masks are used directly. A redundant RGB-difference filter was
  removed because it erased genuine skin/clothing pixels similar to the background.
- Contact shadows are recovered from the rendered ground pixels adjoining each actor.
  The pose and silhouette remain the original CARLA render.
- Staged-source naming follows the anonymous manuscript. Header stopping counts
  were recomputed using the existing paper's 0.2 m/s for one second rule.

Raw articulation limitations remain visible. There is no MotionBricks refinement,
foot-locking, pose cleanup or independently relocated pedestrian in this figure.
CARLA road placement is a visualization of the measured relative geometry, not
a reconstruction of the original source map.

## Rotation correction and continuation, November 17

The installed converter reads XYZ Euler angles although its wrapper advertises
axis-angle. The figure pipeline now converts IK matrices explicitly and matches
the female avatar reference. All six panels were rendered again. Source joints,
root paths, placements, cameras and metric BEVs are unchanged. The rotation
regression and the existing source/geometry checks pass. The previous bundles are
retained with `_before_rotation_fix` names. PDF photo quality is 85 at 1920 pixels
to keep the paper file size manageable. The original lossless renders are retained.

Keep the second main.tex hook as well as the figure hook on the next sync.
`\input{figures/carla_main_20261117/retargeting_text.tex}` replaces Mohan's TODO
with the verified retargeting paragraph. `drafts/mohan_motionbricks_20261117/`
contains the two-paragraph plan, figure concept, citation and eight-page proposal
against Erik's newer `bd0bcdb` paper. The MotionBricks concept is not a result.
It is deliberately not included in the live main.tex. The layout proposal must be
reconciled with Erik's new experiments before it becomes the main manuscript.
