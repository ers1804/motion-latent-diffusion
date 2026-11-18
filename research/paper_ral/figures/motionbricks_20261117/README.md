## Presentation revision, November 17, late evening

The latest figure replaces the single stills with matched steps, paired head-pose
close-ups, a walking/holding/restarting sequence, and projected road paths.
All traces come from the same accepted controls. No motions changed. Head pose
is the accurate label: this edit primarily tilts the head rather than changing yaw.
See `temporal/README.md` and `temporal/temporal_data.json` for definitions and views.
The earlier single-frame version remains in Git history and the earlier Mac outputs.
Regenerate the latest PDF with compose_figure.py (delegates to temporal/).

# Completed MotionBricks CARLA realization — November 17, 2026

The figure shows one EgoPed motion proposal, its refined realization, and three
authored variants. It is a single-clip illustration, not a benchmark or a new
dataset. There was no training. The main CARLA figure remains unrefined EgoPed.

## Accepted motion and checks

- Source: corrected November 13 export, `08_correct_walk_nuscenes_M005_1`, sample 2.
  The input is an EgoPed generation, not recorded human motion or pose ground truth.
- Five clips: source, refined, head turn, pause/restart, wider pass. Each has 229
  samples at 30 Hz, spanning timestamps 0.0–7.6 s. The vehicle trajectory is shared.
- Refinement and head-turn clips retain the source root path and timing through
  explicit root projection. Native MotionBricks predictions are separately saved.
  Refinement's native, retimed path has mean/max deviation 0.030/0.152 m before projection.
- The pause has an exactly stationary root from 3.0–4.0 s, with 0.5 s speed ramps.
  Its below-0.2 m/s interval is 2.933–4.067 s (1.133 s). The speed ramps explain
  why this interval is longer than the one-second exact hold.
- The route edit reaches a 0.40 m lateral offset. Closest vehicle-center distance
  increases from 2.185 to 2.584 m. Minimum skeleton-to-car-footprint clearance is
  0.814 m across all five clips; nearest surveyed static footprints exceed 4.3 m.
- Native CARLA contact IK anchors support feet, with smooth entry/exit weights.
  Full-weight planted-foot intervals have numerical-zero horizontal drift in the
  controls. This is an enforced constraint, not a learned quality score. Ground
  tests use ankle/toe proxies plus visual inspection, not a mesh collision oracle.
- All 1,145 motion frames were inspected in ordered contact sheets. We checked
  walking, the whole pause and restart, route transitions, feet, head orientation,
  camera framing and car passage. No visible foot penetration or car occlusion
  remains in the accepted detail views. Source skating remains intentionally visible.
- CARLA bone readback agrees with saved controls within 0.007 mm in every frame.
  Both RGB sensors match the requested simulator frame. The full body stays in
  both camera frustums. All encoded clips contain exactly 229 frames at 30 fps.
- Fastest angular changes are approximately 22 degrees/frame at toe joints, already
  present before contact IK; they were included in the full-frame visual review.

## Reused setup and corrections

The working 5090 installation is
`/home/ivlab/Documents/motionbricks_demo_2026-11-15/repo`, commit `54fc563`.
The existing `rare_scenarios/adapter.py` SMPL2CARLA bridge and
`fable_revision/calibrated_pose.py` SOMA adapter were checked and their rest-frame
conversion and joint mapping reused. The main figure already uses SMPL2CARLA.
The MotionBricks realization additionally uses native-rig contact IK. Leaf-head
orientation is retained explicitly; using child directions alone loses head yaw.
CARLA's fixed skeleton-root transform is inverted before applying native controls.

`model_provenance.json` pins the existing model and modified skeleton; neither was
changed. Four-pose moving context is used. Native output durations are retimed to
the prescribed timestamps. The stop, head turn and lateral offset are authored
controls, not new EgoPed predictions, inferred human intent or validated responses.

## Files and replay

- `generated.npz`, `contacts.npz`: MotionBricks outputs after prescribed retiming
  and root projection; `native.npz` retains predictions before projection.
- `targets.npz`, `generation.json`: exact target poses, schedules and intervals.
- `carla_before_contact.npz`, `carla_final.npz`, `contact_audit.npz`: separate
  native controls before/after contact correction and their contact anchors.
- `renders/<mode>/render_manifest.json`: all timestamps, sensor frame IDs, camera
  matrices and returned foot positions. RGB PNGs remain on the 3090.
- `media/01_refinement_review.mp4`, `media/02_variations_review.mp4`: full-length
  synchronized comparisons. Individual scene/detail clips are also retained.
- `figure_panels/*.json`: exact figure frame IDs and crop rectangles. Source,
  refinement, head turn and wider pass use the same rectangle at 2.333 s. The
  pause uses 3.600 s. Crops preserve image geometry.
- `validation.json`, `motion_metrics.json`, `render_audit.json`,
  `geometry_check.json`, `media/video_validation.json`: reproducible diagnostics.

On the 3090, run `retarget_final.py` and `validate_motion.py` with the `wham` Python.
Run `render_final.py --mode <mode> --size 1280` with `carla16` Python, one renderer
at a time, in Town10HD_Opt. Capture holds each prescribed pose until bone readback
matches, then reads both sensors for that frame. This is kinematic replay of 30 Hz
motion timestamps, not a closed-loop or physically validated pedestrian simulation.
The renderer saves every frame, making later video edits independent of inference.

The earlier pilot's failed 0.70 s pause remains in `motionbricks_pilot_20261117`.
The main figure's three-scene supplementary video is still a separate follow-up.
