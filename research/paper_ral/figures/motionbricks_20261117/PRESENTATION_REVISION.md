# Revised presentation: motion is visible in the figure

This revision replaces the single-frame panels while keeping the approved
504 x 142 pt branching layout, typography, colors and arrow structure.
No accepted motion, model, target, contact correction or root constraint changed.
Existing full-length videos remain valid.

- Direct retarget and refinement show identical times (2.20, 2.43, 2.67 s) with
  identical crop rectangles. Left-toe lift uses the same 0–20 cm scale.
- The head panel uses new, actual CARLA close-ups: refined walk and edited pose
  at 3.00 s, then the edited motion at 5.00 s. The earlier label “Head turn” was
  too specific: the edit is primarily a tilt. It is now “Head pose”. Its trace
  is the geodesic orientation difference from the refined head, not yaw alone.
- Pause frames show walking at 2.40 s, holding at 3.60 s and restarting at 4.70 s.
  The full root-speed trace shades the exact 3–4 s hold.
- The wider-pass panel uses a new overhead camera at 2.33 s. Full source and
  edited root paths are projected through its actual camera matrix. The blue
  dashed line is the original path; purple is the lateral edit. The original
  and edited paths share endpoints. The offset trace peaks at 40 cm.

Every trace covers all 229 accepted samples over 7.6 s. Dots identify selected
times. Body crops follow the pedestrian and must not be interpreted as showing
world displacement; the overhead panel and path trace supply that information.

The left-toe trace is the CARLA left toe-joint height minus its 3.5 cm ground
proxy, in centimeters. It illustrates articulation, not a measured human-motion
quality score. Root speed uses numpy.gradient with actual timestamps. Route
offset is horizontal displacement from the source; this edit is purely lateral.
The 33.62-degree head difference includes all rotational components.

All RGB images come from CARLA replay of the saved accepted controls. No images
were generated or pose-edited. `temporal_data.json` saves frame IDs, source hash,
crop rectangles, camera projection, all traces and their definitions. Head and
overhead camera manifests are retained for corresponding supplementary views.

Reproduction: run prepare_temporal.py, make_head_renderer.py, the generated
head renderer for refined frame 90 and edited frames 90/150, make_path_renderer.py,
the generated path renderer for wider frame 70, then finish_temporal_assets.py.
Use the existing CARLA/wham environments on the 3090. compose_temporal.py uses
ReportLab/Pillow and can run locally after copying this folder's panels/JSON.
