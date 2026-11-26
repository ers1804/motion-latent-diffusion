"""Render one teaser scene in Blender: generated body motion next to the ego vehicle.

  blender -b -P render_teaser.py -- --npz <teaser npz> --row 137 --seed 0 --out a.png

Joints are drawn as generated, unsmoothed. Writes <out>.json with the image position of
each time stamp, so the figure script can label them as vector text.

With --map (research/src/teaser_map.py) the road layout of the recorded scene lies under the
motion: road surface, walkways, crosswalks, lane markings and stop signs from the dataset's
own map. --after adds the recorded pedestrian's path after the window as a dashed arrow.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    import bmesh
    import bpy
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Vector
except ImportError:  # outside Blender only the helpers without geometry work, which is what their test needs
    bpy = None

CHAINS = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
RADIUS = {0: .10, 1: .085, 2: .085, 3: .10, 4: .06, 5: .06, 6: .105, 7: .045, 8: .045, 9: .115, 10: .04, 11: .04,
          12: .05, 13: .07, 14: .07, 15: .105, 16: .05, 17: .05, 18: .04, 19: .04, 20: .03, 21: .03}
PED, PED_PALE, PED_OTHER = (.03, .13, .55), (.50, .63, .90), (.55, .68, .92)
CAR, CAR_PALE, ROAD = (.72, .03, .12), (.96, .68, .72), (.80, .81, .83)
GROUND, GRID_ON_MAP = (.83, .84, .85), (.66, .68, .71)  # with a map: paler ground, so that the road reads as road
MAP_LAYERS = [  # class in the map file, colour, height above the ground (draw order)
    ("road", (.50, .51, .55), .004), ("walkway", (.78, .71, .55), .007), ("crosswalk", (.97, .97, .97), .010),
    ("marking_white", (.97, .97, .97), .010), ("marking_yellow", (.93, .74, .22), .010)]
SIGN_RED, RECORDED = (.80, .05, .05), (.03, .03, .04)
MIN_STEP_PED, MIN_STEP_CAR = .6, 3.5  # closer time stamps would overlap: only the later one is drawn
STILL_PED, STILL_CAR = .6, .5  # a mark also stands for an earlier time stamp if the track stayed this close to it
# m from the point whose path the data holds to the middle of the drawn vehicle. The processed nuScenes scenes hold
# the pose of the roof lidar (x to the right, 1.84 m above the ground), which sits 0.94 m ahead of the rear axle and
# 0.45 m behind the middle of the vehicle. Waymo's vehicle frame sits lower and further back (its front camera is
# 1.5 m ahead of it), taken here as the rear axle. Any other source is drawn around its point, with a note.
BODY_AHEAD = {"nuscenes": .45, "waymo": 1.45}
FPS = 20


def to_blender(p):
    """Motion frame (x left, y up, z forward) to Blender (z up)."""
    p = np.asarray(p)
    return np.stack([p[..., 0], -p[..., 2], p[..., 1]], -1)


def material(name, rgb, rough=.55):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1)
    b.inputs["Roughness"].default_value = rough
    return m


def fade(pale, full, a):
    return tuple(p + (f - p) * a for p, f in zip(pale, full))


def thin(frames, track, min_step, still):
    """Keep the last time stamp and any earlier one far enough from the next kept one.

    Returns {kept frame: earliest frame it stands for}. A dropped time stamp widens the span of the kept one
    only if the track stayed within `still` of the kept position on every frame in between. A vehicle that
    waits from 0 to 4.8 s is drawn once and labelled with that span. One that creeps 3 m in that time is
    drawn once and keeps the later time stamp alone.
    """
    track = np.asarray(track)[:, :2]
    frames = sorted(set(frames))
    keep = {frames[-1]: frames[-1]}
    last = frames[-1]
    for f in reversed(frames[:-1]):
        away = np.linalg.norm(track[f:last + 1] - track[last], axis=1)
        if away[0] >= min_step:
            keep[f] = last = f
        elif away.max() <= still and keep[last] > f:
            keep[last] = f
    return dict(sorted(keep.items()))


def source_of(path):
    """'nuscenes/train/260_16.json' -> 'nuscenes'. '' if the path does not have the form <source>/<split>/<file>."""
    path = path if isinstance(path, (str, bytes)) else np.asarray(path).item()
    parts = Path(path.decode() if isinstance(path, bytes) else path).parts
    return parts[-3] if len(parts) >= 3 else ""


def vehicle_marks(frames, ego, body_ahead):
    """Where the vehicle is drawn: [(frame, earliest frame the mark stands for, heading, middle of the body)].

    ego: (T, >=2) path of the point the data holds. The body's middle lies `body_ahead` metres in front of it.
    """
    ego = np.asarray(ego)[:, :2]
    marks = []
    for f, since in thin(frames, ego, MIN_STEP_CAR, STILL_CAR).items():
        v = ego[min(f + 10, len(ego) - 1)] - ego[max(f - 10, 0)]
        if np.linalg.norm(v) < .3:  # stationary: the last heading it moved with, or the first one it sets off with
            moving = np.where(np.linalg.norm(np.diff(ego, axis=0), axis=1) > .02)[0]
            i = moving[moving <= f][-1] if (moving <= f).any() else (moving[0] if len(moving) else 0)
            v = ego[min(i + 1, len(ego) - 1)] - ego[i]
        heading = math.atan2(v[1], v[0])
        marks.append((f, since, heading, ego[f] + body_ahead * np.array([math.cos(heading), math.sin(heading)])))
    return marks


def add_mesh(name, bm, mat):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for poly in me.polygons:
        poly.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    me.materials.append(mat)
    return ob


def capsule(bm, a, b, r0, r1):
    a, b = Vector(a), Vector(b)
    d = b - a
    if d.length > 1e-5:
        q = d.to_track_quat("Z", "Y").to_matrix().to_4x4()
        m = q.copy()
        m.translation = (a + b) / 2
        bmesh.ops.create_cone(bm, cap_ends=True, segments=16, diameter1=r0, diameter2=r1, depth=d.length, matrix=m)
    for c, r in ((a, r0), (b, r1)):
        bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=10, diameter=r, matrix=_translate(c))


def _translate(c):
    from mathutils import Matrix
    return Matrix.Translation(c)


def body(name, joints, mat):
    bm = bmesh.new()
    for chain in CHAINS:
        for i, j in zip(chain[:-1], chain[1:]):
            capsule(bm, joints[i], joints[j], RADIUS[i], RADIUS[j])
    capsule(bm, joints[16], joints[17], .06, .06)  # shoulders
    capsule(bm, joints[1], joints[2], .09, .09)  # hips
    return add_mesh(name, bm, mat)


def ribbon(name, pts, width, mat, z, arrow=False):
    """Flat path on the ground; the last segment optionally widens into an arrow head."""
    pts = np.asarray(pts)[:, :2]
    tangent = np.gradient(pts, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-6)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], 1)
    bm = bmesh.new()
    rows = []
    for p, n in zip(pts, normal):
        rows.append([bm.verts.new((*(p + sign * n * width / 2), z)) for sign in (-1, 1)])
    for r0, r1 in zip(rows[:-1], rows[1:]):
        bm.faces.new((r0[0], r0[1], r1[1], r1[0]))
    if arrow:
        tip, n = pts[-1] + tangent[-1] * width * 4, normal[-1]
        bm.faces.new((bm.verts.new((*(pts[-1] - n * width * 1.8), z)), bm.verts.new((*(pts[-1] + n * width * 1.8), z)), bm.verts.new((*tip, z))))
    bm.normal_update()
    return add_mesh(name, bm, mat)


def ground(p, z):
    """(x, z) of the motion frame on the ground to Blender."""
    return (p[0], -p[1], z)


def triangle_mesh(name, tris, mat, z):
    """Flat triangles from the map file, all facing up."""
    bm = bmesh.new()
    for tri in tris:
        a, b, c = (np.array(ground(p, z)) for p in tri)
        if np.cross(b - a, c - a)[2] < 0:
            b, c = c, b
        bm.faces.new([bm.verts.new(v) for v in (a, b, c)])
    ob = add_mesh(name, bm, mat)
    for poly in ob.data.polygons:
        poly.use_smooth = False
    return ob


def stop_sign(name, pos, toward, red, pole):
    """Stylised stop sign, larger than life and turned to the camera so that it can be read at this scale."""
    base = Vector(ground(pos, 0))
    bm = bmesh.new()
    capsule(bm, base, base + Vector((0, 0, 2.3)), .05, .05)
    add_mesh(name + "_pole", bm, pole)
    bm = bmesh.new()
    facing = (Vector(toward) - (base + Vector((0, 0, 2.6))))
    m = facing.to_track_quat("Z", "Y").to_matrix().to_4x4()
    m.translation = base + Vector((0, 0, 2.6))
    bmesh.ops.create_cone(bm, cap_ends=True, segments=8, diameter1=.62, diameter2=.62, depth=.08, matrix=m @ _rot_z(math.pi / 8))
    ob = add_mesh(name, bm, red)
    for poly in ob.data.polygons:
        poly.use_smooth = False
    return (base + Vector((0, 0, 3.3))).to_tuple()


def dashed_arrow(name, pts, mat, z, width=.2, dash=.9, gap=.3):
    """Dashed path on the ground that ends in an arrow head."""
    pts = np.asarray(pts, dtype=float)
    step = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.concatenate([[True], step > 1e-4])
    pts = pts[keep]
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    head = 4 * width
    at = lambda d: np.array([np.interp(d, s, pts[:, k]) for k in range(2)])
    start = 0.0
    while start < s[-1] - head:
        end = min(start + dash, s[-1] - head)
        if end - start > .1:
            ribbon(f"{name}_{start:.2f}", [ground(at(d), 0) for d in np.linspace(start, end, 6)], width, mat, z)
        start += dash + gap
    tip, back = at(s[-1]), at(max(s[-1] - head, 0))
    direction = (tip - back) / max(np.linalg.norm(tip - back), 1e-6)
    normal = np.array([-direction[1], direction[0]])
    bm = bmesh.new()
    bm.faces.new([bm.verts.new(ground(q, z)) for q in (back - normal * width * 1.8, back + normal * width * 1.8, tip)])
    bm.normal_update()
    add_mesh(name + "_head", bm, mat)
    return [*ground(tip, 0)[:2], .3]


def car(name, mid, heading, mat, tyre):
    """Stylised vehicle around `mid`, for scale only; the data holds a path, not a shape.

    With the middle from vehicle_marks, the gap between the front of the body and a pedestrian is the recorded
    one to within the difference between this body (4.5 m long) and the real vehicle, about 0.2 m for nuScenes.
    """
    rot = _rot_z(heading)
    base = Vector((mid[0], mid[1], 0))

    def boxes(parts, material_, bevel):
        bm = bmesh.new()
        for size, c in parts:
            m = rot.copy()
            m.translation = base + rot.to_3x3() @ Vector(c)
            bmesh.ops.create_cube(bm, size=1, matrix=m @ _scale(size))
        ob = add_mesh(name, bm, material_)
        mod = ob.modifiers.new("b", "BEVEL")
        mod.width, mod.segments = bevel, 4
        for poly in ob.data.polygons:
            poly.use_smooth = False

    boxes([((4.5, 1.82, .58), (0, 0, .62))], mat, .16)
    boxes([((2.35, 1.6, .5), (-.2, 0, 1.12))], tyre, .12)
    boxes([((2.35, 1.6, .06), (-.2, 0, 1.39))], mat, .025)
    bm = bmesh.new()
    for x in (-1.45, 1.45):
        for y in (-.92, .92):
            m = rot.copy()
            m.translation = base + rot.to_3x3() @ Vector((x, y, .33))
            bmesh.ops.create_cone(bm, cap_ends=True, segments=24, diameter1=.33, diameter2=.33, depth=.22, matrix=m @ _rot_x())
    add_mesh(name + "_w", bm, tyre)


def _rot_z(a):
    from mathutils import Matrix
    return Matrix.Rotation(a, 4, "Z")


def _rot_x():
    from mathutils import Matrix
    return Matrix.Rotation(math.pi / 2, 4, "X")


def _scale(s):
    from mathutils import Matrix
    return Matrix.Diagonal((*s, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--row", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--times", type=float, nargs="*", default=[0, 2.4, 4.8, 7.2, 9.75])
    ap.add_argument("--car-times", type=float, nargs="*", help="time stamps of the vehicle, if not those of the pedestrian "
                    "(a vehicle that drives over the place where the pedestrian stood earlier would be drawn on top of them)")
    ap.add_argument("--cam", type=float, nargs=3, default=[-23, -10.5, 11], help="camera position")
    ap.add_argument("--target", type=float, nargs=3, default=[2, -10.5, .4])
    ap.add_argument("--lens", type=float, default=36)
    ap.add_argument("--res", type=int, nargs=2, default=[1600, 800])
    ap.add_argument("--samples", type=int, default=96)
    ap.add_argument("--map", help="road layout from research/src/teaser_map.py")
    ap.add_argument("--after", action="store_true", help="draw the recorded pedestrian's path after the window (needs --map)")
    args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])

    d = np.load(args.npz)
    L = int(d[f"len_{args.row}"])
    joints = to_blender(d[f"joints_{args.row}"][:, :L])  # (seeds, T, 22, 3)
    floor = joints[args.seed, :, :, 2].min()
    joints[..., 2] -= floor
    ego = d[f"ego_{args.row}"][:L]
    ego = np.stack([ego[:, 0], -ego[:, 1], np.full(L, .04)], -1)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = args.res
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"

    world = bpy.data.worlds.new("w")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
    world.node_tree.nodes["Background"].inputs[1].default_value = .62
    scene.world = world
    sun = bpy.data.lights.new("s", "SUN")
    sun.energy, sun.angle = 2.0, math.radians(14)
    so = bpy.data.objects.new("s", sun)
    so.rotation_euler = (math.radians(40), math.radians(15), math.radians(-35))
    bpy.context.collection.objects.link(so)

    # ground with a 5 m grid. The model sees no map; with --map the recorded scene's road layout is drawn for the reader
    layout = json.load(open(args.map))[str(args.row)] if args.map else None
    bpy.ops.mesh.primitive_plane_add(size=400)
    bpy.context.object.data.materials.append(material("g", GROUND if layout else ROAD, rough=.9))
    grid = material("grid", GRID_ON_MAP if layout else (.60, .62, .65))
    if layout:
        for cls, rgb, z in MAP_LAYERS:
            if layout["triangles"].get(cls):
                triangle_mesh("map_" + cls, layout["triangles"][cls], material("map_" + cls, rgb, rough=.9), z)
    bm = bmesh.new()
    r = .02 if layout else .03
    for v in np.arange(-100, 101, 5.0):
        capsule(bm, (v, -100, .005), (v, 100, .005), r, r)
        capsule(bm, (-100, v, .005), (100, v, .005), r, r)
    add_mesh("grid", bm, grid)

    # other seeds, chosen path, ego path
    other = material("o", PED_OTHER)
    for s in range(joints.shape[0]):
        if s != args.seed:
            ribbon(f"o{s}", joints[s, ::2, 0], .07, other, .02)
    ribbon("root", joints[args.seed, ::2, 0], .13, material("r", PED), .03)
    ribbon("ego", ego[::2], .2, material("e", CAR), .03)

    # time-lapse: pedestrian and vehicle share one fade from early to late
    dark = material("w", (.08, .08, .09))
    labels = []
    frames = [min(int(t * FPS), L - 1) for t in args.times]
    span = max(frames[-1], 1)
    for f, since in thin(frames, joints[args.seed, :, 0], MIN_STEP_PED, STILL_PED).items():
        body(f"p{f}", joints[args.seed, f], material(f"pm{f}", fade(PED_PALE, PED, f / span), rough=.5))
        labels.append(dict(kind="ped", t=f / FPS, since=since / FPS, at=joints[args.seed, f, 15].tolist()))
    car_frames = [min(int(t * FPS), L - 1) for t in args.car_times] if args.car_times else frames
    source = source_of(d[f"path_{args.row}"]) if f"path_{args.row}" in d else ""
    if source not in BODY_AHEAD:
        print(f"source '{source}' of row {args.row}: not known which point of the vehicle its path follows, body drawn around it")
    for f, since, heading, mid in vehicle_marks(car_frames, ego, BODY_AHEAD.get(source, 0.0)):
        car(f"c{f}", mid, heading, material(f"cm{f}", fade(CAR_PALE, CAR, f / span), rough=.35), dark)
        labels.append(dict(kind="car", t=f / FPS, since=since / FPS, at=[float(mid[0]), float(mid[1]), 1.6]))

    if layout:
        if layout["signs"]:
            red, pole = material("sign", SIGN_RED, rough=.4), material("pole", (.35, .36, .38))
            tops = [stop_sign(f"sign{i}", p, args.cam, red, pole) for i, p in enumerate(layout["signs"])]
            stop = ego[-1][:2]  # label the sign nearest to where the vehicle ends (both in Blender coordinates)
            nearest = int(np.argmin([np.linalg.norm(np.array(ground(p, 0))[:2] - stop) for p in layout["signs"]]))
            labels.append(dict(kind="sign", at=list(tops[nearest])))
        if args.after:
            path = np.array(layout["recorded_after"]).reshape(-1, 2)  # 10 Hz, row k is k / 10 s after its start
            moved = np.linalg.norm(path - path[0], axis=1) if len(path) else np.zeros(1)
            if moved.max() > 1.0:
                first = int(np.argmax(moved > .3))  # frames after the window until the recorded pedestrian sets off
                tip = dashed_arrow("after", path[first:], material("after", RECORDED), .035)
                labels.append(dict(kind="after", at=tip, t=layout["recorded_after_start_s"] + first / 10))
            else:
                print(f"--after: the recorded pedestrian moves {moved.max():.1f} m after the window, no arrow drawn")

    cam = bpy.data.cameras.new("c")
    cam.lens = args.lens
    co = bpy.data.objects.new("c", cam)
    bpy.context.collection.objects.link(co)
    co.location = args.cam
    co.rotation_euler = (Vector(args.target) - Vector(args.cam)).to_track_quat("-Z", "Y").to_euler()
    scene.camera = co
    bpy.context.view_layer.update()

    for lab in labels:
        uv = world_to_camera_view(scene, co, Vector(lab["at"]))
        lab["uv"] = [uv.x, uv.y, uv.z]
    with open(args.out + ".json", "w") as f:
        json.dump(labels, f, indent=1)
    scene.render.filepath = args.out
    scene.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
