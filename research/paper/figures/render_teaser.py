"""Render one teaser scene in Blender: generated body motion next to the ego vehicle.

  blender -b -P render_teaser.py -- --npz <teaser npz> --row 137 --seed 0 --out a.png

Joints are drawn as generated, unsmoothed. Writes <out>.json with the image position of
each time stamp, so the figure script can label them as vector text.
"""
import argparse
import json
import math
import sys

import bmesh
import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

CHAINS = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
RADIUS = {0: .10, 1: .085, 2: .085, 3: .10, 4: .06, 5: .06, 6: .105, 7: .045, 8: .045, 9: .115, 10: .04, 11: .04,
          12: .05, 13: .07, 14: .07, 15: .105, 16: .05, 17: .05, 18: .04, 19: .04, 20: .03, 21: .03}
PED, PED_PALE, PED_OTHER = (.03, .13, .55), (.50, .63, .90), (.55, .68, .92)
CAR, CAR_PALE, ROAD = (.72, .03, .12), (.96, .68, .72), (.80, .81, .83)
MIN_STEP_PED, MIN_STEP_CAR = .6, 3.5  # closer time stamps would overlap: only the later one is drawn
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


def thin(frames, track, min_step):
    """Keep the last time stamp and any earlier one far enough from the next kept one."""
    keep = [frames[-1]]
    for f in reversed(frames[:-1]):
        if np.linalg.norm(track[f][:2] - track[keep[-1]][:2]) >= min_step:
            keep.append(f)
    return sorted(keep)


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


def car(name, pos, heading, mat, tyre):
    """Stylised vehicle, for scale only; the data holds its path, not its shape."""
    rot = _rot_z(heading)
    base = Vector((pos[0], pos[1], 0))

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
    ap.add_argument("--cam", type=float, nargs=3, default=[-23, -10.5, 11], help="camera position")
    ap.add_argument("--target", type=float, nargs=3, default=[2, -10.5, .4])
    ap.add_argument("--lens", type=float, default=36)
    ap.add_argument("--res", type=int, nargs=2, default=[1600, 800])
    ap.add_argument("--samples", type=int, default=96)
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

    # ground with a 5 m grid: the data has no map, so no road is drawn
    bpy.ops.mesh.primitive_plane_add(size=400)
    bpy.context.object.data.materials.append(material("g", ROAD, rough=.9))
    grid = material("grid", (.60, .62, .65))
    bm = bmesh.new()
    for v in np.arange(-100, 101, 5.0):
        capsule(bm, (v, -100, .005), (v, 100, .005), .03, .03)
        capsule(bm, (-100, v, .005), (100, v, .005), .03, .03)
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
    for f in thin(frames, joints[args.seed, :, 0], MIN_STEP_PED):
        body(f"p{f}", joints[args.seed, f], material(f"pm{f}", fade(PED_PALE, PED, f / span), rough=.5))
        labels.append(dict(kind="ped", t=f / FPS, at=joints[args.seed, f, 15].tolist()))
    for f in thin(frames, ego, MIN_STEP_CAR):
        back, ahead = max(f - 10, 0), min(f + 10, L - 1)
        v = ego[ahead] - ego[back]
        if np.linalg.norm(v[:2]) < .3:  # stationary: keep the last heading it moved with
            moving = np.where(np.linalg.norm(np.diff(ego[:, :2], axis=0), axis=1) > .02)[0]
            i = moving[moving <= f][-1] if (moving <= f).any() else (moving[0] if len(moving) else 0)
            v = ego[i + 1] - ego[i]
        car(f"c{f}", ego[f], math.atan2(v[1], v[0]), material(f"cm{f}", fade(CAR_PALE, CAR, f / span), rough=.35), dark)
        labels.append(dict(kind="car", t=f / FPS, at=[*ego[f][:2].tolist(), 1.6]))

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
