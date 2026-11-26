#!/usr/bin/env python3
"""Road layout under the teaser figure, taken from the datasets' own maps.

The model never sees a map. The teaser draws one so that a reader can tell why the recorded scene went the way
it did (a pedestrian waiting at a curb, a vehicle halting at a stop sign). No position is drawn by hand: road
surface, walkways, crosswalks, lane markings and stop signs come from the nuScenes map expansion and from the
map features of the Waymo segment, and they are placed by a rigid fit of the recorded vehicle and pedestrian
paths of the scene to the same paths in the teaser file. Line styles follow the maps where they record one.
Schematic are the zebra bars inside a mapped crosswalk, the dash rhythm, and nuScenes road dividers, which
carry no type and are drawn solid white.

Two stages, because the Waymo records need TensorFlow and the geometry needs shapely >= 2.1:

  <waymo env>/python research/src/teaser_map.py waymo-raw --npz research/data/teaser_rs1_pooled_scene_val.npz --out /tmp/waymo_raw.json
  <shapely env>/python research/src/teaser_map.py build --npz research/data/teaser_rs1_pooled_scene_val.npz --waymo-raw /tmp/waymo_raw.json

The second call writes research/data/teaser_map_<name of the npz>.json, which render_teaser.py reads with --map.
All coordinates in it are metres on the ground in the frame of the joints, as (x, z) of the motion frame.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

NAS = os.environ.get("EGOPED_NAS", "/home/erik/NAS/methods/diffusion_gen")
DRIVESTUDIO = os.environ.get("EGOPED_DRIVESTUDIO", f"{NAS}/data/drivestudio")
NUSCENES = os.environ.get("EGOPED_NUSCENES", "/home/erik/NAS/publicdatasets/nuscenes")
WAYMO = os.environ.get("EGOPED_WAYMO", "/home/erik/NAS/publicdatasets/waymo_perception_old_version/individual_files/training")
SCENE_HZ, MOTION_HZ = 10, 20  # frames of the processed scenes, frames of the motion
PAD = 45.0  # metres of map kept around the paths
MAX_RMS = 0.5  # a fit worse than this means the wrong scene or the wrong window
MIN_MARGIN, FAR = 1.5, 0.5  # the best window must fit this many times better than any window over FAR seconds away
MAX_ROAD_SHARE = 0.6  # of the clipped area; more means the road leaked through a gap in its edges
ANCHOR_S = 1.0  # seconds of the pedestrian's path that enter the fit, see fit_window
STRIPE, MARK = 0.6, 0.14  # crosswalk stripe and gap, width of a lane marking
DASH, GAP = 3.0, 4.5


# ---------------------------------------------------------------- placing the scene in the teaser frame
def rigid(a, b, reflect):
    """R, t with b ~ a @ R.T + t in 2D. det(R) is +1, or -1 with reflect."""
    ca, cb = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
    d = np.sign(np.linalg.det(vt.T @ u.T)) * (-1 if reflect else 1)
    r = vt.T @ np.diag([1, d]) @ u.T
    return r, cb - ca @ r.T


def resample(t, times, track):
    return np.stack([np.interp(t, times, track[:, k]) for k in range(track.shape[1])], 1)


def fit_window(ego_world, ped_world, ego_teaser, ped_teaser, step=0.025):
    """Where the teaser window starts in the scene and how the scene's world frame maps into the teaser frame.

    ego_world, ped_world: (frames, 2) at SCENE_HZ, NaN where the pedestrian is not tracked.
    ego_teaser, ped_teaser: (L, 2) at MOTION_HZ. Both handednesses are tried, because the motion frame has x to
    the left. Returns a dict with start_s, R, t, reflect, rms and the rms of the other handedness.

    The vehicle path is metric in the motion frame and enters whole. The pedestrian's path does not: the motion
    representation stretches the root path by the ratio of the canonical to the fitted leg length
    (uniform_skeleton, 4 to 6% in the scenes of the teaser) and turns it by the few degrees between the body's
    facing and the box heading. Only its first ANCHOR_S seconds enter, which fix the origin and the handedness.
    """
    times = np.arange(len(ego_world)) / SCENE_HZ
    length = len(ego_teaser)
    anchor = min(length, int(round(ANCHOR_S * MOTION_HZ)))
    target = np.concatenate([ego_teaser, ped_teaser[:anchor]])
    best, profile = {}, {False: [], True: []}
    for reflect in (False, True):
        for start in np.arange(0, times[-1] - (length - 1) / MOTION_HZ + 1e-9, step):
            t = start + np.arange(length) / MOTION_HZ
            ped = resample(t, times, ped_world)
            if np.isnan(ped).any():  # the pedestrian must be tracked over the whole window
                continue
            source = np.concatenate([resample(t, times, ego_world), ped[:anchor]])
            r, shift = rigid(source, target, reflect)
            rms = float(np.sqrt((np.linalg.norm(source @ r.T + shift - target, axis=1) ** 2).mean()))
            profile[reflect].append((float(start), rms))
            if reflect not in best or rms < best[reflect]["rms"]:
                best[reflect] = dict(start_s=float(start), R=r, t=shift, reflect=reflect, rms=rms)
    if not best:
        raise ValueError("the pedestrian is not tracked over any window of this length")
    chosen = min(best.values(), key=lambda b: b["rms"])
    others = [b["rms"] for k, b in best.items() if k != chosen["reflect"]]
    chosen["rms_other_handedness"] = others[0] if others else float("nan")
    elsewhere = [rms for start, rms in profile[chosen["reflect"]] if abs(start - chosen["start_s"]) > FAR]
    chosen["rms_elsewhere"] = min(elsewhere) if elsewhere else float("nan")
    return chosen


def ambiguous(fit):
    """Why a fit must not be trusted, or None. A straight path with a standing pedestrian fits anywhere, and mirrored."""
    floor = max(fit["rms"], 0.02)  # a perfect fit must not make every other one look bad
    if fit["rms_other_handedness"] < 2 * floor:
        return f"the mirrored placement fits as well ({fit['rms_other_handedness']:.2f} m against {fit['rms']:.2f} m)"
    if fit["rms_elsewhere"] < MIN_MARGIN * floor:
        return f"a window over {FAR} s away fits as well ({fit['rms_elsewhere']:.2f} m against {fit['rms']:.2f} m)"
    return None


def window_frames(start_s, length, ped_world):
    """Scene frames the window covers, and the pedestrian's tracked path from its end on.

    Returns first, last and the path from frame `last`, cut at the first frame without a track, so that row k of
    the path is always k / SCENE_HZ seconds after frame `last`.
    """
    first = int(np.floor(start_s * SCENE_HZ + 1e-9))
    last = int(np.ceil((start_s + (length - 1) / MOTION_HZ) * SCENE_HZ - 1e-9))
    tail = ped_world[last:]
    gaps = np.flatnonzero(np.isnan(tail).any(1))
    return first, last, tail[:gaps[0]] if len(gaps) else tail


def scene_tracks(dataset, scene, obj):
    """World paths of the vehicle and of one tracked object, from the processed scene (10 Hz)."""
    root = Path(DRIVESTUDIO) / dataset / "drivestudio_output" / scene
    n = len(list((root / "ego_pose").glob("*.txt")))
    ego = np.array([np.loadtxt(root / "ego_pose" / f"{i:03d}.txt")[:2, 3] for i in range(n)])
    ann = json.load(open(root / "instances" / "instances_info.json"))[obj]["frame_annotations"]
    ped = np.full((n, 2), np.nan)
    ped[ann["frame_idx"]] = np.array(ann["obj_to_world"])[:, :2, 3]
    return ego, ped


def parse_path(path):
    """'nuscenes/train/146_11.json' -> ('nuscenes', '146', '11'). Mirrored copies (M...) have no scene of their own."""
    dataset, _, name = Path(path).parts[-3:]
    scene, obj = Path(name).stem.split("_")
    if scene.startswith("M"):
        raise ValueError(f"{path} is a mirrored copy, its road layout would have to be mirrored as well")
    return dataset, scene, obj


# ---------------------------------------------------------------- raw map layers, in world coordinates
def nuscenes_layers(scene, around, radius, frames=None):
    """Layers of the nuScenes map expansion near a world point, read from the JSON without the devkit.

    frames: length of the processed scene. It must match the scene's key frames (2 Hz, four frames in between),
    which guards the assumption that the processed scenes are numbered in the order of scene.json.
    """
    scenes = json.load(open(f"{NUSCENES}/v1.0-trainval/scene.json"))
    logs = {log["token"]: log for log in json.load(open(f"{NUSCENES}/v1.0-trainval/log.json"))}
    record = scenes[int(scene)]  # the processed scenes are numbered in the order of scene.json
    location = logs[record["log_token"]]["location"]
    if frames is not None and frames != (record["nbr_samples"] - 1) * 5 + 1:
        raise SystemExit(f"processed scene {scene} has {frames} frames, {record['name']} would give {(record['nbr_samples'] - 1) * 5 + 1}: not the same scene")
    m = json.load(open(f"{NUSCENES}/maps/expansion/{location}.json"))
    node = {n["token"]: (n["x"], n["y"]) for n in m["node"]}
    polygon = {p["token"]: p for p in m["polygon"]}
    line = {ln["token"]: ln for ln in m["line"]}

    def near(points):  # by bounding box: a long feature can cross the area with no vertex inside it
        p, c = np.asarray(points), np.asarray(around)
        return bool((p.min(0) < c + radius).all() and (p.max(0) > c - radius).all())

    def shape(token):
        p = polygon[token]
        return dict(exterior=[node[t] for t in p["exterior_node_tokens"]], holes=[[node[t] for t in h["node_tokens"]] for h in p["holes"]])

    layers = {}
    for name in ("drivable_area", "walkway", "ped_crossing"):
        shapes = [shape(t) for rec in m[name] for t in rec.get("polygon_tokens", [rec.get("polygon_token")])]
        layers[name] = [s for s in shapes if near(s["exterior"])]
    layers["road_divider"] = [dict(line=pts, kind="SOLID_WHITE") for pts in ([node[t] for t in line[rec["line_token"]]["node_tokens"]] for rec in m["road_divider"]) if near(pts)]
    layers["lane_divider"] = []
    for rec in m["lane_divider"]:
        pts = [node[t] for t in line[rec["line_token"]]["node_tokens"]]
        kinds = [s["segment_type"] for s in rec["lane_divider_segments"] if s["segment_type"] != "NIL"]
        if near(pts) and kinds:  # one style per divider: its most frequent segment type, e.g. DOUBLE_DASHED_WHITE
            layers["lane_divider"].append(dict(line=pts, kind=max(sorted(set(kinds)), key=kinds.count)))
    return dict(source=f"nuScenes {record['name']} ({location}), map expansion {m['version']}", description=record["description"], layers=layers)


def waymo_raw(scene, out):
    """Map features of one Waymo segment, written as JSON. Needs tensorflow and waymo_open_dataset."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import tensorflow as tf
    from waymo_open_dataset import dataset_pb2
    from waymo_open_dataset.protos import map_pb2

    files = sorted(f for f in os.listdir(WAYMO) if f.endswith(".tfrecord"))  # the processed scenes follow this order
    frame = dataset_pb2.Frame()
    for raw in tf.data.TFRecordDataset(os.path.join(WAYMO, files[int(scene)]), compression_type="").take(1):
        frame.ParseFromString(raw.numpy())
    if not frame.map_features:
        raise SystemExit(f"{files[int(scene)]} carries no map features (they exist from release 1.4.2 on)")
    points = lambda polyline: [[p.x, p.y] for p in polyline]
    layers = dict(road_edge=[], road_line=[], lane=[], crosswalk=[], stop_sign=[])
    for feature in frame.map_features:
        kind = feature.WhichOneof("feature_data")
        if kind == "road_edge":
            layers[kind].append(dict(line=points(feature.road_edge.polyline)))
        elif kind == "road_line":
            layers[kind].append(dict(line=points(feature.road_line.polyline), kind=map_pb2.RoadLine.RoadLineType.Name(feature.road_line.type)))
        elif kind == "lane":
            layers[kind].append(dict(line=points(feature.lane.polyline)))
        elif kind == "crosswalk":
            layers[kind].append(dict(exterior=points(feature.crosswalk.polygon)))
        elif kind == "stop_sign":
            layers[kind].append(dict(position=[feature.stop_sign.position.x, feature.stop_sign.position.y]))
    pose = np.array(frame.pose.transform).reshape(4, 4)[:2, 3].tolist()
    json.dump(dict(source=f"Waymo Open Dataset segment {frame.context.name} ({frame.context.stats.location})", scene=str(scene), first_pose=pose, layers=layers), open(out, "w"))
    print("wrote", out, {k: len(v) for k, v in layers.items()})


# ---------------------------------------------------------------- geometry, in the teaser frame
def stripes(poly, stripe=STRIPE):
    """Zebra bars of a crosswalk: across its short side, repeated along its long side."""
    from shapely import affinity
    from shapely.geometry import box

    corners = np.array(poly.minimum_rotated_rectangle.exterior.coords)[:4]
    e0, e1 = corners[1] - corners[0], corners[2] - corners[1]
    long_edge, short_edge = (e0, e1) if np.linalg.norm(e0) >= np.linalg.norm(e1) else (e1, e0)
    length, width = np.linalg.norm(long_edge), np.linalg.norm(short_edge)
    angle = np.degrees(np.arctan2(long_edge[1], long_edge[0]))
    centre = corners.mean(0)
    count = int((length + stripe) // (2 * stripe))  # bars and gaps of one width, a bar at either end
    left = -(count * stripe + (count - 1) * stripe) / 2
    bars = []
    for k in range(count):
        x0 = left + 2 * stripe * k
        bar = affinity.translate(affinity.rotate(box(x0, -width, x0 + stripe, width), angle, origin=(0, 0)), *centre)
        bars.append(bar.intersection(poly))
    return [b for b in bars if not b.is_empty]


def dashes(line, dash=DASH, gap=GAP):
    from shapely.ops import substring

    return [substring(line, s, min(s + dash, line.length)) for s in np.arange(0, line.length, dash + gap)]


def marking(line, kind, width=MARK):
    """Polygons of one lane marking. kind holds SOLID or BROKEN/DASHED, DOUBLE, WHITE or YELLOW."""
    parts = dashes(line) if ("BROKEN" in kind or "DASHED" in kind) else [line]
    if "DOUBLE" in kind:
        parts = [p.offset_curve(side * width) for p in parts for side in (-1, 1)]
    return [p.buffer(width / 2, cap_style=2) for p in parts if p.length > 0]


def road_from_edges(edges, lanes, clip):
    """Road surface where the map gives only its edges: the cells of the edge network that hold a lane centre."""
    from shapely.ops import polygonize, unary_union

    cells = [c for c in polygonize(unary_union(list(edges) + [clip.exterior])) if c.representative_point().within(clip)]  # edges reach past the clip
    centres = unary_union(list(lanes)).intersection(clip)  # the lines, not their vertices: a lane may have only two
    road = unary_union([c for c in cells if c.buffer(-0.3).intersects(centres)])
    if road.area > MAX_ROAD_SHARE * clip.area:
        raise SystemExit(f"the road covers {road.area / clip.area:.0%} of the map section: its edges have a gap and the surface leaked")
    return road


def triangles(geometry):
    """Any polygonal geometry as a list of triangles [[x, z] * 3], holes respected."""
    import shapely

    if not hasattr(shapely, "constrained_delaunay_triangles"):
        raise SystemExit("shapely >= 2.1 is needed for the triangulation")
    out = []
    polys = [g for g in getattr(geometry, "geoms", [geometry]) if g.geom_type in ("Polygon", "MultiPolygon") and not g.is_empty]
    for poly in polys:
        for part in getattr(poly, "geoms", [poly]):
            for tri in shapely.constrained_delaunay_triangles(part).geoms:
                if tri.area > 1e-6:
                    out.append(np.round(np.array(tri.exterior.coords)[:3], 3).tolist())
    return out


def build_row(raw, to_teaser, clip_bounds, dataset):
    """Drawable layers of one panel: triangles per class, stop signs, all in the teaser frame."""
    from shapely.geometry import LineString, Point, Polygon, box
    from shapely.ops import unary_union

    clip = box(*clip_bounds)
    layers = raw["layers"]
    tf_poly = lambda g: Polygon(to_teaser(g["exterior"]), [to_teaser(h) for h in g.get("holes", [])]).buffer(0)
    tf_line = lambda g: LineString(to_teaser(g["line"]))
    out, signs = {}, []
    if dataset == "nuscenes":
        road = unary_union([tf_poly(g) for g in layers["drivable_area"]]).intersection(clip)
        out["walkway"] = unary_union([tf_poly(g) for g in layers["walkway"]]).intersection(clip).difference(road)
        crossings = [tf_poly(g) for g in layers["ped_crossing"]]
        marks = [(tf_line(g), g["kind"]) for g in layers["road_divider"] + layers["lane_divider"]]
    else:
        road = road_from_edges([tf_line(g) for g in layers["road_edge"]], [tf_line(g) for g in layers["lane"]], clip)
        crossings = [tf_poly(g) for g in layers["crosswalk"]]
        marks = [(tf_line(g), g["kind"]) for g in layers["road_line"]]
        positions = [g["position"] for g in layers["stop_sign"]]
        signs = [np.round(p, 3).tolist() for p in to_teaser(positions) if clip.contains(Point(p))] if positions else []
    crossings = [c.intersection(clip) for c in crossings if c.intersects(clip)]
    out["road"] = road
    out["crosswalk"] = unary_union([bar for c in crossings for bar in stripes(c)]).intersection(road) if crossings else Polygon()
    zebra = unary_union(crossings) if crossings else Polygon()
    for colour in ("WHITE", "YELLOW"):
        polys = [p for line, kind in marks if colour in kind for p in marking(line, kind)]
        out[f"marking_{colour.lower()}"] = (unary_union(polys).intersection(road).difference(zebra) if polys else Polygon())
    return dict(triangles={k: triangles(v) for k, v in out.items()}, signs=signs, road=road)


def build(args):
    from shapely.geometry import Point

    d = np.load(args.npz)
    waymo = json.load(open(args.waymo_raw)) if args.waymo_raw else None
    result = {}
    for row in [int(r) for r in d["rows"]]:
        dataset, scene, obj = parse_path(str(d[f"path_{row}"]))
        length = int(d[f"len_{row}"])
        ego_teaser = d[f"ego_{row}"][:length].astype(float)
        ped_teaser = d[f"gt_{row}"][:length, 0][:, [0, 2]].astype(float)
        ego_world, ped_world = scene_tracks(dataset, scene, obj)
        fit = fit_window(ego_world, ped_world, ego_teaser, ped_teaser)
        if fit["rms"] > MAX_RMS:
            raise SystemExit(f"row {row}: the scene does not fit the teaser window (rms {fit['rms']:.2f} m)")
        if ambiguous(fit):
            raise SystemExit(f"row {row}: the placement is not unique, {ambiguous(fit)}")
        to_teaser = lambda pts, f=fit: np.asarray(pts, dtype=float)[:, :2] @ f["R"].T + f["t"]
        both = np.concatenate([ego_teaser, ped_teaser])
        bounds = (*(both.min(0) - PAD), *(both.max(0) + PAD))
        if dataset == "nuscenes":
            raw = nuscenes_layers(scene, ego_world.mean(0), 2 * PAD + np.ptp(ego_world, 0).max(), frames=len(ego_world))
        elif dataset == "waymo":
            if waymo is None or waymo["scene"] != str(int(scene)):
                raise SystemExit(f"row {row} needs --waymo-raw for segment {scene}, see the module docstring")
            if np.linalg.norm(np.array(waymo["first_pose"]) - ego_world[0]) > 0.5:
                raise SystemExit("the Waymo record is not the processed scene (first poses differ)")
            raw = waymo
        else:
            raise SystemExit(f"no map source for {dataset}")
        drawn = build_row(raw, to_teaser, bounds, dataset)
        on_road = float(np.mean([drawn["road"].contains(Point(p)) for p in ego_teaser[::4]]))
        if on_road < 0.9:
            raise SystemExit(f"row {row}: only {on_road:.0%} of the vehicle path lies on the mapped road, the placement is wrong")
        ped_on_road = float(np.mean([drawn["road"].contains(Point(p)) for p in ped_teaser[::4]]))
        first, last, tail = window_frames(fit["start_s"], length, ped_world)
        after = to_teaser(tail) if len(tail) else np.zeros((0, 2))
        result[str(row)] = dict(
            source=raw["source"], path=str(d[f"path_{row}"]), window_start_s=fit["start_s"], scene_frames=[first, last],
            fit_rms_m=round(fit["rms"], 3), fit_rms_other_handedness_m=round(fit["rms_other_handedness"], 3), fit_rms_elsewhere_m=round(fit["rms_elsewhere"], 3),
            vehicle_path_on_road=round(on_road, 3), pedestrian_path_on_road=round(ped_on_road, 3),
            triangles=drawn["triangles"], signs=drawn["signs"],
            recorded_after=np.round(after, 3).tolist(), recorded_after_start_s=round(last / SCENE_HZ - fit["start_s"], 2),
        )
        print(f"row {row}: {raw['source']}; window starts at {fit['start_s']:.2f} s, fit rms {fit['rms']:.2f} m "
              f"(other handedness {fit['rms_other_handedness']:.2f} m, best window elsewhere {fit['rms_elsewhere']:.2f} m), "
              f"{on_road:.0%} of the vehicle path and {ped_on_road:.0%} of the recorded pedestrian path on the road, "
              + ", ".join(f"{k} {len(v)}" for k, v in drawn["triangles"].items()) + f", signs {len(drawn['signs'])}, recorded after the window {len(after)} frames")
    out = Path(args.out) if args.out else Path(__file__).resolve().parent.parent / "data" / f"teaser_map_{Path(args.npz).stem.replace('teaser_', '')}.json"
    json.dump(result, open(out, "w"))
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="stage", required=True)
    w = sub.add_parser("waymo-raw", help="map features of the Waymo segment of the teaser (TensorFlow environment)")
    w.add_argument("--npz", required=True)
    w.add_argument("--out", required=True)
    b = sub.add_parser("build", help="fit, clip and triangulate (shapely >= 2.1)")
    b.add_argument("--npz", required=True)
    b.add_argument("--waymo-raw")
    b.add_argument("--out")
    args = ap.parse_args()
    if args.stage == "waymo-raw":
        d = np.load(args.npz)
        scenes = [parse_path(str(d[f"path_{int(r)}"])) for r in d["rows"]]
        segments = [scene for dataset, scene, _ in scenes if dataset == "waymo"]
        if len(segments) != 1:
            raise SystemExit(f"expected one Waymo row in the teaser file, found {len(segments)}")
        waymo_raw(str(int(segments[0])), args.out)
    else:
        build(args)


if __name__ == "__main__":
    main()
