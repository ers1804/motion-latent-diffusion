#!/usr/bin/env python3
"""External test set from the labelled Munich drives: root paths only, no pose fits.

The models are frozen and nothing is selected on this set. The protocol below was fixed and committed before any
model output on it was looked at (research/review_action_plan_2026-11-13.md, "Munich external test").

1. Population. Every pedestrian (adult, child, construction worker, police officer) in the tables LABEL_VERSION that
   is not labelled sitting or lying on half or more of its keyframes.
2. Heading. The rotation in ego_pose.json is 90 degrees off the direction of travel and all global boxes inherit
   it, stored = t_ego + Rz(-90 deg)(p - t_ego). correct_heading undoes that about the vehicle pose of the keyframe.
3. Sequences. Box centres (2 Hz) and vehicle poses are resampled linearly to 20 fps. A track gives one window of at
   most 196 frames, centred on the frame of closest approach to the vehicle, which is the interaction crop of the
   loader without its jitter. Tracks under 40 frames are dropped.
4. Frame. The pedestrian's box at the window start: origin at its centre, +z along its heading, +x to its left.
   That is the frame the models saw in training for every track of up to 196 frames. For longer tracks they saw
   the vehicle in the frame of the track's first pose (the stale frame of the paper), which this set does not copy.
   The existing dataset builds its vehicle input the same way, from the labelled box and not from the pose fit
   (data_pipeline step 1, ego_traj_in_ped). check_against_dataset reproduces dataset files from the boxes
   (vehicle path to a median 3 to 5 cm on 40 nuScenes and 40 Waymo files).
   Variant "label" (primary) takes the labelled box heading. In these labels it lies within 30 degrees of the walking
   direction for 83% of walkers, against 100% (nuScenes) and 89% (Waymo) of the walkers in the files checked, so
   variant "motion" (sensitivity) takes the direction of the first MOTION_SECONDS of the window instead whenever
   the pedestrian covers MOTION_MIN there. Labels, windows and subsets are the same in both.
5. Recorded stop. The label "pedestrian.standing" on STOP_KEYFRAMES consecutive keyframes inside the window (1 s).
   The box centres jitter by about 0.1 m per keyframe, so the paper's speed rule on the box path is secondary.
6. Interactive subset, from the window. The pedestrian is within R_PATH of a position the vehicle reaches up to
   PET_MAX seconds later (in front, vehicle moving or waiting), or of one it left up to PET_MAX seconds earlier
   (behind), or comes within CLOSE of the path the vehicle drives (close pass). The vehicle must drive at least
   MIN_EGO_TRAVEL in the window. See kind_of.

Files: sequences_<variant>.npz (paths, kept out of git) and manifest.json (labels, subsets, hashes of the paths).
The loader needs one JSON per sequence with 263-dimensional features. materialize writes them with only the root
channels filled, which is all that the root metrics read (recover_root_rot_pos).

Usage: python research/src/munich_eval.py --build
       python research/src/munich_eval.py --materialize <dir>      # both variants; EGOPED_MUNICH=<dir> afterwards
       python research/src/munich_eval.py --check-dataset 40       # box construction against existing files
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from intention_labels import is_stopping  # noqa: E402

LABELS = os.environ.get("MUNICH_LABELS", "/home/erik/NAS2/car_data/dataset_annotations_ayadata")
OUT = HERE.parent / "data" / "munich_v1"
STATS = HERE.parent / "data" / "scene_split_v1"  # the statistics the frozen models were trained with
STAT_FILES = ("Mean.npy", "Std.npy", "Ego_Mean.npy", "Ego_Std.npy")
LABEL_VERSION = "v2.2-mini"
PEDESTRIAN = ("human.pedestrian.adult", "human.pedestrian.child", "human.pedestrian.construction_worker",
              "human.pedestrian.police_officer")
FPS, MAX_LEN, MIN_LEN = 20, 196, 40
SITTING_SHARE = 0.5   # share of keyframes labelled sitting or lying from which a track is left out
STOP_KEYFRAMES = 3    # consecutive keyframes 0.5 s apart, i.e. standing for 1 s
R_PATH = 1.5          # m, within this distance of a position of the vehicle = in its path
PET_MAX = 5.0         # s between the pedestrian and the vehicle at the same spot
MOVING = 1.0          # m/s, above this the vehicle counts as moving when the pedestrian is in its path
CLOSE = 2.5           # m, clearance of a close pass
MIN_EGO_TRAVEL = 3.0  # m, a vehicle that drives less than this has no path to interact with
MOTION_SECONDS, MOTION_MIN = 2.0, 1.5  # variant "motion": heading from the first 2 s if they cover 1.5 m
VARIANTS = ("label", "motion")
ROOT_HEIGHT = 0.9     # m, constant, not read by any root metric
NJOINTS, NFEATS = 22, 263


def yaw_of(q):
    """Yaw of a nuScenes quaternion (w, x, y, z)."""
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def correct_heading(xy, yaw, pivot):
    """Stored global box positions and yaws -> true ones, turned by +90 degrees about the vehicle position."""
    d = np.asarray(xy, dtype=float) - pivot
    return pivot + np.stack([-d[:, 1], d[:, 0]], axis=1), np.asarray(yaw, dtype=float) + np.pi / 2


def to_ped_frame(xy, origin, yaw):
    """World xy -> (left, forward) of a pedestrian standing at origin and heading along yaw: the dataset's x and z."""
    d = np.asarray(xy, dtype=float) - origin
    left, forward = np.array([-np.sin(yaw), np.cos(yaw)]), np.array([np.cos(yaw), np.sin(yaw)])
    return np.stack([d @ left, d @ forward], axis=1)


def window_of(ped, ego):
    """(start, length) of the evaluated window: the whole track, or 196 frames centred on the closest approach."""
    n = len(ped)
    if n <= MAX_LEN:
        return 0, n
    centre = int(np.argmin(np.linalg.norm(ped - ego, axis=1)))
    return max(0, min(centre - MAX_LEN // 2, n - MAX_LEN)), MAX_LEN


def standing_stop(frames, standing, inside):
    """True if the pedestrian is labelled standing on STOP_KEYFRAMES consecutive keyframes of the window."""
    run = best = 0
    prev = None
    for f, s, ok in zip(frames, standing, inside):
        run = (run + 1 if run and f == prev + 1 else 1) if (s and ok) else 0
        best, prev = max(best, run), f
    return best >= STOP_KEYFRAMES


def _smooth(v, k=FPS):
    if len(v) < 2:
        return v
    return np.convolve(np.r_[np.repeat(v[0], k // 2), v, np.repeat(v[-1], k - k // 2 - 1)], np.ones(k) / k, mode="valid")


def measures(ped, ego):
    """Geometry of one window, ped and ego (n, 2) at 20 fps in one metric frame."""
    n = len(ped)
    d = np.linalg.norm(ped - ego, axis=1)
    dm = np.linalg.norm(ped[:, None] - ego[None], axis=2)  # [pedestrian time, vehicle time]
    step = np.linalg.norm(np.diff(ego, axis=0), axis=1) * FPS
    ego_speed = _smooth(np.r_[step, step[-1]])  # one value per frame, the last step repeated
    out = dict(dmin=float(d.min()), clearance=float(dm[np.triu_indices(n)].min()),
               ego_travel=float(np.linalg.norm(np.diff(ego, axis=0), axis=1).sum()),
               ped_travel=float(np.linalg.norm(np.diff(ped, axis=0), axis=1).sum()),
               pet_front=None, pet_behind=None, ego_speed_front=None, ego_slows_front=0.0)
    tp, te = np.where(dm < R_PATH)
    front = te > tp + 2
    if front.any():
        i = int(np.argmin((te - tp)[front]))
        a, b = int(tp[front][i]), int(te[front][i])
        before = ego_speed[max(0, a - 4 * FPS):a + 1]
        out.update(pet_front=float((b - a) / FPS), ego_speed_front=float(ego_speed[a]),
                   ego_slows_front=float(max(0.0, before.max() - ego_speed[max(0, a - 4 * FPS):b + 1].min())))
    behind = tp > te + 2
    if behind.any():
        out["pet_behind"] = float((tp - te)[behind].min() / FPS)
    return out


def kind_of(m):
    """The kind of path interaction of a window, or "none". The kinds are exclusive, in this order.

    front_moving: in the vehicle's path, the vehicle arrives within PET_MAX and moves while the pedestrian is there.
    front_waiting: the same with the vehicle standing or creeping while the pedestrian is there.
    behind: reaches a spot the vehicle left at most PET_MAX earlier.
    close_pass: within CLOSE of where the vehicle is or later drives, without one of the above. This includes a
    pedestrian who was in the path more than PET_MAX before the vehicle came.
    A vehicle that drives less than MIN_EGO_TRAVEL in the window has no path, so the window is "none".
    """
    if m["ego_travel"] < MIN_EGO_TRAVEL:
        return "none"
    front, behind = m["pet_front"], m["pet_behind"]
    if front is not None and front <= PET_MAX:
        return "front_moving" if m["ego_speed_front"] > MOVING else "front_waiting"
    if behind is not None and behind <= PET_MAX:
        return "behind"
    return "close_pass" if m["clearance"] < CLOSE else "none"


def encode_root(path):
    """(n, 2) root path (x, z) from the origin -> (n, 263) features whose root channels decode to that path.

    recover_root_rot_pos integrates channel 0 (turning) and channels 1:3 (velocity in the root frame). With no
    turning the root frame is the frame of the path, so the velocities are plain differences.
    """
    feats = np.zeros((len(path), NFEATS), dtype=np.float32)
    v = np.diff(np.asarray(path, dtype=np.float64), axis=0)
    feats[:-1, 1:3] = v
    feats[-1, 1:3] = v[-1]
    feats[:, 3] = ROOT_HEIGHT
    return feats


def content_hash(names, ped, ego, length):
    h = hashlib.sha256("\n".join(names).encode())
    for a in (ped, ego, length):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def scene_dirs(labels=LABELS):
    """All scene folders in a fixed order. The index is the scene id, so it stays when more scenes get labels."""
    root = Path(labels)
    return sorted(d for z in sorted(root.iterdir()) if z.is_dir() and z.name.startswith("z")
                  for d in sorted(z.iterdir()) if d.is_dir() and not d.name.startswith("@"))


def read_scene(scene_dir, version=LABEL_VERSION):
    """Pedestrian tracks of one scene at its keyframes, heading corrected, and the vehicle's positions."""
    base = Path(scene_dir) / version
    table = lambda name: json.loads((base / f"{name}.json").read_text())  # noqa: E731
    category = {c["token"]: c["name"] for c in table("category")}
    attribute = {a["token"]: a["name"] for a in table("attribute")}
    instance = {i["token"]: category[i["category_token"]] for i in table("instance")}
    samples = sorted(table("sample"), key=lambda s: s["timestamp"])
    order = {s["token"]: i for i, s in enumerate(samples)}
    time = {s["token"]: s["timestamp"] for s in samples}
    ego = {e["token"]: e for e in table("ego_pose")}
    key_ego = {d["sample_token"]: ego[d["ego_pose_token"]] for d in table("sample_data")
               if d["is_key_frame"] and "LIDAR_TOP" in d["filename"]}
    ego_all = sorted(ego.values(), key=lambda e: e["timestamp"])
    by_instance = {}
    raw = (base / "sample_annotation.json").read_bytes()
    for a in json.loads(raw):
        if instance[a["instance_token"]] in PEDESTRIAN:
            by_instance.setdefault(a["instance_token"], []).append(a)
    tracks = []
    for token, rows in sorted(by_instance.items()):
        rows.sort(key=lambda a: order[a["sample_token"]])
        pivot = np.array([key_ego[a["sample_token"]]["translation"][:2] for a in rows])
        xy, yaw = correct_heading(np.array([a["translation"][:2] for a in rows]),
                                  np.array([yaw_of(a["rotation"]) for a in rows]), pivot)
        names = [[attribute[x] for x in a["attribute_tokens"]] for a in rows]
        tracks.append(dict(
            instance=token, category=instance[token], frame=np.array([order[a["sample_token"]] for a in rows]),
            t=np.array([time[a["sample_token"]] for a in rows]), xy=xy, yaw=yaw,
            standing=np.array(["pedestrian.standing" in n for n in names]),
            sitting=np.array(["pedestrian.sitting_lying_down" in n for n in names]),
            visibility=float(np.median([int(a["visibility_token"]) if str(a["visibility_token"]).isdigit() else 0 for a in rows])),
            lidar=float(np.median([a.get("num_lidar_pts", 0) for a in rows]))))
    return dict(t0=samples[0]["timestamp"], ego_t=np.array([e["timestamp"] for e in ego_all]),
                ego_xy=np.array([e["translation"][:2] for e in ego_all]), tracks=tracks,
                annotations=len(json.loads(raw)), labels_sha256=hashlib.sha256(raw).hexdigest())


def resample(track, ego_t, ego_xy):
    """One track and the vehicle on a common 20 fps grid: times, pedestrian xy, pedestrian yaw, vehicle xy."""
    grid = np.arange(track["t"][0], track["t"][-1] + 1e-6, 1 / FPS)
    column = lambda t, x: np.stack([np.interp(grid, t, x[:, i]) for i in range(2)], axis=1)  # noqa: E731
    return grid, column(track["t"], track["xy"]), np.interp(grid, track["t"], np.unwrap(track["yaw"])), column(ego_t, ego_xy)


def motion_heading(ped, label_yaw):
    """Heading of variant "motion": the direction walked in the first MOTION_SECONDS, else the labelled one."""
    d = ped[min(len(ped) - 1, int(MOTION_SECONDS * FPS))] - ped[0]
    return float(np.arctan2(d[1], d[0])) if np.linalg.norm(d) >= MOTION_MIN else float(label_yaw)


def sequence_of(track, ego_t, ego_xy):
    """The evaluated window of one track in the pedestrian's frame (per variant) with its labels, or the reason
    it is left out."""
    if track["sitting"].mean() >= SITTING_SHARE:
        return "sitting"
    grid, ped, yaw, ego = resample(track, ego_t, ego_xy)
    if len(grid) < MIN_LEN:
        return "short"
    start, n = window_of(ped, ego)
    heading = dict(label=float(yaw[start]), motion=motion_heading(ped[start:start + n], yaw[start]))
    frames = {v: (to_ped_frame(ped[start:start + n], ped[start], h), to_ped_frame(ego[start:start + n], ped[start], h))
              for v, h in heading.items()}
    p, e = frames["label"]
    turned = np.degrees(abs((heading["motion"] - heading["label"] + np.pi) % (2 * np.pi) - np.pi))
    inside = (track["t"] >= grid[start] - 1e-3) & (track["t"] <= grid[start + n - 1] + 1e-3)
    m = measures(p, e)
    m.update(kind=kind_of(m), stop=bool(standing_stop(track["frame"], track["standing"], inside)),
             stop_speed=bool(is_stopping(p)), start=int(start), frames=int(n), t_start=float(grid[start]),
             standing_share=float(track["standing"][inside].mean()), gaps=int((np.diff(track["frame"]) > 1).sum()),
             visibility=track["visibility"], lidar=track["lidar"], category=track["category"], instance=track["instance"],
             motion_heading_offset=float(turned))
    return dict(frames=frames, row=m)


def build(labels=LABELS, out=OUT):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows, paths, left_out, scenes, unlabelled = [], [], Counter(), {}, []
    for index, scene_dir in enumerate(scene_dirs(labels)):
        if not (scene_dir / LABEL_VERSION / "sample_annotation.json").exists():
            unlabelled.append(scene_dir.name)
            continue
        scene = read_scene(scene_dir)
        if not scene["annotations"]:
            unlabelled.append(scene_dir.name)
            continue
        scenes[f"{index:03d}"] = dict(scene=scene_dir.name, zone=scene_dir.parent.name, labels_sha256=scene["labels_sha256"])
        for obj, track in enumerate(scene["tracks"]):
            seq = sequence_of(track, scene["ego_t"], scene["ego_xy"])
            if isinstance(seq, str):
                left_out[seq] += 1
                continue
            row = dict(name=f"U{index:03d}_{obj}", scene=scene_dir.name, zone=scene_dir.parent.name, **seq["row"])
            row["t_start"] -= scene["t0"]
            rows.append(row)
            paths.append(seq["frames"])
    n = len(rows)
    length = np.array([r["frames"] for r in rows], dtype=np.int64)
    names = [r["name"] for r in rows]
    hashes = {}
    for variant in VARIANTS:
        ped, ego = np.zeros((n, MAX_LEN, 2), np.float32), np.zeros((n, MAX_LEN, 2), np.float32)
        for i in range(n):
            ped[i, :length[i]], ego[i, :length[i]] = paths[i][variant]
        np.savez_compressed(out / f"sequences_{variant}.npz", names=np.array(names), ped=ped, ego=ego, length=length)
        hashes[variant] = content_hash(names, ped, ego, length)
    kinds = Counter(r["kind"] for r in rows)
    manifest = dict(
        protocol=dict(label_version=LABEL_VERSION, categories=PEDESTRIAN, fps=FPS, max_len=MAX_LEN, min_len=MIN_LEN,
                      sitting_share=SITTING_SHARE, stop_keyframes=STOP_KEYFRAMES, r_path=R_PATH, pet_max=PET_MAX,
                      moving=MOVING, close=CLOSE, min_ego_travel=MIN_EGO_TRAVEL, window="centred on the closest approach",
                      frame="pedestrian box at the window start, x left, z forward", variants=VARIANTS,
                      motion_seconds=MOTION_SECONDS, motion_min=MOTION_MIN),
        counts=dict(sequences=n, scenes=len({r["scene"] for r in rows}), labelled_scenes=len(scenes),
                    left_out=dict(left_out), stop=int(sum(r["stop"] for r in rows)),
                    stop_speed=int(sum(r["stop_speed"] for r in rows)),
                    interactive=n - kinds["none"], kinds=dict(kinds),
                    interactive_scenes=len({r["scene"] for r in rows if r["kind"] != "none"}),
                    heading_from_motion=int(sum(r["motion_heading_offset"] > 0 for r in rows)),
                    heading_moved_over_30_deg=int(sum(r["motion_heading_offset"] > 30 for r in rows))),
        content_sha256=hashes, unlabelled_scenes=unlabelled, scenes=scenes, sequences=rows)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest["counts"], indent=1))
    return manifest


def load_sequences(variant="label", folder=OUT):
    """(names, ped, ego, length, manifest) of one variant, refused if the paths are not those of the manifest."""
    d = np.load(Path(folder) / f"sequences_{variant}.npz")
    names = [str(x) for x in d["names"]]
    manifest = json.loads((Path(folder) / "manifest.json").read_text())
    got = content_hash(names, d["ped"], d["ego"], d["length"])
    if got != manifest["content_sha256"][variant]:
        raise ValueError(f"sequences_{variant}.npz in {folder} is not the set of the manifest ({got})")
    return names, d["ped"], d["ego"], d["length"], manifest


def sample_json(name, ped, ego):
    """One sequence in the format of the dataset files. Every joint sits at the root, only the root is scored."""
    n = len(ped)
    joints = np.zeros((n, NJOINTS, 3), dtype=np.float32)
    joints[:, :, 0], joints[:, :, 1], joints[:, :, 2] = ped[:, :1], ROOT_HEIGHT, ped[:, 1:]
    ego3 = np.zeros((n, 3), dtype=np.float32)
    ego3[:, [0, 2]] = ego
    scene, obj = name[1:].split("_")
    return dict(scene_id=scene, object_id=obj, ego_in_ped_frame=ego3.tolist(), ped_in_ped_frame=joints.tolist(),
                vectors_263=encode_root(ped).tolist())


def materialize(target, folder=OUT, stats=STATS):
    """Write both variants where the loader finds them: <target>/<variant>/data/diffusion/munich/val/*.json and
    <target>/<variant>/split/ with the list and the training statistics."""
    for variant in VARIANTS:
        names, ped, ego, length, _ = load_sequences(variant, folder)
        data, split = Path(target) / variant / "data" / "diffusion" / "munich" / "val", Path(target) / variant / "split"
        data.mkdir(parents=True, exist_ok=True)
        split.mkdir(parents=True, exist_ok=True)
        for i, name in enumerate(names):
            (data / f"{name[1:]}.json").write_text(json.dumps(sample_json(name, ped[i, :length[i]], ego[i, :length[i]])))
        (split / "val.txt").write_text("\n".join(names) + "\n")
        for f in STAT_FILES:
            shutil.copyfile(Path(stats) / f, split / f)
        print(f"{variant}: {len(names)} sequences in {data}, list and statistics in {split}")


def check_against_dataset(k, source="nuscenes", seed=0):
    """Rebuild existing dataset files from the labelled boxes with the functions above and report the difference.

    Takes box poses and vehicle poses from the pose pipeline's inputs (10 Hz), keeps the frames the pose fit kept,
    and compares vehicle path and root path with the dataset file. Needs the NAS and joblib.
    """
    import random

    import joblib

    nas = os.environ.get("EGOPED_NAS", "/home/erik/NAS/methods/diffusion_gen") + "/data"
    files = sorted(p for p in Path(f"{nas}/diffusion/{source}/full_dataset").glob("*.json") if not p.name.startswith("M"))
    random.Random(seed).shuffle(files)
    ego_err, root_err, forward = [], [], []
    for f in files[:k]:
        scene, obj = f.stem.split("_")
        base = Path(f"{nas}/drivestudio/{source}/drivestudio_output/{scene}")
        try:
            info = json.loads((base / "instances" / "instances_info.json").read_text())[obj]["frame_annotations"]
            smpl = joblib.load(base / "humanpose" / "smpl.pkl")
            valid = np.asarray(smpl[int(obj) if int(obj) in smpl else obj]["valid_mask"]).astype(bool)
        except Exception as ex:  # noqa: BLE001
            print(f"  {f.stem}: skipped, {type(ex).__name__}")
            continue
        frames, box = np.array(info["frame_idx"]), np.array(info["obj_to_world"])
        keep = np.where(valid[:len(frames)])[0]
        if len(keep) < 5:
            continue
        box = box[keep]
        vehicle = np.array([np.loadtxt(base / "ego_pose" / f"{int(frames[i]):03d}.txt")[:2, 3] for i in keep])
        yaw0, origin = np.arctan2(box[0, 1, 0], box[0, 0, 0]), box[0, :2, 3]
        up = lambda x: np.stack([np.interp(np.linspace(0, len(x) - 1, 2 * len(x)), np.arange(len(x)), x[:, i])  # noqa: E731
                                 for i in range(2)], axis=1)
        e, p = up(to_ped_frame(vehicle, origin, yaw0)), up(to_ped_frame(box[:, :2, 3], origin, yaw0))
        j = json.loads(f.read_text())
        e_ref = np.array(j["ego_in_ped_frame"])[:, [0, 2]]
        p_ref = np.array(j["ped_in_ped_frame"])[:, 0][:, [0, 2]]
        n, m = min(len(e), len(e_ref)), min(len(p), len(p_ref))
        ego_err.append(np.median(np.linalg.norm(e[:n] - e_ref[:n], axis=1)))
        root_err.append(np.median(np.linalg.norm(p[:m] - p_ref[:m], axis=1)))
        if np.linalg.norm(p_ref[-1]) > 3:
            forward.append(np.degrees(np.arctan2(abs(p_ref[-1, 0]), p_ref[-1, 1])))
    q = lambda x: np.percentile(x, [50, 90]).round(3).tolist()  # noqa: E731
    print(f"{source}: {len(ego_err)} files rebuilt from boxes. Median difference per file, 50th and 90th percentile: "
          f"vehicle path {q(ego_err)} m, root path {q(root_err)} m")
    print(f"  walkers (net path over 3 m): {np.mean(np.array(forward) < 30):.0%} of {len(forward)} head within 30 deg of +z")
    return ego_err, root_err


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--materialize", metavar="DIR")
    ap.add_argument("--folder", default=str(OUT), help="where the sequence files and the manifest are")
    ap.add_argument("--check-dataset", type=int, metavar="K")
    args = ap.parse_args()
    if args.build:
        build()
    if args.materialize:
        materialize(args.materialize, folder=args.folder)
    if args.check_dataset:
        for s in ("nuscenes", "waymo"):
            check_against_dataset(args.check_dataset, s)
