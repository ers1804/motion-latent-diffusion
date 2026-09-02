#!/usr/bin/env python3
"""Cheap text pseudo-ground-truth for the ego-motion dataset (item 10b).

Motivation: our JSONs have no captions, but a text-conditioned in-domain
baseline needs them. Rather than hand-annotating or paying for an LLM, we
derive captions deterministically from kinematics — the same family of
heuristics as `intention_labels.py`, extended into a small template grammar.
CPU-only, reproducible, auditable.

TWO VOCABULARIES (the choice is scientifically load-bearing):
  * "body"  — describes only the pedestrian's own motion, in HumanML3D caption
              style ("a person walks forward and then stops"). Comparable to
              the pretrained MLD's training distribution.
  * "ego"   — additionally verbalizes the vehicle relation ("...as a vehicle
              approaches from the left"). This turns text into a *channel for
              ego information*, so training on it answers: can a compressed
              linguistic description replace the raw 196-step trajectory?

Attributes extracted per sample (all from root/joint trajectories + ego):
  gait      : stands still | walks slowly | walks | walks quickly
  stop      : whether a sustained low-speed phase occurs (>=1 s under 0.2 m/s)
  onset     : starts from standing / already moving
  turn      : straight | turns left | turns right (net heading change)
  ego_rel   : approaches | moves away | passes | stays near   (range trend)
  ego_side  : left | right | front                            (mean bearing)
  proximity : whether ego comes within 5 m

Usage:
  python research/src/synth_captions.py --demo          # show captions for a few samples
  python research/src/synth_captions.py --write-all     # dump captions.json for train+val
"""
import argparse
import json
import math
import random
from glob import glob
from pathlib import Path

import numpy as np

BASE = "/home/erik/NAS/methods/diffusion_gen/data/diffusion"
FPS = 20.0
STOP_SPEED, STOP_FRAMES = 0.2, 20     # matches intention_labels.py
SLOW, FAST = 0.7, 1.6                 # m/s boundaries for gait wording
TURN_DEG = 35.0                       # net heading change to call it a turn
NEAR_M = 5.0


def _attrs(ped_xz, ego_xz):
    v = np.linalg.norm(np.diff(ped_xz, axis=0), axis=1) * FPS
    moving = v[v > STOP_SPEED]
    mean_speed = float(moving.mean()) if len(moving) else 0.0

    # sustained stop?
    run = best = 0
    for s in (v < STOP_SPEED):
        run = run + 1 if s else 0
        best = max(best, run)
    stop = best >= STOP_FRAMES
    starts_moving = bool((v[:STOP_FRAMES] < STOP_SPEED).all()) if len(v) > STOP_FRAMES else False

    if mean_speed == 0.0:
        gait = "stands still"
    elif mean_speed < SLOW:
        gait = "walks slowly"
    elif mean_speed > FAST:
        gait = "walks quickly"
    else:
        gait = "walks"

    # net heading change of the pedestrian's own path
    d = np.diff(ped_xz, axis=0)
    d = d[np.linalg.norm(d, axis=1) > 1e-3]
    turn = "straight"
    if len(d) > 4:
        a0 = math.atan2(*d[:max(1, len(d)//5)].mean(0)[::-1])
        a1 = math.atan2(*d[-max(1, len(d)//5):].mean(0)[::-1])
        dth = math.degrees((a1 - a0 + math.pi) % (2 * math.pi) - math.pi)
        if dth > TURN_DEG:
            turn = "turns left"
        elif dth < -TURN_DEG:
            turn = "turns right"

    # ego relation
    rng = np.linalg.norm(ego_xz - ped_xz, axis=1)
    first, last, lo = rng[:len(rng)//4].mean(), rng[-len(rng)//4:].mean(), rng.min()
    if last < first - 1.0:
        ego_rel = "approaches"
    elif last > first + 1.0:
        ego_rel = "moves away"
    elif lo < first - 1.0:
        ego_rel = "passes"
    else:
        ego_rel = "stays near"
    rel = ego_xz - ped_xz
    bearing = math.degrees(math.atan2(rel[:, 1].mean(), rel[:, 0].mean()))
    ego_side = "left" if bearing > 30 else ("right" if bearing < -30 else "front")
    return dict(gait=gait, stop=stop, starts_moving=starts_moving, turn=turn,
                ego_rel=ego_rel, ego_side=ego_side, near=bool(lo < NEAR_M),
                mean_speed=round(mean_speed, 2), min_range=round(float(lo), 2))


def caption(a, vocab="body", rng=None, subj=None):
    if subj is None:
        subj = "a person"
    # --- body clause
    if a["gait"] == "stands still":
        body = f"{subj} stands still"
    else:
        body = f"{subj} {a['gait']} forward"
        if a["starts_moving"]:
            body = f"{subj} starts walking"
        if a["turn"] != "straight":
            body += f" and {a['turn']}"
        if a["stop"]:
            body += " and then stops"
    if vocab == "body":
        return body + "."
    # --- ego clause
    verb = {"approaches": "a vehicle approaches",
            "moves away": "a vehicle drives away",
            "passes": "a vehicle passes by",
            "stays near": "a vehicle waits nearby"}[a["ego_rel"]]
    where = "" if a["ego_side"] == "front" else f" from the {a['ego_side']}"
    close = " close by" if a["near"] else ""
    return f"{body} as {verb}{where}{close}."


def load(fp):
    d = json.load(open(fp))
    ped = np.asarray(d["ped_in_ped_frame"], dtype=np.float32)[:, 0, :][:, [0, 2]]
    ego = np.asarray(d["ego_in_ped_frame"], dtype=np.float32)[:, [0, 2]]
    T = min(len(ped), len(ego))
    return ped[:T], ego[:T]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--write-all", action="store_true")
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()
    rng = random.Random(0)

    if args.demo:
        files = sorted(sum((glob(f"{BASE}/{s}/val/*.json") for s in ("ava", "nuscenes", "waymo")), []))
        rng.shuffle(files)
        for fp in files[:args.n]:
            a = _attrs(*load(fp))
            subj = random.Random(Path(fp).stem).choice(["a person", "a man", "a woman", "a pedestrian"])
            print(f"{Path(fp).stem:>12s} | speed={a['mean_speed']:.2f} minrange={a['min_range']:>5.1f}")
            print(f"             body: {caption(a, 'body', subj=subj)}")
            print(f"             ego : {caption(a, 'ego', subj=subj)}")
        return

    if args.write_all:
        out = {}
        for split in ("train", "val"):
            for src in ("ava", "nuscenes", "waymo"):
                for fp in glob(f"{BASE}/{src}/{split}/*.json"):
                    try:
                        a = _attrs(*load(fp))
                    except Exception:
                        continue
                    subj = random.Random(Path(fp).stem).choice(["a person", "a man", "a woman", "a pedestrian"])
                    out[f"{src}/{split}/{Path(fp).stem}"] = {
                        "body": caption(a, "body", subj=subj),
                        "ego": caption(a, "ego", subj=subj),
                        "attrs": a,
                    }
        dst = Path("research/data/synth_captions.json")
        json.dump(out, open(dst, "w"), indent=0)
        print(f"wrote {dst} ({len(out)} captions)")


if __name__ == "__main__":
    main()
