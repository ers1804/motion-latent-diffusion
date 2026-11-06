#!/usr/bin/env python3
"""Pick test conditions for the qualitative figure by a fixed rule, from teaser_generate.py --scan output.

Per source (nuScenes, Waymo), the conditions where the vehicle comes closest to the pedestrian in each outcome:
  correct_stop  ground truth stops, model stops (p_stop >= 0.8 over the K=5 dump samples)
  correct_walk  ground truth walks, model never stops (p_stop = 0)
  missed_stop   ground truth stops, model never stops            (failure)
  false_stop    ground truth walks, model stops (p_stop >= 0.8)  (failure)
  worst_ade     the highest mean ADE                             (failure)
plus every AVA test sequence (there are six) and four rows drawn at random with seed 0.

Usage: python research/src/select_qualitatives.py --scan research/data/teaser_scan_rs1_pooled_scene_test.json \
           --dump research/data/ade_fde_rs1_pooled_s1234_ep1399_cfg15_scene_test_k5.npz --out <json>
"""
import argparse
import json

import numpy as np

PER_CATEGORY = 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = json.load(open(args.scan))
    ade = np.load(args.dump)["ades"].mean(1)
    for r in rows:
        r["ade"] = float(ade[r["row"]])
        r["source"] = r["path"].split("/")[0]
    rules = {
        "correct_stop": lambda r: r["gt_stop"] and r["p_stop"] >= .8,
        "correct_walk": lambda r: not r["gt_stop"] and r["p_stop"] == 0,
        "missed_stop": lambda r: r["gt_stop"] and r["p_stop"] == 0,
        "false_stop": lambda r: not r["gt_stop"] and r["p_stop"] >= .8,
    }
    picked = {}
    for source in ("nuscenes", "waymo"):
        pool = [r for r in rows if r["source"] == source]
        for name, rule in rules.items():
            for r in sorted(filter(rule, pool), key=lambda r: r["min_dist"])[:PER_CATEGORY]:
                picked.setdefault(r["row"], (name, r))
        worst = max(pool, key=lambda r: r["ade"])
        picked.setdefault(worst["row"], ("worst_ade", worst))
    for r in rows:
        if r["source"] == "ava":
            picked.setdefault(r["row"], ("ava", r))
    rest = [r for r in rows if r["row"] not in picked]
    for i in np.random.default_rng(0).choice(len(rest), 4, replace=False):
        picked.setdefault(rest[i]["row"], ("random", rest[i]))

    out = [dict(category=c, **{k: r[k] for k in ("row", "path", "source", "L", "gt_stop", "p_stop", "ade", "min_dist",
                                                  "ego_mean_speed", "ego_start_speed", "ego_end_speed")})
           for c, r in picked.values()]
    out.sort(key=lambda r: (r["source"], r["category"], r["row"]))
    json.dump(out, open(args.out, "w"), indent=1)
    for r in out:
        print(f"{r['row']:5d} {r['category']:13s} {r['path']:28s} L={r['L']:3d} gt_stop={r['gt_stop']!s:5s} "
              f"p_stop={r['p_stop']:.1f} ADE={r['ade']:.2f} min_dist={r['min_dist']:.1f}")
    print(len(out), "conditions; rows:", " ".join(str(r["row"]) for r in out))


if __name__ == "__main__":
    main()
