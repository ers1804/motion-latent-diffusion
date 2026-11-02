#!/usr/bin/env python3
"""Known answers for compare_resplit: no GPU, no data. Usage: python research/src/test_compare_resplit.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_resplit import beats, cell_of, compare, metrics, scene_key  # noqa: E402


def fake(ade, p, stop):
    ade = np.asarray(ade, dtype=float)
    return dict(ade=ade, minade=ade * .5, p=np.asarray(p, dtype=float), stop=np.asarray(stop, dtype=float))


def main():
    assert cell_of("rs1_egoped_ia_s1234") == "egoped_ia" and cell_of("rs1_uncond_s1234") == "uncond"
    assert scene_key("nuscenes/val/M216_9.json") == scene_key("nuscenes/val/216_3.json")
    assert scene_key("waymo/val/216_3.json") != scene_key("nuscenes/val/216_3.json")

    n = 200
    stop = (np.arange(n) % 2).astype(float)
    r = metrics(fake(np.ones(n), stop, stop), np.arange(n))
    assert r["ECS"] == 1 and r["Brier"] == 0 and r["ADE"] == 1 and r["minADE"] == .5
    assert metrics(fake(np.ones(n), np.full(n, .3), stop), np.arange(n))["ECS"] == 0

    clusters = {i: np.array([2 * i, 2 * i + 1]) for i in range(n // 2)}
    rng = np.random.default_rng(1)
    base = 2 + rng.normal(0, .3, n)

    # First cell clearly better on ADE, on every seed pair.
    good = [fake(base - .5 + d, stop, stop) for d in (0, .01, -.01)]
    bad = [fake(base + d, stop * .5, stop) for d in (0, .01, -.01)]
    out = compare(good, bad, clusters, n=300)
    assert abs(out["ADE"][0] - .5) < 1e-9 and beats(out["ADE"]), out["ADE"]
    assert abs(out["ECS"][0] - .5) < 1e-9 and beats(out["ECS"])
    assert abs(compare(bad, good, clusters, n=300)["ADE"][0] + .5) < 1e-9
    assert beats(compare(bad, good, clusters, n=300)["ADE"])  # the rule also holds for a negative delta

    # Tight CI, but one seed pair disagrees: the rule must refuse.
    split = [fake(base - .5, stop, stop), fake(base + .1, stop, stop)]
    out = compare(split, [fake(base, stop, stop)], clusters, n=300)
    assert out["ADE"][4] < 0 < out["ADE"][5] and not beats(out["ADE"]), out["ADE"]

    # Identical cells are no difference.
    out = compare(bad, bad, clusters, n=100)
    assert out["ADE"][0] == 0 and not beats(out["ADE"])
    print("all checks passed")


if __name__ == "__main__":
    main()
