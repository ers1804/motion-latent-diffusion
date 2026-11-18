#!/usr/bin/env python3
"""Known answers for energy_score and retrieval_baseline: no GPU, no data. Usage: python research/src/test_energy_retrieval.py"""
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_resplit import beats, compare  # noqa: E402
from energy_score import SIGN_E, energy_terms, load_e, metrics_e, path_distance  # noqa: E402
from retrieval_baseline import drop_replaced, nearest, retrieve  # noqa: E402


def offset(n, t, dx):
    """n paths of t frames that stand at x = dx."""
    p = np.zeros((n, t, 2)); p[..., 0] = dx
    return p


def main():
    n, t = 4, 10
    lens = np.array([10, 10, 5, 5])
    gt = offset(n, t, 0.0)

    # Distance: only the true length counts. Garbage after it must not change the result.
    a = offset(n, t, 3.0); a[2:, 5:] = 99.0
    assert np.allclose(path_distance(a, gt, lens), 3.0)

    # Two samples at +1 and -1 around the recording: error 1, spread 2, energy 0.
    gen = np.stack([offset(n, t, 1.0), offset(n, t, -1.0)], axis=1)
    err, spread = energy_terms(gen, gt, lens)
    assert np.allclose(err, 1.0) and np.allclose(spread, 2.0) and np.allclose(err - 0.5 * spread, 0.0)
    # Both samples at +1: the same error, no spread, energy 1. Concentrating on a wrong answer is not rewarded.
    gen = np.stack([offset(n, t, 1.0), offset(n, t, 1.0)], axis=1)
    err, spread = energy_terms(gen, gt, lens)
    assert np.allclose(err, 1.0) and np.allclose(spread, 0.0)
    # One sample: the energy score is the ADE.
    err, spread = energy_terms(offset(n, t, 2.0)[:, None], gt, lens)
    assert np.allclose(err, 2.0) and np.allclose(spread, 0.0)
    # The spread averages over pairs of different samples (0, 1, 3 -> pairs 1, 3, 2 -> mean 2), not over all K^2.
    gen = np.stack([offset(n, t, 0.0), offset(n, t, 1.0), offset(n, t, 3.0)], axis=1)
    assert np.allclose(energy_terms(gen, gt, lens)[1], 2.0)

    # The same through a dump on disk, which is the path of the report: +1 and -1 give energy 0, +1 and +1 give 1.
    # A padded frame that is not finite must not reach the score.
    with tempfile.TemporaryDirectory() as tmp:
        for name, second, want in (("spread", -1.0, 0.0), ("sharp", 1.0, 1.0)):
            g = np.stack([offset(n, t, 1.0), offset(n, t, second)], axis=1)
            g[2:, :, 5:] = np.nan
            f = Path(tmp) / f"{name}.npz"
            np.savez(f, ades=np.ones((n, 2)), fdes=np.ones((n, 2)), paths=np.array(["nuscenes/val/1_1.json"] * n),
                     roots_gen=g.astype("float32"), roots_gt=gt.astype("float32"), lengths=lens)
            m = load_e(f)
            with np.errstate(all="ignore"), warnings.catch_warnings():  # every recording here stands, so ECS is undefined
                warnings.simplefilter("ignore")
                assert np.allclose(m["energy"], want) and np.isclose(metrics_e(m, np.arange(n))["Energy"], want), (name, m["energy"])

    # A forecaster that draws from the distribution of the recording scores better than a sharper and a wider one.
    rng = np.random.default_rng(0)
    m, k = 4000, 5
    truth = rng.normal(0, 1, (m, 1, 1)) * np.ones((m, t, 2)) * [1.0, 0.0]
    score = {}
    for name, sd in (("sharp", 0.3), ("right", 1.0), ("wide", 3.0)):
        g = rng.normal(0, sd, (m, k, 1, 1)) * np.ones((m, k, t, 2)) * [1.0, 0.0]
        e, s = energy_terms(g, truth, np.full(m, t))
        score[name] = (e - 0.5 * s).mean()
    assert score["right"] < score["sharp"] and score["right"] < score["wide"], score

    # The rule on the new metric: a cell with a lower energy in every scene beats the other one.
    def fake(energy):
        z = np.zeros(m)
        return dict(ade=z + 1, minade=z + 1, p=z + .5, stop=(np.arange(m) % 2).astype(float), k=5, energy=np.asarray(energy))
    clusters = {i: np.arange(i * 40, (i + 1) * 40) for i in range(m // 40)}
    noise = rng.normal(0, .2, m)
    r = compare([fake(1 + noise)], [fake(1.3 + noise[::-1])], clusters, n=300, metrics_fn=metrics_e, sign=SIGN_E)
    assert beats(r["Energy"]) and r["Energy"][0] > 0 and not beats(r["ADE"]), r

    # Retrieval: only the vehicle path over the query's length counts, and shorter bank rows are no candidates.
    bank_ego = np.stack([offset(1, t, x)[0] for x in (0.0, 1.0, 2.0, 5.0, 0.1)])
    bank_len = np.array([10, 10, 10, 10, 4])
    rows, dist = nearest(offset(1, t, 0.2)[0], 6, bank_ego, bank_len, k=2)
    assert list(rows) == [0, 1] and np.allclose(dist, [0.2, 0.8]), (rows, dist)  # row 4 is closest but too short
    rows, _ = nearest(offset(1, t, 0.2)[0], 4, bank_ego, bank_len, k=2)
    assert list(rows) == [4, 0], rows
    late = offset(1, t, 0.0)[0]; late[6:, 0] = 50.0  # differs from row 0 only after the query's length
    rows, dist = nearest(late, 6, bank_ego, bank_len, k=1)
    assert list(rows) == [0] and dist[0] == 0
    rows, _ = nearest(offset(1, t, 1.5)[0], 6, bank_ego, bank_len, k=2)  # tie between rows 1 and 2: bank order
    assert list(rows) == [1, 2], rows

    bank_root = np.arange(5)[:, None, None] * np.ones((5, t, 2))
    roots, rows, _ = retrieve(np.stack([offset(1, t, 0.2)[0], offset(1, t, 4.0)[0]]), np.array([6, 10]), bank_ego, bank_len, bank_root, k=2)
    assert roots.shape == (2, 2, t, 2) and list(rows[1]) == [3, 2]
    assert np.allclose(roots[0, 0, :6], 0) and np.allclose(roots[0, 1, :6], 1) and np.allclose(roots[0, :, 6:], 0)
    assert np.allclose(roots[1, 0], 3) and np.allclose(roots[1, 1], 2)

    # Rows that the loader replaced by their neighbor are dropped, a run of them keeps its last row.
    motion = np.array([[1.0], [2.0], [2.0], [3.0], [4.0], [4.0], [4.0], [5.0]])[:, None, :] * np.ones((8, 3, 2))
    keep = drop_replaced(motion, np.array([3, 3, 3, 3, 3, 3, 3, 3]))
    assert list(keep) == [0, 2, 3, 6, 7], keep
    keep = drop_replaced(motion, np.array([3, 3, 2, 3, 3, 3, 3, 3]))  # equal features, other length: not a replacement
    assert list(keep) == [0, 1, 2, 3, 6, 7], keep
    wrapped = np.concatenate([motion, motion[:1]])  # the loader gave the last row the first sample
    assert list(drop_replaced(wrapped, np.full(9, 3))) == [0, 2, 3, 6, 7], drop_replaced(wrapped, np.full(9, 3))
    assert list(drop_replaced(motion[:1], np.array([3]))) == [0]
    print("all checks passed")


if __name__ == "__main__":
    main()
