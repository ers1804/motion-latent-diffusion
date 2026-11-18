#!/usr/bin/env python3
"""Retrieval baseline on the scene split: the motions recorded with the K most similar vehicle paths.

Does the generator add anything to a search of the training data? For every held-out condition the K=5 training
sequences are taken whose vehicle path is closest to the condition's, and their recorded root paths stand in for
the K samples of a generator. Nothing is trained and nothing is tuned:

  - The vehicle path is the one the models receive (the pedestrian's frame, here in metres), and only it is
    matched. Distance: mean distance of the two vehicle paths over the length of the condition.
  - Only training sequences at least as long as the condition can be matched, and their motion is cut to the
    condition's length, which the generators are given as well.
  - The bank is the training partition with one fixed window per sequence (the window rule of the held-out
    partitions). A track that the loader replaces by its neighbor (under 2 s) is in the bank once.

The dump has the layout of eval_ade_fde.py --save-roots, so energy_score.py and compare_resplit.py score it as
they score a model. Protocol: research/review_action_plan_2026-11-13.md, "Energy score and retrieval baseline".

Usage: python research/src/retrieval_baseline.py --splits scene_val            # then, after the commit, scene_test
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
K = 5


def nearest(query_ego, query_len, bank_ego, bank_len, k=K):
    """Bank rows with the k closest vehicle paths over the first query_len frames, closest first, and their distances.

    Rows shorter than the query are not candidates. Ties keep the order of the bank.
    """
    candidates = np.flatnonzero(bank_len >= query_len)
    assert len(candidates) >= k, (query_len, len(candidates))
    diff = bank_ego[candidates, :query_len].astype(np.float64) - query_ego[None, :query_len].astype(np.float64)
    dist = np.linalg.norm(diff, axis=-1).mean(1)
    order = np.argsort(dist, kind="stable")[:k]
    return candidates[order], dist[order]


def drop_replaced(motion, lengths):
    """Rows that are themselves, not the neighbor the loader returned for a track it skips.

    EgoMotionDataset returns sample i+1 for a sample i that is too short or unreadable, so such a row equals the
    next one, and the last row would equal the first. Keeps the last row of every run of equal neighbors.
    """
    n = len(motion)
    nxt = [(i + 1) % n for i in range(n)]
    same_as_next = np.array([n > 1 and lengths[i] == lengths[j] and np.array_equal(motion[i], motion[j])
                             for i, j in enumerate(nxt)])
    assert not same_as_next.all()
    return np.flatnonzero(~same_as_next)


def retrieve(q_ego, q_len, bank_ego, bank_len, bank_root, k=K):
    """roots (N, k, T, 2) with zeros after the condition's length, bank rows (N, k), distances (N, k)."""
    n, t = len(q_ego), bank_root.shape[1]
    roots = np.zeros((n, k, t, 2), dtype=np.float32)
    rows = np.zeros((n, k), dtype=np.int64)
    dists = np.zeros((n, k))
    for i in range(n):
        length = int(q_len[i])
        rows[i], dists[i] = nearest(q_ego[i], length, bank_ego, bank_len, k)
        roots[i, :, :length] = bank_root[rows[i], :length]
    return roots, rows, dists


def read_partition(dm, dataset):
    """ego in metres (N, T, 2), root paths (N, T, 2), lengths, normalized motion and file names of one partition."""
    import torch
    from torch.utils.data import DataLoader
    from mld.data.EgoMotion import ego_motion_collate
    egos, motions, lengths = [], [], []
    for batch in DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8, collate_fn=ego_motion_collate):
        egos.append(batch["ego"]); motions.append(batch["motion"]); lengths.extend(int(v) for v in batch["length"])
    ego, motion = torch.cat(egos).numpy(), torch.cat(motions)
    if dataset.ego_mean is not None:  # undo _normalize_ego, so that the distance is in metres
        ego = ego * (dataset.ego_std + 1e-8) + dataset.ego_mean
    else:
        ego = ego * dataset.ego_scale
    root = dm.feats2joints(motion)[:, :, 0, [0, 2]].numpy()
    paths = np.array(["/".join(Path(p).parts[-3:]) for p in dataset.sample_paths])
    assert len(paths) == len(lengths), (len(paths), len(lengths))
    return ego.astype(np.float32), root.astype(np.float32), np.array(lengths), motion.numpy(), paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["scene_val"], choices=["scene_val", "scene_test"])
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent.parent / "data"))
    args = ap.parse_args()
    splits, data = args.splits, Path(args.data)  # build_cfg overwrites sys.argv
    from compare_resplit import scene_key
    from regressor_baseline import SCENE, build_cfg, get_loaders

    bank = None
    for split in splits:
        dm = get_loaders(build_cfg(SCENE, scene=True, test_split=split.removeprefix("scene_")))
        if bank is None:
            train = dm.train_dataset
            train.deterministic_crop = True  # one fixed window per training sequence, as on the held-out partitions
            assert not train.interaction_crop
            ego, root, lengths, motion, paths = read_partition(dm, train)
            keep = drop_replaced(motion, lengths)
            bank = dict(ego=ego[keep], root=root[keep], lengths=lengths[keep], paths=paths[keep])
            print(f"bank: {len(keep)} training sequences ({len(paths) - len(keep)} replaced rows dropped), "
                  f"{(bank['lengths'] == ego.shape[1]).mean():.0%} of full length")
        q_ego, q_root, q_len, _, q_paths = read_partition(dm, dm.test_dataset)
        assert not set(q_paths) & set(bank["paths"])
        assert not {scene_key(p) for p in q_paths} & {scene_key(p) for p in bank["paths"]}, "a scene is on both sides"
        roots, rows, dists = retrieve(q_ego, q_len, bank["ego"], bank["lengths"], bank["root"])
        err = np.linalg.norm(roots - q_root[:, None], axis=-1)  # (N, K, T)
        ades = np.array([[err[i, k, :n].mean() for k in range(K)] for i, n in enumerate(q_len)])
        fdes = np.array([[err[i, k, n - 1] for k in range(K)] for i, n in enumerate(q_len)])
        out = data / f"ade_fde_retrieval_{split}_k5.npz"
        np.savez(out, ades=ades, fdes=fdes, paths=q_paths, roots_gen=roots, roots_gt=q_root, lengths=q_len,
                 neighbor_paths=bank["paths"][rows], neighbor_dist=dists)
        print(f"[{split}] N={len(q_len)} ADE={ades.mean():.3f} minADE={ades.min(1).mean():.3f} "
              f"vehicle-path distance to the nearest {dists[:, 0].mean():.2f} m, to the fifth {dists[:, -1].mean():.2f} m; saved {out.name}")


if __name__ == "__main__":
    main()
