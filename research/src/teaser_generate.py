#!/usr/bin/env python3
"""Generate the motions behind the teaser figure: one model, two ego conditions.

Each noise seed is decoded under both conditions, so a difference between the two
motions comes from the ego trajectory alone. Output holds full joints, not only roots:

  python research/src/teaser_generate.py --model h4 --split val_test --scan      # rank condition pairs
  python research/src/teaser_generate.py --model h4 --split val_test --rows 12 345 --seeds 8
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_ade_fde import FV, MODELS, NAS, SCENE, VT  # noqa: E402
from intention_labels import is_stopping  # noqa: E402

FPS = 20
OUT = Path(__file__).resolve().parent.parent / "data"


def build(spec, split):
    scene = split.startswith("scene_")
    split_dir = SCENE if scene else (VT if split == "val_test" else FV)
    overrides = [
        f"DATASET.EGOMOTION.ROOT=[{NAS}/data/diffusion/ava, {NAS}/data/diffusion/nuscenes, {NAS}/data/diffusion/waymo]",
        f"DATASET.EGOMOTION.MEAN_STD_PATH={split_dir}",
        f"DATASET.EGOMOTION.EGO_MEAN_STD_PATH={split_dir}",
        "DATASET.EGOMOTION.INTERACTION_CROP=False",
        "DATASET.EGOMOTION.INTERACTION_WEIGHTED_SAMPLING=False",
        "DATASET.EGOMOTION.DETERMINISTIC_CROP=True",
        f"TEST.CHECKPOINTS={spec['ckpt']}",
        "METRIC.TYPE=['EgoMotionMetrics']",
        "TEST.REPLICATION_TIMES=1",
        f"TEST.SPLIT={'test' if split == 'scene_test' else 'val'}",
        f"model.guidance_scale={spec['guidance']}",
        "TEST.BATCH_SIZE=1",
    ]
    if scene:
        overrides.append("DATASET.EGOMOTION.SCENE_SPLIT=True")
    if spec.get("window_frame"):  # models trained with the ego re-anchored at the window start (rs2_*)
        overrides.append("DATASET.EGOMOTION.EGO_WINDOW_FRAME=True")
    sys.argv = ["teaser", "--cfg", spec["cfg"], "--nodebug", "--overrides"] + overrides

    from mld.config import parse_args
    from mld.data.get_data import get_datasets
    from mld.models.get_model import get_model

    cfg = parse_args(phase="test")
    cfg.FOLDER = "results"
    datamodule = get_datasets(cfg, logger=None, phase="test")[0]
    datamodule.setup(stage="test")
    return cfg, datamodule, get_model


def raw_ego(dataset, row):
    """Denormalised ego window of one row, in metres in the pedestrian's start frame."""
    item = dataset[row]
    return item, np.asarray(item["ego"]) * dataset.ego_std + dataset.ego_mean


def scan(args, dataset):
    dump = np.load(args.dump or OUT / f"ade_fde_{args.model}_{args.split}_k5.npz")
    gen, gt, lens = dump["roots_gen"], dump["roots_gt"], dump["lengths"]
    if "paths" in dump:  # dumps written on another machine must list the same rows in the same order
        assert list(dump["paths"]) == ["/".join(Path(p).parts[-3:]) for p in dataset.sample_paths]
    rows = []
    for i in range(len(lens)):
        L = int(lens[i])
        _, ego = raw_ego(dataset, i)
        ego = ego[:L]
        speed = np.linalg.norm(np.diff(ego, axis=0), axis=1) * FPS
        dist = np.linalg.norm(ego - gt[i, :L], axis=1)
        rows.append(dict(
            row=i, path="/".join(Path(dataset.sample_paths[i]).parts[-3:]), L=L,
            p_stop=float(np.mean([is_stopping(gen[i, k, :L]) for k in range(gen.shape[1])])),
            gt_stop=bool(is_stopping(gt[i, :L])),
            ego_mean_speed=float(speed.mean()), ego_start_speed=float(speed[:20].mean()),
            ego_end_speed=float(speed[-20:].mean()), ego_path=float(speed.sum() / FPS),
            min_dist=float(dist.min()), ego_in_10m=float((dist < 10).mean()),
            gen_travel=float(np.mean([np.linalg.norm(gen[i, k, L - 1] - gen[i, k, 0]) for k in range(gen.shape[1])])),
        ))
    out = OUT / f"teaser_scan_{args.model}_{args.split}.json"
    out.write_text(json.dumps(rows, indent=0))
    print("saved", out)


def generate(args, cfg, datamodule, get_model, spec):
    device = torch.device(args.device)
    load = torch.load
    torch.load = lambda *a, **k: load(*a, **{"map_location": "cpu", **k})  # some submodules load with no map_location
    model = get_model(cfg, datamodule)
    model.load_state_dict(torch.load(spec["ckpt"], map_location="cpu")["state_dict"])
    model = model.to(device).eval()
    dataset = datamodule.test_dataset
    result = {}
    for row in args.rows:
        item, ego = raw_ego(dataset, row)
        L = int(item["length"])
        batch = dict(ego=torch.as_tensor(item["ego"])[None].to(device), length=[L],
                     motion=torch.as_tensor(item["motion"])[None].to(device),
                     ego_length=[L], text=[item["text"]])
        joints = []
        with torch.no_grad():
            for seed in range(args.seeds):
                torch.manual_seed(seed)
                joints.append(model.test_diffusion_forward(batch)["joints_rst"][0].cpu().numpy())
            gt = datamodule.feats2joints(batch["motion"].cpu())[0].numpy()
        result[f"joints_{row}"] = np.stack(joints).astype("float32")  # (seeds, T, 22, 3)
        result[f"gt_{row}"] = gt.astype("float32")
        result[f"ego_{row}"] = ego.astype("float32")
        result[f"len_{row}"] = np.array(L)
        result[f"path_{row}"] = np.array("/".join(Path(dataset.sample_paths[row]).parts[-3:]))
    result["rows"] = np.array(args.rows)
    to_joint_frame(result, args.split, dataset)
    out = OUT / f"teaser_{args.model}_{args.split}.npz"
    np.savez(out, **result)
    print("saved", out)


def to_joint_frame(d, split, dataset=None):
    """ego_<row> into the frame of the joints, for drawing; the ego as the model saw it moves to ego_model_<row>.

    The dataset hands a windowed track its ego in the frame of the track's first pose (coupling_breakdown.py).
    """
    from coupling_breakdown import load_geometry

    rows = [int(r) for r in d["rows"]]
    lens = [int(d[f"len_{r}"]) for r in rows]
    roots = np.zeros((len(rows), max(lens), 2), dtype="float32")
    for i, r in enumerate(rows):
        roots[i, :lens[i]] = d[f"gt_{r}"][:lens[i], 0][:, [0, 2]]
    egos, offset, yaw, _ = load_geometry(split, [str(d[f"path_{r}"]) for r in rows], lens, roots, dataset)
    for i, r in enumerate(rows):
        d[f"ego_model_{r}"] = np.array(d[f"ego_{r}"])
        ego = d[f"ego_model_{r}"].copy()
        ego[:len(egos[i])] = egos[i]
        d[f"ego_{r}"] = ego.astype("float32")
        print(f"row {r}: ego moved by {offset[i]:.2f} m and {yaw[i]:.1f} deg into the frame of the joints")


def reframe(path, split):
    """Bring a file written before 2026-11-13 into the layout above."""
    d = dict(np.load(path))
    if any(k.startswith("ego_model_") for k in d):
        print("already in the frame of the joints:", path)
        return
    to_joint_frame(d, split)
    np.savez(path, **d)
    print("rewrote", path)


def main():
    if "--reframe" in sys.argv:  # python teaser_generate.py --reframe <npz> <split>
        return reframe(sys.argv[sys.argv.index("--reframe") + 1], sys.argv[sys.argv.index("--reframe") + 2])
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cfg")
    ap.add_argument("--ckpt")
    ap.add_argument("--guidance", type=float, default=10)
    ap.add_argument("--split", default="val_test", choices=["val_test", "scene_val", "scene_test"])
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--dump", help="K=5 root dump for --scan, if not named after the model")
    ap.add_argument("--rows", type=int, nargs="*", default=[])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--window-frame", action="store_true", help="rs2_* models: ego in the frame of the window start")
    args = ap.parse_args()
    spec = dict(cfg=args.cfg, ckpt=args.ckpt, guidance=args.guidance) if args.ckpt else dict(MODELS[args.model])
    spec["window_frame"] = args.window_frame or spec.get("window_frame", False) or args.model.startswith("rs2_")
    cfg, datamodule, get_model = build(spec, args.split)
    if args.scan:
        scan(args, datamodule.test_dataset)
    else:
        generate(args, cfg, datamodule, get_model, spec)


if __name__ == "__main__":
    main()
