#!/usr/bin/env python3
"""Draw root trajectories from PedGen's pretrained no-context model (ICLR 2025, plan step 6).

`pedgen_no_context` takes no scene, goal, body shape or ego input, so one bank of samples serves every
split. Its authors' sampling code is used as is, from a clone at commit 7c3f55d; lightning and smplx are
only needed for parts this script does not call and are stubbed.

Output frame matches our dumps: 20 fps, start at the origin, x left and z forward of the initial facing.
PedGen is 60 frames at 30 fps, so 40 frames at 20 fps (1.95 s) is the longest common horizon.

Usage: python research/src/pedgen_baseline.py --repo <PedGen clone> --ckpt <pedgen_no_context.ckpt> --out <npz>
"""
import argparse
import sys
import types

import numpy as np
import torch
import yaml


def stub_dependencies():
    class LightningModule(torch.nn.Module):
        @property
        def device(self):
            return next(self.parameters()).device

    class SMPLLayer(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()

    def module(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m

    module("lightning")
    module("lightning.pytorch", LightningModule=LightningModule)
    module("smplx", SMPLLayer=SMPLLayer)
    module("smplx.lbs", vertices2joints=None)


def to_canonical(samples, rotation_6d_to_matrix):
    """(M, 60, 147) -> root (M, 40, 2) as (x, z) in the frame of the initial facing."""
    trans = torch.cumsum(samples[..., :3], dim=1)
    R = rotation_6d_to_matrix(samples[..., 3:9])  # (M, 60, 3, 3): columns are the body axes in the world
    up = R[..., 1].mean(dim=(0, 1))
    up = up / up.norm()
    facing = R[:, 0, :, 2]
    facing = facing - (facing * up).sum(-1, keepdim=True) * up
    e_z = facing / facing.norm(dim=-1, keepdim=True)
    e_x = torch.cross(up.expand_as(e_z), e_z, dim=-1)
    root = torch.stack([(trans * e_x[:, None]).sum(-1), (trans * e_z[:, None]).sum(-1)], dim=-1)

    index = torch.arange(40, device=root.device) * 1.5
    lo, frac = index.long(), (index % 1)[None, :, None]
    root = root[:, lo] * (1 - frac) + root[:, lo + 1] * frac
    height = (trans * up).sum(-1)
    return root - root[:, :1], up, height


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=15000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    stub_dependencies()
    sys.path.insert(0, args.repo)
    from pedgen.model.pedgen_model import PedGenModel
    from pedgen.utils.rot import rotation_6d_to_matrix

    conf = yaml.safe_load(open(f"{args.repo}/cfgs/pedgen_no_context.yaml"))["model"]
    model = PedGenModel(**conf)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=True)["state_dict"]
    state = {k: v for k, v in state.items() if not k.startswith("smpl.")}
    model.load_state_dict(state)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    torch.manual_seed(args.seed)
    frames, feats = conf["diffuser_conf"]["num_frames"], conf["diffuser_conf"]["input_feats"]
    roots, ups, heights = [], [], []
    for start in range(0, args.n, args.batch):
        b = min(args.batch, args.n - start)
        zeros = torch.zeros(b, frames, feats, device=device)  # the first local translation is inpainted to 0
        cond = model.cond_embed.unsqueeze(0).repeat(b, 1)
        x = model.sample_ddim_progressive(zeros, cond)
        x = model.smooth_motion(x[:, None])[:, 0]  # eval_step smooths with DCT before use
        root, up, height = to_canonical(x, rotation_6d_to_matrix)
        roots.append(root.cpu().numpy().astype("float32"))
        ups.append(up.cpu().numpy())
        heights.append(height.std(1).cpu().numpy())
        print(f"{start + b}/{args.n} up {up.cpu().numpy().round(3)}", flush=True)

    roots = np.concatenate(roots)
    speed = np.linalg.norm(np.diff(roots, axis=1), axis=-1).mean(1) * 20
    print("up axis mean", np.mean(ups, axis=0).round(3), "| vertical std median", np.median(np.concatenate(heights)).round(3),
          "| speed m/s p10/p50/p90", np.percentile(speed, [10, 50, 90]).round(2),
          "| end z median", np.median(roots[:, -1, 1]).round(2), "| end x median", np.median(roots[:, -1, 0]).round(2))
    np.savez_compressed(args.out, roots=roots)
    print("saved", args.out)


if __name__ == "__main__":
    main()
