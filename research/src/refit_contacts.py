#!/usr/bin/env python3
"""De-slide the pseudo-GT poses: anchor footfalls to the ground, preserving the tracked root.

The audit (research/src/pose_quality_audit.py) found the labels slide (foot-skate 0.97) and the
decomposition (skate_decomposition.py) found the slide is already in the VAE. This module tests
whether the labels can be corrected at all, which is the precondition for any retraining.

What is preserved: the root trajectory (the component the audit found sound), the whole upper body,
timing, and overall walking speed. What is rewritten: the legs (hip->knee->ankle->toe).

Method: plan footsteps along the measured root path at a cadence implied by the measured speed, hold
each foot at its planted position through stance, arc it to the next during swing, then solve 2-link
IK for the knee with the template's own femur/tibia lengths. Foot skate then goes to ~0 by
construction.

HONEST CAVEAT: the legs become procedural gait. Measured leg swing is only ~0.73x what the root
speed requires, so this is part correction and part synthesis. The result is physically consistent
but no longer a pose estimate; it is a different trade, not strictly better ground truth.
"""
import numpy as np

LEGS = {"R": (2, 5, 8, 11), "L": (1, 4, 7, 10)}   # pelvis-side hip, knee, ankle, toe
SIDE = {"R": +1.0, "L": -1.0}
STANCE_FRAC = 0.62        # fraction of the gait cycle each foot spends planted
SWING_LIFT = 0.10         # m, peak foot lift during swing
FPS = 20.0


def _unit(v, eps=1e-8):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + eps)


def _headings(root_h, smooth=5):
    """Per-frame forward/lateral unit vectors in the horizontal plane, from the root path."""
    T = len(root_h)
    d = np.gradient(root_h, axis=0)
    k = np.ones(smooth) / smooth
    d = np.stack([np.convolve(d[:, i], k, mode="same") for i in range(2)], 1)
    n = np.linalg.norm(d, axis=1)
    good = n > 1e-6
    if not good.any():
        d = np.tile(np.array([0.0, 1.0]), (T, 1))
    else:                                   # hold the last valid heading through stationary spans
        idx = np.where(good, np.arange(T), 0)
        np.maximum.accumulate(idx, out=idx)
        d = d[idx]
    fwd = _unit(d)
    lat = np.stack([-fwd[:, 1], fwd[:, 0]], 1)
    return fwd, lat


def _ik_knee(hip, ankle, l1, l2, fwd3):
    """2-link IK: knee position given hip and ankle, bending toward `fwd3`."""
    seg = ankle - hip
    d = np.linalg.norm(seg, axis=-1, keepdims=True)
    dmax = (l1 + l2) * 0.999
    d_c = np.clip(d, abs(l1 - l2) + 1e-4, dmax)
    ankle = hip + _unit(seg) * d_c                      # pull the ankle in if over-extended
    u = _unit(ankle - hip)
    a = (l1 ** 2 - l2 ** 2 + d_c ** 2) / (2 * d_c)
    h = np.sqrt(np.maximum(l1 ** 2 - a ** 2, 0.0))
    n = fwd3 - u * np.sum(fwd3 * u, axis=-1, keepdims=True)
    bad = np.linalg.norm(n, axis=-1) < 1e-6
    if bad.any():
        alt = np.tile(np.array([0.0, 1.0, 0.0]), (len(n), 1))
        n[bad] = (alt - u * np.sum(alt * u, axis=-1, keepdims=True))[bad]
    return hip + u * a + _unit(n) * h, ankle


def refit(J, fps=FPS):
    """J: (T,22,3) joints. Returns de-slid joints, same shape, root and upper body untouched."""
    J = np.asarray(J, dtype=np.float64).copy()
    T = len(J)
    if T < 20:
        return J
    root_h = J[:, 0][:, [0, 2]]
    fwd, lat = _headings(root_h)
    step = np.linalg.norm(np.diff(root_h, axis=0), axis=1)
    speed = np.concatenate([step[:1], step]) * fps
    v_ref = np.median(speed[speed > 0.3]) if (speed > 0.3).any() else 0.0
    if v_ref <= 0.0:
        return J                                     # standing still: nothing to anchor
    L_step = float(np.clip(0.55 * v_ref, 0.25, 0.90))
    T_cycle = max(2.0 * L_step / v_ref, 0.5)         # seconds for a full L+R cycle
    per = T_cycle * fps

    ground = np.percentile(J[:, :, 1], 1.0)
    for leg, (hip_i, knee_i, ank_i, toe_i) in LEGS.items():
        l1 = float(np.median(np.linalg.norm(J[:, hip_i] - J[:, knee_i], axis=-1)))
        l2 = float(np.median(np.linalg.norm(J[:, knee_i] - J[:, ank_i], axis=-1)))
        toe_off = J[:, toe_i] - J[:, ank_i]
        toe_len = float(np.median(np.linalg.norm(toe_off[:, [0, 2]], axis=-1)))
        toe_dy = float(np.median(toe_off[:, 1]))
        foot_y = float(np.percentile(J[:, ank_i, 1], 5))
        hip_w = float(np.median(np.linalg.norm(J[:, hip_i][:, [0, 2]] - root_h, axis=-1)))

        phase = (np.arange(T) / per + (0.5 if leg == "R" else 0.0)) % 1.0
        td = [0] + [t for t in range(1, T) if phase[t] < phase[t - 1]] + [T]
        anchors = {}
        for t0 in td[:-1]:                            # plant position at each touchdown
            anchors[t0] = np.array([
                root_h[t0, 0] + fwd[t0, 0] * 0.5 * L_step + lat[t0, 0] * SIDE[leg] * hip_w,
                foot_y,
                root_h[t0, 1] + fwd[t0, 1] * 0.5 * L_step + lat[t0, 1] * SIDE[leg] * hip_w])
        keys = sorted(anchors)
        new_ank = np.zeros((T, 3))
        for i, t0 in enumerate(keys):
            t1 = keys[i + 1] if i + 1 < len(keys) else T
            A0 = anchors[t0]
            A1 = anchors[keys[i + 1]] if i + 1 < len(keys) else A0
            n = t1 - t0
            ns = max(int(round(n * STANCE_FRAC)), 1)
            new_ank[t0:t0 + ns] = A0                              # stance: planted
            if t0 + ns < t1:                                      # swing: arc to the next plant
                m = t1 - (t0 + ns)
                w = (np.arange(m) + 1) / m
                arc = A0[None] * (1 - w[:, None]) + A1[None] * w[:, None]
                arc[:, 1] += SWING_LIFT * np.sin(np.pi * w)
                new_ank[t0 + ns:t1] = arc
        new_ank[:, 1] = np.maximum(new_ank[:, 1], ground)
        fwd3 = np.stack([fwd[:, 0], np.zeros(T), fwd[:, 1]], 1)
        knee, ank = _ik_knee(J[:, hip_i], new_ank, l1, l2, fwd3)
        J[:, knee_i] = knee
        J[:, ank_i] = ank
        J[:, toe_i] = ank + fwd3 * toe_len + np.array([0.0, toe_dy, 0.0])
    return J
