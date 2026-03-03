# Trajectory Guidance for Motion Latent Diffusion

## Overview

This document describes two methods for improving ego-trajectory fidelity in MLD (Motion Latent Diffusion): **gradient-based guidance** during DDIM sampling, and **post-hoc trajectory injection** after generation. Both are implemented in `demo_ego_motion.py` and `mld/models/modeltype/mld.py`.

---

## Background

### MLD Architecture

MLD operates in a compressed latent space. A VAE encodes motion sequences $\mathbf{x} \in \mathbb{R}^{T \times 263}$ (HumanML3D format) into latents $\mathbf{z} \in \mathbb{R}^{1 \times 256}$, and a denoiser network performs diffusion in this latent space.

The 263-dimensional motion features are structured as:

| Columns | Feature |
|---------|---------|
| 0 | Root angular velocity (y-axis) |
| 1 | Root x-velocity (local frame) |
| 2 | Root z-velocity (local frame) |
| 3 | Root height (y) |
| 4–66 | Joint positions (local, 21 joints × 3) |
| 67–193 | Joint velocities (21 joints × 3 + root × 3) |
| 194–259 | Joint rotations (6D, 22 joints × 3) |
| 260–262 | Foot contact labels (4 binary) |

### Root Trajectory Recovery

Global root positions are recovered from the local velocity features via:

$$r_0 = 0, \quad \theta_t = \sum_{i=0}^{t} \omega_i$$

$$\Delta p_t^{\text{global}} = R(\theta_t) \cdot \begin{pmatrix} v_t^x \\ v_t^z \end{pmatrix}$$

$$p_t = \sum_{i=0}^{t} \Delta p_i^{\text{global}}$$

where $\omega_i$ is the angular velocity (col 0), $v_t^x, v_t^z$ are local-frame velocities (cols 1–2), and $R(\theta)$ is the 2D rotation matrix for heading angle $\theta$.

This is implemented in `mld/transforms/joints2rots/quaternion.py::recover_root_rot_pos`.

### DDIM Sampling

Standard DDIM reverse step from $\mathbf{z}_t$ to $\mathbf{z}_{t-1}$:

$$\hat{\mathbf{z}}_0 = \frac{\mathbf{z}_t - \sqrt{1 - \bar{\alpha}_t} \, \boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t)}{\sqrt{\bar{\alpha}_t}}$$

$$\mathbf{z}_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \, \hat{\mathbf{z}}_0 + \sqrt{1 - \bar{\alpha}_{t-1}} \, \boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t)$$

---

## Method 1: Gradient-Based Trajectory Guidance

### Approach

Inspired by [GMD (Guided Motion Diffusion)](https://github.com/korrawe/guided-motion-diffusion), we apply classifier-free guidance by computing gradients of a trajectory loss with respect to the predicted clean latent $\hat{\mathbf{z}}_0$ at each DDIM step.

### Algorithm

At each reverse step $t$:

1. **Predict clean latent**: Compute $\hat{\mathbf{z}}_0$ from $(\mathbf{z}_t, \boldsymbol{\epsilon}_\theta)$ via the DDIM formula.

2. **Decode to motion**: Pass $\hat{\mathbf{z}}_0$ through the frozen VAE decoder:
   $$\hat{\mathbf{x}} = \text{Dec}(\hat{\mathbf{z}}_0) \in \mathbb{R}^{T \times 263}$$

3. **Extract predicted trajectory**: Apply `recover_root_rot_pos` to get predicted root XZ positions $\hat{\mathbf{p}} \in \mathbb{R}^{T \times 2}$.

4. **Compute trajectory loss** against target positions $\mathbf{p}^*$ with mask $\mathbf{m}$ (1 at observed frames):
   $$\mathcal{L} = \frac{1}{|\mathbf{m}|} \sum_{i} m_i \left\| \hat{\mathbf{p}}_i - \mathbf{p}^*_i \right\|^2$$

5. **Backpropagate** to get gradient $\mathbf{g} = \nabla_{\hat{\mathbf{z}}_0} \mathcal{L}$.

6. **Correct** with adaptive clamp (see below):
   $$\hat{\mathbf{z}}_0' = \hat{\mathbf{z}}_0 - s \cdot \text{clamp}(\mathbf{g})$$
   where $s$ is a scale hyperparameter.

7. **Re-derive** $\mathbf{z}_{t-1}$ using corrected $\hat{\mathbf{z}}_0'$ with the original noise $\boldsymbol{\epsilon}_\theta$:
   $$\mathbf{z}_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \, \hat{\mathbf{z}}_0' + \sqrt{1 - \bar{\alpha}_{t-1}} \, \boldsymbol{\epsilon}_\theta$$

### Adaptive Clamping

**Problem**: Raw gradient magnitudes vary wildly across timesteps. At high noise levels ($t > 250$), $\hat{\mathbf{z}}_0$ is essentially random, and gradient norms are 28–180× larger than at low-$t$ steps. Applying a constant scale $s$ either:
- Is too small at low-$t$ (no effect), or
- Is too large at high-$t$ (catastrophic divergence, ADE > 15 on some samples).

**Solution**: Clamp the correction magnitude to a fixed fraction $\rho$ of the predicted latent's norm:

$$\mathbf{c} = s \cdot \mathbf{g}$$

$$\mathbf{c}' = \mathbf{c} \cdot \min\!\left(1, \; \frac{\rho \, \|\hat{\mathbf{z}}_0\|}{\|\mathbf{c}\|}\right)$$

With $\rho = 0.10$ (10% of $\|\hat{\mathbf{z}}_0\|$), this naturally adapts: at high-$t$ where $\hat{\mathbf{z}}_0$ is noisy (norm ~150) and gradients are huge (~5600 correction norm), the clamp limits corrections to ~15. At low-$t$ where corrections are already small, the clamp is inactive.

### Gradient Flow Verification

The full chain is differentiable:

$$\hat{\mathbf{z}}_0 \xrightarrow{\text{VAE Dec}} \hat{\mathbf{x}} \xrightarrow{\text{recover\_root}} \hat{\mathbf{p}} \xrightarrow{\mathcal{L}} \mathbb{R}$$

Verified empirically: gradient norm at $\hat{\mathbf{z}}_0$ is ~2.5, with element-wise absolute mean ~0.039.

### Hyperparameters

| Parameter | Flag | Default | Notes |
|-----------|------|---------|-------|
| Scale $s$ | `--guide_scale` | 200 | Magnitude before clamping |
| Cap ratio $\rho$ | hardcoded | 0.10 | 10% of $\|\hat{\mathbf{z}}_0\|$ |
| Start timestep | `--guide_start_t` | 1000 | Guide at all steps (clamping handles high-$t$) |
| Stop timestep | `--guide_stop_t` | 0 | Guide until final step |

---

## Method 2: Post-Hoc Trajectory Injection

### Approach

After generation, directly replace the root velocity features (columns 1–2) in the decoded motion with velocities derived from the target trajectory. All body pose features are preserved from the diffusion model's output.

### Algorithm

Given decoded motion $\hat{\mathbf{x}} \in \mathbb{R}^{T \times 263}$ and target global root positions $\mathbf{p}^* \in \mathbb{R}^{T \times 2}$:

1. **Extract predicted heading**: From the generated angular velocities $\omega_t = \hat{\mathbf{x}}_{t,0}$:
   $$\theta_t = \sum_{i=0}^{t} \omega_i$$

2. **Compute target global velocities**:
   $$\Delta \mathbf{p}^*_t = \mathbf{p}^*_t - \mathbf{p}^*_{t-1}, \quad \Delta \mathbf{p}^*_0 = \mathbf{p}^*_0$$

3. **Rotate to local frame** using the model's predicted heading:
   $$\begin{pmatrix} v_t^x \\ v_t^z \end{pmatrix} = R(-\theta_t) \cdot \Delta \mathbf{p}^*_t$$

4. **Replace** columns 1–2:
   $$\hat{\mathbf{x}}_{t,1} \leftarrow v_t^x, \quad \hat{\mathbf{x}}_{t,2} \leftarrow v_t^z$$

This is exact by construction: the replaced velocities, when integrated forward with the same heading angles, reproduce $\mathbf{p}^*$ exactly.

### Properties

- **ADE = 0, FDE = 0** by construction
- Body pose, joint rotations, velocities, and foot contacts are untouched
- Heading (angular velocity, col 0) is preserved from the model
- No retraining required — purely a post-processing step

---

## Experimental Results

### Setup

- **Dataset**: 10 samples from AVA train split
- **Seed**: 42 (set per-sample for reproducibility)
- **Metrics**: ADE (Average Displacement Error) and FDE (Final Displacement Error) in meters, comparing predicted root XZ trajectory vs. ground-truth ego trajectory
- **Inference**: 50 DDIM steps (from 1000 training steps)

### Results

| Method | Det. ADE | Det. FDE | Stoch. ADE | Stoch. FDE |
|--------|----------|----------|------------|------------|
| Baseline | 4.28 | 9.72 | 5.53 | 11.46 |
| Gradient guidance | 3.31 (−23%) | 7.03 (−28%) | 3.38 (−39%) | 7.21 (−37%) |
| Post-hoc injection | 0.00 | 0.00 | 0.00 | 0.00 |
| Guidance + injection | 0.00 | 0.00 | 0.00 | 0.00 |

**Checkpoints**: Both trained for 5000 epochs on AVA + NuScenes + Waymo with the new VAE.
- Deterministic: VAE uses $\mathbf{z} = \boldsymbol{\mu}$ (posterior mean)
- Stochastic: VAE samples $\mathbf{z} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2)$

### Observations

1. **Deterministic > stochastic at baseline** (ADE 4.28 vs 5.53), likely because the deterministic VAE produces a tighter, more learnable latent distribution for the denoiser.

2. **Gradient guidance closes the gap**: After guidance, both checkpoints converge to ~3.3 ADE / ~7.1 FDE. The stochastic model benefits proportionally more (−39% vs −23%).

3. **Guidance is robust**: The adaptive 10% clamp eliminated catastrophic failures. Without it, scale=200 caused ADE > 15 on 1/10 samples (sample 0003_36 diverged to ADE=15.24). With the clamp, worst-case ADE was ~8.

4. **Injection is perfect** for trajectory metrics but doesn't affect body motion quality.

### Scale Sensitivity (Deterministic, sample 0001_33)

| Scale | ADE | FDE | Notes |
|-------|-----|-----|-------|
| 0 (baseline) | 2.68 | 5.51 | |
| 50 | 2.02 | 3.72 | |
| 100 | 1.53 | 2.23 | |
| 150 | 1.30 | 1.67 | |
| 200 | 1.17 | 1.38 | Best |
| 225 | diverged | diverged | Without clamp |

This motivated the adaptive clamp — the optimal scale varies per sample and per timestep.

---

## Usage

```bash
# Baseline (no guidance)
python demo_ego_motion.py \
  --config configs/config_ego_motion_new_vae_det.yaml \
  --checkpoint /path/to/checkpoint.ckpt \
  --data_dir /path/to/ava/train \
  --seed 42 --trajectories

# Gradient guidance
python demo_ego_motion.py \
  --config configs/config_ego_motion_new_vae_det.yaml \
  --checkpoint /path/to/checkpoint.ckpt \
  --data_dir /path/to/ava/train \
  --seed 42 --trajectories \
  --guide --guide_scale 200

# Post-hoc trajectory injection
python demo_ego_motion.py \
  --config configs/config_ego_motion_new_vae_det.yaml \
  --checkpoint /path/to/checkpoint.ckpt \
  --data_dir /path/to/ava/train \
  --seed 42 --trajectories \
  --inject_traj

# Combined (guidance steers the latent, injection ensures exact trajectory)
python demo_ego_motion.py \
  --config configs/config_ego_motion_new_vae_det.yaml \
  --checkpoint /path/to/checkpoint.ckpt \
  --data_dir /path/to/ava/train \
  --seed 42 --trajectories \
  --guide --guide_scale 200 --inject_traj
```

---

## Bug Fix: `prediction_type` Mismatch

During analysis, a critical inference bug was discovered. The stochastic configs had `PREDICT_EPSILON: True` in the model config but `prediction_type: sample` in the DDIM scheduler YAML. This mismatch meant the scheduler interpreted the model's noise prediction $\boldsymbol{\epsilon}_\theta$ as a clean-sample prediction $\hat{\mathbf{z}}_0$, producing garbage trajectories.

**Fix** (applied to `mld.py` + both stochastic YAML configs):
- `mld/models/modeltype/mld.py`: Explicitly overrides `scheduler.config.prediction_type` based on `PREDICT_EPSILON`
- `configs/config_ego_motion_new_vae_stoch.yaml`: `prediction_type: sample` → `prediction_type: epsilon`
- `configs/config_ego_motion_old_vae_stoch.yaml`: same

Note: training was unaffected because the loss is computed as raw MSE between predicted and target noise/sample, bypassing the scheduler entirely.

---

## Files Modified

| File | Changes |
|------|---------|
| `mld/models/modeltype/mld.py` | Fixed `prediction_type` override; added `_diffusion_reverse_guided()` method |
| `demo_ego_motion.py` | Added `--seed`, `--guide*`, `--inject_traj` flags; `inject_trajectory()` function; guidance plumbing in `generate_motion()` |
| `configs/config_ego_motion_new_vae_stoch.yaml` | `prediction_type: epsilon` |
| `configs/config_ego_motion_old_vae_stoch.yaml` | `prediction_type: epsilon` |
