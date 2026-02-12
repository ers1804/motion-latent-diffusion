# Ego-Conditioned MLD: Complete Change Summary

### Overview
This branch adds **ego vehicle trajectory conditioning** to the Motion Latent Diffusion (MLD) model. Instead of generating motion from text, it generates pedestrian motion conditioned on the ego vehicle's trajectory (useful for autonomous driving scenarios).

---

### New Files Created

| File | Description |
|------|-------------|
| `mld/data/EgoMotion.py` | Dataset class for loading ego+motion pairs from JSON files. Supports multiple data sources (waymo, nuscenes, ava). |
| `mld/models/architectures/ego_encoder.py` | Transformer encoder for ego trajectory. Input: `(B, T, 2)` → Output: `(B, T, 256)`. Also includes `EgoEncoderPooled` variant (unused). |
| `configs/config_ego_motion.yaml` | Main config file for ego-conditioned training. |

---

### Modified Files

#### 1. `mld/models/modeltype/mld.py`
**Key changes:**
- **Line 59-60**: Initialize `ego_encoder` when `condition == 'ego'`
- **Line 122-124**: Set `feats2joints` for ego condition
- **Lines 240-248**: `forward()` - ego condition handling with CFG
- **Lines 539-555**: NEW METHOD `_compute_diffusion_loss()` - bypasses torchmetrics for gradient issues
- **Lines 594-606**: `train_diffusion_forward()` - ego CFG with input zeroing (per-sample dropout)
- **Lines 635-641**: `test_diffusion_forward()` - ego condition handling
- **Lines 901-917**: `allsplit_step()` - direct loss computation for diffusion stage
- **Lines 927-929**: `allsplit_step()` - ego evaluation routing
- **Line 507**: `train_vae_forward()` - include ego in joints recovery

#### 2. `mld/models/architectures/mld_denoiser.py`
**Key changes:**
- **Lines 78-93**: `__init__()` - ego condition initialization (time projection, embedding projection)
- **Lines 192-204**: `forward()` - ego embedding handling with time concatenation

#### 3. `mld/data/get_data.py`
**Key changes:**
- **Line 9**: Import `EgoMotionDataModule, ego_motion_collate`
- **Lines 61-62**: `get_collate_fn()` - return ego collate function
- **Line 76**: `dataset_module_map` - add egomotion entry
- **Lines 139-173**: `get_datasets()` - full egomotion dataset initialization

#### 4. `configs/modules/denoiser.yaml`
**Key change:**
- **Line 4**: Commented out `text_encoded_dim: 768` to prevent OmegaConf override

#### 5. `train.py`
**Key change:**
- **Line 121**: Uses `RichProgressBar()` (requires rich==12.6.0)

---

### Critical Dependencies

```bash
# Required package versions (different from original MLD)
pip install torchmetrics==0.7.3   # Downgraded from newer versions
pip install rich==12.6.0          # Downgraded from 14.x for PL 1.9.5 compatibility
```

**Full environment:**
- Python 3.9.25
- PyTorch 1.12.1
- pytorch-lightning 1.9.5
- torchmetrics 0.7.3
- rich 12.6.0

---

### Data Format

Each JSON file in `{dataset}/train/` or `{dataset}/val/`:
```json
{
  "scene_id": "string",
  "object_id": "string", 
  "ego_in_ped_frame": [[x, y, z], ...],  // Ego trajectory, uses (x, z)
  "vectors_263": [[263 features], ...]    // HumanML3D format motion
}
```

---

### How to Train

```bash
python train.py --cfg configs/config_ego_motion.yaml --nodebug
```

**Important:** Always use `--nodebug` flag due to a bug in `config.py` line 170 where DEBUG is incorrectly set.

---

### Config Customization

Edit `configs/config_ego_motion.yaml`:
- `TRAIN.PRETRAINED_VAE`: Path to pre-trained VAE checkpoint
- `DATASET.EGOMOTION.ROOT`: List of data directories
- `DATASET.EGOMOTION.MEAN_STD_PATH`: Path to Mean.npy/Std.npy (must match VAE training)
- `model.guidance_uncondp`: CFG dropout rate (default 0.1 = 10%)

---

### Architecture Summary

```
Input: Ego trajectory (B, T_ego, 2)
         ↓
    EgoEncoder (Transformer)
         ↓
    Ego embedding (B, T_ego, 256)
         ↓
    MLD Denoiser (with CFG)
         ↓
    Latent z (B, 1, 256)
         ↓
    Pre-trained VAE Decoder
         ↓
Output: Motion (B, T_motion, 263)
```

---

### Known Issues & Workarounds

| Issue | Workaround |
|-------|------------|
| `DEBUG: True` despite config | Use `--nodebug` flag |
| `text_encoded_dim: 768` override | Comment out in `configs/modules/denoiser.yaml` |
| Loss gradient errors with torchmetrics > 0.7 | Use `torchmetrics==0.7.3` + direct loss computation |
| Rich progress bar crash | Use `rich==12.6.0` |

---

### Ego Normalization (Current Workaround)

**Current approach:** Simple scaling by `EGO_SCALE` (default 50.0 meters)
```python
ego = ego / self.ego_scale  # Roughly puts values in [-1, 1] range
```

**TODO:** Calculate proper ego mean/std from the training data:
```python
# Future: use proper normalization like motion
ego = (ego - ego_mean) / (ego_std + 1e-8)

## example code below: we cannot use previous method, as now mean is lateral, and we also dont wanna create combined folder crap
import numpy as np
import json
from glob import glob

# Collect all ego trajectories
all_ego = []
for json_file in glob("data/diffusion/*/train/*.json"):
    with open(json_file) as f:
        data = json.load(f)
    ego_3d = np.array(data["ego_in_ped_frame"])
    ego_2d = ego_3d[:, [0, 2]]  # x, z only
    all_ego.append(ego_2d)

all_ego = np.concatenate(all_ego, axis=0)  # (N_total, 2)

# Compute stats
ego_mean = all_ego.mean(axis=0)  # (2,)
ego_std = all_ego.std(axis=0)    # (2,)

print(f"Ego Mean: {ego_mean}")
print(f"Ego Std: {ego_std}")

# Save to your stats folder
np.save("path/to/your_stats/Ego_Mean.npy", ego_mean)
np.save("path/to/your_stats/Ego_Std.npy", ego_std)
```

The dataset class (`EgoMotion.py`) already supports `ego_mean` and `ego_std` parameters - they're just not being used yet. To enable:
1. Compute `ego_mean.npy` and `ego_std.npy` from training data
2. Set `DATASET.EGOMOTION.EGO_MEAN_STD_PATH` in config
3. Update `_normalize_ego()` in dataset to use them

---

### Not Yet Implemented

| Feature | Status | Notes |
|---------|--------|-------|
| Training | Done | Works with `--nodebug` flag |
| Validation | Done | Uses `test_diffusion_forward` |
| Ego mean/std normalization | Not Done | Using scaling workaround |
| Demo/Inference (`demo.py`) | Not Done | Needs ego-specific demo script |
| Visualization | Not Done | Need to adapt render scripts for ego input |
| Past pedestrian conditioning | Not Done | Phase 2 (see below) |

---

# Implementation Roadmap

#### Phase 1: Ego-Only Conditioning (CURRENT - In Progress)
- [x] EgoEncoder architecture
- [x] Dataset loading (waymo, nuscenes, ava)
- [x] Training pipeline with CFG
- [x] Validation pipeline
- [ ] Proper ego normalization (mean/std)
- [ ] Demo/inference script
- [ ] Visualization

#### Phase 2: Ego + Past Pedestrian Conditioning (NEXT)
After ego-only works, add past pedestrian motion as additional conditioning for **prediction mode**.

**Architecture (from plan):**
```
Ego Trajectory (B, T_ego, 2)     Past Ped Motion (B, T_obs, 263)
        │                                 │
        ▼                                 ▼
   EgoEncoder                    VAE Encoder (frozen)
        │                                 │
        ▼                                 ▼
   (B, T_ego, 256)                   (B, 1, 256)
        │                                 │
        └────────────┬────────────────────┘
                     │
                     ▼
              Concat (seq dim)
                     │
                     ▼
              (B, T_ego+1, 256) → Denoiser → Future Motion
```

**Key files to create:**
- `mld/models/architectures/ego_pastped_encoder.py` - Combined encoder
- `configs/config_ego_pastped.yaml` - Prediction mode config

**CFG strategy for dual conditioning:**
| Dropout Scenario | Result | Use Case |
|-----------------|--------|----------|
| Drop nothing | Ego + Past Ped | Full prediction |
| Drop past_ped only | Ego only | Generation mode |
| Drop ego only | Past Ped only | Regularization |
| Drop both | Unconditional | Random motion |

**Files to MODIFY:**

| File | Changes |
|------|---------|
| `mld/models/modeltype/mld.py` | Add `condition == 'ego_pastped'` handling in `__init__`, `train_diffusion_forward`, `test_diffusion_forward`, `forward` |
| `mld/models/architectures/mld_denoiser.py` | Add `'ego_pastped'` to condition checks (can reuse `'ego'` logic - same output shape) |
| `mld/data/EgoMotion.py` | Add `past_ped`, `past_ped_lengths` to `__getitem__` output; split motion into past (observation) and future (target) |
| `mld/data/get_data.py` | Minor update to pass prediction mode flag to dataset |

**Config updates for Phase 2:**
```yaml
# In config_ego_pastped.yaml
model:
  condition: 'ego_pastped'  # Instead of 'ego'
  
  ego_encoder:
    target: mld.models.architectures.ego_pastped_encoder.EgoPastPedEncoder
    params:
      ego_input_dim: 2
      latent_dim: 256
      guidance_uncondp_ego: 0.1
      guidance_uncondp_pastped: 0.2  # Higher dropout to help generation mode

DATASET:
  EGOMOTION:
    OBS_LEN: 20   # ~1 sec at 20fps observation
    PRED_LEN: 80  # ~4 sec prediction
```

Full implementation details can be discussed later.

---

### Quick Reference: Data Format

**JSON structure per sample:**
```json
{
  "scene_id": "string",
  "object_id": "string",
  "ego_in_ped_frame": [[x, y, z], ...],  // Uses (x, z) only - lateral motion
  "vectors_263": [[263 features], ...]    // HumanML3D format
}
```

**Batch structure:**
```python
batch = {
    "ego": (B, T_ego, 2),      # Normalized ego trajectory
    "motion": (B, T_motion, 263),  # Target motion
    "length": [int, ...],      # Motion lengths
    "ego_length": [int, ...],  # Ego lengths
    # Future (Phase 2):
    "past_ped": (B, T_obs, 263),
    "past_ped_lengths": [int, ...]
}
```