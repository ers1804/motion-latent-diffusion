# Research Log — Ego-Conditioned Pedestrian Motion Generation

## 2026-03-27 — Bootstrap

### State at Start
Two experiments actively running on helma H100 cluster:
1. **Diffusion training** (job 326049, ~22h): `ego_motion_diffusion_interaction_crop_weighted_1_helma`
   - Latent-4 stochastic VAE, interaction_crop + weighted_sampling
   - Epoch ~4200/5000, loss ≈ 0.302
2. **VAE training** (job 326050, ~22h): `ego_motion_vae_latent_8_wo_traj_interaction_crop_weighted_sampling`
   - Latent-8, KL annealing (0→500 epochs), cosine LR warmup
   - Epoch ~5699/6000, loss ≈ 0.0147

### Best Known Metrics
From `ego_motion_diffusion_interaction_crop_weighted_1` (crashed, latent-4, interaction-aware):
- FID: 7.5029
- Diversity: 5.6653 (GT: 5.3433)
- R-precision top-1/2/3: 0.8254 / 0.8917 / 0.9179

### Research Question Identified
Can interaction-aware training + larger latent spaces improve ego-conditioned motion generation?

### Key Design Decisions
- **Model**: MLD denoiser conditioned on EgoEncoderPooled (transformer + mean pool → 256d)
- **VAE**: MLD-VAE, motion in HumanML3D 263D representation
- **Data**: AVA + nuScenes + Waymo (interaction split)
- **Interaction features**: crop window to closest-approach frame + weight samples by interaction score

### Hypotheses Formed
- H2: interaction-aware training improves FID/diversity (currently running)
- H3: latent-8 VAE gives more capacity (VAE running, diffusion training TBD)
- H4: ego encoder architecture (cross-attn vs mean pool) improves R-precision
- H5: CFG guidance scale sweep (5/10/15/20) is cheap and might help
- H6: joint fine-tuning of VAE+diffusion improves alignment

### Next Steps
1. Monitor running jobs; evaluate once checkpoint ready
2. Literature search on pedestrian motion generation, CFG tuning, interaction-aware synthesis
3. Prepare guidance-scale sweep as next quick experiment (post checkpoint)

## 2026-03-27 — Eval Job Fix & Pipeline Preparation

### Issue: Eval Job 327929 Failed (0 samples loaded)
- Root cause: `TEST.SPLIT=test` used, but helma data archives only contain `train/` and `val/` directories
- `test.txt` and `val.txt` in the split list path are identical (11786 entries each)
- Fix: changed `TEST.SPLIT=test` → `TEST.SPLIT=val` in both eval scripts
- Committed and pushed; pulled on helma; submitted new eval job 327939

### Training Status Update (as of 2026-03-27 ~14:47 CET)
- **run_005 diffusion** (job 326049): epoch ~4500/5000, loss=0.292. ~6.5h remaining.
- **run_006 VAE latent-8** (job 326050): epoch ~5900/6000, loss=0.0143. ~1h remaining.

### Pipeline Ready for H3
- `configs/config_ego_motion_new_vae_stoch_latent_8.yaml` — latent-8 diffusion config
- `slurm/pretrain_ego_encoder_latent_8_helma.sh` — ego encoder pretraining
- `slurm/diffusion_training_latent_8_helma.sh` — H3 diffusion training (24h job)
- `slurm/eval_latent_8.sh` — evaluation script for H3 (prepared today)
- `slurm/eval_cfg_sweep.sh` — CFG sweep {5,10,15,20} on latent-4 model

### Next Actions
1. Eval job 327939 running — await FID/Diversity/R-Precision results
2. When VAE finishes: `sbatch pretrain_ego_encoder_latent_8_helma.sh` → `sbatch diffusion_training_latent_8_helma.sh`
3. After run_005 completes at epoch 5000: eval final checkpoint
4. Submit CFG sweep once run_005 eval confirms setup works

## 2026-03-27 — Ego Encoder Pretraining Fix & Resubmission

### Issue: Job 327946 Failed (FileNotFoundError)
- Error: `FileNotFoundError: '.../data/vae/mean_std_txt/ava_nuscenes_waymo/Mean.npy'`
- Root cause: `pretrain_ego_encoder.py` reads `DATASET.EGOMOTION.MEAN_STD_PATH` from config, which contains the local NAS path `/mnt/md0/erik/nas/...`. The slurm script extracts data to `$TMPDIR/data/` but didn't override this path.
- `pretrain_ego_encoder.py` had no mechanism to override `MEAN_STD_PATH` (only `--data_roots` existed)

### Fix Applied
- Added `--mean_std_path` CLI argument to `pretrain_ego_encoder.py`
- `build_cfg()` now overrides both `DATASET.EGOMOTION.MEAN_STD_PATH` and `DATASET.EGOMOTION.EGO_MEAN_STD_PATH` when `--mean_std_path` is provided
- Updated `slurm/pretrain_ego_encoder_latent_8_helma.sh` to pass `--mean_std_path "$TMPDIR/data/vae/mean_std_txt/ava_nuscenes_waymo"`
- Committed (2922363), pushed, pulled on helma, resubmitted as **job 327955**

### Current Job Status
- **327955** ego_enc_pretrain — RUNNING (just started, h13-13)
- **327950** eval_cfg_sweep — RUNNING (extracting data, h13-24)
- **326049** diffusion_train (H2 run_005) — RUNNING (23h51m, h14-06)

## 2026-03-30 — Inner Loop Results: FID Regression + H3 Pipeline Started

### Key Results from Completed Jobs (2026-03-27)

**Job 327955 (Ego encoder latent-8)**: COMPLETED. 200/200 epochs. `best.pt` saved.
Cosine similarity ~0.92 at epoch 200. H3 pipeline fully ready.

**Job 327991 (Eval H2 epoch=4599, CFG=5)**: COMPLETED. REGRESSION DETECTED.
- FID: 8.400 ± 0.102
- R-prec@1: 0.733
- Diversity: 5.757
- This is WORSE than epoch=4399 (FID=6.603). Training loss didn't diverge (~0.286), so FID oscillation is independent of loss. **Epoch=4399 is the best H2 checkpoint.**

**Job 327990 (H2 resume epoch 4599→4999)**: COMPLETED. Final epoch=4999, loss=0.286.
- NOTE: Job 327881 was an untracked duplicate resume from the same checkpoint, running in parallel.
  This caused PyTorch Lightning to append "-v1" suffixes to checkpoints. The duplicate training
  was submitted by someone (possibly user manually) and ran 15:22-19:34 on 2026-03-27.
  Checkpoint integrity may be affected.

### New Submissions (2026-03-30)

- **Job 329788**: Eval H2 epoch=4999 at CFG=5 — checks if model recovered from epoch=4599 regression
- **Job 329789**: H3 diffusion training (latent-8 VAE + new ego encoder) — the key H3 experiment

### Outer Loop Reflection

Three major findings from this research session:
1. **CFG=5 is optimal for FID** (10.8% improvement vs CFG=15)
2. **Epoch=4399 is the best H2 checkpoint** — model degraded from epoch 4399 → 4599 → unclear for 4999
3. **H3 pipeline is running** — ego encoder (cos~0.92) proves latent-8 encoder alignment works

The H2 FID regression raises an important question about training stability. The loss plateau (~0.287) doesn't predict sampling quality oscillation. For future training runs, we should save more frequent checkpoints and run periodic FID validation to catch the best model.

## 2026-03-30 — H2 Regression Confirmed: Epoch=4999 Result

**Job 329788 (Eval H2 epoch=4999, CFG=5)**: COMPLETED.
- FID: 7.510 ± 0.084
- R-prec@1: 0.671
- Diversity: 5.816
- MultiModality: 3.613 ± 0.483

**FID regression is CONFIRMED and PERMANENT**: The model did NOT recover to epoch=4399 levels.
- epoch=4399: FID=6.603 ← BEST (confirmed baseline)
- epoch=4599: FID=8.400 ← regression
- epoch=4999: FID=7.510 ← partial recovery, still 13.7% worse than best

The oscillation appears persistent: even after 400 more training epochs, the model settled at FID=7.51, not recovering to 6.60. **Conclusion: epoch=4399 is definitively the best H2 checkpoint.**

**H3 training (job 329789)**: Confirmed running on h13-23. Training started at ~08:42 2026-03-30.
No checkpoints saved yet (~30 min elapsed). 24h wall time. First checkpoint expected ~epoch 100-200.

### Current Research State
- H2 DONE: best FID=6.603 (epoch=4399, CFG=5) — this is the H2 baseline
- H3 RUNNING: 24h training on latent-8 model — critical experiment
- H4 (cross-attn encoder): planned, untested
- Key question: will H3 (latent-8) beat H2's FID=6.603?

## 2026-03-30 — CFG=7 Result + H3 Timing Discovery

### CFG=7 Eval (Job 329801): COMPLETED

FID=6.716 ± 0.068, R-prec@1=0.724, Diversity=5.771, MM=3.291 ± 0.435

**Full CFG curve now complete (H2 epoch=4399):**
- CFG=5: FID=6.603, R-prec@1=0.671 (best FID)
- CFG=7: FID=6.716, R-prec@1=0.724 ← **best FID/R-prec tradeoff**
- CFG=10: FID=6.963, R-prec@1=0.767
- CFG=15: FID=7.400, R-prec@1=0.797 (prior default)
- CFG=20: FID=7.856, R-prec@1=0.815

CFG=7 is the recommended operating point: only 1.7% FID cost vs CFG=5, but 7.9% better R-prec alignment.

### H3 Timing Issue Discovered

H3 latent-8 training is running at **~51.6 seconds/epoch** vs H2's ~17.6s/epoch — 3x slower.
At 24h wall time: ~1674 epochs maximum. H2's best checkpoint was epoch=4399.
**H3 needs ~3× 24h job segments** to reach comparable training depth.

Actions taken:
- Created `resume_h3_diffusion_helma.sh` for chaining job segments
- Updated eval script to target epoch=4399 checkpoint (may need intermediate evals)
- Plan: eval at epoch~1599 (end of first segment), resume for second 24h segment

### Job 329818 (unrelated)
Confirmed: "h7_lower_peak_lr" job is from workspace `v103fe12-jepa` (JEPA scene encoder project) — a different project under the same user account. Not related to this research.

## 2026-03-31 — H3 Segment-1 Complete: epoch=2299

### H3 Training Segment-1 (Job 329789): FINISHED

Job completed at ~08:23 on 2026-03-31 (started 08:42 on 2026-03-30, ~23.7h elapsed).

**Key facts:**
- Final checkpoint: `epoch=2299.ckpt`, loss=0.309
- **Actual epoch rate: ~37s/epoch** (faster than estimated 51.6s/epoch — prior estimate was wrong)
- Revised timing: 24h → ~2340 epochs per segment (not ~1674)
- H3 now needs only **2 segments total** to reach epoch~4639 (past H2's best at 4399)

**Revised H3 plan:**
- Segment 1 (job 329789): done, epoch=2299
- Segment 2 (job 330925): running, resume from epoch=2299, expected final ~epoch=4639
- One more eval at epoch~4399 will give the fair H2 comparison

### Actions Taken
1. Updated `eval_h3_cfg5_helma.sh` EPOCH to 2299
2. Submitted **H3 intermediate eval**: job 330924 (epoch=2299, CFG=5), running on h13-15
3. Submitted **H3 segment-2 training**: job 330925, running on h13-23, resume from epoch=2299

### Open Question
Will H3 (latent-8, epoch=2299) show meaningful FID improvement over H2 baseline (FID=6.603)?
At only 52% of H2's best epoch depth, we expect H3 to be undertrained — but will show training trajectory.

## 2026-04-01 — H3 Definitive Eval: REJECTED; Direction Set to H4

### H3 Definitive Eval Results (epoch=4399, jobs 333132/333133)

Both eval jobs completed in ~5 minutes each. Results:

| Config | FID | FID CI | R-prec@1 | Diversity | MM |
|--------|-----|--------|----------|-----------|-----|
| H3 CFG=5 | 7.563 | ±0.078 | 0.676 | 5.842 | 3.245 |
| H3 CFG=7 | 8.131 | ±0.103 | 0.733 | 5.830 | 2.989 |
| H2 CFG=5 (baseline) | 6.603 | ±0.067 | 0.671 | 5.779 | 3.503 |
| H2 CFG=7 | 6.716 | ±0.068 | 0.724 | 5.771 | 3.291 |

**H3 HYPOTHESIS REJECTED**: Latent-8 is 14.5% worse FID than latent-4 at the same epoch. The same-capacity denoiser cannot model an 8D latent distribution as well as 4D.

Key observation: R-prec@1 is essentially identical (H3=0.676 vs H2=0.671 at CFG=5). This tells us the bottleneck for conditioning quality is NOT the latent dimension — it's the ego encoder architecture.

### Outer Loop Reflection: Direction → H4

With H3 rejected, the next hypothesis is H4 — cross-attention ego conditioning:
- **What**: Use `trans_dec` arch in the diffusion denoiser so the latent `z` queries the full T=196 ego sequence via cross-attention (instead of a single mean-pooled token)
- **Why**: H3 showed R-prec is decoupled from latent dim; EgoEncoderPooled (196→1 token) compresses away temporal structure
- **Prediction**: R-prec@1 improves (>0.700 at CFG=5); FID may also improve via better temporal alignment
- **Cost**: ~1 day ego encoder pretraining + ~2 days diffusion training (latent-4, same as H2)

The `trans_dec` arch already exists in `mld_denoiser.py:130`. H4 requires:
1. Pre-train `EgoEncoder` (no pooling, full sequence) with contrastive loss
2. Train MLD with `arch=trans_dec`, passing full ego sequence as K/V

### Current State
- **H2** (best): FID=6.603, epoch=4399, CFG=5 — confirmed best latent-4 result
- **H3** (rejected): latent-8 is consistently worse
- **H4** (next): cross-attention ego conditioning — implementation planning pending

## 2026-04-02 — H4 Implementation: Code Changes + Protocol Commit + Job Submitted

### Code Changes for H4

Three files modified to support `EgoEncoder` (full T=196 sequence, no pooling):

1. **`pretrain_ego_encoder.py`** (×2 fixes): `squeeze(1)` → `mean(dim=1)` for EgoEncoder compatibility.
   - `squeeze(1)` only works for T=1 (EgoEncoderPooled). For EgoEncoder, T=196, squeeze does nothing → wrong shape (B, 196, 256) instead of (B, 256).
   - `mean(dim=1)` works for both T=1 (EgoEncoderPooled) and T=196 (EgoEncoder) — backward-compatible.

2. **`mld/models/modeltype/mld.py:1002`**: `squeeze(1)` → `mean(dim=1)` for `ego_emb` in `rs_set`.
   - Same issue: R-precision metric needs `ego_emb` to be (B, 256) for cosine similarity.

3. **New files created**:
   - `configs/config_ego_motion_new_vae_stoch_latent_4_trans_dec.yaml`: H4 config (EgoEncoder target, arch=trans_dec, same latent-4 VAE as H2)
   - `slurm/pretrain_ego_encoder_h4_helma.sh`: 4h ego encoder pretraining job
   - `slurm/diffusion_training_h4_trans_dec_helma.sh`: 24h diffusion training segment-1
   - `slurm/resume_h4_diffusion_helma.sh`: segment-2 resume
   - `slurm/eval_h4_cfg5_helma.sh`, `slurm/eval_h4_cfg7_helma.sh`: eval at epoch=4399

### Architectural Detail: trans_dec with EgoEncoder

H4 changes the conditioning from 2 K/V tokens (time_emb + 1 mean-pooled ego token) to 197 K/V tokens (time_emb + 196 ego timestep tokens). The denoiser latent `z` (queries) now attends to the full temporal ego sequence at every denoising step. This preserves spatial-temporal structure that EgoEncoderPooled discards.

### Protocol Commit + Job Submitted

- Protocol committed: `f63c596 research(protocol): H4 cross-attention ego conditioning (trans_dec arch)`
- Pushed to `local-dev-erik`
- Submitted H4 ego encoder pretraining: **job 343502** (helma, 4h, h100)
  - Output: `/hnvme/workspace/v103fe12-ped_gen/models/ego_encoder/ego_encoder_h4_trans_dec/checkpoints/best.pt`

### Plan After Pretraining
1. Wait for job 343502 to complete (~4h)
2. Submit `diffusion_training_h4_trans_dec_helma.sh` for segment-1 (~24h, ~2300 epochs)
3. After segment-1: eval at intermediate checkpoint, resume for segment-2
4. Eval at epoch~4399: compare R-prec@1 and FID vs H2 baseline

## 2026-04-02 — H4 Pretraining Fix: NAS Not Mounted on All Helma Nodes

### Job 343502 Failed (3 min)

```
FileNotFoundError: /mnt/md0/erik/nas/methods/methods/diffusion_gen/models/vae/
  ego_motion_vae_latent_4_wo_traj_interaction_crop_weighted_sampling/checkpoints/epoch=5999.ckpt
```

The NAS mount `/mnt/md0/erik/nas` is not accessible from node h11-02. The H4 config had the NAS
path for `PRETRAINED_VAE`. The helma workspace already has the same checkpoint at:
`/hnvme/workspace/v103fe12-ped_gen/models/vae/ego_motion_vae_latent_4_wo_traj_interaction_crop_weighted_sampling/checkpoints/epoch=5999.ckpt`

**Fix**: Updated `configs/config_ego_motion_new_vae_stoch_latent_4_trans_dec.yaml` to use helma path.
This also fixes the H4 diffusion training which uses the same config.

**Resubmitted**: Job **343503** (same script, fixed config). Expected completion ~4h.

**Lesson**: Always use `/hnvme/` paths in configs for helma jobs. NAS paths are node-dependent.
