#!/usr/bin/env python3
"""Generate definitive-eval SLURM scripts for an in-domain text model as a CFG sweep.

Usage:  python slurm/review/make_text_eval_sweep.py <vocab: ego|egoonly> <epoch> [cfgs]
  e.g.  python slurm/review/make_text_eval_sweep.py egoonly 3099 1,2.5,5,10

Clones slurm/review/rv_eval_h2_nopipe.sh (the proven helma eval pattern) and adds the
text-model overrides mirrored from the training scripts. R-Precision is UNDEFINED for text
models by design (dim-1 placeholder -> 0.0); do not report it. Then: git push BEFORE sbatch.
"""
import os, re, sys
vocab, epoch = sys.argv[1], int(sys.argv[2])
cfgs = (sys.argv[3] if len(sys.argv) > 3 else "1,2.5,5,10").split(",")
assert vocab in ("ego", "egoonly")
W = "/hnvme/workspace/v103fe12-ped_gen"; REPO = f"{W}/motion-latent-diffusion"
run = f"ego_motion_diffusion_text_{vocab}"
src = open("slurm/review/rv_eval_h2_nopipe.sh").read()
for cfg in cfgs:
    tag = cfg.replace(".", "p"); name = f"rv_eval_text_{vocab}_ep{epoch}_cfg{tag}"
    s = src.replace("rv_eval_h2_nopipe", name)
    s = re.sub(r"^CHECKPOINT=.*$", f"CHECKPOINT={W}/models/mld/{run}/checkpoints/epoch={epoch}.ckpt", s, flags=re.M)
    s = re.sub(r"--cfg \S+", "--cfg configs/config_ego_motion_new_vae_stoch_latent_4_trans_dec.yaml", s)
    s = re.sub(r'"model\.guidance_scale=[0-9.]+"', f'"model.guidance_scale={cfg}"', s)
    s = s.replace("    \"METRIC.TYPE=['EgoMotionMetrics']\" \\",
                  "    \"METRIC.TYPE=['EgoMotionMetrics']\" \\\n"
                  "    \"model.latent_dim=[4, 256]\" \\\n"
                  "    \"model.condition=text\" \\\n"
                  "    \"model.denoiser.params.text_encoded_dim=768\" \\\n"
                  f"    \"DATASET.EGOMOTION.CAPTIONS_PATH={REPO}/research/data/synth_captions.json\" \\\n"
                  f"    \"DATASET.EGOMOTION.CAPTION_VOCAB={vocab}\" \\")
    assert "condition=text" in s and f"guidance_scale={cfg}" in s and f"epoch={epoch}" in s and f"NAME={name}" in s
    p = f"slurm/review/{name}.sh"; open(p, "w").write(s); os.chmod(p, 0o755); print("written", p)
