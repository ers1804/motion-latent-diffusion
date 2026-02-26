"""
Baseline 1: Unconditional MLD

Runs the trained ego-conditioned diffusion model with all ego trajectories
replaced by zeros.  Because the model was trained with 10 % null-ego dropout
(guidance_uncondp = 0.1), it has a well-defined unconditional prior.  With
CFG enabled (guidance_scale > 1), both the "cond" and "uncond" denoiser calls
receive the same zero-ego embedding, so classifier-free guidance cancels out
and the output is the pure unconditional sample.

Purpose: isolate the effect of ego conditioning by comparing FID / Diversity
         against the model trained with ego conditioning.

Usage:
    python baselines/eval_uncond.py \
        --cfg configs/config_ego_motion_baseline_uncond.yaml \
        --cfg_assets configs/assets.yaml
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mld.callback import ProgressLogger
from mld.config import parse_args
from mld.data.get_data import get_datasets
from mld.models.get_model import get_model
from mld.utils.logger import create_logger
from baselines.baseline_utils import print_metrics, save_metrics


# ---------------------------------------------------------------------------
# Zero-ego patches
# ---------------------------------------------------------------------------

def _patch_zero_ego(model):
    """
    Monkeypatch test_diffusion_forward and ego_eval so that batch["ego"]
    is zeroed before any encoding takes place.
    """
    _orig_tdf = model.test_diffusion_forward
    _orig_ee = model.ego_eval

    def _zero_ego_tdf(batch, **kwargs):
        batch = dict(batch)
        batch["ego"] = torch.zeros_like(batch["ego"])
        return _orig_tdf(batch, **kwargs)

    def _zero_ego_ee(batch, **kwargs):
        batch = dict(batch)
        batch["ego"] = torch.zeros_like(batch["ego"])
        return _orig_ee(batch, **kwargs)

    model.test_diffusion_forward = _zero_ego_tdf
    model.ego_eval = _zero_ego_ee


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = parse_args(phase="test")
    cfg.FOLDER = cfg.TEST.FOLDER

    logger = create_logger(cfg, phase="test")
    output_dir = Path(
        os.path.join(cfg.FOLDER, str(cfg.model.model_type), str(cfg.NAME),
                     "samples_" + cfg.TIME)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.SEED_VALUE)

    if cfg.ACCELERATOR == "gpu":
        os.environ["PYTHONWARNINGS"] = "ignore"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    datasets = get_datasets(cfg, logger=logger, phase="test")[0]
    model = get_model(cfg, datasets)
    logger.info("model {} loaded".format(cfg.model.model_type))

    metric_monitor = {
        "APE root": "Metrics/APE_root",
        "APE mean pose": "Metrics/APE_mean_pose",
    }
    callbacks = [
        pl.callbacks.RichProgressBar(),
        ProgressLogger(metric_monitor=metric_monitor),
    ]

    trainer = pl.Trainer(
        benchmark=False,
        max_epochs=cfg.TRAIN.END_EPOCH,
        accelerator=cfg.ACCELERATOR,
        devices=list(range(len(cfg.DEVICE))),
        default_root_dir=cfg.FOLDER_EXP,
        reload_dataloaders_every_n_epochs=1,
        log_every_n_steps=cfg.LOGGER.LOG_EVERY_STEPS,
        deterministic=False,
        detect_anomaly=False,
        enable_progress_bar=True,
        logger=None,
        callbacks=callbacks,
    )

    logger.info(
        "Loading checkpoint from {}".format(cfg.TEST.CHECKPOINTS)
    )
    state_dict = torch.load(cfg.TEST.CHECKPOINTS, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict)

    # --- Apply zero-ego patch AFTER weights are loaded ---
    _patch_zero_ego(model)
    logger.info("Zero-ego patch applied: model will generate unconditionally.")

    all_metrics: dict = {}
    replication_times = cfg.TEST.REPLICATION_TIMES

    for i in range(replication_times):
        metrics_type = ", ".join(cfg.METRIC.TYPE)
        logger.info(f"Evaluating {metrics_type} - Replication {i}")
        metrics = trainer.test(model, datamodule=datasets)[0]

        if "EgoMotionMetrics" in metrics_type:
            logger.info(f"Evaluating MultiModality - Replication {i}")
            datasets.mm_mode(True)
            mm_metrics = trainer.test(model, datamodule=datasets)[0]
            metrics.update(mm_metrics)
            datasets.mm_mode(False)

        for key, item in metrics.items():
            all_metrics.setdefault(key, []).append(item)

    # Aggregate
    all_metrics_new = {}
    for key, values in all_metrics.items():
        arr = np.array(values)
        mean = np.mean(arr)
        ci = 1.96 * np.std(arr) / np.sqrt(replication_times)
        all_metrics_new[key + "/mean"] = float(mean)
        all_metrics_new[key + "/conf_interval"] = float(ci)

    print_metrics(all_metrics_new, title="Baseline: Unconditional MLD")

    metric_file = output_dir.parent / f"metrics_{cfg.TIME}.json"
    all_metrics_new.update({k: list(v) for k, v in all_metrics.items()})
    save_metrics(all_metrics_new, str(metric_file))
    logger.info(f"Done. Metrics saved to {metric_file}")


if __name__ == "__main__":
    main()
