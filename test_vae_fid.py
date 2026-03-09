"""
VAE-only FID: encode GT motions through the VAE and decode them,
then compute FID / Diversity / R-Precision using EgoMotionMetrics.

This establishes the reconstruction floor — the diffusion model
cannot do better than this.

Usage:
    python test_vae_fid.py --cfg configs/config_ego_motion_new_vae_stoch.yaml \
                           --cfg_assets configs/assets.yaml
"""

import os
import json
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from rich import get_console
from rich.table import Table
from omegaconf import OmegaConf

from mld.config import parse_args
from mld.data.get_data import get_datasets
from mld.models.get_model import get_model
from mld.utils.logger import create_logger


def print_table(title, metrics):
    table = Table(title=title)
    table.add_column("Metrics", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    for key, value in metrics.items():
        table.add_row(key, str(value))
    console = get_console()
    console.print(table, justify="center")


def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def main():
    cfg = parse_args(phase="test")
    cfg.FOLDER = cfg.TEST.FOLDER
    logger = create_logger(cfg, phase="test")
    output_dir = Path(
        os.path.join(cfg.FOLDER, str(cfg.model.model_type), str(cfg.NAME),
                     "vae_fid_" + cfg.TIME))
    output_dir.mkdir(parents=True, exist_ok=True)

    pl.seed_everything(cfg.SEED_VALUE)

    if cfg.ACCELERATOR == "gpu":
        os.environ["PYTHONWARNINGS"] = "ignore"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    datasets = get_datasets(cfg, logger=logger, phase="test")[0]
    logger.info("datasets module {} initialized".format("".join(cfg.TRAIN.DATASETS)))

    model = get_model(cfg, datasets)
    logger.info("model {} loaded".format(cfg.model.model_type))

    # Load checkpoint
    logger.info("Loading checkpoints from {}".format(cfg.TEST.CHECKPOINTS))
    state_dict = torch.load(cfg.TEST.CHECKPOINTS, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict)

    device = torch.device("cuda" if cfg.ACCELERATOR == "gpu" else "cpu")
    model = model.to(device)
    model.eval()

    # Monkey-patch ego_eval to use VAE encode→decode instead of diffusion
    original_ego_eval = model.ego_eval

    def vae_only_ego_eval(batch):
        lengths = batch["length"]
        feats_ref = batch["motion"].detach()

        # VAE encode → decode (the reconstruction path)
        with torch.no_grad():
            z, dist_m = model.vae.encode(feats_ref, lengths)
            feats_rst = model.vae.decode(z, lengths)

        # Re-encode the reconstruction to get its latent
        with torch.no_grad():
            recons_z, dist_rm = model.vae.encode(feats_rst, lengths)

        joints_rst = model.feats2joints(feats_rst)
        joints_ref = model.feats2joints(feats_ref)

        # Ego embedding (still from the real ego trajectory)
        ego = batch["ego"]
        if model.do_classifier_free_guidance:
            uncond_ego = torch.zeros_like(ego)
            ego_cat = torch.cat([uncond_ego, ego], dim=0)
        else:
            ego_cat = ego
        cond_emb = model.ego_encoder(ego_cat)
        ego_emb = cond_emb[-len(lengths):].squeeze(1).detach()

        rs_set = {
            "m_rst": feats_rst,
            "m_ref": feats_ref,
            "lat_t": z.permute(1, 0, 2),
            "lat_m": z.permute(1, 0, 2),      # GT latent
            "lat_rm": recons_z.permute(1, 0, 2),  # Reconstruction latent
            "joints_rst": joints_rst,
            "joints_ref": joints_ref,
            "ego_emb": ego_emb,
        }

        # t2m motion encoder embeddings for FID/Diversity
        m_lens = torch.tensor(lengths, device=feats_rst.device) // 4
        m_lens = torch.clamp(m_lens, min=1)
        m_lens_sorted, sort_idx = m_lens.sort(descending=True)
        unsort_idx = sort_idx.argsort()

        with torch.no_grad():
            recons_mov = model.t2m_moveencoder(feats_rst[..., :-4]).detach()
            lat_rm_sorted = model.t2m_motionencoder(recons_mov[sort_idx], m_lens_sorted)
            rs_set["t2m_lat_rm"] = lat_rm_sorted[unsort_idx]

            motion_mov = model.t2m_moveencoder(feats_ref[..., :-4]).detach()
            lat_m_sorted = model.t2m_motionencoder(motion_mov[sort_idx], m_lens_sorted)
            rs_set["t2m_lat_m"] = lat_m_sorted[unsort_idx]

        # Mean-pool VAE latent for R-precision
        rs_set["motion_lat_pooled"] = recons_z.permute(1, 0, 2).mean(dim=1).detach()

        return rs_set

    # Patch the model
    model.ego_eval = vae_only_ego_eval

    # Use PL trainer for test loop (same as test.py)
    callbacks = [pl.callbacks.RichProgressBar()]
    trainer = pl.Trainer(
        benchmark=False,
        max_epochs=cfg.TRAIN.END_EPOCH,
        accelerator=cfg.ACCELERATOR,
        devices=list(range(len(cfg.DEVICE))),
        default_root_dir=cfg.FOLDER_EXP,
        reload_dataloaders_every_n_epochs=1,
        deterministic=False,
        detect_anomaly=False,
        enable_progress_bar=True,
        logger=None,
        callbacks=callbacks,
    )

    all_metrics = {}
    replication_times = cfg.TEST.REPLICATION_TIMES

    for i in range(replication_times):
        metrics_type = ", ".join(cfg.METRIC.TYPE)
        logger.info(f"[VAE-only] Evaluating {metrics_type} - Replication {i}")
        metrics = trainer.test(model, datamodule=datasets)[0]

        if "EgoMotionMetrics" in metrics_type:
            logger.info(f"[VAE-only] Evaluating MultiModality - Replication {i}")
            datasets.mm_mode(True)
            mm_metrics = trainer.test(model, datamodule=datasets)[0]
            metrics.update(mm_metrics)
            datasets.mm_mode(False)

        for key, item in metrics.items():
            if key not in all_metrics:
                all_metrics[key] = [item]
            else:
                all_metrics[key] += [item]

    all_metrics_new = {}
    for key, item in all_metrics.items():
        mean, conf_interval = get_metric_statistics(np.array(item), replication_times)
        all_metrics_new[key + "/mean"] = mean
        all_metrics_new[key + "/conf_interval"] = conf_interval

    print_table("VAE-only FID (Reconstruction Floor)", all_metrics_new)
    all_metrics_new.update(all_metrics)

    metric_file = output_dir.parent / f"vae_fid_{cfg.TIME}.json"
    with open(metric_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics_new, f, indent=4)
    logger.info(f"VAE-only FID done, metrics saved to {str(metric_file)}")


if __name__ == "__main__":
    main()
