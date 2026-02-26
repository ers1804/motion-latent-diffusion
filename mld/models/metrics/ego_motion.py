from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Metric

from .utils import (
    calculate_activation_statistics_np,
    calculate_frechet_distance_np,
    calculate_diversity_np,
    euclidean_distance_matrix,
    calculate_top_k,
)


class EgoMotionMetrics(Metric):
    """
    FID, Diversity, and R-Precision for ego-conditioned motion generation.

    - FID / Diversity: computed in 512D space from the pretrained HumanML3D
      t2m motion encoder (same motion representation, compatible weights).
    - R-Precision: vehicle-motion alignment accuracy in the shared 256D VAE
      latent space (mean-pooled generated motion latent vs ego encoder output).
      Uses cosine similarity via L2-normalized euclidean distance.
    """

    full_state_update = True

    def __init__(
        self,
        top_k: int = 3,
        R_size: int = 32,
        diversity_times: int = 300,
        dist_sync_on_step: bool = True,
        **kwargs,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.name = "fid, diversity, and r-precision for ego-conditioned generation"
        self.top_k = top_k
        self.R_size = R_size
        self.diversity_times = diversity_times

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq", default=torch.tensor(0), dist_reduce_fx="sum")

        self.metrics = []

        # FID
        self.add_state("FID", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.metrics.append("FID")

        # Diversity
        self.add_state("Diversity", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("gt_Diversity", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.metrics.extend(["Diversity", "gt_Diversity"])

        # R-Precision (ego-motion alignment, Top-1 … Top-top_k)
        for k in range(1, top_k + 1):
            self.add_state(
                f"R_precision_top_{k}",
                default=torch.tensor(0.0),
                dist_reduce_fx="sum",
            )
            self.metrics.append(f"R_precision_top_{k}")

        # Cached embeddings accumulated across batches
        self.add_state("recmotion_embeddings", default=[], dist_reduce_fx=None)
        self.add_state("gtmotion_embeddings", default=[], dist_reduce_fx=None)
        self.add_state("ego_embeddings", default=[], dist_reduce_fx=None)
        self.add_state("motion_latents", default=[], dist_reduce_fx=None)

    def update(
        self,
        recmotion_embeddings: Tensor,  # (B, 512) from t2m_motionencoder — generated
        gtmotion_embeddings: Tensor,   # (B, 512) from t2m_motionencoder — GT
        motion_latents: Tensor,        # (B, 256) mean-pooled VAE latent — generated
        ego_embeddings: Tensor,        # (B, 256) ego encoder output
        lengths: List[int],
    ):
        self.count += sum(lengths)
        self.count_seq += len(lengths)

        self.recmotion_embeddings.append(recmotion_embeddings.detach())
        self.gtmotion_embeddings.append(gtmotion_embeddings.detach())
        self.motion_latents.append(motion_latents.detach())
        self.ego_embeddings.append(ego_embeddings.detach())

    def compute(self, sanity_flag):
        count_seq = self.count_seq.item()

        metrics = {metric: getattr(self, metric) for metric in self.metrics}

        if sanity_flag:
            return metrics

        # Shuffle to avoid order bias
        shuffle_idx = torch.randperm(count_seq)
        all_gen = torch.cat(self.recmotion_embeddings, dim=0).cpu()[shuffle_idx]   # (N, 512)
        all_gt = torch.cat(self.gtmotion_embeddings, dim=0).cpu()[shuffle_idx]     # (N, 512)
        all_ego = torch.cat(self.ego_embeddings, dim=0).cpu()[shuffle_idx]         # (N, 256)
        all_mot = torch.cat(self.motion_latents, dim=0).cpu()[shuffle_idx]         # (N, 256)

        # ── FID ──────────────────────────────────────────────────────────────
        gen_np = all_gen.numpy()
        gt_np = all_gt.numpy()
        mu_gen, cov_gen = calculate_activation_statistics_np(gen_np)
        mu_gt, cov_gt = calculate_activation_statistics_np(gt_np)
        metrics["FID"] = calculate_frechet_distance_np(mu_gt, cov_gt, mu_gen, cov_gen)

        # ── Diversity ─────────────────────────────────────────────────────────
        assert count_seq > self.diversity_times, (
            f"Need > {self.diversity_times} test samples for Diversity, got {count_seq}. "
            "Reduce TEST.DIVERSITY_TIMES or use a larger test set."
        )
        metrics["Diversity"] = calculate_diversity_np(gen_np, self.diversity_times)
        metrics["gt_Diversity"] = calculate_diversity_np(gt_np, self.diversity_times)

        # ── R-Precision (vehicle-motion alignment) ────────────────────────────
        # L2-normalize both embeddings so euclidean distance ∝ cosine distance.
        ego_norm = F.normalize(all_ego, dim=1)  # (N, 256)
        mot_norm = F.normalize(all_mot, dim=1)  # (N, 256)

        assert count_seq > self.R_size, (
            f"Need > {self.R_size} test samples for R-Precision, got {count_seq}."
        )
        top_k_mat = torch.zeros((self.top_k,))
        for i in range(count_seq // self.R_size):
            group_ego = ego_norm[i * self.R_size:(i + 1) * self.R_size]   # (R, 256)
            group_mot = mot_norm[i * self.R_size:(i + 1) * self.R_size]   # (R, 256)
            # Lower euclidean distance = better alignment
            dist_mat = euclidean_distance_matrix(group_ego, group_mot).nan_to_num()  # (R, R)
            argsmax = torch.argsort(dist_mat, dim=1)  # ascending
            top_k_mat += calculate_top_k(argsmax, top_k=self.top_k).sum(axis=0)

        R_count = count_seq // self.R_size * self.R_size
        for k in range(self.top_k):
            metrics[f"R_precision_top_{k + 1}"] = top_k_mat[k] / R_count

        return {**metrics}
