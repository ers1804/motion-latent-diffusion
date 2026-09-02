import inspect
import os
from mld.transforms.rotation2xyz import Rotation2xyz
import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchmetrics import MetricCollection
import time
from mld.config import instantiate_from_config
from os.path import join as pjoin
from mld.models.architectures import (
    mld_denoiser,
    mld_vae,
    vposert_vae,
    t2m_motionenc,
    t2m_textenc,
    vposert_vae,
)
from mld.models.losses.mld import MLDLosses
from mld.models.modeltype.base import BaseModel
from mld.utils.temos_utils import remove_padding, lengths_to_mask

from .base import BaseModel


class MLD(BaseModel):
    """
    Stage 1 vae
    Stage 2 diffusion
    """

    def __init__(self, cfg, datamodule, **kwargs):
        super().__init__()

        self.cfg = cfg

        self.stage = cfg.TRAIN.STAGE
        self.condition = cfg.model.condition
        self.is_vae = cfg.model.vae
        self.predict_epsilon = cfg.TRAIN.ABLATION.PREDICT_EPSILON
        self.nfeats = cfg.DATASET.NFEATS
        self.njoints = cfg.DATASET.NJOINTS
        self.debug = cfg.DEBUG
        self.latent_dim = cfg.model.latent_dim
        self.guidance_scale = cfg.model.guidance_scale
        self.guidance_uncodp = cfg.model.guidance_uncondp
        self.deterministic_z = getattr(cfg.TRAIN, 'DETERMINISTIC_Z', False)
        self.datamodule = datamodule

        try:
            self.vae_type = cfg.model.vae_type
        except:
            self.vae_type = cfg.model.motion_vae.target.split(
                ".")[-1].lower().replace("vae", "")

        # Initialize condition encoder based on condition type
        if self.condition in ["text", "text_uncond"]:
            self.text_encoder = instantiate_from_config(cfg.model.text_encoder)
        elif self.condition in ["ego"]:
            self.ego_encoder = instantiate_from_config(cfg.model.ego_encoder)

        if self.vae_type != "no":
            self.vae = instantiate_from_config(cfg.model.motion_vae)

        # Don't train the motion encoder and decoder
        if self.stage == "diffusion":
            if self.vae_type in ["mld", "vposert","actor"]:
                self.vae.training = False
                for p in self.vae.parameters():
                    p.requires_grad = False
            elif self.vae_type == "no":
                pass
            else:
                self.motion_encoder.training = False
                for p in self.motion_encoder.parameters():
                    p.requires_grad = False
                self.motion_decoder.training = False
                for p in self.motion_decoder.parameters():
                    p.requires_grad = False

        self.denoiser = instantiate_from_config(cfg.model.denoiser)
        if self.predict_epsilon:
            cfg.model.scheduler.params['prediction_type'] = 'epsilon'
            cfg.model.noise_scheduler.params['prediction_type'] = 'epsilon'
        else:
            cfg.model.scheduler.params['prediction_type'] = 'sample'
            cfg.model.noise_scheduler.params['prediction_type'] = 'sample'
        self.scheduler = instantiate_from_config(cfg.model.scheduler)
        self.noise_scheduler = instantiate_from_config(
            cfg.model.noise_scheduler)

        if self.condition in ["text", "text_uncond"] or "EgoMotionMetrics" in cfg.METRIC.TYPE:
            self._get_t2m_evaluator(cfg)

        if cfg.TRAIN.OPTIM.TYPE.lower() == "adamw":
            self.optimizer = AdamW(lr=cfg.TRAIN.OPTIM.LR,
                                   params=self.parameters())
        else:
            raise NotImplementedError(
                "Do not support other optimizer for now.")

        # LR Scheduler
        self.lr_scheduler = None
        sched_cfg = getattr(cfg.TRAIN.OPTIM, 'LR_SCHEDULER', None)
        if sched_cfg and getattr(sched_cfg, 'TYPE', '') == 'CosineAnnealingWarmup':
            warmup_epochs = getattr(sched_cfg, 'WARMUP_EPOCHS', 0)
            min_lr = getattr(sched_cfg, 'MIN_LR', 0.0)
            total_epochs = cfg.TRAIN.END_EPOCH
            if warmup_epochs > 0:
                warmup_sched = LinearLR(
                    self.optimizer,
                    start_factor=1e-2,  # start at 1% of base LR
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                )
                cosine_sched = CosineAnnealingLR(
                    self.optimizer,
                    T_max=total_epochs - warmup_epochs,
                    eta_min=min_lr,
                )
                self.lr_scheduler = SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_sched, cosine_sched],
                    milestones=[warmup_epochs],
                )
            else:
                self.lr_scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=total_epochs,
                    eta_min=min_lr,
                )

        if cfg.LOSS.TYPE == "mld":
            self._losses = MetricCollection({
                split: MLDLosses(vae=self.is_vae, mode="xyz", cfg=cfg)
                for split in ["losses_train", "losses_test", "losses_val"]
            })
        else:
            raise NotImplementedError(
                "MotionCross model only supports mld losses.")

        self.losses = {
            key: self._losses["losses_" + key]
            for key in ["train", "test", "val"]
        }

        self.metrics_dict = cfg.METRIC.TYPE
        self.configure_metrics()

        # If we want to overide it at testing time
        self.sample_mean = False
        self.fact = None
        self.do_classifier_free_guidance = self.guidance_scale > 1.0
        if self.condition in ['text', 'text_uncond']:
            self.feats2joints = datamodule.feats2joints
        elif self.condition == 'ego':
            # Use datamodule's feats2joints if available, else identity
            self.feats2joints = getattr(datamodule, 'feats2joints', lambda x: x)
        elif self.condition == 'action':
            self.rot2xyz = Rotation2xyz(smpl_path=cfg.DATASET.SMPL_PATH)
            self.feats2joints_eval = lambda sample, mask: self.rot2xyz(
                sample.view(*sample.shape[:-1], 6, 25).permute(0, 3, 2, 1),
                mask=mask,
                pose_rep='rot6d',
                glob=True,
                translation=True,
                jointstype='smpl',
                vertstrans=True,
                betas=None,
                beta=0,
                glob_rot=None,
                get_rotations_back=False)
            self.feats2joints = lambda sample, mask: self.rot2xyz(
                sample.view(*sample.shape[:-1], 6, 25).permute(0, 3, 2, 1),
                mask=mask,
                pose_rep='rot6d',
                glob=True,
                translation=True,
                jointstype='vertices',
                vertstrans=False,
                betas=None,
                beta=0,
                glob_rot=None,
                get_rotations_back=False)

    def _get_t2m_evaluator(self, cfg):
        """
        load T2M text encoder and motion encoder for evaluating
        """
        # init module
        self.t2m_textencoder = t2m_textenc.TextEncoderBiGRUCo(
            word_size=cfg.model.t2m_textencoder.dim_word,
            pos_size=cfg.model.t2m_textencoder.dim_pos_ohot,
            hidden_size=cfg.model.t2m_textencoder.dim_text_hidden,
            output_size=cfg.model.t2m_textencoder.dim_coemb_hidden,
        )

        self.t2m_moveencoder = t2m_motionenc.MovementConvEncoder(
            input_size=cfg.DATASET.NFEATS - 4,
            hidden_size=cfg.model.t2m_motionencoder.dim_move_hidden,
            output_size=cfg.model.t2m_motionencoder.dim_move_latent,
        )

        self.t2m_motionencoder = t2m_motionenc.MotionEncoderBiGRUCo(
            input_size=cfg.model.t2m_motionencoder.dim_move_latent,
            hidden_size=cfg.model.t2m_motionencoder.dim_motion_hidden,
            output_size=cfg.model.t2m_motionencoder.dim_motion_latent,
        )
        # load pretrianed
        dataname = cfg.TEST.DATASETS[0]
        dataname = "t2m" if dataname == "humanml3d" else dataname
        t2m_checkpoint = torch.load(
            os.path.join(cfg.model.t2m_path, dataname,
                         "t2m/text_mot_match/model/finest.tar"))
        self.t2m_textencoder.load_state_dict(t2m_checkpoint["text_encoder"])
        self.t2m_moveencoder.load_state_dict(
            t2m_checkpoint["movement_encoder"])
        self.t2m_motionencoder.load_state_dict(
            t2m_checkpoint["motion_encoder"])

        # freeze params
        self.t2m_textencoder.eval()
        self.t2m_moveencoder.eval()
        self.t2m_motionencoder.eval()
        for p in self.t2m_textencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_moveencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_motionencoder.parameters():
            p.requires_grad = False

    def sample_from_distribution(
        self,
        dist,
        *,
        fact=None,
        sample_mean=False,
    ) -> Tensor:
        fact = fact if fact is not None else self.fact
        sample_mean = sample_mean if sample_mean is not None else self.sample_mean

        if sample_mean:
            return dist.loc.unsqueeze(0)

        # Reparameterization trick
        if fact is None:
            return dist.rsample().unsqueeze(0)

        # Resclale the eps
        eps = dist.rsample() - dist.loc
        z = dist.loc + fact * eps

        # add latent size
        z = z.unsqueeze(0)
        return z

    def forward(self, batch):
        lengths = batch["length"]
        if self.cfg.TEST.COUNT_TIME:
            self.starttime = time.time()

        if self.stage in ['diffusion', 'vae_diffusion']:
            # diffusion reverse
            if self.condition in ["text", "text_uncond"]:
                texts = batch["text"]
                if self.do_classifier_free_guidance:
                    uncond_tokens = [""] * len(texts)
                    if self.condition == 'text':
                        uncond_tokens.extend(texts)
                    elif self.condition == 'text_uncond':
                        uncond_tokens.extend(uncond_tokens)
                    texts = uncond_tokens
                cond_emb = self.text_encoder(texts)
            elif self.condition == "ego":
                ego = batch["ego"]
                if self.do_classifier_free_guidance:
                    # Create uncond (zeros) and concat with real ego
                    uncond_ego = torch.zeros_like(ego)
                    ego = torch.cat([uncond_ego, ego], dim=0)
                cond_emb = self.ego_encoder(ego)
            else:
                raise TypeError(f"condition type {self.condition} not supported in forward")
            z = self._diffusion_reverse(cond_emb, lengths)
        elif self.stage in ['vae']:
            motions = batch['motion']
            z, dist_m = self.vae.encode(motions, lengths)

        with torch.no_grad():
            # ToDo change mcross actor to same api
            if self.vae_type in ["mld","actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)

        if self.cfg.TEST.COUNT_TIME:
            self.endtime = time.time()
            elapsed = self.endtime - self.starttime
            self.times.append(elapsed)
            if len(self.times) % 100 == 0:
                meantime = np.mean(
                    self.times[-100:]) / self.cfg.TEST.BATCH_SIZE
                print(
                    f'100 iter mean Time (batch_size: {self.cfg.TEST.BATCH_SIZE}): {meantime}',
                )
            if len(self.times) % 1000 == 0:
                meantime = np.mean(
                    self.times[-1000:]) / self.cfg.TEST.BATCH_SIZE
                print(
                    f'1000 iter mean Time (batch_size: {self.cfg.TEST.BATCH_SIZE}): {meantime}',
                )
                with open(pjoin(self.cfg.FOLDER_EXP, 'times.txt'), 'w') as f:
                    for line in self.times:
                        f.write(str(line))
                        f.write('\n')
        joints = self.feats2joints(feats_rst.detach().cpu())
        return remove_padding(joints, lengths)

    def gen_from_latent(self, batch):
        z = batch["latent"]
        lengths = batch["length"]

        feats_rst = self.vae.decode(z, lengths)

        # feats => joints
        joints = self.feats2joints(feats_rst.detach().cpu())
        return remove_padding(joints, lengths)

    def recon_from_motion(self, batch):
        feats_ref = batch["motion"]
        length = batch["length"]

        z, dist = self.vae.encode(feats_ref, length)
        feats_rst = self.vae.decode(z, length)

        # feats => joints
        joints = self.feats2joints(feats_rst.detach().cpu())
        joints_ref = self.feats2joints(feats_ref.detach().cpu())
        return remove_padding(joints,
                              length), remove_padding(joints_ref, length)

    def _diffusion_reverse(self, encoder_hidden_states, lengths=None):
        # init latents
        bsz = encoder_hidden_states.shape[0]
        if self.do_classifier_free_guidance:
            bsz = bsz // 2
        if self.vae_type == "no":
            assert lengths is not None, "no vae (diffusion only) need lengths for diffusion"
            latents = torch.randn(
                (bsz, max(lengths), self.cfg.DATASET.NFEATS),
                device=encoder_hidden_states.device,
                dtype=torch.float,
            )
        else:
            latents = torch.randn(
                (bsz, self.latent_dim[0], self.latent_dim[-1]),
                device=encoder_hidden_states.device,
                dtype=torch.float,
            )

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        # set timesteps
        self.scheduler.set_timesteps(
            self.cfg.model.scheduler.num_inference_timesteps)
        timesteps = self.scheduler.timesteps.to(encoder_hidden_states.device)
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs = {}
        if "eta" in set(
                inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta

        # reverse
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (torch.cat(
                [latents] *
                2) if self.do_classifier_free_guidance else latents)
            lengths_reverse = (lengths * 2 if self.do_classifier_free_guidance
                               else lengths)
            # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # predict the noise residual
            noise_pred = self.denoiser(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
                lengths=lengths_reverse,
            )[0]
            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond)
                # text_embeddings_for_guidance = encoder_hidden_states.chunk(
                #     2)[1] if self.do_classifier_free_guidance else encoder_hidden_states
            latents = self.scheduler.step(noise_pred, t, latents,
                                              **extra_step_kwargs).prev_sample
            # if self.predict_epsilon:
            #     latents = self.scheduler.step(noise_pred, t, latents,
            #                                   **extra_step_kwargs).prev_sample
            # else:
            #     # predict x for standard diffusion model
            #     # compute the previous noisy sample x_t -> x_t-1
            #     latents = self.scheduler.step(noise_pred,
            #                                   t,
            #                                   latents,
            #                                   **extra_step_kwargs).prev_sample

        # [batch_size, 1, latent_dim] -> [1, batch_size, latent_dim]
        latents = latents.permute(1, 0, 2)
        return latents

    # ------------------------------------------------------------------
    # Guided reverse diffusion  (GMD-style trajectory guidance in latent space)
    # ------------------------------------------------------------------
    def _diffusion_reverse_guided(
        self,
        encoder_hidden_states,
        lengths=None,
        traj_target=None,
        traj_mask=None,
        guide_scale=100.0,
        guide_start_t=1000,
        guide_stop_t=0,
        normalize_grad=False,
        verbose=False,
        scale_by_alpha=False,
    ):
        """
        Denoising loop with per-step trajectory guidance.

        At every timestep *t* the predicted clean latent z0 is decoded through
        the frozen VAE, the root (pelvis) trajectory is extracted, and an L2
        loss against ``traj_target`` is back-propagated to z0.
        The resulting gradient nudges the sample toward trajectories
        that match the target.

        Args:
            encoder_hidden_states: conditioning embedding (with CFG doubling)
            lengths:               list of motion lengths
            traj_target:           (B, T, 2) target root (x, z) positions
                                   **in denormalized (real-world) metres**.
            traj_mask:             (B, T) bool — which frames to guide on.
                                   ``None`` → guide on all frames.
            guide_scale:           gradient multiplier (like classifier scale
                                   in classifier-guided diffusion).
            guide_start_t:         only guide when t <= this value
            guide_stop_t:          stop guiding when t < this value
            normalize_grad:        if True, normalize gradient to unit
                                   direction (step size = guide_scale only).
            verbose:               if True, print per-step diagnostics.
            scale_by_alpha:        if True, multiply effective scale by
                                   alpha_bar_t (weaker at noisy steps).
        """
        from mld.data.humanml.scripts.motion_process import recover_root_rot_pos

        bsz = encoder_hidden_states.shape[0]
        if self.do_classifier_free_guidance:
            bsz = bsz // 2

        # --- init latents ------------------------------------------------
        if self.vae_type == "no":
            assert lengths is not None
            latents = torch.randn(
                (bsz, max(lengths), self.cfg.DATASET.NFEATS),
                device=encoder_hidden_states.device, dtype=torch.float,
            )
        else:
            latents = torch.randn(
                (bsz, self.latent_dim[0], self.latent_dim[-1]),
                device=encoder_hidden_states.device, dtype=torch.float,
            )
        latents = latents * self.scheduler.init_noise_sigma

        # --- scheduler setup ---------------------------------------------
        self.scheduler.set_timesteps(
            self.cfg.model.scheduler.num_inference_timesteps)
        timesteps = self.scheduler.timesteps.to(encoder_hidden_states.device)

        extra_step_kwargs = {}
        if "eta" in set(
                inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta

        # --- precompute mean / std for denorm (keep on device) -----------
        dm = self.datamodule
        if hasattr(dm, 'mean') and dm.mean is not None:
            mean_t = torch.tensor(dm.mean, dtype=torch.float32,
                                  device=latents.device)
            std_t  = torch.tensor(dm.std,  dtype=torch.float32,
                                  device=latents.device)
        else:
            mean_t = std_t = None

        # --- denoising loop with guidance --------------------------------
        for i, t in enumerate(timesteps):
            t_int = int(t)
            do_guide = (
                traj_target is not None
                and t_int <= guide_start_t
                and t_int >= guide_stop_t
            )

            # duplicate for CFG
            latent_model_input = (
                torch.cat([latents] * 2)
                if self.do_classifier_free_guidance else latents
            )
            lengths_reverse = (
                lengths * 2 if self.do_classifier_free_guidance else lengths
            )

            # ---- denoiser forward (always no_grad) ----------------------
            with torch.no_grad():
                noise_pred = self.denoiser(
                    sample=latent_model_input,
                    timestep=t,
                    encoder_hidden_states=encoder_hidden_states,
                    lengths=lengths_reverse,
                    return_dict=False,
                )[0]

            # CFG combination
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_cond - noise_pred_uncond)

            # scheduler step → get x_{t-1} *and* predicted x0
            step_out = self.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs)
            latents_prev = step_out.prev_sample
            pred_z0 = step_out.pred_original_sample       # (B, 1, 256)

            # ---- trajectory guidance ------------------------------------
            if do_guide and pred_z0 is not None:
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[
                        timesteps[i + 1]] if i + 1 < len(timesteps)
                    else self.scheduler.final_alpha_cumprod
                )

                # Decode predicted z0 through frozen VAE → features
                z0_for_decode = pred_z0.detach().requires_grad_(True)
                z0_perm = z0_for_decode.permute(1, 0, 2)  # (1, B, 256)
                feats = self.vae.decode(z0_perm, lengths)  # (B, T, 263)

                # Denormalize
                if mean_t is not None:
                    feats_denorm = feats * std_t + mean_t
                else:
                    feats_denorm = feats

                # Extract root trajectory
                _, r_pos = recover_root_rot_pos(feats_denorm)
                pred_root_xz = r_pos[..., [0, 2]]        # (B, T, 2)

                # Compute trajectory loss
                T_pred = pred_root_xz.shape[1]
                T_tgt  = traj_target.shape[1]
                T_min  = min(T_pred, T_tgt)
                pred_crop = pred_root_xz[:, :T_min]
                tgt_crop  = traj_target[:, :T_min]

                if traj_mask is not None:
                    mask_crop = traj_mask[:, :T_min].unsqueeze(-1).float()
                else:
                    mask_crop = torch.ones_like(pred_crop)

                diff = (pred_crop - tgt_crop) * mask_crop
                loss = (diff ** 2).sum() / mask_crop.sum().clamp(min=1)

                # Gradient w.r.t. predicted z0
                grad_z0 = torch.autograd.grad(loss, z0_for_decode)[0]
                grad_norm = grad_z0.norm()

                if normalize_grad and grad_norm > 1e-8:
                    grad_z0 = grad_z0 / grad_norm

                effective_scale = (
                    guide_scale * float(alpha_prod_t) if scale_by_alpha
                    else guide_scale
                )

                # Compute raw correction
                correction = effective_scale * grad_z0

                # Adaptive clamp: limit correction norm to 10% of pred_z0
                # norm.  Prevents catastrophic divergence on samples where
                # the gradient is disproportionately large.
                z0_norm = pred_z0.norm().clamp(min=1e-8)
                max_correction_norm = 0.1 * z0_norm
                corr_norm = correction.norm()
                if corr_norm > max_correction_norm:
                    correction = correction * (max_correction_norm / corr_norm)

                if verbose:
                    print(f"  t={t_int:4d} loss={loss.item():.6f} "
                          f"|grad|={grad_norm.item():.6f} "
                          f"|z0|={z0_norm.item():.4f} "
                          f"|corr|={correction.norm().item():.4f} "
                          f"pred_end={pred_root_xz[0,-1].detach().cpu().tolist()} "
                          f"tgt_end={tgt_crop[0,-1].detach().cpu().tolist()}")

                # Correct z0, then re-derive x_{t-1} using the ORIGINAL
                # eps (preserves the denoiser's noise estimate, only shifts
                # the clean signal).
                pred_z0_corrected = pred_z0 - correction
                eps_implied = (
                    (latents - alpha_prod_t.sqrt() * pred_z0)
                    / (1 - alpha_prod_t).sqrt().clamp(min=1e-8)
                )
                latents_prev = (
                    alpha_prod_t_prev.sqrt() * pred_z0_corrected
                    + (1 - alpha_prod_t_prev).sqrt() * eps_implied
                )

            latents = latents_prev.detach()

        # (B, 1, 256) → (1, B, 256)
        latents = latents.permute(1, 0, 2)
        return latents
    # ------------------------------------------------------------------

    def _diffusion_reverse_tsne(self, encoder_hidden_states, lengths=None):
        # init latents
        bsz = encoder_hidden_states.shape[0]
        if self.do_classifier_free_guidance:
            bsz = bsz // 2
        if self.vae_type == "no":
            assert lengths is not None, "no vae (diffusion only) need lengths for diffusion"
            latents = torch.randn(
                (bsz, max(lengths), self.cfg.DATASET.NFEATS),
                device=encoder_hidden_states.device,
                dtype=torch.float,
            )
        else:
            latents = torch.randn(
                (bsz, self.latent_dim[0], self.latent_dim[-1]),
                device=encoder_hidden_states.device,
                dtype=torch.float,
            )

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        # set timesteps
        self.scheduler.set_timesteps(
            self.cfg.model.scheduler.num_inference_timesteps)
        timesteps = self.scheduler.timesteps.to(encoder_hidden_states.device)
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs = {}
        if "eta" in set(
                inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta

        # reverse
        latents_t = []
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (torch.cat(
                [latents] *
                2) if self.do_classifier_free_guidance else latents)
            lengths_reverse = (lengths * 2 if self.do_classifier_free_guidance
                               else lengths)
            # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # predict the noise residual
            noise_pred = self.denoiser(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
                lengths=lengths_reverse,
            )[0]
            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond)
                # text_embeddings_for_guidance = encoder_hidden_states.chunk(
                #     2)[1] if self.do_classifier_free_guidance else encoder_hidden_states
            latents = self.scheduler.step(noise_pred, t, latents,
                                              **extra_step_kwargs).prev_sample
            # [batch_size, 1, latent_dim] -> [1, batch_size, latent_dim]
            latents_t.append(latents.permute(1,0,2))
        # [1, batch_size, latent_dim] -> [t, batch_size, latent_dim]
        latents_t = torch.cat(latents_t)
        return latents_t

    def _diffusion_process(self, latents, encoder_hidden_states, lengths=None):
        """
        heavily from https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py
        """
        # our latent   [batch_size, n_token=1 or 5 or 10, latent_dim=256]
        # sd  latent   [batch_size, [n_token0=64,n_token1=64], latent_dim=4]
        # [n_token, batch_size, latent_dim] -> [batch_size, n_token, latent_dim]
        latents = latents.permute(1, 0, 2)

        # Sample noise that we'll add to the latents
        # [batch_size, n_token, latent_dim]
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each motion
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz, ),
            device=latents.device,
        )
        timesteps = timesteps.long()
        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_latents = self.noise_scheduler.add_noise(latents.clone(), noise,
                                                       timesteps)
        # Predict the noise residual
        noise_pred = self.denoiser(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            lengths=lengths,
            return_dict=False,
        )[0]
        # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
        if self.cfg.LOSS.LAMBDA_PRIOR != 0.0:
            noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
            noise, noise_prior = torch.chunk(noise, 2, dim=0)
        else:
            noise_pred_prior = 0
            noise_prior = 0
        n_set = {
            "noise": noise,
            "noise_prior": noise_prior,
            "noise_pred": noise_pred,
            "noise_pred_prior": noise_pred_prior,
        }
        if not self.predict_epsilon:
            n_set["pred"] = noise_pred
            n_set["latent"] = latents
        return n_set

    def train_vae_forward(self, batch):
        feats_ref = batch["motion"]
        lengths = batch["length"]

        if self.vae_type in ["mld", "vposert", "actor"]:
            motion_z, dist_m = self.vae.encode(feats_ref, lengths)
            feats_rst = self.vae.decode(motion_z, lengths)
        else:
            raise TypeError("vae_type must be mcross or actor")

        # prepare for metric
        recons_z, dist_rm = self.vae.encode(feats_rst, lengths)

        # joints recover
        if self.condition in ["text", "ego"]:
            joints_rst = self.feats2joints(feats_rst)
            joints_ref = self.feats2joints(feats_ref)
        elif self.condition == "action":
            mask = batch["mask"]
            joints_rst = self.feats2joints(feats_rst, mask)
            joints_ref = self.feats2joints(feats_ref, mask)

        if dist_m is not None:
            if self.is_vae:
                # Create a centred normal distribution to compare with
                mu_ref = torch.zeros_like(dist_m.loc)
                scale_ref = torch.ones_like(dist_m.scale)
                dist_ref = torch.distributions.Normal(mu_ref, scale_ref)
            else:
                dist_ref = dist_m

        # cut longer part over max length
        min_len = min(feats_ref.shape[1], feats_rst.shape[1])
        min_joints_len = min(joints_ref.shape[1], joints_rst.shape[1])
        rs_set = {
            "m_ref": feats_ref[:, :min_len, :],
            "m_rst": feats_rst[:, :min_len, :],
            # [bs, ntoken, nfeats]<= [ntoken, bs, nfeats]
            "lat_m": motion_z.permute(1, 0, 2),
            "lat_rm": recons_z.permute(1, 0, 2),
            "joints_ref": joints_ref[:, :min_joints_len, ...],
            "joints_rst": joints_rst[:, :min_joints_len, ...],
            "dist_m": dist_m,
            "dist_ref": dist_ref,
            "lengths": lengths,
        }
        return rs_set
    
    def _compute_diffusion_loss(self, rs_set):
        """Compute diffusion training loss directly (bypasses torchmetrics)."""
        import torch.nn.functional as F
        
        if self.predict_epsilon:
            # Predict noise (your case)
            loss = F.mse_loss(rs_set['noise_pred'], rs_set['noise'])
        else:
            # Predict x
            loss = F.mse_loss(rs_set['pred'], rs_set['latent'])
        
        # Add prior loss if configured (yours is 0.0, so this won't run)
        if self.cfg.LOSS.LAMBDA_PRIOR != 0.0:
            prior_loss = F.mse_loss(rs_set['noise_pred_prior'], rs_set['noise_prior'])
            loss = loss + self.cfg.LOSS.LAMBDA_PRIOR * prior_loss
    
        return loss

    def _compute_vae_loss(self, rs_set):
        """Compute VAE training loss directly (bypasses torchmetrics) for backprop.
        
        Uses length-based masking so that zero-padded frames do not
        contribute to reconstruction / joint / trajectory losses.
        """
        import torch.nn.functional as F
        from mld.data.humanml.scripts.motion_process import recover_root_rot_pos

        device = rs_set["m_rst"].device
        lengths = rs_set["lengths"]
        loss = torch.tensor(0.0, device=device)

        # --- Build frame mask from lengths ---
        m_rst = rs_set["m_rst"]  # (B, T, 263)
        m_ref = rs_set["m_ref"]
        feat_mask = lengths_to_mask(lengths, device, max_len=m_ref.shape[1])  # (B, T)

        # --- Feature reconstruction loss (masked) ---
        # Average over feature dim first, then mask over frames.
        # This keeps the same loss scale as reduction='mean' but excludes padding.
        rec_loss = F.smooth_l1_loss(m_rst, m_ref, reduction='none').mean(dim=-1)  # (B, T)
        rec_loss = (rec_loss * feat_mask).sum() / feat_mask.sum()
        loss = loss + self.cfg.LOSS.LAMBDA_REC * rec_loss

        # --- Joint reconstruction loss (masked) ---
        j_rst = rs_set["joints_rst"]  # (B, T, 22, 3)
        j_ref = rs_set["joints_ref"]
        joint_mask = lengths_to_mask(lengths, device, max_len=j_ref.shape[1])  # (B, T)

        joint_loss = F.smooth_l1_loss(j_rst, j_ref, reduction='none').mean(dim=(-1, -2))  # (B, T)
        joint_loss = (joint_loss * joint_mask).sum() / joint_mask.sum()
        loss = loss + self.cfg.LOSS.LAMBDA_JOINT * joint_loss

        # --- Trajectory loss (masked) ---
        if self.cfg.LOSS.LAMBDA_TRAJ != 0.0:
            _, rst_pos = recover_root_rot_pos(m_rst)
            _, ref_pos = recover_root_rot_pos(m_ref)
            traj_rst = rst_pos[..., [0, 2]]  # (B, T, 2)
            traj_ref = ref_pos[..., [0, 2]]
            traj_mask = lengths_to_mask(lengths, device, max_len=traj_ref.shape[1])  # (B, T)

            traj_loss = F.mse_loss(traj_rst, traj_ref, reduction='none').mean(dim=-1)  # (B, T)
            traj_loss = (traj_loss * traj_mask).sum() / traj_mask.sum()
            loss = loss + self.cfg.LOSS.LAMBDA_TRAJ * traj_loss

        # KL loss against N(0, I) — with optional annealing
        kl_weight = self._get_kl_weight()
        if kl_weight > 0.0:
            kl = torch.distributions.kl_divergence(
                rs_set["dist_m"], rs_set["dist_ref"]
            ).mean()
            loss = loss + kl_weight * kl

        return loss

    def _get_kl_weight(self):
        """Compute the current KL weight, with optional linear annealing."""
        kl_anneal_cfg = getattr(self.cfg.LOSS, 'KL_ANNEAL', None)
        if kl_anneal_cfg is not None and getattr(kl_anneal_cfg, 'ENABLED', False):
            start_epoch = kl_anneal_cfg.START_EPOCH
            end_epoch = kl_anneal_cfg.END_EPOCH
            start_w = kl_anneal_cfg.START_WEIGHT
            end_w = kl_anneal_cfg.END_WEIGHT
            current_epoch = self.current_epoch
            if current_epoch <= start_epoch:
                return start_w
            elif current_epoch >= end_epoch:
                return end_w
            else:
                # Linear interpolation
                progress = (current_epoch - start_epoch) / (end_epoch - start_epoch)
                return start_w + progress * (end_w - start_w)
        else:
            return self.cfg.LOSS.LAMBDA_KL

    def train_diffusion_forward(self, batch):
        feats_ref = batch["motion"]
        lengths = batch["length"]
        # motion encode
        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                z, dist = self.vae.encode(feats_ref, lengths)
                if self.deterministic_z:
                    # Use posterior mean instead of sample — eliminates
                    # the irreducible noise floor from stochastic encoding.
                    # Useful for debugging: loss should go to ~0 on single
                    # sample overfit if the denoiser is working correctly.
                    z = dist.loc
            elif self.vae_type == "no":
                z = feats_ref.permute(1, 0, 2)
            else:
                raise TypeError("vae_type must be mcross or actor")

        if self.condition in ["text", "text_uncond"]:
            text = batch["text"]
            # classifier free guidance: randomly drop text during training
            text = [
                "" if np.random.rand(1) < self.guidance_uncodp else i
                for i in text
            ]
            # text encode
            cond_emb = self.text_encoder(text)
        elif self.condition in ['action']:
            action = batch['action']
            # text encode
            cond_emb = action
        # elif self.condition in ['ego']:
        #     ego = batch["ego"]  # (B, T_ego, 2)
            
        #     # Classifier-free guidance: randomly drop ego conditioning
        #     if np.random.rand(1) < self.guidance_uncodp:
        #         # Use zeros as "unconditional" embedding
        #         cond_emb = torch.zeros(
        #             ego.shape[0], ego.shape[1], self.latent_dim[-1],
        #             device=ego.device, dtype=ego.dtype
        #         )
        #     else:
        #         cond_emb = self.ego_encoder(ego)
        elif self.condition in ['ego']:
            ego = batch["ego"]  # (B, T_ego, 2)
            
            # Classifier-free guidance: randomly drop ego conditioning PER SAMPLE
            if self.training:
                B = ego.shape[0]
                drop_mask = torch.rand(B, device=ego.device) < self.guidance_uncodp  # (B,)
                # Zero input for dropped samples (matches inference unconditional)
                ego = ego.clone()
                ego[drop_mask] = 0.0
            
            # Encode (zeros become unconditional embedding)
            cond_emb = self.ego_encoder(ego)  # (B, T_ego, 256)
        else:
            raise TypeError(f"condition type {self.condition} not supported")

        # diffusion process return with noise and noise_pred
        n_set = self._diffusion_process(z, cond_emb, lengths)
        return {**n_set}

    def test_diffusion_forward(self, batch, finetune_decoder=False):
        lengths = batch["length"]

        if self.condition in ["text", "text_uncond"]:
            # get text embeddings
            if self.do_classifier_free_guidance:
                uncond_tokens = [""] * len(lengths)
                if self.condition == 'text':
                    texts = batch["text"]
                    uncond_tokens.extend(texts)
                elif self.condition == 'text_uncond':
                    uncond_tokens.extend(uncond_tokens)
                texts = uncond_tokens
            cond_emb = self.text_encoder(texts)
        elif self.condition in ['action']:
            cond_emb = batch['action']
            if self.do_classifier_free_guidance:
                cond_emb = torch.cat(
                    cond_emb,
                    torch.zeros_like(batch['action'],
                                     dtype=batch['action'].dtype))
        elif self.condition in ['ego']:
            ego = batch["ego"]  # (B, T_ego, 2)
            if self.do_classifier_free_guidance:
                # Create uncond (zeros) and concat with real ego
                uncond_ego = torch.zeros_like(ego)
                ego = torch.cat([uncond_ego, ego], dim=0)
            cond_emb = self.ego_encoder(ego)
        else:
            raise TypeError(f"condition type {self.condition} not supported")

        # diffusion reverse
        with torch.no_grad():
            z = self._diffusion_reverse(cond_emb, lengths)

        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)
            else:
                raise TypeError("vae_type must be mcross or actor or mld")

        joints_rst = self.feats2joints(feats_rst)

        rs_set = {
            "m_rst": feats_rst,
            # [bs, ntoken, nfeats]<= [ntoken, bs, nfeats]
            "lat_t": z.permute(1, 0, 2),
            "joints_rst": joints_rst,
        }

        # Store real ego embedding for R-precision metric.
        # cond_emb is (2B, 1, 256) with CFG or (B, 1, 256) without.
        # The last len(lengths) entries are always the real (non-zero) embeddings.
        if self.condition in ['ego']:
            # mean(dim=1): works for EgoEncoderPooled (T=1) and EgoEncoder/H4 (T=196)
            rs_set["ego_emb"] = cond_emb[-len(lengths):].mean(dim=1).detach()  # (B, 256)

        # prepare gt/refer for metric
        if "motion" in batch.keys() and not finetune_decoder:
            feats_ref = batch["motion"].detach()
            with torch.no_grad():
                if self.vae_type in ["mld", "vposert", "actor"]:
                    motion_z, dist_m = self.vae.encode(feats_ref, lengths)
                    recons_z, dist_rm = self.vae.encode(feats_rst, lengths)
                elif self.vae_type == "no":
                    motion_z = feats_ref
                    recons_z = feats_rst

            joints_ref = self.feats2joints(feats_ref)

            rs_set["m_ref"] = feats_ref
            rs_set["lat_m"] = motion_z.permute(1, 0, 2)
            rs_set["lat_rm"] = recons_z.permute(1, 0, 2)
            rs_set["joints_ref"] = joints_ref
        return rs_set

    def ego_eval(self, batch):
        """
        Ego-conditioned evaluation pass.

        Wraps test_diffusion_forward and computes the additional motion
        embeddings needed by EgoMotionMetrics:

          - t2m_lat_rm / t2m_lat_m : 512-D embeddings from the pretrained
            HumanML3D t2m_motionencoder (used for FID and Diversity).
          - motion_lat_pooled       : 256-D mean-pooled VAE latent of the
            generated motion (used for R-Precision).
          - ego_emb                 : 256-D ego encoder output (already added
            by test_diffusion_forward, used for R-Precision).
        """
        rs_set = self.test_diffusion_forward(batch)
        lengths = batch["length"]
        if "ego_emb" not in rs_set:
            # Text-conditioned model evaluated on ego data: no ego embedding
            # exists. Insert a deliberately dim-1 placeholder so the shape guard
            # in EgoMotionMetrics reports R-Precision as UNDEFINED (0.0) rather
            # than silently producing a plausible-looking fake number.
            rs_set["ego_emb"] = torch.zeros(
                len(lengths), 1, device=rs_set["m_rst"].device)

        # t2m motion encoder expects lengths divided by 4
        # (two stride-2 convolutions in MovementConvEncoder).
        m_lens = torch.tensor(lengths, device=rs_set["m_rst"].device) // 4
        m_lens = torch.clamp(m_lens, min=1)

        # pack_padded_sequence requires lengths sorted in descending order.
        m_lens_sorted, sort_idx = m_lens.sort(descending=True)
        unsort_idx = sort_idx.argsort()

        with torch.no_grad():
            # Generated motion → 512-D t2m embedding
            recons_mov = self.t2m_moveencoder(rs_set["m_rst"][..., :-4]).detach()
            lat_rm_sorted = self.t2m_motionencoder(recons_mov[sort_idx], m_lens_sorted)
            rs_set["t2m_lat_rm"] = lat_rm_sorted[unsort_idx]  # (B, 512)

            if "m_ref" in rs_set:
                # GT motion → 512-D t2m embedding
                motion_mov = self.t2m_moveencoder(rs_set["m_ref"][..., :-4]).detach()
                lat_m_sorted = self.t2m_motionencoder(motion_mov[sort_idx], m_lens_sorted)
                rs_set["t2m_lat_m"] = lat_m_sorted[unsort_idx]  # (B, 512)

        # Mean-pool VAE latent for R-precision: (B, ntoken, 256) → (B, 256)
        rs_set["motion_lat_pooled"] = rs_set["lat_rm"].mean(dim=1).detach()  # (B, 256)

        return rs_set

    def t2m_eval(self, batch):
        texts = batch["text"]
        motions = batch["motion"].detach().clone()
        lengths = batch["length"]
        word_embs = batch["word_embs"].detach().clone()
        pos_ohot = batch["pos_ohot"].detach().clone()
        text_lengths = batch["text_len"].detach().clone()

        # start
        start = time.time()

        if self.trainer.datamodule.is_mm:
            texts = texts * self.cfg.TEST.MM_NUM_REPEATS
            motions = motions.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                dim=0)
            lengths = lengths * self.cfg.TEST.MM_NUM_REPEATS
            word_embs = word_embs.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            pos_ohot = pos_ohot.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                  dim=0)
            text_lengths = text_lengths.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)

        if self.stage in ['diffusion', 'vae_diffusion']:
            # diffusion reverse
            if self.do_classifier_free_guidance:
                uncond_tokens = [""] * len(texts)
                if self.condition == 'text':
                    uncond_tokens.extend(texts)
                elif self.condition == 'text_uncond':
                    uncond_tokens.extend(uncond_tokens)
                texts = uncond_tokens
            text_emb = self.text_encoder(texts)
            z = self._diffusion_reverse(text_emb, lengths)
        elif self.stage in ['vae']:
            if self.vae_type in ["mld", "vposert", "actor"]:
                z, dist_m = self.vae.encode(motions, lengths)
            else:
                raise TypeError("Not supported vae type!")
            if self.condition in ['text_uncond']:
                # uncond random sample
                z = torch.randn_like(z)

        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)

        # end time
        end = time.time()
        self.times.append(end - start)

        # joints recover
        joints_rst = self.feats2joints(feats_rst)
        joints_ref = self.feats2joints(motions)

        # renorm for t2m evaluators
        feats_rst = self.datamodule.renorm4t2m(feats_rst)
        motions = self.datamodule.renorm4t2m(motions)

        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=motions.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        feats_rst = feats_rst[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens,
                           self.cfg.DATASET.HUMANML3D.UNIT_LEN,
                           rounding_mode="floor")

        recons_mov = self.t2m_moveencoder(feats_rst[..., :-4]).detach()
        recons_emb = self.t2m_motionencoder(recons_mov, m_lens)
        motion_mov = self.t2m_moveencoder(motions[..., :-4]).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        # t2m text encoder
        text_emb = self.t2m_textencoder(word_embs, pos_ohot,
                                        text_lengths)[align_idx]

        rs_set = {
            "m_ref": motions,
            "m_rst": feats_rst,
            "lat_t": text_emb,
            "lat_m": motion_emb,
            "lat_rm": recons_emb,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
        }
        return rs_set

    def a2m_eval(self, batch):
        actions = batch["action"]
        actiontexts = batch["action_text"]
        motions = batch["motion"].detach().clone()
        lengths = batch["length"]

        if self.do_classifier_free_guidance:
            cond_emb = torch.cat((torch.zeros_like(actions), actions))

        if self.stage in ['diffusion', 'vae_diffusion']:
            z = self._diffusion_reverse(cond_emb, lengths)
        elif self.stage in ['vae']:
            if self.vae_type in ["mld", "vposert","actor"]:
                z, dist_m = self.vae.encode(motions, lengths)
            else:
                raise TypeError("vae_type must be mcross or actor")

        with torch.no_grad():
            if self.vae_type in ["mld", "vposert","actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)
            else:
                raise TypeError("vae_type must be mcross or actor or mld")

        mask = batch["mask"]
        joints_rst = self.feats2joints(feats_rst, mask)
        joints_ref = self.feats2joints(motions, mask)
        joints_eval_rst = self.feats2joints_eval(feats_rst, mask)
        joints_eval_ref = self.feats2joints_eval(motions, mask)

        rs_set = {
            "m_action": actions,
            "m_ref": motions,
            "m_rst": feats_rst,
            "m_lens": lengths,
            "joints_rst": joints_rst,
            "joints_ref": joints_ref,
            "joints_eval_rst": joints_eval_rst,
            "joints_eval_ref": joints_eval_ref,
        }
        return rs_set

    def a2m_gt(self, batch):
        actions = batch["action"]
        actiontexts = batch["action_text"]
        motions = batch["motion"].detach().clone()
        lengths = batch["length"]
        mask = batch["mask"]

        joints_ref = self.feats2joints(motions.to('cuda'), mask.to('cuda'))

        rs_set = {
            "m_action": actions,
            "m_text": actiontexts,
            "m_ref": motions,
            "m_lens": lengths,
            "joints_ref": joints_ref,
        }
        return rs_set

    def eval_gt(self, batch, renoem=True):
        motions = batch["motion"].detach().clone()
        lengths = batch["length"]

        # feats_rst = self.datamodule.renorm4t2m(feats_rst)
        if renoem:
            motions = self.datamodule.renorm4t2m(motions)

        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=motions.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens,
                           self.cfg.DATASET.HUMANML3D.UNIT_LEN,
                           rounding_mode="floor")

        word_embs = batch["word_embs"].detach()
        pos_ohot = batch["pos_ohot"].detach()
        text_lengths = batch["text_len"].detach()

        motion_mov = self.t2m_moveencoder(motions[..., :-4]).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        # t2m text encoder
        text_emb = self.t2m_textencoder(word_embs, pos_ohot,
                                        text_lengths)[align_idx]

        # joints recover
        joints_ref = self.feats2joints(motions)

        rs_set = {
            "m_ref": motions,
            "lat_t": text_emb,
            "lat_m": motion_emb,
            "joints_ref": joints_ref,
        }
        return rs_set

    def allsplit_step(self, split: str, batch, batch_idx):
        if split in ["train", "val"]:
            if self.stage == "vae":
                rs_set = self.train_vae_forward(batch)
                rs_set["lat_t"] = rs_set["lat_m"]
            elif self.stage == "diffusion":
                rs_set = self.train_diffusion_forward(batch)
            elif self.stage == "vae_diffusion":
                vae_rs_set = self.train_vae_forward(batch)
                diff_rs_set = self.train_diffusion_forward(batch)
                t2m_rs_set = self.test_diffusion_forward(batch,
                                                         finetune_decoder=True)
                # merge results
                rs_set = {
                    **vae_rs_set,
                    **diff_rs_set,
                    "gen_m_rst": t2m_rs_set["m_rst"],
                    "gen_joints_rst": t2m_rs_set["joints_rst"],
                    "lat_t": t2m_rs_set["lat_t"],
                }
            else:
                raise ValueError(f"Not support this stage {self.stage}!")

            # loss = self.losses[split].update(rs_set)
            # if loss is None:
            #     raise ValueError(
            #         "Loss is None, this happend with torchmetrics > 0.7")
            
            # Compute loss directly for backprop (bypasses torchmetrics issues)
            if self.stage == "diffusion":
                loss = self._compute_diffusion_loss(rs_set)
            elif self.stage == "vae":
                # Sync the KL weight for logging consistency
                kl_weight = self._get_kl_weight()
                if hasattr(self.losses[split], '_params') and 'kl_motion' in self.losses[split]._params:
                    self.losses[split]._params['kl_motion'] = kl_weight
                loss = self._compute_vae_loss(rs_set)
            else:
                # For vae or vae_diffusion stages, use original method
                loss = self.losses[split].update(rs_set)
                if loss is None:
                    raise ValueError("Loss is None, this happened with torchmetrics > 0.7")

            # Still update metrics for logging (but don't use return value)
            if self.stage in ["diffusion", "vae"]:
                self.losses[split].update(rs_set)

            # Log LR and KL weight for monitoring
            if split == "train" and self.stage == "vae":
                current_lr = self.optimizer.param_groups[0]['lr']
                self.log("train/lr", current_lr, prog_bar=False)
                self.log("train/kl_weight", self._get_kl_weight(), prog_bar=False)

        # Compute the metrics - currently evaluate results from text to motion
        if split in ["val", "test"]:
            # EgoMotionMetrics implies our EgoMotion batches, which carry no
            # HumanML3D fields (word_embs/pos_ohot); route to ego_eval even for
            # text-conditioned models trained on our data (item 10b rung 3).
            if "EgoMotionMetrics" in self.metrics_dict:
                rs_set = self.ego_eval(batch)
            elif self.condition in ['text', 'text_uncond']:
                # use t2m evaluators
                rs_set = self.t2m_eval(batch)
            elif self.condition == 'action':
                # use a2m evaluators
                rs_set = self.a2m_eval(batch)
            elif self.condition == 'ego':
                # use ego_eval (includes t2m embeddings) when EgoMotionMetrics
                # is active, otherwise fall back to plain diffusion forward
                if "EgoMotionMetrics" in self.metrics_dict:
                    rs_set = self.ego_eval(batch)
                else:
                    rs_set = self.test_diffusion_forward(batch)
            # MultiModality evaluation sperately
            if self.trainer.datamodule.is_mm:
                metrics_dicts = ['MMMetrics']
            else:
                metrics_dicts = self.metrics_dict

            for metric in metrics_dicts:
                if metric == "TemosMetric":
                    phase = split if split != "val" else "eval"
                    dataset_name = eval(f"self.cfg.{phase.upper()}.DATASETS")[0].lower()
                    if dataset_name not in ["humanml3d", "kit"]:
                        # Skip TemosMetric for non-HumanML3D datasets (e.g., egomotion)
                        # APE and AVE metrics only support humanml3d and kit datasets
                        continue

                    getattr(self, metric).update(rs_set["joints_rst"],
                                                 rs_set["joints_ref"],
                                                 batch["length"])
                elif metric == "TM2TMetrics":
                    getattr(self, metric).update(
                        # lat_t, latent encoded from diffusion-based text
                        # lat_rm, latent encoded from reconstructed motion
                        # lat_m, latent encoded from gt motion
                        # rs_set['lat_t'], rs_set['lat_rm'], rs_set['lat_m'], batch["length"])
                        rs_set["lat_t"],
                        rs_set["lat_rm"],
                        rs_set["lat_m"],
                        batch["length"],
                    )
                elif metric == "UncondMetrics":
                    getattr(self, metric).update(
                        recmotion_embeddings=rs_set["lat_rm"],
                        gtmotion_embeddings=rs_set["lat_m"],
                        lengths=batch["length"],
                    )
                elif metric == "MRMetrics":
                    getattr(self, metric).update(rs_set["joints_rst"],
                                                 rs_set["joints_ref"],
                                                 batch["length"])
                elif metric == "EgoMotionMetrics":
                    getattr(self, metric).update(
                        recmotion_embeddings=rs_set["t2m_lat_rm"],
                        gtmotion_embeddings=rs_set["t2m_lat_m"],
                        motion_latents=rs_set["motion_lat_pooled"],
                        ego_embeddings=rs_set["ego_emb"],
                        lengths=batch["length"],
                    )
                elif metric == "MMMetrics":
                    # For ego condition use t2m embeddings; otherwise VAE latents
                    if self.condition == 'ego' and "t2m_lat_rm" in rs_set:
                        getattr(self, metric).update(
                            rs_set["t2m_lat_rm"].unsqueeze(0), batch["length"]
                        )
                    else:
                        getattr(self, metric).update(
                            rs_set["lat_rm"].unsqueeze(0), batch["length"]
                        )
                elif metric == "HUMANACTMetrics":
                    getattr(self, metric).update(rs_set["m_action"],
                                                 rs_set["joints_eval_rst"],
                                                 rs_set["joints_eval_ref"],
                                                 rs_set["m_lens"])
                elif metric == "UESTCMetrics":
                    # the stgcn model expects rotations only
                    getattr(self, metric).update(
                        rs_set["m_action"],
                        rs_set["m_rst"].view(*rs_set["m_rst"].shape[:-1], 6,
                                             25).permute(0, 3, 2, 1)[:, :-1],
                        rs_set["m_ref"].view(*rs_set["m_ref"].shape[:-1], 6,
                                             25).permute(0, 3, 2, 1)[:, :-1],
                        rs_set["m_lens"])
                else:
                    raise TypeError(f"Not support this metric {metric}")

        # return forward output rather than loss during test
        if split in ["test"]:
            return rs_set["joints_rst"], batch["length"]
        return loss
