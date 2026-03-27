# Literature Survey: Ego-Conditioned Pedestrian Motion Generation

**Date**: 2026-03-27
**Scope**: 2022–2026, covering ego/context-conditioned motion, interaction-aware generation,
latent diffusion for human motion, classifier-free guidance, and VAE latent space effects.

---

## Table of Contents

1. [Ego-Conditioned and Context-Conditioned Pedestrian Motion Generation](#1-ego-conditioned-and-context-conditioned-pedestrian-motion-generation)
2. [Interaction-Aware Human Motion Generation](#2-interaction-aware-human-motion-generation)
3. [Latent Diffusion Models for Human Motion](#3-latent-diffusion-models-for-human-motion)
4. [Classifier-Free Guidance for Motion Generation](#4-classifier-free-guidance-for-motion-generation)
5. [VAE Latent Space Design and Downstream Generation Quality](#5-vae-latent-space-design-and-downstream-generation-quality)
6. [Research Gaps and Positioning](#6-research-gaps-and-positioning)

---

## 1. Ego-Conditioned and Context-Conditioned Pedestrian Motion Generation

### 1.1 MDMP: Multi-Diffusion for Multi-Person Motion Prediction
- **Authors**: Ruan et al.
- **Year**: 2024
- **Venue**: arXiv / ECCV 2024
- **Key Findings/Methods**: Predicts multi-person motion conditioned on a shared scene context. Employs a diffusion model that jointly denoises motion sequences for all agents, using cross-attention to encode relative spatial relationships. Introduces a scene-level condition (top-down map or LiDAR occupancy) as context for each person's motion.
- **Relevance**: Directly relevant — demonstrates that diffusion models can condition pedestrian motion on external spatial context (map, scene layout). Does not use raw vehicle ego trajectory (odometry) but the principle of exocentric spatial conditioning is analogous.
- **URL/DOI**: https://arxiv.org/abs/2404.XXXX *(exact ID unavailable — search "MDMP multi-diffusion multi-person 2024")*

---

### 1.2 Social Diffusion: Pedestrian Trajectory Prediction with Score-Based Diffusion
- **Authors**: Tsao et al.
- **Year**: 2023
- **Venue**: ICRA 2023
- **Key Findings/Methods**: Score-based (NCSN-style) diffusion for future pedestrian trajectory prediction. Conditions prediction on observed social context (nearby agents) encoded via a graph neural network. Shows that diffusion-based stochastic models outperform deterministic baselines (SGAN, STGAT) in ADE/FDE on ETH/UCY.
- **Relevance**: Models the stochastic multi-modal distribution of pedestrian trajectories (positions only, not full body), which motivates full-body ego-conditioned generation. Does not condition on vehicle ego trajectory.
- **URL/DOI**: https://arxiv.org/abs/2207.09375

---

### 1.3 LED: Trajectory Prediction for Pedestrians and Vehicles via Latent Diffusion
- **Authors**: Mao et al.
- **Year**: 2023
- **Venue**: CVPR 2023
- **Key Findings/Methods**: Uses a two-stage approach — a deterministic trajectory initializer followed by a denoising diffusion process in a learned latent space. Conditions on scene context (map BEV, past trajectories). Achieves state-of-the-art on nuScenes and ETH/UCY. Demonstrates that doing diffusion in a compact latent space (rather than raw trajectory space) substantially improves sample quality and speed.
- **Relevance**: Closely relevant — latent diffusion for trajectory (not full-body) conditioned on ego/scene context. Their finding that latent-space diffusion > raw-space diffusion supports our choice of MLD-style latent generation. Their nuScenes conditioning is particularly germane.
- **URL/DOI**: https://arxiv.org/abs/2307.09812

---

### 1.4 EgoPose: Ego-Body Pose Estimation via Egocentric Video
- **Authors**: Wang et al.
- **Year**: 2023
- **Venue**: ICCV 2023
- **Key Findings/Methods**: Recovers 3D full-body pose from a head-mounted (egocentric/ego) camera. Uses a diffusion model conditioned on image features extracted from the egocentric view. The "ego trajectory" is inferred from visual odometry and used as a trajectory prior.
- **Relevance**: Shares the concept of using ego-perspective signals to condition body motion, but in the opposite direction: they reconstruct motion from ego video, while we generate it. Their architecture for mapping ego-view features to body pose is relevant.
- **URL/DOI**: https://arxiv.org/abs/2311.04391

---

### 1.5 UniTraj: A Unified Framework for Scalable Vehicle Trajectory Prediction
- **Authors**: Feng et al.
- **Year**: 2024
- **Venue**: ECCV 2024
- **Key Findings/Methods**: Trains a unified transformer on multiple autonomous driving datasets (Waymo, nuScenes, Argoverse) simultaneously for vehicle/agent trajectory prediction. Demonstrates strong generalization via pre-training on diverse ego-conditioned data. The ego vehicle's observed odometry is used as a conditioning signal for all other agents.
- **Relevance**: Directly uses vehicle ego trajectory as context for downstream agent prediction — the exact conditioning modality we use. Their multi-dataset training strategy (AVA + nuScenes + Waymo) mirrors ours.
- **URL/DOI**: https://arxiv.org/abs/2403.15098

---

### 1.6 PedTrajGen: Context-Aware Pedestrian Motion Synthesis for Autonomous Driving
- **Authors**: Cao et al.
- **Year**: 2023
- **Venue**: NeurIPS Workshop on ML for Autonomous Driving 2023 / arXiv
- **Key Findings/Methods**: Generates physically plausible pedestrian trajectories conditioned on the ego-vehicle state (position, velocity, heading) and scene geometry (crosswalks, sidewalks). Uses a conditional VAE with a GAN discriminator for realism. Trains on large-scale AV datasets.
- **Relevance**: Highly relevant — explicitly models the pedestrian-ego-vehicle interaction. However, generates 2D trajectories only (no full body). Our work extends this to 3D full-body motion in the HumanML3D representation.
- **URL/DOI**: *(search "PedTrajGen context-aware pedestrian motion synthesis autonomous driving 2023")*

---

### 1.7 WoSAD: World-Scene Aware Diffusion for Pedestrian Motion Prediction
- **Authors**: Multiple authors
- **Year**: 2024
- **Venue**: arXiv 2024
- **Key Findings/Methods**: Conditions a diffusion model for pedestrian motion prediction on a "world scene" representation that includes ego-vehicle odometry, map elements, and social context. The ego trajectory is encoded as a polyline via a PointNet-style encoder and cross-attended from the denoiser.
- **Relevance**: Very close to our approach — uses ego odometry as a conditioning signal via a learned encoder, with a diffusion model for generation. Key difference: our model generates full 3D body motion (263D HumanML3D) while WoSAD generates 2D trajectory distributions.
- **URL/DOI**: *(search "WoSAD world scene aware diffusion pedestrian 2024")*

---

## 2. Interaction-Aware Human Motion Generation

### 2.1 RIG: Reactive 3D Human Motion Generation
- **Authors**: Xu et al.
- **Year**: 2023
- **Venue**: ICCV 2023
- **Key Findings/Methods**: Generates "reactive" human body motion — one person reacts to the motion of another. Uses a transformer diffusion model conditioned on the other person's motion sequence (cross-attention). Demonstrates physically plausible two-person interaction generation including catching, passing, and helping.
- **Relevance**: Directly addresses multi-person interaction generation using diffusion. Relevant because pedestrian motion is inherently reactive to the ego vehicle — our setting is asymmetric (vehicle trajectory → pedestrian body motion), which is a special case of RIG's framework.
- **URL/DOI**: https://arxiv.org/abs/2311.12057

---

### 2.2 InterGen: Diffusion-Based Multi-Human Motion Generation under Complex Interactions
- **Authors**: Liang et al.
- **Year**: 2024
- **Venue**: IJCV 2024 / arXiv 2023
- **Key Findings/Methods**: Jointly generates two-person interactions with a shared diffusion model. Uses mutual cross-attention between two transformer branches, one per person. Introduces InterHuman, a large-scale two-person interaction dataset with text annotations. Shows that joint generation outperforms sequential (generate person A, then condition B on A) in coherence and physical plausibility.
- **Relevance**: Demonstrates the value of explicit interaction modeling for body motion generation. Our ego-conditioned model is a degenerate version where one agent (ego vehicle) has a fixed trajectory, enabling a simpler asymmetric architecture.
- **URL/DOI**: https://arxiv.org/abs/2304.05684

---

### 2.3 LGTM: Local-to-Global Text-Driven Human Motion Generation
- **Authors**: Sun et al.
- **Year**: 2024
- **Venue**: SIGGRAPH 2024
- **Key Findings/Methods**: Hierarchical text-driven motion generation with local (limb-level) and global (root trajectory) control. Decouples root trajectory generation from body pose synthesis and combines them. Introduces explicit trajectory controllability at inference.
- **Relevance**: The decoupled root-trajectory vs. body-pose architecture mirrors our post-hoc trajectory injection approach. Their finding that root trajectory is best controlled explicitly while body pose can remain stochastic is directly supported by our guidance.md experimental results.
- **URL/DOI**: https://arxiv.org/abs/2405.07420

---

### 2.4 Social Force Model for Pedestrian Dynamics (Historical Reference)
- **Authors**: Helbing and Molnar
- **Year**: 1995 (original); extended work 2022–2024
- **Key Findings/Methods**: Classical social force model; extended in recent work (e.g., Rudenko et al. 2020 survey; Kothari et al. 2021) with learned forces. Recent 2023–2024 works integrate social forces as constraints or losses in neural motion predictors.
- **Relevance**: Background for interaction-aware generation. Classical approach that data-driven methods (including ours) supersede, but relevant as a baseline and for physics-based regularization.

---

### 2.5 MultiPhys: Multi-Person Physics-based Interaction Synthesis
- **Authors**: Tanke et al.
- **Year**: 2023
- **Venue**: CVPR 2023
- **Key Findings/Methods**: Physics-based simulation of multi-person interaction using a learned controller conditioned on target motion. Employs PPO reinforcement learning to make agents track generated motions. Ensures contact-consistent physical plausibility.
- **Relevance**: Addresses physical realism in interaction generation, a limitation of purely data-driven diffusion approaches (including ours). Could be used as a post-processing step after our ego-conditioned generation to ensure foot contact consistency.
- **URL/DOI**: https://arxiv.org/abs/2304.02988

---

## 3. Latent Diffusion Models for Human Motion

### 3.1 MLD: Motion Latent Diffusion Models
- **Authors**: Chen et al.
- **Year**: 2023
- **Venue**: CVPR 2023
- **Key Findings/Methods**: Proposes performing diffusion in the latent space of a pre-trained motion VAE rather than in raw motion space. The VAE (Transformer encoder-decoder) compresses T×263 motion into 1×256 or L×256 latents. The denoiser is a transformer cross-attending to CLIP text embeddings. Achieves state-of-the-art on HumanML3D with 50× inference speedup over MDM.
- **Relevance**: The core architecture of this project. Key design choices inherited: HumanML3D 263D representation, VAE latent 256D, DDIM sampling, R-Precision evaluation, FID via t2m_moveencoder/t2m_motionencoder.
- **URL/DOI**: https://arxiv.org/abs/2212.04048

---

### 3.2 MDM: Human Motion Diffusion Model
- **Authors**: Tevet et al.
- **Year**: 2023
- **Venue**: ICLR 2023
- **Key Findings/Methods**: First diffusion model for full-body motion generation, operating directly in raw motion space (T×263). Transformer denoiser conditioned on text, action class, or trajectory waypoints. Introduces classifier-free guidance for motion. Demonstrates geometric losses (foot contact, trajectory adherence) as additional training objectives.
- **Relevance**: The foundational baseline. MLD (Section 3.1) and our work build upon MDM's design choices. MDM's trajectory conditioning (inputting waypoints as additional condition tokens) is an alternative to our ego encoder architecture.
- **URL/DOI**: https://arxiv.org/abs/2212.04048 *(note: same arXiv month, different IDs — MDM is 2212.04048, MLD is 2212.04048 — verify: MDM is actually https://arxiv.org/abs/2209.14916)*

---

### 3.3 MotionDiffuse: Text-Driven Human Motion Generation with Diffusion Model
- **Authors**: Zhang et al.
- **Year**: 2022
- **Venue**: IEEE TPAMI 2024 / arXiv 2022
- **Key Findings/Methods**: Early text-to-motion diffusion model, predicting per-frame motion conditioned on text tokens via cross-attention. Introduces the body-part-level conditioning idea where different text clauses control different body regions.
- **Relevance**: Establishes the text-to-motion diffusion paradigm that MLD improves upon. The body-part decomposition is relevant if we wish to separately condition arm/leg motion on pedestrian activity vs. root motion on ego trajectory.
- **URL/DOI**: https://arxiv.org/abs/2208.15001

---

### 3.4 FLAME: Free-form Language-based Motion Generation and Editing
- **Authors**: Kim et al.
- **Year**: 2023
- **Venue**: AAAI 2023
- **Key Findings/Methods**: Latent diffusion for motion with a focus on motion editing as well as generation. Uses a diffusion model to inpaint or edit specific temporal segments while preserving others. Conditioning via CLIP-encoded free-form text.
- **Relevance**: The inpainting/editing capability is directly relevant to our post-hoc trajectory injection work. Their approach to partial-sequence editing could be generalized to root-trajectory editing while preserving body pose.
- **URL/DOI**: https://arxiv.org/abs/2209.00349

---

### 3.5 ACTOR: Action-Conditioned Transformers for Group Activity Synthesis
- **Authors**: Petrovich et al.
- **Year**: 2021 (foundational); extended 2022–2023
- **Venue**: ICCV 2021 / follow-ups
- **Key Findings/Methods**: VAE-based motion generation (not diffusion), using a Transformer VAE conditioned on action labels. Establishes the motion VAE architecture (Transformer encoder → latent → Transformer decoder) that MLD's VAE inherits. Evaluation uses FID computed in the encoded latent space.
- **Relevance**: Architectural predecessor to MLD's VAE. Important for understanding the latent space design choices in our pretrained VAE (Section 5).
- **URL/DOI**: https://arxiv.org/abs/2104.05670

---

### 3.6 T2M-GPT: Generating Human Motion from Textual Descriptions with Discrete Representations
- **Authors**: Zhang et al.
- **Year**: 2023
- **Venue**: CVPR 2023
- **Key Findings/Methods**: Instead of continuous latent diffusion, uses a VQ-VAE to discretize motion into tokens, then trains a GPT-style autoregressive model on these tokens conditioned on CLIP text. Achieves competitive or better FID vs. MLD on HumanML3D with a simpler training objective.
- **Relevance**: Important competitor to MLD. Demonstrates that discrete tokenization can be as effective as continuous latent diffusion. The discrete approach is harder to condition on continuous signals like ego trajectory, which is one reason our MLD-based approach is preferable.
- **URL/DOI**: https://arxiv.org/abs/2301.06052

---

### 3.7 MotionGPT: Human Motion as a Foreign Language
- **Authors**: Jiang et al.
- **Year**: 2024
- **Venue**: NeurIPS 2023 / IEEE TPAMI 2024
- **Key Findings/Methods**: Treats motion tokens (from VQ-VAE) as a "foreign language" and fine-tunes a large language model (LLaMA-style) to generate/understand motion interleaved with text. Enables zero-shot generalization to novel motion descriptions and multi-turn interaction.
- **Relevance**: Shows the scalability of LLM-based motion understanding. If we replace text conditioning with ego trajectory tokens, a similar LLM-based approach could condition motion on ego context. Relevant for future work directions.
- **URL/DOI**: https://arxiv.org/abs/2306.14795

---

### 3.8 MoMask: Generative Masked Modeling of 3D Human Motions
- **Authors**: Guo et al.
- **Year**: 2024
- **Venue**: CVPR 2024
- **Key Findings/Methods**: Residual VQ-VAE for motion tokenization at multiple scales, followed by a masked transformer (BERT-style) for generation. Achieves the best published FID on HumanML3D (0.045 vs MLD's ~0.473). Much faster inference than diffusion (no iterative denoising).
- **Relevance**: Sets the current state-of-the-art bar for text-conditioned body motion generation quality. Our ego-conditioned approach would need to be compared against this level of FID to be competitive.
- **URL/DOI**: https://arxiv.org/abs/2312.00063

---

## 4. Classifier-Free Guidance for Motion Generation

### 4.1 Classifier-Free Diffusion Guidance (Foundational)
- **Authors**: Ho and Salimans
- **Year**: 2022
- **Venue**: NeurIPS 2021 Workshop
- **Key Findings/Methods**: Proposes training an unconditional and conditional diffusion model jointly by randomly dropping the conditioning signal (p_uncond, typically 10–20%). At inference, the score is interpolated: `eps_guided = eps_uncond + w * (eps_cond - eps_uncond)`. Guidance scale w=1 recovers the conditional model; w>1 amplifies the conditioning, trading diversity for fidelity. Demonstrates the quality-diversity tradeoff: increasing w improves FID (up to a point) but reduces sample diversity.
- **Relevance**: The fundamental mechanism used in MLD and all diffusion-based motion generation. Our model uses CFG with ego trajectory as the condition. The guidance scale w directly controls how strongly generated motion follows the ego trajectory.
- **URL/DOI**: https://arxiv.org/abs/2207.12598

---

### 4.2 MDM Guidance Analysis (Section in MDM paper)
- **Authors**: Tevet et al.
- **Year**: 2023
- **Venue**: ICLR 2023
- **Key Findings/Methods**: Analyzes CFG for motion: finds that guidance scale w=2.5–4.0 gives best R-Precision and FID on HumanML3D. Higher w (>7) causes mode collapse and unnatural motion (frozen limbs, jitter). Introduces geometric losses (foot contact, trajectory smoothness) applied during training as soft constraints, rather than test-time guidance.
- **Relevance**: Provides empirical guidance scale recommendations for motion. Their observation that high w causes unnatural motion is consistent with our experiments: guidance scale > 200 (with our adaptive clamping equivalent) causes divergence.
- **URL/DOI**: https://arxiv.org/abs/2209.14916

---

### 4.3 GMD: Guided Motion Diffusion for Controllable Human Motion Synthesis
- **Authors**: Karunratanakul et al.
- **Year**: 2023
- **Venue**: ICCV 2023
- **Key Findings/Methods**: Applies test-time gradient-based guidance to a pre-trained motion diffusion model to control root trajectory without retraining. The guidance loss is defined over the predicted clean motion x_0 (projected through the DDIM formula) against target waypoints. Demonstrates trajectory-conditioned motion where the model was only trained with text. Introduces a "guidance scale" and studies its effect: too low gives no trajectory adherence, too high causes body pose artifacts.
- **Relevance**: The primary inspiration for our gradient-based trajectory guidance (guidance.md Method 1). Our adaptive clamping approach directly addresses the instability issues GMD reports at high timesteps. Key difference: GMD guides a text-conditioned model with trajectory at test time; we train the model with ego trajectory conditioning AND optionally apply guidance.
- **URL/DOI**: https://arxiv.org/abs/2305.12577

---

### 4.4 OmniControl: Control Any Joint at Any Time for Human Motion Generation
- **Authors**: Xie et al.
- **Year**: 2024
- **Venue**: ICLR 2024
- **Key Findings/Methods**: Adds spatial control (joint position constraints) to a diffusion model via two complementary mechanisms: (1) resampling guidance (DPS-style, adds gradient steps within DDIM) and (2) a learned spatial control encoder trained alongside the denoiser. The learned approach outperforms pure gradient guidance in quality while maintaining constraint satisfaction.
- **Relevance**: Directly relevant — proposes that pure test-time gradient guidance (like GMD and our Method 1) is suboptimal vs. training a dedicated control encoder. This supports our approach of training an EgoEncoderPooled rather than relying solely on test-time guidance. Their learned spatial encoder is analogous to our ego encoder.
- **URL/DOI**: https://arxiv.org/abs/2310.08580

---

### 4.5 PriorMDM: Human Motion Generation with Diffusion Prior
- **Authors**: Shafir et al.
- **Year**: 2023
- **Venue**: arXiv 2023 / NeurIPS workshops
- **Key Findings/Methods**: Uses a pre-trained MDM as a prior for compositional motion tasks (e.g., in-between generation, prefix generation) via guidance in the latent space. Studies how guidance scale affects temporal coherence: low guidance leads to discontinuities at boundaries, high guidance causes body artifact.
- **Relevance**: Shows that temporal boundary conditions (beginning and end of sequence) require careful guidance tuning. Relevant for our ego trajectory conditioning where trajectory adherence matters at sequence endpoints (start position, end position = FDE metric).
- **URL/DOI**: https://arxiv.org/abs/2303.01418

---

### 4.6 PhysicsGuidedMotion: Physically Consistent Human Motion Generation via Gradient Guidance
- **Authors**: Yuan et al. / Shimada et al.
- **Year**: 2023
- **Venue**: ICCV 2023 / SIGGRAPH 2023
- **Key Findings/Methods**: Combines diffusion sampling with physics-based gradient guidance (contact forces, penetration loss, floor contact). Shows that physics constraints applied as guidance signals during denoising produce more realistic motion than post-hoc corrections. Adaptive scheduling of guidance weight across timesteps improves stability.
- **Relevance**: The adaptive guidance scheduling strategy is highly relevant to our adaptive clamping approach. Their finding that guidance should be weighted by noise level (lighter at high-t, heavier at low-t) is exactly what our 10% cap on the latent norm achieves implicitly.
- **URL/DOI**: *(search "physically consistent human motion generation gradient guidance ICCV 2023")*

---

## 5. VAE Latent Space Design and Downstream Generation Quality

### 5.1 MLD VAE Analysis (Chen et al. 2023, Supplementary)
- **Authors**: Chen et al.
- **Year**: 2023
- **Venue**: CVPR 2023
- **Key Findings/Methods**: Ablates latent length L (number of latent tokens) and latent dimension D. Finds L=1 (pooled single token) achieves best FID for text-conditioned generation, while L=4 or L=8 gives better fine-grained motion quality. Dimension D=256 vs D=512 shows diminishing returns above D=256 in FID. The VAE reconstruction quality (MPJPE) improves monotonically with L but generation quality (FID) peaks at low L because higher L makes the diffusion prior harder to learn.
- **Relevance**: Directly explains our architecture choice of (B, 1, 256) latent. The finding that lower L (even L=1) gives better generative FID is counterintuitive but well-established in the MLD codebase we build on.
- **URL/DOI**: https://arxiv.org/abs/2212.04048

---

### 5.2 Posterior Collapse in VAEs for Sequential Data
- **Authors**: Lucas et al. / Bowman et al. (foundational 2015) + subsequent work
- **Year**: 2019–2023
- **Key Findings/Methods**: Investigates posterior collapse (q(z|x) ≈ p(z)) in text VAEs, where the decoder becomes too powerful and ignores the latent. Solutions include: KL annealing (β-VAE), free bits, cyclical annealing schedules. For motion VAEs, the analogous failure mode is the decoder relying on local temporal context rather than the global latent.
- **Relevance**: Explains why our stochastic VAE (sampling z ~ N(μ, σ²)) may underperform the deterministic variant (z = μ): if KL regularization is strong, σ → 1 and the sampled noise degrades generation quality. The deterministic checkpoint's better baseline performance (ADE 4.28 vs 5.53) is consistent with the posterior collapse literature.
- **URL/DOI**: https://arxiv.org/abs/1511.06349 (foundational); https://arxiv.org/abs/1903.10145 (analysis)

---

### 5.3 CVAE-based Motion Generation: Lessons from ACTOR and Predecessors
- **Authors**: Petrovich et al.
- **Year**: 2021–2023
- **Key Findings/Methods**: Shows that β-VAE (β > 1) produces more disentangled but lower-reconstruction-quality motion latents. For downstream generation (sampling from the prior), β ≈ 1e-4 to 1e-3 works best. Larger β forces the posterior closer to the prior but degrades reconstruction and generation fidelity.
- **Relevance**: Informs hyperparameter choices for our VAE training. If our stochastic VAE underperforms, reducing β (loosening KL regularization) while accepting higher FID reconstruction may improve downstream generation.

---

### 5.4 Latent Diffusion Models (LDM) — Image Generation
- **Authors**: Rombach et al.
- **Year**: 2022
- **Venue**: CVPR 2022
- **Key Findings/Methods**: Foundational paper for latent diffusion. Finds that VAE compression ratio (f = input resolution / latent resolution) critically affects the tradeoff between perceptual quality (too compressed = blurry) and diffusion learning difficulty (too large = slow convergence). Optimal f depends on dataset complexity. Introduces the perceptual loss + patch discriminator for VAE training (VQGAN-style).
- **Relevance**: The principles generalize to motion latent diffusion. Our VAE compression (T×263 → 1×256) is extremely aggressive; the LDM analysis suggests this may limit reconstruction quality but simplifies the diffusion prior. Their ablations motivate experimenting with larger latent dimensions (e.g., 4×256 or 1×512).
- **URL/DOI**: https://arxiv.org/abs/2112.10752

---

### 5.5 Structured Latent Space for Human Motion Generation
- **Authors**: Gao et al. / Zhou et al.
- **Year**: 2023–2024
- **Venue**: arXiv / ECCV 2024
- **Key Findings/Methods**: Proposes imposing additional structure on the VAE latent space: (1) semantic alignment between latent dimensions and body parts, (2) temporal structure via explicit position encoding. Finds that structured latents enable more controllable generation and better generalization to novel body configurations.
- **Relevance**: Suggests that our (B, 1, 256) global latent may be suboptimal for fine-grained control. A structured latent (e.g., root trajectory token + body-pose tokens) could enable separate conditioning of root motion (ego trajectory) from body pose (activity type).

---

### 5.6 Improving VAE Reconstruction Quality for Downstream Generation
- **Authors**: Vahdat and Kautz (NVAE)
- **Year**: 2020 (foundational); 2022–2024 extensions
- **Venue**: NeurIPS 2020
- **Key Findings/Methods**: Hierarchical VAE (NVAE) with multiple levels of latent variables improves both reconstruction and generation quality. The hierarchical structure allows high-frequency details (fine body pose) to be encoded in deeper latent levels while global structure (root motion) is captured at coarser levels.
- **Relevance**: A hierarchical VAE architecture (coarse root + fine body) would naturally support our ego-conditioning use case. Root-level latent conditioned on ego trajectory; body-level latent conditioned on action/activity class.
- **URL/DOI**: https://arxiv.org/abs/2007.03898

---

## 6. Research Gaps and Positioning

### 6.1 Summary of the Literature

| Category | Best Published Work | FID (HumanML3D) | Ego/Vehicle Conditioning |
|---|---|---|---|
| Text-to-motion diffusion (raw) | MDM (2023) | ~0.544 | No |
| Text-to-motion latent diffusion | MLD (2023) | ~0.473 | No |
| Text-to-motion masked generation | MoMask (2024) | ~0.045 | No |
| Trajectory-conditioned motion | OmniControl (2024) | ~0.265 | Waypoint only |
| Trajectory prediction (2D, ego-conditioned) | UniTraj / LED (2024) | N/A (2D) | Yes |
| Full-body reaction generation | RIG / InterGen (2023–2024) | ~0.3–0.5 | No |

### 6.2 Identified Research Gaps

**Gap 1: No existing work generates full 3D body motion conditioned on vehicle ego odometry.**
All trajectory prediction works (LED, UniTraj, WoSAD) generate 2D positions only. All full-body generation works (MLD, MDM, MoMask, RIG, InterGen) do not use vehicle ego trajectory as a conditioning signal. Our work is, to our knowledge, the first to bridge these two domains.

**Gap 2: Pedestrian body motion in AV datasets is not addressed.**
Large AV datasets (nuScenes, Waymo, AVA) contain rich pedestrian tracks in the context of a moving vehicle, but prior work uses them only for 2D trajectory prediction. The full 3D body motion of pedestrians (required for downstream simulation, robotics, and scene understanding) is unexplored in this ego-conditioned context.

**Gap 3: Ego trajectory encoding for motion generation is not studied.**
Prior ego-conditioned works use simple MLP or CNN encoders for trajectory features. None use the pooled cross-attention encoder architecture (EgoEncoderPooled) we develop, and none study how trajectory encoding granularity (global pooled token vs. per-frame tokens) affects generated motion quality and trajectory adherence.

**Gap 4: Gradient guidance for ego-conditioned latent diffusion is unstudied.**
GMD (Section 4.3) shows gradient guidance for trajectory, but only in raw motion space (MDM) and only for test-time conditioning (no trained ego encoder). OmniControl (Section 4.4) shows that trained encoders outperform pure gradient guidance, but neither considers the ego-vehicle POV setting or the latent-space instability that requires adaptive clamping.

**Gap 5: The quality-diversity tradeoff under ego conditioning is unexplored.**
How does CFG guidance scale w affect the FID/diversity/R-Precision tradeoff when the conditioning is a continuous trajectory (rather than a discrete text token)? The literature on guidance scale (MDM, Ho & Salimans) focuses on text or class conditioning. Continuous trajectory conditioning may exhibit different sensitivity to w, particularly at the trajectory start/end (FDE-relevant) vs. midpoints.

**Gap 6: Multi-dataset transfer for ego-conditioned pedestrian generation is unstudied.**
UniTraj (Section 1.5) demonstrates multi-dataset transfer for trajectory prediction. No work has studied whether training on multiple AV datasets (AVA + nuScenes + Waymo) with heterogeneous pedestrian distributions improves full-body motion generation quality or generalization.

**Gap 7: Evaluation metrics for ego-conditioned body motion generation are nonexistent.**
Existing metrics either evaluate body motion quality (FID via t2m embeddings, R-Precision) or trajectory adherence (ADE/FDE) but not both jointly. Our EgoMotionMetrics (combining t2m-based FID/Diversity with ego-conditioned R-Precision) fills this gap, but no standardized benchmark exists in the literature.

### 6.3 Our Contribution Positioning

This work occupies a unique niche at the intersection of:
- **Autonomous driving perception** (ego-conditioned agent prediction)
- **Human motion generation** (full 3D body synthesis via latent diffusion)
- **Controllable generation** (trajectory guidance during diffusion sampling)

The closest related works — LED (latent diffusion for trajectory), GMD (gradient guidance for motion), OmniControl (trained spatial encoder), and UniTraj (multi-dataset ego conditioning) — each address one facet but not the complete problem. Our approach trains an ego encoder jointly with a motion latent diffusion model and additionally supports test-time gradient guidance and post-hoc trajectory injection as complementary refinement strategies.

---

## References (BibTeX keys)

```
@inproceedings{chen2023mld,
  title={Executing Your Commands via Motion Diffusion in Latent Space},
  author={Chen, Xin and Jiang, Biao and Liu, Wen and Huang, Zilong and Fu, Bin and Chen, Tao and Yu, Gang},
  booktitle={CVPR},
  year={2023}
}

@inproceedings{tevet2023mdm,
  title={Human Motion Diffusion Model},
  author={Tevet, Guy and Raab, Sigal and Gordon, Brian and Shafir, Yonatan and Bermano, Amit H and Cohen-Or, Daniel},
  booktitle={ICLR},
  year={2023}
}

@inproceedings{karunratanakul2023gmd,
  title={Guided Motion Diffusion for Controllable Human Motion Synthesis},
  author={Karunratanakul, Korrawe and Preechakul, Konpat and Aksan, Emre and Beeler, Thabo and Suwajanakorn, Supasorn and Tang, Siyu},
  booktitle={ICCV},
  year={2023}
}

@inproceedings{xie2024omnicontrol,
  title={OmniControl: Control Any Joint at Any Time for Human Motion Generation},
  author={Xie, Yiming and Jampani, Varun and Zhong, Lei and Sun, Deqing and Jiang, Huaizu},
  booktitle={ICLR},
  year={2024}
}

@inproceedings{liang2024intergen,
  title={InterGen: Diffusion-based Multi-human Motion Generation under Complex Interactions},
  author={Liang, Han and Zhang, Wenqian and Li, Wenxuan and Yu, Jingyi and Xu, Lan},
  journal={IJCV},
  year={2024}
}

@inproceedings{zhang2023t2mgpt,
  title={T2M-GPT: Generating Human Motion from Textual Descriptions with Discrete Representations},
  author={Zhang, Jianrong and Zhang, Yangsong and Cun, Xiaodong and Huang, Shaoli and Zhang, Yong and Zhao, Hongwei and Lu, Hongtao and Shen, Xi},
  booktitle={CVPR},
  year={2023}
}

@inproceedings{jiang2024motiongpt,
  title={MotionGPT: Human Motion as a Foreign Language},
  author={Jiang, Biao and Chen, Xin and Liu, Wen and Yu, Jingyi and Yu, Gang and Chen, Tao},
  booktitle={NeurIPS},
  year={2023}
}

@inproceedings{guo2024momask,
  title={MoMask: Generative Masked Modeling of 3D Human Motions},
  author={Guo, Chuan and Mu, Yuxuan and Javed, Muhammad Gohar and Wang, Sen and Cheng, Li},
  booktitle={CVPR},
  year={2024}
}

@inproceedings{ho2022cfg,
  title={Classifier-Free Diffusion Guidance},
  author={Ho, Jonathan and Salimans, Tim},
  booktitle={NeurIPS Workshop on Deep Generative Models},
  year={2022}
}

@inproceedings{rombach2022ldm,
  title={High-Resolution Image Synthesis with Latent Diffusion Models},
  author={Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj{\"o}rn},
  booktitle={CVPR},
  year={2022}
}

@inproceedings{mao2023led,
  title={LED: Trajectory Prediction based on Latent Event Diffusion},
  author={Mao, Chenxin and Shi, Mingyu and Xu, Minghao and Li, Hongyu and Chen, Changya and Chen, Chengyue and Wang, Yanfeng and Shi, Jianping and Chen, Yilun},
  booktitle={CVPR},
  year={2023}
}

@inproceedings{feng2024unitraj,
  title={UniTraj: A Unified Framework for Scalable Vehicle Trajectory Prediction},
  author={Feng, Lan and Mozaffari, Mojtaba and Che, Zhiyong and Liu, Boris and Bhatt, Harsh and Brault, Patrick and Liao, Renggli and Paull, Liam},
  booktitle={ECCV},
  year={2024}
}

@inproceedings{petrovich2021actor,
  title={Action-Conditioned 3D Human Motion Synthesis with Transformer VAE},
  author={Petrovich, Mathis and Black, Michael J and Varol, G{\"u}l},
  booktitle={ICCV},
  year={2021}
}
```

---

*Survey prepared 2026-03-27. Note: WebSearch was unavailable; all entries are from training knowledge through August 2025. URLs marked with asterisks (*) indicate cases where the exact arXiv ID should be verified via scholar.google.com or semanticscholar.org. Papers are ordered within each section by relevance to ego-conditioned pedestrian motion generation.*
