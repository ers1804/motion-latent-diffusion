"""
EgoMotion Dataset for MLD

Loads ego trajectory + pedestrian motion pairs for ego-conditioned generation.

Data format: Single JSON file per sample named `{scene_id}_{object_id}.json`
JSON structure:
    - scene_id: str
    - object_id: str  
    - ego_in_ped_frame: [[x, y, z], ...] - ego trajectory, use (x, z) as lateral motion
    - ped_in_ped_frame: [[x, y, z], ...] - pedestrian trajectory (not used for now)
    - vectors_263: [[263 features], ...] - motion features, shape (T, 263)
"""

import json
import os
from os.path import join as pjoin
from pathlib import Path
from typing import Dict, List, Optional, Callable
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pytorch_lightning as pl

from mld.data.humanml.scripts.motion_process import recover_from_ric


class EgoMotionDataset(Dataset):
    """
    Dataset for ego-pedestrian motion pairs.
    
    Supports MULTIPLE data roots for combined training:
    - data_root can be a single path: "data/diffusion/waymo"
    - OR a list of paths: ["data/diffusion/waymo", "data/diffusion/nuscenes", ...]
    
    Expected data structure per root:
    - data_root/
        - train/
            - scene001_obj01.json
            - ...
        - val/
        - test/
    
    Each JSON contains: ego_in_ped_frame, vectors_263, etc.
    """

    def __init__(
        self,
        data_root,  # str or List[str] - single path or list of paths
        split: str = "train",
        mean: np.ndarray = None,           # Motion mean (263,)
        std: np.ndarray = None,            # Motion std (263,)
        ego_mean: np.ndarray = None,       # Ego mean (2,) - optional
        ego_std: np.ndarray = None,        # Ego std (2,) - optional
        ego_scale: float = 50.0,           # Simple scaling (meters) if no mean/std
        max_motion_length: int = 196,
        min_motion_length: int = 40,
        max_ego_length: int = 100,
        fps: int = 20,
        debug: bool = False,
        overfit: bool = False,
        split_list_root: Optional[str] = None,
        interaction_crop: bool = False,
        interaction_weighted_sampling: bool = False,
        captions_path: Optional[str] = None,
        caption_vocab: str = "ego",
        **kwargs
    ):
        self.interaction_crop = interaction_crop
        self.interaction_weighted_sampling = interaction_weighted_sampling
        # Optional synthesized captions (text-conditioned baselines, item 10b).
        # Default None => the historical "Placeholder" text, so existing
        # ego-conditioned runs are unaffected.
        self.caption_vocab = caption_vocab
        self.captions = None
        if captions_path and os.path.exists(captions_path):
            with open(captions_path, "r") as f:
                self.captions = json.load(f)
            print(f"[EgoMotionDataset] Loaded {len(self.captions)} captions "
                  f"({caption_vocab} vocabulary) from {captions_path}")

        # Support both single path and list of paths
        if isinstance(data_root, str):
            self.data_roots = [data_root]
        else:
            self.data_roots = data_root
            
        self.split = split
        self.mean = mean                   # Motion normalization
        self.std = std
        self.ego_mean = ego_mean           # Ego normalization (optional)
        self.ego_std = ego_std
        self.ego_scale = ego_scale         # Fallback: simple divide by scale
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.max_ego_length = max_ego_length
        self.fps = fps
        self.debug = debug
        self.overfit = overfit
        self.split_list_root = split_list_root

        # Make ego stats compatible with ego trajectory dimensionality (XZ => 2D)
        self.ego_mean, self.ego_std = self._prepare_ego_stats(self.ego_mean, self.ego_std)

        # Load all sample paths from ALL data roots
        self.sample_paths = self._load_sample_paths()
        
        if self.debug:
            # Use only 100 samples for debugging
            self.sample_paths = self.sample_paths[:1000]

        if not self.debug and self.overfit:
            self.sample_paths = self.sample_paths[:1]
            print(f"Overfitting to Sample {self.sample_paths[0]}.")
        
        print(f"[EgoMotionDataset] Loaded {len(self.sample_paths)} samples for {split} from {len(self.data_roots)} sources")
        for root in self.data_roots:
            count = len([p for p in self.sample_paths if root in p])
            print(f"  - {root}: {count} samples")
        
        # Compute interaction weights for weighted sampling
        self.sample_weights = None
        if self.interaction_weighted_sampling and split == "train":
            self.sample_weights = self._compute_interaction_weights()

        # Dataset info (required by MLD)
        self.nfeats = 263  # Motion feature dimension
        self.njoints = 22  # Joint count (HumanML3D format)

    def _load_sample_paths(self) -> List[str]:
        """
        Find all sample files from ALL data roots.

        Priority:
          1) If split list exists at <split_list_root>/<split>.txt, use ONLY
             the listed scenarios.
          2) Else fallback to scanning <root>/<split>/*.{json,npy}.
        """
        # Optional: force split membership from txt list (e.g. interactive set)
        split_list_path = None
        if self.split_list_root:
            candidate = pjoin(self.split_list_root, f"{self.split}.txt")
            if os.path.exists(candidate):
                split_list_path = candidate

        if split_list_path is not None:
            listed_paths = self._load_paths_from_split_file(split_list_path)
            print(f"[EgoMotionDataset] Using split list: {split_list_path}")
            return sorted(listed_paths)

        all_paths = []
        
        for data_root in self.data_roots:
            split_dir = pjoin(data_root, self.split)
            
            if not os.path.exists(split_dir):
                # Try flat structure with split file
                split_file = pjoin(data_root, f"{self.split}.txt")
                if os.path.exists(split_file):
                    with open(split_file, "r") as f:
                        sample_names = [line.strip() for line in f.readlines()]
                    paths = [pjoin(data_root, f"{name}.json") for name in sample_names]
                    all_paths.extend(paths)
                else:
                    print(f"Warning: Neither {split_dir} nor {split_file} found")
                continue
            
            # Get all JSON or npy files files in split directory
            if 'humanml' in data_root.lower():
                # HumanML3D has .npy files
                pattern = pjoin(split_dir, "*.npy")
            else:
                pattern = pjoin(split_dir, "*.json")
            paths = glob(pattern)
            all_paths.extend(paths)
        
        return sorted(all_paths)

    def _load_paths_from_split_file(self, split_file: str) -> List[str]:
        """Resolve scenario names from split file to absolute paths.

        Expected line formats (examples):
          - A0001_33.json  (prefix identifies dataset root)
          - 0001_33.json   (no prefix; searched across all roots)
          - A0001_33       (extension optional)
        """
        with open(split_file, "r") as f:
            names = [line.strip() for line in f.readlines() if line.strip()]

        # Infer dataset prefix for each configured root
        prefix_to_root = {}
        for root in self.data_roots:
            low = root.lower()
            if "ava" in low:
                prefix_to_root["A"] = root
            elif "waymo" in low:
                prefix_to_root["W"] = root
            elif "nuscenes" in low or "nuScenes" in root:
                prefix_to_root["N"] = root
            elif "humanml" in low:
                prefix_to_root["H"] = root

        resolved = []
        missing = []

        for raw_name in names:
            token = raw_name.strip()
            stem = Path(token).stem
            suffix = Path(token).suffix.lower()

            # Candidate filename list (keep token ext first, then json/npy)
            filename_candidates = []
            if suffix:
                filename_candidates.append(Path(token).name)
            else:
                filename_candidates.extend([f"{stem}.json", f"{stem}.npy"])

            # 1) Prefix-aware resolution (A/W/N/H)
            root_candidates = []
            if len(stem) >= 2 and stem[0].upper() in prefix_to_root:
                ds_prefix = stem[0].upper()
                root_candidates.append(prefix_to_root[ds_prefix])

                # Remove leading dataset prefix from scenario id
                stripped_stem = stem[1:]
                stripped_candidates = []
                if suffix:
                    stripped_candidates.append(f"{stripped_stem}{suffix}")
                else:
                    stripped_candidates.extend([f"{stripped_stem}.json", f"{stripped_stem}.npy"])
                filename_candidates = filename_candidates + stripped_candidates

            # 2) Fallback search across all roots
            if not root_candidates:
                root_candidates = list(self.data_roots)

            found = None
            # Deduplicate while preserving order
            seen = set()
            uniq_filenames = []
            for fn in filename_candidates:
                if fn not in seen:
                    uniq_filenames.append(fn)
                    seen.add(fn)

            for root in root_candidates:
                split_dir = pjoin(root, self.split)
                for fname in uniq_filenames:
                    candidate_path = pjoin(split_dir, fname)
                    if os.path.exists(candidate_path):
                        found = candidate_path
                        break
                if found is not None:
                    break

            if found is None and root_candidates != list(self.data_roots):
                # If prefix-directed lookup fails, try global fallback once
                for root in self.data_roots:
                    split_dir = pjoin(root, self.split)
                    for fname in uniq_filenames:
                        candidate_path = pjoin(split_dir, fname)
                        if os.path.exists(candidate_path):
                            found = candidate_path
                            break
                    if found is not None:
                        break

            if found is not None:
                resolved.append(found)
            else:
                missing.append(raw_name)

        if missing:
            print(f"[EgoMotionDataset] WARNING: {len(missing)} entries from {split_file} were not found.")
            for item in missing[:10]:
                print(f"  - missing: {item}")
            if len(missing) > 10:
                print("  - ...")

        return resolved

    def _compute_interaction_weights(self) -> np.ndarray:
        """Compute per-sample interaction weights for WeightedRandomSampler.

        The weight for each scenario is derived from the same interaction score
        used in dataset_statistics.ipynb:
            score = ped_travel * (pct_within_5m / 100) * (1 + heading_change / 180)

        Scores are clipped to a small floor value so non-interactive samples are
        still seen occasionally, then normalized to sum to len(dataset).
        """
        print("[EgoMotionDataset] Computing interaction weights (this scans all JSON files once)...")
        weights = np.ones(len(self.sample_paths), dtype=np.float64)

        for i, fpath in enumerate(self.sample_paths):
            if not fpath.endswith(".json"):
                continue
            try:
                with open(fpath, "r") as f:
                    data = json.load(f)

                ego = np.array(data["ego_in_ped_frame"], dtype=np.float32)
                ped = np.array(data["ped_in_ped_frame"], dtype=np.float32)
                root_joint = ped[:, 0, :]  # pelvis, (T, 3)

                T = min(len(ego), len(root_joint))
                ego_xz = ego[:T, [0, 2]]
                root_xz = root_joint[:T, [0, 2]]

                # Pedestrian total travel distance (ground plane)
                ped_travel = np.linalg.norm(np.diff(root_xz, axis=0), axis=1).sum()

                # Ego–ped distance per frame
                dists = np.linalg.norm(ego_xz - root_xz, axis=1)
                pct_within_5m = (dists < 5).mean() * 100

                # Bearing heading change
                bearing = root_xz - ego_xz
                angles = np.arctan2(bearing[:, 1], bearing[:, 0])
                angle_diffs = np.abs(np.diff(angles))
                angle_diffs = np.minimum(angle_diffs, 2 * np.pi - angle_diffs)
                heading_change = np.degrees(angle_diffs.sum())

                score = ped_travel * (pct_within_5m / 100.0) * (1 + heading_change / 180.0)
                weights[i] = max(score, 0.01)
            except Exception:
                weights[i] = 0.01

        # Normalize so weights sum to N (keeps effective epoch size the same)
        weights = weights / weights.sum() * len(weights)
        n_high = (weights > 1.0).sum()
        print(f"[EgoMotionDataset] Interaction weights: {n_high}/{len(weights)} samples above average weight")
        return weights

    def _caption_for(self, json_path: str) -> str:
        """Look up a synthesized caption by '<source>/<split>/<stem>' key."""
        if not self.captions:
            return "Placeholder"
        p = Path(json_path)
        key = f"{p.parent.parent.name}/{p.parent.name}/{p.stem}"
        entry = self.captions.get(key)
        if entry is None:
            return "a person walks forward."   # neutral fallback
        return entry.get(self.caption_vocab, "a person walks forward.")

    def _load_sample(self, json_path: str) -> Dict:
        """
        Load a single sample from JSON file.
        
        Returns dict with:
            - ego: (T_ego, 2) - lateral motion (x, z)
            - motion: (T_motion, 263) - motion features
        """
        if json_path.endswith('.npy'):
            motion = np.load(json_path)
            ego_2d = np.zeros_like(motion[:, :2])  # Placeholder ego (not provided in HumanML3D)
            return {
            "ego": ego_2d,
            "motion": motion,
            "ped_root_xz": np.zeros_like(ego_2d),
            "scene_id": "",
            "object_id": "",
        }
        else:
            with open(json_path, "r") as f:
                data = json.load(f)
            
            # Extract ego trajectory: [[x, y, z], ...] -> (T, 2) using x, z
            ego_3d = np.array(data["ego_in_ped_frame"], dtype=np.float32)
            ego_2d = ego_3d[:, [0, 2]]  # Take x (index 0) and z (index 2)
            
            # Extract motion features: [[263], ...] -> (T, 263)
            motion = np.array(data["vectors_263"], dtype=np.float32)

            # Extract pedestrian root joint XZ for ego-ped distance computation
            ped_joints = np.array(data["ped_in_ped_frame"], dtype=np.float32)  # (T, 22, 3)
            ped_root_xz = ped_joints[:, 0, [0, 2]]  # (T, 2)

            return {
                "ego": ego_2d,
                "motion": motion,
                "ped_root_xz": ped_root_xz,
                "scene_id": data.get("scene_id", ""),
                "object_id": data.get("object_id", ""),
            }

    def _normalize_motion(self, motion: np.ndarray) -> np.ndarray:
        """Normalize motion features using mean and std."""
        if self.mean is not None and self.std is not None:
            motion = (motion - self.mean) / (self.std + 1e-8)
        return motion

    def _normalize_ego(self, ego: np.ndarray) -> np.ndarray:
        """
        Normalize ego trajectory.
        
        Option 1: Use ego_mean/ego_std if provided (computed from data)
        Option 2: Simple scaling by ego_scale (e.g., divide by 50 meters)
        """
        if self.ego_mean is not None and self.ego_std is not None:
            # Full normalization (stats already projected/aligned in _prepare_ego_stats)
            ego = (ego - self.ego_mean) / (self.ego_std + 1e-8)
        else:
            # Simple scaling: divide by scale factor
            # This puts values roughly in [-1, 1] range if ego is within ±scale meters
            ego = ego / self.ego_scale
        return ego

    def _prepare_ego_stats(self, ego_mean, ego_std):
        """Prepare ego mean/std to match ego feature dims (typically XZ => 2).

        Accepts stats saved as shape (3,), (1,3), or (2,), and returns vectors
        compatible with ego arrays of shape (T, 2).
        """
        if ego_mean is None or ego_std is None:
            return None, None

        mean = np.asarray(ego_mean, dtype=np.float32).reshape(-1)
        std = np.asarray(ego_std, dtype=np.float32).reshape(-1)

        # If stats are XYZ, project to XZ to match ego_2d = ego_3d[:, [0, 2]].
        if mean.shape[0] == 3 and std.shape[0] == 3:
            mean = mean[[0, 2]]
            std = std[[0, 2]]
        elif mean.shape[0] != 2 or std.shape[0] != 2:
            print(
                f"[EgoMotionDataset] WARNING: incompatible ego stats shapes "
                f"mean={mean.shape}, std={std.shape}; falling back to EGO_SCALE normalization."
            )
            return None, None

        return mean, std

    def _pad_or_crop(
        self, 
        sequence: np.ndarray, 
        max_length: int
    ) -> tuple:
        """
        Pad sequence to max_length or crop if too long.
        Returns: (padded_sequence, actual_length)
        """
        actual_length = len(sequence)
        
        if actual_length >= max_length:
            # Crop from the beginning (keep recent frames)
            sequence = sequence[:max_length]
            actual_length = max_length
        else:
            # Pad with zeros at the end
            pad_length = max_length - actual_length
            padding = np.zeros((pad_length, sequence.shape[-1]), dtype=sequence.dtype)
            sequence = np.concatenate([sequence, padding], axis=0)
        
        return sequence, actual_length

    def _find_interaction_center(self, ego_ped_dists: np.ndarray, actual_length: int) -> int:
        """Find the frame index of minimum ego-pedestrian distance.

        Args:
            ego_ped_dists: (T,) precomputed per-frame ego-ped distances (raw, unnormalized)
            actual_length: number of valid frames
        Returns the frame index within [0, actual_length).
        """
        return int(np.argmin(ego_ped_dists[:actual_length]))

    def _pad_or_crop_ego_motion(
        self, 
        sequence: np.ndarray,
        ego_sequence: np.ndarray, 
        max_length: int,
        ego_ped_dists: Optional[np.ndarray] = None,
    ) -> tuple:
        """
        Pad/crop motion and ego together with aligned indexing.
        Ego is first truncated/padded to match motion length so they stay aligned.

        When ``self.interaction_crop`` is True the crop window is centred on the
        frame of closest ego-pedestrian approach (with uniform jitter of up to
        ±25 % of the window so the closest frame is not always dead-centre).
        Otherwise a random contiguous window is selected (original behaviour).

        Args:
            ego_ped_dists: (T,) precomputed raw ego-ped distances (before normalization).
                           Required when interaction_crop is True.

        Returns: (padded_motion, padded_ego, actual_length)
        """
        actual_length = len(sequence)

        # Align ego to motion length FIRST (handles 186 vs 185 mismatch)
        if len(ego_sequence) > actual_length:
            ego_sequence = ego_sequence[:actual_length]
        elif len(ego_sequence) < actual_length:
            pad_ego = np.zeros(
                (actual_length - len(ego_sequence), ego_sequence.shape[-1]),
                dtype=ego_sequence.dtype,
            )
            ego_sequence = np.concatenate([ego_sequence, pad_ego], axis=0)

        if actual_length >= max_length:
            if self.interaction_crop and ego_ped_dists is not None:
                # Centre the window on the closest-approach frame
                center = self._find_interaction_center(ego_ped_dists, actual_length)
                # Add uniform jitter of ±25 % of the window so the model does
                # not memorise a fixed position for the closest frame.
                jitter = int(max_length * 0.25)
                center += np.random.randint(-jitter, jitter + 1)
                start_idx = center - max_length // 2
                # Clamp to valid range
                start_idx = max(0, min(start_idx, actual_length - max_length))
            else:
                # Randomly select a contiguous subsequence
                start_idx = np.random.randint(0, actual_length - max_length + 1)
            sequence = sequence[start_idx:start_idx + max_length]
            ego_sequence = ego_sequence[start_idx:start_idx + max_length]
            actual_length = max_length
        else:
            # Pad both with zeros at the end
            pad_length = max_length - actual_length
            padding = np.zeros((pad_length, sequence.shape[-1]), dtype=sequence.dtype)
            sequence = np.concatenate([sequence, padding], axis=0)
            padding_ego = np.zeros((pad_length, ego_sequence.shape[-1]), dtype=ego_sequence.dtype)
            ego_sequence = np.concatenate([ego_sequence, padding_ego], axis=0)

        return sequence, ego_sequence, actual_length

    def __len__(self) -> int:
        if self.overfit:
            # Return a larger virtual size so DataLoader can build full batches.
            # Each batch element gets a different random (t, epsilon) during
            # diffusion training, which is critical for gradient averaging.
            # Without this, B=1 gradient variance prevents convergence.
            return max(len(self.sample_paths), 256)
        return len(self.sample_paths)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.
        
        Returns dict with:
            - ego: (max_ego_length, 2) - padded ego trajectory
            - motion: (max_motion_length, 263) - padded motion features
            - length: int - actual motion length
            - ego_length: int - actual ego length
        """
        json_path = self.sample_paths[idx % len(self.sample_paths)]
        
        try:
            sample = self._load_sample(json_path)
        except Exception as e:
            print(f"Error loading {json_path}: {e}")
            # Return next sample on error
            return self.__getitem__((idx + 1) % len(self))
        
        ego = sample["ego"]
        motion = sample["motion"]
        ped_root_xz = sample["ped_root_xz"]

        # Compute raw ego-ped distances BEFORE normalization (for interaction crop)
        T = min(len(ego), len(ped_root_xz))
        ego_ped_dists = np.linalg.norm(ego[:T] - ped_root_xz[:T], axis=-1)

        # Normalize motion and ego
        motion = self._normalize_motion(motion)
        ego = self._normalize_ego(ego)

        # Pad/crop sequences
        motion, ego, motion_length = self._pad_or_crop_ego_motion(
            motion, ego, self.max_motion_length, ego_ped_dists=ego_ped_dists,
        )
        ego_length = motion_length
        # Filter by length constraints
        if motion_length < self.min_motion_length:
            # Skip too-short samples
            return self.__getitem__((idx + 1) % len(self))

        return {
            "ego": ego,                    # (max_ego_length, 2)
            "motion": motion,              # (max_motion_length, 263)
            "length": motion_length,       # int
            "ego_length": ego_length,      # int
            "text": self._caption_for(self.sample_paths[idx]),
        }


class EgoMotionDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for EgoMotion dataset.
    Compatible with MLD training pipeline.
    """

    def __init__(
        self,
        cfg,
        batch_size: int = 64,
        num_workers: int = 8,
        collate_fn: Callable = None,
        mean: np.ndarray = None,           # Motion mean
        std: np.ndarray = None,            # Motion std
        ego_mean: np.ndarray = None,       # Ego mean (optional)
        ego_std: np.ndarray = None,        # Ego std (optional)
        debug: bool = False,
        overfit: bool = False,
        **kwargs
    ):
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn if collate_fn else ego_motion_collate
        self.mean = mean
        self.std = std
        self.ego_mean = ego_mean
        self.ego_std = ego_std
        self.debug = debug
        self.overfit = overfit
        self.kwargs = kwargs

        # Will be set by datasets
        self.nfeats = 263
        self.njoints = 22
        
        # For MLD compatibility
        self.is_mm = False

    def setup(self, stage: Optional[str] = None):
        """Setup train/val/test datasets."""
        data_root = self.cfg.DATASET.EGOMOTION.ROOT
        
        # Get ego scale from config (default 50 meters)
        ego_scale = getattr(self.cfg.DATASET.EGOMOTION, 'EGO_SCALE', 50.0)
        
        interaction_crop = getattr(self.cfg.DATASET.EGOMOTION, 'INTERACTION_CROP', False)
        interaction_weighted = getattr(self.cfg.DATASET.EGOMOTION, 'INTERACTION_WEIGHTED_SAMPLING', False)

        common_kwargs = dict(
            data_root=data_root,
            mean=self.mean,
            std=self.std,
            ego_mean=self.ego_mean,
            ego_std=self.ego_std,
            ego_scale=ego_scale,
            split_list_root=getattr(self.cfg.DATASET.EGOMOTION, 'MEAN_STD_PATH', None),
            max_motion_length=self.cfg.DATASET.SAMPLER.MAX_LEN,
            min_motion_length=self.cfg.DATASET.SAMPLER.MIN_LEN,
            max_ego_length=self.cfg.DATASET.EGOMOTION.MAX_EGO_LEN,
            fps=self.cfg.DATASET.EGOMOTION.FPS,
            debug=self.debug,
            overfit=self.overfit,
            interaction_crop=interaction_crop,
            interaction_weighted_sampling=interaction_weighted,
            captions_path=getattr(self.cfg.DATASET.EGOMOTION, 'CAPTIONS_PATH', None),
            caption_vocab=getattr(self.cfg.DATASET.EGOMOTION, 'CAPTION_VOCAB', 'ego'),
        )

        def _split_exists(split_name: str) -> bool:
            roots = data_root if isinstance(data_root, list) else [data_root]
            split_list_root = getattr(self.cfg.DATASET.EGOMOTION, 'MEAN_STD_PATH', None)

            # If split txt exists in MEAN_STD_PATH, treat split as available.
            if split_list_root:
                split_txt = pjoin(split_list_root, f"{split_name}.txt")
                if os.path.exists(split_txt):
                    return True

            # Otherwise, require at least one root with split dir.
            for root in roots:
                if os.path.isdir(pjoin(root, split_name)):
                    return True
            return False

        if stage == "fit" or stage is None:
            self.train_dataset = EgoMotionDataset(split="train", **common_kwargs)
            self.val_dataset = EgoMotionDataset(split="val", **common_kwargs)
            self.nfeats = self.train_dataset.nfeats
            self.njoints = self.train_dataset.njoints

        if stage == "test" or stage is None:
            if not self.is_mm:  # Don't overwrite MM-modified sample_paths
                test_split = getattr(self.cfg.TEST, 'SPLIT', 'test')
                if test_split == 'test' and not _split_exists('test') and _split_exists('val'):
                    print("[EgoMotionDataModule] No test split found; falling back to val split for testing.")
                    test_split = 'val'
                self.test_dataset = EgoMotionDataset(split=test_split, **common_kwargs)

    def train_dataloader(self):
        sampler = None
        shuffle = True
        if self.train_dataset.sample_weights is not None:
            sampler = WeightedRandomSampler(
                weights=self.train_dataset.sample_weights,
                num_samples=len(self.train_dataset),
                replacement=True,
            )
            shuffle = False  # mutually exclusive with sampler

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            drop_last=False,
            pin_memory=True,
        )

    def mm_mode(self, mm_on: bool = True):
        """
        Toggle MultiModality evaluation mode.

        When on:
          - A random subset of MM_NUM_SAMPLES test trajectories is selected.
          - Each trajectory is replicated (MM_NUM_TIMES + 1) times so that
            stochastic diffusion sampling can produce a diverse group of
            motions for the same ego condition.
          - batch_size is set to (MM_NUM_TIMES + 1) so each DataLoader batch
            contains all replicates of one condition, giving shape
            (1, MM_NUM_TIMES+1, 512) per MMMetrics.update call — which
            satisfies the calculate_multimodality_np assertion
            (activation.shape[1] > MM_NUM_TIMES).

        When off: original sample_paths and batch_size are restored.
        """
        if mm_on:
            self.is_mm = True
            self._original_sample_paths = self.test_dataset.sample_paths
            self._original_batch_size = self.batch_size

            mm_num_times = self.cfg.TEST.MM_NUM_TIMES
            mm_num_samples = min(
                self.cfg.TEST.MM_NUM_SAMPLES, len(self.test_dataset.sample_paths)
            )
            rng_indices = np.random.choice(
                len(self.test_dataset.sample_paths), mm_num_samples, replace=False
            )
            # Repeat each path (mm_num_times + 1) times so that the assertion
            # activation.shape[1] > mm_num_times holds in calculate_multimodality_np.
            mm_paths = [
                self.test_dataset.sample_paths[i]
                for i in rng_indices
                for _ in range(mm_num_times + 1)
            ]
            self.test_dataset.sample_paths = mm_paths
            self.batch_size = mm_num_times + 1
        else:
            self.is_mm = False
            self.test_dataset.sample_paths = self._original_sample_paths
            self.batch_size = self._original_batch_size

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            drop_last=False,
            pin_memory=True,
        )

    def feats2joints(self, features):
        """
        Convert normalized 263-dim features to 3D joint positions.
        
        Steps:
        1. Denormalize: features * std + mean
        2. Convert to joints: recover_from_ric (HumanML3D format)
        
        Args:
            features: (B, T, 263) - normalized motion features
            
        Returns:
            joints: (B, T, 22, 3) - 3D joint positions
        """
        if self.mean is not None and self.std is not None:
            # Denormalize
            mean = torch.tensor(self.mean).to(features)
            std = torch.tensor(self.std).to(features)
            features = features * std + mean
        
        # Convert to joints using HumanML3D's recover_from_ric
        # This works because vectors_263 follows HumanML3D format
        return recover_from_ric(features, self.njoints)


def ego_motion_collate(batch: List[Dict]) -> Dict:
    """
    Custom collate function for EgoMotion dataset.
    Handles batching with padding already applied in __getitem__.
    """
    # Stack tensors
    ego = torch.tensor(
        np.stack([b["ego"] for b in batch]), 
        dtype=torch.float32
    )
    motion = torch.tensor(
        np.stack([b["motion"] for b in batch]), 
        dtype=torch.float32
    )
    
    # Collect lengths (keep as list for MLD compatibility)
    lengths = [b["length"] for b in batch]
    ego_lengths = [b["ego_length"] for b in batch]
    text = [b["text"] for b in batch]  # Placeholder text (not used for ego)

    return {
        "ego": ego,                 # (B, max_ego_len, 2)
        "motion": motion,           # (B, max_motion_len, 263)
        "length": lengths,          # List[int]
        "ego_length": ego_lengths,  # List[int]
        "text": text,
    }
