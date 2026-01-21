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
from typing import Dict, List, Optional, Callable
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
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
        **kwargs
    ):
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

        # Load all sample paths from ALL data roots
        self.sample_paths = self._load_sample_paths()
        
        if self.debug:
            # Use only 100 samples for debugging
            self.sample_paths = self.sample_paths[:100]
        
        print(f"[EgoMotionDataset] Loaded {len(self.sample_paths)} samples for {split} from {len(self.data_roots)} sources")
        for root in self.data_roots:
            count = len([p for p in self.sample_paths if root in p])
            print(f"  - {root}: {count} samples")
        
        # Dataset info (required by MLD)
        self.nfeats = 263  # Motion feature dimension
        self.njoints = 22  # Joint count (HumanML3D format)

    def _load_sample_paths(self) -> List[str]:
        """
        Find all JSON files from ALL data roots.
        """
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
            
            # Get all JSON files in split directory
            pattern = pjoin(split_dir, "*.json")
            paths = glob(pattern)
            all_paths.extend(paths)
        
        return sorted(all_paths)

    def _load_sample(self, json_path: str) -> Dict:
        """
        Load a single sample from JSON file.
        
        Returns dict with:
            - ego: (T_ego, 2) - lateral motion (x, z)
            - motion: (T_motion, 263) - motion features
        """
        with open(json_path, "r") as f:
            data = json.load(f)
        
        # Extract ego trajectory: [[x, y, z], ...] -> (T, 2) using x, z
        ego_3d = np.array(data["ego_in_ped_frame"], dtype=np.float32)
        ego_2d = ego_3d[:, [0, 2]]  # Take x (index 0) and z (index 2)
        
        # Extract motion features: [[263], ...] -> (T, 263)
        motion = np.array(data["vectors_263"], dtype=np.float32)
        
        return {
            "ego": ego_2d,
            "motion": motion,
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
            # Full normalization
            ego = (ego - self.ego_mean) / (self.ego_std + 1e-8)
        else:
            # Simple scaling: divide by scale factor
            # This puts values roughly in [-1, 1] range if ego is within ±scale meters
            ego = ego / self.ego_scale
        return ego

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

    def __len__(self) -> int:
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
        json_path = self.sample_paths[idx]
        
        try:
            sample = self._load_sample(json_path)
        except Exception as e:
            print(f"Error loading {json_path}: {e}")
            # Return next sample on error
            return self.__getitem__((idx + 1) % len(self))
        
        ego = sample["ego"]
        motion = sample["motion"]

        # Normalize motion and ego
        motion = self._normalize_motion(motion)
        ego = self._normalize_ego(ego)

        # Pad/crop sequences
        ego, ego_length = self._pad_or_crop(ego, self.max_ego_length)
        motion, motion_length = self._pad_or_crop(motion, self.max_motion_length)

        # Filter by length constraints
        if motion_length < self.min_motion_length:
            # Skip too-short samples
            return self.__getitem__((idx + 1) % len(self))

        return {
            "ego": ego,                    # (max_ego_length, 2)
            "motion": motion,              # (max_motion_length, 263)
            "length": motion_length,       # int
            "ego_length": ego_length,      # int
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
        self.kwargs = kwargs

        # Will be set by datasets
        self.nfeats = 263
        self.njoints = 22
        
        # For MLD compatibility
        self.is_mm = False  # No multimodality evaluation for ego

    def setup(self, stage: Optional[str] = None):
        """Setup train/val/test datasets."""
        data_root = self.cfg.DATASET.EGOMOTION.ROOT
        
        # Get ego scale from config (default 50 meters)
        ego_scale = getattr(self.cfg.DATASET.EGOMOTION, 'EGO_SCALE', 50.0)
        
        common_kwargs = dict(
            data_root=data_root,
            mean=self.mean,
            std=self.std,
            ego_mean=self.ego_mean,
            ego_std=self.ego_std,
            ego_scale=ego_scale,
            max_motion_length=self.cfg.DATASET.SAMPLER.MAX_LEN,
            min_motion_length=self.cfg.DATASET.SAMPLER.MIN_LEN,
            max_ego_length=self.cfg.DATASET.EGOMOTION.MAX_EGO_LEN,
            fps=self.cfg.DATASET.EGOMOTION.FPS,
            debug=self.debug,
        )

        if stage == "fit" or stage is None:
            self.train_dataset = EgoMotionDataset(split="train", **common_kwargs)
            self.val_dataset = EgoMotionDataset(split="val", **common_kwargs)
            self.nfeats = self.train_dataset.nfeats
            self.njoints = self.train_dataset.njoints

        if stage == "test" or stage is None:
            self.test_dataset = EgoMotionDataset(split="test", **common_kwargs)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
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

    return {
        "ego": ego,                 # (B, max_ego_len, 2)
        "motion": motion,           # (B, max_motion_len, 263)
        "length": lengths,          # List[int]
        "ego_length": ego_lengths,  # List[int]
    }
