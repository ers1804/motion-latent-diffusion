"""
Ego Trajectory Encoder for MLD

Encodes ego vehicle trajectory (in pedestrian-centric coordinates) 
as conditioning for the diffusion denoiser.

Input: Ego trajectory (B, T, 2) - lateral motion only (x, z)
Output: Conditioning embedding (B, T, latent_dim) for concat with latent z
"""

import torch
import torch.nn as nn
from typing import Optional

from mld.models.operator import PositionalEncoding


class EgoEncoder(nn.Module):
    """
    Transformer-based encoder for ego vehicle trajectory.
    Replaces text encoder (CLIP) in original MLD for ego-conditioned generation.
    """

    def __init__(
        self,
        input_dim: int = 2,              # (x, z) lateral motion
        latent_dim: int = 256,           # Output dim (match VAE latent)
        num_layers: int = 4,             # Transformer layers
        num_heads: int = 4,              # Attention heads
        ff_size: int = 1024,             # Feedforward hidden size
        dropout: float = 0.1,
        activation: str = "gelu",
        **kwargs
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Project ego coordinates to latent dimension
        self.input_projection = nn.Linear(input_dim, latent_dim)

        # Positional encoding for temporal sequence
        self.pos_encoder = PositionalEncoding(latent_dim, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
            batch_first=False,  # MLD uses (seq, batch, dim) format internally
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )

        # Layer norm for output
        self.output_norm = nn.LayerNorm(latent_dim)

    def forward(
        self, 
        ego: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode ego trajectory.

        Args:
            ego: (B, T, 2) - Ego trajectory in ped-centric coords (x, z)
            mask: (B, T) - Optional mask, 1 for valid positions, 0 for padding

        Returns:
            (B, T, latent_dim) - Encoded ego features for conditioning
        """
        # Project input: (B, T, 2) -> (B, T, latent_dim)
        x = self.input_projection(ego)

        # Transpose for transformer: (B, T, D) -> (T, B, D)
        x = x.permute(1, 0, 2)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Create padding mask if provided
        # Transformer expects True for positions to IGNORE
        if mask is not None:
            src_key_padding_mask = ~mask.bool()
        else:
            src_key_padding_mask = None

        # Transformer forward
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Transpose back: (T, B, D) -> (B, T, D)
        x = x.permute(1, 0, 2)

        # Apply layer norm
        x = self.output_norm(x)

        return x


class EgoEncoderPooled(EgoEncoder):
    """
    Ego encoder that pools to a single vector (like CLIP does for text).
    Use this if you want (B, 1, latent_dim) output instead of full sequence.
    """

    def __init__(self, pool_type: str = "mean", **kwargs):
        super().__init__(**kwargs)
        self.pool_type = pool_type

    def forward(
        self, 
        ego: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Returns: (B, 1, latent_dim) - Single pooled vector per sample
        """
        # Get full sequence encoding
        x = super().forward(ego, mask)  # (B, T, D)

        # Pool along time dimension
        if mask is not None:
            # Masked mean pooling
            mask_expanded = mask.unsqueeze(-1).float()  # (B, T, 1)
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            if self.pool_type == "mean":
                x = x.mean(dim=1)  # (B, D)
            elif self.pool_type == "first":
                x = x[:, 0, :]  # (B, D)
            elif self.pool_type == "last":
                x = x[:, -1, :]  # (B, D)
            else:
                raise ValueError(f"Unknown pool_type: {self.pool_type}")

        # Add sequence dimension back: (B, D) -> (B, 1, D)
        x = x.unsqueeze(1)

        return x
