"""
EE Position Extractor: predicts end-effector xyz from action expert features.
Auxiliary task to encourage spatial encoding in suffix_out.

Input:  features [B, feature_dim] — mean-pooled suffix_out
Output: ee_pos [B, 3] — predicted EE position (in normalized space)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class EEPositionExtractor(nn.Module):
    def __init__(self, feature_dim: int = 1024, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: [B, feature_dim] -> [B, 3]"""
        return self.mlp(features)
