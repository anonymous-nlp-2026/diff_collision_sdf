"""
SDF MLP Decoder for A1 auxiliary task.
4-layer MLP, hidden=256, Softplus(beta=5), skip connection re-injecting xyz at layer 2.
Validated in KG-3 gradient tests.

Input:
  features: [B, feature_dim] — mean-pooled suffix_out from action expert
  query_points: [B, N, 3] — 3D query positions in workspace coords

Output:
  sdf_values: [B, N] — predicted signed distance
  sdf_gradients: [B, N, 3] — spatial gradients (optional)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SDFDecoder(nn.Module):
    def __init__(self, feature_dim: int = 1024, hidden_dim: int = 256):
        super().__init__()
        # Layer 0: concat(feature, xyz) -> hidden
        self.layer0 = nn.Linear(feature_dim + 3, hidden_dim)
        # Layer 1: hidden -> hidden
        self.layer1 = nn.Linear(hidden_dim, hidden_dim)
        # Layer 2 (skip): hidden + xyz -> hidden
        self.layer2 = nn.Linear(hidden_dim + 3, hidden_dim)
        # Layer 3: hidden -> 1
        self.layer3 = nn.Linear(hidden_dim, 1)
        self.activation = nn.Softplus(beta=5)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        features: torch.Tensor,
        query_points: torch.Tensor,
        return_gradients: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            features: [B, feature_dim] mean-pooled action expert features
            query_points: [B, N, 3]
            return_gradients: compute spatial gradients for Eikonal loss

        Returns:
            sdf: [B, N]
            grad_sdf: [B, N, 3] or None
        """
        B, N, _ = query_points.shape

        if return_gradients:
            xyz = query_points.detach().requires_grad_(True)
        else:
            xyz = query_points

        feat = features.unsqueeze(1).expand(-1, N, -1)  # [B, N, feat_dim]
        x = torch.cat([feat, xyz], dim=-1)  # [B, N, feat_dim+3]

        h = self.activation(self.layer0(x))
        h = self.activation(self.layer1(h))
        h = torch.cat([h, xyz], dim=-1)  # skip: re-inject xyz
        h = self.activation(self.layer2(h))
        sdf = self.layer3(h).squeeze(-1)  # [B, N]

        if return_gradients:
            grad_outputs = torch.ones_like(sdf)
            (grad_sdf,) = torch.autograd.grad(
                outputs=sdf,
                inputs=xyz,
                grad_outputs=grad_outputs,
                create_graph=self.training,
                retain_graph=True,
            )
            return sdf, grad_sdf

        return sdf, None


def compute_sdf_loss(
    sdf_pred: torch.Tensor,
    sdf_gt: torch.Tensor,
    sdf_gradients: torch.Tensor | None,
    clamp_delta: float = 0.05,
    eikonal_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Clamped L1 + Eikonal regularization."""
    pred_c = sdf_pred.clamp(-clamp_delta, clamp_delta)
    gt_c = sdf_gt.clamp(-clamp_delta, clamp_delta)
    sdf_l1 = (pred_c - gt_c).abs().mean()

    losses = {"sdf_l1": sdf_l1}

    if sdf_gradients is not None:
        grad_norm = sdf_gradients.norm(dim=-1)
        eikonal = ((grad_norm - 1.0) ** 2).mean()
        losses["eikonal"] = eikonal
        losses["sdf_total"] = sdf_l1 + eikonal_weight * eikonal
    else:
        losses["sdf_total"] = sdf_l1

    return losses
