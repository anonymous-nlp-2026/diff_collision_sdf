"""
SDF Gradient-Guided Action Refinement for A3.

Uses ∂SDF/∂(suffix features) to shift action expert features toward collision-free
regions, then trains the model to output the shifted predictions directly.

Algorithm:
  1. Pool suffix_out → predict EE position → compute SDF at EE
  2. For samples where SDF < margin (near collision):
     a. Compute ∂SDF/∂suffix_out via autograd
     b. Shift suffix_out toward higher SDF: refined = suffix + α * grad
     c. Compute v_t (original) and v_t_refined from action_out_proj
     d. refine_loss = MSE(v_t, v_t_refined.detach())
  3. Model learns to directly output collision-aware actions

Input:
  suffix_out: [B, H, 1024] — action expert features (from forward hook)
  sdf_decoder: SDFDecoder
  ee_extractor: EEPositionExtractor
  action_out_proj: nn.Linear(1024 → 32)

Output:
  refine_loss: scalar or None
  info: dict with logging metrics

Dependencies:
  colliforce/sdf_decoder.py, colliforce/ee_extractor.py
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SDFGradientRefiner(nn.Module):
    """SDF gradient-guided action refinement.

    Computes the SDF spatial gradient w.r.t. suffix features and uses it to
    produce a "collision-aware" action target. The refinement loss pushes the
    model to predict these safer actions directly.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        margin: float = 0.01,
        max_refinement_steps: int = 1,
        grad_clip: float = 1.0,
    ):
        """
        Args:
            alpha: Gradient step size. Larger = stronger collision avoidance push.
            margin: SDF threshold (positive). Refinement activates when SDF < margin,
                    i.e., before actual collision (SDF=0) to provide a safety buffer.
            max_refinement_steps: Iterative refinement steps. 1 is sufficient for
                    training; higher values are useful at inference.
            grad_clip: Per-sample L2 norm cap on ∂SDF/∂suffix_out. Prevents
                    exploding gradients from sharp SDF boundaries.
        """
        super().__init__()
        self.alpha = alpha
        self.margin = margin
        self.max_refinement_steps = max_refinement_steps
        self.grad_clip = grad_clip

    def _clip_grad_per_sample(self, grad: torch.Tensor, B: int) -> torch.Tensor:
        """Clip gradient L2 norm independently per sample."""
        flat = grad.reshape(B, -1)
        norms = flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = torch.clamp(self.grad_clip / norms, max=1.0)
        return grad * scale.view(B, 1, 1)

    def forward(
        self,
        suffix_out: torch.Tensor,
        sdf_decoder: nn.Module,
        ee_extractor: nn.Module,
        action_out_proj: nn.Module,
    ) -> tuple[torch.Tensor | None, dict]:
        """Compute gradient-guided refinement loss.

        Returns (None, info) when no samples need refinement (all SDF >= margin).
        """
        B, H, D = suffix_out.shape
        pooled = suffix_out.mean(dim=1).float()  # [B, 1024]

        # Predict EE position → SDF at EE
        ee_pred = ee_extractor(pooled)  # [B, 3]
        sdf_at_ee, _ = sdf_decoder(pooled, ee_pred.unsqueeze(1))  # [B, 1]
        sdf_at_ee = sdf_at_ee.squeeze(-1)  # [B]

        info = {
            "sdf_at_ee_mean": sdf_at_ee.mean().item(),
            "sdf_at_ee_min": sdf_at_ee.min().item(),
        }

        needs_refine = sdf_at_ee < self.margin
        n_refine = needs_refine.sum().item()
        info["n_refine"] = float(n_refine)
        info["refine_ratio"] = n_refine / B

        if n_refine == 0:
            info["refine_loss"] = 0.0
            return None, info

        # --- Iterative gradient refinement ---
        suffix_cur = suffix_out
        for step in range(self.max_refinement_steps):
            # Recompute SDF from current (possibly shifted) features
            if step > 0:
                pooled_c = suffix_cur.mean(dim=1).float()
                ee_c = ee_extractor(pooled_c)
                sdf_c, _ = sdf_decoder(pooled_c, ee_c.unsqueeze(1))
                sdf_c = sdf_c.squeeze(-1)
                active = sdf_c < self.margin
                if not active.any():
                    break
            else:
                active = needs_refine

            # ∂SDF/∂suffix_out — direction to increase SDF (escape collision)
            # create_graph=False: saves memory; v_t_refined is detached anyway
            sdf_for_grad = sdf_at_ee[active].sum() if step == 0 else sdf_c[active].sum()
            # retain_graph=False: SDF decoder graph is separate from model graph
            grad_suffix = torch.autograd.grad(
                sdf_for_grad, suffix_cur,
                retain_graph=False,
                create_graph=False,
            )[0]  # [B, H, D]

            grad_suffix = self._clip_grad_per_sample(grad_suffix, B)

            # Gradient ascent on SDF (move toward higher SDF = safer)
            mask = active.float().view(B, 1, 1)
            suffix_cur = suffix_cur + self.alpha * grad_suffix * mask

        # Original and refined action predictions
        v_t = action_out_proj(suffix_out)       # [B, H, action_dim]
        v_t_refined = action_out_proj(suffix_cur)  # [B, H, action_dim]

        # Refinement loss: train model to directly output collision-safe actions.
        # Detach refined target so gradient only flows through v_t → suffix_out → model.
        refine_loss = F.mse_loss(
            v_t[needs_refine],
            v_t_refined[needs_refine].detach(),
        )

        info["refine_loss"] = refine_loss.item()
        info["action_shift_mean"] = (
            (v_t_refined - v_t)[needs_refine].abs().mean().item()
        )

        return refine_loss, info
