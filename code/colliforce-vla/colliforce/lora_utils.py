"""Shared LoRA injection module, extracted from train_a3_hook.py.

Provides manual LoRA (low-rank adaptation) without PEFT dependency.
LoRALinear wraps nn.Linear with frozen base weights + trainable lora_A/lora_B.
"""

import logging

import torch
import torch.nn as nn

TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    """Drop-in nn.Linear replacement with low-rank adapters.
    Base weight frozen; only lora_A/lora_B are trainable.
    """

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        in_f, out_f = base.in_features, base.out_features
        _device = base.weight.device
        _dtype = base.weight.dtype
        self.lora_A = nn.Parameter(torch.randn(in_f, rank, device=_device, dtype=_dtype) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_f, device=_device, dtype=_dtype))
        self.scaling = alpha / rank

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        base_out = self.base(x.to(self.base.weight.dtype))
        lora_out = (x.float() @ self.lora_A.float() @ self.lora_B.float()) * self.scaling
        return base_out + lora_out.to(base_out.dtype)


def inject_lora(module, lora_rank=16, lora_alpha=16.0, target_modules=None):
    """Inject LoRA adapters into matching nn.Linear layers within module.

    Walks all sub-modules. For any nn.Linear whose attribute name matches
    an entry in target_modules, replaces it with LoRALinear.

    Args:
        module: nn.Module to inject LoRA into.
        lora_rank: Rank of the LoRA decomposition.
        lora_alpha: Scaling factor (effective scale = alpha / rank).
        target_modules: List of attribute names to replace.
            Defaults to TARGET_MODULES (q/k/v/o_proj + gate/up/down_proj).

    Returns:
        List of LoRA parameters (lora_A, lora_B pairs) for the optimizer.
    """
    if target_modules is None:
        target_modules = list(TARGET_MODULES)

    lora_params = []
    for name, child in list(module.named_modules()):
        if not name:
            continue
        parts = name.split(".")
        attr_name = parts[-1]
        if attr_name in target_modules and isinstance(child, nn.Linear):
            parent_path = ".".join(parts[:-1])
            parent = module.get_submodule(parent_path) if parent_path else module
            lora_mod = LoRALinear(child, rank=lora_rank, alpha=lora_alpha)
            setattr(parent, attr_name, lora_mod)
            lora_params.extend([lora_mod.lora_A, lora_mod.lora_B])

    return lora_params


def inject_lora_pi0(model, pali_rank=16, pali_alpha=16.0,
                    expert_rank=32, expert_alpha=32.0, target_modules=None):
    """Pi0-specific LoRA injection: separate rank/alpha for backbone vs expert.

    PaliGemma (Gemma 2B) -> pali_rank/pali_alpha
    Action Expert (Gemma 300M) -> expert_rank/expert_alpha

    Args:
        model: Pi0 model (unwrapped, not DDP).
        pali_rank: LoRA rank for PaliGemma backbone.
        pali_alpha: LoRA alpha for PaliGemma backbone.
        expert_rank: LoRA rank for action expert.
        expert_alpha: LoRA alpha for action expert.
        target_modules: Override target module names.

    Returns:
        (all_lora_params, pali_lora_params, expert_lora_params)
    """
    pali_model = model.paligemma_with_expert.paligemma.model.language_model
    expert_model = model.paligemma_with_expert.gemma_expert.model

    pali_lora_params = inject_lora(pali_model, lora_rank=pali_rank,
                                   lora_alpha=pali_alpha,
                                   target_modules=target_modules)
    expert_lora_params = inject_lora(expert_model, lora_rank=expert_rank,
                                     lora_alpha=expert_alpha,
                                     target_modules=target_modules)

    all_params = pali_lora_params + expert_lora_params
    logging.info("LoRA injected: PaliGemma rank=%d (%d params), Expert rank=%d (%d params)",
                 pali_rank, sum(p.numel() for p in pali_lora_params),
                 expert_rank, sum(p.numel() for p in expert_lora_params))
    return all_params, pali_lora_params, expert_lora_params
