"""Verify that pi0_base weights load correctly into PI0Pytorch model."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "openpi"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "openpi", "src"))
os.environ['MUJOCO_GL'] = 'egl'

import torch
import safetensors.torch
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
import openpi.models.pi0_config as pi0_config

WEIGHT_DIR = "<DATA_ROOT>/openpi_weights/pi0_base"
WEIGHT_FILE = os.path.join(WEIGHT_DIR, "model.safetensors")

print(f"Weight file: {WEIGHT_FILE}")
print(f"File size: {os.path.getsize(WEIGHT_FILE) / 1024**3:.2f} GB")

# Check file integrity
from safetensors import safe_open
with safe_open(WEIGHT_FILE, framework='pt') as f:
    sf_keys = set(f.keys())
    total_params = 0
    for k in sf_keys:
        shape = f.get_tensor(k).shape
        p = 1
        for s in shape:
            p *= s
        total_params += p
    print(f"Safetensors: {len(sf_keys)} tensors, {total_params:,} params")

# Build model
config = pi0_config.Pi0Config(
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora",
    action_dim=7,
    action_horizon=50,
)
print(f"\nBuilding PI0Pytorch model...")
model = PI0Pytorch(config)
model_keys = set(model.state_dict().keys())
print(f"Model state_dict: {len(model_keys)} keys")

# Check key overlap
common = sf_keys & model_keys
only_sf = sf_keys - model_keys
only_model = model_keys - sf_keys

print(f"\nKey matching:")
print(f"  Common: {len(common)}")
print(f"  Only in safetensors: {len(only_sf)}")
print(f"  Only in model: {len(only_model)}")

if only_sf:
    print(f"  Safetensors-only examples: {sorted(only_sf)[:5]}")
if only_model:
    print(f"  Model-only examples: {sorted(only_model)[:5]}")

# Try loading
print(f"\nLoading weights...")
safetensors.torch.load_model(model, WEIGHT_FILE, strict=False)
print("SUCCESS: Weights loaded")

# Count loaded vs random params
loaded = 0
total = 0
for name, param in model.named_parameters():
    p = param.numel()
    total += p
    if name in sf_keys:
        loaded += p
print(f"Loaded params: {loaded:,} / {total:,} ({100*loaded/total:.1f}%)")
print(f"\nDone. Weight loading verified.")
