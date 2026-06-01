"""Merge LoRA checkpoint into standard safetensors format for serve_policy.py.

LoRA checkpoints save: X.base.weight, X.lora_A (in_f, rank), X.lora_B (rank, out_f)
Standard format needs: X.weight

Merge formula: weight = base_weight + (lora_A @ lora_B).T * (alpha/rank)
For our training: pali alpha/rank = 16/16 = 1.0, expert alpha/rank = 32/32 = 1.0
"""
import os
import sys
import torch
from safetensors.torch import load_file, save_file


def merge_lora_checkpoint(ckpt_dir, output_path=None, scaling=1.0):
    if output_path is None:
        output_path = os.path.join(ckpt_dir, "model_merged.safetensors")

    weight_path = os.path.join(ckpt_dir, "model.safetensors")
    print(f"Loading {weight_path}...")
    state_dict = load_file(weight_path, device="cpu")
    print(f"  {len(state_dict)} keys loaded")

    merged = {}
    n_merged = 0
    n_passthrough = 0

    base_keys = [k for k in state_dict if k.endswith('.base.weight')]

    for key in base_keys:
        prefix = key[:-len('.base.weight')]
        lora_a_key = prefix + '.lora_A'
        lora_b_key = prefix + '.lora_B'

        base_weight = state_dict[key]
        if lora_a_key in state_dict and lora_b_key in state_dict:
            lora_a = state_dict[lora_a_key].float()  # (in_f, rank)
            lora_b = state_dict[lora_b_key].float()  # (rank, out_f)
            delta = (lora_a @ lora_b).T * scaling  # (out_f, in_f)
            merged_weight = base_weight.float() + delta
            merged[prefix + '.weight'] = merged_weight.to(base_weight.dtype)
            n_merged += 1
        else:
            merged[prefix + '.weight'] = base_weight
            n_merged += 1

    processed_keys = set()
    for key in base_keys:
        prefix = key[:-len('.base.weight')]
        processed_keys.add(key)
        processed_keys.add(prefix + '.lora_A')
        processed_keys.add(prefix + '.lora_B')

    for key in state_dict:
        if key not in processed_keys:
            merged[key] = state_dict[key]
            n_passthrough += 1

    print(f"  Merged {n_merged} LoRA layers, {n_passthrough} passthrough keys")
    print(f"  Output: {len(merged)} total keys")
    print(f"  Saving to {output_path}...")
    save_file(merged, output_path)
    print(f"  Done. Size: {os.path.getsize(output_path) / 1e9:.2f} GB")
    return output_path


if __name__ == "__main__":
    ckpt_dirs = sys.argv[1:]
    if not ckpt_dirs:
        print("Usage: python merge_lora_checkpoint.py <ckpt_dir1> [ckpt_dir2] ...")
        sys.exit(1)

    for d in ckpt_dirs:
        print(f"\n=== Merging: {d} ===")
        merge_lora_checkpoint(d, scaling=1.0)
