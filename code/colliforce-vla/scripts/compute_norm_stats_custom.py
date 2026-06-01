"""Compute normalization stats for our custom LeRobot datasets."""
import os
import sys
import json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "openpi"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "openpi", "src"))

os.environ["HF_LEROBOT_HOME"] = "<DATA_ROOT>/colliforce_data"

import openpi.shared.normalize as normalize
import pyarrow.parquet as pq

CONFIGS = {
    "spatial_bowl": {
        "repo_id": "safelibero_spatial",
        "config_name": "baseline_spatial_bowl",
    },
    "object_pudding": {
        "repo_id": "safelibero_object",
        "config_name": "baseline_object_pudding",
    },
}

DATA_ROOT = "<DATA_ROOT>/colliforce_data"


def compute_stats(repo_id, config_name):
    print(f"Computing norm stats for {repo_id}...")
    
    data_dir = os.path.join(DATA_ROOT, repo_id, "data")
    parquet_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".parquet")]
    
    if not parquet_files:
        # Try nested
        for root, dirs, files in os.walk(data_dir):
            for f in files:
                if f.endswith(".parquet"):
                    parquet_files.append(os.path.join(root, f))
    
    print(f"  Found {len(parquet_files)} parquet files")
    
    all_states = []
    all_actions = []
    
    for pf in sorted(parquet_files):
        table = pq.read_table(pf)
        df = table.to_pandas()
        print(f"  {pf}: {len(df)} rows, columns: {list(df.columns)[:10]}")
        
        if "state" in df.columns:
            states = np.stack(df["state"].values)
            all_states.append(states)
        
        if "actions" in df.columns:
            actions = np.stack(df["actions"].values)
            all_actions.append(actions)
    
    states = np.concatenate(all_states)
    actions = np.concatenate(all_actions)
    
    print(f"  Total frames: {len(states)}")
    print(f"  State shape: {states.shape}, Actions shape: {actions.shape}")
    
    norm_stats = {
        "state": normalize.NormStats(
            mean=states.mean(axis=0).astype(np.float32),
            std=np.maximum(states.std(axis=0).astype(np.float32), 1e-6),
        ),
        "actions": normalize.NormStats(
            mean=actions.mean(axis=0).astype(np.float32),
            std=np.maximum(actions.std(axis=0).astype(np.float32), 1e-6),
        ),
    }
    
    output_dir = f"./assets/{config_name}/{repo_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    normalize.save(output_dir, norm_stats)
    print(f"  Saved to {output_dir}")
    print(f"  State mean: {norm_stats['state'].mean}")
    print(f"  State std:  {norm_stats['state'].std}")
    print(f"  Actions mean: {norm_stats['actions'].mean}")
    print(f"  Actions std:  {norm_stats['actions'].std}")


for name, cfg in CONFIGS.items():
    compute_stats(cfg["repo_id"], cfg["config_name"])

print("Done!")
