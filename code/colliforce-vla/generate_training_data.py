"""
LIBERO -> ColliForce-VLA Training Data Pipeline

Generates training data from LIBERO demos with:
- Camera images (agentview + eye_in_hand, 224x224)
- Actions (7-DoF delta EE)
- Proprioceptive state (eef_pos + eef_axisangle + gripper_qpos = 8-dim)
- EE position GT (3D world coords)
- SDF GT (pre-computed, per-scene)

Output: HDF5 files in <DATA_ROOT>/colliforce_data/
"""

import os
import sys
import json
import math
import h5py
import numpy as np
import pathlib
import torch
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

sys.path.insert(0, './LIBERO')

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark_dict

# === Config ===
OUTPUT_DIR = '<DATA_ROOT>/colliforce_data'
DEMO_DIR = '<DATA_ROOT>/libero_datasets'
SDF_DIR = './data'
IMG_SIZE = 224
MAX_EPISODES = 50  # per task

TASKS = {
    'spatial_bowl': {
        'suite': 'libero_spatial',
        'task_idx': 0,
        'sdf_file': 'sdf_gt_spatial_bowl.npz',
    },
    'object_chocolate': {
        'suite': 'libero_object',
        'task_idx': 8,
        'sdf_file': 'sdf_gt_object_chocolate.npz',
    },
}


def quat2axisangle(quat):
    q = quat.copy()
    if q[3] > 1.0:
        q[3] = 1.0
    elif q[3] < -1.0:
        q[3] = -1.0
    den = np.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


def resize_batch(imgs, size):
    from PIL import Image
    return np.array([np.array(Image.fromarray(img).resize((size, size), Image.BILINEAR)) for img in imgs])


def inspect_demo_hdf5(demo_path):
    with h5py.File(demo_path, 'r') as f:
        demos = sorted(list(f['data'].keys()))
        print(f"  Demos: {len(demos)}")
        ep = demos[0]
        print(f"  First episode '{ep}':")
        for k in f[f'data/{ep}']:
            if k in ['obs', 'next_obs']:
                print(f"    {k}/")
                for obs_k in f[f'data/{ep}/{k}']:
                    shape = f[f'data/{ep}/{k}/{obs_k}'].shape
                    print(f"      {obs_k}: {shape}")
            elif isinstance(f[f'data/{ep}/{k}'], h5py.Dataset):
                print(f"    {k}: {f[f'data/{ep}/{k}'].shape}")
        if 'env_args' in f['data'].attrs:
            env_args = json.loads(f['data'].attrs['env_args'])
            print(f"  env_args type: {env_args.get('type', 'unknown')}")


def detect_obs_keys(obs_keys):
    """Detect which naming convention the HDF5 uses and return key mapping."""
    mapping = {}

    # Image keys
    if 'agentview_image' in obs_keys:
        mapping['agentview'] = 'agentview_image'
    elif 'agentview_rgb' in obs_keys:
        mapping['agentview'] = 'agentview_rgb'

    if 'robot0_eye_in_hand_image' in obs_keys:
        mapping['wrist'] = 'robot0_eye_in_hand_image'
    elif 'robot0_eye_in_hand_rgb' in obs_keys:
        mapping['wrist'] = 'robot0_eye_in_hand_rgb'
    elif 'eye_in_hand_rgb' in obs_keys:
        mapping['wrist'] = 'eye_in_hand_rgb'

    # EE position
    if 'robot0_eef_pos' in obs_keys:
        mapping['ee_pos'] = 'robot0_eef_pos'
    elif 'ee_pos' in obs_keys:
        mapping['ee_pos'] = 'ee_pos'

    # EE orientation
    if 'robot0_eef_quat' in obs_keys:
        mapping['ee_ori'] = 'robot0_eef_quat'
        mapping['ori_format'] = 'quat'
    elif 'ee_ori' in obs_keys:
        mapping['ee_ori'] = 'ee_ori'
        mapping['ori_format'] = 'axisangle'

    # Gripper
    if 'robot0_gripper_qpos' in obs_keys:
        mapping['gripper'] = 'robot0_gripper_qpos'
    elif 'gripper_states' in obs_keys:
        mapping['gripper'] = 'gripper_states'

    return mapping


def process_task(task_name, task_cfg):
    print(f"\n{'='*60}")
    print(f"Processing: {task_name}")
    print(f"{'='*60}")

    suite_name = task_cfg['suite']
    task_idx = task_cfg['task_idx']
    sdf_file = task_cfg['sdf_file']

    benchmark_dict = get_benchmark_dict()
    suite = benchmark_dict[suite_name]()
    task = suite.get_task(task_idx)
    print(f"  Task: {task.name}")
    print(f"  Language: {task.language}")

    # Find demo HDF5
    demo_rel = suite.get_task_demonstration(task_idx)
    demo_path = os.path.join(DEMO_DIR, demo_rel)
    if not os.path.exists(demo_path):
        alt_path = os.path.join(get_libero_path('datasets'), demo_rel)
        if os.path.exists(alt_path):
            demo_path = alt_path
        else:
            print(f"  ERROR: Demo file not found: {demo_path}")
            print(f"  Also tried: {alt_path}")
            return None

    print(f"  Demo file: {demo_path}")
    inspect_demo_hdf5(demo_path)

    # Load SDF GT
    sdf_path = os.path.join(SDF_DIR, sdf_file)
    sdf_data = np.load(sdf_path)
    sdf_query_points = sdf_data['query_points']
    sdf_values = sdf_data['sdf_values']
    print(f"  SDF GT: {len(sdf_values)} points, range [{sdf_values.min():.4f}, {sdf_values.max():.4f}]")

    # Load demo data
    with h5py.File(demo_path, 'r') as f:
        demos = sorted(list(f['data'].keys()))
        n_episodes = min(len(demos), MAX_EPISODES)

        first_ep = demos[0]
        obs_keys = list(f[f'data/{first_ep}/obs'].keys()) if 'obs' in f[f'data/{first_ep}'] else []
        key_map = detect_obs_keys(obs_keys)

        print(f"  Obs keys: {obs_keys}")
        print(f"  Key mapping: {key_map}")

        has_images = 'agentview' in key_map
        has_robot_state = 'ee_pos' in key_map and 'ee_ori' in key_map and 'gripper' in key_map

        print(f"  Has images: {has_images}, Has robot state: {has_robot_state}")

        if has_images and has_robot_state:
            return process_from_obs(f, demos, n_episodes, task, sdf_query_points, sdf_values, task_name, key_map)
        else:
            print("  ERROR: Missing required observation keys for direct extraction.")
            print("  Cannot proceed without images and robot state.")
            return None


def process_from_obs(f, demos, n_episodes, task, sdf_query_points, sdf_values, task_name, key_map):
    print("  Mode: extracting from stored observations")
    print(f"  Orientation format: {key_map.get('ori_format', 'unknown')}")

    all_episodes = []
    for ep_idx in range(n_episodes):
        ep = demos[ep_idx]
        obs_grp = f[f'data/{ep}/obs']
        actions = f[f'data/{ep}/actions'][()]

        # Images
        agentview = obs_grp[key_map['agentview']][()]
        if 'wrist' in key_map:
            eye_in_hand = obs_grp[key_map['wrist']][()]
        else:
            eye_in_hand = np.zeros_like(agentview)

        # Resize if needed
        if agentview.shape[1] != IMG_SIZE:
            agentview = resize_batch(agentview, IMG_SIZE)
            eye_in_hand = resize_batch(eye_in_hand, IMG_SIZE)

        # Robot state
        eef_pos = obs_grp[key_map['ee_pos']][()]          # [T, 3]
        eef_ori_raw = obs_grp[key_map['ee_ori']][()]      # [T, 3] or [T, 4]
        gripper = obs_grp[key_map['gripper']][()]          # [T, 2]

        # Convert orientation to axis-angle if needed
        if key_map['ori_format'] == 'quat':
            eef_axisangle = np.array([quat2axisangle(q) for q in eef_ori_raw])
        else:
            eef_axisangle = eef_ori_raw  # already axis-angle

        # Build 8-dim state: eef_pos(3) + eef_axisangle(3) + gripper(2)
        state = np.concatenate([eef_pos, eef_axisangle, gripper], axis=-1)

        # EE position GT
        ee_positions = eef_pos.copy()

        episode_data = {
            'agentview': agentview,
            'eye_in_hand': eye_in_hand,
            'state': state.astype(np.float32),
            'actions': actions.astype(np.float32),
            'ee_positions': ee_positions.astype(np.float32),
        }
        all_episodes.append(episode_data)

        if (ep_idx + 1) % 10 == 0:
            print(f"    Processed {ep_idx + 1}/{n_episodes} episodes")

    return save_dataset(all_episodes, task, sdf_query_points, sdf_values, task_name)


def save_dataset(episodes, task, sdf_query_points, sdf_values, task_name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f'{task_name}_train.hdf5')

    print(f"\n  Saving to {out_path}")
    with h5py.File(out_path, 'w') as f:
        f.attrs['task_name'] = task_name
        f.attrs['language_instruction'] = task.language
        f.attrs['n_episodes'] = len(episodes)
        f.attrs['img_size'] = IMG_SIZE
        f.attrs['action_dim'] = 7
        f.attrs['state_dim'] = 8

        sdf_grp = f.create_group('sdf_gt')
        sdf_grp.create_dataset('query_points', data=sdf_query_points.astype(np.float32))
        sdf_grp.create_dataset('sdf_values', data=sdf_values.astype(np.float32))

        data_grp = f.create_group('data')
        for i, ep in enumerate(episodes):
            ep_grp = data_grp.create_group(f'episode_{i:04d}')
            ep_grp.attrs['length'] = len(ep['actions'])

            obs_grp = ep_grp.create_group('observations')
            obs_grp.create_dataset('agentview', data=ep['agentview'],
                                   compression='gzip', compression_opts=4)
            obs_grp.create_dataset('eye_in_hand', data=ep['eye_in_hand'],
                                   compression='gzip', compression_opts=4)
            obs_grp.create_dataset('state', data=ep['state'])

            ep_grp.create_dataset('actions', data=ep['actions'])
            ep_grp.create_dataset('ee_positions', data=ep['ee_positions'])

    file_size = os.path.getsize(out_path) / (1024**2)
    print(f"  Saved: {out_path} ({file_size:.1f} MB)")

    lengths = [len(ep['actions']) for ep in episodes]
    all_actions = np.concatenate([ep['actions'] for ep in episodes])
    all_ee = np.concatenate([ep['ee_positions'] for ep in episodes])
    print(f"\n  === Statistics for {task_name} ===")
    print(f"  Episodes: {len(episodes)}")
    print(f"  Avg episode length: {np.mean(lengths):.0f} (min={min(lengths)}, max={max(lengths)})")
    print(f"  Action dim: {episodes[0]['actions'].shape[-1]}")
    print(f"  Image shape: {episodes[0]['agentview'].shape[1:]}")
    print(f"  State dim: {episodes[0]['state'].shape[-1]}")
    print(f"  Action range: [{all_actions.min():.4f}, {all_actions.max():.4f}]")
    print(f"  EE pos range: x=[{all_ee[:,0].min():.3f},{all_ee[:,0].max():.3f}], "
          f"y=[{all_ee[:,1].min():.3f},{all_ee[:,1].max():.3f}], "
          f"z=[{all_ee[:,2].min():.3f},{all_ee[:,2].max():.3f}]")
    print(f"  SDF GT: {len(sdf_values)} points, range [{sdf_values.min():.4f}, {sdf_values.max():.4f}]")

    return {
        'task_name': task_name,
        'language': task.language,
        'n_episodes': len(episodes),
        'avg_length': float(np.mean(lengths)),
        'min_length': int(min(lengths)),
        'max_length': int(max(lengths)),
        'action_range': [float(all_actions.min()), float(all_actions.max())],
        'ee_pos_range': {
            'x': [float(all_ee[:,0].min()), float(all_ee[:,0].max())],
            'y': [float(all_ee[:,1].min()), float(all_ee[:,1].max())],
            'z': [float(all_ee[:,2].min()), float(all_ee[:,2].max())],
        },
        'sdf_n_points': len(sdf_values),
        'sdf_range': [float(sdf_values.min()), float(sdf_values.max())],
        'file_path': out_path,
        'file_size_mb': file_size,
    }


if __name__ == '__main__':
    print("ColliForce-VLA Training Data Pipeline")
    print(f"Output: {OUTPUT_DIR}")

    results = {}
    for task_name, task_cfg in TASKS.items():
        try:
            result = process_task(task_name, task_cfg)
            if result:
                results[task_name] = result
        except Exception as e:
            import traceback
            print(f"\n  ERROR processing {task_name}: {e}")
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("Pipeline Complete")
    print(f"{'='*60}")
    for name, r in results.items():
        print(f"\n{name}:")
        print(f"  File: {r['file_path']} ({r['file_size_mb']:.1f} MB)")
        print(f"  Episodes: {r['n_episodes']}, Avg length: {r['avg_length']:.0f}")

    # Save results JSON
    if results:
        json_path = os.path.join(OUTPUT_DIR, 'pipeline_results.json')
        with open(json_path, 'w') as jf:
            json.dump(results, jf, indent=2)
        print(f"\nResults saved to {json_path}")
