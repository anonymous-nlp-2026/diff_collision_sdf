"""Convert raw LIBERO HDF5 demos to LeRobot format for openpi training."""
import os
import sys
import json
import shutil
import argparse
import numpy as np
from PIL import Image

os.environ["HF_LEROBOT_HOME"] = "<DATA_ROOT>/colliforce_data"

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME
import h5py

SCENES = {
    "spatial_bowl": {
        "hdf5": "<DATA_ROOT>/libero_datasets/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5",
        "repo_id": "safelibero_spatial",
    },
    "object_pudding": {
        "hdf5": "<DATA_ROOT>/libero_datasets/libero_object/pick_up_the_chocolate_pudding_and_place_it_in_the_basket_demo.hdf5",
        "repo_id": "safelibero_object",
    },
}


def resize_image(img_array, target_size=(256, 256)):
    img = Image.fromarray(img_array)
    img = img.resize(target_size, Image.BILINEAR)
    return np.array(img)


def convert_scene(scene_name):
    cfg = SCENES[scene_name]
    hdf5_path = cfg["hdf5"]
    repo_id = cfg["repo_id"]

    print(f"Converting {scene_name}: {hdf5_path} -> {repo_id}")

    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        print(f"  Removing existing {output_path}")
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=20,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=4,
        image_writer_processes=2,
    )

    with h5py.File(hdf5_path, "r") as f:
        problem_info = json.loads(f["data"].attrs.get("problem_info", "{}"))
        task_str = problem_info.get("language_instruction", scene_name)
        num_demos = int(f["data"].attrs.get("num_demos", 0))
        if num_demos == 0:
            num_demos = len([k for k in f["data"].keys() if k.startswith("demo_")])

        print(f"  Task: {task_str}")
        print(f"  Num demos: {num_demos}")

        total_frames = 0
        for demo_idx in range(num_demos):
            demo_key = f"demo_{demo_idx}"
            demo = f["data"][demo_key]
            actions = demo["actions"][:]
            obs = demo["obs"]
            T = actions.shape[0]

            agentview = obs["agentview_rgb"][:]
            eye_in_hand = obs["eye_in_hand_rgb"][:]
            ee_pos = obs["ee_pos"][:]
            ee_ori = obs["ee_ori"][:]
            gripper = obs["gripper_states"][:]

            for t in range(T):
                img = resize_image(agentview[t])
                wrist = resize_image(eye_in_hand[t])
                state = np.concatenate([ee_pos[t], ee_ori[t], gripper[t]]).astype(np.float32)

                dataset.add_frame({
                    "image": img,
                    "wrist_image": wrist,
                    "state": state,
                    "actions": actions[t].astype(np.float32),
                    "task": task_str,
                })

            dataset.save_episode()
            total_frames += T

            if (demo_idx + 1) % 10 == 0:
                print(f"  Processed {demo_idx + 1}/{num_demos} demos ({total_frames} frames)")

    dataset.finalize()
    print(f"  Done: {total_frames} frames, saved to {output_path}")
    print(f"  Dataset size: {sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file()) / 1024**2:.1f} MB")
    return total_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=list(SCENES.keys()) + ["all"], default="all")
    args = parser.parse_args()

    scenes = list(SCENES.keys()) if args.scene == "all" else [args.scene]
    for s in scenes:
        convert_scene(s)


if __name__ == "__main__":
    main()
