"""serve_policy with LoRA injection for A1-SDF checkpoints."""
import argparse
import logging
import pathlib
import socket

import safetensors.torch
import torch

from openpi.models_pytorch import pi0_pytorch
from openpi.policies import policy as _policy
from openpi.serving import websocket_policy_server
from openpi.training import config as _config
from openpi.shared import normalize as _normalize
import openpi.transforms as transforms

from colliforce.lora_utils import inject_lora_pi0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pi05_libero")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    train_config = _config.get_config(args.config)

    logging.info("Creating model with LoRA injection...")
    model = pi0_pytorch.PI0Pytorch(config=train_config.model)
    inject_lora_pi0(model)

    weight_path = f"{args.checkpoint_dir}/model.safetensors"
    logging.info("Loading checkpoint: %s", weight_path)
    safetensors.torch.load_model(model, weight_path)

    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

    ckpt_dir = pathlib.Path(args.checkpoint_dir)
    # Find norm_stats.json in the checkpoint's assets directory
    assets_dir = ckpt_dir / "assets"
    norm_stats_files = list(assets_dir.rglob("norm_stats.json"))
    if norm_stats_files:
        norm_stats_dir = norm_stats_files[0].parent
        logging.info("Loading norm stats from: %s", norm_stats_dir)
        norm_stats = _normalize.load(norm_stats_dir)
    else:
        raise FileNotFoundError(f"No norm_stats.json found under {assets_dir}")

    policy = _policy.Policy(
        model,
        transforms=[
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
        metadata=train_config.policy_metadata,
        is_pytorch=True,
        pytorch_device=device,
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Serving on %s:%d (ip: %s)", hostname, args.port, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=args.port, metadata=train_config.policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
