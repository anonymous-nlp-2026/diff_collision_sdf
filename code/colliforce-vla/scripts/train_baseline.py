"""
Baseline: Vanilla pi0 fine-tune on SafeLIBERO MVP scenarios.
No SDF, no collision awareness — serves as Phase B baseline.
Uses manual LoRA injection (matching A3) for fair comparison.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/train_baseline.py --scene spatial_bowl --seed 0
    CUDA_VISIBLE_DEVICES=3 python scripts/train_baseline.py --scene spatial_bowl --dry_run
"""

import argparse
import os
import sys
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "openpi"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "openpi", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["HF_LEROBOT_HOME"] = "<DATA_ROOT>/colliforce_data"

import openpi.models.pi0_config as pi0_config
import openpi.training.config as _config
from openpi.training.optimizer import CosineDecaySchedule
from scripts.train_pytorch import TrainHooks, train_loop
from colliforce.lora_utils import inject_lora_pi0


SCENE_CONFIG = {
    "spatial_bowl": {
        "description": "SafeLIBERO Spatial: bowl between plate and ramekin (Level II)",
        "data_subdir": "safelibero_spatial",
    },
    "object_pudding": {
        "description": "SafeLIBERO Object: chocolate pudding to basket (Level I)",
        "data_subdir": "safelibero_object",
    },
}

CHECKPOINT_BASE = "<DATA_ROOT>/checkpoints"
DATA_BASE = "<DATA_ROOT>/colliforce_data"
WEIGHTS_BASE = "<DATA_ROOT>/openpi_weights"


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline pi0 fine-tune")
    parser.add_argument("--scene", type=str, required=True, choices=list(SCENE_CONFIG.keys()))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_train_steps", type=int, default=30000)
    parser.add_argument("--lr", type=float, default=2.5e-5)
    parser.add_argument("--save_interval", type=int, default=2000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true", help="Quick 2-step test without wandb")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--data_dir", type=str, default=None, help="Override data directory")
    parser.add_argument("--weight_path", type=str, default=None, help="Override base weight path")
    return parser.parse_args()


class BaselineLoRAHooks(TrainHooks):
    """Inject manual LoRA and freeze non-LoRA params for fair comparison with A1/A3."""

    def on_model_created(self, model, device):
        _m = model.module if hasattr(model, "module") else model

        # Inject LoRA: PaliGemma rank=16, Action Expert rank=32
        self.lora_params, self._pali_lora_params, self._expert_lora_params = inject_lora_pi0(
            _m, pali_rank=16, pali_alpha=16.0, expert_rank=32, expert_alpha=32.0)

        # Freeze all non-LoRA params
        for name, p in _m.named_parameters():
            if "lora" not in name.lower():
                p.requires_grad = False

        n_lora = sum(p.numel() for p in self.lora_params)
        n_total = sum(p.numel() for p in _m.parameters())
        n_trainable = sum(p.numel() for p in _m.parameters() if p.requires_grad)
        logging.info("=== Baseline LoRA Hooks ===")
        logging.info("  Total params: %.2fM", n_total / 1e6)
        logging.info("  LoRA params: %.2fM (PaliGemma: %.2fM, Expert: %.2fM)",
                     n_lora / 1e6,
                     sum(p.numel() for p in self._pali_lora_params) / 1e6,
                     sum(p.numel() for p in self._expert_lora_params) / 1e6)
        logging.info("  Trainable params: %.2fM", n_trainable / 1e6)

    def extra_params(self):
        return []

    def on_step_start(self, global_step, model):
        if global_step < 100 and global_step % 10 == 0:
            _m = model.module if hasattr(model, "module") else model
            trainable = {n: p.numel() for n, p in _m.named_parameters() if p.requires_grad}
            print(f"[Step {global_step}] Trainable params: {sum(trainable.values()):,}")
            print(f"[Step {global_step}] LoRA params: {sum(v for n, v in trainable.items() if 'lora' in n):,}")


def build_config(args) -> _config.TrainConfig:
    scene_cfg = SCENE_CONFIG[args.scene]
    data_dir = args.data_dir or os.path.join(DATA_BASE, scene_cfg["data_subdir"])
    weight_path = args.weight_path or os.path.join(WEIGHTS_BASE, "pi0_base")
    exp_name = args.exp_name or f"baseline_{args.scene}_seed{args.seed}"

    num_steps = 2 if args.dry_run else args.num_train_steps

    model_config = pi0_config.Pi0Config(
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
    )

    config = _config.TrainConfig(
        name=f"baseline_{args.scene}",
        project_name="colliforce-vla",
        exp_name=exp_name,
        model=model_config,
        data=_config.LeRobotLiberoDataConfig(
            repo_id=scene_cfg["data_subdir"],
            base_config=_config.DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        pytorch_weight_path=weight_path,
        ema_decay=None,
        lr_schedule=CosineDecaySchedule(
            warmup_steps=min(1000, num_steps // 3),
            peak_lr=args.lr,
            decay_steps=num_steps,
            decay_lr=args.lr / 10,
        ),
        num_train_steps=num_steps,
        batch_size=args.batch_size,
        save_interval=args.save_interval if not args.dry_run else num_steps,
        log_interval=args.log_interval if not args.dry_run else 1,
        checkpoint_base_dir=CHECKPOINT_BASE,
        seed=args.seed,
        wandb_enabled=not args.dry_run,
        overwrite=args.dry_run,
        resume=args.resume,
    )

    return config


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        logging.warning("CUDA_VISIBLE_DEVICES not set, will use default GPU")

    scene_cfg = SCENE_CONFIG[args.scene]
    logging.info("Scene: %s", scene_cfg["description"])
    logging.info("Seed: %d, Dry run: %s", args.seed, args.dry_run)

    data_dir = args.data_dir or os.path.join(DATA_BASE, scene_cfg["data_subdir"])
    weight_path = args.weight_path or os.path.join(WEIGHTS_BASE, "pi0_base")

    if not os.path.exists(data_dir):
        logging.error("Data directory not found: %s", data_dir)
        sys.exit(1)

    weight_file = os.path.join(weight_path, "model.safetensors")
    if not os.path.exists(weight_file):
        logging.error("Base weight file not found: %s", weight_file)
        sys.exit(1)

    config = build_config(args)
    hooks = BaselineLoRAHooks()
    logging.info("Checkpoint dir: %s", config.checkpoint_dir)
    logging.info("Weight path: %s (verified)", config.pytorch_weight_path)
    logging.info("Training steps: %d, Batch size: %d, LR: %.2e",
                 config.num_train_steps, config.batch_size, args.lr)

    train_loop(config, hooks=hooks)


if __name__ == "__main__":
    main()
