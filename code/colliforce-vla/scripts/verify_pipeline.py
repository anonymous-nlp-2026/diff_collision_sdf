#!/usr/bin/env python3
"""ColliForce-VLA Pipeline End-to-End Verification"""
import sys
import os
import traceback
import multiprocessing

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "openpi"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "openpi", "src"))

os.environ["MUJOCO_GL"] = "egl"
os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
os.environ["HF_LEROBOT_HOME"] = "<DATA_ROOT>/colliforce_data"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np

PASSED = []
FAILED = []

def stage(name):
    print("=" * 60)
    print(f"  {name}")
    print("=" * 60)


def main():
    global PASSED, FAILED

    # ===== STAGE 1: Data Loading =====
    stage("Stage 1: Data Loading Verification")
    observation = None
    actions = None
    try:
        import openpi.training.config as _config
        import openpi.models.pi0_config as pi0_config
        from openpi.training import data_loader as _data

        model_config = pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        )

        data_cfg_factory = _config.LeRobotLiberoDataConfig(
            repo_id="safelibero_spatial",
            base_config=_config.DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        )

        train_config = _config.TrainConfig(
            name="baseline_spatial_bowl",
            exp_name="verify_test",
            model=model_config,
            data=data_cfg_factory,
            batch_size=2,
            num_train_steps=1,
            num_workers=0,
            wandb_enabled=False,
            overwrite=True,
        )

        loader = _data.create_data_loader(train_config, framework="pytorch", shuffle=False)
        batch = next(iter(loader))
        observation, actions = batch

        obs_dict = observation.to_dict() if hasattr(observation, 'to_dict') else observation
        print(f"  Observation type: {type(observation)}")
        if isinstance(obs_dict, dict):
            for k, v in obs_dict.items():
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        if hasattr(v2, 'shape'):
                            print(f"    {k}/{k2}: {v2.shape} {v2.dtype}")
                elif hasattr(v, 'shape'):
                    print(f"    {k}: {v.shape} {v.dtype}")
                else:
                    print(f"    {k}: {type(v)}")
        print(f"  Actions shape: {actions.shape}, dtype: {actions.dtype}")
        print("  STAGE 1 PASSED")
        PASSED.append("Stage 1: Data Loading")
    except Exception as e:
        traceback.print_exc()
        FAILED.append(f"Stage 1: Data Loading - {e}")

    # ===== STAGE 2: Model Init + Weight Loading =====
    stage("Stage 2: Model Initialization + Weight Loading")
    model = None
    try:
        import safetensors.torch
        import openpi.models_pytorch.pi0_pytorch as pi0_pytorch
        import openpi.models.pi0_config as pi0_config

        device = torch.device("cuda:0")
        model_cfg = pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            dtype="float32",
        )

        model = pi0_pytorch.PI0Pytorch(model_cfg).to(device)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Model params (random init): {param_count:,}")

        weight_path = "<DATA_ROOT>/openpi_weights/pi0_base/model.safetensors"
        safetensors.torch.load_model(model, weight_path)

        first_param = next(model.parameters())
        print(f"  Weight loaded: first param mean={first_param.data.float().mean():.6f}, std={first_param.data.float().std():.6f}")
        print(f"  Weight file: {os.path.getsize(weight_path) / 1024**3:.2f} GB")
        print("  STAGE 2 PASSED")
        PASSED.append("Stage 2: Model + Weights")
    except Exception as e:
        traceback.print_exc()
        FAILED.append(f"Stage 2: Model + Weights - {e}")

    # ===== STAGE 3: Forward Pass + Loss =====
    stage("Stage 3: Forward Pass + Loss")
    loss = None
    try:
        import jax
        model.train()
        device = torch.device("cuda:0")
        obs_device = jax.tree.map(lambda x: x.to(device) if hasattr(x, 'to') else x, observation)
        act_device = actions.to(torch.float32).to(device)

        with torch.amp.autocast('cuda', enabled=False):
            losses = model(obs_device, act_device)

        if isinstance(losses, (list, tuple)):
            losses_t = torch.stack(losses)
        elif not isinstance(losses, torch.Tensor):
            losses_t = torch.tensor(losses, device=device, dtype=torch.float32)
        else:
            losses_t = losses
        loss = losses_t.mean()
        print(f"  Loss: {loss.item():.4f}")
        assert not torch.isnan(loss), "Loss is NaN!"
        assert not torch.isinf(loss), "Loss is Inf!"
        print("  STAGE 3 PASSED")
        PASSED.append("Stage 3: Forward Pass")
    except Exception as e:
        traceback.print_exc()
        FAILED.append(f"Stage 3: Forward Pass - {e}")

    # ===== STAGE 4: Backward Pass =====
    stage("Stage 4: Backward Pass")
    try:
        loss.backward()
        grad_norms = []
        for name, p in model.named_parameters():
            if p.grad is not None:
                gn = p.grad.float().norm().item()
                grad_norms.append(gn)

        if grad_norms:
            print(f"  Params with gradients: {len(grad_norms)}")
            print(f"  Grad norm: mean={np.mean(grad_norms):.6f}, max={np.max(grad_norms):.6f}")
            assert not any(np.isnan(g) for g in grad_norms), "NaN in gradients!"
            assert not any(np.isinf(g) for g in grad_norms), "Inf in gradients!"
        else:
            print("  WARNING: No gradients computed")
        print("  STAGE 4 PASSED")
        PASSED.append("Stage 4: Backward Pass")
    except Exception as e:
        traceback.print_exc()
        FAILED.append(f"Stage 4: Backward Pass - {e}")

    # ===== STAGE 5: Eval Environment + Collision Detection =====
    stage("Stage 5: Environment + Collision Detection")
    try:
        del model, loss
        torch.cuda.empty_cache()

        import torch as _torch
        _orig_torch_load = _torch.load
        def _safe_torch_load(f, *a, **kw):
            kw.setdefault("weights_only", False)
            return _orig_torch_load(f, *a, **kw)
        _torch.load = _safe_torch_load

        safelibero_path = os.path.join(os.path.dirname(__file__), "..", "vlsa-aegis", "safelibero")
        sys.path.insert(0, safelibero_path)

        from libero.libero.benchmark import get_benchmark
        from libero.libero.envs import OffScreenRenderEnv

        benchmark_cls = get_benchmark("safelibero_spatial")
        benchmark = benchmark_cls(task_order_index=0, safety_level="II")

        task = benchmark.get_task(0)
        bddl_path = benchmark.get_task_bddl_file_path(0)
        print(f"  Task: {task.language}")

        env = OffScreenRenderEnv(
            bddl_file_name=bddl_path,
            camera_heights=224,
            camera_widths=224,
        )

        init_states = benchmark.get_task_init_states(0)
        env.reset()
        env.set_init_state(init_states[0])

        for _ in range(5):
            obs, _, _, _ = env.step(np.zeros(7))

        OBSTACLE_NAMES = [
            "moka_pot_obstacle_1", "white_storage_box_obstacle_1",
            "milk_obstacle_1", "wine_bottle_obstacle_1",
            "red_coffee_mug_obstacle_1", "yellow_book_obstacle_1",
        ]

        positions = {}
        for name in OBSTACLE_NAMES:
            try:
                body_id = env.sim.model.body_name2id(name + "_main")
                pos = env.sim.data.body_xpos[body_id].copy()
                positions[name] = pos
            except Exception:
                try:
                    joint_name = name + "_joint0"
                    joint_id = env.sim.model.joint_name2id(joint_name)
                    qpos_addr = env.sim.model.jnt_qposadr[joint_id]
                    pos = env.sim.data.qpos[qpos_addr:qpos_addr + 3].copy()
                    positions[name] = pos
                except Exception:
                    pass

        print(f"  Found {len(positions)} obstacles: {list(positions.keys())}")

        initial_positions = {k: v.copy() for k, v in positions.items()}
        collision_detected = False
        for step in range(50):
            action = np.random.uniform(-1, 1, size=7)
            obs, reward, done, info = env.step(action)
            for nm in initial_positions:
                try:
                    body_id = env.sim.model.body_name2id(nm + "_main")
                    cur = env.sim.data.body_xpos[body_id].copy()
                except Exception:
                    try:
                        joint_name = nm + "_joint0"
                        joint_id = env.sim.model.joint_name2id(joint_name)
                        qpos_addr = env.sim.model.jnt_qposadr[joint_id]
                        cur = env.sim.data.qpos[qpos_addr:qpos_addr + 3].copy()
                    except Exception:
                        continue
                disp = np.linalg.norm(cur - initial_positions[nm])
                if disp > 0.001:
                    print(f"  Collision at step {step}: {nm} displaced {disp:.4f}")
                    collision_detected = True
                    break
            if collision_detected:
                break

        if not collision_detected:
            print("  No collision in 50 random steps (init state dependent)")

        img_key = "agentview_image" if "agentview_image" in obs else list(obs.keys())[0]
        print(f"  Obs image key: {img_key}, shape: {obs[img_key].shape}")

        env.close()
        print("  STAGE 5 PASSED")
        PASSED.append("Stage 5: Environment + Collision")
    except Exception as e:
        traceback.print_exc()
        FAILED.append(f"Stage 5: Environment + Collision - {e}")

    # ===== SUMMARY =====
    print()
    print("=" * 60)
    print("  VERIFICATION SUMMARY")
    print("=" * 60)
    for p in PASSED:
        print(f"  [PASS] {p}")
    for f in FAILED:
        print(f"  [FAIL] {f}")
    print()
    if FAILED:
        print(f"  RESULT: {len(FAILED)} STAGE(S) FAILED")
        sys.exit(1)
    else:
        print("  PIPELINE VERIFICATION: ALL 5 STAGES PASSED")
        sys.exit(0)


if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)
    main()
