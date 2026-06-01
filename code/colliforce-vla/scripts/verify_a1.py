"""
A1 SDF Decoder integration dry-run verification.
Loads pi0 model + weights, attaches SDF decoder + EE extractor,
runs 1 training step, checks dimensions/losses/gradients.

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts/verify_a1.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "openpi"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "openpi", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["HF_LEROBOT_HOME"] = "<DATA_ROOT>/colliforce_data"

import numpy as np
import torch
import torch.nn.functional as F

import jax
import safetensors.torch

import openpi.models.pi0_config as pi0_config
import openpi.models_pytorch.pi0_pytorch as pi0_pt
import openpi.training.config as _config
import openpi.training.data_loader as _data
from openpi.training.optimizer import CosineDecaySchedule
from openpi.training.config import AssetsConfig

from colliforce.sdf_decoder import SDFDecoder, compute_sdf_loss
from colliforce.ee_extractor import EEPositionExtractor


RESULTS = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    msg = f"[{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    RESULTS.append((name, condition, detail))
    return condition


def main():
    print("=" * 70)
    print("A1 SDF Decoder Integration — Dry-Run Verification")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    torch.manual_seed(42)
    np.random.seed(42)

    # ---- 1. SDF Decoder standalone ----
    print("\n--- Test 1: SDF Decoder standalone ---")
    sdf_dec = SDFDecoder(feature_dim=1024, hidden_dim=256).to(device)
    sdf_dec.train()

    n_params_sdf = sum(p.numel() for p in sdf_dec.parameters())
    print(f"SDF Decoder params: {n_params_sdf:,}")

    feat_dummy = torch.randn(2, 1024, device=device)
    pts_dummy = torch.randn(2, 64, 3, device=device)
    sdf_out, grad_out = sdf_dec(feat_dummy, pts_dummy, return_gradients=True)

    check("SDF output shape", sdf_out.shape == (2, 64), f"got {sdf_out.shape}")
    check("SDF gradient shape", grad_out.shape == (2, 64, 3), f"got {grad_out.shape}")
    check("SDF output finite", torch.isfinite(sdf_out).all().item())
    check("SDF gradient finite", torch.isfinite(grad_out).all().item())

    gt_dummy = torch.randn(2, 64, device=device) * 0.02
    losses = compute_sdf_loss(sdf_out, gt_dummy, grad_out, clamp_delta=0.05, eikonal_weight=0.1)
    check("SDF L1 loss finite", torch.isfinite(losses["sdf_l1"]).item(), f"val={losses['sdf_l1'].item():.6f}")
    check("Eikonal loss finite", torch.isfinite(losses["eikonal"]).item(), f"val={losses['eikonal'].item():.6f}")
    check("SDF total loss finite", torch.isfinite(losses["sdf_total"]).item(), f"val={losses['sdf_total'].item():.6f}")

    # ---- 2. EE Extractor standalone ----
    print("\n--- Test 2: EE Extractor standalone ---")
    ee_ext = EEPositionExtractor(feature_dim=1024, hidden_dim=256).to(device)
    ee_out = ee_ext(feat_dummy)
    check("EE output shape", ee_out.shape == (2, 3), f"got {ee_out.shape}")
    check("EE output finite", torch.isfinite(ee_out).all().item())

    # ---- 3. Load pi0 model + weights ----
    print("\n--- Test 3: Load pi0 model ---")
    t0 = time.time()
    model_cfg = pi0_config.Pi0Config(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        dtype="float32",
    )
    model = pi0_pt.PI0Pytorch(model_cfg).to(device)
    print(f"Model created in {time.time()-t0:.1f}s")

    weight_path = "<DATA_ROOT>/openpi_weights/pi0_base/model.safetensors"
    t0 = time.time()
    safetensors.torch.load_model(model, weight_path)
    print(f"Weights loaded in {time.time()-t0:.1f}s")

    check("action_out_proj exists", hasattr(model, "action_out_proj"))
    check("action_out_proj shape",
          model.action_out_proj.weight.shape[1] == 1024,
          f"got {model.action_out_proj.weight.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params/1e9:.2f}B")

    # ---- 4. Hook suffix_out ----
    print("\n--- Test 4: Hook suffix_out ---")
    hook_storage = {"features": None}

    def hook_fn(module, input, output):
        hook_storage["features"] = input[0]

    hook_handle = model.action_out_proj.register_forward_hook(hook_fn)
    check("Hook registered", True)

    # ---- 5. Load real data batch ----
    print("\n--- Test 5: Load data + forward pass ---")
    config = _config.TrainConfig(
        name="a1_verify",
        project_name="colliforce-vla",
        exp_name="a1_verify",
        model=model_cfg,
        data=_config.LeRobotLiberoDataConfig(
            repo_id="safelibero_spatial",
            base_config=_config.DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
            assets=AssetsConfig(assets_dir="./assets/baseline_spatial_bowl"),
        ),
        pytorch_weight_path="<DATA_ROOT>/openpi_weights/pi0_base",
        num_train_steps=2,
        batch_size=2,
        wandb_enabled=False,
        overwrite=True,
    )

    loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
    observation, actions = next(iter(loader))
    observation = jax.tree.map(lambda x: x.to(device), observation)
    actions = actions.to(torch.float32).to(device)

    B = actions.shape[0]
    print(f"Batch size: {B}")
    print(f"Actions shape: {actions.shape}")
    print(f"State shape: {observation.state.shape}")
    print(f"State[:, :3] (EE pos): {observation.state[:, :3]}")

    check("Batch loaded", B > 0)
    check("State has 8 dims", observation.state.shape[-1] >= 8, f"got {observation.state.shape[-1]}")

    # Forward pass
    model.train()
    t0 = time.time()
    action_losses = model(observation, actions)
    fwd_time = time.time() - t0
    action_loss = action_losses.mean()

    print(f"Forward pass: {fwd_time:.2f}s")
    check("Action loss finite", torch.isfinite(action_loss).item(), f"val={action_loss.item():.6f}")

    # ---- 6. Verify hook capture ----
    print("\n--- Test 6: Verify hook capture ---")
    suffix_out = hook_storage["features"]
    check("Hook captured suffix_out", suffix_out is not None)
    if suffix_out is not None:
        check("suffix_out shape [B,50,1024]",
              suffix_out.shape == (B, 50, 1024),
              f"got {suffix_out.shape}")
        check("suffix_out finite", torch.isfinite(suffix_out).all().item())
        check("suffix_out dtype float32", suffix_out.dtype == torch.float32, f"got {suffix_out.dtype}")

    # ---- 7. SDF loss with real features ----
    print("\n--- Test 7: SDF loss with real features ---")
    pooled = suffix_out.mean(dim=1)  # [B, 1024]
    check("Pooled features shape", pooled.shape == (B, 1024), f"got {pooled.shape}")

    sdf_data = np.load("./data/sdf_gt_spatial_bowl.npz")
    sdf_query_all = torch.tensor(sdf_data["query_points"], dtype=torch.float32, device=device)
    sdf_values_all = torch.tensor(sdf_data["sdf_values"], dtype=torch.float32, device=device)

    n_query = 64
    indices = torch.randint(0, len(sdf_query_all), (n_query,))
    query_pts = sdf_query_all[indices].unsqueeze(0).expand(B, -1, -1)
    gt_sdf = sdf_values_all[indices].unsqueeze(0).expand(B, -1)

    sdf_dec = sdf_dec.to(device)
    pred_sdf, grad_sdf = sdf_dec(pooled.detach(), query_pts, return_gradients=True)
    sdf_losses = compute_sdf_loss(pred_sdf, gt_sdf, grad_sdf, clamp_delta=0.05, eikonal_weight=0.1)

    check("Pred SDF shape", pred_sdf.shape == (B, n_query), f"got {pred_sdf.shape}")
    check("SDF L1 reasonable", sdf_losses["sdf_l1"].item() < 1.0, f"val={sdf_losses['sdf_l1'].item():.6f}")
    check("Eikonal reasonable", sdf_losses["eikonal"].item() < 100.0, f"val={sdf_losses['eikonal'].item():.6f}")
    print(f"SDF L1: {sdf_losses['sdf_l1'].item():.6f}, Eikonal: {sdf_losses['eikonal'].item():.6f}")

    # ---- 8. EE position loss ----
    print("\n--- Test 8: EE position loss ---")
    ee_ext = ee_ext.to(device)
    ee_pred = ee_ext(pooled.detach())
    ee_gt = observation.state[:, :3].float()
    ee_loss = F.mse_loss(ee_pred, ee_gt)
    check("EE loss finite", torch.isfinite(ee_loss).item(), f"val={ee_loss.item():.6f}")
    print(f"EE pred: {ee_pred[0].detach().cpu().numpy()}")
    print(f"EE gt:   {ee_gt[0].detach().cpu().numpy()}")

    # ---- 9. Phase 1 backward (frozen backbone) ----
    print("\n--- Test 9: Phase 1 backward (frozen backbone) ---")

    # Freeze backbone
    for p in model.parameters():
        p.requires_grad = False
    for p in sdf_dec.parameters():
        p.requires_grad = True
    for p in ee_ext.parameters():
        p.requires_grad = True

    # Re-run forward (need fresh computation graph)
    hook_storage["features"] = None
    action_losses_p1 = model(observation, actions)
    action_loss_p1 = action_losses_p1.mean()
    suffix_out_p1 = hook_storage["features"]
    pooled_p1 = suffix_out_p1.mean(dim=1).detach()

    pred_sdf_p1, grad_sdf_p1 = sdf_dec(pooled_p1, query_pts, return_gradients=True)
    sdf_losses_p1 = compute_sdf_loss(pred_sdf_p1, gt_sdf, grad_sdf_p1)
    ee_pred_p1 = ee_ext(pooled_p1)
    ee_loss_p1 = F.mse_loss(ee_pred_p1, ee_gt)

    total_loss_p1 = 0.1 * sdf_losses_p1["sdf_total"] + 0.01 * ee_loss_p1
    total_loss_p1.backward()

    check("Total loss P1 finite", torch.isfinite(total_loss_p1).item(), f"val={total_loss_p1.item():.6f}")

    # Check gradients
    sdf_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in sdf_dec.parameters())
    ee_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in ee_ext.parameters())
    backbone_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                           for p in model.parameters() if p.numel() > 0)

    check("Phase 1: SDF decoder has gradients", sdf_has_grad)
    check("Phase 1: EE extractor has gradients", ee_has_grad)
    check("Phase 1: backbone has NO gradients", not backbone_has_grad)

    # Check no NaN in gradients
    sdf_nan = any(p.grad is not None and torch.isnan(p.grad).any() for p in sdf_dec.parameters())
    ee_nan = any(p.grad is not None and torch.isnan(p.grad).any() for p in ee_ext.parameters())
    check("Phase 1: no NaN in SDF grads", not sdf_nan)
    check("Phase 1: no NaN in EE grads", not ee_nan)

    # Zero grads
    for p in sdf_dec.parameters():
        if p.grad is not None:
            p.grad = None
    for p in ee_ext.parameters():
        if p.grad is not None:
            p.grad = None

    # ---- 10. Phase 2 backward (unfrozen LoRA) ----
    print("\n--- Test 10: Phase 2 backward (unfrozen LoRA) ---")

    # Unfreeze all model parameters for Phase 2 (PyTorch pi0 has no separate LoRA params)
    unfrozen_count = 0
    for p in model.parameters():
        p.requires_grad = True
        unfrozen_count += 1
    print(f"Model params unfrozen: {unfrozen_count}")

    # Re-run forward
    hook_storage["features"] = None
    action_losses_p2 = model(observation, actions)
    action_loss_p2 = action_losses_p2.mean()
    suffix_out_p2 = hook_storage["features"]
    pooled_p2 = suffix_out_p2.mean(dim=1)  # NOT detached in Phase 2

    pred_sdf_p2, grad_sdf_p2 = sdf_dec(pooled_p2, query_pts, return_gradients=True)
    sdf_losses_p2 = compute_sdf_loss(pred_sdf_p2, gt_sdf, grad_sdf_p2)
    ee_pred_p2 = ee_ext(pooled_p2)
    ee_loss_p2 = F.mse_loss(ee_pred_p2, ee_gt)

    total_loss_p2 = action_loss_p2 + 0.1 * sdf_losses_p2["sdf_total"] + 0.01 * ee_loss_p2
    total_loss_p2.backward()

    check("Total loss P2 finite", torch.isfinite(total_loss_p2).item(), f"val={total_loss_p2.item():.6f}")

    model_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                        for p in model.parameters())
    sdf_has_grad_p2 = any(p.grad is not None and p.grad.abs().sum() > 0 for p in sdf_dec.parameters())
    ee_has_grad_p2 = any(p.grad is not None and p.grad.abs().sum() > 0 for p in ee_ext.parameters())

    check("Phase 2: model params have gradients", model_has_grad)
    check("Phase 2: SDF decoder has gradients", sdf_has_grad_p2)
    check("Phase 2: EE extractor has gradients", ee_has_grad_p2)

    # NaN check
    any_nan = False
    for n, p in model.named_parameters():
        if p.grad is not None and torch.isnan(p.grad).any():
            print(f"  NaN gradient in: {n}")
            any_nan = True
    for p in sdf_dec.parameters():
        if p.grad is not None and torch.isnan(p.grad).any():
            any_nan = True
    for p in ee_ext.parameters():
        if p.grad is not None and torch.isnan(p.grad).any():
            any_nan = True

    check("Phase 2: no NaN in any gradients", not any_nan)

    # ---- 11. Full 1-step training cycle ----
    print("\n--- Test 11: Full 1-step optimizer cycle ---")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_params += list(sdf_dec.parameters()) + list(ee_ext.parameters())
    test_optim = torch.optim.AdamW(trainable_params, lr=2.5e-5, weight_decay=1e-4)

    # Zero grads
    test_optim.zero_grad(set_to_none=True)

    # Forward
    hook_storage["features"] = None
    action_losses_step = model(observation, actions)
    action_loss_step = action_losses_step.mean()
    suffix_out_step = hook_storage["features"]
    pooled_step = suffix_out_step.mean(dim=1)

    pred_sdf_step, grad_sdf_step = sdf_dec(pooled_step, query_pts, return_gradients=True)
    sdf_losses_step = compute_sdf_loss(pred_sdf_step, gt_sdf, grad_sdf_step)
    ee_pred_step = ee_ext(pooled_step)
    ee_loss_step = F.mse_loss(ee_pred_step, ee_gt)

    total_loss_step = action_loss_step + 0.1 * sdf_losses_step["sdf_total"] + 0.01 * ee_loss_step

    # Backward + clip + step
    total_loss_step.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
    test_optim.step()

    check("Optimizer step completed", True)
    check("Grad norm finite", torch.isfinite(torch.tensor(float(grad_norm))).item(), f"val={float(grad_norm):.4f}")

    # Cleanup
    hook_handle.remove()

    # ---- Summary ----
    print("\n" + "=" * 70)
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"RESULTS: {passed} passed, {failed} failed, {len(RESULTS)} total")

    if failed > 0:
        print("\nFailed checks:")
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"  FAIL: {name} — {detail}")
    else:
        print("\nAll checks passed!")

    # Write report
    report_path = "./data/a1_integration_report.txt"
    with open(report_path, "w") as f:
        f.write("A1 SDF Decoder Integration Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Device: {device}\n")
        if torch.cuda.is_available():
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
            f.write(f"Peak memory: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB\n")
        f.write(f"\nResults: {passed}/{len(RESULTS)} passed\n\n")
        for name, ok, detail in RESULTS:
            status = "PASS" if ok else "FAIL"
            f.write(f"[{status}] {name}")
            if detail:
                f.write(f" — {detail}")
            f.write("\n")
        f.write(f"\nSDF Decoder params: {n_params_sdf:,}\n")
        f.write(f"EE Extractor params: {sum(p.numel() for p in ee_ext.parameters()):,}\n")
        f.write(f"Model params: {total_params/1e9:.2f}B\n")

    print(f"\nReport written to {report_path}")

    if torch.cuda.is_available():
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")

    print("=" * 70)
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
