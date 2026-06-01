# SDF Pipeline Report

## 1. Mesh Statistics

### Scene 1: spatial_bowl (Spatial — "Pick up bowl between plate and ramekin")

| Metric | Value |
|--------|-------|
| Total visual meshes (group=1) | 68 |
| Watertight before repair | 19 (27.9%) |
| Watertight after repair | 21 (30.9%) |
| Repair success | 2/49 non-watertight fixed |

Repaired meshes: `robot0_g14_vis`, `robot0_g19_vis`

Key watertight objects (scene obstacles):
- `akita_black_bowl_1_g0` (127566v) — watertight
- `akita_black_bowl_2_g0` (127566v) — watertight
- `cookies_1_g0` (53562v) — watertight
- `glazed_rim_porcelain_ramekin_1_g0` (48822v) — watertight
- `plate_1_g0` (10392v) — NOT watertight (repair failed)
- `flat_stove_1_burner` (356v) — watertight
- `wooden_cabinet_1_g10/g17/g21/g28/g32/g39` — watertight (cabinet parts)

Non-watertight: mostly robot arm visual meshes (complex multi-part geometry), `plate_1_g0`, `mount0_pedestal_vis`.

### Scene 2: object_chocolate (Object — "Pick up chocolate pudding, place in basket")

| Metric | Value |
|--------|-------|
| Total visual meshes (group=1) | 60 |
| Watertight before repair | 13 (21.7%) |
| Watertight after repair | 17 (28.3%) |
| Repair success | 4/47 non-watertight fixed |

Repaired meshes: `robot0_g14_vis`, `robot0_g19_vis`, `ketchup_1_g0`, `alphabet_soup_1_g0`

Key watertight objects:
- `chocolate_pudding_1_g0` (54162v) — watertight
- `basket_1_g0` (199170v) — watertight
- `orange_juice_1_g0` (59796v) — watertight
- `bbq_sauce_1_g0` (48270v) — watertight
- `salad_dressing_1_g0` (58830v) — watertight
- `ketchup_1_g0` (46605v) — repaired to watertight
- `alphabet_soup_1_g0` (41577v) — repaired to watertight

## 2. SDF Ground Truth Statistics

| Metric | spatial_bowl | object_chocolate |
|--------|-------------|-----------------|
| Query points | 4084 | 4084 |
| SDF range | [-0.0500, 0.0500] | [-0.0500, 0.0500] |
| SDF mean | 0.0178 | 0.0163 |
| SDF std | 0.0221 | 0.0218 |
| Near-surface (|SDF|<0.02) | 61.0% | 61.5% |
| Inside (SDF<0) | 20.4% | 23.8% |

Sampling: 80% near-surface (dual-scale Gaussian σ₁=0.025, σ₂=0.008) + 20% uniform bbox.
Truncation: clipped to [-0.05, 0.05].

## 3. Gradient Direction Verification

| Scene | Accuracy | Tested Points | Status |
|-------|----------|---------------|--------|
| spatial_bowl | **98.7%** (296/300) | 300 | PASS (≥90%) |
| object_chocolate | **99.7%** (299/300) | 300 | PASS (≥90%) |

Method: Finite difference (eps=1e-4) on near-surface points (|SDF|<0.02). Verify that moving 0.001m along gradient direction increases SDF value (moves away from obstacle).

**No blockers.**

## 4. Sign Flip Verification

pysdf convention: positive = inside, negative = outside.
Standard SDF convention: negative = inside, positive = outside.
Pipeline applies `sdf = -pysdf_output` at computation time.

**Collision cost check: `ReLU(-SDF + margin)`**
- Inside points (SDF < 0): cost > 0 — **PASS** (both scenes)
- Outside points (SDF > margin): cost == 0 — **PASS** (both scenes)

## 5. Visualization

- `artifacts/sdf_slice_spatial_bowl.png` — XY slice at z=0.868m (median vertex height), shows SDF contours with zero-level set (black) and gradient arrows (green)
- `artifacts/sdf_slice_object_chocolate.png` — XY slice at z=0.089m, same format

Grid resolution: 30×30 for computation efficiency.

## 6. Output Files

| File | Path | Size |
|------|------|------|
| SDF GT (spatial) | `data/sdf_gt_spatial_bowl.npz` | 113K |
| SDF GT (object) | `data/sdf_gt_object_chocolate.npz` | 113K |
| Vis (spatial) | `artifacts/sdf_slice_spatial_bowl.png` | 116K |
| Vis (object) | `artifacts/sdf_slice_object_chocolate.png` | 120K |
| Full results | `artifacts/sdf_pipeline_results.json` | 23K |

## 7. Notes

- Non-watertight meshes (primarily robot arm parts) are excluded from SDF computation. This is acceptable since SDF GT is used for scene obstacle avoidance, and robot self-collision is handled separately.
- `plate_1_g0` in spatial scene is not watertight and could not be repaired. This may affect SDF accuracy near the plate.
- Pipeline script: `<PROJECT_ROOT>/colliforce-vla/sdf_pipeline.py` (reusable for other LIBERO scenes)
