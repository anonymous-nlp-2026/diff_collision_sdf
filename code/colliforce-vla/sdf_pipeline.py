"""SDF Pipeline: Extract meshes, compute SDF GT, verify gradients, visualize."""
import os
import sys
import json
import time
import numpy as np
import trimesh
import mujoco

sys.path.insert(0, './LIBERO')
os.environ.setdefault('LIBERO_CONFIG_PATH', '<PROJECT_ROOT>/.libero')
os.environ['MUJOCO_GL'] = 'egl'

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from pysdf import SDF

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = './data'
ARTIFACT_DIR = './artifacts'


def extract_visual_meshes(model, data):
    meshes = []
    for geom_id in range(model.ngeom):
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if model.geom_group[geom_id] != 1:
            continue
        mesh_id = model.geom_dataid[geom_id]
        vs = model.mesh_vertadr[mesh_id]
        vc = model.mesh_vertnum[mesh_id]
        fs = model.mesh_faceadr[mesh_id]
        fc = model.mesh_facenum[mesh_id]
        vertices = model.mesh_vert[vs:vs+vc].copy()
        faces = model.mesh_face[fs:fs+fc].copy()
        geom_pos = data.geom_xpos[geom_id]
        geom_rot = data.geom_xmat[geom_id].reshape(3, 3)
        world_verts = (geom_rot @ vertices.T).T + geom_pos
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"
        mesh = trimesh.Trimesh(vertices=world_verts, faces=faces)
        meshes.append({
            'name': name, 'mesh': mesh, 'watertight': mesh.is_watertight,
            'vertices': len(vertices), 'faces': len(faces),
        })
    return meshes


def try_repair(meshes):
    results = []
    for m in meshes:
        if m['watertight']:
            results.append({'name': m['name'], 'was_watertight': True, 'now_watertight': True, 'method': 'N/A'})
            continue
        mc = m['mesh'].copy()
        trimesh.repair.fill_holes(mc)
        trimesh.repair.fix_normals(mc)
        trimesh.repair.fix_winding(mc)
        now_wt = mc.is_watertight
        if now_wt:
            m['mesh'] = mc
            m['watertight'] = True
        results.append({'name': m['name'], 'was_watertight': False, 'now_watertight': now_wt,
                        'method': 'fill_holes+fix_normals+fix_winding'})
    return results


def compute_scene_sdf(meshes, n_points=4096, bbox_pad=0.05, sigma1=0.025, sigma2=0.008):
    wt = [m for m in meshes if m['watertight']]
    if not wt:
        print("ERROR: No watertight meshes!")
        return None, None
    all_verts = np.concatenate([m['mesh'].vertices for m in wt])
    bbox_min = all_verts.min(axis=0) - bbox_pad
    bbox_max = all_verts.max(axis=0) + bbox_pad
    n_near = int(n_points * 0.8)
    n_uniform = n_points - n_near
    near_points = []
    n_per = n_near // len(wt)
    for m in wt:
        sp = m['mesh'].sample(n_per)
        h = n_per // 2
        near_points.append(sp[:h] + np.random.randn(h, 3) * sigma1)
        near_points.append(sp[h:] + np.random.randn(n_per - h, 3) * sigma2)
    near_points = np.concatenate(near_points)
    uniform = np.random.uniform(bbox_min, bbox_max, (n_uniform, 3))
    qp = np.concatenate([near_points, uniform])
    sdf_stack = []
    for m in wt:
        f = SDF(m['mesh'].vertices, m['mesh'].faces)
        raw = np.array(f(qp))
        sdf_stack.append(-raw)  # sign flip
    ws = np.min(np.stack(sdf_stack), axis=0)
    ws = np.clip(ws, -0.05, 0.05)
    return qp, ws


def sdf_at_point(pt, sdf_funcs):
    vals = []
    for f in sdf_funcs:
        vals.append(-f(pt.reshape(1, 3))[0])
    return min(vals)


def verify_gradients(qp, sv, sdf_funcs, eps=1e-4, n_test=300):
    near = np.where(np.abs(sv) < 0.02)[0]
    if len(near) > n_test:
        near = np.random.choice(near, n_test, replace=False)
    if len(near) == 0:
        print("WARNING: No near-surface points")
        return 0.0, 0
    correct = total = 0
    for idx in near:
        pt = qp[idx]
        grad = np.zeros(3)
        for d in range(3):
            pp = pt.copy(); pp[d] += eps
            pm = pt.copy(); pm[d] -= eps
            grad[d] = (sdf_at_point(pp, sdf_funcs) - sdf_at_point(pm, sdf_funcs)) / (2*eps)
        gn = np.linalg.norm(grad)
        if gn < 1e-8:
            continue
        total += 1
        moved = pt + 0.001 * grad / gn
        if sdf_at_point(moved, sdf_funcs) > sv[idx]:
            correct += 1
    acc = correct / total if total > 0 else 0
    print(f"Gradient accuracy: {correct}/{total} = {acc:.1%}")
    return acc, total


def visualize_slice(sdf_funcs, meshes, z_slice, save_path, label, res=30):
    wt = [m for m in meshes if m['watertight']]
    av = np.concatenate([m['mesh'].vertices for m in wt])
    xr = np.linspace(av[:,0].min()-0.03, av[:,0].max()+0.03, res)
    yr = np.linspace(av[:,1].min()-0.03, av[:,1].max()+0.03, res)
    xx, yy = np.meshgrid(xr, yr)
    gp = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, z_slice)])
    gs = np.array([sdf_at_point(p, sdf_funcs) for p in gp]).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(10, 8))
    cs = ax.contourf(xx, yy, gs, levels=20, cmap='RdBu')
    ax.contour(xx, yy, gs, levels=[0], colors='black', linewidths=2)
    plt.colorbar(cs, label='SDF (m)')
    eps = 1e-4
    skip = max(1, res // 8)
    for i in range(0, len(xr), skip):
        for j in range(0, len(yr), skip):
            pt = np.array([xr[i], yr[j], z_slice])
            g = np.zeros(2)
            for d in range(2):
                pp = pt.copy(); pp[d] += eps
                pm = pt.copy(); pm[d] -= eps
                g[d] = (sdf_at_point(pp, sdf_funcs) - sdf_at_point(pm, sdf_funcs)) / (2*eps)
            n = np.linalg.norm(g)
            if n > 1e-6:
                ax.arrow(xr[i], yr[j], 0.008*g[0]/n, 0.008*g[1]/n, head_width=0.002, color='green', alpha=0.7)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_title(f'SDF z={z_slice:.3f}m — {label}')
    ax.set_aspect('equal'); plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()
    print(f"Saved: {save_path}")


def verify_collision_sign(sv):
    margin = 0.01
    cost = np.maximum(-sv + margin, 0)
    inside = sv < 0
    outside = sv > margin
    ip = bool(np.all(cost[inside] > 0)) if inside.any() else True
    op = bool(np.all(cost[outside] == 0)) if outside.any() else True
    print(f"  Inside: {inside.sum()}, cost>0: {ip}")
    print(f"  Outside(sdf>{margin}): {outside.sum()}, cost==0: {op}")
    return ip and op


def process_scene(suite_name, task_idx, label):
    print(f"\n{'='*60}")
    print(f"Processing: {label} ({suite_name} task {task_idx})")
    print(f"{'='*60}")
    bd = benchmark.get_benchmark_dict()
    suite = bd[suite_name]()
    task = suite.get_task(task_idx)
    print(f"Task: {task.name}")

    bddl_base = get_libero_path('bddl_files')
    bddl_full = os.path.join(bddl_base, task.problem_folder, task.bddl_file)
    print(f"BDDL: {bddl_full}")

    print("Loading environment...")
    env = OffScreenRenderEnv(bddl_file_name=bddl_full, camera_heights=128, camera_widths=128)
    env.reset()
    model = env.sim.model._model
    data = env.sim.data._data
    print(f"Model: {model.ngeom} geoms, {model.nmesh} meshes")

    print("\n--- Extract visual meshes ---")
    meshes = extract_visual_meshes(model, data)
    print(f"Visual meshes (group=1): {len(meshes)}")
    for m in meshes:
        print(f"  {m['name']}: {m['vertices']}v/{m['faces']}f wt={m['watertight']}")
    wt_before = sum(1 for m in meshes if m['watertight'])

    print("\n--- Repair ---")
    repair = try_repair(meshes)
    for r in repair:
        if not r['was_watertight']:
            print(f"  {r['name']}: {'FIXED' if r['now_watertight'] else 'FAILED'}")
    wt_after = sum(1 for m in meshes if m['watertight'])
    print(f"Watertight: {wt_before} -> {wt_after} / {len(meshes)}")

    print("\n--- Compute SDF ---")
    t0 = time.time()
    qp, sv = compute_scene_sdf(meshes, n_points=4096)
    print(f"Time: {time.time()-t0:.1f}s")
    if qp is None:
        env.close()
        return None

    print(f"Points: {qp.shape}, SDF: [{sv.min():.4f}, {sv.max():.4f}]")
    print(f"Mean: {sv.mean():.4f}, Std: {sv.std():.4f}")
    ns = np.abs(sv) < 0.02
    ins = sv < 0
    print(f"Near-surface: {ns.sum()} ({ns.mean():.1%}), Inside: {ins.sum()} ({ins.mean():.1%})")

    print("\n--- Sign verification ---")
    sign_ok = verify_collision_sign(sv)
    print(f"Sign: {'PASS' if sign_ok else 'FAIL'}")

    wt_meshes = [m for m in meshes if m['watertight']]
    sdf_funcs = [SDF(m['mesh'].vertices, m['mesh'].faces) for m in wt_meshes]

    print("\n--- Gradient verification ---")
    t0 = time.time()
    gacc, gtot = verify_gradients(qp, sv, sdf_funcs, n_test=300)
    print(f"Time: {time.time()-t0:.1f}s")
    if gacc < 0.9:
        print(f"BLOCKER: {gacc:.1%} < 90%!")
    else:
        print(f"PASS: {gacc:.1%} >= 90%")

    save_path = os.path.join(DATA_DIR, f"sdf_gt_{label}.npz")
    np.savez(save_path, query_points=qp, sdf_values=sv)
    print(f"Saved: {save_path}")

    print("\n--- Visualization ---")
    all_z = np.concatenate([m['mesh'].vertices[:,2] for m in wt_meshes])
    z_slice = float(np.median(all_z))
    print(f"z_slice={z_slice:.3f}")
    vis_path = os.path.join(ARTIFACT_DIR, f"sdf_slice_{label}.png")
    t0 = time.time()
    visualize_slice(sdf_funcs, meshes, z_slice, vis_path, label, res=30)
    print(f"Vis time: {time.time()-t0:.1f}s")

    env.close()
    return {
        'scene_label': label, 'task_name': task.name,
        'total_meshes': len(meshes), 'watertight_before': wt_before, 'watertight_after': wt_after,
        'repair_results': repair, 'n_query_points': len(qp),
        'sdf_min': float(sv.min()), 'sdf_max': float(sv.max()),
        'sdf_mean': float(sv.mean()), 'sdf_std': float(sv.std()),
        'near_surface_ratio': float(ns.mean()), 'inside_ratio': float(ins.mean()),
        'sign_ok': sign_ok, 'grad_accuracy': gacc, 'grad_total': gtot,
        'z_slice': z_slice, 'sdf_save_path': save_path, 'vis_save_path': vis_path,
    }


if __name__ == '__main__':
    np.random.seed(42)
    results = []
    r1 = process_scene('libero_spatial', 0, 'spatial_bowl')
    if r1: results.append(r1)
    r2 = process_scene('libero_object', 8, 'object_chocolate')
    if r2: results.append(r2)
    with open(os.path.join(ARTIFACT_DIR, 'sdf_pipeline_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for r in results:
        print(f"\n{r['scene_label']}:")
        print(f"  Meshes: {r['watertight_after']}/{r['total_meshes']} watertight")
        print(f"  SDF: [{r['sdf_min']:.4f}, {r['sdf_max']:.4f}]")
        print(f"  Near-surface: {r['near_surface_ratio']:.1%}, Inside: {r['inside_ratio']:.1%}")
        print(f"  Sign: {'PASS' if r['sign_ok'] else 'FAIL'}")
        print(f"  Gradient: {r['grad_accuracy']:.1%} (n={r['grad_total']})")
        print(f"  Status: {'BLOCKER' if r['grad_accuracy']<0.9 else 'PASS'}")
