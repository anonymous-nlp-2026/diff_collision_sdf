"""
dfCAR Threshold Sensitivity Analysis

For each eval condition, computes at multiple delta thresholds:
- filtered_CAR: safe fraction among episodes with ee_disp > delta
- fake_safe_rate: (safe & ee_disp <= delta) / total
- valid_episodes: count with ee_disp > delta
"""
import json, os, re, csv, sys, hashlib
import numpy as np
from pathlib import Path
from collections import defaultdict

THRESHOLDS_MM = [2, 5, 10, 20, 50, 100, 150, 200, 300]
THRESHOLDS_M = [t / 1000.0 for t in THRESHOLDS_MM]

OUTPUT_DIR = Path("./analysis/threshold_sensitivity")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEARCH_DIRS = [
    "./eval_results",
    "<DATA_ROOT>/eval_results",
]

def find_eval_jsons():
    found = []
    for d in SEARCH_DIRS:
        p = Path(d)
        if not p.exists():
            continue
        for f in sorted(p.glob("*.json")):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                results = data.get("results", [])
                if results and results[0].get("episodes"):
                    ep0 = results[0]["episodes"][0]
                    if "max_ee_displacement" in ep0:
                        found.append(f)
            except Exception:
                pass
    return found


def parse_condition(filepath, config):
    """Extract condition label from checkpoint path (not config.seed)."""
    ckpt = config.get("checkpoint", "")

    # Extract seed from checkpoint path: seed0, seed1, seed2...
    seed_m = re.search(r"seed(\d+)", ckpt)
    seed = seed_m.group(1) if seed_m else "?"

    # Extract lambda
    lam_m = re.search(r"_l(\d{2,3})", ckpt)
    lam = "?"
    if lam_m:
        val = lam_m.group(1)
        if val == "005":
            lam = "0.05"
        elif val == "01":
            lam = "0.1"
        else:
            lam = f"0.{val}"

    # Checkpoint type from path
    if "/best" in ckpt:
        ckpt_type = "best"
    else:
        ckpt_m = re.search(r"/(\d+)/?$", ckpt)
        ckpt_type = f"{int(ckpt_m.group(1))//1000}k" if ckpt_m else "other"

    label = f"s{seed}_lam{lam}_{ckpt_type}"
    return label, seed, lam, ckpt_type


def load_episodes(filepath):
    with open(filepath) as f:
        data = json.load(f)
    config = data.get("config", {})
    records = []
    for task in data.get("results", []):
        task_name = task.get("task_name", "unknown")
        for ep in task.get("episodes", []):
            ee_disp = ep.get("max_ee_displacement")
            collided = ep.get("collided")
            obs_disp = ep.get("max_displacement", 0)
            if ee_disp is not None and collided is not None:
                records.append({
                    "ee_disp": ee_disp,
                    "collided": collided,
                    "obs_disp": obs_disp,
                    "success": ep.get("success", False),
                    "task": task_name,
                })
    return config, records


def deduplicate_files(files):
    """Deduplicate by episode data content."""
    seen = {}
    unique = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        # Hash just the episode data
        episodes_str = json.dumps([
            [(ep.get("max_ee_displacement"), ep.get("collided"))
             for ep in task.get("episodes", [])]
            for task in data.get("results", [])
        ], sort_keys=True)
        h = hashlib.md5(episodes_str.encode()).hexdigest()
        if h not in seen:
            seen[h] = f
            unique.append(f)
        else:
            print(f"  [dedup] {f.name} is duplicate of {seen[h].name}, skipping")
    return unique


def compute_metrics(records, delta):
    active = [r for r in records if r["ee_disp"] > delta]
    inactive = [r for r in records if r["ee_disp"] <= delta]
    n_total = len(records)
    n_active = len(active)

    safe_active = sum(1 for r in active if not r["collided"])
    safe_inactive = sum(1 for r in inactive if not r["collided"])

    filtered_car = safe_active / n_active if n_active > 0 else None
    fake_safe_rate = safe_inactive / n_total if n_total > 0 else 0.0
    raw_car = sum(1 for r in records if not r["collided"]) / n_total if n_total > 0 else None

    return {
        "filtered_CAR": filtered_car,
        "raw_CAR": raw_car,
        "fake_safe_rate": fake_safe_rate,
        "valid_episodes": n_active,
        "excluded_episodes": len(inactive),
        "total_episodes": n_total,
    }


def analyze_bimodality(all_displacements):
    if len(all_displacements) < 10:
        return "insufficient data"

    vals = np.sort(all_displacements)
    diffs = np.diff(vals)
    median_diff = np.median(diffs)

    max_gap_idx = np.argmax(diffs)
    max_gap = diffs[max_gap_idx]
    gap_location = (vals[max_gap_idx] + vals[max_gap_idx + 1]) / 2

    p10 = np.percentile(vals, 10)
    p25 = np.percentile(vals, 25)
    p50 = np.percentile(vals, 50)
    p75 = np.percentile(vals, 75)
    p90 = np.percentile(vals, 90)

    below_100mm = np.sum(vals < 0.1)
    above_100mm = np.sum(vals >= 0.1)

    return {
        "n": len(vals),
        "min": float(vals[0]),
        "max": float(vals[-1]),
        "mean": float(np.mean(vals)),
        "median": float(p50),
        "p10": float(p10),
        "p25": float(p25),
        "p75": float(p75),
        "p90": float(p90),
        "max_gap": float(max_gap),
        "max_gap_location_m": float(gap_location),
        "max_gap_location_mm": float(gap_location * 1000),
        "median_gap": float(median_diff),
        "gap_ratio": float(max_gap / median_diff) if median_diff > 0 else float("inf"),
        "below_100mm": int(below_100mm),
        "above_100mm": int(above_100mm),
    }


def plot_distributions(condition_data, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = sorted(condition_data.keys())
    n = len(conditions)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), squeeze=False)

    for i, cond in enumerate(conditions):
        ax = axes[i, 0]
        records = condition_data[cond]
        ee_vals = np.array([r["ee_disp"] for r in records]) * 1000
        safe_mask = np.array([not r["collided"] for r in records])

        safe_vals = ee_vals[safe_mask]
        coll_vals = ee_vals[~safe_mask]

        bins = np.linspace(0, max(ee_vals.max() + 50, 1300), 60)
        ax.hist(safe_vals, bins=bins, alpha=0.6, label=f"safe (n={len(safe_vals)})", color="steelblue")
        ax.hist(coll_vals, bins=bins, alpha=0.6, label=f"collision (n={len(coll_vals)})", color="salmon")

        for t in [100]:
            ax.axvline(t, color="black", linestyle="--", linewidth=1.5, label=f"current δ={t}mm")

        ax.set_title(f"{cond} (N={len(records)})", fontsize=11)
        ax.set_xlabel("Max EE Displacement (mm)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved histogram: {output_path}")


def plot_sensitivity_curves(csv_rows, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cond_data = defaultdict(lambda: {"thresholds": [], "cars": [], "valid": []})
    for row in csv_rows:
        c = row["condition"]
        cond_data[c]["thresholds"].append(row["threshold_mm"])
        cond_data[c]["cars"].append(row["filtered_CAR"])
        cond_data[c]["valid"].append(row["valid_episodes"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    colors = plt.cm.tab10(np.linspace(0, 1, len(cond_data)))
    for (cond, d), color in zip(sorted(cond_data.items()), colors):
        ts = d["thresholds"]
        cars = [c if c is not None else np.nan for c in d["cars"]]
        valids = d["valid"]
        ax1.plot(ts, cars, "o-", label=cond, color=color, markersize=4)
        ax2.plot(ts, valids, "s-", label=cond, color=color, markersize=4)

    ax1.axvline(100, color="black", linestyle="--", alpha=0.5, label="δ=100mm (current)")
    ax1.set_ylabel("filtered_CAR")
    ax1.set_title("dfCAR Threshold Sensitivity")
    ax1.legend(fontsize=8, loc="best")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    ax2.axvline(100, color="black", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Threshold δ (mm)")
    ax2.set_ylabel("Valid Episodes (ee_disp > δ)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved sensitivity curves: {output_path}")


def plot_combined_histogram(all_records, output_path):
    """Single combined histogram of ALL ee_displacements."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ee_vals = np.array([r["ee_disp"] for r in all_records]) * 1000
    safe_mask = np.array([not r["collided"] for r in all_records])

    fig, ax = plt.subplots(figsize=(10, 4))
    bins = np.linspace(0, max(ee_vals.max() + 50, 1300), 80)
    ax.hist(ee_vals[safe_mask], bins=bins, alpha=0.6, label=f"safe (n={safe_mask.sum()})", color="steelblue")
    ax.hist(ee_vals[~safe_mask], bins=bins, alpha=0.6, label=f"collision (n={(~safe_mask).sum()})", color="salmon")

    for t_mm, ls, lbl in [(100, "--", "δ=100mm (current)"), (200, ":", "δ=200mm")]:
        ax.axvline(t_mm, color="black", linestyle=ls, linewidth=1.5, alpha=0.7, label=lbl)

    ax.set_xlabel("Max EE Displacement (mm)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"EE Displacement Distribution (all conditions, N={len(all_records)})", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1400)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved combined histogram: {output_path}")


def main():
    print("=" * 60)
    print("dfCAR Threshold Sensitivity Analysis")
    print("=" * 60)

    # Phase 1: discover files
    print("\n[1] Discovering eval JSONs with max_ee_displacement...")
    files = find_eval_jsons()
    print(f"  Found {len(files)} files")
    for f in files:
        with open(f) as fh:
            config = json.load(fh).get("config", {})
        ckpt = config.get("checkpoint", "N/A")
        print(f"    {f.name}  ->  ckpt: ...{ckpt[-60:]}")

    files = deduplicate_files(files)
    print(f"  After dedup: {len(files)} files")

    # Phase 2: load data per condition
    print("\n[2] Loading data and parsing conditions...")
    condition_data = defaultdict(list)
    condition_files = defaultdict(list)
    all_records = []

    for f in files:
        config, records = load_episodes(f)
        label, seed, lam, ckpt_type = parse_condition(f, config)
        n_tasks = len(json.load(open(f)).get("results", []))
        print(f"  {f.name} -> {label} ({len(records)} episodes across {n_tasks} tasks)")
        condition_data[label].extend(records)
        condition_files[label].append(str(f))
        all_records.extend(records)

    # Phase 3: compute metrics at each threshold
    print("\n[3] Computing metrics at each threshold...")
    csv_rows = []
    for cond in sorted(condition_data.keys()):
        records = condition_data[cond]
        for delta_m, delta_mm in zip(THRESHOLDS_M, THRESHOLDS_MM):
            metrics = compute_metrics(records, delta_m)
            row = {
                "condition": cond,
                "threshold_mm": delta_mm,
                "threshold_m": delta_m,
                "filtered_CAR": metrics["filtered_CAR"],
                "raw_CAR": metrics["raw_CAR"],
                "fake_safe_rate": metrics["fake_safe_rate"],
                "valid_episodes": metrics["valid_episodes"],
                "excluded_episodes": metrics["excluded_episodes"],
                "total_episodes": metrics["total_episodes"],
            }
            csv_rows.append(row)

    # Also compute aggregate (all conditions pooled)
    for delta_m, delta_mm in zip(THRESHOLDS_M, THRESHOLDS_MM):
        metrics = compute_metrics(all_records, delta_m)
        csv_rows.append({
            "condition": "ALL_POOLED",
            "threshold_mm": delta_mm,
            "threshold_m": delta_m,
            **metrics,
        })

    csv_path = OUTPUT_DIR / "threshold_sensitivity.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"  Saved CSV: {csv_path}")

    # Print summary table
    print("\n[4] Summary Table:")
    print(f"{'Condition':<25} {'δ(mm)':>6} {'fCAR':>8} {'rawCAR':>8} {'fakeSafe':>8} {'valid':>6} {'excl':>6} {'total':>6}")
    print("-" * 85)
    for row in csv_rows:
        fcar = f"{row['filtered_CAR']:.3f}" if row["filtered_CAR"] is not None else "N/A"
        rcar = f"{row['raw_CAR']:.3f}" if row["raw_CAR"] is not None else "N/A"
        print(f"{row['condition']:<25} {row['threshold_mm']:>6} {fcar:>8} {rcar:>8} "
              f"{row['fake_safe_rate']:.3f} {row['valid_episodes']:>6} {row['excluded_episodes']:>6} {row['total_episodes']:>6}")

    # Phase 4: bimodality analysis
    print("\n[5] EE Displacement Distribution Analysis (GLOBAL):")
    all_ee = np.array([r["ee_disp"] for r in all_records])
    print(f"  Total episodes: {len(all_ee)}")
    bio = analyze_bimodality(all_ee)
    if isinstance(bio, dict):
        print(f"  Range: [{bio['min']*1000:.1f}, {bio['max']*1000:.1f}] mm")
        print(f"  Mean: {bio['mean']*1000:.1f} mm, Median: {bio['median']*1000:.1f} mm")
        print(f"  P10={bio['p10']*1000:.1f}mm  P25={bio['p25']*1000:.1f}mm  P75={bio['p75']*1000:.1f}mm  P90={bio['p90']*1000:.1f}mm")
        print(f"  Largest gap: {bio['max_gap']*1000:.1f}mm at {bio['max_gap_location_mm']:.1f}mm (ratio: {bio['gap_ratio']:.1f}x)")
        print(f"  Below 100mm: {bio['below_100mm']}  Above 100mm: {bio['above_100mm']}")

        if bio["below_100mm"] == 0:
            print(f"\n  *** KEY FINDING: ALL {bio['above_100mm']} episodes have ee_disp > 100mm ***")
            print(f"  *** Minimum ee_disp = {bio['min']*1000:.1f}mm >> δ=100mm ***")
            print(f"  *** δ=100mm threshold filters out ZERO episodes ***")
            print(f"  *** dfCAR == raw CAR for any δ ∈ [0, {bio['min']*1000:.0f}mm) ***")

    # Per-condition analysis
    print("\n  Per-condition min/max ee_disp:")
    for cond in sorted(condition_data.keys()):
        records = condition_data[cond]
        ee = np.array([r["ee_disp"] for r in records])
        print(f"    {cond}: [{ee.min()*1000:.1f}, {ee.max()*1000:.1f}] mm, mean={ee.mean()*1000:.1f}mm")

    # Phase 5: plots
    print("\n[6] Generating plots...")
    try:
        plot_distributions(condition_data, OUTPUT_DIR / "ee_displacement_distribution.png")
        plot_sensitivity_curves(csv_rows, OUTPUT_DIR / "sensitivity_curves.png")
        plot_combined_histogram(all_records, OUTPUT_DIR / "ee_displacement_combined.png")
    except Exception as e:
        print(f"  Plot error: {e}")
        import traceback
        traceback.print_exc()

    # Summary JSON
    summary = {
        "files_found": len(files),
        "total_episodes": len(all_records),
        "conditions": {c: len(r) for c, r in condition_data.items()},
        "condition_files": dict(condition_files),
        "bimodality_global": bio if isinstance(bio, dict) else {"note": bio},
        "key_finding": (
            f"All {len(all_records)} episodes have max_ee_displacement > {bio['min']*1000:.1f}mm. "
            f"The δ=100mm threshold filters out 0 episodes. "
            f"dfCAR is identical to raw CAR for any δ < {bio['min']*1000:.0f}mm."
        ) if isinstance(bio, dict) else "N/A",
        "thresholds_mm": THRESHOLDS_MM,
    }
    summary_path = OUTPUT_DIR / "analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved summary: {summary_path}")

    # List conditions missing ee_displacement
    print("\n[7] Files WITHOUT max_ee_displacement (cannot analyze):")
    for d in SEARCH_DIRS:
        p = Path(d)
        if not p.exists():
            continue
        for f in sorted(p.glob("*.json")):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                results = data.get("results", [])
                if results and results[0].get("episodes"):
                    ep0 = results[0]["episodes"][0]
                    if "max_ee_displacement" not in ep0:
                        config = data.get("config", {})
                        ckpt = config.get("checkpoint", "")
                        # Infer type from filename
                        fname = f.stem
                        if "baseline" in fname or "pi05" in fname:
                            tag = "[BASELINE]"
                        elif "oracle" in fname:
                            tag = "[ORACLE]"
                        elif "a1" in fname or "a3" in fname:
                            tag = "[SDF-trained]"
                        else:
                            tag = "[?]"
                        print(f"  {tag} {f.name}")
            except Exception:
                pass

    print("\n" + "=" * 60)
    print("Output files:")
    for p in sorted(OUTPUT_DIR.iterdir()):
        if p.suffix in (".csv", ".json", ".png"):
            print(f"  {p}")
    print("=" * 60)


if __name__ == "__main__":
    main()
