import json, csv
import numpy as np

base = "<DATA_ROOT>/eval_results"
experiments = [
    ("LoRA s0", f"{base}/lora_baseline_s0_reeval_safelibero_object_levelII_seed7.json"),
    ("LoRA s1", f"{base}/reeval_lora_s1_30k.json"),
    ("SDF λ=0.005 s0", f"{base}/reeval_sdf_s0_l005_30k.json"),
    ("SDF λ=0.005 s1", f"{base}/reeval_sdf_s1_l005_30k.json"),
    ("SDF λ=0.005 s2", f"{base}/reeval_sdf_s2_l005_30k.json"),
    ("SDF λ=0.01 s0", f"{base}/reeval_sdf_s0_l01_30k.json"),
    ("SDF λ=0.01 s1", f"{base}/reeval_sdf_s1_l01_30k.json"),
    ("SDF λ=0.01 s2", f"{base}/reeval_sdf_s2_l01_30k.json"),
]

thresholds_mm = [2, 5, 10, 20, 50, 100, 200, 300]

results = []
for label, path in experiments:
    with open(path) as f:
        data = json.load(f)

    episodes = []
    for task in data["results"]:
        episodes.extend(task["episodes"])

    print(f"{label}: {len(episodes)} episodes")

    for thresh_mm in thresholds_mm:
        thresh_m = thresh_mm / 1000.0
        active = [ep for ep in episodes if ep.get("max_ee_displacement", 0) >= thresh_m]
        if len(active) > 0:
            safe = sum(1 for ep in active if not ep.get("collided", True))
            dfCAR = safe / len(active) * 100
        else:
            dfCAR = None
        results.append({
            "label": label,
            "threshold_mm": thresh_mm,
            "dfCAR": round(dfCAR, 2) if dfCAR is not None else "",
            "n_active": len(active),
            "n_total": len(episodes)
        })

out = "<DATA_ROOT>/threshold_sensitivity_v2.csv"
with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["label", "threshold_mm", "dfCAR", "n_active", "n_total"])
    writer.writeheader()
    writer.writerows(results)

print(f"\nWritten to {out}")
