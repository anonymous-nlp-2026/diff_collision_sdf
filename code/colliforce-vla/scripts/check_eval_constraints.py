#!/usr/bin/env python3
"""Check eval results against hard constraints and compare with LoRA baseline."""
import json
import sys
import numpy as np

LORA_BASELINE = {
    "s0": {"TSR": 51.5, "CAR": 56.5},
    "s1": {"TSR": 51.5, "CAR": 52.5},
    "mean": {"TSR": 51.5, "CAR": 54.5},
}

FROZEN_EE_THRESHOLD = 0.005

def analyze(path):
    with open(path) as f:
        data = json.load(f)

    results = data["results"]
    config = data.get("config", {})

    print(f"{'='*60}")
    print(f"File: {path}")
    print(f"Config: suite={config.get('suite')}, model_type={config.get('model_type')}, seed={config.get('seed')}")
    print(f"Checkpoint: {config.get('checkpoint', 'N/A')}")
    print(f"{'='*60}")
    print()

    print(f"{'Task':<8} {'TSR':>6} {'CAR':>6} {'Frozen':>8} {'AvgEEDisp':>10}")
    print(f"{'-'*8} {'-'*6} {'-'*6} {'-'*8} {'-'*10}")

    overall_tsr = []
    overall_car = []
    any_frozen = False

    for i, task in enumerate(results):
        tsr = task["TSR"] * 100
        car = task["CAR"] * 100
        n_ep = task["n_episodes"]
        episodes = task.get("episodes", [])

        frozen_count = sum(1 for ep in episodes if ep.get("max_ee_displacement", 1.0) < FROZEN_EE_THRESHOLD)
        avg_ee_disp = task.get("avg_max_ee_displacement", 0)
        frozen_pct = frozen_count / n_ep * 100 if n_ep > 0 else 0

        overall_tsr.append(task["TSR"])
        overall_car.append(task["CAR"])

        flag = " ***" if frozen_count > 0 else ""
        if frozen_count > 0:
            any_frozen = True
        print(f"task_{i:<4} {tsr:5.1f}% {car:5.1f}% {frozen_pct:6.0f}% {avg_ee_disp:10.4f}{flag}")

    avg_tsr = np.mean(overall_tsr) * 100
    avg_car = np.mean(overall_car) * 100

    print(f"{'-'*8} {'-'*6} {'-'*6} {'-'*8} {'-'*10}")
    print(f"{'Overall':<8} {avg_tsr:5.1f}% {avg_car:5.1f}%")
    print()

    if avg_tsr < 35:
        print("!!! WARNING: TSR < 35% — below viability threshold !!!")
    if any_frozen:
        print("!!! CRITICAL: Frozen EE episodes detected — check for degenerate policy !!!")

    print()
    print("--- LoRA-only baseline comparison ---")
    print(f"  LoRA mean: TSR={LORA_BASELINE['mean']['TSR']:.1f}%  CAR={LORA_BASELINE['mean']['CAR']:.1f}%")
    print(f"  This run:  TSR={avg_tsr:.1f}%  CAR={avg_car:.1f}%")
    delta_tsr = avg_tsr - LORA_BASELINE["mean"]["TSR"]
    delta_car = avg_car - LORA_BASELINE["mean"]["CAR"]
    print(f"  Delta:     TSR={delta_tsr:+.1f}pp  CAR={delta_car:+.1f}pp")
    print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <eval_result.json> [eval_result2.json ...]")
        print("Checks eval results against hard constraints and compares with LoRA baseline.")
        sys.exit(0)
    for path in sys.argv[1:]:
        analyze(path)
