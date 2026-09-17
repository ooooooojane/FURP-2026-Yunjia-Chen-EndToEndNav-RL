#!/usr/bin/env python3
"""表 2 跨批稳定性:每个(模型, 设备)跑 5 个独立批次,每批 1000 次,保存 raw timings。

回应 GPT 复检意见 1.5:单批 1000 次只能描述批内形态,不能说明跨运行稳定性。
输出:results/table2_repeat_batches.npz(raw) + 打印跨批 mean/std/min-max。
"""
import json
import statistics

import numpy as np
import torch

N = 1000
WARMUP = 50
N_BATCHES = 5

MODELS = {
    "cnntd3": "exported/CNNTD3_actor.pt",
    "sac": "exported/SAC_actor.pt",
    "ppo": "exported/PPO_actor.pt",
}


def one_batch(model, device, x):
    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return times


import time

if __name__ == "__main__":
    import sys
    devices = sys.argv[1:] if len(sys.argv) > 1 else ["cpu", "mps"]
    torch.manual_seed(7)
    raw = {}
    summary = {}
    for dev in devices:
        for name, path in MODELS.items():
            model = torch.jit.load(path).eval()
            x = torch.randn(1, 185, device=dev)
            model = model.to(dev)
            # 预热
            for _ in range(WARMUP):
                model(x)
            if dev == "mps":
                torch.mps.synchronize()
            batches = []
            means = []
            for b in range(N_BATCHES):
                t = one_batch(model, dev, x)
                batches.append(t)
                means.append(statistics.mean(t))
                print(f"{name} {dev} batch{b+1}: mean {means[-1]:.4f} ms", flush=True)
            raw[f"{name}_{dev}"] = np.array(batches)
            summary[f"{name}_{dev}"] = {
                "batch_means": [round(m, 4) for m in means],
                "cross_batch_mean": round(statistics.mean(means), 4),
                "cross_batch_std": round(statistics.stdev(means), 4),
                "min_batch_mean": round(min(means), 4),
                "max_batch_mean": round(max(means), 4),
            }
            print(f"  → 跨批: {summary[f'{name}_{dev}']}", flush=True)
    np.savez_compressed("results/table2_repeat_batches.npz", **raw)
    with open("results/table2_repeat_batches.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("[*] 已保存: results/table2_repeat_batches.npz + .json")
