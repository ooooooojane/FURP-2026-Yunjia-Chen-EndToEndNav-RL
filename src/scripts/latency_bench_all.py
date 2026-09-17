#!/usr/bin/env python3
"""表 2 整批复测:三个 DRL 策略(CNNTD3/SAC/PPO, TorchScript)在指定设备上同批测量。

协议与 latency_bench.py 一致(预热 50 / 测 1000 / mean+p50+p95+p99),
固定随机种子保证可复现。M5 CPU 与 M5 GPU(MPS)在 Mac 上跑,RTX 4070(CUDA)在主机上跑。

用法(Code/scripts/ 下,对应环境):
    python latency_bench_all.py --device cpu    # Mac M5 CPU
    python latency_bench_all.py --device mps    # Mac M5 GPU
    python latency_bench_all.py --device cuda   # 主机 RTX 4070
"""
import argparse
import json
import statistics
import time

import torch

N = 1000
WARMUP = 50

MODELS = {
    "cnntd3": "exported/CNNTD3_actor.pt",
    "sac": "exported/SAC_actor.pt",
    "ppo": "exported/PPO_actor.pt",
}


def bench(model, device, n=N, warmup=WARMUP):
    x = torch.randn(1, 185, device=device)
    model = model.to(device)
    model.eval()
    for _ in range(warmup):
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    s = sorted(times)
    return {
        "mean_ms": round(statistics.mean(s), 4),
        "p50_ms": round(s[n // 2], 4),
        "p95_ms": round(s[int(n * 0.95)], 4),
        "p99_ms": round(s[int(n * 0.99)], 4),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--out", default="results/table2_latency_fresh.json")
    args = ap.parse_args()

    torch.manual_seed(7)
    results = {"n": N, "warmup": WARMUP, "device": args.device, "models": {}}
    for name, path in MODELS.items():
        m = torch.jit.load(path)
        results["models"][name] = bench(m, args.device)
        print(f"{name} [{args.device}]: {json.dumps(results['models'][name])}")
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[*] 已保存: {args.out}")
