#!/usr/bin/env python3
"""统一批次延迟测量:PyTorch FP32 / ORT FP32 / ORT INT8 一次跑完,同协议。

用途:表 4 压缩权衡的唯一出口(GPT 审阅 P0-4.5:跨脚本/跨批次不能做比值)。
协议与 latency_bench.py 完全一致:预热 50 次,正式 1000 次,报 mean/p50/p95/p99。
输入固定随机种子,保证可复现。

用法(在 Code/scripts/ 下,drl 环境):
    python latency_bench_unified.py
"""
import json
import statistics
import time

import numpy as np
import torch

N = 1000
WARMUP = 50

# 注意:剪枝部署文件为 CNNTD3_pruned*.pt(尺寸与 compression_report 吻合);
# CNNTD3_actor_pruned*.pt 是等大的未剪完整 actor,禁止用于评估。
MODELS = {
    "pytorch_fp32": "exported/CNNTD3_actor.pt",          # TorchScript 原始
    "pruned10": "exported/compressed/CNNTD3_pruned10.pt",
    "pruned30": "exported/compressed/CNNTD3_pruned30.pt",
    "pruned50": "exported/compressed/CNNTD3_pruned50.pt",
    "pruned50_ft": "exported/compressed/CNNTD3_pruned50_ft.pt",
    "ort_fp32": "exported/compressed/CNNTD3.onnx",       # ONNX FP32
    "ort_int8": "exported/compressed/CNNTD3_int8_realcalib.onnx",  # ONNX INT8 真实校准
}


def bench_torch(model, n=N, warmup=WARMUP):
    x = torch.randn(1, 185)
    model.eval()
    for _ in range(warmup):
        model(x)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t0) * 1000)
    return times


def bench_ort(sess, n=N, warmup=WARMUP):
    x = np.random.randn(1, 185).astype(np.float32)
    in_name = sess.get_inputs()[0].name
    for _ in range(warmup):
        sess.run(None, {in_name: x})
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        sess.run(None, {in_name: x})
        times.append((time.perf_counter() - t0) * 1000)
    return times


def stats(times):
    s = sorted(times)
    return {
        "mean_ms": round(statistics.mean(s), 4),
        "p50_ms": round(s[N // 2], 4),
        "p95_ms": round(s[int(N * 0.95)], 4),
        "p99_ms": round(s[int(N * 0.99)], 4),
    }


if __name__ == "__main__":
    import onnxruntime as ort

    np.random.seed(7)
    torch.manual_seed(7)

    results = {"n": N, "warmup": WARMUP, "input": "[1,185] random", "variants": {}}

    # PyTorch TorchScript 变体(原始 + 剪枝档)
    for name in ["pytorch_fp32", "pruned10", "pruned30", "pruned50", "pruned50_ft"]:
        m = torch.jit.load(MODELS[name])
        results["variants"][name] = stats(bench_torch(m))
        print(f"{name}: {results['variants'][name]}")

    # ORT 变体
    for name, is_int8 in [("ort_fp32", False), ("ort_int8", True)]:
        s = ort.InferenceSession(MODELS[name], providers=["CPUExecutionProvider"])
        results["variants"][name] = stats(bench_ort(s))
        print(f"{name}: {results['variants'][name]}")

    print(json.dumps(results, indent=2))
    with open("results/table4_latency_unified.json", "w") as f:
        json.dump(results, f, indent=2)
    print("[*] 已保存: results/table4_latency_unified.json")
