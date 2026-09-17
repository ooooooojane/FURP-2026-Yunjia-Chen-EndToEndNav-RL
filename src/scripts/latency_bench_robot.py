#!/usr/bin/env python3
"""板载三模型推理延迟对比 v2(真机用, GPT审阅修正版)。

修正点(GPT 2026-08-18审阅):
  1. TorchScript 路径加 torch.inference_mode() —— 与真实节点 no_grad 一致(原版缺省,
     含 autograd 开销, 41× 被夸大);
  2. 保存全部 raw samples(每模型每批 1000 次);
  3. 每模型独立运行 5 批, 报告每批统计与合并分布(p99 离群可定位);
  4. 输入使用真实导航状态(calib_states.npy 前 N 个), 三模型同一输入集。

用法(机器人上, 模型文件在当前目录):
    python3 latency_bench_robot.py [批数, 默认5]
输出: 控制台 + latency_robot_raw_<日期>.json(全部样本)
"""
import json
import os
import statistics
import sys
import time
from datetime import datetime

import numpy as np

N = 1000
WARMUP = 50

MODELS = [
    ("torchscript_fp32", "cnntd3_actor.pt", "torch"),
    ("ort_fp32", "CNNTD3_ir9.onnx", "ort"),
    ("ort_int8", "CNNTD3_int8_ir9.onnx", "ort"),
]


def load_inputs(n_inputs=200):
    """真实导航状态作为输入(与校准数据同源, 三模型同一输入集)"""
    p = "calib_states.npy"
    if os.path.exists(p):
        d = np.load(p).astype(np.float32)
        return d[:n_inputs]
    return np.random.randn(n_inputs, 185).astype(np.float32)


def bench_torch(path, inputs):
    import torch
    m = torch.jit.load(path).eval()
    x = torch.from_numpy(inputs[0].reshape(1, -1))
    with torch.inference_mode():
        for _ in range(WARMUP):
            m(x)
    ts = []
    with torch.inference_mode():
        for inp in inputs:
            xi = torch.from_numpy(inp.reshape(1, -1))
            t0 = time.perf_counter()
            m(xi)
            ts.append((time.perf_counter() - t0) * 1000)
    return ts


def bench_ort(path, inputs):
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    x = inputs[0].reshape(1, -1)
    for _ in range(WARMUP):
        sess.run(None, {iname: x})
    ts = []
    for inp in inputs:
        xi = inp.reshape(1, -1)
        t0 = time.perf_counter()
        sess.run(None, {iname: xi})
        ts.append((time.perf_counter() - t0) * 1000)
    return ts


def stats(ts):
    s = sorted(ts)
    n = len(s)
    return {"mean_ms": round(statistics.mean(s), 4),
            "p50_ms": round(s[n // 2], 4),
            "p95_ms": round(s[int(n * 0.95)], 4),
            "p99_ms": round(s[int(n * 0.99)], 4),
            "min_ms": round(s[0], 4),
            "max_ms": round(s[-1], 4)}


if __name__ == "__main__":
    n_batches = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    inputs = load_inputs(N)   # 200 真实状态循环使用
    out = {"n_per_batch": N, "warmup": WARMUP, "n_batches": n_batches,
           "input_source": "calib_states.npy(真实导航状态) 或随机", "models": {}}

    print(f"{'版本':16s} {'尺寸KB':>7s} {'批均mean':>9s} {'合并p50':>8s} {'合并p95':>8s} {'合并p99':>8s}")
    for name, path, kind in MODELS:
        if not os.path.exists(path):
            print(f"{name}: 缺文件 {path}, 跳过")
            continue
        fn = bench_torch if kind == "torch" else bench_ort
        batches, batch_stats = [], []
        for b in range(n_batches):
            t = fn(path, inputs)
            batches.append(t)
            batch_stats.append(stats(t))
            print(f"  {name} 批{b+1}: mean={stats(t)['mean_ms']} p99={stats(t)['p99_ms']} max={stats(t)['max_ms']}", flush=True)
        merged = [x for b in batches for x in b]
        out["models"][name] = {
            "batch_stats": batch_stats,
            "merged": stats(merged),
            "size_kb": os.path.getsize(path) // 1024,
            "raw": merged,
        }
        m = out["models"][name]
        print(f"{name:16s} {m['size_kb']:7d} {statistics.mean([s['mean_ms'] for s in batch_stats]):9.3f} "
              f"{m['merged']['p50_ms']:8.3f} {m['merged']['p95_ms']:8.3f} {m['merged']['p99_ms']:8.3f}", flush=True)

    fname = f"latency_robot_raw_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(fname, "w") as f:
        json.dump(out, f)
    print(f"[*] raw 已保存: {fname}")
