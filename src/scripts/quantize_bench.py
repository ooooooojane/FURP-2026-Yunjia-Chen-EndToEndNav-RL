#!/usr/bin/env python3
"""模型压缩实验(任务 #4 正确版): 结构化剪枝 + ONNX 静态量化 INT8

用法(Mac drl 环境):
  python quantize_bench.py

流程:
  1. 从新检查点加载 CNNTD3 Actor(与导出 .pt 同一权重)
  2. 基线: 尺寸 + CPU 延迟
  3. 结构化剪枝 10/30/50%: 按 L2 范数删整条神经元(MLP 部分),
     重新导出 .pt(尺寸/延迟真实变化)
  4. ONNX 静态量化 INT8: 导出 ONNX → 校准 → 量化 → ORT 推理延迟
输出: exported/compressed/*.pt + onnx, 以及压缩报告 JSON
"""
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from robot_nav.models.CNNTD3.CNNTD3 import CNNTD3  # noqa: E402

STATE_DIM, ACTION_DIM = 185, 2
OUT = ROOT / "exported" / "compressed"
OUT.mkdir(parents=True, exist_ok=True)

# torch.load 的 CPU 映射(检查点在 GPU 上训练的)
_orig_load = torch.load
torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "map_location": "cpu"})


def load_actor():
    """从新检查点加载 actor(和 exported/CNNTD3_actor.pt 同一权重)"""
    model = CNNTD3(state_dim=STATE_DIM, action_dim=ACTION_DIM, max_action=1,
                   device="cpu", save_every=0, load_model=True, model_name="CNNTD3",
                   load_directory=str(ROOT / "robot_nav/models/CNNTD3/checkpoint"))
    model.actor.eval()
    return model.actor


def bench_torch(model, n=1000):
    """测量 torch 模型 CPU 推理延迟(ms)"""
    x = torch.randn(1, STATE_DIM)
    with torch.no_grad():
        for _ in range(50):
            model(x)
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return round(statistics.mean(times), 4), round(times[int(n * 0.95)], 4)


def bench_ort(session, n=1000):
    """测量 onnxruntime CPU 推理延迟(ms)"""
    x = np.random.randn(1, STATE_DIM).astype(np.float32)
    for _ in range(50):
        session.run(None, {session.get_inputs()[0].name: x})
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        session.run(None, {session.get_inputs()[0].name: x})
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return round(statistics.mean(times), 4), round(times[int(n * 0.95)], 4)


def structured_prune(actor, ratio):
    """结构化剪枝: 按 L2 范数删除 MLP 的整条神经元(真实减少计算量)

    只剪 MLP 部分(layer_1/layer_2/layer_3), 保留卷积编码器:
      layer_1: 36 -> h1 (删输出神经元, 即 layer_1 的行 + layer_2 的列)
      layer_2: h1 -> h2 (删输出神经元, 即 layer_2 的行 + layer_3 的列)
    """
    h1 = max(round(400 * (1 - ratio)), 50)
    h2 = max(round(300 * (1 - ratio)), 50)

    w1, b1 = actor.layer_1.weight.detach(), actor.layer_1.bias.detach()
    w2, b2 = actor.layer_2.weight.detach(), actor.layer_2.bias.detach()
    w3, b3 = actor.layer_3.weight.detach(), actor.layer_3.bias.detach()

    # layer_1 的输出神经元重要性 = 其 L2 范数(行范数)
    imp1 = w1.norm(dim=1)
    keep1 = torch.argsort(imp1, descending=True)[:h1].sort().values
    # layer_2 的输出神经元重要性(行范数), 输入维度同步裁剪
    imp2 = w2.norm(dim=1)
    keep2 = torch.argsort(imp2, descending=True)[:h2].sort().values

    new_l1 = torch.nn.Linear(36, h1)
    new_l1.weight.data = w1[keep1]
    new_l1.bias.data = b1[keep1]
    new_l2 = torch.nn.Linear(h1, h2)
    new_l2.weight.data = w2[keep2][:, keep1]
    new_l2.bias.data = b2[keep2]
    new_l3 = torch.nn.Linear(h2, ACTION_DIM)
    new_l3.weight.data = w3[:, keep2]
    new_l3.bias.data = b3

    new_actor = torch.nn.Module()
    # 复用原卷积/嵌入层(参数共享, 保持一致性)
    new_actor.cnn1 = actor.cnn1
    new_actor.cnn2 = actor.cnn2
    new_actor.cnn3 = actor.cnn3
    new_actor.goal_embed = actor.goal_embed
    new_actor.action_embed = actor.action_embed
    new_actor.layer_1 = new_l1
    new_actor.layer_2 = new_l2
    new_actor.layer_3 = new_l3

    def forward(x):
        laser = x[:, :180].unsqueeze(1)
        feat = torch.nn.functional.leaky_relu(new_actor.cnn1(laser))
        feat = torch.nn.functional.leaky_relu(new_actor.cnn2(feat))
        feat = torch.nn.functional.leaky_relu(new_actor.cnn3(feat))
        feat = feat.flatten(1)
        goal = torch.nn.functional.leaky_relu(new_actor.goal_embed(x[:, 180:183]))
        act = torch.nn.functional.leaky_relu(new_actor.action_embed(x[:, 183:185]))
        h = torch.cat([feat, goal, act], dim=1)
        h = torch.nn.functional.leaky_relu(new_actor.layer_1(h))
        h = torch.nn.functional.leaky_relu(new_actor.layer_2(h))
        return torch.tanh(new_actor.layer_3(h))

    new_actor.forward = forward
    new_actor.eval()
    return new_actor


if __name__ == "__main__":
    print("== 加载新权重 actor ==")
    actor = load_actor()
    x = torch.randn(1, STATE_DIM)
    with torch.no_grad():
        base_out = actor(x)

    report = {}

    # 1) 基线
    mean, p95 = bench_torch(actor)
    base_pt = ROOT / "exported" / "CNNTD3_actor.pt"
    report["baseline"] = {"size_mb": round(base_pt.stat().st_size / 1024 / 1024, 3),
                          "mean_ms": mean, "p95_ms": p95}
    print("基线: %.3f MB, %.4f ms" % (report["baseline"]["size_mb"], mean))

    # 2) 结构化剪枝 10/30/50%
    for ratio in (0.1, 0.3, 0.5):
        pruned = structured_prune(actor, ratio)
        with torch.no_grad():
            out = pruned(x)
            diff = (out - base_out).abs().max().item()
        mean, p95 = bench_torch(pruned)
        tag = f"pruned{int(ratio * 100)}"
        traced = torch.jit.trace(pruned, x)
        fn = OUT / f"CNNTD3_{tag}.pt"
        torch.jit.save(traced, str(fn))
        report[tag] = {"size_mb": round(fn.stat().st_size / 1024 / 1024, 3),
                       "mean_ms": mean, "p95_ms": p95,
                       "max_out_diff": round(diff, 4)}
        print("剪枝 %d%%: %.3f MB, %.4f ms, 输出最大偏差 %.4f" % (
            int(ratio * 100), report[tag]["size_mb"], mean, diff))

    # 3) ONNX 静态量化 INT8
    try:
        import onnx
        import onnxruntime as ort
        from onnxruntime.quantization import (
            CalibrationDataReader, QuantFormat, QuantType, quantize_static)

        onnx_path = OUT / "CNNTD3.onnx"
        torch.onnx.export(actor, x, str(onnx_path),
                          input_names=["obs"], output_names=["action"],
                          opset_version=17)
        # 校准数据(随机状态; 机制验证用, 成功率影响由 C 步评估)
        class RandomReader(CalibrationDataReader):
            def __init__(self, n=200):
                self.data = [{"obs": np.random.randn(1, STATE_DIM).astype(np.float32)}
                             for _ in range(n)]
                self.i = 0
            def get_next(self):
                if self.i < len(self.data):
                    d = self.data[self.i]
                    self.i += 1
                    return d
                return None
        q_path = OUT / "CNNTD3_int8.onnx"
        quantize_static(str(onnx_path), str(q_path), RandomReader(),
                        quant_format=QuantFormat.QDQ, weight_type=QuantType.QInt8,
                        per_channel=True)
        sess = ort.InferenceSession(str(q_path), providers=["CPUExecutionProvider"])
        mean, p95 = bench_ort(sess)
        report["int8_onnx"] = {"size_mb": round(q_path.stat().st_size / 1024 / 1024, 3),
                               "mean_ms": mean, "p95_ms": p95}
        print("ONNX INT8: %.3f MB, %.4f ms" % (report["int8_onnx"]["size_mb"], mean))
    except Exception as e:
        report["int8_onnx"] = {"error": str(e)}
        print("ONNX INT8 失败:", e)

    with open(OUT / "compression_report.json", "w") as f:
        json.dump(report, f, indent=1)
    print("报告已保存:", OUT / "compression_report.json")
