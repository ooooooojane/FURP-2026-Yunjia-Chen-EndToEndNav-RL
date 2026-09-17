# Source data for the report

Every number in *Benchmarking Edge Deployment of DRL Navigation Policies* traces
back to a file in this folder. Nothing here was retyped by hand — each table in
the report was generated from these files.

## Table 2 — cross-platform inference latency

Model-only latency, batch = 1, warm, device-resident input. 5 independent
batches × 1000 forward passes = 5000 samples per (model, platform) cell.
Values are `[mean, p50, p95, p99]` in ms.

| File | Contents |
|---|---|
| `table2_latency_crossbatch.json` | **Authoritative.** 9 cells: `{cnntd3,sac,ppo} × {cpu, mps, cuda}` |
| `table2_repeat_batches.npz` | Raw per-batch timings behind the above |
| `table2_repeat_batches.json` | Batch-level summary |

Key values: CNNTD3 CPU `0.1204` ms, RTX 4070 `0.3073` ms.
Synchronisation differs by platform — no sync on CPU, `torch.mps.synchronize()`
after each forward on MPS, `torch.cuda.synchronize()` on CUDA. This measures
*blocking model-only* latency; it excludes host→device transfer and the ROS path.

> **Caveat carried from the report:** MPS (M5 GPU) rows are unstable across
> batches — sample SD up to 0.17 ms, with occasional high-variance batches.
> They are reported as indicative. CPU is the basis for deployment advice.

## Table 3 — navigation success rate

999 paired scenarios, both seeds, all four methods on **identical** scenarios
(made possible by the deterministic seeding patch — see `../patches/`).
Merged column in the report: seed 42 (500 episodes) + seed 7 (499 episodes).

| File | Method | Episodes |
|---|---|---|
| `mp42_{cnntd3,sac,ppo,neupan}.json` | 4 methods | 500 (seed 42) |
| `mp7_{cnntd3,sac,ppo,neupan}.json` | 4 methods | 499 (seed 7) |

Merged: CNNTD3 **68.8%**, SAC **83.7%**, PPO **30.9%**, NeuPAN **2.5%**.
Adjacent-ranking differences are all significant by exact paired McNemar
(SAC vs CNNTD3 p≈2.4×10⁻²⁷; CNNTD3 vs PPO p≈1.6×10⁻⁸²; PPO vs NeuPAN p≈1.6×10⁻⁷⁶).

> **Provenance note:** an earlier protocol that ran a physical `reset_clear`
> probe *during* evaluation produced different numbers (CNNTD3 75.2% /
> SAC 89.6% / PPO 25.0% / NeuPAN 3.3%). Those were withdrawn — the probe
> changed each episode's starting pose. The files here are the corrected
> generation-stage-probe protocol. Do not mix the two sets.

## Table 4 — compression trade-off (CNNTD3)

200 deterministically paired scenarios per tier, same scenarios across all tiers.

| Column | Source file |
|---|---|
| Success rate | `mp_matrix_{baseline,fp32,int8,pruned10,pruned30,pruned50,ft50}.json` |
| Model size | `../scripts/exported/compressed/compression_report.json` |
| Latency | `table4_latency_unified.json`, raw in `table4_crossbatch.npz` |

Headline results: ONNX Runtime gives **6.3×** speedup (PyTorch 0.1204 ms →
ORT FP32 0.0191 ms). INT8 cuts size by **71%** (0.538 → 0.157 MB) with no
statistically significant task-level degradation (72.5% vs 72.0%, McNemar
p = 1.0). Pruning to 50% *does* hurt (72.5% → 61.0%, p≈2.9×10⁻⁴); fine-tuning
recovers most of it (→ 70.0%, p≈0.011).

> **Do not use `CNNTD3_actor_pruned*.pt`** for evaluation. Those are equal-size
> artifacts, not actual pruned models — `PROVENANCE_MANIFEST.json` records this
> and gives SHA-256s for the real deployment files.

## Table 5 / Table IV — real-robot trials (WHEELTEC S100)

On-board sensor-to-command latency, measured on the robot (Jetson Orin Super,
ROS1 Noetic). Columns are `callback_ms` (LiDAR callback → velocity command,
includes inference) and `sensor_to_cmd_ms` (full path).

| File | Deployment format | n | Mean `sensor_to_cmd` |
|---|---|---|---|
| `latency_original.csv` | TorchScript FP32 | 930 | **89.1 ms** ← primary figure |
| `latency_fp32.csv` | ONNX Runtime FP32 | 618 | 87.2 ms |
| `latency_int8.csv` | ONNX Runtime INT8 | 552 | 87.2 ms |
| `latency.csv` | combined link test | 488 | 92.1 ms (secondary) |
| `latency_robot_raw_19700101_0844.json` | model-only, 5×1000 | — | TS FP32 1.539 / ORT FP32 0.092 / ORT INT8 0.100 ms |

The `19700101` timestamp in the filename is not a mistake — the Jetson has no
RTC, so it boots at the Unix epoch.

> These five files previously existed **only** as loose files on one machine,
> with no backup. They back the report's primary real-robot claim, so they are
> committed here.

## Matched-geometry diagnostic

| File | Contents |
|---|---|
| `matched_double_VC_results.json` | Final converged run — 20 trials, trajectories / commands / shield activations / termination reason |
| `matched_double_VC_trajectory.png` | Companion figure |

Nine earlier `matched_scene_results_v*.json` files (35 MB) were an exploratory
sweep across shield modes and obstacle geometries and are superseded by this one;
they are intentionally not included.

> **Honest caveat recorded during the work:** in the VC scene the obstacle
> near-edge sits at ±0.25 m against a 0.174 m half-width, so it is a
> narrow-corridor scene with spontaneous left/right alternation — **not** a
> forced double-bypass.

## Provenance

`PROVENANCE_MANIFEST.json` is the authoritative registry: model SHA-256s, seeds,
script versions, and generation dates for every artifact above.

> **Known discrepancy:** no `eval_compare.py` on disk matches any of the three
> hashes recorded in the manifest. That script's outputs were withdrawn anyway
> (it produced the void 75.2/89.6/25.0/3.3 numbers), and the authoritative
> success-rate script is `eval_multi_policy.py`. `eval_compare.py` is therefore
> not published here. Flagged rather than silently dropped.
