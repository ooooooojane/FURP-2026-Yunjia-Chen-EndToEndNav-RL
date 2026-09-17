# Benchmarking Edge Deployment of DRL Navigation Policies

**Latency, Compression, and Real-Robot Validation**

> **FURP 2026** — Faculty Undergraduate Research Practice
> Undergraduate Research Group · Faculty of Science and Engineering
> University of Nottingham Ningbo China

## Deliverables

| | |
|---|---|
|  **Final report** | [`FURP_Summer_Report.pdf`](FURP_Summer_Report.pdf) |
|  **Poster** | [`FURP_Showcase.pdf`](FURP_Showcase.pdf) |
|  **Code** | [`src/`](src/) — map in [`src/README.md`](src/README.md) |
|  **Source data** | [`src/paper_data/`](src/paper_data/) — which file backs which table |
|  **Upstream patch** | [`src/patches/`](src/patches/) — what we changed, as a diff |
|  **Weekly log** | [`docs/00_weekly.md`](docs/00_weekly.md) |

## Abstract

Autonomous mobile robot navigation maps sensor observations to motion commands
within an on-board control loop. A deployable policy must be timely, compact and
competent, yet these properties are often evaluated separately. This report
defines a benchmark that combines model-only latency and artifact size, paired
closed-loop task outcomes, and execution in a physical sensing and control
stack, where each source of evidence has a stated limit on the claims it
supports. We apply it to three DRL policies and one optimization-based planner
using CPU and GPU execution, 999 paired simulation scenarios, 200 paired
compression trials, and 50 formal trials on a WHEELTEC S100.

Model-only latency for CNNTD3, SAC and PPO ranges from 0.03 to 0.70 ms, but
**latency ordering does not track navigation success** among the tested
policies. ONNX Runtime provides a 6.3-fold speedup, while INT8 reduces size by
71% without a statistically significant task-level degradation. Fine-tuning
recovers most of the performance lost at 50% pruning. On the robot, model-only
inference is small relative to the 87–89 ms sensor-to-command path.

**Two contributions.** First, a benchmark for edge navigation with fixed
model-only timing, paired simulation, and physical-robot protocols, keeping
latency, size and task outcomes as separate metrics. Second, its application —
which shows which deployment changes affect latency or storage, and which leave
task behavior preserved.

## Project info

| Field | Entry |
|---|---|
| Student name | Yunjia Chen |
| Student ID | 20808871 |
| Project title | End-to-End Navigation for an AMR with Reinforcement Learning |
| Project tag | EndToEndNav-RL |
| Track | Research |
| Supervising faculty | Tianxiang Cui |
| Project lead | Fuhua Jia |
| Team or individual | Individual |
| Upstream work extended | [`reiniscimurs/DRL-robot-navigation-IR-SIM`](https://github.com/reiniscimurs/DRL-robot-navigation-IR-SIM) — supplies the CNNTD3 / SAC / PPO implementations benchmarked here; our modifications are in [`src/patches/`](src/patches/) |
| SOTA baseline added | NeuPAN (Han et al., *IEEE T-RO*, 2025) — optimization-based planner, evaluated under the identical protocol |

## Repository structure

```
/docs
 ├── 00_weekly.md          ← weekly log index
 ├── weekly_progress/      ← individual weekly entries (W1–W6)
 ├── meeting_notes/        ← key takeaways from team meetings
 └── checklists/           ← programme checkpoints
/src
 ├── README.md             ← code map: which script produces what
 ├── scripts/              ← self-contained working directory
 ├── paper_data/           ← source data for every number in the report
 └── patches/              ← modifications to the upstream project
FURP_Showcase.pdf          ← poster (repository root, as required)
FURP_Summer_Report.pdf     ← final report
```

## Reproducing the results

Training and simulation code is the upstream project plus
[`src/patches/upstream_modifications.patch`](src/patches/), applied against
commit `31e1a4d`. That patch is what makes the work possible: it adds
**deterministic seeding across all three RNGs**, so obstacle layouts are
reproducible across processes and every policy can be evaluated on byte-identical
scenarios — the precondition for the paired McNemar tests in the report.

```bash
git clone https://github.com/reiniscimurs/DRL-robot-navigation-IR-SIM.git
cd DRL-robot-navigation-IR-SIM
git checkout 31e1a4d511bb607e6ea38f4f8fccc842fbc7dd77
git apply /path/to/upstream_modifications.patch
```

Then see [`src/README.md`](src/README.md) for the per-script detail and
[`src/paper_data/README.md`](src/paper_data/README.md) for the data.

**Not included:** raw ROS `.bag` recordings (456 MB), trained checkpoints
(65 MB), and the withdrawn `eval_compare.py`. Reasons are documented in
[`src/README.md`](src/README.md#deliberately-not-included).

## FURP programme

**The three rules for the certificate** — all three must be satisfied:

1. **Attend > 50%** of programme activities.
2. **Submit a poster** — placed as `FURP_Showcase.pdf` in this repo root.
3. **Present at the Poster Showcase.**

Research Track minimum: successful replication of a cited work with at least
**10% innovation** — reproduce the work *and* add something new.

Any leave of absence or withdrawal must be notified by email; a verbal or chat
message is not sufficient.

---

*Bridging the gap between classroom knowledge and cutting-edge research.*
