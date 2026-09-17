#!/usr/bin/env python3
"""Generate IEEE-ready latency and compression figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, LogFormatterMathtext, NullFormatter
import numpy as np


ROOT = Path("/Users/janechan/Desktop/Code")
OUT = ROOT / "figures"
RAW = ROOT / "scripts/results/table2_repeat_batches.npz"

COLORS = {
    "CNNTD3": "#2a78d6",
    "SAC": "#eb6834",
    "PPO": "#1baf7a",
    "baseline": "#0b0b0b",
    "pruned10": "#1c5cab",
    "pruned30": "#4c87cb",
    "pruned50": "#86b6ef",
    "ft50": "#1c5cab",
    "ort": "#eb6834",
    "int8": "#e34948",
}

LATENCY_AUTH = {
    "cnntd3_cpu": (0.1204, 0.1188, 0.1306, 0.1511),
    "sac_cpu": (0.0395, 0.0392, 0.0411, 0.0445),
    "ppo_cpu": (0.0295, 0.0293, 0.0302, 0.0316),
    "cnntd3_mps": (0.6962, 0.6398, 0.9442, 1.3587),
    "sac_mps": (0.4477, 0.4005, 0.6837, 1.1528),
    "ppo_mps": (0.5133, 0.3803, 0.9297, 3.0550),
    "cnntd3_cuda": (0.3073, 0.3200, 0.3352, 0.3456),
    "sac_cuda": (0.1462, 0.1506, 0.1603, 0.1664),
    "ppo_cuda": (0.1068, 0.1110, 0.1190, 0.1249),
}

COMP = [
    ("Baseline", "baseline", 0.538, 0.1204, 72.5, "o"),
    ("Pruned 10%", "pruned10", 0.444, 0.1178, 72.5, "o"),
    ("Pruned 30%", "pruned30", 0.286, 0.1173, 76.0, "o"),
    ("Pruned 50%", "pruned50", 0.164, 0.1221, 61.0, "o"),
    ("Pruned 50% + FT", "ft50", 0.164, 0.1177, 70.0, "o"),
    ("ORT FP32", "ort", 0.520, 0.0191, 72.0, "s"),
    ("INT8", "int8", 0.157, 0.0282, 72.0, "D"),
]

SHORT_LABEL = {
    "baseline": "Base",
    "pruned10": "P10",
    "pruned30": "P30",
    "pruned50": "P50",
    "ft50": "P50+FT",
    "ort": "ORT",
    "int8": "INT8",
}


def choose_font() -> tuple[str, str]:
    for candidate in ("Times New Roman", "TeX Gyre Termes", "Times"):
        try:
            path = font_manager.findfont(candidate, fallback_to_default=False)
            return candidate, path
        except ValueError:
            continue
    raise RuntimeError(
        "Neither Times New Roman nor TeX Gyre Termes/Times is installed; "
        "refusing to silently fall back to DejaVu Sans."
    )


def setup_style() -> tuple[str, str]:
    font_name, font_path = choose_font()
    mpl.rcParams.update({
        "font.family": font_name,
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.55,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    return font_name, font_path


def clean_axis(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#6f6f6f")
    ax.spines["bottom"].set_color("#6f6f6f")
    ax.grid(axis=grid_axis, which="major", color="#e1e0d9", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#333333")


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.02)


def ecdf(values):
    x = np.sort(np.asarray(values).reshape(-1))
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def make_latency(data):
    panels = [
        ("cpu", "(a) M5 CPU"),
        ("mps", "(b) M5 GPU (MPS)"),
        ("cuda", "(c) RTX 4070 (CUDA)"),
    ]
    names = [("cnntd3", "CNNTD3"), ("sac", "SAC"), ("ppo", "PPO")]
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 1.88), sharey=True)
    for i, (platform, title) in enumerate(panels):
        ax = axs[i]
        all_values = []
        for key_name, display in names:
            values = data[f"{key_name}_{platform}"].reshape(-1)
            all_values.append(values)
            x, y = ecdf(values)
            ax.step(x, y, where="post", lw=1.35, color=COLORS[display], label=display)
        concat = np.concatenate(all_values)
        positive = concat[concat > 0]
        lo = np.percentile(positive, 0.05) / 1.15
        hi = np.percentile(positive, 99.95) * 1.15
        ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, 1.01)
        ax.set_yticks([0, 0.5, 0.9, 1.0])
        tick_map = {
            "cpu": [0.03, 0.05, 0.10],
            "mps": [0.20, 0.50, 1.00, 2.00],
            "cuda": [0.10, 0.20, 0.30],
        }
        ax.xaxis.set_major_locator(FixedLocator(tick_map[platform]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(title, pad=3)
        ax.set_xlabel("Inference latency (ms)", labelpad=2)
        clean_axis(ax, "y")

    axs[0].set_ylabel("Empirical CDF", labelpad=2)
    axs[0].legend(loc="lower right", frameon=True, facecolor="white",
                  edgecolor="#bdbdbd", framealpha=0.95, borderpad=0.3,
                  handlelength=1.5, labelspacing=0.25)
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.25, top=0.87, wspace=0.20)
    save(fig, "fig1_latency_ecdf")
    plt.close(fig)


LEFT_OFFSETS = {
    "baseline": (-31, 13),
    "pruned10": (-8, 13),
    "pruned30": (-8, -14),
    "pruned50": (7, 0),
    "ft50": (7, -7),
    "ort": (-18, -17),
    "int8": (7, 11),
}
RIGHT_OFFSETS = {
    "baseline": (-28, 13),
    "pruned10": (-7, -15),
    "pruned30": (-7, 12),
    "pruned50": (7, -12),
    "ft50": (7, 11),
    "ort": (-25, 11),
    "int8": (7, 8),
}


def draw_compression(ax_success, ax_latency):
    by_key = {key: (size, latency, success) for _, key, size, latency, success, _ in COMP}
    for a, b in (("baseline", "pruned50"), ("pruned50", "ft50"), ("ort", "int8")):
        xa, la, sa = by_key[a]
        xb, lb, sb = by_key[b]
        for ax, ya, yb in ((ax_success, sa, sb), (ax_latency, la, lb)):
            ax.annotate("", xy=(xb, yb), xytext=(xa, ya),
                        arrowprops=dict(arrowstyle="->", color="#c0c0c0", lw=0.55,
                                        shrinkA=4, shrinkB=5, mutation_scale=6), zorder=1)

    for name, key, size, latency, success, marker in COMP:
        hollow = key == "ft50"
        face = "white" if hollow else COLORS[key]
        edge = COLORS[key]
        for ax, y, offsets in ((ax_success, success, LEFT_OFFSETS),
                               (ax_latency, latency, RIGHT_OFFSETS)):
            ax.scatter(size, y, s=28, marker=marker, facecolor=face, edgecolor=edge,
                       linewidth=1.0, zorder=3)
            dx, dy = offsets[key]
            ax.annotate(SHORT_LABEL[key], (size, y), xytext=(dx, dy), textcoords="offset points",
                        fontsize=6.8, ha="left", va="center", color="#202020", zorder=4)

    ax_success.set_xlim(0.125, 0.575)
    ax_success.set_ylim(58, 79.5)
    ax_success.set_xlabel("Model size (MB)", labelpad=2)
    ax_success.set_ylabel("Success rate (%)", labelpad=2)
    ax_success.set_title("(a) Navigation performance", pad=3)
    ax_success.set_yticks([60, 65, 70, 75])
    clean_axis(ax_success, "y")

    ax_latency.set_xlim(0.125, 0.575)
    ax_latency.set_yscale("log")
    ax_latency.set_ylim(0.015, 0.18)
    ax_latency.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
    ax_latency.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax_latency.set_xlabel("Model size (MB)", labelpad=2)
    ax_latency.set_ylabel("Mean latency (ms)", labelpad=2)
    ax_latency.set_title("(b) Runtime cost", pad=3)
    clean_axis(ax_latency, "y")


def make_compression_wide():
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 1.72))
    draw_compression(axs[0], axs[1])
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.28, top=0.84, wspace=0.28)
    save(fig, "fig2_compression")
    plt.close(fig)


def make_compression_single():
    fig, axs = plt.subplots(2, 1, figsize=(3.5, 3.2))
    draw_compression(axs[0], axs[1])
    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.10, top=0.95, hspace=0.62)
    save(fig, "fig2_compression_singlecol")
    plt.close(fig)


def write_notes(font_name, font_path, data):
    lines = [
        "# Figure 1 notes",
        "",
        "- Figure: `fig1_latency_ecdf.pdf/.png` (7.16 × 1.88 in).",
        "- Script: `scripts/make_figures/make_paper_figures.py`.",
        "- Raw source: `scripts/results/table2_repeat_batches.npz`; 5 batches × 1000 samples per policy/platform.",
        "- ECDF curves use all raw samples. NeuPAN is intentionally omitted because raw timing samples are unavailable.",
        f"- Font: `{font_name}` from `{font_path}`; PDF uses TrueType/fonttype 42.",
        "- Authoritative labels use the paper's frozen summary values. Direct NumPy percentiles can differ in the last reported digits because the frozen export used a different percentile/rounding path.",
        "",
        "## Consistency check (raw calculation; not substituted into the paper)",
        "",
        "| key | raw mean | raw p50 | raw p95 | raw p99 | authoritative mean/p50/p95/p99 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for key, auth in LATENCY_AUTH.items():
        values = data[key].reshape(-1)
        p50, p95, p99 = np.percentile(values, [50, 95, 99])
        lines.append(
            f"| {key} | {values.mean():.6f} | {p50:.6f} | {p95:.6f} | {p99:.6f} | "
            f"{auth[0]:.4f}/{auth[1]:.4f}/{auth[2]:.4f}/{auth[3]:.4f} |"
        )
    lines += [
        "",
        "## Suggested caption",
        "",
        "Empirical cumulative distributions of model-only inference latency over 5,000 measurements per policy and platform. CPU execution exhibited the most concentrated distributions, whereas MPS showed more pronounced upper tails. All tested DRL configurations remained below the millisecond scale for most measurements.",
        "",
    ]
    (OUT / "fig1_notes.md").write_text("\n".join(lines), encoding="utf-8")

    comp_lines = [
        "# Figure 2 notes",
        "",
        "- Figures: `fig2_compression.pdf/.png` (wide, 7.16 × 1.72 in) and `fig2_compression_singlecol.pdf/.png` (single-column, 3.5 × 3.2 in).",
        "- Script: `scripts/make_figures/make_paper_figures.py`.",
        "- Sources: frozen Table 4 values; latency from `table4_latency_unified.json`, size from `compression_report.json` size fields only, success from 200 paired scenarios.",
        f"- Font: `{font_name}` from `{font_path}`; PDF uses TrueType/fonttype 42.",
        "- Arrows indicate processing/translation paths only, not independent causal identification.",
        "- Direct-label abbreviations: Base=baseline, P10/P30/P50=10%/30%/50% pruning, P50+FT=50% pruning followed by fine-tuning, ORT=ONNX Runtime FP32.",
        "- No confidence intervals were invented. Paired McNemar conclusions belong in the caption/text.",
        "",
        "## Values plotted",
        "",
        "| Version | Size (MB) | Mean latency (ms) | Success (%) |",
        "|---|---:|---:|---:|",
    ]
    for name, _, size, latency, success, _ in COMP:
        comp_lines.append(f"| {name} | {size:.3f} | {latency:.4f} | {success:.1f} |")
    comp_lines += [
        "",
        "## Suggested caption",
        "",
        "INT8 showed no detected success-rate degradation in 200 paired scenarios, while reducing model size by 71%. Fifty-percent pruning significantly reduced success, which was significantly recovered by fine-tuning; equivalence to the baseline was not formally established.",
        "",
    ]
    (OUT / "fig2_notes.md").write_text("\n".join(comp_lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    font_name, font_path = setup_style()
    data = np.load(RAW)
    missing = sorted(set(LATENCY_AUTH) - set(data.files))
    if missing:
        raise RuntimeError(f"Missing latency arrays: {missing}")
    make_latency(data)
    make_compression_wide()
    make_compression_single()
    write_notes(font_name, font_path, data)
    manifest = {
        "font": {"name": font_name, "path": font_path},
        "outputs": sorted(p.name for p in OUT.glob("fig*.pdf")),
        "raw_latency": str(RAW),
    }
    (OUT / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
