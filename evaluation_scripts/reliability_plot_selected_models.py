"""Reliability diagrams for o3 and Grok 4.1 Fast (NR)."""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

SCORED_DIR = "../mcqa_evaluation/output/results/scored"
OUTPUT_PDF = "reliability_diagrams_selected.pdf"
OUTPUT_PNG = "reliability_diagrams_selected.png"
CW_THRESHOLD = 0.8
N_BINS = 10

SELECTED_MODELS = [
    ("o3-2025-04-16", "tempDefault"),                # underconfident
    ("grok-4-1-fast-non-reasoning", "temp0_0"),      # overconfident
]
MODEL_NAMES = {
    "o3-2025-04-16": "o3",
    "grok-4-1-fast-non-reasoning": "Grok 4.1 Fast (NR)",
}
COLORS = {
    "o3-2025-04-16": "#283593",
    "grok-4-1-fast-non-reasoning": "#E63946",
}
BLACK = "#2C2C2A"
GRAY = "#888780"


def compute_ece(pairs):
    if not pairs:
        return None
    bins = [[] for _ in range(N_BINS)]
    for conf, corr in pairs:
        bins[min(int(conf * N_BINS), N_BINS - 1)].append((conf, corr))
    ece = 0.0
    for b in bins:
        if b:
            bin_acc = sum(c for _, c in b) / len(b)
            bin_conf = sum(c for c, _ in b) / len(b)
            ece += (len(b) / len(pairs)) * abs(bin_acc - bin_conf)
    return round(ece, 4)


def compute_cw_rate(records, threshold=0.8):
    wrong = [r for r in records if r.get("correct") == False and r.get("confidence") is not None]
    if not wrong:
        return None
    return round(sum(1 for r in wrong if r["confidence"] >= threshold) / len(wrong), 4)


def load_scored_raw(model_key, temp_label):
    path = os.path.join(SCORED_DIR, f"scored_raw_{model_key}_{temp_label}.json")
    if not os.path.exists(path):
        print(f"file not found: {path}")
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def reliability_diagram(ax, records, model_name, color):
    valid = [r for r in records if r.get("confidence") is not None and r.get("correct") is not None]
    if not valid:
        ax.text(0.5, 0.5, "No confidence data", ha="center", va="center", fontsize=8,
                color=GRAY, transform=ax.transAxes)
        ax.set_title(model_name, fontsize=10, color=BLACK, pad=3)
        return

    bin_edges = np.linspace(0, 1, N_BINS + 1)
    bin_acc, bin_conf, bin_left = [], [], []
    for i in range(N_BINS):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = [r for r in valid if lo <= r["confidence"] < hi]
        if in_bin:
            bin_acc.append(np.mean([int(r["correct"]) for r in in_bin]))
            bin_conf.append(np.mean([r["confidence"] for r in in_bin]))
            bin_left.append(lo)

    ax.bar(bin_left, bin_acc, width=0.1, facecolor="none", edgecolor="#AAAAAA",
           linewidth=0.6, align="edge", zorder=1)
    for left, bacc, bconf in zip(bin_left, bin_acc, bin_conf):
        gap_color, gap_bottom = ("#E63946", bacc) if bconf > bacc else ("#2E8B57", bconf)
        ax.bar(left, abs(bconf - bacc), bottom=gap_bottom, width=0.1, alpha=0.30,
               color=gap_color, align="edge", zorder=2)

    ax.plot([0, 1], [0, 1], "--", color=GRAY, alpha=0.6, linewidth=0.9, zorder=3)
    if bin_conf and bin_acc:
        ax.plot(bin_conf, bin_acc, "o-", color=color, markersize=6, linewidth=1.8, zorder=4,
                markeredgecolor="white", markeredgewidth=0.5)

    ece = compute_ece([(r["confidence"], int(r["correct"])) for r in valid])
    cw_rate = compute_cw_rate(valid, CW_THRESHOLD)
    ece_str = f"ECE={ece:.3f}" if ece is not None else "ECE=N/A"
    cw_str = f"CW={cw_rate * 100:.1f}%" if cw_rate is not None else "CW=N/A"
    ax.text(0.04, 0.96, f"{ece_str}  {cw_str}", transform=ax.transAxes, fontsize=8,
            color=BLACK, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85,
                      edgecolor="#D3D1C7", linewidth=0.5))

    ax.set_xlim(max(0.0, min(r["confidence"] for r in valid) - 0.05), 1.02)
    ax.set_ylim(0, 1.3)
    ax.set_title(model_name, fontsize=9, color=BLACK, pad=3)
    ax.set_xlabel("Confidence", fontsize=9, color=BLACK)
    ax.set_ylabel("Accuracy", fontsize=9, color=BLACK)
    ax.tick_params(labelsize=9, colors=BLACK)
    for spine in ax.spines.values():
        spine.set_edgecolor("#D3D1C7")
        spine.set_linewidth(0.6)
    ax.grid(color="#E8E6DC", linewidth=0.3, linestyle="--", alpha=0.6)


def main():
    fig, axes = plt.subplots(2, 1, figsize=(2.8, 4.15))
    plt.subplots_adjust(hspace=0.45, left=0.22, right=0.97, top=0.93, bottom=0.10)

    for ax, (model_key, temp_label) in zip(axes, SELECTED_MODELS):
        temp_disp = (temp_label.replace("temp", "").replace("_", ".")
                     .replace("Default", "Default Temp.").replace("0.0", "Temp=0"))
        reliability_diagram(ax, load_scored_raw(model_key, temp_label),
                            f"{MODEL_NAMES.get(model_key, model_key)} ({temp_disp})",
                            COLORS.get(model_key, "#378ADD"))

    plt.savefig(OUTPUT_PDF, bbox_inches="tight", dpi=300)
    plt.savefig(OUTPUT_PNG, bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()
