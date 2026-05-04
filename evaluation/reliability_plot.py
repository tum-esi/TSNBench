import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


SCORED_DIR      = "../mcqa_evaluation/output/results/accuracy_new_new"
OUTPUT_PDF   = "reliability_diagrams.pdf"
OUTPUT_PNG   = "reliability_diagrams.png"
CW_THRESHOLD = 0.8
N_BINS       = 10

# Model display names
MODEL_NAMES = {
    "grok-4-1-fast-reasoning"    : "Grok 4.1 Fast",
    "grok-4-1-fast-non-reasoning": "Grok 4.1 Fast (NR)",
    "deepseek-chat"              : "DeepSeek-V3.2 (NT)",
    "gpt-4o-2024-08-06"          : "GPT-4o",
    "gpt-4o-mini-2024-07-18"     : "GPT-4o mini",
    "Llama-3.3-70B-Instruct"     : "Llama 3.3 70B",
    "mistral-medium-2508"        : "Mistral Medium 3.1",
    "mistral-large-2512"         : "Mistral Large 3",
    "claude-sonnet-4-5-20250929" : "Claude Sonnet 4.5",
    "o3-2025-04-16"              : "o3",
    "gpt-5-2025-08-07"           : "GPT-5",
    "deepseek-reasoner"          : "DeepSeek-V3.2 (T)",
    "gemini-2.5-flash"           : "Gemini 2.5 Flash",
    "Llama-3.2-1B"               : "Llama 3.2 1B",
    "Qwen3-8B"                   : "Qwen3 8B",
    "ministral-8b-2512"          : "Ministral 3 8B",
}

# Model order for grid layout
MODEL_ORDER = [
    "grok-4-1-fast-reasoning",
    "grok-4-1-fast-non-reasoning",
    "deepseek-chat",
    "gpt-4o-2024-08-06",
    "gpt-4o-mini-2024-07-18",
    "Llama-3.3-70B-Instruct",
    "mistral-medium-2508",
    "mistral-large-2512",
    "claude-sonnet-4-5-20250929",
    "o3-2025-04-16",
    "gpt-5-2025-08-07",
    "deepseek-reasoner",
    "gemini-2.5-flash",
    "Llama-3.2-1B",
    "Qwen3-8B",
    "ministral-8b-2512",
]

COLORS = [
    "#E63946", "#2E8B57", "#378ADD", "#BA7517",
    "#9C27B0", "#FF6F00", "#1D9E75", "#D85A30",
    "#C62828", "#283593", "#00838F", "#558B2F",
    "#4527A0", "#AD1457", "#00695C", "#F57F17",
]

BLACK = "#2C2C2A"
GRAY  = "#888780"


def compute_ece(pairs):
    if not pairs:
        return None
    bins = [[] for _ in range(N_BINS)]
    for conf, corr in pairs:
        idx = min(int(conf * N_BINS), N_BINS - 1)
        bins[idx].append((conf, corr))
    total = len(pairs)
    ece   = 0.0
    for b in bins:
        if not b:
            continue
        bin_acc  = sum(c for _, c in b) / len(b)
        bin_conf = sum(c for c, _ in b) / len(b)
        ece += (len(b) / total) * abs(bin_acc - bin_conf)
    return round(ece, 4)


def compute_cw_rate(records, threshold=0.8):
    wrong = [r for r in records
             if r.get("correct") == False and
             r.get("confidence") is not None]
    if not wrong:
        return None
    conf_wrong = [r for r in wrong if r["confidence"] >= threshold]
    return round(len(conf_wrong) / len(wrong), 4)


def load_scored_raw(scored_dir, model_key, temp_label):
    fname = f"scored_raw_{model_key}_{temp_label}.json"
    fpath = os.path.join(scored_dir, fname)
    if not os.path.exists(fpath):
        return []
    with open(fpath, encoding="utf-8") as f:
        return json.load(f)


def find_best_temp(scored_dir, model_key):
    """Find best temp label: prefer temp0_0 over temp0_7 over tempDefault."""
    priority = ["temp0_0", "tempDefault", "temp0_7"]
    for label in priority:
        fname = f"scored_raw_{model_key}_{label}.json"
        if os.path.exists(os.path.join(scored_dir, fname)):
            return label
    return None


def reliability_diagram(ax, records, model_name, color):
    valid = [r for r in records
             if r.get("confidence") is not None and
             r.get("correct") is not None]

    if not valid:
        ax.text(0.5, 0.5, "No confidence data",
                ha="center", va="center", fontsize=8,
                color=GRAY, transform=ax.transAxes)
        ax.set_title(model_name, fontsize=10, color=BLACK, pad=3)
        return

    bin_edges = np.linspace(0, 1, N_BINS + 1)
    bin_acc   = []
    bin_conf  = []
    bin_count = []
    bin_left  = []

    for i in range(N_BINS):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = [r for r in valid
                  if lo <= r["confidence"] < hi]
        if in_bin:
            bin_acc.append(np.mean([int(r["correct"]) for r in in_bin]))
            bin_conf.append(np.mean([r["confidence"]  for r in in_bin]))
            bin_count.append(len(in_bin))
            bin_left.append(lo)

    ax.bar(bin_left, bin_acc, width=0.1,
           facecolor="none", edgecolor="#AAAAAA",
           linewidth=0.6, align="edge", zorder=1)

    for left, bacc, bconf in zip(bin_left, bin_acc, bin_conf):
        if bconf > bacc:
            gap_color  = "#E63946"   # red — overconfident
            gap_bottom = bacc
        else:
            gap_color  = "#2E8B57"   # green — underconfident
            gap_bottom = bconf
        gap_height = abs(bconf - bacc)
        ax.bar(left, gap_height, bottom=gap_bottom,
               width=0.1, alpha=0.30, color=gap_color,
               align="edge", zorder=2)

    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], "--", color=GRAY,
            alpha=0.6, linewidth=0.9, zorder=3)

    if bin_conf and bin_acc:
        ax.plot(bin_conf, bin_acc, "o-", color=color,
                markersize=6, linewidth=1.8, zorder=4,
                markeredgecolor="white", markeredgewidth=0.5)

    pairs   = [(r["confidence"], int(r["correct"])) for r in valid]
    ece     = compute_ece(pairs)
    cw_rate = compute_cw_rate(valid, CW_THRESHOLD)

    ece_str = f"ECE={ece:.3f}" if ece     is not None else "ECE=N/A"
    cw_str  = f"CW={cw_rate*100:.1f}%"   if cw_rate is not None else "CW=N/A"

    ax.text(0.04, 0.96, f"{ece_str}  {cw_str}",
            transform=ax.transAxes, fontsize=8,
            color=BLACK, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.25",
                      facecolor="white", alpha=0.85,
                      edgecolor="#D3D1C7", linewidth=0.5))

    if valid:
        min_conf = min(r["confidence"] for r in valid)
        x_lo     = max(0.0, min_conf - 0.05)
        ax.set_xlim(x_lo, 1.02)
    else:
        ax.set_xlim(0, 1.02)

    ax.set_ylim(0, 1.3)
    ax.set_title(model_name, fontsize=9, color=BLACK, pad=3)
    ax.tick_params(labelsize=9, colors=BLACK)
    for spine in ax.spines.values():
        spine.set_edgecolor("#D3D1C7")
        spine.set_linewidth(0.6)
    ax.grid(color="#E8E6DC", linewidth=0.3, linestyle="--", alpha=0.6)


def main():
    available = {}
    for model_key in MODEL_ORDER:
        temp_label = find_best_temp(SCORED_DIR, model_key)
        if temp_label:
            available[model_key] = temp_label

    if os.path.isdir(SCORED_DIR):
        for fname in os.listdir(SCORED_DIR):
            if not fname.startswith("scored_raw_") or not fname.endswith(".json"):
                continue
            stem = fname.replace("scored_raw_", "").replace(".json", "")
            for label in ["temp0_0", "tempDefault", "temp0_7"]:
                if stem.endswith(f"_{label}"):
                    mk = stem[:-len(f"_{label}")]
                    if mk not in available:
                        available[mk] = label
                    break

    n_models = len(available)
    if n_models == 0:
        print(f"No scored_raw files found in {SCORED_DIR}")
        return

    print(f"Found {n_models} models with scored raw files")

    n_cols = 4
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 3.2, n_rows * 3.0)
    )
    plt.subplots_adjust(
        hspace=0.45, wspace=0.35,
        left=0.06, right=0.97,
        top=0.93, bottom=0.06
    )

    if n_rows == 1:
        axes = [axes] if n_cols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    ordered_keys = [k for k in MODEL_ORDER if k in available]
    extras       = [k for k in available  if k not in MODEL_ORDER]
    ordered_keys += extras

    for idx, model_key in enumerate(ordered_keys):
        temp_label = available[model_key]
        records    = load_scored_raw(SCORED_DIR, model_key, temp_label)
        display    = MODEL_NAMES.get(model_key, model_key)
        color      = COLORS[idx % len(COLORS)]
        temp_disp  = temp_label.replace("temp", "").replace("_", ".").replace("Default", "Default Temp.").replace("0.0", "Temp=0")

        ax = axes[idx]
        reliability_diagram(ax, records, f"{display} ({temp_disp})", color)

        if idx % n_cols == 0:
            ax.set_ylabel("Accuracy", fontsize=9, color=BLACK)
        # ax.set_xlabel("Confidence", fontsize=10, color=BLACK)
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel("Confidence", fontsize=9, color=BLACK)

    for idx in range(len(ordered_keys), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        "Gray Dashed diagonal = perfect calibration  |  "
        "Red shaded portion = overconfident  |  Green shaded portion = underconfident",
        fontsize=10, color=BLACK, y=0.98
    )

    plt.savefig(OUTPUT_PDF, bbox_inches="tight", dpi=300)
    plt.savefig(OUTPUT_PNG, bbox_inches="tight", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()