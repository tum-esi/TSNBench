import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


CBS_SCORED_DIRS = [
    "../open_end_evaluation/output/CBS/score_new",
    # "../open_end_evaluation/output/CBS/one_switch_topo/score",
    # "../open_end_evaluation/output/CBS/medium_mesh_topo/score",

]

CQF_SCORED_DIRS = [
    "../open_end_evaluation/output/CQF/score_new",
    # "../open_end_evaluation/output/CQF/one_switch_topo/score",
    # "../open_end_evaluation/output/CQF/medium_mesh_topo/score",
]

MCQA_SUMMARY    = "../mcqa_evaluation/output/results/full_mcqa_summary.json"
MIN_TC_COVERAGE = 0.50
TOTAL_TCS       = 100

MODEL_ORDER = [
    ("Grok 4.1 Fast (R)",      "grok-4-1-fast-reasoning",       False),
    ("Grok 4.1 Fast (NR)",     "grok-4-1-fast-non-reasoning",   False),
    ("DeepSeek-V3.2 (NT)",     "deepseek-chat",                 False),
    ("GPT-4o",                 "gpt-4o-2024-08-06",             False),
    ("GPT-4o mini",            "gpt-4o-mini-2024-07-18",        False),
    ("Llama 3.3",              "Llama-3.3-70B-Instruct",        False),
    ("Mistral Medium 3.1",     "mistral-medium-2508",           False),
    ("Mistral Large 3",        "mistral-large-2512",            False),
    ("Claude Sonnet 4.5",      "claude-sonnet-4-5-20250929",    False),
    ("o3",                     "o3-2025-04-16",                 True),
    ("GPT-5",                  "gpt-5-2025-08-07",              True),
    ("DeepSeek-V3.2 (T)",      "deepseek-reasoner",             True),
    ("Gemini 2.5 Flash",       "gemini-2.5-flash",              False),
    ("Llama 3.2 1B",           "Llama-3.2-1B",                  False),
    ("Qwen3 8B",               "Qwen3-8B",                      False),
    ("Ministral 3 8B",         "ministral-8b-2512",             False),
]


def load_flow_errors_multi(scored_dirs: list, prompt_type: str) -> dict:
    """
    Aggregate flow_records_avg across multiple scored directories.
    Per-model errors are pooled across all directories.
    Returns: {model_key: {"errors": [...], "scored_tcs": int}}
    """
    model_data  = {}
    total_files = 0

    for scored_dir in scored_dirs:
        if not os.path.isdir(scored_dir):
            print(f"  SKIP (not found): {scored_dir}")
            continue

        files = sorted([
            f for f in os.listdir(scored_dir)
            if f.startswith("scored_openend_") and
               f"_{prompt_type}_" in f and
               f.endswith(".json")
        ])

        for fname in files:
            stem      = fname.replace("scored_openend_", "").replace(".json", "")
            parts     = stem.split(f"_{prompt_type}_")
            model_key = parts[0]

            with open(os.path.join(scored_dir, fname), encoding="utf-8") as f:
                data = json.load(f)

            enough_tcs = data.get("enough_tcs", True)
            scored_tcs = data.get("scored_tcs", 0)      # ← track TC count

            flow_errors = []
            if enough_tcs:
                for fr in data.get("flow_records_avg", []):
                    err = fr.get("abs_error")
                    if err is not None:
                        flow_errors.append(err)

            if model_key not in model_data:
                model_data[model_key] = {"errors": [], "scored_tcs": 0}

            model_data[model_key]["errors"].extend(flow_errors)
            model_data[model_key]["scored_tcs"] += scored_tcs   # ← accumulate

            total_files += 1

            print(f"  {prompt_type} | {scored_dir} | {model_key}: "
                  f"{len(flow_errors)} errors (total so far: "
                  f"{len(model_data[model_key]['errors'])})")

    print(f"  Total scored files loaded: {total_files} | "
          f"Models found: {len(model_data)}")
    return model_data


def load_mcqa_accuracy(summary_file: str) -> dict:
    if not os.path.exists(summary_file):
        print(f"  WARNING: MCQA summary not found: {summary_file}")
        return {}
    with open(summary_file, encoding="utf-8") as f:
        data = json.load(f)
    result = {}
    for s in data:
        key = s.get("model", "")
        acc = s.get("accuracy_overall")
        if key and acc is not None:
            result[key] = round(acc * 100, 1) if acc <= 1.0 else round(acc, 1)
    return result


print("Loading CBS flow errors (100 TCs)...")
cbs_errors = load_flow_errors_multi(CBS_SCORED_DIRS, "CBS")

print("\nLoading CQF flow errors (100 TCs)...")
cqf_errors = load_flow_errors_multi(CQF_SCORED_DIRS, "CQF")

print("\nLoading MCQA accuracy...")
mcqa_acc = load_mcqa_accuracy(MCQA_SUMMARY)

display_names = [m[0] for m in MODEL_ORDER]
model_keys    = [m[1] for m in MODEL_ORDER]
is_reasoning  = [m[2] for m in MODEL_ORDER]

mcqa_vals = []
cbs_data  = []
cqf_data  = []

for key in model_keys:
    mcqa_vals.append(mcqa_acc.get(key))

    cbs     = cbs_errors.get(key, {})
    cqf     = cqf_errors.get(key, {})
    cbs_tcs = cbs.get("scored_tcs", 0)
    cqf_tcs = cqf.get("scored_tcs", 0)

    if cbs_tcs >= MIN_TC_COVERAGE * TOTAL_TCS and cbs.get("errors"):
        cbs_data.append(cbs["errors"])
    else:
        cbs_data.append(None)
        print(f"  CBS {key}: {cbs_tcs}/{TOTAL_TCS} TCs → SUPPRESSED")

    if cqf_tcs >= MIN_TC_COVERAGE * TOTAL_TCS and cqf.get("errors"):
        cqf_data.append(cqf["errors"])
    else:
        cqf_data.append(None)
        print(f"  CQF {key}: {cqf_tcs}/{TOTAL_TCS} TCs → SUPPRESSED")


BLUE  = "#378ADD"
CORAL = "#E63946"
TEAL  = "#1B5E20"
RED   = "#E24B4A"
BLACK = "#2C2C2A"

y   = np.arange(len(display_names))
rng = np.random.default_rng(42)

fig2, (ax1b, ax2b) = plt.subplots(
    1, 2, figsize=(14, 6.5),
    gridspec_kw={"width_ratios": [0.8, 2.2]}
)
plt.subplots_adjust(wspace=0.06, left=0.13, right=0.98,
                    top=0.88, bottom=0.10)

# --- MCQA panel ---
ax1b.barh(y, mcqa_vals, 0.55, color=BLUE, alpha=0.85, zorder=3)
for i, val in enumerate(mcqa_vals):
    ax1b.text(val + 0.4, y[i], f"{val:.0f}%",
              va="center", ha="left", fontsize=10, color=BLUE)
ax1b.set_yticks(y)
ax1b.set_yticklabels(display_names, fontsize=13, color=BLACK)
ax1b.set_xlim(0, 120)
ax1b.set_ylim(-0.6, len(display_names) - 0.4)
ax1b.set_title("MCQA Accuracy", fontsize=12, color=BLACK, pad=5)
ax1b.set_xlabel("MCQA (%)", fontsize=12, color=BLACK, loc="left")
ax1b.annotate(
    "higher is better",
    xy=(1.0, -0.077), xytext=(0.5, -0.077),
    xycoords="axes fraction", textcoords="axes fraction",
    fontsize=11, color=RED, ha="left", va="center",
    arrowprops=dict(arrowstyle="->", color=RED, lw=1.0)
)
ax1b.tick_params(colors=BLACK, labelsize=12)
for spine in ax1b.spines.values():
    spine.set_visible(True)
    spine.set_edgecolor("#888780")
    spine.set_linewidth(0.7)
ax1b.grid(axis="x", color="#D3D1C7", linewidth=0.3, linestyle="--", alpha=0.5)

# --- CBS vs CQF box plot panel ---
h2 = 0.50

cbs_plot2, cbs_pos2 = [], []
cqf_plot2, cqf_pos2 = [], []
for i, (cbs, cqf) in enumerate(zip(cbs_data, cqf_data), start=1):
    if cbs and len(cbs) > 0:
        cbs_plot2.append(cbs)
        cbs_pos2.append(i + h2 / 2)
    if cqf and len(cqf) > 0:
        cqf_plot2.append(cqf)
        cqf_pos2.append(i - h2 / 2)


def make_bp2(ax, plot_data, positions, color):
    if not plot_data:
        return
    bp = ax.boxplot(
        plot_data, positions=positions, vert=False,
        patch_artist=True, notch=False, widths=h2 * 0.82,
        flierprops=dict(marker=".", markersize=3,
                        markerfacecolor=color, markeredgecolor=color,
                        alpha=0.5, linestyle="none"),
        medianprops=dict(linewidth=2.0),
        whiskerprops=dict(linewidth=0.8, linestyle="--"),
        capprops=dict(linewidth=0.8),
        boxprops=dict(linewidth=1.1)
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.2)
        patch.set_edgecolor(color)
    for el in bp["medians"]:
        el.set_color(color)
        el.set_linewidth(2.0)
    for el in bp["whiskers"] + bp["caps"]:
        el.set_color(color)
        el.set_alpha(0.65)
    for pos, d in zip(positions, plot_data):
        arr = np.array(d)
        ax.scatter(arr, pos + rng.uniform(-0.07, 0.07, len(arr)),
                   color=color, alpha=0.3, s=5, zorder=2, linewidths=0)
        ax.scatter(np.mean(arr), pos, marker="D", color=color,
                   s=18, zorder=5, edgecolors="white", linewidths=0.4)


make_bp2(ax2b, cbs_plot2, cbs_pos2, CORAL)
make_bp2(ax2b, cqf_plot2, cqf_pos2, TEAL)

all_vals2 = [v for d in cbs_data + cqf_data if d for v in d]
xmax2 = max(all_vals2) * 1.06 if all_vals2 else 5000

for i, (cbs, cqf) in enumerate(zip(cbs_data, cqf_data), start=1):
    if not cbs or len(cbs) == 0:
        ax2b.text(xmax2 * 0.01, i + h2 / 2, "--", va="center", ha="left",
                  fontsize=8, color=CORAL, style="italic")
    if not cqf or len(cqf) == 0:
        ax2b.text(xmax2 * 0.01, i - h2 / 2, "--", va="center", ha="left",
                  fontsize=8, color=TEAL, style="italic")

ax2b.set_yticks(range(1, len(display_names) + 1))
ax2b.set_yticklabels([])
ax2b.set_ylim(0.3, len(display_names) + 0.7)
ax2b.set_xlim(0, 5000)
ax2b.set_title("CBS vs CQF - Per-Flow MAE (µs) | 100 Test Cases (TCs)", fontsize=12,
               color=BLACK, pad=5)
# ax2b.set_title("CBS vs CQF - Per-Flow MAE (µs) | One-Switch Topology", fontsize=12,
#                color=BLACK, pad=5)
ax2b.set_xlabel("MAE (µs)", fontsize=12, color=BLACK)
ax2b.annotate(
    "lower is better",
    xy=(0.15, -0.077), xytext=(0.35, -0.077),
    xycoords="axes fraction", textcoords="axes fraction",
    fontsize=11, color=RED, ha="right", va="center",
    arrowprops=dict(arrowstyle="->", color=RED, lw=1.0)
)
ax2b.tick_params(colors=BLACK, labelsize=12)
for spine in ax2b.spines.values():
    spine.set_visible(True)
    spine.set_edgecolor("#888780")
    spine.set_linewidth(0.7)
ax2b.grid(axis="x", color="#D3D1C7", linewidth=0.3, linestyle="--", alpha=0.5)

handles2 = [
    mpatches.Patch(color=CORAL, alpha=0.4, label="CBS"),
    mpatches.Patch(color=TEAL,  alpha=0.4, label="CQF"),
]
ax2b.legend(handles=handles2, fontsize=13, framealpha=0.6,
            edgecolor="#D3D1C7", loc="lower right",
            ncol=2, handlelength=1.2, handletextpad=0.4, columnspacing=0.8)

plt.tight_layout()
plt.savefig("mcqa_cbs_vs_cqf_one_switch_topology.pdf", dpi=300)
plt.show()



