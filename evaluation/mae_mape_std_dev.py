import os
import json
import numpy as np
from collections import defaultdict


CBS_RAW_DIRS = [
    "../open_end_evaluation/output/CBS",
    # "../open_end_evaluation/output/CBS/one_switch_topo",
    # "../open_end_evaluation/output/CBS/medium_mesh_topo",
]

CQF_RAW_DIRS = [
    "../open_end_evaluation/output/CQF",
    # "../open_end_evaluation/output/CQF/one_switch_topo",
    # "../open_end_evaluation/output/CQF/medium_mesh_topo",
]

CBS_GROUND_TRUTH = "../ground_truth/CBS/WCD_open_ended_CBS.json"
CQF_GROUND_TRUTH = "../ground_truth/CQF/WCD_open_ended_CQF.json"

MIN_FLOW_COVERAGE = 0.8
MIN_SCORED_TCS    = 50      # out of 100 now
EXACT_MATCH_TOL   = 0.01

TC_LIST_OVERRIDE = None

MODEL_DISPLAY = {
    "grok-4-1-fast-reasoning"    : "Grok 4.1 Fast",
    "grok-4-1-fast-non-reasoning": "Grok 4.1 Fast (Non-Reasoning)",
    "deepseek-chat"              : "DeepSeek-V3.2 (Non-Thinking)",
    "gpt-4o-2024-08-06"          : "GPT-4o",
    "gpt-4o-mini-2024-07-18"     : "GPT-4o mini",
    "Llama-3.3-70B-Instruct"     : "Llama 3.3 70B",
    "mistral-medium-2508"        : "Mistral Medium 3.1",
    "mistral-large-2512"         : "Mistral Large 3",
    "claude-sonnet-4-5-20250929" : "Claude Sonnet 4.5",
    "o3-2025-04-16"              : "o3",
    "gpt-5-2025-08-07"           : "GPT-5",
    "deepseek-reasoner"          : "DeepSeek-V3.2 (Thinking)",
    "gemini-2.5-flash"           : "Gemini 2.5 Flash",
    "Llama-3.2-1B"               : "Llama 3.2 1B",
    "Qwen3-8B"                   : "Qwen3 8B",
    "ministral-8b-2512"          : "Ministral 3 8B",
}


def is_trivial(wcd_values: dict) -> bool:
    if not wcd_values:
        return False
    values = [v for v in wcd_values.values() if v is not None]
    return len(values) > 0 and all(v == 0.0 for v in values)


def get_valid_runs(runs: list) -> list:
    return [
        r for r in runs
        if r.get("wcd_values")
        and not r.get("invalid")
        and not is_trivial(r.get("wcd_values", {}))
    ]


def compute_wcd_from_runs(valid_runs: list) -> dict:
    if not valid_runs:
        return {}

    if len(valid_runs) == 1:
        # Single run — use directly, no averaging needed
        return dict(valid_runs[0]["wcd_values"])

    # Multiple runs — average per flow
    all_flow_ids = set()
    for r in valid_runs:
        all_flow_ids.update(r["wcd_values"].keys())

    avg_wcd = {}
    for flow_id in all_flow_ids:
        values = [
            r["wcd_values"].get(flow_id)
            for r in valid_runs
            if r["wcd_values"].get(flow_id) is not None
        ]
        if values:
            avg_wcd[flow_id] = float(np.mean(values))

    return avg_wcd


def compute_tc_metrics_from_wcd(avg_wcd: dict, gt_flows: dict) -> dict:
    if not avg_wcd or is_trivial(avg_wcd):
        return None

    total_flows = len([v for v in gt_flows.values() if v and v > 0])
    valid_flows = sum(
        1 for fid, gt_val in gt_flows.items()
        if gt_val and gt_val > 0 and avg_wcd.get(fid) is not None
    )

    if total_flows == 0:
        return None

    coverage = valid_flows / total_flows
    if coverage < MIN_FLOW_COVERAGE:
        return None

    abs_errors    = []
    pct_errors    = []
    exact_matches = 0

    for flow_id, gt_val in gt_flows.items():
        if gt_val is None or gt_val == 0:
            continue
        pred_val = avg_wcd.get(flow_id)
        if pred_val is None:
            continue

        abs_err = abs(pred_val - gt_val)
        pct_err = abs_err / gt_val * 100
        abs_errors.append(abs_err)
        pct_errors.append(pct_err)
        if pct_err <= (EXACT_MATCH_TOL * 100):
            exact_matches += 1

    if not abs_errors:
        return None

    return {
        "mae"     : float(np.mean(abs_errors)),
        "mape"    : float(np.mean(pct_errors)),
        "em_rate" : exact_matches / total_flows * 100,
        "coverage": coverage,
        "n_runs"  : len(valid_runs) if False else None,  # set below
    }


def compute_tc_run_std(runs: list, gt_flows: dict) -> float:
    valid_runs = get_valid_runs(runs)
    if len(valid_runs) < 2:
        return 0.0

    mae_per_run = []
    for r in valid_runs:
        wcd = r["wcd_values"]
        abs_errors = []
        for fid, gt_val in gt_flows.items():
            if gt_val is None or gt_val == 0:
                continue
            pred = wcd.get(fid)
            if pred is None:
                continue
            abs_errors.append(abs(pred - gt_val))
        if abs_errors:
            mae_per_run.append(float(np.mean(abs_errors)))

    return float(np.std(mae_per_run)) if len(mae_per_run) >= 2 else 0.0




def load_raw_files_multi(raw_dirs: list, prompt_type: str) -> dict:
    model_runs = {}
    total_files = 0

    for raw_dir in raw_dirs:
        if not os.path.isdir(raw_dir):
            print(f"  SKIP (not found): {raw_dir}")
            continue

        raw_files = sorted([
            f for f in os.listdir(raw_dir)
            if f.startswith("raw_openend_") and
               f"_{prompt_type}_" in f and
               f.endswith(".json")
        ])

        for fname in raw_files:
            fpath = os.path.join(raw_dir, fname)
            with open(fpath, encoding="utf-8") as f:
                records = json.load(f)

            stem      = fname.replace("raw_openend_", "").replace(".json", "")
            parts     = stem.split(f"_{prompt_type}_")
            model_key = parts[0]

            if model_key not in model_runs:
                model_runs[model_key] = []
            model_runs[model_key].extend(records)
            total_files += 1

    print(f"  Loaded {total_files} files | {len(model_runs)} models")
    return model_runs



def compute_model_stats(model_runs: dict, ground_truth: dict,
                        tc_list: list) -> dict:
    results = {}

    for model_key, records in model_runs.items():
        tc_groups = defaultdict(list)
        for r in records:
            tc_groups[r["tc_name"]].append(r)

        per_tc_mae     = {}
        per_tc_mape    = {}
        per_tc_em      = {}
        per_tc_run_std = {}
        per_tc_n_runs  = {}   # how many valid runs per TC

        for tc in tc_list:
            runs = tc_groups.get(tc, [])
            gt   = ground_truth.get(tc, {})

            if not gt or not runs:
                continue

            valid_runs = get_valid_runs(runs)
            n_runs     = len(valid_runs)

            if n_runs == 0:
                continue

            # Use single run or average depending on count
            wcd    = compute_wcd_from_runs(valid_runs)
            metrics = compute_tc_metrics_from_wcd(wcd, gt)

            if metrics is None:
                continue

            per_tc_mae[tc]     = metrics["mae"]
            per_tc_mape[tc]    = metrics["mape"]
            per_tc_em[tc]      = metrics["em_rate"]
            per_tc_run_std[tc] = compute_tc_run_std(runs, gt)
            per_tc_n_runs[tc]  = n_runs

        scored_tcs = len(per_tc_mae)
        enough_tcs = scored_tcs >= MIN_SCORED_TCS

        if enough_tcs and per_tc_mae:
            mae_vals  = list(per_tc_mae.values())
            mape_vals = list(per_tc_mape.values())
            em_vals   = list(per_tc_em.values())
            std_vals  = list(per_tc_run_std.values())

            # Count how many TCs had 1 vs 3 runs
            n_single = sum(1 for n in per_tc_n_runs.values() if n == 1)
            n_multi  = sum(1 for n in per_tc_n_runs.values() if n  > 1)

            results[model_key] = {
                "per_tc_mae"     : per_tc_mae,
                "per_tc_mape"    : per_tc_mape,
                "per_tc_em"      : per_tc_em,
                "per_tc_run_std" : per_tc_run_std,
                "per_tc_n_runs"  : per_tc_n_runs,
                "mean_mae"       : float(np.mean(mae_vals)),
                "std_mae"        : float(np.std(mae_vals)),
                "median_mae"     : float(np.median(mae_vals)),
                "mean_mape"      : float(np.mean(mape_vals)),
                "std_mape"       : float(np.std(mape_vals)),
                "mean_em"        : float(np.mean(em_vals)),
                "std_em"         : float(np.std(em_vals)),
                "avg_run_std"    : float(np.mean(std_vals)) if std_vals else 0.0,
                "scored_tcs"     : scored_tcs,
                "enough_tcs"     : True,
                "n_single_run_tcs": n_single,
                "n_multi_run_tcs" : n_multi,
            }
        else:
            results[model_key] = {
                "per_tc_mae"     : per_tc_mae,
                "per_tc_mape"    : per_tc_mape,
                "per_tc_em"      : per_tc_em,
                "per_tc_run_std" : per_tc_run_std,
                "per_tc_n_runs"  : per_tc_n_runs,
                "mean_mae"       : None,
                "std_mae"        : None,
                "median_mae"     : None,
                "mean_mape"      : None,
                "std_mape"       : None,
                "mean_em"        : None,
                "std_em"         : None,
                "avg_run_std"    : None,
                "scored_tcs"     : scored_tcs,
                "enough_tcs"     : False,
                "n_single_run_tcs": 0,
                "n_multi_run_tcs" : 0,
            }

    return results


def fmt(val, decimals=1):
    if val is None:
        return "--"
    return f"{val:.{decimals}f}"


def fmt_pm(mean_val, std_val, decimals=1):
    if mean_val is None or std_val is None:
        return "--"
    return f"${mean_val:.{decimals}f} \\pm {std_val:.{decimals}f}$"


def print_console_summary(cbs_results: dict, cqf_results: dict,
                           tc_list: list):
    print("\n" + "=" * 140)
    print(f"  Total TCs evaluated: {len(tc_list)}")
    print(f"  MIN_SCORED_TCS threshold: {MIN_SCORED_TCS}")
    print("=" * 140)
    print(f"{'Model':<40} {'CBS MAE±std':>18} {'CBS MAPE':>12} "
          f"{'CBS EM':>7} {'CBS n/100':>10} {'CBS 1run':>9} "
          f"{'CQF MAE±std':>18} {'CQF MAPE':>12} "
          f"{'CQF EM':>7} {'CQF n/100':>10} {'CQF 1run':>9}")
    print("-" * 140)

    for mk, display in MODEL_DISPLAY.items():
        cbs = cbs_results.get(mk, {})
        cqf = cqf_results.get(mk, {})

        cbs_mae  = f"{fmt(cbs.get('mean_mae'))}±{fmt(cbs.get('std_mae'))}" \
                   if cbs.get('mean_mae') is not None else "--"
        cbs_mape = fmt(cbs.get('mean_mape'))
        cbs_em   = fmt(cbs.get('mean_em'))
        cbs_n    = f"{cbs.get('scored_tcs', 0)}/{len(tc_list)}"
        cbs_1r   = str(cbs.get('n_single_run_tcs', 0))

        cqf_mae  = f"{fmt(cqf.get('mean_mae'))}±{fmt(cqf.get('std_mae'))}" \
                   if cqf.get('mean_mae') is not None else "--"
        cqf_mape = fmt(cqf.get('mean_mape'))
        cqf_em   = fmt(cqf.get('mean_em'))
        cqf_n    = f"{cqf.get('scored_tcs', 0)}/{len(tc_list)}"
        cqf_1r   = str(cqf.get('n_single_run_tcs', 0))

        print(f"{display:<40} {cbs_mae:>18} {cbs_mape:>12} {cbs_em:>7} "
              f"{cbs_n:>10} {cbs_1r:>9} "
              f"{cqf_mae:>18} {cqf_mape:>12} {cqf_em:>7} "
              f"{cqf_n:>10} {cqf_1r:>9}")

    print("=" * 140)


def print_latex_summary_table(cbs_results: dict, cqf_results: dict):
    print("\n% ════════════════════════════════════════════════════════════════")
    print("% LATEX TABLE — CBS and CQF summary across 100 TCs")
    print("% Model & CBS MAE±std & CBS MAPE±std & CBS Median & CBS EM "
          "& CQF MAE±std & CQF MAPE±std & CQF Median & CQF EM \\\\")
    print("% ════════════════════════════════════════════════════════════════")
    print("\\midrule")

    for mk, display in MODEL_DISPLAY.items():
        cbs = cbs_results.get(mk, {})
        cqf = cqf_results.get(mk, {})

        if cbs.get("enough_tcs") and cbs.get("mean_mae") is not None:
            cbs_mae    = fmt_pm(cbs["mean_mae"],  cbs["std_mae"])
            cbs_mape   = fmt_pm(cbs["mean_mape"], cbs["std_mape"])
            cbs_median = fmt(cbs["median_mae"])
        else:
            cbs_mae = cbs_mape = cbs_median = "--"

        if cqf.get("enough_tcs") and cqf.get("mean_mae") is not None:
            cqf_mae    = fmt_pm(cqf["mean_mae"],  cqf["std_mae"])
            cqf_mape   = fmt_pm(cqf["mean_mape"], cqf["std_mape"])
            cqf_median = fmt(cqf["median_mae"])
        else:
            cqf_mae = cqf_mape = cqf_median = "--"

        print(f"{display} & {cbs_mae} & {cbs_mape} & {cbs_median} "
              f"& {cqf_mae} & {cqf_mape} & {cqf_median} \\\\")

    print("\\midrule")
    print("\\textbf{Overall} & & & & & & & & \\\\")
    print("\\bottomrule")


def print_latex_per_tc_table(results: dict, prompt_type: str,
                              tc_list: list):
    print(f"\n% ─── Per-TC MAE (µs) — {prompt_type} "
          f"({'  '.join(tc_list[:5])} ...) ───")
    print("% Model & " + " & ".join(tc_list) + " \\\\")
    print("\\midrule")

    for mk, display in MODEL_DISPLAY.items():
        data    = results.get(mk, {})
        per_tc  = data.get("per_tc_mae", {})
        run_std = data.get("per_tc_run_std", {})
        n_runs  = data.get("per_tc_n_runs", {})
        enough  = data.get("enough_tcs", False)

        cells = []
        for tc in tc_list:
            if not enough:
                cells.append("--")
                continue
            mae = per_tc.get(tc)
            std = run_std.get(tc, 0.0)
            nr  = n_runs.get(tc, 0)
            if mae is None:
                cells.append("--")
            elif nr == 1:
                # Single run — no ± std, add superscript to mark it
                cells.append(f"{fmt(mae)}$^*$")
            elif std > 0.5:
                cells.append(f"${fmt(mae)} \\pm {fmt(std)}$")
            else:
                cells.append(fmt(mae))

        print(f"{display} & " + " & ".join(cells) + " \\\\")

    print("\n% $^*$ Single run result (no averaging)")


def print_excluded(cbs_results: dict, cqf_results: dict, tc_list: list):
    print("\n% ─── Models excluded (scored_tcs < MIN_SCORED_TCS) ──────────────")
    for mk, display in MODEL_DISPLAY.items():
        cbs = cbs_results.get(mk, {})
        cqf = cqf_results.get(mk, {})
        if not cbs.get("enough_tcs") and mk in cbs_results:
            print(f"%   CBS  {display}: "
                  f"scored={cbs.get('scored_tcs', 0)}/{len(tc_list)} → --")
        if not cqf.get("enough_tcs") and mk in cqf_results:
            print(f"%   CQF  {display}: "
                  f"scored={cqf.get('scored_tcs', 0)}/{len(tc_list)} → --")

    qwen_key = "Qwen3-8B"
    qwen_cbs = cbs_results.get(qwen_key, {})
    qwen_cqf = cqf_results.get(qwen_key, {})
    print(f"%   Qwen3-8B: CBS scored={qwen_cbs.get('scored_tcs', 0)}/{len(tc_list)}  "
          f"CQF scored={qwen_cqf.get('scored_tcs', 0)}/{len(tc_list)}")


def main():
    print("Loading ground truth...")
    with open(CBS_GROUND_TRUTH, encoding="utf-8") as f:
        cbs_gt = json.load(f)
    with open(CQF_GROUND_TRUTH, encoding="utf-8") as f:
        cqf_gt = json.load(f)

    if TC_LIST_OVERRIDE:
        tc_list = TC_LIST_OVERRIDE
    else:
        def tc_sort_key(tc):
            num = ''.join(filter(str.isdigit, tc))
            return int(num) if num else 0
        tc_list = sorted(set(list(cbs_gt.keys()) + list(cqf_gt.keys())),
                         key=tc_sort_key)

    print(f"  CBS ground truth: {len(cbs_gt)} TCs")
    print(f"  CQF ground truth: {len(cqf_gt)} TCs")
    print(f"  TC list ({len(tc_list)} total): {tc_list[:5]} ... {tc_list[-3:]}")

    print("\nLoading CBS raw files...")
    cbs_model_runs = load_raw_files_multi(CBS_RAW_DIRS, "CBS")

    print("\nLoading CQF raw files...")
    cqf_model_runs = load_raw_files_multi(CQF_RAW_DIRS, "CQF")

    print("\nComputing statistics...")
    cbs_results = compute_model_stats(cbs_model_runs, cbs_gt, tc_list)
    cqf_results = compute_model_stats(cqf_model_runs, cqf_gt, tc_list)
    print(f"  CBS: {len(cbs_results)} models computed")
    print(f"  CQF: {len(cqf_results)} models computed")

    print_console_summary(cbs_results, cqf_results, tc_list)
    print_latex_summary_table(cbs_results, cqf_results)
    print_excluded(cbs_results, cqf_results, tc_list)


if __name__ == "__main__":
    main()