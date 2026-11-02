"""API cost (USD) and latency (ms) per model."""

import copy
import csv
import json
import os
from collections import defaultdict

import mae_cb_fc_ablation as m

MCQA_DIR = "../mcqa_evaluation/output/results"
MCQA_TEMP_PREFERENCE = ["tempDefault", "temp0_0", "temp0_7"]

RAW_DIRS = {
    mech: [
        f"../open_end_evaluation/output/{mech}",
        f"../open_end_evaluation/output/{mech}/one_switch_topo",
        f"../open_end_evaluation/output/{mech}/medium_mesh_topo",
        f"../open_end_evaluation/output/{mech}/last_run",
        f"../open_end_evaluation/output/{mech}/tc5",
    ]
    for mech in ("CBS", "CQF")
}

TARGET_RUNS = 3
MIN_VALID_TCS = 50
OUT_FILE = "results/cost_latency.csv"

# USD per 1M tokens (input, output)
PRICING = {
    "grok-4-1-fast-reasoning": {"input": 0.20, "output": 0.50},
    "grok-4-1-fast-non-reasoning": {"input": 0.20, "output": 0.50},
    "deepseek-chat": {"input": 0.028, "output": 0.42},
    "gpt-4o-2024-08-06": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini-2024-07-18": {"input": 0.15, "output": 0.60},
    "Llama-3.3-70B-Instruct": {"input": 0.59, "output": 0.79},
    "mistral-medium-2508": {"input": 0.40, "output": 2.00},
    "mistral-large-2512": {"input": 0.50, "output": 1.50},
    "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},
    "o3-2025-04-16": {"input": 2.00, "output": 8.00},
    "gpt-5-2025-08-07": {"input": 1.25, "output": 10.00},
    "deepseek-reasoner": {"input": 0.028, "output": 0.42},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "Llama-3.2-1B": {"input": 0.10, "output": 0.10},
    "Qwen3-8B": {"input": 0.07, "output": 0.18},
    "ministral-8b-2512": {"input": 0.15, "output": 0.15},
}


def is_valid(r):
    return not (r.get("api_error") or r.get("invalid_response") or r.get("invalid"))


def cost_and_latency(r, model):
    in_rate = PRICING[model]["input"] / 1_000_000
    out_rate = PRICING[model]["output"] / 1_000_000
    cost = (r.get("tokens_in") or 0) * in_rate + (r.get("tokens_out") or 0) * out_rate
    return cost, r.get("latency_ms") or 0


def mcqa_file(model):
    files = [f for f in os.listdir(MCQA_DIR) if f.startswith(f"raw_{model}_") and f.endswith(".json")]
    if not files:
        return None
    rank = lambda f: next((i for i, t in enumerate(MCQA_TEMP_PREFERENCE) if t in f), len(MCQA_TEMP_PREFERENCE))
    return os.path.join(MCQA_DIR, min(files, key=rank))


def mcqa_totals(model):
    path = mcqa_file(model)
    if path is None:
        return None, None
    with open(path, encoding="utf-8") as f:
        records = [r for r in json.load(f) if is_valid(r)]
    if not records:
        return None, None
    cost = latency = 0
    for r in records:
        c, l = cost_and_latency(r, model)
        cost += c
        latency += l
    return round(cost, 4), int(latency)


def openend_totals(model, mech):
    runs_per_tc = defaultdict(list)
    prefix = f"raw_openend_{model}_{mech}_"
    for d in RAW_DIRS[mech]:
        for fname in os.listdir(d):
            if not (fname.startswith(prefix) and fname.endswith(".json")):
                continue
            with open(os.path.join(d, fname), encoding="utf-8") as f:
                for r in json.load(f):
                    if not is_valid(r) or ("wcd_values" in r and r["wcd_values"] is None):
                        continue
                    runs_per_tc[r.get("tc_name")].append(cost_and_latency(r, model))
    if not runs_per_tc:
        return None, None

    cost = latency = 0.0
    for runs in runs_per_tc.values():
        scale = TARGET_RUNS / len(runs)
        cost += sum(c for c, _ in runs) * scale
        latency += sum(l for _, l in runs) * scale
    return round(cost, 4), int(latency)


def valid_tc_counts(mech):
    with open(m.GROUND_TRUTH[mech], encoding="utf-8") as f:
        gt = json.load(f)
    runs, _ = m.select_temperature(m.load_by_model_temp(m.CB_DIRS[mech], mech))
    if m.REPARSE:
        runs = m.reparse_records(runs)
    elif m.NORMALIZE_FLOW_IDS:
        runs = copy.deepcopy(runs)
    if m.NORMALIZE_FLOW_IDS:
        runs = m.normalize_model_runs(runs)
    scores = m.score_all(runs, gt, sorted(gt), id_check=True)
    return {mk: len(s["per_tc_mae"]) for mk, s in scores.items()}


def fmt(v):
    if v is None:
        return "--"
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def main():
    n_valid = {mech: valid_tc_counts(mech) for mech in ("CBS", "CQF")}

    header = ["Model", "MCQA cost", "MCQA latency",
              "CBS cost", "CBS latency", "CQF cost", "CQF latency"]
    rows = []
    for model in PRICING:
        row = [m.MODEL_DISPLAY.get(model, model), *mcqa_totals(model)]
        for mech in ("CBS", "CQF"):
            if n_valid[mech].get(model, 0) >= MIN_VALID_TCS:
                row += openend_totals(model, mech)
            else:
                row += [None, None]
        rows.append([row[0]] + [fmt(v) for v in row[1:]])

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([header] + rows)

    widths = [max(len(r[i]) for r in [header] + rows) for i in range(len(header))]
    for r in [header] + rows:
        print(r[0].ljust(widths[0] + 2) + "".join(c.rjust(w + 2) for c, w in zip(r[1:], widths[1:])))


if __name__ == "__main__":
    main()
