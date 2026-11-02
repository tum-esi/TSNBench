"""Per-TC MAE for the medium-mesh TCs."""

import copy
import csv
import json
import os
from collections import defaultdict

import numpy as np

import mae_cb_fc_ablation as m

MECHANISMS = ["CBS", "CQF"]
SETTING = "CB"  # "CB" or "FC"
MESH_DIRS = {
    "CBS": "../open_end_evaluation/output/CBS/medium_mesh_topo",
    "CQF": "../open_end_evaluation/output/CQF/medium_mesh_topo",
}
N_SHOW = 10
SMALL_MODELS = ["Llama-3.2-1B", "Qwen3-8B", "ministral-8b-2512"]
DECIMALS = 2
OUT_DIR = "results"

TRIVIAL = "0"
INVALID = "--"


def tc_number(tc):
    return int("".join(c for c in tc if c.isdigit()) or 0)


def mesh_tcs(mech):
    tcs = set()
    d = MESH_DIRS[mech]
    for fname in os.listdir(d):
        if fname.startswith("raw_openend_") and f"_{mech}_" in fname and fname.endswith(".json"):
            with open(os.path.join(d, fname), encoding="utf-8") as f:
                tcs.update(r["tc_name"] for r in json.load(f) if isinstance(r, dict) and "tc_name" in r)
    return sorted(tcs, key=tc_number)[:N_SHOW]


def load_runs(mech):
    dirs = m.CB_DIRS if SETTING == "CB" else m.FC_DIRS
    runs, _ = m.select_temperature(m.load_by_model_temp(dirs[mech], mech))
    if SETTING == "FC":
        for mk in set(runs) & m.RETIRED_FC:
            del runs[mk]
    if m.REPARSE:
        runs = m.reparse_records(runs)
    elif m.NORMALIZE_FLOW_IDS:
        runs = copy.deepcopy(runs)
    if m.NORMALIZE_FLOW_IDS:
        runs = m.normalize_model_runs(runs)
    return runs


def tc_mae(records, gt):
    answered = [r for r in records if r.get("wcd_values") and not r.get("invalid")]
    if not answered:
        return INVALID
    if all(m.is_trivial(r["wcd_values"]) for r in answered):
        return TRIVIAL

    valid = m.get_valid_runs(records)
    renumbered = [r for r in valid if m.has_unknown_ids(r, gt)]
    if renumbered:
        if m.MISMATCH_SCOPE == "tc":
            return INVALID
        valid = [r for r in valid if r not in renumbered]
    if not valid:
        return INVALID

    errors = m.tc_errors(m.compute_wcd_from_runs(valid), gt)
    if errors is None:
        return INVALID
    return f"{np.mean(errors[0]):.{DECIMALS}f}"


def model_order(models):
    order = [mk for mk in m.MODEL_DISPLAY if mk in models]
    order += sorted(set(models) - set(m.MODEL_DISPLAY))
    order = [mk for mk in order if mk not in m.SKIP_MODELS]
    return [mk for mk in order if mk not in SMALL_MODELS] + [mk for mk in order if mk in SMALL_MODELS]


def per_tc_table(mech):
    with open(m.GROUND_TRUTH[mech], encoding="utf-8") as f:
        gt = json.load(f)
    tcs = [tc for tc in mesh_tcs(mech) if tc in gt]
    runs = load_runs(mech)

    rows = []
    for mk in model_order(runs):
        by_tc = defaultdict(list)
        for r in runs[mk]:
            by_tc[r["tc_name"]].append(r)
        rows.append([m.MODEL_DISPLAY.get(mk, mk)] + [tc_mae(by_tc[tc], gt[tc]) for tc in tcs])
    return ["Model"] + tcs, rows


def print_table(header, rows):
    width = max(len(r[0]) for r in [header] + rows) + 2
    for r in [header] + rows:
        print(r[0].ljust(width) + "".join(c.rjust(10) for c in r[1:]))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for mech in MECHANISMS:
        header, rows = per_tc_table(mech)
        path = os.path.join(OUT_DIR, f"medium_mesh_per_tc_mae_{mech.lower()}_{SETTING.lower()}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows([header] + rows)
        print(f"\n{mech} ({SETTING}) per-TC MAE (us)")
        print_table(header, rows)


if __name__ == "__main__":
    main()
