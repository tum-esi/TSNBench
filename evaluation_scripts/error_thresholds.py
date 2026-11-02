"""Share of flows with a WCD error below 50, 100, 500 and 1000 us."""

import copy
import csv
import json
import math
import os
from collections import defaultdict

import numpy as np

import mae_cb_fc_ablation as m

SETTING = "CB"  # "CB" or "FC"
MIN_FLOW_COVERAGE = 0.8
MIN_SCORED_TCS = 50
THRESHOLDS_US = (50, 100, 500, 1000)
MECHANISMS = ["CBS", "CQF"]
OUT_DIR = "results"


def is_trivial(wcd_values):
    if not wcd_values:
        return False
    values = [v for v in wcd_values.values() if v is not None]
    return len(values) > 0 and all(v == 0.0 for v in values)


def get_valid_runs(runs):
    return [r for r in runs
            if r.get("wcd_values") and not r.get("invalid")
            and not is_trivial(r.get("wcd_values", {}))]


def compute_wcd_from_runs(valid_runs):
    if not valid_runs:
        return {}
    if len(valid_runs) == 1:
        return dict(valid_runs[0]["wcd_values"])
    all_flow_ids = set()
    for r in valid_runs:
        all_flow_ids.update(r["wcd_values"].keys())
    avg_wcd = {}
    for fid in all_flow_ids:
        vals = [r["wcd_values"].get(fid) for r in valid_runs
                if r["wcd_values"].get(fid) is not None]
        if vals:
            avg_wcd[fid] = float(np.mean(vals))
    return avg_wcd


def tc_flow_errors(avg_wcd, gt_flows):
    if not avg_wcd or is_trivial(avg_wcd):
        return None
    total = len([v for v in gt_flows.values() if v and v > 0])
    covered = sum(1 for fid, g in gt_flows.items()
                  if g and g > 0 and avg_wcd.get(fid) is not None)
    if total == 0 or covered / total < MIN_FLOW_COVERAGE:
        return None
    errors = []
    for fid, g in gt_flows.items():
        if g is None or g == 0:
            continue
        p = avg_wcd.get(fid)
        if p is None or not math.isfinite(p):
            continue
        errors.append(abs(p - g))
    return errors or None


def load_runs(raw_dirs, mech):
    model_runs, _ = m.select_temperature(m.load_by_model_temp(raw_dirs, mech))
    if m.REPARSE:
        model_runs = m.reparse_records(model_runs)
    else:
        model_runs = copy.deepcopy(model_runs)
    if m.NORMALIZE_FLOW_IDS:
        model_runs = m.normalize_model_runs(model_runs)
    if SETTING == "FC":
        for mk in set(model_runs) & m.RETIRED_FC:
            del model_runs[mk]
    return model_runs


def apply_flow_id_rule(model_runs, gt):
    out = {}
    for mk, records in model_runs.items():
        bad = [r for r in records if gt.get(r["tc_name"]) and r.get("wcd_values")
               and m.has_unknown_ids(r, gt[r["tc_name"]])]
        if bad:
            if m.MISMATCH_SCOPE == "tc":
                bad_tcs = {r["tc_name"] for r in bad}
                records = [r for r in records if r["tc_name"] not in bad_tcs]
            else:
                bad_ids = {id(r) for r in bad}
                records = [r for r in records if id(r) not in bad_ids]
        out[mk] = records
    return out


def threshold_pct(model_runs, gt, tc_list):
    results = {}
    for model_key, records in model_runs.items():
        tc_groups = defaultdict(list)
        for r in records:
            tc_groups[r["tc_name"]].append(r)

        per_tc_band = {t: [] for t in THRESHOLDS_US}
        for tc in tc_list:
            runs, gt_flows = tc_groups.get(tc, []), gt.get(tc, {})
            if not runs or not gt_flows:
                continue
            valid_runs = get_valid_runs(runs)
            if not valid_runs:
                continue
            errors = tc_flow_errors(compute_wcd_from_runs(valid_runs), gt_flows)
            if errors is None:
                continue
            for t in THRESHOLDS_US:
                per_tc_band[t].append(sum(e < t for e in errors) / len(errors))

        ok = len(per_tc_band[THRESHOLDS_US[0]]) >= MIN_SCORED_TCS
        results[model_key] = {t: 100 * float(np.mean(per_tc_band[t])) if ok else None
                              for t in THRESHOLDS_US}
    return results


def fmt(v):
    return "--" if v is None else f"{v:.1f}"


def main():
    dirs = m.CB_DIRS if SETTING == "CB" else m.FC_DIRS
    gt = {}
    for mech in MECHANISMS:
        with open(m.GROUND_TRUTH[mech], encoding="utf-8") as f:
            gt[mech] = json.load(f)
    tc_list = sorted(set(gt["CBS"]) | set(gt["CQF"]),
                     key=lambda tc: int("".join(filter(str.isdigit, tc)) or 0))

    res = {}
    for mech in MECHANISMS:
        runs = apply_flow_id_rule(load_runs(dirs[mech], mech), gt[mech])
        res[mech] = threshold_pct(runs, gt[mech], tc_list)

    header = ["Model"] + [f"{mech} <{t}" for mech in MECHANISMS for t in THRESHOLDS_US]
    rows = []
    for mk, name in m.MODEL_DISPLAY.items():
        if mk not in res["CBS"] and mk not in res["CQF"]:
            continue
        cells = []
        for mech in MECHANISMS:
            r = res[mech].get(mk) or {t: None for t in THRESHOLDS_US}
            cells += [fmt(r[t]) for t in THRESHOLDS_US]
        if all(c == "--" for c in cells):
            continue
        rows.append([name] + cells)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"abs_error_thresholds_{SETTING.lower()}.csv"),
              "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([header] + rows)

    widths = [max(len(r[i]) for r in [header] + rows) for i in range(len(header))]
    print(f"\n% of flows with absolute WCD error below each threshold ({SETTING})")
    for r in [header] + rows:
        print(r[0].ljust(widths[0] + 2) + "".join(c.rjust(w + 2) for c, w in zip(r[1:], widths[1:])))


if __name__ == "__main__":
    main()
