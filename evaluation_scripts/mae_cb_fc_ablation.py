"""WCD MAE per model, CB vs FC."""

import os
import re
import csv
import json
import copy
import math
import numpy as np
from collections import defaultdict

REPARSE = True

CB_DIRS = {
    "CBS": [
        "../open_end_evaluation/output/CBS",
        "../open_end_evaluation/output/CBS/one_switch_topo",
        "../open_end_evaluation/output/CBS/medium_mesh_topo",
        "../open_end_evaluation/output/CBS/last_run",
        "../open_end_evaluation/output/CBS/tc5",
    ],
    "CQF": [
        "../open_end_evaluation/output/CQF",
        "../open_end_evaluation/output/CQF/one_switch_topo",
        "../open_end_evaluation/output/CQF/medium_mesh_topo",
        "../open_end_evaluation/output/CQF/last_run",
        "../open_end_evaluation/output/CQF/tc5",
    ],
}

FC_DIRS = {
    "CBS": ["../open_end_evaluation/output/CBS/camera_ready/all/nc_equation"],
    "CQF": ["../open_end_evaluation/output/CQF/camera_ready/all/formula"],
}

GROUND_TRUTH = {
    "CBS": "../open_end_evaluation/output/ground_truth/CBS/WCD_open_ended_CBS.json",
    "CQF": "../open_end_evaluation/output/ground_truth/CQF/WCD_open_ended_CQF.json",
}

MIN_FLOW_COVERAGE = 0.8
MIN_SCORED_TCS = 50
MISMATCH_SCOPE = "run"     # "run" or "tc"
TEMP_PREFERENCE = ["temp0_0", "temp0_7", "tempDefault"]
DDOF = 0
NORMALIZE_FLOW_IDS = True
OUT_DIR = "results"

MECHANISMS = ["CBS", "CQF"]
SETTINGS = ["CB", "FC"]

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

SKIP_MODELS = set()

# retired before the FC runs
RETIRED_FC = {
    "grok-4-1-fast-reasoning",
    "grok-4-1-fast-non-reasoning",
    "deepseek-chat",
    "deepseek-reasoner",
}


def load_by_model_temp(raw_dirs, prompt_type):
    by_model_temp = defaultdict(lambda: defaultdict(list))
    for raw_dir in raw_dirs:
        if not os.path.isdir(raw_dir):
            print(f"  SKIP (not found): {raw_dir}")
            continue
        for fname in sorted(os.listdir(raw_dir)):
            if not (fname.startswith("raw_openend_")
                    and f"_{prompt_type}_" in fname
                    and fname.endswith(".json")):
                continue
            with open(os.path.join(raw_dir, fname), encoding="utf-8") as f:
                records = json.load(f)
            if not isinstance(records, list) or (records and not isinstance(records[0], dict)):
                print(f"  SKIP (bad structure): {fname}")
                continue
            stem = fname.replace("raw_openend_", "").replace(".json", "")
            model_key, temp_label = stem.split(f"_{prompt_type}_", 1)
            by_model_temp[model_key][temp_label].extend(records)
    return by_model_temp


def select_temperature(by_model_temp):
    model_runs, model_temp = {}, {}
    for model_key, temps in sorted(by_model_temp.items()):
        for t in TEMP_PREFERENCE:
            if t in temps:
                model_runs[model_key] = temps[t]
                model_temp[model_key] = t
                break
    return model_runs, model_temp


def reparse_records(model_runs):
    from wcd_parser import parse_response
    out = {}
    for mk, records in model_runs.items():
        new_records = []
        for r in records:
            r2 = copy.deepcopy(r)
            raw = r.get("raw_response")
            if raw is not None:
                parsed = parse_response(raw)
                wcd = parsed.get("wcd_values")
                r2["wcd_values"] = wcd
                r2["invalid"]    = parsed.get("invalid", not wcd)
                r2["refusal"]    = parsed.get("refusal", False)
                r2["parse_note"] = parsed.get("parse_note", "")
            new_records.append(r2)
        out[mk] = new_records
    return out


# "15", "f15", "Flow 15" -> "F15"
_FID_RE = re.compile(r"^\s*(?:f(?:low)?)?[\s_\-#:]*0*(\d+)\s*$", re.IGNORECASE)


def normalize_fid(key):
    if isinstance(key, bool):
        return str(key)
    if isinstance(key, (int, float)) and float(key).is_integer():
        return f"F{int(key)}"
    m = _FID_RE.match(str(key))
    if m:
        return f"F{int(m.group(1))}"
    return str(key).strip().upper()


def _to_number(v):
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).strip())
    except ValueError:
        return None


def normalize_wcd_ids(wcd):
    if not wcd:
        return wcd, False, 0
    out, renamed, collisions = {}, False, 0
    for k, v in wcd.items():
        nk = normalize_fid(k)
        if nk != k:
            renamed = True
        if nk in out:
            collisions += 1
            if out[nk] is None:
                out[nk] = _to_number(v)
            continue
        out[nk] = _to_number(v)
    return out, renamed, collisions


def normalize_model_runs(model_runs):
    for records in model_runs.values():
        for r in records:
            r["wcd_values"], _, _ = normalize_wcd_ids(r.get("wcd_values"))
    return model_runs


def is_trivial(wcd_values):
    if not wcd_values:
        return False
    values = [v for v in wcd_values.values() if v is not None]
    return len(values) > 0 and all(v == 0.0 for v in values)


def get_valid_runs(runs):
    return [r for r in runs
            if r.get("wcd_values")
            and not r.get("invalid")
            and not is_trivial(r.get("wcd_values", {}))]


def compute_wcd_from_runs(valid_runs):
    if not valid_runs:
        return {}
    if len(valid_runs) == 1:
        return dict(valid_runs[0]["wcd_values"])
    flow_ids = set()
    for r in valid_runs:
        flow_ids.update(r["wcd_values"].keys())
    avg = {}
    for fid in flow_ids:
        vals = [r["wcd_values"][fid] for r in valid_runs
                if r["wcd_values"].get(fid) is not None]
        if vals:
            avg[fid] = float(np.mean(vals))
    return avg


def tc_errors(avg_wcd, gt_flows):
    if not avg_wcd or is_trivial(avg_wcd):
        return None
    gt = {fid: v for fid, v in gt_flows.items() if v and v > 0}
    if not gt:
        return None
    n_pred = sum(1 for fid in gt if avg_wcd.get(fid) is not None)
    if n_pred / len(gt) < MIN_FLOW_COVERAGE:
        return None
    errs = [abs(avg_wcd[fid] - g) for fid, g in gt.items()
            if avg_wcd.get(fid) is not None and math.isfinite(avg_wcd[fid])]
    return (errs, len(gt)) if errs else None


_FLOW_ID_RE = re.compile(r"^F\d+$")


def has_unknown_ids(run, gt_flows):
    all_ids = set(gt_flows.keys())
    pred = set((run.get("wcd_values") or {}).keys())
    return any(_FLOW_ID_RE.match(fid) and fid not in all_ids for fid in pred)


def score_model(records, gt_all, tc_list, id_check=False):
    tc_groups = defaultdict(list)
    for r in records:
        tc_groups[r["tc_name"]].append(r)

    per_tc_mae, per_tc_runs = {}, {}
    id_mismatch = []
    for tc in tc_list:
        gt = gt_all.get(tc)
        if not gt:
            continue
        valid = get_valid_runs(tc_groups.get(tc, []))
        if id_check:
            bad = [r for r in valid if has_unknown_ids(r, gt)]
            if bad:
                id_mismatch += [(tc, r.get("run")) for r in bad]
                if MISMATCH_SCOPE == "tc":
                    continue
                valid = [r for r in valid if r not in bad]
        if not valid:
            continue
        e = tc_errors(compute_wcd_from_runs(valid), gt)
        if e is None:
            continue
        per_tc_mae[tc]  = float(np.mean(e[0]))
        per_tc_runs[tc] = len(valid)

    n = len(per_tc_mae)
    ok = n >= MIN_SCORED_TCS
    vals = list(per_tc_mae.values())
    return {
        "per_tc_mae"  : per_tc_mae,
        "per_tc_runs" : per_tc_runs,
        "scored_tcs"  : n,
        "enough_tcs"  : ok,
        "mean_mae"    : float(np.mean(vals))            if ok else None,
        "std_mae"     : float(np.std(vals, ddof=DDOF))  if ok else None,
        "median_mae"  : float(np.median(vals))          if ok else None,
        "id_mismatch_runs": id_mismatch,
    }


def score_all(model_runs, gt_all, tc_list, id_check=False):
    return {mk: score_model(recs, gt_all, tc_list, id_check)
            for mk, recs in model_runs.items()}


def delta_pct(cb, fc):
    if cb is None or fc is None or cb == 0:
        return None
    return (cb - fc) / cb * 100.0


def fmt(v, d=1):
    return "--" if v is None else f"{v:.{d}f}"


def ordered_models(*result_dicts):
    present = set()
    for d in result_dicts:
        present.update(d.keys())
    order = [mk for mk in MODEL_DISPLAY if mk in present]
    order += sorted(present - set(MODEL_DISPLAY))
    return [mk for mk in order if mk not in SKIP_MODELS]


def save_table(name, header, rows):
    rows = [[str(c) for c in r] for r in rows]
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"{name}.csv"), "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([header] + rows)

    widths = [max(len(r[i]) for r in [header] + rows) for i in range(len(header))]
    print(f"\n{name}")
    for r in [header] + rows:
        print(r[0].ljust(widths[0] + 2) + "".join(c.rjust(w + 2) for c, w in zip(r[1:], widths[1:])))


def mae_table(res, mech):
    header = ["Model"]
    for s in SETTINGS:
        header += [f"{s} MAE", f"{s} std", f"{s} median", f"{s} n"]
    header.append("Delta (%)")

    rows = []
    for mk in ordered_models(*(res[s][mech] for s in SETTINGS)):
        cb = res["CB"][mech].get(mk, {})
        fc = {} if mk in RETIRED_FC else res["FC"][mech].get(mk, {})
        row = [MODEL_DISPLAY.get(mk, mk)]
        for st in (cb, fc):
            row += [fmt(st.get("mean_mae")), fmt(st.get("std_mae")), fmt(st.get("median_mae")),
                    st.get("scored_tcs", "--")]
        d = delta_pct(cb.get("mean_mae"), fc.get("mean_mae"))
        row.append("--" if d is None else f"{d:+.1f}")
        rows.append(row)
    return header, rows


def main():
    gt = {}
    for mech in MECHANISMS:
        with open(GROUND_TRUTH[mech], encoding="utf-8") as f:
            gt[mech] = json.load(f)

    def tc_key(tc):
        num = "".join(filter(str.isdigit, tc))
        return int(num) if num else 0
    tc_list = sorted(set().union(*(gt[m].keys() for m in MECHANISMS)), key=tc_key)

    dirs = {"CB": CB_DIRS, "FC": FC_DIRS}
    res = {s: {} for s in SETTINGS}
    for setting in SETTINGS:
        for mech in MECHANISMS:
            runs, _ = select_temperature(load_by_model_temp(dirs[setting][mech], mech))
            if REPARSE:
                runs = reparse_records(runs)
            elif NORMALIZE_FLOW_IDS:
                runs = copy.deepcopy(runs)
            if NORMALIZE_FLOW_IDS:
                runs = normalize_model_runs(runs)
            res[setting][mech] = score_all(runs, gt[mech], tc_list, id_check=True)

    for mech in MECHANISMS:
        save_table(f"mae_{mech.lower()}", *mae_table(res, mech))


if __name__ == "__main__":
    main()
