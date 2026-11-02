"""CBS results on the Orion topology, CB vs FC."""

import copy
import csv
import json
import os

import numpy as np

import mae_cb_fc_ablation as m

MECH = "CBS"
SETTINGS = ["CB", "FC"]
ORION_DIRS = {
    "CB": ["../open_end_evaluation/output/CBS/camera_ready/orion/closed_book"],
    "FC": ["../open_end_evaluation/output/CBS/camera_ready/orion/nc_equation"],
}
GROUND_TRUTH = "../open_end_evaluation/output/ground_truth/CBS/WCD_open_ended_CBS_orion.json"
MIN_VALID_TCS = 5
EXCLUDE = set(m.RETIRED_FC)
OUT_DIR = "results"


def tc_number(tc):
    return int("".join(c for c in tc if c.isdigit()) or 0)


def fmt(v):
    return "--" if v is None else f"{v:.1f}"


def display(mk):
    return m.MODEL_DISPLAY.get(mk, mk)


def per_tc_mae(setting, gt, tcs):
    runs, _ = m.select_temperature(m.load_by_model_temp(ORION_DIRS[setting], MECH))
    for mk in set(runs) & EXCLUDE:
        del runs[mk]
    if m.REPARSE:
        runs = m.reparse_records(runs)
    elif m.NORMALIZE_FLOW_IDS:
        runs = copy.deepcopy(runs)
    if m.NORMALIZE_FLOW_IDS:
        runs = m.normalize_model_runs(runs)
    scores = m.score_all(runs, gt, tcs, id_check=True)
    return {mk: s["per_tc_mae"] for mk, s in scores.items()}


def summarise(per_tc):
    vals = list(per_tc.values())
    if len(vals) < MIN_VALID_TCS:
        return {"n": len(vals), "mean": None, "std": None, "median": None}
    return {
        "n": len(vals),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=m.DDOF)),
        "median": float(np.median(vals)),
    }


def model_order(models):
    order = [mk for mk in m.MODEL_DISPLAY if mk in models]
    order += sorted(set(models) - set(m.MODEL_DISPLAY))
    return [mk for mk in order if mk not in EXCLUDE and mk not in m.SKIP_MODELS]


def summary_table(stats, models):
    # same models for CB and FC
    both = [mk for mk in models if all(stats[s][mk]["mean"] is not None for s in SETTINGS)]
    fc_lower = sum(stats["FC"][mk]["mean"] < stats["CB"][mk]["mean"] for mk in both)

    header = ["Setting", "Models", "Median MAE", "Best model", "Best MAE", "FC lowers MAE"]
    rows = []
    for s in SETTINGS:
        means = {mk: stats[s][mk]["mean"] for mk in both}
        best = min(means, key=means.get) if means else None
        rows.append([
            s,
            len(both),
            fmt(float(np.median(list(means.values()))) if means else None),
            display(best) if best else "--",
            fmt(means.get(best)),
            f"{fc_lower} of {len(both)}" if s == "FC" else "--",
        ])
    return header, rows


def per_model_table(stats, models):
    header = ["Model"]
    for s in SETTINGS:
        header += [f"{s} MAE", f"{s} std", f"{s} median", f"{s} n"]
    header.append("Delta (%)")

    rows = []
    for mk in models:
        row = [display(mk)]
        for s in SETTINGS:
            st = stats[s][mk]
            row += [fmt(st["mean"]), fmt(st["std"]), fmt(st["median"]), st["n"]]
        d = m.delta_pct(stats["CB"][mk]["mean"], stats["FC"][mk]["mean"])
        row.append("--" if d is None else f"{d:+.1f}")
        rows.append(row)
    return header, rows


def per_tc_table(per_tc, models, tcs):
    rows = [[display(mk)] + [fmt(per_tc.get(mk, {}).get(tc)) for tc in tcs] for mk in models]
    return ["Model"] + tcs, rows


def save(name, header, rows):
    rows = [[str(c) for c in r] for r in rows]
    with open(os.path.join(OUT_DIR, f"{name}.csv"), "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([header] + rows)

    widths = [max(len(r[i]) for r in [header] + rows) for i in range(len(header))]
    print(f"\n{name}")
    for r in [header] + rows:
        print(r[0].ljust(widths[0] + 2) + "".join(c.rjust(w + 2) for c, w in zip(r[1:], widths[1:])))


def main():
    with open(GROUND_TRUTH, encoding="utf-8") as f:
        gt = json.load(f)
    tcs = sorted(gt, key=tc_number)

    per_tc = {s: per_tc_mae(s, gt, tcs) for s in SETTINGS}
    models = model_order(set(per_tc["CB"]) | set(per_tc["FC"]))
    stats = {s: {mk: summarise(per_tc[s].get(mk, {})) for mk in models} for s in SETTINGS}

    os.makedirs(OUT_DIR, exist_ok=True)
    save("orion_cbs_summary", *summary_table(stats, models))
    save("orion_cbs_per_model", *per_model_table(stats, models))
    for s in SETTINGS:
        save(f"orion_cbs_per_tc_{s.lower()}", *per_tc_table(per_tc[s], models, tcs))


if __name__ == "__main__":
    main()
