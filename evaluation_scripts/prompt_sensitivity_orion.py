"""Prompt sensitivity on the Orion TCs."""

import copy
import csv
import json
import math
import os

import numpy as np

import mae_cb_fc_ablation as m

MECHANISMS = ["CBS", "CQF"]
CONDITIONS = ["original", "prompt_one", "prompt_two"]
LABELS = {"original": "CB", "prompt_one": "Variant A", "prompt_two": "Variant B"}
DIRS = {
    mech: {
        "original": [f"../open_end_evaluation/output/{mech}/camera_ready/orion/closed_book"],
        "prompt_one": [f"../open_end_evaluation/output/{mech}/camera_ready/orion/prompt_one"],
        "prompt_two": [f"../open_end_evaluation/output/{mech}/camera_ready/orion/prompt_two"],
    }
    for mech in MECHANISMS
}
GROUND_TRUTH = {
    mech: f"../open_end_evaluation/output/ground_truth/{mech}/WCD_open_ended_{mech}_orion.json"
    for mech in MECHANISMS
}
EXCLUDE = set(m.RETIRED_FC)
OUT_DIR = "results"


def tc_number(tc):
    return int("".join(c for c in tc if c.isdigit()) or 0)


def fmt(v):
    return "--" if v is None else f"{v:.1f}"


def load_runs(mech, cond):
    runs, _ = m.select_temperature(m.load_by_model_temp(DIRS[mech][cond], mech))
    for mk in set(runs) & EXCLUDE:
        del runs[mk]
    if m.REPARSE:
        runs = m.reparse_records(runs)
    elif m.NORMALIZE_FLOW_IDS:
        runs = copy.deepcopy(runs)
    if m.NORMALIZE_FLOW_IDS:
        runs = m.normalize_model_runs(runs)
    return runs


def stats(values, min_tcs):
    if len(values) < min_tcs:
        return {"n": len(values), "mean": None, "std": None, "median": None}
    return {"n": len(values), "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=m.DDOF)), "median": float(np.median(values))}


def model_order(models):
    order = [mk for mk in m.MODEL_DISPLAY if mk in models]
    order += sorted(set(models) - set(m.MODEL_DISPLAY))
    return [mk for mk in order if mk not in EXCLUDE and mk not in m.SKIP_MODELS]


def score_model(per_tc, min_tcs):
    have = [c for c in CONDITIONS if per_tc[c]]
    per_prompt = {c: stats(list(per_tc[c].values()), min_tcs) if c in have else None
                  for c in CONDITIONS}

    # TCs valid under all prompts
    paired, common = {c: None for c in CONDITIONS}, []
    if "original" in have and len(have) >= 2:
        common = set.intersection(*(set(per_tc[c]) for c in have))
        for c in have:
            paired[c] = stats([per_tc[c][t] for t in common], min_tcs)

    means = [paired[c]["mean"] for c in CONDITIONS if paired[c] and paired[c]["mean"] is not None]
    spread = {"range": None, "std": None, "cv": None}
    if len(means) == len(CONDITIONS):
        std = float(np.std(means))
        spread = {"range": float(max(means) - min(means)), "std": std,
                  "cv": float(std / np.mean(means) * 100) if np.mean(means) else None}
    return {"paired": paired, "n_paired": len(common), "per_prompt": per_prompt, **spread}


def save(name, header, rows):
    rows = [[str(c) for c in r] for r in rows]
    with open(os.path.join(OUT_DIR, f"{name}.csv"), "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([header] + rows)

    widths = [max(len(r[i]) for r in [header] + rows) for i in range(len(header))]
    print(f"\n{name}")
    for r in [header] + rows:
        print(r[0].ljust(widths[0] + 2) + "".join(c.rjust(w + 2) for c, w in zip(r[1:], widths[1:])))


def run(mech):
    with open(GROUND_TRUTH[mech], encoding="utf-8") as f:
        gt = json.load(f)
    runs = {c: load_runs(mech, c) for c in CONDITIONS}

    # TCs the variants were run on
    tcs = sorted({r["tc_name"] for c in CONDITIONS[1:] for recs in runs[c].values() for r in recs}
                 & set(gt), key=tc_number)
    min_tcs = math.ceil(0.5 * len(tcs))
    per_tc = {c: {mk: s["per_tc_mae"] for mk, s in m.score_all(runs[c], gt, tcs, id_check=True).items()}
              for c in CONDITIONS}

    models = model_order({mk for c in CONDITIONS[1:] for mk in runs[c]})
    results = {mk: score_model({c: per_tc[c].get(mk, {}) for c in CONDITIONS}, min_tcs)
               for mk in models}

    cv_rows = []
    for mk in models:
        r = results[mk]
        cv_rows.append([m.MODEL_DISPLAY.get(mk, mk)]
                       + [fmt(r["paired"][c]["mean"] if r["paired"][c] else None) for c in CONDITIONS]
                       + [fmt(r["range"]), fmt(r["std"]), fmt(r["cv"]), r["n_paired"]])
    save(f"prompt_sensitivity_orion_cv_{mech.lower()}",
         ["Model"] + [LABELS[c] for c in CONDITIONS] + ["Range", "Std", "CV (%)", "n"], cv_rows)

    header, prompt_rows = ["Model"], []
    for c in CONDITIONS:
        header += [f"{LABELS[c]} MAE", f"{LABELS[c]} std", f"{LABELS[c]} median", f"{LABELS[c]} n"]
    for mk in models:
        row = [m.MODEL_DISPLAY.get(mk, mk)]
        for c in CONDITIONS:
            s = results[mk]["per_prompt"][c] or {"n": 0, "mean": None, "std": None, "median": None}
            row += [fmt(s["mean"]), fmt(s["std"]), fmt(s["median"]), s["n"]]
        prompt_rows.append(row)
    save(f"prompt_sensitivity_orion_per_prompt_{mech.lower()}", header, prompt_rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for mech in MECHANISMS:
        run(mech)


if __name__ == "__main__":
    main()
