import os
import json

SCORED_DIR  = "CQF/score"
PROMPT_TYPE = "CQF"

TC_LIST = [
    "TC1", "TC2", "TC3", "TC4", "TC6", "TC7", "TC8", "TC9", "TC10",
    "TC11", "TC12", "TC13", "TC14", "TC15", "TC16", "TC17", "TC18",
    "TC19", "TC20", "TC21"
]

scored_files = sorted([
    f for f in os.listdir(SCORED_DIR)
    if f.startswith("scored_openend_") and
       f"_{PROMPT_TYPE}_" in f and
       f.endswith(".json")
])

if not scored_files:
    print(f"No scored files found in {SCORED_DIR}")
    exit()

rows = []

for fname in scored_files:
    fpath = os.path.join(SCORED_DIR, fname)

    with open(fpath, encoding="utf-8") as f:
        data = json.load(f)

    model      = data.get("model", "unknown")
    tc_results = data.get("tc_results", {})
    enough_tcs = data.get("enough_tcs", True)

    values = []

    if not enough_tcs:
        print(f"% {model} — enough_tcs=False "
              f"(scored_tcs={data.get('scored_tcs')}/{data.get('total_tcs')}) "
              f"→ genuine MAE suppressed")

        for tc in TC_LIST:
            tc_data = tc_results.get(tc)
            if tc_data is None:
                values.append("--")
                continue

            scenario = tc_data.get("metrics_avg", {}).get("scenario", "")

            if scenario == "trivial_zeros":
                values.append("0")
            else:
                values.append("--")

        rows.append((model, values))
        continue

    for tc in TC_LIST:
        tc_data = tc_results.get(tc)
        if tc_data is None:
            values.append("--")
            continue

        metrics_avg = tc_data.get("metrics_avg", {})
        scenario    = metrics_avg.get("scenario", "")
        mae         = metrics_avg.get("mae_us")

        if scenario == "trivial_zeros":
            values.append("0")
        elif scenario == "invalid_missing" or mae is None:
            values.append("--")
        else:
            values.append(str(round(mae, 2)))

    rows.append((model, values))


best_per_tc = []
for col_idx in range(len(TC_LIST)):
    col_values = []
    for _, values in rows:
        v = values[col_idx]
        if v not in ("--", "0"):
            try:
                col_values.append(float(v))
            except ValueError:
                pass
    best_per_tc.append(min(col_values) if col_values else None)


for model, values in rows:
    formatted = []
    for col_idx, v in enumerate(values):
        best = best_per_tc[col_idx]
        if v not in ("--", "0") and best is not None:
            try:
                if round(float(v), 2) == round(best, 2):
                    formatted.append(f"\\textbf{{{v}}}")
                else:
                    formatted.append(v)
            except ValueError:
                formatted.append(v)
        else:
            formatted.append(v)

    row = f"{model} & " + " & ".join(formatted) + " \\\\"
    print(row)


print("\n% Best MAE per TC (for reference):")
best_row = " & ".join(
    str(round(b, 2)) if b is not None else "--"
    for b in best_per_tc
)
print(f"% Best & {best_row} \\\\")