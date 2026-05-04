import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import os
import json
import logging
from collections import Counter
from statistics import mean, median
from datetime import datetime

OPEN_END_DIR = "output/CBS"
GROUND_TRUTH_FILE = "../ground_truth/CBS/WCD_open_ended_CBS.json"
SCORED_DIR = "output/CBS/score"

PROMPT_TYPE = "CBS"
EXACT_MATCH_TOL = 0.01
MIN_FLOW_COVERAGE = 0.8
MIN_SCORED_TCS = 50     # out of 100

os.makedirs(OPEN_END_DIR, exist_ok=True)
os.makedirs(SCORED_DIR,   exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            os.path.join(SCORED_DIR, "scoring.log"),
            encoding="utf-8",
            mode="w"
        ),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


def is_trivial(wcd_values: dict) -> bool:
    if not wcd_values:
        return False
    values = [v for v in wcd_values.values() if v is not None]
    return len(values) > 0 and all(v == 0.0 for v in values)


def get_majority_wcd(runs: list) -> dict:
    valid_runs = [
        r for r in runs
        if r.get("wcd_values")
           and not r["invalid"]
           and not is_trivial(r.get("wcd_values", {}))  # ← exclude trivial runs
    ]
    if not valid_runs:
        return {}

    all_flow_ids = set()
    for r in valid_runs:
        all_flow_ids.update(r["wcd_values"].keys())

    majority_wcd = {}
    for flow_id in all_flow_ids:
        values = []
        for r in valid_runs:
            v = r["wcd_values"].get(flow_id)
            if v is not None:
                values.append(round(v, 1))
        if values:
            majority_wcd[flow_id] = Counter(values).most_common(1)[0][0]

    return majority_wcd


def get_average_wcd(runs: list) -> dict:
    valid_runs = [
        r for r in runs
        if r.get("wcd_values")
           and not r["invalid"]
           and not is_trivial(r.get("wcd_values", {}))  # ← exclude trivial runs
    ]
    if not valid_runs:
        return {}

    all_flow_ids = set()
    for r in valid_runs:
        all_flow_ids.update(r["wcd_values"].keys())

    average_wcd = {}
    for flow_id in all_flow_ids:
        values = []
        for r in valid_runs:
            v = r["wcd_values"].get(flow_id)
            if v is not None:
                values.append(v)
        if values:
            average_wcd[flow_id] = round(mean(values), 4)

    return average_wcd


def flag_high_variance(runs: list, threshold: float = 10.0) -> list:
    flagged = []
    all_flow_ids = set()
    for r in runs:
        if r.get("wcd_values"):
            all_flow_ids.update(r["wcd_values"].keys())

    for flow_id in all_flow_ids:
        values = [
            r["wcd_values"].get(flow_id)
            for r in runs
            if r.get("wcd_values") and r["wcd_values"].get(flow_id)
        ]
        values = [v for v in values if v and v > 0]
        if len(values) >= 2 and max(values) / min(values) > threshold:
            flagged.append({
                "flow_id": flow_id,
                "values" : values,
                "ratio"  : round(max(values) / min(values), 1)
            })
    return flagged


def compute_flow_errors(wcd_values: dict, gt_flows: dict) -> list:
    flow_records = []
    for flow_id, gt_val in gt_flows.items():
        if gt_val is None or gt_val == 0:
            continue

        pred_val = wcd_values.get(flow_id)

        if pred_val is None:
            flow_records.append({
                "flow_id"     : flow_id,
                "ground_truth": gt_val,
                "predicted"   : None,
                "abs_error"   : None,
                "pct_error"   : None,
                "exact_match" : False,
                "missing"     : True,
            })
            continue

        abs_error   = abs(pred_val - gt_val)
        pct_error   = abs_error / gt_val * 100
        exact_match = pct_error <= (EXACT_MATCH_TOL * 100)

        flow_records.append({
            "flow_id"     : flow_id,
            "ground_truth": gt_val,
            "predicted"   : pred_val,
            "abs_error"   : round(abs_error, 4),
            "pct_error"   : round(pct_error, 4),
            "exact_match" : exact_match,
            "missing"     : False,
        })
    return flow_records


def compute_tc_metrics(flow_records: list, wcd_values: dict) -> dict:
    if is_trivial(wcd_values):
        return {
            "mae_us"          : None,
            "mape_pct"        : None,
            "exact_match_rate": 0.0,
            "total_flows"     : len(flow_records),
            "valid_flows"     : 0,
            "missing_flows"   : 0,
            "trivial"         : True,
            "scenario"        : "trivial_zeros",
            "coverage"        : 0.0,
            "coverage_failed" : False,
        }

    valid   = [f for f in flow_records if not f["missing"] and f["abs_error"] is not None]
    missing = [f for f in flow_records if f["missing"]]

    if not valid:
        return {
            "mae_us"          : None,
            "mape_pct"        : None,
            "exact_match_rate": 0.0,
            "total_flows"     : len(flow_records),
            "valid_flows"     : 0,
            "missing_flows"   : len(missing),
            "trivial"         : False,
            "scenario"        : "invalid_missing",
            "coverage"        : 0.0,
            "coverage_failed" : False,
        }

    total_flows = len(flow_records)
    valid_flows = len(valid)
    coverage    = valid_flows / total_flows if total_flows > 0 else 0.0

    if coverage < MIN_FLOW_COVERAGE:
        log.warning(
            f"    Coverage check failed: {valid_flows}/{total_flows} flows "
            f"({round(coverage * 100, 1)}%) — below {MIN_FLOW_COVERAGE * 100}% threshold "
            f"→ treated as invalid_missing"
        )
        return {
            "mae_us"          : None,
            "mape_pct"        : None,
            "exact_match_rate": 0.0,
            "total_flows"     : total_flows,
            "valid_flows"     : valid_flows,
            "missing_flows"   : len(missing),
            "trivial"         : False,
            "scenario"        : "invalid_missing",
            "coverage"        : round(coverage, 4),
            "coverage_failed" : True,
        }

    mae     = mean(f["abs_error"] for f in valid)
    mape    = mean(f["pct_error"] for f in valid)
    em_rate = sum(1 for f in valid if f["exact_match"]) / total_flows * 100

    return {
        "mae_us"          : round(mae, 4),
        "mape_pct"        : round(mape, 4),
        "exact_match_rate": round(em_rate, 4),
        "total_flows"     : total_flows,
        "valid_flows"     : valid_flows,
        "missing_flows"   : len(missing),
        "trivial"         : False,
        "scenario"        : "genuine_attempt",
        "coverage"        : round(coverage, 4),
        "coverage_failed" : False,
    }


def compute_model_summary(model_key: str,
                          temp_label: str,
                          tc_results: dict,
                          all_runs: list) -> dict:
    per_tc_mae_maj  = []
    per_tc_mape_maj = []
    per_tc_mae_avg  = []
    per_tc_mape_avg = []
    per_tc_em       = []
    all_errors      = []
    trivial_tcs     = 0
    invalid_tcs     = 0
    coverage_failed = 0

    for tc_name, tc_data in tc_results.items():
        m_maj = tc_data["metrics"]
        m_avg = tc_data["metrics_avg"]

        if m_maj["scenario"] == "trivial_zeros":
            trivial_tcs += 1
            continue

        if m_maj["scenario"] == "invalid_missing":
            invalid_tcs += 1
            if m_maj.get("coverage_failed"):
                coverage_failed += 1
            continue

        if m_maj["mae_us"] is not None:
            per_tc_mae_maj.append(m_maj["mae_us"])
            per_tc_mape_maj.append(m_maj["mape_pct"])
            per_tc_em.append(m_maj["exact_match_rate"])

        if m_avg["mae_us"] is not None:
            per_tc_mae_avg.append(m_avg["mae_us"])
            per_tc_mape_avg.append(m_avg["mape_pct"])

        # Collect flow-level errors from majority vote records
        for f in tc_data["flow_records"]:
            if f["abs_error"] is not None:
                all_errors.append(f["abs_error"])

    total_tcs  = len(tc_results)
    scored_tcs = len(per_tc_mae_maj)
    enough_tcs = scored_tcs >= MIN_SCORED_TCS

    if not enough_tcs:
        log.warning(
            f"  {model_key}: only {scored_tcs}/{total_tcs} TCs scored "
            f"(minimum {MIN_SCORED_TCS} required) → MAE/MAPE reported as null"
        )

    tokens_in  = [r["tokens_in"]  for r in all_runs if r.get("tokens_in")  is not None]
    tokens_out = [r["tokens_out"] for r in all_runs if r.get("tokens_out") is not None]
    latencies  = [r["latency_ms"] for r in all_runs if r.get("latency_ms") is not None]

    return {
        "model"                : model_key,
        "prompt_type"          : PROMPT_TYPE,
        "temperature_label"    : temp_label,
        "timestamp"            : datetime.utcnow().isoformat(),
        "total_tcs"            : total_tcs,
        "scored_tcs"           : scored_tcs,
        "trivial_tcs"          : trivial_tcs,
        "invalid_tcs"          : invalid_tcs,
        "coverage_failed_tcs"  : coverage_failed,
        "trivial_rate"         : round(trivial_tcs    / total_tcs, 4) if total_tcs else 0,
        "invalid_rate"         : round(invalid_tcs    / total_tcs, 4) if total_tcs else 0,
        "coverage_failed_rate" : round(coverage_failed / total_tcs, 4) if total_tcs else 0,
        "enough_tcs"           : enough_tcs,
        "min_scored_tcs"       : MIN_SCORED_TCS,
        "mae_us_majority"      : round(mean(per_tc_mae_maj),  4)
                                 if enough_tcs and per_tc_mae_maj  else None,
        "mape_pct_majority"    : round(mean(per_tc_mape_maj), 4)
                                 if enough_tcs and per_tc_mape_maj else None,
        "mae_us_average"       : round(mean(per_tc_mae_avg),  4)
                                 if enough_tcs and per_tc_mae_avg  else None,
        "mape_pct_average"     : round(mean(per_tc_mape_avg), 4)
                                 if enough_tcs and per_tc_mape_avg else None,
        "exact_match_rate"     : round(mean(per_tc_em), 4)
                                 if enough_tcs and per_tc_em else None,
        "error_min"            : round(min(all_errors), 4)
                                 if enough_tcs and all_errors else None,
        "error_q1"             : round(sorted(all_errors)[int(len(all_errors)*0.25)], 4)
                                 if enough_tcs and all_errors else None,
        "error_median"         : round(median(all_errors), 4)
                                 if enough_tcs and all_errors else None,
        "error_q3"             : round(sorted(all_errors)[int(len(all_errors)*0.75)], 4)
                                 if enough_tcs and all_errors else None,
        "error_max"            : round(max(all_errors), 4)
                                 if enough_tcs and all_errors else None,
        "error_mean"           : round(mean(all_errors), 4)
                                 if enough_tcs and all_errors else None,
        "total_flow_errors"    : len(all_errors),
        "flat_errors"          : all_errors if enough_tcs else [],
        "total_tokens_in"      : sum(tokens_in)                   if tokens_in  else 0,
        "total_tokens_out"     : sum(tokens_out)                  if tokens_out else 0,
        "total_tokens"         : sum(tokens_in) + sum(tokens_out) if tokens_in  else 0,
        "avg_tokens_in"        : round(mean(tokens_in),  1)       if tokens_in  else 0,
        "avg_tokens_out"       : round(mean(tokens_out), 1)       if tokens_out else 0,
        "avg_latency_ms"       : round(mean(latencies),  1)       if latencies  else 0,
        "max_latency_ms"       : max(latencies)                   if latencies  else 0,
        "total_time_seconds"   : round(sum(latencies) / 1000, 2)  if latencies  else 0,
    }


def main():
    if not os.path.exists(GROUND_TRUTH_FILE):
        log.error(f"Ground truth not found: {GROUND_TRUTH_FILE}")
        return

    with open(GROUND_TRUTH_FILE, encoding="utf-8") as f:
        ground_truth = json.load(f)
    log.info(f"Loaded ground truth: {len(ground_truth)} TCs")

    raw_files = sorted([
        f for f in os.listdir(OPEN_END_DIR)
        if f.startswith("raw_openend_") and
           f"_{PROMPT_TYPE}_" in f and
           f.endswith(".json")
    ])

    if not raw_files:
        log.error(f"No raw result files found in {OPEN_END_DIR}")
        return

    log.info(f"Found {len(raw_files)} result files (all temperatures)")
    for f in raw_files:
        log.info(f"  {f}")

    all_summaries = []
    model_errors  = {}

    for fname in raw_files:
        raw_path = os.path.join(OPEN_END_DIR, fname)

        stem       = fname.replace("raw_openend_", "").replace(".json", "")
        parts      = stem.split(f"_{PROMPT_TYPE}_")
        model_key  = parts[0]
        temp_label = parts[1] if len(parts) > 1 else "unknown"

        log.info(f"\n{'='*70}")
        log.info(f"Model: {model_key} | Temp: {temp_label}")

        with open(raw_path, encoding="utf-8") as f:
            raw_results = json.load(f)

        tc_groups = {}
        for r in raw_results:
            tc_groups.setdefault(r["tc_name"], []).append(r)

        tc_results     = {}
        scored_records = []

        for tc_name, runs in sorted(tc_groups.items()):
            gt_flows = ground_truth.get(tc_name, {})
            if not gt_flows:
                log.warning(f"  No ground truth for {tc_name} — skipping")
                continue

            majority_wcd = get_majority_wcd(runs)
            average_wcd  = get_average_wcd(runs)
            hv_flags     = flag_high_variance(runs)

            if hv_flags:
                log.warning(f"  {tc_name}: {len(hv_flags)} high-variance flows "
                            f"— {[f['flow_id'] for f in hv_flags]}")

            # Majority vote scoring
            flow_records_majority = compute_flow_errors(majority_wcd, gt_flows)
            tc_metrics_majority   = compute_tc_metrics(flow_records_majority, majority_wcd)

            # Average scoring
            flow_records_average  = compute_flow_errors(average_wcd, gt_flows)
            tc_metrics_average    = compute_tc_metrics(flow_records_average, average_wcd)

            for fr in flow_records_majority:
                fr["tc_name"] = tc_name
                fr["model"]   = model_key

            for fr in flow_records_average:
                fr["tc_name"] = tc_name
                fr["model"]   = model_key

            tc_results[tc_name] = {
                "metrics"          : tc_metrics_majority,
                "metrics_avg"      : tc_metrics_average,
                "flow_records"     : flow_records_majority,
                "flow_records_avg" : flow_records_average,
                "majority_wcd"     : majority_wcd,
                "average_wcd"      : average_wcd,
                "hv_flags"         : hv_flags,
            }

            scored_records.extend(flow_records_majority)

            cov_maj = tc_metrics_majority.get("coverage", 1.0)
            cov_tag = f" [COVERAGE FAILED: {round(cov_maj * 100, 1)}%]" \
                      if tc_metrics_majority.get("coverage_failed") else ""

            tag = {
                "trivial_zeros"  : " [TRIVIAL — all zeros]",
                "invalid_missing": " [INVALID — no output]",
                "genuine_attempt": "",
            }.get(tc_metrics_majority["scenario"], "")

            log.info(
                f"  {tc_name}{tag}{cov_tag}: "
                f"MAE(maj)={tc_metrics_majority['mae_us']} µs  "
                f"MAE(avg)={tc_metrics_average['mae_us']} µs  "
                f"MAPE(maj)={tc_metrics_majority['mape_pct']}%  "
                f"MAPE(avg)={tc_metrics_average['mape_pct']}%  "
                f"flows={tc_metrics_majority['valid_flows']}/{tc_metrics_majority['total_flows']}  "
                f"coverage={round(cov_maj * 100, 1)}%"
            )

        summary = compute_model_summary(model_key, temp_label, tc_results, raw_results)
        all_summaries.append(summary)
        model_errors[f"{model_key}_{temp_label}"] = summary["flat_errors"]

        log.info(
            f"  SUMMARY → "
            f"scored={summary['scored_tcs']}/{summary['total_tcs']} "
            f"(min={MIN_SCORED_TCS}) "
            f"enough={summary['enough_tcs']}  "
            f"MAE(maj)={summary['mae_us_majority']} µs  "
            f"MAE(avg)={summary['mae_us_average']} µs  "
            f"MAPE(maj)={summary['mape_pct_majority']}%  "
            f"MAPE(avg)={summary['mape_pct_average']}%  "
            f"EM={summary['exact_match_rate']}%  "
            f"trivial={summary['trivial_tcs']}/{summary['total_tcs']}  "
            f"invalid={summary['invalid_tcs']}/{summary['total_tcs']}  "
            f"cov_failed={summary['coverage_failed_tcs']}/{summary['total_tcs']}"
        )

        flow_records_avg_out = []
        for tc_name_key, tc_data in tc_results.items():
            scenario = tc_data["metrics_avg"].get("scenario", "")
            if scenario != "genuine_attempt":
                continue
            for fr in tc_data.get("flow_records_avg", []):
                if fr.get("abs_error") is not None:
                    flow_records_avg_out.append(fr)

        scored_path = os.path.join(
            SCORED_DIR,
            f"scored_openend_{model_key}_{PROMPT_TYPE}_{temp_label}.json"
        )
        out = {k: v for k, v in summary.items() if k != "flat_errors"}
        out["tc_results"] = {
            tc: {
                "metrics"     : d["metrics"],
                "metrics_avg" : d["metrics_avg"],
                "hv_flags"    : d["hv_flags"]
            }
            for tc, d in tc_results.items()
        }
        out["flow_records"]     = scored_records
        out["flow_records_avg"] = flow_records_avg_out   # per-flow avg errors for box plot

        with open(scored_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

        log.info(
            f"  Saved: {scored_path} | "
            f"flow_records={len(scored_records)} | "
            f"flow_records_avg={len(flow_records_avg_out)}"
        )

    errors_path = os.path.join(SCORED_DIR, f"model_errors_{PROMPT_TYPE}.json")
    with open(errors_path, "w", encoding="utf-8") as f:
        json.dump(model_errors, f, indent=2, ensure_ascii=False)
    log.info(f"\nModel errors saved: {errors_path}")

    clean = [{k: v for k, v in s.items() if k != "flat_errors"}
             for s in all_summaries]
    full_path = os.path.join(SCORED_DIR, f"full_scored_summary_{PROMPT_TYPE}.json")
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)
    log.info(f"Full summary saved: {full_path}")

    log.info(f"\n{'='*120}")
    log.info(
        f"{'Model':<30} {'Temp':<12} {'Scored':>7} {'MAE-maj':>8} {'MAE-avg':>8} "
        f"{'MAPE-maj':>9} {'MAPE-avg':>9} {'EM(%)':>7} "
        f"{'Trivial':>8} {'Invalid':>8} {'CovFail':>8}"
    )
    log.info(f"{'-'*120}")

    for s in sorted(clean,
                    key=lambda x: (x["mae_us_majority"] is None,
                                   x["mae_us_majority"] or 99999)):
        mae_maj  = f"{s['mae_us_majority']:.1f}"    if s["mae_us_majority"]   is not None else "--"
        mae_avg  = f"{s['mae_us_average']:.1f}"     if s["mae_us_average"]    is not None else "--"
        mape_maj = f"{s['mape_pct_majority']:.1f}"  if s["mape_pct_majority"] is not None else "--"
        mape_avg = f"{s['mape_pct_average']:.1f}"   if s["mape_pct_average"]  is not None else "--"
        em       = f"{s['exact_match_rate']:.1f}"   if s["exact_match_rate"]  is not None else "--"
        scored   = f"{s['scored_tcs']}/{s['total_tcs']}"
        trivial  = f"{s['trivial_tcs']}/{s['total_tcs']}"
        invalid  = f"{s['invalid_tcs']}/{s['total_tcs']}"
        covfail  = f"{s['coverage_failed_tcs']}/{s['total_tcs']}"

        log.info(
            f"{s['model']:<30} "
            f"{s['temperature_label']:<12} "
            f"{scored:>7} "
            f"{mae_maj:>8} "
            f"{mae_avg:>8} "
            f"{mape_maj:>9} "
            f"{mape_avg:>9} "
            f"{em:>7} "
            f"{trivial:>8} "
            f"{invalid:>8} "
            f"{covfail:>8}"
        )
    log.info(f"{'='*120}")
    log.info("Scoring complete.")


if __name__ == "__main__":
    main()