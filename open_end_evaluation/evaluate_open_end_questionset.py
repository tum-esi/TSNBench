import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import os
import re
import json
import time
import logging
import warnings
from pathlib import Path
from datetime import datetime
from statistics import mean
from collections import Counter

warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

from google.genai import types as genai_types
from config import (
    MODELS, RESULTS_DIR,
    TEMPERATURE_WITH, TEMPERATURE_WITHOUT,
    compute_cost
)

TC_DIR         = "../dataset/open_ended"
PROMPT_TYPE    = "CQF"     # "CQF" or "CBS"
RUNS_PER_TC    = 3
OPEN_END_DIR   = "output/CQF/last_run"

# TC_LIST = [
#     "TC1","TC2","TC3","TC4","TC5",
#     "TC6","TC7","TC8","TC9","TC10",
#     "TC11","TC12","TC13","TC14","TC15",
#     "TC16","TC17","TC18","TC19","TC20",
#     "TC21"
# ]

# TC_LIST = [
#     "TC1","TC2","TC3","TC4",
#     "TC6","TC7","TC8","TC9","TC10",
#     "TC11","TC12","TC13","TC14","TC15",
#     "TC16","TC17","TC18","TC19","TC20","TC21"
# ]

# TC_LIST = [
#     "TC41","TC42","TC43","TC44","TC45",
#     "TC46","TC47","TC48","TC49","TC50",
#     "TC51"
# ]

# TC_LIST = [
#     "TC71","TC72","TC73","TC74","TC75",
#     "TC76","TC77","TC78","TC79","TC80",
#     "TC81","TC82","TC83","TC84","TC85",
#     "TC86","TC87","TC88","TC89","TC90",
#     "TC91","TC92","TC93","TC94","TC95",
#     "TC96","TC97","TC98","TC99","TC100"
# ]

TC_LIST = [
    "TC22","TC23","TC24","TC25","TC26",
    "TC27","TC28","TC29","TC30","TC31",
    "TC32","TC33","TC34","TC35","TC36",
    "TC37","TC38","TC39","TC40",
    "TC52","TC53","TC54","TC55","TC56",
    "TC57","TC58","TC59","TC60","TC61",
    "TC62","TC63","TC64","TC65","TC66",
    "TC67","TC68","TC69","TC70"
]


os.makedirs(OPEN_END_DIR, exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            os.path.join(OPEN_END_DIR, "openend_evaluation.log"),
            encoding="utf-8"
        ),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

for noisy in ["httpx", "httpcore", "openai", "anthropic",
              "mistralai", "google", "urllib3"]:
    logging.getLogger(noisy).setLevel(logging.ERROR)


def build_prompt(tc_name: str) -> str:
    topo_file = os.path.join(TC_DIR, f"{tc_name}_topo.txt")
    flows_file = os.path.join(TC_DIR, f"{tc_name}_flows.txt")
    route_file = os.path.join(TC_DIR, f"{tc_name}_route.txt")

    for f in [topo_file, flows_file, route_file]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing input file: {f}")

    if PROMPT_TYPE == "CQF":
        from CQF import CQF
        return CQF(topo_file, flows_file, route_file)
    elif PROMPT_TYPE == "CBS":
        from CBS import CBS
        return CBS(topo_file, flows_file, route_file)
    else:
        raise ValueError(f"Unknown PROMPT_TYPE: {PROMPT_TYPE}. Use 'CQF' or 'CBS'.")


def parse_response(raw: str) -> dict:
    empty = {
        "wcd_values"  : None,
        "confidence"  : None,
        "flow_profile": None,
        "reason"      : None,
        "invalid"     : True,
        "refusal"     : False,
        "parse_note"  : ""
    }

    if not raw:
        return {**empty, "parse_note": "empty_response"}

    clean = raw.strip()
    clean = re.sub(r"^```(?:json)?", "", clean).strip()
    clean = re.sub(r"```$",          "", clean).strip()

    refusal_patterns = [
        "i cannot", "i can't", "i am not able", "i'm not able",
        "unable to", "cannot answer", "not enough information",
        "insufficient information", "cannot calculate",
        "not possible to determine"
    ]
    if any(p in clean.lower() for p in refusal_patterns):
        return {**empty, "invalid": False, "refusal": True,
                "parse_note": "refusal_detected"}

    start = clean.find("{")
    end   = clean.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(clean[start:end + 1])

            wcd_raw    = parsed.get("WCD_us", {})
            wcd_values = {}
            bad_values = []

            for flow_id, val in wcd_raw.items():
                try:
                    wcd_values[flow_id] = float(val)
                except (TypeError, ValueError):
                    num_match = re.search(r"[\d.]+", str(val))
                    if num_match:
                        wcd_values[flow_id] = float(num_match.group())
                        bad_values.append(flow_id)
                    else:
                        wcd_values[flow_id] = None
                        bad_values.append(flow_id)

            confidence_raw = parsed.get("confidence")
            if isinstance(confidence_raw, (int, float)):
                confidence = round(max(0.0, min(1.0, float(confidence_raw))), 4)
            else:
                confidence = None

            if not wcd_raw:
                note = "json_parsed_no_wcd_field"
            elif bad_values:
                note = f"wcd_not_numeric:{bad_values}"
            else:
                note = "ok"

            return {
                "wcd_values"  : wcd_values if wcd_values else None,
                "confidence"  : confidence,
                "flow_profile": parsed.get("flow_profile"),
                "reason"      : parsed.get("reason"),
                "invalid"     : not wcd_values or all(v is None for v in wcd_values.values()),
                "refusal"     : False,
                "parse_note"  : note
            }

        except json.JSONDecodeError as e:
            log.warning(f"  JSON parse error: {e} — trying fallback extractor")

    wcd_values = {}
    patterns = [
        r"['\"]?(F\d+)['\"]?\s*[:=]\s*([\d.]+)",
        r"(?:WCD|delay)\s+(?:of|for)?\s*(F\d+)\s+is\s+([\d.]+)",
        r"(F\d+)[^:=\d]+([\d.]+)\s*(?:µs|us|microseconds)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, clean, re.IGNORECASE):
            flow_id = match.group(1).upper()
            try:
                wcd_values[flow_id] = float(match.group(2))
            except ValueError:
                pass

    if wcd_values:
        log.warning(f"  Extracted {len(wcd_values)} WCD values from plain text: {wcd_values}")
        return {
            "wcd_values"  : wcd_values,
            "confidence"  : None,
            "flow_profile": None,
            "reason"      : None,
            "invalid"     : False,
            "refusal"     : False,
            "parse_note"  : "extracted_from_plain_text"
        }

    log.warning(f"  Could not extract WCD values. Raw[:300]: {raw[:300]}")
    return {**empty, "parse_note": "parse_failed_no_wcd_extractable"}


def compute_wcd_consistency(runs: list) -> dict:
    if not runs:
        return {}
    all_flow_ids = set()
    for r in runs:
        if r.get("wcd_values"):
            all_flow_ids.update(r["wcd_values"].keys())

    consistency = {}
    for flow_id in all_flow_ids:
        values = []
        for r in runs:
            wcd = r.get("wcd_values") or {}
            v   = wcd.get(flow_id)
            if v is not None:
                values.append(v)

        if not values:
            consistency[flow_id] = 0.0
            continue

        rounded = [round(v, 1) for v in values]
        most_common_count = Counter(rounded).most_common(1)[0][1]
        consistency[flow_id] = round(most_common_count / RUNS_PER_TC, 4)

    return consistency


def call_model(model_cfg: dict, prompt: str, temperature: float) -> dict:
    api      = model_cfg["api"]
    client   = model_cfg["client"]
    model_id = model_cfg["model_id"]
    result   = {
        "raw_response": "",
        "tokens_in"   : 0,
        "tokens_out"  : 0,
        "latency_ms"  : 0,
        "error"       : None
    }

    max_tokens = model_cfg.get("max_tokens", 4096)
    start  = time.time()

    try:
        if api == "anthropic":
            kwargs = dict(
                model      = model_id,
                max_tokens = max_tokens,
                messages   = [{"role": "user", "content": prompt}]
            )
            if not model_cfg.get("skip_temperature", False):
                kwargs["temperature"] = temperature
            full_text = []
            tokens_in = 0
            tokens_out = 0
            with client.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    full_text.append(text)
                final_msg = stream.get_final_message()
                tokens_in = final_msg.usage.input_tokens
                tokens_out = final_msg.usage.output_tokens
            result["raw_response"] = "".join(full_text)
            result["tokens_in"] = tokens_in
            result["tokens_out"] = tokens_out

        elif api == "gemini":
            gen_config_kwargs = {"max_output_tokens": max_tokens}
            if not model_cfg.get("skip_temperature", False):
                gen_config_kwargs["temperature"] = temperature
            resp = client.models.generate_content(
                model    = model_id,
                contents = prompt,
                config   = genai_types.GenerateContentConfig(**gen_config_kwargs)
            )
            result["raw_response"] = resp.text
            if hasattr(resp, "usage_metadata"):
                result["tokens_in"]  = resp.usage_metadata.prompt_token_count or 0
                result["tokens_out"] = resp.usage_metadata.candidates_token_count or 0

        elif api == "mistral":
            kwargs = dict(
                model      = model_id,
                max_tokens = max_tokens,
                messages   = [{"role": "user", "content": prompt}],
                stream     = False
            )
            if not model_cfg.get("skip_temperature", False):
                kwargs["temperature"] = temperature
            for attempt in range(5):
                try:
                    resp = client.chat.complete(**kwargs)
                    result["raw_response"] = resp.choices[0].message.content
                    result["tokens_in"]    = resp.usage.prompt_tokens
                    result["tokens_out"]   = resp.usage.completion_tokens
                    break
                except Exception as mistral_e:
                    err = str(mistral_e)
                    if "429" in err or "rate_limited" in err or "1300" in err:
                        wait = 60 * (attempt + 1)
                        log.warning(f"  Mistral 429 — waiting {wait}s retry {attempt+1}/5")
                        time.sleep(wait)
                    elif "503" in err or "upstream" in err or "overflow" in err:
                        wait = 30 * (attempt + 1)
                        log.warning(f"  Mistral 503 — waiting {wait}s retry {attempt+1}/5")
                        time.sleep(wait)
                    else:
                        log.warning(f"  Mistral attempt {attempt+1}/5 failed: {mistral_e}")
                        time.sleep(2 ** attempt)
                    if attempt == 4:
                        raise mistral_e

        else:
            max_tokens_param = model_cfg.get("max_tokens_param", "max_tokens")
            kwargs = {
                "model"         : model_id,
                max_tokens_param: max_tokens,
                "messages"      : [{"role": "user", "content": prompt}]
            }
            if model_cfg["type"] != "Reasoning" and not model_cfg.get("skip_temperature", False):
                kwargs["temperature"] = temperature
            resp = client.chat.completions.create(**kwargs)
            result["raw_response"] = resp.choices[0].message.content
            result["tokens_in"]    = resp.usage.prompt_tokens
            result["tokens_out"]   = resp.usage.completion_tokens

    except Exception as e:
        result["error"] = str(e)
        log.warning(f"  API error ({model_cfg['key']}): {e}")

    result["latency_ms"] = int((time.time() - start) * 1000)
    return result

def evaluate_tc(
    tc_name     : str,
    model_cfg   : dict,
    temperature : float
) -> list:
    try:
        prompt = build_prompt(tc_name)
    except Exception as e:
        log.error(f"  Failed to build prompt for {tc_name}: {e}")
        return [{
            "tc_name"       : tc_name,
            "model"         : model_cfg["key"],
            "prompt_type"   : PROMPT_TYPE,
            "run"           : run_idx,
            "temperature" : None if (model_cfg.get("skip_temperature", False) or
                         model_cfg["type"] == "Reasoning") else temperature,
            "timestamp"     : datetime.utcnow().isoformat(),
            "wcd_values"    : None,
            "confidence"    : None,
            "flow_profile"  : None,
            "reason"        : None,
            "parse_note"    : "prompt_build_failed",
            "invalid"       : True,
            "refusal"       : False,
            "tokens_in"     : 0,
            "tokens_out"    : 0,
            "total_tokens"  : 0,
            "cost_usd"      : 0.0,
            "latency_ms"    : 0,
            "response_chars": 0,
            "raw_response"  : "",
            "api_error"     : str(e),
        } for run_idx in range(1, RUNS_PER_TC + 1)]

    runs = []

    for run_idx in range(1, RUNS_PER_TC + 1):
        api_result = call_model(model_cfg, prompt, temperature)
        parsed     = parse_response(api_result["raw_response"])
        cost       = compute_cost(
            model_cfg["pricing_key"],
            api_result["tokens_in"],
            api_result["tokens_out"]
        )

        record = {
            "tc_name"       : tc_name,
            "model"         : model_cfg["key"],
            "prompt_type"   : PROMPT_TYPE,
            "run"           : run_idx,
            "temperature" : None if (model_cfg.get("skip_temperature", False) or
                         model_cfg["type"] == "Reasoning") else temperature,
            "timestamp"     : datetime.utcnow().isoformat(),
            "wcd_values"    : parsed["wcd_values"],
            "confidence"    : parsed["confidence"],
            "flow_profile"  : parsed["flow_profile"],
            "reason"        : parsed["reason"],
            "parse_note"    : parsed.get("parse_note", ""),
            "invalid"       : parsed["invalid"],
            "refusal"       : parsed["refusal"],
            "tokens_in"     : api_result["tokens_in"],
            "tokens_out"    : api_result["tokens_out"],
            "total_tokens"  : api_result["tokens_in"] + api_result["tokens_out"],
            "cost_usd"      : round(cost, 8),
            "latency_ms"    : api_result["latency_ms"],
            "response_chars": len(api_result["raw_response"]),
            "raw_response"  : api_result["raw_response"],
            "api_error"     : api_result["error"],
        }
        runs.append(record)

        wcd_summary = parsed["wcd_values"] if parsed["wcd_values"] else f"INVALID({parsed['parse_note']})"
        log.info(
            f"  {tc_name} run {run_idx} | {model_cfg['key']} | "
            f"WCD={wcd_summary} | "
            f"conf={parsed['confidence']} | "
            f"lat={api_result['latency_ms']}ms"
        )
        time.sleep(1.0)

    return runs


def compute_summary(
    model_cfg   : dict,
    all_runs    : list,
    temperature : float
) -> dict:
    total_runs = len(all_runs)
    total_tcs  = len(set(r["tc_name"] for r in all_runs))

    tc_groups = {}
    for r in all_runs:
        tc_groups.setdefault(r["tc_name"], []).append(r)

    all_consistency = []
    for tc_name, runs in tc_groups.items():
        tc_consistency = compute_wcd_consistency(runs)
        if tc_consistency:
            all_consistency.extend(tc_consistency.values())

    valid      = [r for r in all_runs if not r["invalid"] and not r["refusal"]]
    invalid    = [r for r in all_runs if r["invalid"]]
    refused    = [r for r in all_runs if r["refusal"]]
    api_errors = [r for r in all_runs if r["api_error"]]

    parse_notes = Counter(r.get("parse_note", "") for r in all_runs)

    latencies  = [r["latency_ms"] for r in all_runs if not r["api_error"]]
    costs      = [r["cost_usd"]   for r in all_runs]
    tokens_in  = [r["tokens_in"]  for r in all_runs]
    tokens_out = [r["tokens_out"] for r in all_runs]
    confs      = [r["confidence"] for r in all_runs if r["confidence"] is not None]

    return {
        "model"              : model_cfg["key"],
        "organization"       : model_cfg["org"],
        "type"               : model_cfg["type"],
        "weights"            : model_cfg["weights"],
        "prompt_type"        : PROMPT_TYPE,
        "temperature": None if (model_cfg.get("skip_temperature", False) or
                                model_cfg["type"] == "Reasoning") else temperature,
        "runs_per_tc"        : RUNS_PER_TC,
        "timestamp"          : datetime.utcnow().isoformat(),
        "total_tcs"          : total_tcs,
        "total_runs"         : total_runs,
        "valid_runs"         : len(valid),
        "invalid_runs"       : len(invalid),
        "refusals"           : len(refused),
        "failed_api_calls"   : len(api_errors),
        "invalid_rate"       : round(len(invalid) / total_runs, 4) if total_runs else 0,
        "refusal_rate"       : round(len(refused) / total_runs, 4) if total_runs else 0,
        "parse_notes"        : dict(parse_notes),
        "avg_consistency"    : round(mean(all_consistency), 4) if all_consistency else None,
        "min_consistency"    : round(min(all_consistency), 4)  if all_consistency else None,
        "mae_us"             : None,
        "mape_percent"       : None,
        "exact_match_rate"   : None,
        "avg_confidence"     : round(mean(confs), 4) if confs else None,
        "total_tokens_in"    : sum(tokens_in),
        "total_tokens_out"   : sum(tokens_out),
        "total_cost_usd"     : round(sum(costs), 6),
        "avg_cost_per_run"   : round(mean(costs), 8) if costs else 0,
        "avg_latency_ms"     : round(mean(latencies), 1) if latencies else 0,
        "max_latency_ms"     : max(latencies) if latencies else 0,
        "total_time_seconds" : round(sum(latencies) / 1000, 2) if latencies else 0,
    }


def evaluate_model(
    model_cfg   : dict,
    tc_list     : list,
    temperature : float
) -> tuple:

    skip_temp = (
            model_cfg.get("skip_temperature", False) or
            model_cfg["type"] == "Reasoning"
    )
    temp_label = "tempDefault" if skip_temp else f"temp{temperature}".replace(".", "_")
    log.info(f"{'='*70}")
    log.info(f"Model : {model_cfg['key']} | Prompt: {PROMPT_TYPE} | Temp: {temperature}")
    log.info(f"TCs   : {len(tc_list)} | Runs per TC: {RUNS_PER_TC} | Total calls: {len(tc_list) * RUNS_PER_TC}")
    log.info(f"{'='*70}")

    all_runs    = []
    model_start = time.time()

    for tc_name in tc_list:
        runs = evaluate_tc(tc_name, model_cfg, temperature)
        all_runs.extend(runs)

    model_elapsed = round(time.time() - model_start, 2)
    log.info(f"  Completed in {model_elapsed}s")

    raw_path = os.path.join(
        OPEN_END_DIR,
        f"raw_openend_{model_cfg['key']}_{PROMPT_TYPE}_{temp_label}.json"
    )
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(all_runs, f, indent=2, ensure_ascii=False)
    log.info(f"  Raw saved: {raw_path}")

    summary = compute_summary(model_cfg, all_runs, temperature)
    summary["wall_time_seconds"] = model_elapsed

    summary_path = os.path.join(
        OPEN_END_DIR,
        f"summary_openend_{model_cfg['key']}_{PROMPT_TYPE}_{temp_label}.json"
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info(f"  Summary saved: {summary_path}")

    return all_runs, summary


def main():
    if not os.path.isdir(TC_DIR):
        log.error(f"TC directory not found: {TC_DIR}")
        return

    missing = [tc for tc in TC_LIST
               if not os.path.exists(os.path.join(TC_DIR, f"{tc}_topo.txt"))]
    if missing:
        log.warning(f"Missing TC folders: {missing}")
    tc_list = [tc for tc in TC_LIST if tc not in missing]

    if not tc_list:
        log.error("No valid TC folders found. Exiting.")
        return

    log.info(f"Prompt type    : {PROMPT_TYPE}")
    log.info(f"Test cases     : {len(tc_list)}")
    log.info(f"Runs per TC    : {RUNS_PER_TC}")
    log.info(f"Models         : {len(MODELS)}")
    log.info(f"Temp settings  : {TEMPERATURE_WITHOUT} (without) | {TEMPERATURE_WITH} (with)")
    log.info(f"Output dir     : {OPEN_END_DIR}")
    log.info(f"NOTE: WCD accuracy computed by error_calculation.py")

    temp_one = sum(1 for m in MODELS if m.get("skip_temperature", False) or m["type"] == "Reasoning")
    temp_two = len(MODELS) - temp_one
    total_calls = len(tc_list) * RUNS_PER_TC * (temp_one + temp_two * 2)
    log.info(f"Models once (no temp): {temp_one} | Models twice (with/without): {temp_two}")
    log.info(f"Total API calls: {total_calls}")

    all_summaries = []

    for model_cfg in MODELS:
        skip_temp = (
            model_cfg.get("skip_temperature", False) or
            model_cfg["type"] == "Reasoning"
        )
        temperatures_to_run = [TEMPERATURE_WITHOUT] if skip_temp else [TEMPERATURE_WITHOUT, TEMPERATURE_WITH]

        if skip_temp:
            log.info(f"  {model_cfg['key']} — no temperature support, running once only")

        for temperature in temperatures_to_run:
            try:
                _, summary = evaluate_model(model_cfg, tc_list, temperature)
                all_summaries.append(summary)
            except Exception as e:
                log.error(f"Failed: {model_cfg['key']} temp={temperature} | {e}", exc_info=True)

    full_path = os.path.join(OPEN_END_DIR, f"full_summary_openend_{PROMPT_TYPE}.json")
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)
    log.info(f"Full summary saved: {full_path}")

    log.info(f"\n{'='*80}")
    log.info(f"{'Model':<28} {'Temp':>5} {'Valid':>6} {'Invalid':>8} {'Consist':>8} {'Cost$':>8} {'Lat(ms)':>9}")
    log.info(f"{'-'*80}")
    for s in sorted(all_summaries, key=lambda x: x["valid_runs"], reverse=True):
        consist = f"{s['avg_consistency']:.2f}" if s["avg_consistency"] is not None else "N/A"
        log.info(
            f"{s['model']:<28} "
            f"{str(s['temperature']) if s['temperature'] is not None else 'Def':>5} "
            f"{s['valid_runs']:>6} "
            f"{s['invalid_runs']:>8} "
            f"{consist:>8} "
            f"{s['total_cost_usd']:>8.4f} "
            f"{s['avg_latency_ms']:>9.0f}"
        )
    log.info(f"{'='*80}")
    log.info(f"Evaluation complete. Run error_calculation.py for MAE and MAPE values.")


if __name__ == "__main__":
    main()