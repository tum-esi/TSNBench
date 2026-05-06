import os
import json
import logging
from collections import defaultdict

MCQA_DIR   = "output/results"
CBS_DIR    = "../open_end_evaluation/output/CBS"
CQF_DIR    = "../open_end_evaluation/output/CQF"
CBS_SCORED_DIR = "../open_end_evaluation/output/CBS/score"
CQF_SCORED_DIR = "../open_end_evaluation/output/CQF/score"

MODEL_NAMES = {
    "grok-4-1-fast-reasoning"        : "Grok 4.1 Fast",
    "grok-4-1-fast-non-reasoning"    : "Grok 4.1 Fast (NR)",
    "deepseek-chat"                  : "DeepSeek-V3.2 (NT)",
    "gpt-4o-2024-08-06"              : "GPT-4o",
    "gpt-4o-mini-2024-07-18"         : "GPT-4o mini",
    "Llama-3.3-70B-Instruct"         : "Llama 3.3",
    "mistral-medium-2508"            : "Mistral Medium 3.1",
    "mistral-large-2512"             : "Mistral Large 3",
    "claude-sonnet-4-5-20250929"     : "Claude Sonnet 4.5",
    "o3-2025-04-16"                  : "o3",
    "gpt-5-2025-08-07"               : "GPT-5",
    "deepseek-reasoner"              : "DeepSeek-V3.2 (T)",
    "gemini-2.5-flash"               : "Gemini 2.5 Flash",
    "Llama-3.2-1B"                   : "Llama 3.2 1B",
    "Qwen3-8B"                       : "Qwen3 8B",
    "ministral-8b-2512"              : "Ministral 3 8B",
}

MODEL_PRICING = {
    "grok-4-1-fast-reasoning"       : {"input":  0.20, "output": 0.50},
    "grok-4-1-fast-non-reasoning"   : {"input":  0.20, "output": 0.50},
    "o3-2025-04-16"                 : {"input": 2.00, "output": 8.00},
    "gpt-4o-2024-08-06"             : {"input":  2.50, "output": 10.00},
    "gpt-4o-mini-2024-07-18"        : {"input":  0.15, "output":  0.60},
    "gpt-5-2025-08-07"              : {"input":  1.25, "output":  10.00},
    "deepseek-chat"                 : {"input":  0.028, "output":  0.42},
    "deepseek-reasoner"             : {"input":  0.028, "output":  0.42},
    "gemini-2.5-flash"              : {"input":  0.30, "output":  2.50},
    "claude-sonnet-4-5-20250929"    : {"input":  3.00, "output":  15.00},
    "Llama-3.3-70B-Instruct"        : {"input":  0.59, "output":  0.79},
    "mistral-medium-2508"           : {"input":  0.40, "output":  2.00},
    "mistral-large-2512"            : {"input":  0.50, "output":  1.50},
    "Llama-3.2-1B"               : {"input": 0.10,  "output": 0.10},
    "Qwen3-8B"                      : {"input": 0.07,  "output": 0.18},
    "ministral-8b-2512"             : {"input":  0.15, "output":  0.15}
}


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler("cost_latency_table.log", encoding="utf-8", mode="w"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


def check_enough_tcs(scored_dir: str, model_key: str,
                     prompt_type: str, temp_label: str) -> bool:
    if not temp_label:
        return False

    scored_fname = f"scored_openend_{model_key}_{prompt_type}_{temp_label}.json"
    scored_path  = os.path.join(scored_dir, scored_fname)

    if not os.path.exists(scored_path):
        return False

    with open(scored_path, encoding="utf-8") as f:
        data = json.load(f)

    return data.get("enough_tcs", False)


def aggregate_raw(fpath: str, model_key: str) -> tuple:
    if not os.path.exists(fpath):
        return None, None, 0

    with open(fpath, encoding="utf-8") as f:
        records = json.load(f)

    if not records:
        return 0.0, 0, 0

    pricing     = MODEL_PRICING.get(model_key, {"input": 0.0, "output": 0.0})
    input_rate  = pricing["input"]  / 1_000_000
    output_rate = pricing["output"] / 1_000_000

    total_cost    = 0.0
    total_latency = 0
    valid_count   = 0

    for r in records:
        if r.get("api_error"):
            continue
        if r.get("invalid_response"):
            continue
        if r.get("invalid"):
            continue
        if "wcd_values" in r and r.get("wcd_values") is None:
            continue

        tokens_in  = r.get("tokens_in",  0) or 0
        tokens_out = r.get("tokens_out", 0) or 0
        total_cost    += tokens_in * input_rate + tokens_out * output_rate
        total_latency += r.get("latency_ms", 0) or 0
        valid_count   += 1

    if valid_count == 0:
        return None, None, 0

    return round(total_cost, 4), int(total_latency), valid_count


def find_raw_file(directory: str, model_key: str,
                  prompt_type: str = None) -> tuple:
    if not os.path.isdir(directory):
        return None, None

    files = os.listdir(directory)

    if prompt_type:
        prefix  = f"raw_openend_{model_key}_{prompt_type}_"
        matches = [f for f in files if f.startswith(prefix) and f.endswith(".json")]
    else:
        prefix  = f"raw_{model_key}_"
        matches = [f for f in files if f.startswith(prefix) and f.endswith(".json")]

    if not matches:
        return None, None

    priority = {"tempDefault": 0, "temp0_0": 1, "temp0_7": 2}

    def sort_key(fname):
        for label, rank in priority.items():
            if label in fname:
                return rank
        return 99

    matches.sort(key=sort_key)
    best = matches[0]

    temp_label = "unknown"
    for label in ["tempDefault", "temp0_0", "temp0_7"]:
        if label in best:
            temp_label = label
            break

    return os.path.join(directory, best), temp_label


def main():
    all_model_keys = set()

    if os.path.isdir(MCQA_DIR):
        for fname in os.listdir(MCQA_DIR):
            if fname.startswith("raw_") and fname.endswith(".json"):
                stem = fname.replace("raw_", "").replace(".json", "")
                for label in ["_tempDefault", "_temp0_0", "_temp0_7"]:
                    if stem.endswith(label):
                        all_model_keys.add(stem[:-len(label)])
                        break

    for d, prompt in [(CBS_DIR, "CBS"), (CQF_DIR, "CQF")]:
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            if (fname.startswith("raw_openend_") and
                    f"_{prompt}_" in fname and fname.endswith(".json")):
                stem  = fname.replace("raw_openend_", "").replace(".json", "")
                parts = stem.split(f"_{prompt}_")
                if len(parts) >= 1:
                    all_model_keys.add(parts[0])

    log.info(f"Found {len(all_model_keys)} unique model keys\n")

    rows = []

    for model_key in sorted(all_model_keys):
        display = MODEL_NAMES.get(model_key, model_key)

        mcqa_path, mcqa_temp = find_raw_file(MCQA_DIR, model_key)
        if mcqa_path:
            mcqa_cost, mcqa_lat, mcqa_n = aggregate_raw(mcqa_path, model_key)
        else:
            mcqa_cost, mcqa_lat, mcqa_n = None, None, 0

        cbs_path, cbs_temp = find_raw_file(CBS_DIR, model_key, "CBS")
        if cbs_path and check_enough_tcs(CBS_SCORED_DIR, model_key, "CBS", cbs_temp):
            cbs_cost, cbs_lat, cbs_n = aggregate_raw(cbs_path, model_key)
        else:
            cbs_cost, cbs_lat, cbs_n = None, None, 0

        cqf_path, cqf_temp = find_raw_file(CQF_DIR, model_key, "CQF")
        if cqf_path and check_enough_tcs(CQF_SCORED_DIR, model_key, "CQF", cqf_temp):
            cqf_cost, cqf_lat, cqf_n = aggregate_raw(cqf_path, model_key)
        else:
            cqf_cost, cqf_lat, cqf_n = None, None, 0

        temp_display = mcqa_temp or cbs_temp or cqf_temp or "unknown"
        temp_display = (temp_display
                        .replace("temp", "")
                        .replace("_", ".")
                        .replace("Default", "Def"))

        rows.append({
            "display" : display,
            "temp"    : temp_display,
            "mcqa_cost": mcqa_cost, "mcqa_lat": mcqa_lat, "mcqa_n": mcqa_n,
            "cbs_cost" : cbs_cost,  "cbs_lat" : cbs_lat,  "cbs_n" : cbs_n,
            "cqf_cost" : cqf_cost,  "cqf_lat" : cqf_lat,  "cqf_n" : cqf_n,
        })

        log.info(
            f"  {model_key:<40} "
            f"MCQA: n={mcqa_n} cost=${mcqa_cost} lat={mcqa_lat}ms | "
            f"CBS: n={cbs_n} cost=${cbs_cost} lat={cbs_lat}ms | "
            f"CQF: n={cqf_n} cost=${cqf_cost} lat={cqf_lat}ms"
        )

    def col_max(key):
        return max((r[key] for r in rows if r[key] is not None), default=0)

    max_mcqa_c = col_max("mcqa_cost")
    max_mcqa_l = col_max("mcqa_lat")
    max_cbs_c  = col_max("cbs_cost")
    max_cbs_l  = col_max("cbs_lat")
    max_cqf_c  = col_max("cqf_cost")
    max_cqf_l  = col_max("cqf_lat")

    def fmt(val, max_val, is_int=False):
        if val is None:
            return "--"
        s = f"{val:,}" if is_int else f"{val:.4f}"
        if val == max_val:
            return f"\\cellcolor{{gray!20}}\\textbf{{{s}}}"
        return s

    log.info(f"\n\n% {'='*100}")
    log.info("% LaTeX table — Cost and Latency per Model")
    log.info("% Model (temp) & MCQA Cost (\\$) & MCQA Lat (ms) & CBS Cost (\\$) & CBS Lat (ms) & CQF Cost (\\$) & CQF Lat (ms) \\\\")
    log.info(f"% {'='*100}\n")

    for r in rows:
        name   = f"{r['display']} ({r['temp']})"
        mcqa_c = fmt(r["mcqa_cost"], max_mcqa_c)
        mcqa_l = fmt(r["mcqa_lat"],  max_mcqa_l, is_int=True)
        cbs_c  = fmt(r["cbs_cost"],  max_cbs_c)
        cbs_l  = fmt(r["cbs_lat"],   max_cbs_l,  is_int=True)
        cqf_c  = fmt(r["cqf_cost"],  max_cqf_c)
        cqf_l  = fmt(r["cqf_lat"],   max_cqf_l,  is_int=True)

        log.info(f"{name} & {mcqa_c} & {mcqa_l} & {cbs_c} & {cbs_l} & {cqf_c} & {cqf_l} \\\\")

    log.info(f"\n% Saved to: cost_latency_table.log")


if __name__ == "__main__":
    main()
