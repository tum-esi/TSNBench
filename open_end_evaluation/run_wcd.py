"""Runs the CBS or CQF WCD questions on the models in config.py."""

import argparse
import json
import logging
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from statistics import mean

from dotenv import load_dotenv
from google.genai import types as genai_types

load_dotenv()
from config import MODELS, TEMPERATURE_WITHOUT, compute_cost  # noqa: E402

log = logging.getLogger("open_end")

NOISY_LOGGERS = ["httpx", "httpcore", "openai", "anthropic", "mistralai", "google", "urllib3"]

# error text, wait (s)
RETRY_WAITS = [
    (("429", "rate_limited", "1300"), 60),
    (("503", "upstream", "overflow"), 30),
]


def build_prompt(tc_dir, tc, mechanism, variant):
    paths = [os.path.join(tc_dir, f"{tc}_{part}.txt") for part in ("topo", "flows", "route")]
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(p)
    if mechanism == "CQF":
        from prompts_cqf import CQF as make_prompt
    else:
        from prompts_cbs import CBS as make_prompt
    return make_prompt(*paths, variant=variant)


# parser used during the runs (the evaluation re-parses with wcd_parser.py)
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

PLAIN_TEXT_PATTERNS = [
    r"['\"]?(F\d+)['\"]?\s*[:=]\s*(-?\d+(?:\.\d+)?)",
    r"(?:WCD|delay)\s+(?:of|for)?\s*(F\d+)\s+is\s+(-?\d+(?:\.\d+)?)",
    r"(F\d+)[^:=\d]+(-?\d+(?:\.\d+)?)\s*(?:µs|us|microseconds)",
]

REFUSAL_PHRASES = [
    "i cannot", "i can't", "i am not able", "i'm not able", "unable to",
    "cannot answer", "not enough information", "insufficient information",
    "cannot calculate", "not possible to determine",
]

EMPTY_PARSE = {"wcd_values": None, "confidence": None, "flow_profile": None,
               "reason": None, "invalid": True, "refusal": False, "parse_note": ""}


def _to_float(val):
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, dict):
        for v in val.values():
            f = _to_float(v)
            if f is not None:
                return f
        return None
    if isinstance(val, str):
        m = NUM_RE.search(val)
        return float(m.group()) if m else None
    return None


def _find_wcd_json(text):
    decoder = json.JSONDecoder()
    found, pos = None, 0
    while True:
        idx = text.find("{", pos)
        if idx == -1:
            return found
        try:
            obj, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            pos = idx + 1
            continue
        if isinstance(obj, dict) and "WCD_us" in obj:
            found, pos = obj, end
        else:
            pos = idx + 1


def _find_wcd_block(text):
    m = re.search(r'"WCD_us"\s*:\s*\{([^{}]*)\}', text)
    if not m:
        return {}
    return {fid.upper(): float(num)
            for fid, num in re.findall(r'"?(F\d+)"?\s*:\s*"?(-?\d+(?:\.\d+)?)', m.group(1))}


def parse_response(raw):
    if not raw:
        return {**EMPTY_PARSE, "parse_note": "empty_response"}
    clean = raw.strip()

    parsed = _find_wcd_json(clean)
    if parsed is not None:
        wcd_raw = parsed.get("WCD_us") or {}
        if not isinstance(wcd_raw, dict):
            wcd_raw = {}
        wcd = {str(fid).upper(): _to_float(val) for fid, val in wcd_raw.items()}
        bad = [fid for fid, val in wcd_raw.items()
               if not isinstance(val, (int, float)) or isinstance(val, bool)]
        conf = parsed.get("confidence")
        if isinstance(conf, (int, float)) and not isinstance(conf, bool):
            conf = round(max(0.0, min(1.0, float(conf))), 4)
        else:
            conf = None
        if not wcd:
            note = "json_parsed_no_wcd_field"
        else:
            note = f"wcd_not_numeric:{bad}" if bad else "ok"
        return {"wcd_values": wcd or None, "confidence": conf,
                "flow_profile": parsed.get("flow_profile"), "reason": parsed.get("reason"),
                "invalid": not wcd or all(v is None for v in wcd.values()),
                "refusal": False, "parse_note": note}

    wcd = _find_wcd_block(clean)
    if wcd:
        cm = re.search(r'"confidence"\s*:\s*(-?\d+(?:\.\d+)?)', clean)
        return {**EMPTY_PARSE, "wcd_values": wcd, "confidence": float(cm.group(1)) if cm else None,
                "invalid": False, "parse_note": "wcd_block_only"}

    for pattern in PLAIN_TEXT_PATTERNS:
        for m in re.finditer(pattern, clean, re.IGNORECASE):
            wcd[m.group(1).upper()] = float(m.group(2))
    if wcd:
        return {**EMPTY_PARSE, "wcd_values": wcd, "invalid": False,
                "parse_note": "extracted_from_plain_text"}

    if any(p in clean.lower() for p in REFUSAL_PHRASES):
        return {**EMPTY_PARSE, "invalid": False, "refusal": True, "parse_note": "refusal_detected"}
    return {**EMPTY_PARSE, "parse_note": "parse_failed_no_wcd_extractable"}


# each call returns (text, tokens_in, tokens_out)
def with_retries(fn, attempts=5):
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as e:
            if attempt == attempts - 1:
                raise
            err = str(e)
            wait = next((base * (attempt + 1) for markers, base in RETRY_WAITS
                         if any(mk in err for mk in markers)), 2 ** attempt)
            log.warning("retry %d/%d in %ds: %s", attempt + 1, attempts, wait, e)
            time.sleep(wait)


def _messages(prompt):
    return [{"role": "user", "content": prompt}]


def _call_anthropic(cfg, prompt, temperature, max_tokens):
    kwargs = {"model": cfg["model_id"], "max_tokens": max_tokens, "messages": _messages(prompt)}
    if not cfg.get("skip_temperature"):
        kwargs["temperature"] = temperature
    # long requests need streaming
    with cfg["client"].messages.stream(**kwargs) as stream:
        text = "".join(stream.text_stream)
        usage = stream.get_final_message().usage
    return text, usage.input_tokens, usage.output_tokens


def _call_gemini(cfg, prompt, temperature, max_tokens):
    gen_config = {"max_output_tokens": max_tokens}
    if not cfg.get("skip_temperature"):
        gen_config["temperature"] = temperature
    resp = cfg["client"].models.generate_content(
        model=cfg["model_id"], contents=prompt,
        config=genai_types.GenerateContentConfig(**gen_config))
    usage = getattr(resp, "usage_metadata", None)
    if usage is None:
        return resp.text, 0, 0
    return resp.text, usage.prompt_token_count or 0, usage.candidates_token_count or 0


def _call_mistral(cfg, prompt, temperature, max_tokens):
    kwargs = {"model": cfg["model_id"], "max_tokens": max_tokens,
              "messages": _messages(prompt), "stream": False}
    if not cfg.get("skip_temperature"):
        kwargs["temperature"] = temperature
    resp = with_retries(lambda: cfg["client"].chat.complete(**kwargs))
    return resp.choices[0].message.content, resp.usage.prompt_tokens, resp.usage.completion_tokens


def _call_deepseek(cfg, prompt, temperature, max_tokens):
    thinking = cfg.get("thinking_enabled", False)
    kwargs = {"model": cfg["model_id"], "max_tokens": max_tokens, "messages": _messages(prompt),
              "extra_body": {"thinking": {"type": "enabled" if thinking else "disabled"}}}
    if not cfg.get("skip_temperature"):
        kwargs["temperature"] = temperature
    if thinking:
        kwargs["reasoning_effort"] = cfg.get("reasoning_effort", "high")
    resp = with_retries(lambda: cfg["client"].chat.completions.create(**kwargs))
    return resp.choices[0].message.content, resp.usage.prompt_tokens, resp.usage.completion_tokens


def _call_openai(cfg, prompt, temperature, max_tokens):
    kwargs = {"model": cfg["model_id"], cfg.get("max_tokens_param", "max_tokens"): max_tokens,
              "messages": _messages(prompt)}
    if cfg["type"] != "Reasoning" and not cfg.get("skip_temperature"):
        kwargs["temperature"] = temperature
    resp = cfg["client"].chat.completions.create(**kwargs)
    return resp.choices[0].message.content, resp.usage.prompt_tokens, resp.usage.completion_tokens


CALLERS = {"anthropic": _call_anthropic, "gemini": _call_gemini,
           "mistral": _call_mistral, "deepseek": _call_deepseek}


def call_model(cfg, prompt, temperature):
    call = CALLERS.get(cfg["api"], _call_openai)
    start = time.time()
    try:
        text, tokens_in, tokens_out = call(cfg, prompt, temperature, cfg.get("max_tokens", 4096))
        result = {"raw_response": text or "", "tokens_in": tokens_in or 0,
                  "tokens_out": tokens_out or 0, "error": None}
    except Exception as e:
        log.warning("API error (%s): %s", cfg["key"], e)
        result = {"raw_response": "", "tokens_in": 0, "tokens_out": 0, "error": str(e)}
    result["latency_ms"] = int((time.time() - start) * 1000)
    return result


def uses_default_temperature(cfg):
    return cfg.get("skip_temperature", False) or cfg["type"] == "Reasoning"


def run_record(tc, cfg, run, args, parsed, api, cost):
    return {
        "tc_name": tc,
        "model": cfg["key"],
        "prompt_type": args.mechanism,
        "prompt_variant": args.variant,
        "run": run,
        "temperature": None if uses_default_temperature(cfg) else args.temperature,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **parsed,
        "tokens_in": api["tokens_in"],
        "tokens_out": api["tokens_out"],
        "total_tokens": api["tokens_in"] + api["tokens_out"],
        "cost_usd": round(cost, 8),
        "latency_ms": api["latency_ms"],
        "response_chars": len(api["raw_response"]),
        "raw_response": api["raw_response"],
        "api_error": api["error"],
    }


def evaluate_tc(tc, cfg, args):
    try:
        prompt = build_prompt(args.tc_dir, tc, args.mechanism, args.variant)
    except Exception as e:
        log.error("%s: could not build prompt: %s", tc, e)
        api = {"raw_response": "", "tokens_in": 0, "tokens_out": 0, "latency_ms": 0, "error": str(e)}
        parsed = {**EMPTY_PARSE, "parse_note": "prompt_build_failed"}
        return [run_record(tc, cfg, run, args, parsed, api, 0.0) for run in range(1, args.runs + 1)]

    records = []
    for run in range(1, args.runs + 1):
        api = call_model(cfg, prompt, args.temperature)
        parsed = parse_response(api["raw_response"])
        cost = compute_cost(cfg["pricing_key"], api["tokens_in"], api["tokens_out"])
        records.append(run_record(tc, cfg, run, args, parsed, api, cost))
        time.sleep(1.0)
    return records


def consistency_per_flow(runs, runs_per_tc):
    flows = {f for r in runs for f in (r.get("wcd_values") or {})}
    out = {}
    for f in flows:
        values = [round(v, 1) for r in runs
                  if (v := (r.get("wcd_values") or {}).get(f)) is not None]
        out[f] = round(Counter(values).most_common(1)[0][1] / runs_per_tc, 4) if values else 0.0
    return out


def summarise(cfg, runs, args, wall_time):
    by_tc = {}
    for r in runs:
        by_tc.setdefault(r["tc_name"], []).append(r)
    consistency = [c for tc_runs in by_tc.values()
                   for c in consistency_per_flow(tc_runs, args.runs).values()]

    n = len(runs)
    n_invalid = sum(r["invalid"] for r in runs)
    n_refused = sum(r["refusal"] for r in runs)
    latencies = [r["latency_ms"] for r in runs if not r["api_error"]]
    costs = [r["cost_usd"] for r in runs]
    confs = [r["confidence"] for r in runs if r["confidence"] is not None]

    return {
        "model": cfg["key"],
        "organization": cfg["org"],
        "type": cfg["type"],
        "weights": cfg["weights"],
        "prompt_type": args.mechanism,
        "prompt_variant": args.variant,
        "temperature": None if uses_default_temperature(cfg) else args.temperature,
        "runs_per_tc": args.runs,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_tcs": len(by_tc),
        "total_runs": n,
        "valid_runs": sum(not r["invalid"] and not r["refusal"] for r in runs),
        "invalid_runs": n_invalid,
        "refusals": n_refused,
        "failed_api_calls": sum(bool(r["api_error"]) for r in runs),
        "invalid_rate": round(n_invalid / n, 4) if n else 0,
        "refusal_rate": round(n_refused / n, 4) if n else 0,
        "parse_notes": dict(Counter(r.get("parse_note", "") for r in runs)),
        "avg_consistency": round(mean(consistency), 4) if consistency else None,
        "min_consistency": round(min(consistency), 4) if consistency else None,
        "avg_confidence": round(mean(confs), 4) if confs else None,
        "total_tokens_in": sum(r["tokens_in"] for r in runs),
        "total_tokens_out": sum(r["tokens_out"] for r in runs),
        "total_cost_usd": round(sum(costs), 6),
        "avg_cost_per_run": round(mean(costs), 8) if costs else 0,
        "avg_latency_ms": round(mean(latencies), 1) if latencies else 0,
        "max_latency_ms": max(latencies) if latencies else 0,
        "total_time_seconds": round(sum(latencies) / 1000, 2) if latencies else 0,
        "wall_time_seconds": wall_time,
    }


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def evaluate_model(cfg, tcs, args):
    if uses_default_temperature(cfg):
        label = "tempDefault"
    else:
        label = f"temp{args.temperature}".replace(".", "_")
    log.info("%s: %d test cases x %d runs", cfg["key"], len(tcs), args.runs)

    start = time.time()
    runs = [r for tc in tcs for r in evaluate_tc(tc, cfg, args)]
    wall_time = round(time.time() - start, 2)

    stem = f"{cfg['key']}_{args.mechanism}_{label}"
    write_json(os.path.join(args.out_dir, f"raw_openend_{stem}.json"), runs)
    summary = summarise(cfg, runs, args, wall_time)
    write_json(os.path.join(args.out_dir, f"summary_openend_{stem}.json"), summary)
    return summary


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mechanism", required=True, choices=["CBS", "CQF"])
    p.add_argument("--variant", default="closed_book",
                   choices=["closed_book", "formula", "variant_a", "variant_b"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tc-dir", default="input/test_cases/open_ended")
    p.add_argument("--tcs", nargs="+", default=[f"TC{i}" for i in range(1, 101)])
    p.add_argument("--runs", type=int, default=3, help="runs per test case")
    p.add_argument("--temperature", type=float, default=TEMPERATURE_WITHOUT)
    p.add_argument("--models", nargs="+", help="model keys from config.MODELS (default: all)")
    p.add_argument("--print-prompt", action="store_true",
                   help="print the prompt of the first test case and exit")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)

    tcs = [tc for tc in args.tcs if os.path.exists(os.path.join(args.tc_dir, f"{tc}_topo.txt"))]
    missing = [tc for tc in args.tcs if tc not in tcs]
    if missing:
        log.warning("test cases not found in %s: %s", args.tc_dir, missing)
    if not tcs:
        raise SystemExit(f"no test cases found in {args.tc_dir}")

    if args.print_prompt:
        print(build_prompt(args.tc_dir, tcs[0], args.mechanism, args.variant))
        return

    models = [cfg for cfg in MODELS if not args.models or cfg["key"] in args.models]
    os.makedirs(args.out_dir, exist_ok=True)

    summaries = []
    for cfg in models:
        try:
            summaries.append(evaluate_model(cfg, tcs, args))
        except Exception:
            log.exception("%s failed", cfg["key"])
    write_json(os.path.join(args.out_dir, f"full_summary_openend_{args.mechanism}.json"), summaries)


if __name__ == "__main__":
    main()
