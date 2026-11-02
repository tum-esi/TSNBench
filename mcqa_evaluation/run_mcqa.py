"""Runs the MCQA questions on the models in config.py."""

import argparse
import json
import logging
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from statistics import mean, median

from dotenv import load_dotenv
from google.genai import types as genai_types

load_dotenv()
from config import (  # noqa: E402
    MODELS, QUESTIONSET_FILE, RESULTS_DIR, RUNS_PER_QUESTION,
    TEMPERATURE_WITH, TEMPERATURE_WITHOUT, compute_cost,
)

log = logging.getLogger("mcqa")

NOISY_LOGGERS = ["httpx", "httpcore", "openai", "anthropic", "mistralai", "google", "urllib3"]

EVAL_SYSTEM_PROMPT = """You are an expert in Time-Sensitive Networking (TSN).

Answer the following multiple-choice question based solely on your domain knowledge.

STRICT RULES:
- Select exactly one option: A, B, C, D, or E
- Do NOT explain your answer
- Do NOT add any text before or after the JSON

OUTPUT FORMAT — return exactly this JSON and nothing else:
{
  "answer": "B",
  "confidence": 0.85
}

NOTE: answer must be exactly one of: A, B, C, D, E
NOTE: confidence must be a float between 0.0 and 1.0
      where 1.0 = completely certain (highest confidence), 0.0 = complete guess (lowest confidence)
"""

REFUSAL_PHRASES = [
    "i cannot", "i can't", "i am not able", "i'm not able",
    "unable to", "i don't know", "i do not know",
    "cannot answer", "not enough information",
]

LETTERS = ["A", "B", "C", "D", "E"]

# error text, wait (s)
RETRY_WAITS = [
    (("429", "rate_limited", "1300"), 60),
    (("503", "upstream", "overflow"), 30),
]


def build_eval_prompt(question):
    options_text = ""
    for letter in LETTERS:
        if letter in question:
            options_text += f"  {letter}: {question[letter]}\n"
    return f"QUESTION:\n{question['question']}\n\nOPTIONS:\n{options_text}"


def _clamp_confidence(value):
    return round(max(0.0, min(1.0, float(value))), 4)


def parse_eval_response(raw):
    if not raw:
        return {"answer": None, "confidence": None, "refusal": False, "invalid": True}

    clean = raw.strip()
    clean = re.sub(r"^```(?:json)?", "", clean).strip()
    clean = re.sub(r"```$", "", clean).strip()

    if any(p in clean.lower() for p in REFUSAL_PHRASES):
        return {"answer": None, "confidence": None, "refusal": True, "invalid": False}

    try:
        parsed = json.loads(clean)
        if not isinstance(parsed, dict):
            if isinstance(parsed, str) and parsed.strip().upper() in LETTERS:
                return {"answer": parsed.strip().upper(), "confidence": None,
                        "refusal": False, "invalid": False}
            raise json.JSONDecodeError("not a dict", clean, 0)

        answer = str(parsed.get("answer", "")).strip().upper()
        confidence_raw = parsed.get("confidence")
        if isinstance(confidence_raw, (int, float)):
            confidence = _clamp_confidence(confidence_raw)
        elif isinstance(confidence_raw, str):
            try:
                confidence = _clamp_confidence(confidence_raw)
            except ValueError:
                confidence = None
        else:
            confidence = None

        if answer not in LETTERS:
            answer = None
        return {"answer": answer, "confidence": confidence, "refusal": False, "invalid": answer is None}
    except json.JSONDecodeError:
        pass

    match = re.search(r'"answer"\s*:\s*"([A-Ea-e])"', clean)
    if match:
        conf_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', clean)
        confidence = _clamp_confidence(conf_match.group(1)) if conf_match else None
        return {"answer": match.group(1).upper(), "confidence": confidence,
                "refusal": False, "invalid": False}

    match = re.search(r'\b([A-E])\b', clean)
    if match:
        return {"answer": match.group(1).upper(), "confidence": None, "refusal": False, "invalid": False}

    return {"answer": None, "confidence": None, "refusal": False, "invalid": True}


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


def call_model(cfg, system, user, temperature):
    api, client, model_id, max_tokens = cfg["api"], cfg["client"], cfg["model_id"], cfg["max_tokens"]
    use_temperature = not cfg.get("skip_temperature", False)
    result = {"raw_response": "", "tokens_in": 0, "tokens_out": 0, "latency_ms": 0, "error": None}
    start = time.time()

    try:
        if api == "anthropic":
            kwargs = {"model": model_id, "max_tokens": max_tokens, "system": system,
                      "messages": [{"role": "user", "content": user}]}
            if use_temperature:
                kwargs["temperature"] = temperature
            resp = client.messages.create(**kwargs)
            text, tokens_in, tokens_out = resp.content[0].text, resp.usage.input_tokens, resp.usage.output_tokens

        elif api == "gemini":
            gen_config = {"max_output_tokens": max_tokens}
            if use_temperature:
                gen_config["temperature"] = temperature
            resp = client.models.generate_content(
                model=model_id, contents=f"{system}\n\n{user}",
                config=genai_types.GenerateContentConfig(**gen_config))
            usage = getattr(resp, "usage_metadata", None)
            text = resp.text
            tokens_in = (usage.prompt_token_count or 0) if usage else 0
            tokens_out = (usage.candidates_token_count or 0) if usage else 0

        elif api == "mistral":
            kwargs = {"model": model_id, "max_tokens": max_tokens, "stream": False,
                      "messages": [{"role": "system", "content": system},
                                   {"role": "user", "content": user}]}
            if use_temperature:
                kwargs["temperature"] = temperature
            resp = with_retries(lambda: client.chat.complete(**kwargs))
            text, tokens_in, tokens_out = (resp.choices[0].message.content,
                                           resp.usage.prompt_tokens, resp.usage.completion_tokens)

        else:  # OpenAI-compatible APIs
            kwargs = {"model": model_id, cfg.get("max_tokens_param", "max_tokens"): max_tokens,
                      "messages": [{"role": "system", "content": system},
                                   {"role": "user", "content": user}]}
            if cfg["type"] != "Reasoning" and use_temperature:
                kwargs["temperature"] = temperature
            resp = client.chat.completions.create(**kwargs)
            text, tokens_in, tokens_out = (resp.choices[0].message.content,
                                           resp.usage.prompt_tokens, resp.usage.completion_tokens)

        result.update(raw_response=text or "", tokens_in=tokens_in, tokens_out=tokens_out)
    except Exception as e:
        result["error"] = str(e)
        log.warning("API error (%s): %s", cfg["key"], e)

    result["latency_ms"] = int((time.time() - start) * 1000)
    return result


def uses_default_temperature(cfg):
    return cfg.get("skip_temperature", False) or cfg["type"] == "Reasoning"


def evaluate_question(question, cfg, temperature):
    user = build_eval_prompt(question)
    runs = []
    for run in range(1, RUNS_PER_QUESTION + 1):
        api = call_model(cfg, EVAL_SYSTEM_PROMPT, user, temperature)
        parsed = parse_eval_response(api["raw_response"])
        cost = compute_cost(cfg["pricing_key"], api["tokens_in"], api["tokens_out"])
        runs.append({
            "question_id": question.get("question_id"),
            "model": cfg["key"],
            "run": run,
            "temperature": None if uses_default_temperature(cfg) else temperature,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "selected_answer": parsed["answer"],
            "confidence": parsed["confidence"],
            "refusal": parsed["refusal"],
            "invalid_response": parsed["invalid"],
            "tokens_in": api["tokens_in"],
            "tokens_out": api["tokens_out"],
            "total_tokens": api["tokens_in"] + api["tokens_out"],
            "cost_usd": round(cost, 8),
            "latency_ms": api["latency_ms"],
            "response_chars": len(api["raw_response"]),
            "raw_response": api["raw_response"],
            "api_error": api["error"],
            "category": question.get("category", ""),
            "level": question.get("level", ""),
        })
        time.sleep(0.3)
    return runs


def compute_consistency(runs):
    answers = [r["selected_answer"] for r in runs if r["selected_answer"]]
    if not answers:
        return 0.0
    return round(Counter(answers).most_common(1)[0][1] / len(runs), 4)


def compute_summary(cfg, all_runs, temperature):
    by_question = {}
    for r in all_runs:
        by_question.setdefault(r["question_id"], []).append(r)
    consistency = [compute_consistency(runs) for runs in by_question.values()]

    total_runs = len(all_runs)
    latencies = [r["latency_ms"] for r in all_runs if not r["api_error"]]
    latencies_sorted = sorted(latencies)
    p95_idx = min(int(len(latencies_sorted) * 0.95), len(latencies_sorted) - 1) if latencies_sorted else 0
    tokens_in = [r["tokens_in"] for r in all_runs]
    tokens_out = [r["tokens_out"] for r in all_runs]
    costs = [r["cost_usd"] for r in all_runs]
    confidences = [r["confidence"] for r in all_runs if r["confidence"] is not None]
    n_refused = sum(1 for r in all_runs if r["refusal"])
    n_invalid = sum(1 for r in all_runs if r["invalid_response"])

    return {
        "model": cfg["key"],
        "organization": cfg["org"],
        "type": cfg["type"],
        "weights": cfg["weights"],
        "temperature": None if uses_default_temperature(cfg) else temperature,
        "temperature_applied": not uses_default_temperature(cfg),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "accuracy_questionset": None,
        "correct_questions": None,
        "total_questions": len(by_question),
        "total_runs": total_runs,
        "avg_consistency": round(mean(consistency), 4) if consistency else 0.0,
        "min_consistency": round(min(consistency), 4) if consistency else 0.0,
        "refusal_count": n_refused,
        "refusal_rate": round(n_refused / total_runs, 4),
        "invalid_response_count": n_invalid,
        "invalid_response_rate": round(n_invalid / total_runs, 4),
        "failed_api_calls": sum(1 for r in all_runs if r["api_error"]),
        "total_tokens_in": sum(tokens_in),
        "total_tokens_out": sum(tokens_out),
        "total_tokens": sum(tokens_in) + sum(tokens_out),
        "avg_tokens_in": round(mean(tokens_in), 1) if tokens_in else 0,
        "avg_tokens_out": round(mean(tokens_out), 1) if tokens_out else 0,
        "total_cost_usd": round(sum(costs), 6),
        "avg_cost_per_call_usd": round(mean(costs), 8) if costs else 0,
        "avg_latency_ms": round(mean(latencies), 1) if latencies else 0,
        "max_latency_ms": max(latencies) if latencies else 0,
        "min_latency_ms": min(latencies) if latencies else 0,
        "p50_latency_ms": int(median(latencies)) if latencies else 0,
        "p95_latency_ms": latencies_sorted[p95_idx] if latencies_sorted else 0,
        "total_time_seconds": round(sum(latencies) / 1000, 2) if latencies else 0,
        "avg_response_chars": round(mean([r["response_chars"] for r in all_runs]), 1),
        "max_response_chars": max(r["response_chars"] for r in all_runs),
        "position_bias": dict(Counter(r["selected_answer"] for r in all_runs if r["selected_answer"])),
        "avg_confidence": round(mean(confidences), 4) if confidences else None,
        "ece": None,
        "brier_score": None,
    }


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def evaluate_model(cfg, questions, temperature):
    if uses_default_temperature(cfg):
        label = "tempDefault"
    else:
        label = f"temp{temperature}".replace(".", "_")
    log.info("%s: %d questions x %d runs, temperature %s", cfg["key"], len(questions), RUNS_PER_QUESTION, label)

    start = time.time()
    runs = [r for q in questions for r in evaluate_question(q, cfg, temperature)]
    wall_time = round(time.time() - start, 2)

    write_json(os.path.join(RESULTS_DIR, f"raw_{cfg['key']}_{label}.json"), runs)
    summary = compute_summary(cfg, runs, temperature)
    summary["wall_time_seconds"] = wall_time
    write_json(os.path.join(RESULTS_DIR, f"summary_{cfg['key']}_{label}.json"), summary)
    return summary


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--models", nargs="+", help="model keys from config.MODELS (default: all)")
    p.add_argument("--temperatures", nargs="+", type=float,
                   default=[TEMPERATURE_WITHOUT, TEMPERATURE_WITH],
                   help="temperatures for models that support one")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)

    with open(QUESTIONSET_FILE, encoding="utf-8") as f:
        raw = json.load(f)
    questions = list(raw.values()) if isinstance(raw, dict) else raw

    for q in questions:
        if "answer" in q or "explanation" in q or "source_sentence" in q:
            raise SystemExit(f"answer field found in question {q.get('question_id')}: "
                             f"run create_eval_dataset.py first")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summaries = []
    for cfg in MODELS:
        if args.models and cfg["key"] not in args.models:
            continue
        temperatures = args.temperatures[:1] if uses_default_temperature(cfg) else args.temperatures
        for temperature in temperatures:
            try:
                summaries.append(evaluate_model(cfg, questions, temperature))
            except Exception:
                log.exception("%s failed at temperature %s", cfg["key"], temperature)
    write_json(os.path.join(RESULTS_DIR, "full_summary.json"), summaries)


if __name__ == "__main__":
    main()
