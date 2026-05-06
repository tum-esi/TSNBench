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
from statistics import median, mean
from collections import Counter

import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

from google.genai import types as genai_types
from config import (
    MODELS, QUESTIONSET_FILE, RESULTS_DIR,
    RUNS_PER_QUESTION, TEMPERATURE_WITH, TEMPERATURE_WITHOUT,
    compute_cost
)

os.makedirs(RESULTS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            os.path.join(RESULTS_DIR, "evaluation.log"),
            encoding="utf-8"
        ),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

for noisy in ["httpx", "httpcore", "openai", "anthropic",
              "mistralai", "google", "urllib3"]:
    logging.getLogger(noisy).setLevel(logging.ERROR)

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

def build_eval_prompt(question: dict) -> str:
    options_text = ""
    for letter in ["A", "B", "C", "D", "E"]:
        if letter in question:
            options_text += f"  {letter}: {question[letter]}\n"
    return (
        f"QUESTION:\n{question['question']}\n\n"
        f"OPTIONS:\n{options_text}"
    )


def parse_eval_response(raw: str) -> dict:
    if not raw:
        return {"answer": None, "confidence": None, "refusal": False, "invalid": True}

    clean = raw.strip()
    clean = re.sub(r"^```(?:json)?", "", clean).strip()
    clean = re.sub(r"```$",          "", clean).strip()

    refusal_patterns = [
        "i cannot", "i can't", "i am not able", "i'm not able",
        "unable to", "i don't know", "i do not know",
        "cannot answer", "not enough information"
    ]
    if any(p in clean.lower() for p in refusal_patterns):
        return {"answer": None, "confidence": None, "refusal": True, "invalid": False}
    try:
        parsed = json.loads(clean)
        if not isinstance(parsed, dict):
            # If it parsed to a bare letter string e.g. "B"
            if isinstance(parsed, str) and parsed.strip().upper() in ["A", "B", "C", "D", "E"]:
                return {
                    "answer": parsed.strip().upper(),
                    "confidence": None,
                    "refusal": False,
                    "invalid": False
                }
            raise json.JSONDecodeError("not a dict", clean, 0)

        answer = str(parsed.get("answer", "")).strip().upper()

        confidence_raw = parsed.get("confidence")
        if isinstance(confidence_raw, (int, float)):
            confidence = round(max(0.0, min(1.0, float(confidence_raw))), 4)
        elif isinstance(confidence_raw, str):
            try:
                confidence = round(max(0.0, min(1.0, float(confidence_raw))), 4)
            except ValueError:
                confidence = None
        else:
            confidence = None

        if answer not in ["A", "B", "C", "D", "E"]:
            answer = None

        return {
            "answer"    : answer,
            "confidence": confidence,
            "refusal"   : False,
            "invalid"   : answer is None
        }
    except json.JSONDecodeError:
        pass

    match = re.search(r'"answer"\s*:\s*"([A-Ea-e])"', clean)
    if match:
        answer = match.group(1).upper()
        conf_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', clean)
        confidence = round(max(0.0, min(1.0, float(conf_match.group(1)))), 4) if conf_match else None
        return {"answer": answer, "confidence": confidence, "refusal": False, "invalid": False}

    match = re.search(r'\b([A-E])\b', clean)
    if match:
        return {"answer": match.group(1).upper(), "confidence": None, "refusal": False, "invalid": False}

    return {"answer": None, "confidence": None, "refusal": False, "invalid": True}


def call_model(model_cfg: dict, system: str, user: str, temperature: float) -> dict:
    api        = model_cfg["api"]
    client     = model_cfg["client"]
    model_id   = model_cfg["model_id"]
    result     = {
        "raw_response": "",
        "tokens_in"   : 0,
        "tokens_out"  : 0,
        "latency_ms"  : 0,
        "error"       : None
    }

    start = time.time()

    max_tokens = model_cfg["max_tokens"]

    try:
        if api == "anthropic":
            kwargs = dict(
                model      = model_id,
                max_tokens = max_tokens,
                system     = system,
                messages   = [{"role": "user", "content": user}]
            )
            if not model_cfg.get("skip_temperature", False):
                kwargs["temperature"] = temperature
            resp = client.messages.create(**kwargs)
            result["raw_response"] = resp.content[0].text
            result["tokens_in"]    = resp.usage.input_tokens
            result["tokens_out"]   = resp.usage.output_tokens

        elif api == "gemini":
            prompt = f"{system}\n\n{user}"
            gen_config_kwargs = {"max_output_tokens": max_tokens}
            if not model_cfg.get("skip_temperature", False):
                gen_config_kwargs["temperature"] = temperature
            resp   = client.models.generate_content(
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
                messages   = [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user}
                ],
                stream = False
            )
            if not model_cfg.get("skip_temperature", False):
                kwargs["temperature"] = temperature
            for attempt in range(5):
                try:
                    resp = client.chat.complete(**kwargs)
                    result["raw_response"] = resp.choices[0].message.content
                    result["tokens_in"] = resp.usage.prompt_tokens
                    result["tokens_out"] = resp.usage.completion_tokens
                    break  # success — exit retry loop
                except Exception as mistral_e:
                    err = str(mistral_e)
                    if "429" in err or "rate_limited" in err or "1300" in err:
                        wait = 60 * (attempt + 1)  # 60s, 120s, 180s, 240s, 300s
                        log.warning(f"  Mistral rate limit (429) — waiting {wait}s before retry {attempt + 1}/5")
                        time.sleep(wait)
                    elif "503" in err or "upstream" in err or "overflow" in err:
                        wait = 30 * (attempt + 1)  # 30s, 60s, 90s, 120s, 150s
                        log.warning(f"  Mistral server error (503) — waiting {wait}s before retry {attempt + 1}/5")
                        time.sleep(wait)
                    else:
                        log.warning(f"  Mistral attempt {attempt + 1}/5 failed: {mistral_e}")
                        time.sleep(2 ** attempt)
                    if attempt == 4:
                        raise mistral_e

        else:
            max_tokens_param = model_cfg.get("max_tokens_param", "max_tokens")
            kwargs = {
                "model": model_id,
                max_tokens_param: max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ]
            }
            if model_cfg["type"] != "Reasoning" and not model_cfg.get("skip_temperature", False):
                kwargs["temperature"] = temperature

            resp = client.chat.completions.create(**kwargs)
            result["raw_response"] = resp.choices[0].message.content
            result["tokens_in"] = resp.usage.prompt_tokens
            result["tokens_out"] = resp.usage.completion_tokens

    except Exception as e:
        result["error"] = str(e)
        log.warning(f"  API error ({model_cfg['key']}): {e}")

    result["latency_ms"] = int((time.time() - start) * 1000)
    return result


def evaluate_question(
    question    : dict,
    model_cfg   : dict,
    temperature : float
) -> list[dict]:
    system = EVAL_SYSTEM_PROMPT
    user   = build_eval_prompt(question)
    runs   = []

    for run_idx in range(1, RUNS_PER_QUESTION + 1):
        api_result = call_model(model_cfg, system, user, temperature)
        parsed     = parse_eval_response(api_result["raw_response"])

        cost = compute_cost(
            model_cfg["pricing_key"],
            api_result["tokens_in"],
            api_result["tokens_out"]
        )

        record = {
            "question_id"      : question.get("question_id"),
            "model"            : model_cfg["key"],
            "run"              : run_idx,
            "temperature" : None if (model_cfg.get("skip_temperature", False) or
                         model_cfg["type"] == "Reasoning") else temperature,
            "timestamp"        : datetime.utcnow().isoformat(),
            "selected_answer"  : parsed["answer"],
            "confidence"       : parsed["confidence"],
            "refusal"          : parsed["refusal"],
            "invalid_response" : parsed["invalid"],
            "tokens_in"        : api_result["tokens_in"],
            "tokens_out"       : api_result["tokens_out"],
            "total_tokens"     : api_result["tokens_in"] + api_result["tokens_out"],
            "cost_usd"         : round(cost, 8),
            "latency_ms"       : api_result["latency_ms"],
            "response_chars"   : len(api_result["raw_response"]),
            "raw_response"     : api_result["raw_response"],
            "api_error"        : api_result["error"],
            "category"         : question.get("category", ""),
            "level"            : question.get("level", ""),
        }
        runs.append(record)

        log.info(
            f"  Q{question.get('question_id')} run {run_idx} | "
            f"{model_cfg['key']} | "
            f"ans={parsed['answer']} "
            f"conf={parsed['confidence']} "
            f"lat={api_result['latency_ms']}ms"
        )
        time.sleep(0.3)

    return runs


def compute_consistency(runs: list[dict]) -> float:
    answers = [r["selected_answer"] for r in runs if r["selected_answer"]]
    if not answers:
        return 0.0
    most_common_count = Counter(answers).most_common(1)[0][1]
    return round(most_common_count / len(runs), 4)


def compute_summary(
    model_cfg   : dict,
    all_runs    : list[dict],
    temperature : float
) -> dict:

    total_questions = len(set(r["question_id"] for r in all_runs))
    total_runs = len(all_runs)

    consistency_scores = []
    q_groups = {}
    for r in all_runs:
        qid = r["question_id"]
        q_groups.setdefault(qid, []).append(r)

    for qid, runs in q_groups.items():
        consistency_scores.append(compute_consistency(runs))

    latencies = [r["latency_ms"] for r in all_runs if not r["api_error"]]
    latencies_sorted = sorted(latencies)
    p95_idx = min(int(len(latencies_sorted) * 0.95), len(latencies_sorted) - 1) if latencies_sorted else 0

    tokens_in_list  = [r["tokens_in"]  for r in all_runs]
    tokens_out_list = [r["tokens_out"] for r in all_runs]
    costs           = [r["cost_usd"]   for r in all_runs]

    selected_answers = [r["selected_answer"] for r in all_runs if r["selected_answer"]]
    position_bias    = dict(Counter(selected_answers))

    confidences = [r["confidence"] for r in all_runs
                   if r["confidence"] is not None]
    avg_confidence = round(mean(confidences), 4) if confidences else None

    return {
        "model"                  : model_cfg["key"],
        "organization"           : model_cfg["org"],
        "type"                   : model_cfg["type"],
        "weights"                : model_cfg["weights"],
        "temperature": None if (model_cfg.get("skip_temperature", False) or
                                model_cfg["type"] == "Reasoning") else temperature,
        "temperature_applied"    : not (
            model_cfg.get("skip_temperature", False) or
            model_cfg["type"] == "Reasoning"
        ),
        "timestamp"              : datetime.utcnow().isoformat(),
        "accuracy_questionset"   : None,
        "correct_questions"      : None,
        "total_questions"        : total_questions,
        "total_runs"             : total_runs,
        "avg_consistency"        : round(mean(consistency_scores), 4) if consistency_scores else 0.0,
        "min_consistency"        : round(min(consistency_scores), 4)  if consistency_scores else 0.0,
        "refusal_count"          : sum(1 for r in all_runs if r["refusal"]),
        "refusal_rate"           : round(sum(1 for r in all_runs if r["refusal"]) / total_runs, 4),
        "invalid_response_count" : sum(1 for r in all_runs if r["invalid_response"]),
        "invalid_response_rate"  : round(sum(1 for r in all_runs if r["invalid_response"]) / total_runs, 4),
        "failed_api_calls"       : sum(1 for r in all_runs if r["api_error"]),
        "total_tokens_in"        : sum(tokens_in_list),
        "total_tokens_out"       : sum(tokens_out_list),
        "total_tokens"           : sum(tokens_in_list) + sum(tokens_out_list),
        "avg_tokens_in"          : round(mean(tokens_in_list), 1)  if tokens_in_list  else 0,
        "avg_tokens_out"         : round(mean(tokens_out_list), 1) if tokens_out_list else 0,
        "total_cost_usd"         : round(sum(costs), 6),
        "avg_cost_per_call_usd"  : round(mean(costs), 8) if costs else 0,
        "avg_latency_ms"         : round(mean(latencies), 1)         if latencies else 0,
        "max_latency_ms"         : max(latencies)                     if latencies else 0,
        "min_latency_ms"         : min(latencies)                     if latencies else 0,
        "p50_latency_ms"         : int(median(latencies))             if latencies else 0,
        "p95_latency_ms"         : latencies_sorted[p95_idx]          if latencies_sorted else 0,
        "total_time_seconds"     : round(sum(latencies) / 1000, 2)    if latencies else 0,
        "avg_response_chars"     : round(mean([r["response_chars"] for r in all_runs]), 1),
        "max_response_chars"     : max(r["response_chars"] for r in all_runs),
        "position_bias"          : position_bias,
        "avg_confidence"         : avg_confidence,
        "ece"                    : None,
        "brier_score"            : None,
    }


def evaluate_model(
    model_cfg   : dict,
    questions   : list[dict],
    temperature : float
) -> tuple[list[dict], dict]:
    skip_temp = (
            model_cfg.get("skip_temperature", False) or
            model_cfg["type"] == "Reasoning"
    )
    temp_label = "tempDefault" if skip_temp else f"temp{temperature}".replace(".", "_")
    log.info(f"{'='*70}")
    log.info(f"Model: {model_cfg['key']} | Temperature: {temperature}")
    log.info(f"Questions: {len(questions)} | Runs each: {RUNS_PER_QUESTION}")
    log.info(f"{'='*70}")

    all_runs = []
    model_start = time.time()

    for q in questions:
        runs = evaluate_question(q, model_cfg, temperature)
        all_runs.extend(runs)

    model_elapsed = round(time.time() - model_start, 2)
    log.info(f"Model {model_cfg['key']} completed in {model_elapsed}s")

    raw_path = os.path.join(
        RESULTS_DIR,
        f"raw_{model_cfg['key']}_{temp_label}.json"
    )
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(all_runs, f, indent=2, ensure_ascii=False)
    log.info(f"Raw results saved: {raw_path}")
    summary = compute_summary(model_cfg, all_runs, temperature)
    summary["wall_time_seconds"] = model_elapsed

    summary_path = os.path.join(
        RESULTS_DIR,
        f"summary_{model_cfg['key']}_{temp_label}.json"
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info(f"Summary saved: {summary_path}")

    return all_runs, summary


def main():
    if not os.path.exists(QUESTIONSET_FILE):
        log.error(f"File not found: {QUESTIONSET_FILE}")
        return
    with open(QUESTIONSET_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        questions = list(raw.values())
    else:
        questions = raw

    for q in questions:
        if "answer" in q or "explanation" in q or "source_sentence" in q:
            log.error(
                f"Answer field detected in eval input — Q{q.get('question_id')}. "
                f"Run prepare_eval_dataset.py first to strip answers."
            )
            return

    log.info(f"Loaded {len(questions)} questions from {QUESTIONSET_FILE}")
    log.info(f"Input verified — no answer fields present")
    log.info(f"Models to evaluate: {len(MODELS)}")
    log.info(f"Temperature settings: {TEMPERATURE_WITHOUT} (without), {TEMPERATURE_WITH} (with)")
    log.info(f"Runs per question: {RUNS_PER_QUESTION}")
    log.info(f"NOTE: Accuracy will be computed by score_results.py after evaluation")

    temp_one_models = sum(1 for m in MODELS if m.get("skip_temperature", False) or m["type"] == "Reasoning")
    temp_two_models = len(MODELS) - temp_one_models
    total_api_calls = len(questions) * RUNS_PER_QUESTION * (temp_one_models + temp_two_models * 2)
    log.info(f"Models running once (no temperature): {temp_one_models}")
    log.info(f"Models running twice (with/without temperature): {temp_two_models}")
    log.info(f"Total API calls: {total_api_calls}")

    all_summaries = []
    for model_cfg in MODELS:
        skip_temp = (
                model_cfg.get("skip_temperature", False) or
                model_cfg["type"] == "Reasoning"
        )
        temperatures_to_run = [TEMPERATURE_WITHOUT] if skip_temp else [TEMPERATURE_WITHOUT, TEMPERATURE_WITH]
        # temperatures_to_run = [] if skip_temp else [TEMPERATURE_WITH]

        if skip_temp:
            log.info(f"  {model_cfg['key']} does not support temperature — running once only")

        for temperature in temperatures_to_run:
            try:
                _, summary = evaluate_model(model_cfg, questions, temperature)
                all_summaries.append(summary)
            except Exception as e:
                log.error(f"Failed: {model_cfg['key']} temp={temperature} | {e}", exc_info=True)

    full_summary_path = os.path.join(RESULTS_DIR, "full_summary.json")
    with open(full_summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)
    log.info(f"Full summary saved: {full_summary_path}")

    log.info(f"\n{'='*70}")
    log.info(f"{'Model':<25} {'Temp':>5} {'Consist':>8} {'Cost$':>8} {'Lat(ms)':>8} {'Refusal':>8}")
    log.info(f"{'-'*70}")
    for s in sorted(all_summaries, key=lambda x: x["avg_consistency"], reverse=True):
        temp_str = str(s["temperature"]) if s["temperature"] is not None else "Def"
        model_str = s["model"] or "unknown"
        log.info(
            f"{model_str:<25} "
            f"{temp_str:>5} "
            f"{s['avg_consistency']:>8.2f} "
            f"{s['total_cost_usd']:>8.4f} "
            f"{s['avg_latency_ms']:>8.0f} "
            f"{s['refusal_rate']:>8.2%}"
        )
    log.info(f"{'='*70}")
    log.info("Evaluation complete. Run check_accuracy.py to compute accuracy.")


if __name__ == "__main__":
    main()