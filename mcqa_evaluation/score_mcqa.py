"""Scores the MCQA runs."""

import csv
import json
import os
import re
from collections import Counter
from datetime import datetime
from statistics import median, mean

RESULTS_DIR = "./output/results"
ANSWER_KEY_FILE = "./input_dataset/mcqa_tsn_dataset_answer_key.json"
SCORED_DIR = "./output/results/scored"
OUT_DIR = "results"

CW_THRESHOLD = 0.8  # confidently wrong

MODEL_NAMES = {
    "grok-4-1-fast-reasoning": "Grok 4.1 Fast",
    "grok-4-1-fast-non-reasoning": "Grok 4.1 Fast (Non-Reasoning)",
    "deepseek-chat": "DeepSeek-V3.2 (Non-Thinking)",
    "gpt-4o-2024-08-06": "GPT-4o",
    "gpt-4o-mini-2024-07-18": "GPT-4o mini",
    "Llama-3.3-70B-Instruct": "Llama 3.3 70B",
    "mistral-medium-2508": "Mistral Medium 3.1",
    "mistral-large-2512": "Mistral Large 3",
    "claude-sonnet-4-5-20250929": "Claude Sonnet 4.5",
    "o3-2025-04-16": "o3",
    "gpt-5-2025-08-07": "GPT-5",
    "deepseek-reasoner": "DeepSeek-V3.2 (Thinking)",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "Llama-3.2-1B": "Llama 3.2 1B",
    "Qwen3-8B": "Qwen3 8B",
    "ministral-8b-2512": "Ministral 3 8B",
}
MODEL_ORDER = list(MODEL_NAMES)


def compute_consistency(runs):
    answers = [r["selected_answer"] for r in runs if r["selected_answer"]]
    if not answers:
        return 0.0
    most_common_count = Counter(answers).most_common(1)[0][1]
    return round(most_common_count / len(runs), 4)


def majority_accuracy(runs_by_question, answer_key):
    correct = 0
    for qid, runs in runs_by_question.items():
        correct_ans = answer_key.get(qid, {}).get("correct_answer", "")
        answers = [r["selected_answer"] for r in runs if r["selected_answer"]]
        if answers and correct_ans:
            if Counter(answers).most_common(1)[0][0] == correct_ans:
                correct += 1
    return correct


def compute_scored_summary(model_key, temperature, all_runs, answer_key):
    total_runs = len(all_runs)
    total_questions = len(set(r["question_id"] for r in all_runs))

    q_groups = {}
    for r in all_runs:
        q_groups.setdefault(str(r["question_id"]), []).append(r)
    correct_count = majority_accuracy(q_groups, answer_key)
    consistency_scores = [compute_consistency(runs) for runs in q_groups.values()]
    accuracy = round(correct_count / total_questions, 4) if total_questions else 0.0

    # no level: handmade question
    level_stats = {}
    for level in ["easy", "medium", "hard", "lexicon", "human_crafted"]:
        if level == "human_crafted":
            level_runs = [r for r in all_runs
                          if answer_key.get(str(r["question_id"]), {}).get("level", "") == ""]
        else:
            level_runs = [r for r in all_runs
                          if answer_key.get(str(r["question_id"]), {}).get("level") == level]
        if not level_runs:
            level_stats[level] = None
            continue
        lq_groups = {}
        for r in level_runs:
            lq_groups.setdefault(str(r["question_id"]), []).append(r)
        level_stats[level] = round(majority_accuracy(lq_groups, answer_key) / len(lq_groups), 4)

    for r in all_runs:
        correct_ans = answer_key.get(str(r["question_id"]), {}).get("correct_answer", "")
        r["correct_answer"] = correct_ans
        r["correct"] = (r["selected_answer"] == correct_ans) if r["selected_answer"] and correct_ans else False
    for r in all_runs:
        if r.get("level", "") == "":
            r["level"] = "human_crafted"

    # ECE, 10 bins (Guo et al. 2017)
    n_bins = 10
    bins = [[] for _ in range(n_bins)]
    for r in all_runs:
        c = r.get("confidence")
        if c is None:
            continue
        bins[min(int(c * n_bins), n_bins - 1)].append((c, int(r["correct"])))
    ece_total = sum(len(b) for b in bins)
    ece = 0.0
    if ece_total > 0:
        for b in bins:
            if not b:
                continue
            bin_acc = sum(o for _, o in b) / len(b)
            bin_conf = sum(c for c, _ in b) / len(b)
            ece += (len(b) / ece_total) * abs(bin_acc - bin_conf)
    ece = round(ece, 4)

    brier_pairs = [(r["confidence"], int(r["correct"])) for r in all_runs if r["confidence"] is not None]
    brier = round(sum((c - o) ** 2 for c, o in brier_pairs) / len(brier_pairs), 4) if brier_pairs else None

    wrong_runs = [r for r in all_runs if not r["correct"] and r.get("confidence") is not None]
    cw_rate = (round(sum(1 for r in wrong_runs if r["confidence"] >= CW_THRESHOLD) / len(wrong_runs), 4)
               if wrong_runs else None)

    latencies = [r["latency_ms"] for r in all_runs if not r["api_error"]]
    latencies_sorted = sorted(latencies)
    p95_idx = min(int(len(latencies_sorted) * 0.95), len(latencies_sorted) - 1) if latencies_sorted else 0

    tokens_in_list = [r["tokens_in"] for r in all_runs]
    tokens_out_list = [r["tokens_out"] for r in all_runs]
    costs = [r["cost_usd"] for r in all_runs]
    confidences = [r["confidence"] for r in all_runs if r["confidence"] is not None]
    n_refusal = sum(1 for r in all_runs if r["refusal"])
    n_invalid = sum(1 for r in all_runs if r["invalid_response"])

    return {
        "model": model_key,
        "temperature": temperature,
        "timestamp": datetime.utcnow().isoformat(),

        "accuracy_overall": accuracy,
        "accuracy_easy": level_stats.get("easy"),
        "accuracy_medium": level_stats.get("medium"),
        "accuracy_hard": level_stats.get("hard"),
        "accuracy_lexicon": level_stats.get("lexicon"),
        "accuracy_human_crafted": level_stats.get("human_crafted"),
        "correct_questions": correct_count,
        "total_questions": total_questions,
        "total_runs": total_runs,

        "avg_consistency": round(mean(consistency_scores), 4) if consistency_scores else 0.0,
        "min_consistency": round(min(consistency_scores), 4) if consistency_scores else 0.0,

        "refusal_count": n_refusal,
        "refusal_rate": round(n_refusal / total_runs, 4),
        "invalid_response_count": n_invalid,
        "invalid_response_rate": round(n_invalid / total_runs, 4),
        "failed_api_calls": sum(1 for r in all_runs if r["api_error"]),

        "total_tokens_in": sum(tokens_in_list),
        "total_tokens_out": sum(tokens_out_list),
        "total_tokens": sum(tokens_in_list) + sum(tokens_out_list),
        "avg_tokens_in": round(mean(tokens_in_list), 1) if tokens_in_list else 0,
        "avg_tokens_out": round(mean(tokens_out_list), 1) if tokens_out_list else 0,

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
        "ece": ece,
        "brier_score": brier,
        "cw_rate": cw_rate,
        "cw_threshold": CW_THRESHOLD,
    }


def parse_file_name(fname):
    stem = fname.replace("raw_", "").replace(".json", "")
    m = re.search(r"_temp(\d+_\d+)$", stem)
    if m:
        return stem[:stem.rfind("_temp")], float(m.group(1).replace("_", "."))
    if stem.endswith("_tempDefault"):
        return stem.replace("_tempDefault", ""), None
    return stem, 0.0


def temp_label(t):
    return "default" if t is None else str(round(float(t), 1))


def fmt(v, scale=1, dec=1):
    return "--" if v is None else f"{v * scale:.{dec}f}"


def save_table(name, header, rows):
    rows = [[str(c) for c in r] for r in rows]
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"{name}.csv"), "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([header] + rows)

    widths = [max(len(r[i]) for r in [header] + rows) for i in range(len(header))]
    print(f"\n{name}")
    for r in [header] + rows:
        print(r[0].ljust(widths[0] + 2) + "".join(c.rjust(w + 2) for c, w in zip(r[1:], widths[1:])))


def results_table(summaries):
    by_model = {}
    for s in summaries:
        by_model.setdefault(s["model"] or "unknown", []).append(s)

    def pick(entries):
        for want in ("0.0", "default", "0.7"):
            for s in entries:
                if temp_label(s["temperature"]) == want:
                    return s
        return entries[0]

    keys = [k for k in MODEL_ORDER if k in by_model] + sorted(k for k in by_model if k not in MODEL_ORDER)
    header = ["Model", "Temperature", "Accuracy (%)", "Avg consistency", "Avg latency (ms)",
              "Avg confidence", "ECE", "Brier", "CW rate (%)"]
    rows = []
    for k in keys:
        s = pick(by_model[k])
        rows.append([MODEL_NAMES.get(k, k), temp_label(s["temperature"]),
                     fmt(s.get("accuracy_overall"), 100, 1), fmt(s.get("avg_consistency"), 1, 4),
                     fmt(s.get("avg_latency_ms"), 1, 0), fmt(s.get("avg_confidence"), 1, 4),
                     fmt(s.get("ece"), 1, 4), fmt(s.get("brier_score"), 1, 4), fmt(s.get("cw_rate"), 100, 1)])
    return header, rows


def temperature_table(summaries):
    by_model = {}
    for s in summaries:
        by_model.setdefault(s["model"] or "unknown", {})[temp_label(s["temperature"])] = s

    keys = [k for k in MODEL_ORDER if k in by_model] + sorted(k for k in by_model if k not in MODEL_ORDER)
    temps = ["default", "0.0", "0.7"]
    header = (["Model"] + [f"Accuracy (%) {t}" for t in temps] + [f"Consistency {t}" for t in temps])
    rows = []
    for k in keys:
        e = by_model[k]
        rows.append([MODEL_NAMES.get(k, k)]
                    + [fmt(e[t].get("accuracy_overall"), 100, 1) if t in e else "--" for t in temps]
                    + [fmt(e[t].get("avg_consistency"), 1, 2) if t in e else "--" for t in temps])
    return header, rows


def main():
    with open(ANSWER_KEY_FILE, "r", encoding="utf-8") as f:
        answer_key = json.load(f)
    os.makedirs(SCORED_DIR, exist_ok=True)

    raw_files = sorted(f for f in os.listdir(RESULTS_DIR) if f.startswith("raw_") and f.endswith(".json"))
    summaries = []
    for fname in raw_files:
        with open(os.path.join(RESULTS_DIR, fname), "r", encoding="utf-8") as f:
            all_runs = json.load(f)
        if not all_runs:
            continue
        model_key, temperature = parse_file_name(fname)
        summary = compute_scored_summary(model_key, temperature, all_runs, answer_key)
        summaries.append(summary)

        with open(os.path.join(SCORED_DIR, f"scored_{fname}"), "w", encoding="utf-8") as f:
            json.dump(all_runs, f, indent=2, ensure_ascii=False)
        with open(os.path.join(SCORED_DIR, fname.replace("raw_", "scored_summary_")), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(os.path.join(SCORED_DIR, "scored_full_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    save_table("mcqa_results", *results_table(summaries))
    save_table("mcqa_temperatures", *temperature_table(summaries))


if __name__ == "__main__":
    main()
