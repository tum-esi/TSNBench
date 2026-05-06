import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import os
import re
import json
import logging
from datetime import datetime
from statistics import median, mean
from collections import Counter


RESULTS_DIR = "./output/results"
ANSWER_KEY_FILE = "./input_dataset/mcqa_tsn_dataset_answer_key.json"
SCORED_DIR = "./output/results/score"

os.makedirs(SCORED_DIR, exist_ok=True)

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


def compute_consistency(runs: list) -> float:
    answers = [r["selected_answer"] for r in runs if r["selected_answer"]]
    if not answers:
        return 0.0
    most_common_count = Counter(answers).most_common(1)[0][1]
    return round(most_common_count / len(runs), 4)


def compute_scored_summary(model_key: str, temperature: float,
                            all_runs: list, answer_key: dict) -> dict:
    total_runs = len(all_runs)
    total_questions = len(set(r["question_id"] for r in all_runs))
    # Group runs by question_id
    q_groups = {}
    for r in all_runs:
        qid = str(r["question_id"])
        q_groups.setdefault(qid, []).append(r)

    correct_count = 0
    consistency_scores = []

    for qid, runs in q_groups.items():
        key_entry = answer_key.get(qid, {})
        correct_ans = key_entry.get("correct_answer", "")

        answers = [r["selected_answer"] for r in runs if r["selected_answer"]]
        if answers and correct_ans:
            majority = Counter(answers).most_common(1)[0][0]
            if majority == correct_ans:
                correct_count += 1

        consistency_scores.append(compute_consistency(runs))

    accuracy = round(correct_count / total_questions, 4) if total_questions else 0.0

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

        # Group by question, majority vote
        lq_groups = {}
        for r in level_runs:
            qid = str(r["question_id"])
            lq_groups.setdefault(qid, []).append(r)

        lvl_correct = 0
        for qid, runs in lq_groups.items():
            correct_ans = answer_key.get(qid, {}).get("correct_answer", "")
            answers = [r["selected_answer"] for r in runs if r["selected_answer"]]
            if answers and correct_ans:
                majority = Counter(answers).most_common(1)[0][0]
                if majority == correct_ans:
                    lvl_correct += 1

        level_stats[level] = round(lvl_correct / len(lq_groups), 4)

    for r in all_runs:
        qid = str(r["question_id"])
        correct_ans = answer_key.get(qid, {}).get("correct_answer", "")
        r["correct_answer"] = correct_ans
        r["correct"] = (r["selected_answer"] == correct_ans) \
                              if r["selected_answer"] and correct_ans else False
    for r in all_runs:
        if r.get("level", "") == "":
            r["level"] = "human_crafted"

    n_bins = 10
    bins = [[] for _ in range(n_bins)]
    for r in all_runs:
        c = r.get("confidence")
        if c is None:
            continue
        bin_idx = min(int(c * n_bins), n_bins - 1)
        bins[bin_idx].append(int(r["correct"]))

    ece_total = sum(len(b) for b in bins)
    ece = 0.0
    if ece_total > 0:
        for i, b in enumerate(bins):
            if not b:
                continue
            bin_acc  = sum(b) / len(b)
            bin_conf = (i + 0.5) / n_bins
            ece += (len(b) / ece_total) * abs(bin_acc - bin_conf)
    ece = round(ece, 4)

    brier_pairs = [(r["confidence"], int(r["correct"]))
                   for r in all_runs if r["confidence"] is not None]
    brier = round(sum((c - o) ** 2 for c, o in brier_pairs) / len(brier_pairs), 4) \
            if brier_pairs else None

    latencies = [r["latency_ms"] for r in all_runs if not r["api_error"]]
    latencies_sorted = sorted(latencies)
    p95_idx = min(int(len(latencies_sorted) * 0.95),
                           len(latencies_sorted) - 1) if latencies_sorted else 0

    tokens_in_list  = [r["tokens_in"] for r in all_runs]
    tokens_out_list = [r["tokens_out"] for r in all_runs]
    costs  = [r["cost_usd"] for r in all_runs]

    selected_answers = [r["selected_answer"] for r in all_runs if r["selected_answer"]]
    position_bias = dict(Counter(selected_answers))

    confidences = [r["confidence"] for r in all_runs if r["confidence"] is not None]
    avg_confidence = round(mean(confidences), 4) if confidences else None

    return {
        "model"                  : model_key,
        "temperature"            : temperature,
        "timestamp"              : datetime.utcnow().isoformat(),

        # Accuracy
        "accuracy_overall"       : accuracy,
        "accuracy_easy"          : level_stats.get("easy"),
        "accuracy_medium"        : level_stats.get("medium"),
        "accuracy_hard"          : level_stats.get("hard"),
        "accuracy_lexicon"       : level_stats.get("lexicon"),
        "accuracy_human_crafted" : level_stats.get("human_crafted"),
        "correct_questions"      : correct_count,
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
        "avg_tokens_in"          : round(mean(tokens_in_list), 1) if tokens_in_list  else 0,
        "avg_tokens_out"         : round(mean(tokens_out_list), 1) if tokens_out_list else 0,

        "total_cost_usd"         : round(sum(costs), 6),
        "avg_cost_per_call_usd"  : round(mean(costs), 8) if costs else 0,

        "avg_latency_ms"         : round(mean(latencies), 1)      if latencies else 0,
        "max_latency_ms"         : max(latencies)                  if latencies else 0,
        "min_latency_ms"         : min(latencies)                  if latencies else 0,
        "p50_latency_ms"         : int(median(latencies))          if latencies else 0,
        "p95_latency_ms"         : latencies_sorted[p95_idx]       if latencies_sorted else 0,
        "total_time_seconds"     : round(sum(latencies) / 1000, 2) if latencies else 0,

        "avg_response_chars"     : round(mean([r["response_chars"] for r in all_runs]), 1),
        "max_response_chars"     : max(r["response_chars"] for r in all_runs),

        "position_bias"          : position_bias,

        "avg_confidence"         : avg_confidence,
        "ece"                    : ece,
        "brier_score"            : brier,
    }

def main():
    if not os.path.exists(ANSWER_KEY_FILE):
        log.error(f"Answer key not found: {ANSWER_KEY_FILE}")
        return

    with open(ANSWER_KEY_FILE, "r", encoding="utf-8") as f:
        answer_key = json.load(f)
    log.info(f"Loaded answer key: {len(answer_key)} entries")

    raw_files = sorted([
        f for f in os.listdir(RESULTS_DIR)
        if f.startswith("raw_") and f.endswith(".json")
    ])

    if not raw_files:
        log.error(f"No raw_*.json files found in {RESULTS_DIR}")
        return

    log.info(f"Found {len(raw_files)} result files to score")

    all_summaries = []

    for fname in raw_files:
        raw_path = os.path.join(RESULTS_DIR, fname)

        with open(raw_path, "r", encoding="utf-8") as f:
            all_runs = json.load(f)

        if not all_runs:
            log.warning(f"Empty file: {fname}")
            continue
        stem = fname.replace("raw_", "").replace(".json", "")

        temp_match = re.search(r"_temp(\d+_\d+)$", stem)
        if temp_match:
            temp_str = temp_match.group(1).replace("_", ".")
            temperature = float(temp_str)
            model_key = stem[:stem.rfind("_temp")]
        else:
            default_match = re.search(r"_tempDefault$", stem)
            if default_match:
                temperature = None
                model_key = stem.replace("_tempDefault", "")
            else:
                temperature = 0.0
                model_key = stem

        log.info(f"{'='*60}")
        log.info(f"Scoring: {model_key} | temp={temperature} | {len(all_runs)} records")

        summary = compute_scored_summary(model_key, temperature, all_runs, answer_key)
        all_summaries.append(summary)

        log.info(f"  Overall accuracy  : {summary['accuracy_overall']:.1%}")
        log.info(f"  Easy              : {summary['accuracy_easy']:.1%}"   if summary['accuracy_easy']   is not None else "  Easy   : N/A")
        log.info(f"  Medium            : {summary['accuracy_medium']:.1%}" if summary['accuracy_medium'] is not None else "  Medium : N/A")
        log.info(f"  Hard              : {summary['accuracy_hard']:.1%}"   if summary['accuracy_hard']   is not None else "  Hard   : N/A")
        log.info(f"  Human-crafted     : {summary['accuracy_human_crafted']:.1%}"
                 if summary['accuracy_human_crafted'] is not None
                 else "  Human-crafted : N/A")
        log.info(f"  Avg consistency   : {summary['avg_consistency']:.2f}")
        log.info(f"  ECE               : {summary['ece']}")
        log.info(f"  Brier score       : {summary['brier_score']}")
        log.info(f"  Avg confidence    : {summary['avg_confidence']}")
        log.info(f"  Refusal rate      : {summary['refusal_rate']:.2%}")
        log.info(f"  Invalid rate      : {summary['invalid_response_rate']:.2%}")
        log.info(f"  Total cost        : ${summary['total_cost_usd']:.4f}")
        log.info(f"  Avg latency       : {summary['avg_latency_ms']:.0f}ms")

        # Save scored raw results (with correct/incorrect added)
        scored_raw_path = os.path.join(SCORED_DIR, f"scored_{fname}")
        with open(scored_raw_path, "w", encoding="utf-8") as f:
            json.dump(all_runs, f, indent=2, ensure_ascii=False)

        # Save scored summary
        summary_fname = fname.replace("raw_", "scored_summary_")
        summary_path  = os.path.join(SCORED_DIR, summary_fname)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        log.info(f"  Saved: {scored_raw_path}")
        log.info(f"  Saved: {summary_path}")

    # Save full comparison summary
    full_path = os.path.join(SCORED_DIR, "scored_full_summary.json")
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)
    log.info(f"\nFull summary saved: {full_path}")

    # Print comparison table
    log.info(f"\n{'='*85}")
    log.info(f"{'Model':<25} {'Temp':>5} {'Overall':>8} {'Easy':>7} {'Med':>7} {'Hard':>7} {'ECE':>6} {'Cost$':>7}")
    log.info(f"{'-'*85}")
    for s in sorted(all_summaries, key=lambda x: x["accuracy_overall"], reverse=True):
        easy   = f"{s['accuracy_easy']*100:.1f}"   if s["accuracy_easy"]   is not None else "N/A"
        medium = f"{s['accuracy_medium']*100:.1f}" if s["accuracy_medium"] is not None else "N/A"
        hard   = f"{s['accuracy_hard']*100:.1f}"   if s["accuracy_hard"]   is not None else "N/A"
        human = f"{s['accuracy_human_crafted'] * 100:.1f}" if s["accuracy_human_crafted"] is not None else "N/A"
        model_str = s["model"] or "unknown"
        temp_str = str(s["temperature"]) if s["temperature"] is not None else "Def"
        log.info(
            f"{model_str:<25} "
            f"{temp_str:>5} "
            f"{s['accuracy_overall']:>8.1%} "
            f"{easy:>7} "
            f"{medium:>7} "
            f"{hard:>7} "
            f"{human:>7} "
            f"{str(s['ece']):>6} "
            f"{s['total_cost_usd']:>7.4f}"
        )
    log.info(f"{'='*85}")

    log.info(f"\n% LaTeX table rows")
    log.info(f"% Model & Accuracy (\\%) & Avg Consistency & Refusal (\\%) & Avg Latency (ms) & Avg Confidence \\\\")
    log.info(f"% {'=' * 90}")

    for s in sorted(all_summaries, key=lambda x: x["accuracy_overall"], reverse=True):
        temp_val = str(s["temperature"]) if s["temperature"] is not None else "Def"
        model_str = f"{s['model'] or 'unknown'} (temp {temp_val})"
        accuracy = f"{s['accuracy_overall'] * 100:.1f}" if s["accuracy_overall"] is not None else "--"
        consistency = f"{s['avg_consistency']:.2f}" if s["avg_consistency"] is not None else "--"
        refusal = f"{s['refusal_rate'] * 100:.1f}" if s["refusal_rate"] is not None else "--"
        latency = f"{s['avg_latency_ms']:.0f}" if s["avg_latency_ms"] is not None else "--"
        confidence = f"{s['avg_confidence']:.4f}" if s["avg_confidence"] is not None else "--"
        log.info(f"{model_str} & {accuracy} & {consistency} & {refusal} & {latency} & {confidence} \\\\")

    log.info(f"\n% Model comparison — default temp vs temp 0.0")
    log.info(
        f"% {'Model':<35} {'Acc(Def)':>10} {'Acc(0.0)':>10} {'Acc(0.7)':>10} {'Cons(Def)':>11} {'Cons(0.0)':>11} {'Cons(0.7)':>11}")
    log.info(f"% {'-' * 80}")

    # Group summaries by model key
    from collections import defaultdict
    model_groups = defaultdict(dict)
    for s in all_summaries:
        key = s["model"] or "unknown"
        temp = s["temperature"]
        if temp is None:
            label = "default"
        else:
            label = str(round(float(temp), 1))  # "0.0", "0.7", "1.0" etc
        model_groups[key][label] = s

    for model_key in sorted(model_groups.keys()):
        entries = model_groups[model_key]
        s_def = entries.get("default")
        s_zero = entries.get("0.0")
        s_seven = entries.get("0.7")

        acc_def = f"{s_def['accuracy_overall'] * 100:.1f}" if s_def and s_def.get(
            "accuracy_overall") is not None else "--"
        acc_zero = f"{s_zero['accuracy_overall'] * 100:.1f}" if s_zero and s_zero.get(
            "accuracy_overall") is not None else "--"
        acc_seven = f"{s_seven['accuracy_overall'] * 100:.1f}" if s_seven and s_seven.get(
            "accuracy_overall") is not None else "--"
        cons_def = f"{s_def['avg_consistency']:.2f}" if s_def and s_def.get("avg_consistency") is not None else "--"
        cons_zero = f"{s_zero['avg_consistency']:.2f}" if s_zero and s_zero.get("avg_consistency") is not None else "--"
        cons_seven = f"{s_seven['avg_consistency']:.2f}" if s_seven and s_seven.get(
            "avg_consistency") is not None else "--"

        log.info(f"{model_key} & {acc_def} & {acc_zero} & {acc_seven} & {cons_def} & {cons_zero} & {cons_seven} \\\\")
    log.info("Scoring complete.")


if __name__ == "__main__":
    main()