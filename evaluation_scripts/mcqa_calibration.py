"""ECE, Brier score and CW rate of the MCQA runs."""

import csv
import json
import os

SCORED_DIR = "../mcqa_evaluation/output/results/scored"
CW_THRESHOLD = 0.8
OUT_DIR = "results"

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


def compute_cw_rate(records, threshold=0.8):
    wrong = [r for r in records if r.get("correct") == False and r.get("confidence") is not None]
    if not wrong:
        return None
    return round(sum(1 for r in wrong if r["confidence"] >= threshold) / len(wrong), 4)


def load_scored_raw(model_key, temp_label):
    path = os.path.join(SCORED_DIR, f"scored_raw_{model_key}_{temp_label}.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_all_summaries():
    summaries = []
    for fname in sorted(os.listdir(SCORED_DIR)):
        if fname.startswith("scored_summary_") and fname.endswith(".json"):
            with open(os.path.join(SCORED_DIR, fname), encoding="utf-8") as f:
                summaries.append(json.load(f))
    return summaries


def fmt(v, scale=1, dec=4):
    return "--" if v is None else f"{v * scale:.{dec}f}"


def main():
    rows = []
    for s in load_all_summaries():
        model_key = s.get("model", "unknown")
        temperature = s.get("temperature")
        temp_label = "tempDefault" if temperature is None else f"temp{str(temperature).replace('.', '_')}"

        records = load_scored_raw(model_key, temp_label)
        has_conf = any(r.get("confidence") is not None and r.get("correct") is not None for r in records)
        cw_rate = compute_cw_rate(records, CW_THRESHOLD) if has_conf else None

        rows.append({
            "model": MODEL_NAMES.get(model_key, model_key),
            "temp": "Def" if temperature is None else str(temperature),
            "accuracy": s.get("accuracy_overall"),
            "consistency": s.get("avg_consistency"),
            "refusal": s.get("refusal_rate"),
            "latency": s.get("avg_latency_ms"),
            "confidence": s.get("avg_confidence"),
            "ece": s.get("ece"),
            "brier": s.get("brier_score"),
            "cw": cw_rate,
        })
    rows.sort(key=lambda r: r["accuracy"] or 0, reverse=True)

    header = ["Model", "Temperature", "Accuracy (%)", "Avg consistency", "Refusal (%)",
              "Avg latency (ms)", "Avg confidence", "ECE", "Brier", "CW rate (%)"]
    table = [[r["model"], r["temp"], fmt(r["accuracy"], 100, 1), fmt(r["consistency"]),
              fmt(r["refusal"], 100, 1), fmt(r["latency"], 1, 0), fmt(r["confidence"]),
              fmt(r["ece"]), fmt(r["brier"]), fmt(r["cw"], 100, 1)] for r in rows]

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "mcqa_calibration.csv"), "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([header] + table)

    widths = [max(len(r[i]) for r in [header] + table) for i in range(len(header))]
    for r in [header] + table:
        print(r[0].ljust(widths[0] + 2) + "".join(c.rjust(w + 2) for c, w in zip(r[1:], widths[1:])))


if __name__ == "__main__":
    main()
