"""MCQA accuracy of the generator model families vs the other models."""

import json
import os

import numpy as np
from scipy.stats import mannwhitneyu

GENERATOR_FAMILIES = {"Claude", "GPT", "Llama"}
OUT_FILE = "results/mcqa_contamination.json"

# accuracy (%) from Table 2
MCQA_ACCURACY = {
    "claude-sonnet-4-5-20250929": ("Claude", 95.3),
    "gpt-5-2025-08-07": ("GPT", 95.0),
    "gpt-4o-2024-08-06": ("GPT", 91.8),
    "gpt-4o-mini-2024-07-18": ("GPT", 88.3),
    "o3-2025-04-16": ("GPT", 94.7),
    "Llama-3.3-70B-Instruct": ("Llama", 88.9),
    "Llama-3.2-1B": ("Llama", 67.4),
    "deepseek-chat": ("DeepSeek", 94.0),
    "deepseek-reasoner": ("DeepSeek", 94.7),
    "gemini-2.5-flash": ("Gemini", 90.1),
    "mistral-medium-2508": ("Mistral", 92.1),
    "mistral-large-2512": ("Mistral", 92.8),
    "ministral-8b-2512": ("Mistral", 86.9),
    "grok-4-1-fast-reasoning": ("Grok", 93.2),
    "grok-4-1-fast-non-reasoning": ("Grok", 91.7),
    "Qwen3-8B": ("Qwen", 83.7),
}


def main():
    gen = [acc for fam, acc in MCQA_ACCURACY.values() if fam in GENERATOR_FAMILIES]
    other = [acc for fam, acc in MCQA_ACCURACY.values() if fam not in GENERATOR_FAMILIES]
    test = mannwhitneyu(gen, other, alternative="greater")

    result = {
        "generator_families": sorted(GENERATOR_FAMILIES),
        "generator_models": len(gen),
        "generator_mean_accuracy": round(float(np.mean(gen)), 1),
        "other_models": len(other),
        "other_mean_accuracy": round(float(np.mean(other)), 1),
        "difference": round(float(np.mean(gen) - np.mean(other)), 1),
        "mann_whitney_u": float(test.statistic),
        "p_value": round(float(test.pvalue), 4),
    }

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    for k, v in result.items():
        print(f"{k:<26} {v}")


if __name__ == "__main__":
    main()
