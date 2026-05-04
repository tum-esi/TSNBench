import json
import numpy as np

mcqa_results = {
    "claude-sonnet-4-5-20250929" : {"accuracy": 95.3, "generator_family": True,  "family": "Claude"},
    "gpt-5-2025-08-07"           : {"accuracy": 95.0, "generator_family": True,  "family": "GPT"},
    "gpt-4o-2024-08-06"          : {"accuracy": 91.8, "generator_family": True,  "family": "GPT"},
    "gpt-4o-mini-2024-07-18"     : {"accuracy": 88.3, "generator_family": True,  "family": "GPT"},
    "o3-2025-04-16"              : {"accuracy": 94.7, "generator_family": True, "family": "GPT"},
    "deepseek-chat"              : {"accuracy": 94.0, "generator_family": False, "family": "DeepSeek"},
    "deepseek-reasoner"          : {"accuracy": 94.7, "generator_family": False, "family": "DeepSeek"},
    "gemini-2.5-flash"           : {"accuracy": 90.1, "generator_family": False, "family": "Gemini"},
    "Llama-3.3-70B-Instruct"     : {"accuracy": 88.9, "generator_family": True,  "family": "Llama"},
    "llama-3.2-1B"               : {"accuracy": 67.4, "generator_family": True,  "family": "Llama"},
    "mistral-medium-2508"        : {"accuracy": 92.1, "generator_family": False, "family": "Mistral"},
    "mistral-large-2512"         : {"accuracy": 92.8, "generator_family": False, "family": "Mistral"},
    "grok-4-1-fast-reasoning"    : {"accuracy": 93.2, "generator_family": False, "family": "Grok"},
    "grok-4-1-fast-non-reasoning": {"accuracy": 91.7, "generator_family": False, "family": "Grok"},
    "Qwen3-8B"                   : {"accuracy": 83.7, "generator_family": False, "family": "Qwen"},
    "ministral-8b-2512"          : {"accuracy": 86.9, "generator_family": False, "family": "Mistral"},
}

generator_acc     = [v["accuracy"] for v in mcqa_results.values() if v["generator_family"]]
non_generator_acc = [v["accuracy"] for v in mcqa_results.values() if not v["generator_family"]]

print("Generator-family models:")
for k, v in mcqa_results.items():
    if v["generator_family"]:
        print(f"  {k:<40} {v['accuracy']:.1f}%")
print(f"  Mean accuracy: {np.mean(generator_acc):.1f}%")

print("\nNon-generator-family models:")
for k, v in mcqa_results.items():
    if not v["generator_family"]:
        print(f"  {k:<40} {v['accuracy']:.1f}%")
print(f"  Mean accuracy: {np.mean(non_generator_acc):.1f}%")

print(f"\nDifference: {np.mean(generator_acc) - np.mean(non_generator_acc):.1f}%")

from scipy.stats import mannwhitneyu

stat, p_value = mannwhitneyu(
    generator_acc,
    non_generator_acc,
    alternative='greater'
)
print(f"\nMann-Whitney U test:")
print(f"  statistic = {stat:.4f}")
print(f"  p-value   = {p_value:.4f}")
if p_value < 0.05:
    print("  Result: Generator-family models score SIGNIFICANTLY higher → contamination concern")
else:
    print("  Result: No significant difference → contamination concern is NOT supported by data")

