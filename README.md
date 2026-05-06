## TSNBench: Benchmarking LLM Proficiency in Time-Sensitive Networking

> We present TSNBench and demonstrate that multiple-choice benchmarks overestimate LLM capability in safety-critical networking due to poor performance on open-ended timing analysis tasks.

---

## Overview

TSNBench is a benchmark for evaluating large language models (LLMs) on Time-Sensitive Networking (TSN) tasks. It covers two evaluation paradigms:

- **Multiple-Choice Question Answering (MCQA)** - tests conceptual and lexical understanding of TSN standards across difficulty levels.
- **Open-Ended WCD Estimation** - tests whether LLMs can compute Worst-Case Delay (WCD) for Credit-Based Shaper (CBS) and Cyclic Queuing and Forwarding (CQF) scheduling mechanisms from network topology, flows, and routes descriptions.

Our results show that strong MCQA performance does not transfer to open-ended timing analysis, highlighting a critical gap in current LLM capabilities for safety-critical networking.

---

## Repository Structure
```
TSNBench/
├── dataset/
│   ├── mcqa/                          # MCQA questions and MCQA answer keys (JSON)
│   └── open_ended/                    # Open-ended questions (flows, route, and topology) (txt files)
│
├── ground_truth/
│   ├── CBS/
│   │   ├── WCD_open_ended_CBS.json    # Ground truth WCD values for CBS for 100 test cases (TCs)
│   │   └── topology_wise_WCD/         # Per-topology WCD JSON files
│   └── CQF/
│       ├── WCD_open_ended_CQF.json    # Ground truth WCD values for CQF for 100 test cases (TCs)
│       └── topology_wise_WCD/         # Per-topology WCD JSON files
│
├── mcqa_generator/
│   ├── papers/                        # Research documents used for multiple-choice question generation
│   │                                  # (research documents are not included in this repository)
│   ├── output/                        # Generated MCQA questions
│   ├── generator.py                   # Main MCQA question generator
│   └── lexicon_extractor.py           # Extracts TSN-specific keywords from research documents
│
├── mcqa_evaluation/
│   ├── config.py                      # Model configuration files (endpoints)
│   ├── output/                        # Raw and scored results
│   └── *.py                           # Evaluation scripts
│
├── open_end_evaluation/
│   ├── output/                              # Raw model responses and scored results
│   │   ├── CBS/                             # CBS raw files and scored files
│   │   └── CQF/                             # CQF raw files and scored files
│   ├── config.py                            # Model configuration (model endpoints, parameters)
│   ├── CBS.py                               # Prompt template for CBS open-ended evaluation
│   ├── CQF.py                               # Prompt template for CQF open-ended evaluation
│   ├── evaluate_open_end_questionset.py     # Runs open-ended evaluation for all models
│   └── error_calculation.py                 # Computes MAE, MAPE, and Median across all TCs
│
├── evaluation/
│   ├── cost_and_latency_of_mcqa_open_end.py   # Cost and latency reporting across all models
│   ├── mae_mape_std_dev.py                    # MAE, MAPE, and std dev across 100 TCs
│   ├── mcqa_cbs_cqf.py                        # Combined MCQA and open-ended comparison plot
│   └── reliability_plot.py                    # reliability plot (Figure 14)
│
│
├── .env.example                             # Example environment file for API keys
└── requirements.txt                         # Python dependencies
```

## Setup

1. Clone the repository:

```bash
git clone <CODE_REPOSITORY_URL>
cd TSNBench
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure API keys by copying the example environment file and filling in your keys:

```bash
cp .env.example .env
```

The `.env.example` file contains the following variables:

```
ANTHROPIC_API_KEY=
DEEPSEEK_API_KEY=
GOOGLE_API_KEY=
VERTEX_API_KEY=
OPENAI_API_KEY=
XAI_API_KEY=
HF_TOKEN=
MISTRAL_API_KEY=
```

## Dataset

### MCQA
- Located in `dataset/mcqa/`
- Each question has four to five answer choices (A–E) with a single correct answer
- Answer key: `dataset/mcqa/mcqa_tsn_dataset_answer_key.json`

### Open-Ended dataset
- Located in `dataset/open_ended/`
- 100 test cases (TCs) across three topology types: one-switch, medium mesh, ring.

### Ground Truth
- Located in `ground_truth/CBS/` and `ground_truth/CQF/`
- WCD values computed via Network Calculus (NC) and mathematical analysis.
- Per-topology breakdowns available in `topology_wise_wcd/`

### Evaluation Results
- Located in `evaluation_results`
- `mcqa_evaluation.zip` contains the MCQA evaluation results.
- `open_ended_evaluation.zip` contains the open-ended evaluation results.

> **Note:** The research documents and paper drafts used during dataset construction are not included in this repository.

---

## Models Evaluated

| Model | Provider |
|---|---|
| o3, GPT-4o, GPT-4o mini, GPT-5 | OpenAI |
| Claude Sonnet 4.5 | Anthropic |
| Gemini 2.5 Flash | Google |
| DeepSeek-V3.2 (Thinking / Non-Thinking) | DeepSeek |
| Grok 4.1 Fast (Reasoning / Non-Reasoning) | xAI |
| Llama 3.3 70B, Llama 3.2 1B | Meta |
| Mistral Medium 3.1, Mistral Large 3, Ministral 3 8B | Mistral |
| Qwen3 8B | Alibaba |
---

## Running the Benchmark

### Step 1 — Run MCQA Evaluation

```bash
cd mcqa_evaluation
python evaluate_questionset.py
```

Raw results are saved to `mcqa_evaluation/output/raw_{model}_{temp}.json`.

### Step 2 — Score MCQA Results

```bash
python check_accuracy.py
```

Outputs per-model scored summaries and a full comparison table to `output/full_mcqa_summary.json`.

### Step 3 — Run Open-Ended Evaluation

```bash
cd open_end_evaluation
# In config.py and evaluate_open_end_questionset.py, set PROMPT_TYPE to "CBS" or "CQF" before running
python evaluate_open_end_questionset.py
```

Raw results are saved to `output/CBS/raw_openend_{model}_CBS_{temp}.json` and equivalently for CQF.

### Step 4 — Score Open-Ended Results

```bash
python error_calculation.py
```

Scored files and summary JSONs are saved to `output/CBS/score/` and `output/CQF/score/`.

### Step 5 — Compute Performance Metrics

```bash
cd evaluation
python cost_and_latency_of_mcqa_open_end.py
python mae_mape_std_dev.py
python mcqa_cbs_cqf.py
```

Outputs MAE, MAPE, and Median MAE across all 100 TCs per model.

---

## Metrics

### MCQA
- **Overall Accuracy** — majority vote across runs per question
- **Avg Consistency** — agreement across runs per question

### Open-Ended WCD Estimation
- **MAE (µs)** — Mean Absolute Error
- **MAPE (%)** — Mean Absolute Percentage Error
- **Median MAE** — robust central tendency across TCs
---

## Exclusion Criteria

A model is excluded from open-ended results (`--`) if:

1. It successfully evaluated fewer than **50 out of 100 TCs**
2. Within each TC, fewer than **80% of flows** received a WCD estimate
3. All predicted WCD values were zero (trivial failure)

---

## Reproducibility Notes

- All evaluations are primarily run at **temperature = 0** to maximize determinism and reproducibility
- A secondary run at **temperature = 0.7** was conducted for selected models to assess sensitivity to sampling randomness
- Models that do not support explicit temperature control (e.g. o3) use the provider's default sampling, labeled `tempDefault` in result files
- Each TC is run **up to 3 times** and results are averaged to reduce variance
- Raw model responses are saved in full for all models — no post-processing beyond JSON parsing and WCD extraction


## License

This repository is released for research purposes. See `LICENSE` for details.