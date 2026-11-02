# TSNBench: Benchmarking LLM Proficiency in Time-Sensitive Networking

TSNBench evaluates large language models (LLMs) on Time-Sensitive Networking (TSN) with two tasks:

- **Multiple-choice question answering (MCQA):** 939 questions on TSN standards and mechanisms.
- **Open-ended worst-case delay (WCD) estimation:** given a topology, a set of flows and their routes, the model computes the WCD of every flow under the Credit-Based Shaper (CBS) or Cyclic Queuing and Forwarding (CQF). There are 100 synthetic test cases (TC1–TC100) and 10 test cases on the NASA Orion Crew Exploration Vehicle (CEV) network (TC101–TC110).

The repository contains the dataset, the scripts that query the models, the raw model responses used in the paper, and the scripts that compute the tables and figures from these responses.

The dataset is also available on Kaggle: [https://doi.org/10.34740/kaggle/ds/13019279](https://doi.org/10.34740/kaggle/ds/13019279).

## Repository structure

```
TSNBench/
├── dataset/
│   ├── mcqa/                            MCQA questions and answer key
│   └── open_ended/                      TC1–TC100 and orion/TC101–TC110
├── mcqa_evaluation/
│   ├── config.py                        models, API clients and prices
│   ├── run_mcqa.py                      runs the MCQA questions on the models
│   ├── score_mcqa.py                    scores the runs against the answer key
│   ├── input_dataset/                   questions and answer key read by the scripts
│   └── output/results/                  raw model responses (raw_<model>_<temp>.json)
├── open_end_evaluation/
│   ├── config.py                        models, API clients and prices
│   ├── prompts_cbs.py, prompts_cqf.py   prompts
│   ├── run_wcd.py                       runs the open-ended questions on the models
│   ├── input/test_cases/open_ended/     test cases read by the scripts
│   └── output/                          raw model responses and ground truth
├── evaluation_scripts/                  tables and figures of the paper
├── requirements.txt
├── .env.example                         API key names
└── LICENSE
```

## Dataset

### MCQA

`dataset/mcqa/mcqa_tsn_dataset.json` holds the 939 questions (`question_id`, `question`, options `A`–`D`, `level`). `mcqa_tsn_dataset_answer_key.json` maps each `question_id` to the correct option and its category:


### Open-ended test cases

Each test case has three CSV files:

- `TC<n>_topo.txt`: switches and end nodes (`vertex,<name>,<ports>`) and links (`edge,<switch>.<port>,<node>`)
- `TC<n>_flows.txt`: `<flow id>,<source>,<destination>,<period (µs)>,<deadline (µs)>,<payload (bytes)>`
- `TC<n>_route.txt`: `<flow id>,<source>,<destination>,<hop count>,<path>`

| Test cases | Topology |
|---|---|
| TC1–TC40, TC52–TC70 | ring-based (single ring or connected rings) |
| TC41–TC51 | one switch |
| TC71–TC100 | medium mesh |
| TC101–TC110 | NASA Orion CEV |

## Setup

Python 3.9 to 3.12. The pinned versions in `requirements.txt` were tested with Python 3.9, numpy 2.0.2 and scipy 1.13.1 have no wheels for Python 3.13.

```bash
git clone https://github.com/tum-esi/TSNBench.git
cd TSNBench
pip install -r requirements.txt
```

API keys are only needed to query the models again. All tables and figures can be computed from the raw responses in this repository.

```bash
cp .env.example .env    # Windows: copy .env.example .env; then add your keys
```

## Reproducing the paper results

Run each script from its own folder. The tables are printed and written to `results/` in that folder; the figures are saved next to the script.

Run the MCQA scoring first; the calibration and reliability scripts read the scored files it writes to `mcqa_evaluation/output/results/scored/`:

```bash
cd mcqa_evaluation
python score_mcqa.py
cd ../evaluation_scripts
python mae_cb_fc_ablation.py
```

| Script | Paper | Output |
|---|---|---|
| `mcqa_evaluation/score_mcqa.py` | Tables 2 and 10 (accuracy, consistency) | `mcqa_results.csv`, `mcqa_temperatures.csv` |
| `mcqa_calibration.py` | Table 2 (ECE, Brier score, CW rate) | `mcqa_calibration.csv` |
| `reliability_plot_selected_models.py` | Figure 4 | `reliability_diagrams_selected.pdf/.png` |
| `reliability_plot_mcqa.py` | Figure 16 | `reliability_diagrams_all.pdf/.png` |
| `mcqa_contamination.py` | Contamination analysis (Section 4.1) | `mcqa_contamination.json` |
| `mae_cb_fc_ablation.py` | Table 3, Tables 13 and 14 | `mae_cbs.csv`, `mae_cqf.csv` |
| `error_thresholds.py` | Table 4 (`SETTING = "CB"`), Table 15 (`SETTING = "FC"`) | `abs_error_thresholds_<setting>.csv` |
| `cost_latency.py` | Table 11 | `cost_latency.csv` |
| `prompt_sensitivity.py` | Tables 16–19 | `prompt_sensitivity_*_<mech>.csv` |
| `per_tc_one_switch.py` | Tables 20 and 21 | `one_switch_per_tc_mae_<mech>_<setting>.csv` |
| `per_tc_medium_mesh.py` | Tables 22 and 23 | `medium_mesh_per_tc_mae_<mech>_<setting>.csv` |
| `orion_cbs.py` | Tables 24, 26–29 (Orion, CBS) | `orion_cbs_*.csv` |
| `orion_cqf.py` | Tables 25, 30–33 (Orion, CQF) | `orion_cqf_*.csv` |
| `prompt_sensitivity_orion.py` | Orion, prompt sensitivity (not in the paper) | `prompt_sensitivity_orion_*.csv` |

All scripts except the first are in `evaluation_scripts/`. The per-TC and threshold scripts compute the closed-book (CB) values by default; set `SETTING = "FC"` at the top of the script for formula-in-context (FC).

### Scoring

- One temperature per model: 0.0 if available, else 0.7, else the provider default (models without a temperature setting).
- Responses are re-parsed with `evaluation_scripts/wcd_parser.py`. Flow IDs are normalised (`15`, `f15`, `Flow 15` → `F15`), and a run that uses a flow ID that is not in the test case is dropped.
- The predicted WCD of a flow is the average over the valid runs of the test case (up to three).
- A test case is valid if at least 80% of its flows are predicted and the prediction is not all zero.
- The MAE of a test case is the mean absolute error over its flows. A model is reported only with at least 50 valid test cases (5 of 10 for Orion).
- Grok 4.1 Fast, Grok 4.1 Fast (Non-Reasoning) and both DeepSeek-V3.2 models were no longer available for the FC and prompt-sensitivity runs and are reported for CB only.

## Running the models

### MCQA

```bash
cd mcqa_evaluation
python run_mcqa.py
python run_mcqa.py --models gpt-5-2025-08-07 o3-2025-04-16 --temperatures 0.0
```

Every question is asked three times per model and temperature (0.0 and 0.7 by default). Models without a temperature setting are run only with their default. The responses are written to `output/results/raw_<model>_<temp>.json`, with token counts and cost in `summary_<model>_<temp>.json`.

### Open-ended WCD estimation

One mechanism and one prompt per run:

| `--variant` | Prompt |
|---|---|
| `closed_book` | closed-book (CB) |
| `formula` | formula-in-context (FC): CB prompt plus the delay equations |
| `variant_a` | prompt sensitivity, paraphrased instructions |
| `variant_b` | prompt sensitivity, plain-text answer instead of JSON |

```bash
cd open_end_evaluation
python run_wcd.py --mechanism CBS --variant closed_book --out-dir output/CBS
python run_wcd.py --mechanism CQF --variant formula --out-dir output/CQF/camera_ready/all/formula
python run_wcd.py --mechanism CBS --variant closed_book \
    --tc-dir input/test_cases/open_ended/orion --tcs TC101 TC102 TC103 TC104 TC105 TC106 TC107 TC108 TC109 TC110 \
    --out-dir output/CBS/camera_ready/orion/closed_book
```

Other options: `--tcs` (default TC1–TC100), `--runs` (default 3), `--temperature` (default 0.0), `--models`, and `--print-prompt` to print the prompt of the first test case without calling a model. The responses are written to `<out-dir>/raw_openend_<model>_<mechanism>_<temp>.json`, with token counts and cost in `summary_openend_<model>_<mechanism>_<temp>.json`.

### Raw responses used in the paper

`open_end_evaluation/output/<mechanism>/`:

| Folder | Setting |
|---|---|
| `.`, `one_switch_topo/`, `medium_mesh_topo/`, `last_run/`, `tc5/` | CB, TC1–TC100 |
| `camera_ready/all/nc_equation/` (CBS), `camera_ready/all/formula/` (CQF) | FC, TC1–TC100 |
| `camera_ready/all/prompt_one/`, `camera_ready/all/prompt_two/` | Variant A, Variant B |
| `camera_ready/orion/closed_book/` | Orion, CB |
| `camera_ready/orion/nc_equation/` (CBS), `camera_ready/orion/formula/` (CQF) | Orion, FC |
| `camera_ready/orion/prompt_one/`, `camera_ready/orion/prompt_two/` | Orion, Variant A, Variant B |

The CB runs on TC1–TC100 are split over the five CB folders; the scripts merge the records of all five per model.

## Ground truth

`open_end_evaluation/output/ground_truth/<mechanism>/WCD_open_ended_<mechanism>.json` (TC1–TC100) and `..._orion.json` (TC101–TC110) give the WCD of every flow in µs.

- **CBS:** worst-case end-to-end delays from a network-calculus analysis of CBS.
- **CQF:** WCD = (n + 1) · T + n · d<sub>sw</sub> + d<sub>prop</sub> + d<sub>sync</sub>, with n the number of switches on the route, cycle duration T = 50 µs, and 1 µs for the switching delay, the propagation delay and the time-synchronisation error.

## Models

| Model | Provider | Key in `config.py` |
|---|---|---|
| o3 | OpenAI | `o3-2025-04-16` |
| GPT-5 | OpenAI | `gpt-5-2025-08-07` |
| GPT-4o | OpenAI | `gpt-4o-2024-08-06` |
| GPT-4o mini | OpenAI | `gpt-4o-mini-2024-07-18` |
| Claude Sonnet 4.5 | Anthropic | `claude-sonnet-4-5-20250929` |
| Gemini 2.5 Flash | Google | `gemini-2.5-flash` |
| Grok 4.1 Fast | xAI | `grok-4-1-fast-reasoning` |
| Grok 4.1 Fast (Non-Reasoning) | xAI | `grok-4-1-fast-non-reasoning` |
| DeepSeek-V3.2 (Thinking) | DeepSeek | `deepseek-reasoner` |
| DeepSeek-V3.2 (Non-Thinking) | DeepSeek | `deepseek-chat` |
| Mistral Large 3 | Mistral | `mistral-large-2512` |
| Mistral Medium 3.1 | Mistral | `mistral-medium-2508` |
| Ministral 3 8B | Mistral | `ministral-8b-2512` |
| Llama 3.3 70B | Meta (Hugging Face) | `Llama-3.3-70B-Instruct` |
| Llama 3.2 1B | Meta (Hugging Face) | `Llama-3.2-1B` |
| Qwen3 8B | Alibaba (Hugging Face) | `Qwen3-8B` |

## Citation

```bibtex
@inproceedings{debnath2026tsnbench,
  title     = {{TSNBench}: Benchmarking {LLM} Proficiency in Time-Sensitive Networking},
  author    = {Debnath, Rubi and Bujosa, Daniel and Zhao, Luxi and Craciunas, Silviu S. and Pop, Paul and Steinhorst, Sebastian},
  booktitle = {The 40th Conference on Neural Information Processing Systems (NeurIPS 2026), Track on Evaluations and Datasets},
  year      = {2026}
}
```

## License

The code is released under the MIT license, see `LICENSE`. The dataset is released under CC BY 4.0.
