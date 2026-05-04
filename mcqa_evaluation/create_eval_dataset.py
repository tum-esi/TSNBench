"""
prepare_eval_dataset.py
=======================
Splits questionset.json into two files:

  eval_questions.json  — question + options only — passed to models
  answer_key.json      — correct answers only — used to score after evaluation

Run this once before running any evaluation script.
"""

import json
import os

# QUESTIONSET_FILE  = "input/tsn_questionset.json"
# EVAL_DIR          = "input_for_evaluation"
# EVAL_QUESTIONS    = os.path.join(EVAL_DIR, "eval_questions.json")
# ANSWER_KEY        = os.path.join(EVAL_DIR, "answer_key.json")

# QUESTIONSET_FILE  = "input/accepted_after_human_review_questions_format.json"
# EVAL_DIR          = "input_for_evaluation"
# EVAL_QUESTIONS    = os.path.join(EVAL_DIR, "accepted_after_human_review_questions_format.json")
# ANSWER_KEY        = os.path.join(EVAL_DIR, "accepted_after_human_review_questions_format_answer_key.json")

QUESTIONSET_FILE  = "input/tsn_21_April_questionset_after_removing_duplicates_numbered.json"
EVAL_DIR          = "input_dataset"
EVAL_QUESTIONS    = os.path.join(EVAL_DIR, "mcqa_tsn_dataset.json")
ANSWER_KEY        = os.path.join(EVAL_DIR, "mcqa_tsn_dataset_answer_key.json")

os.makedirs(EVAL_DIR, exist_ok=True)

with open(QUESTIONSET_FILE, "r", encoding="utf-8") as f:
    raw = json.load(f)

questions = list(raw.values()) if isinstance(raw, dict) else raw

eval_questions = []
answer_key     = {}

for q in questions:
    qid = q.get("question_id")

    # Stripped record — only what the model should see
    stripped = {"question_id": qid, "question": q["question"]}
    for letter in ["A", "B", "C", "D", "E"]:
        if letter in q:
            stripped[letter] = q[letter]

    # Also carry category and level for per-difficulty accuracy reporting
    #stripped["category"] = q.get("category", "")
    stripped["level"]    = q.get("level", "")

    eval_questions.append(stripped)

    # Answer key — stored separately, never passed to model
    answer_key[str(qid)] = {
        "correct_answer": q.get("answer", "").split(":")[0].strip(),
        "answer_full"   : q.get("answer", ""),
        "level"         : q.get("level", ""),
        "category"      : q.get("category", ""),
    }

# Save stripped questions
with open(EVAL_QUESTIONS, "w", encoding="utf-8") as f:
    json.dump(eval_questions, f, indent=2, ensure_ascii=False)

# Save answer key
with open(ANSWER_KEY, "w", encoding="utf-8") as f:
    json.dump(answer_key, f, indent=2, ensure_ascii=False)

print(f"eval_questions.json : {len(eval_questions)} questions — no answers, no explanations")
print(f"answer_key.json     : {len(answer_key)} entries — correct answers only")
print(f"Saved to: {EVAL_DIR}/")

# Verify no leakage in eval_questions.json
FORBIDDEN = ["answer", "explanation", "source_sentence", "correct_option",
             "generator_model", "paper_id", "votes", "drop_reason"]
leakage_found = False
for q in eval_questions:
    for field in FORBIDDEN:
        if field in q:
            print(f"LEAKAGE: field '{field}' found in question_id={q['question_id']}")
            leakage_found = True
if not leakage_found:
    print("Leakage check passed — no forbidden fields in eval_questions.json")