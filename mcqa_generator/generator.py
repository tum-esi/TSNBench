import os
import re
import json
import time
import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

import sys
import warnings
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import pdfplumber
from openai import OpenAI
import anthropic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("generator.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

anthropic_client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

llama_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN"),
)

GENERATORS = {
    "claude-sonnet" : "claude-sonnet-4-20250514",
    "gpt-4o-mini" : "gpt-4o-mini",
    "llama-3.1-70b" : "meta-llama/Llama-3.1-70B-Instruct:scaleway",
}


DIFFICULTY_LEVELS = ["easy", "medium", "hard"]
QUESTIONS_PER_LEVEL = 5

DIFFICULTY_CATEGORY = {
    "easy"  : "Research Paper - Easy",
    "medium": "Research Paper - Medium",
    "hard"  : "Research Paper - Hard",
}

EASY_PROMPT = """You are an expert in Time-Sensitive Networking (TSN).

Your task is to extract {{TARGET}} EASY multiple-choice questions from the provided research paper text.

WHAT EASY MEANS:
Easy questions test direct factual knowledge. The answer is explicitly stated in the text as a specific fact, value, name, or definition. A reader with basic TSN knowledge can answer these by recalling a clearly stated fact.

QUESTION RULES:
- Every question must be fully self-contained
- The question must be answerable from TSN domain knowledge alone
- Do NOT reference the paper, the authors, the study, the approach, or any figures or tables
- Do NOT use phrases like "according to this paper", "in this study", "the proposed method", "the authors", "as shown", "in the above"
- Expand all acronyms on first use within each question and its options
- All questions must be about different facts — no overlap with each other

OPTION RULES:
- Generate either 4 or 5 answer options per question
- All distractors must be plausible but clearly incorrect to a domain expert
- Use 5 options when strong plausible distractors exist in the content
- Use 4 options when content only supports 4 strong distractors
- Never force a weak 5th distractor just to reach 5
- Never make the correct answer obvious from the question phrasing

SOURCE SENTENCE RULES:
- For each question identify the exact sentence from the text that justifies the correct answer
- Copy it verbatim — do not paraphrase

OUTPUT FORMAT:
Return a valid JSON object exactly as follows — no extra text, no markdown:

{
  "questions": [
    {
      "question": "...",
      "option_1": "...",
      "option_2": "...",
      "option_3": "...",
      "option_4": "...",
      "correct_option": 2,
      "answer": "option_2: ...",
      "explanation": "...",
      "source_sentence": "exact verbatim sentence from text",
      "difficulty": "easy",
      "category": "Research Paper - Easy",
      "generator_model": "{{GENERATOR_MODEL}}",
      "paper_id": "{{PAPER_ID}}",
      "source": "{{SOURCE}}"
    }
  ]
}

NOTE: correct_option is an integer (1-4).
IMPORTANT: Return only the JSON object. Nothing else."""


MEDIUM_PROMPT = """You are an expert in Time-Sensitive Networking (TSN).

Your task is to extract {{TARGET}} MEDIUM multiple-choice questions from the provided research paper text.

WHAT MEDIUM MEANS:
Medium questions test conceptual understanding. The reader must understand how a mechanism works, why a design choice was made, or what the relationship is between two components or parameters. The answer cannot be found by looking up a single fact — it requires understanding the described system.

QUESTION RULES:
- Every question must be fully self-contained
- The question must be answerable from TSN domain knowledge alone
- Do NOT reference the paper, the authors, the study, the approach, or any figures or tables
- Do NOT use phrases like "according to this paper", "in this study", "the proposed method", "the authors", "as shown", "in the above"
- Do NOT ask questions that are answered by a single number or name lookup — those are easy questions
- Expand all acronyms on first use within each question and its options
- All questions must test different concepts — no overlap with each other

OPTION RULES:
- Generate 4 answer options per question
- All distractors must be plausible and test genuine understanding
- Use 5 options when strong plausible distractors exist in the content
- Use 4 options when content only supports 4 strong distractors
- Never force a weak 5th distractor just to reach 5
- Distractors should represent common misconceptions or partial understandings

SOURCE SENTENCE RULES:
- For each question identify the exact sentence from the text that justifies the correct answer
- Copy it verbatim — do not paraphrase

OUTPUT FORMAT:
Return a valid JSON object exactly as follows — no extra text, no markdown:

{
  "questions": [
    {
      "question": "...",
      "option_1": "...",
      "option_2": "...",
      "option_3": "...",
      "option_4": "...",
      "correct_option": 2,
      "answer": "option_2: ...",
      "explanation": "...",
      "source_sentence": "exact verbatim sentence from text",
      "difficulty": "medium",
      "category": "Research Paper - Medium",
      "generator_model": "{{GENERATOR_MODEL}}",
      "paper_id": "{{PAPER_ID}}",
      "source": "{{SOURCE}}"
    }
  ]
}

NOTE: correct_option is an integer (1-4).
IMPORTANT: Return only the JSON object. Nothing else."""


HARD_PROMPT = """You are an expert in Time-Sensitive Networking (TSN).

Your task is to extract {{TARGET}} HARD multiple-choice questions from the provided research paper text.

WHAT HARD MEANS:
Hard questions test deep reasoning and application. The reader must reason about failure conditions, limitations, trade-offs, edge cases, or implications that require combining multiple concepts from the text. The correct answer is not directly stated — it must be derived through reasoning over what is described.

QUESTION RULES:
- Every question must be fully self-contained
- The question must be answerable from TSN domain knowledge alone
- Do NOT reference the paper, the authors, the study, the approach, or any figures or tables
- Do NOT use phrases like "according to this paper", "in this study", "the proposed method", "the authors", "as shown", "in the above"
- Do NOT ask questions answered by recalling a single fact or understanding a single mechanism — those are easy or medium questions
- Expand all acronyms on first use within each question and its options
- All questions must test different reasoning challenges — no overlap with each other

OPTION RULES:
- Generate 4 answer options per question
- All distractors must be sophisticated — plausible to someone with good TSN knowledge but lacking deep understanding
- Use 5 options when strong plausible distractors exist in the content
- Use 4 options when content only supports 4 strong distractors
- Never force a weak 5th distractor just to reach 5
- Distractors should represent technically reasonable but incorrect conclusions

SOURCE SENTENCE RULES:
- For each question identify the exact sentence from the text that best grounds the correct answer
- Copy it verbatim — do not paraphrase

OUTPUT FORMAT:
Return a valid JSON object exactly as follows — no extra text, no markdown:

{
  "questions": [
    {
      "question": "...",
      "option_1": "...",
      "option_2": "...",
      "option_3": "...",
      "option_4": "...",
      "correct_option": 2,
      "answer": "option_2: ...",
      "explanation": "...",
      "source_sentence": "exact verbatim sentence from text",
      "difficulty": "hard",
      "category": "Research Paper - Hard",
      "generator_model": "{{GENERATOR_MODEL}}",
      "paper_id": "{{PAPER_ID}}",
      "source": "{{SOURCE}}"
    }
  ]
}

NOTE: correct_option is an integer (1-4).
IMPORTANT: Return only the JSON object. Nothing else."""


GENERATOR_PROMPTS = {
    "easy"  : EASY_PROMPT,
    "medium": MEDIUM_PROMPT,
    "hard"  : HARD_PROMPT,
}

def parse_json_response(raw: str) -> Optional[dict]:
    if not raw:
        return None
    clean = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass
    start = clean.find("{")
    end   = clean.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(clean[start:end + 1])
        except json.JSONDecodeError as e:
            log.warning(f"JSON parse error: {e}\nRaw: {raw[:300]}")

    return None


def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        log.error(f"Failed to extract text from {pdf_path}: {e}")
    return text.strip()


def clean_paper_text(text: str) -> str:
    abstract_match = re.search(r'\bAbstract\b', text, re.IGNORECASE)
    if abstract_match:
        removed = abstract_match.start()
        text = text[abstract_match.start():]
        log.info(f"Stripped before Abstract ({removed} chars removed)")
    else:
        intro_match = re.search(r'\bIntroduction\b', text, re.IGNORECASE)
        if intro_match:
            text = text[intro_match.start():]
            log.info(f"Abstract not found — stripped before Introduction")
        else:
            log.warning("Neither Abstract nor Introduction found — using full text")
    text = re.sub(
        r'(?:Index Terms?|Keywords?|Key [Ww]ords?)\s*[—:\-]?[^\n]*(?:\n(?!\n)[^\n]*)*',
        '', text, flags=re.IGNORECASE
    )
    text = re.sub(r'https?://[^\s)>\]]+', '', text)
    text = re.sub(r'www\.[^\s)>\]]+', '', text)
    text = re.sub(r'\n\s*(?:github\.com|gitlab\.com|bitbucket\.org|arxiv\.org)[^\n]*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?:available|code|dataset|repository|repo|source)\s+at[^\n]*\n', '', text, flags=re.IGNORECASE)
    noise_pattern = re.compile(
        r'\n\s*(?:(?:\d+\.?\s+|[IVXLC]+\.?\s+)?(?:'
        r'References?|Bibliography|Acknowledgements?|Acknowledgments?|'
        r'Biography|Biographies|About the [Aa]uthor|Author [Bb]iograph|'
        r'Appendix|Appendices'
        r'))\s*\n.*',
        re.IGNORECASE | re.DOTALL
    )
    text = noise_pattern.sub("", text)
    text = re.sub(r'\n\[\d+\]\s+[A-Z].*', '', text, flags=re.DOTALL)
    text = re.sub(
        r'(?:Fig(?:ure)?|Table|Algorithm|Listing|Pseudocode|Alg\.?)'
        r'\s*\.?\s*\d+[.:][^\n]+\n',
        '', text, flags=re.IGNORECASE
    )
    CODE_INDICATORS = re.compile(
        r'(?:'
        r'\bif\b.+:\s*$|'          
        r'\bfor\b.+:\s*$|'          
        r'\bwhile\b.+:\s*$|'        
        r'\bdef\b\s+\w+\s*\(|'     
        r'\breturn\b\s+\w|'         
        r'\bprint\s*\(|'            
        r'\w+\s*[+\-\*/]=\s*\w|'   
        r'\w+\s*:=\s*\w|'           
        r'\w+\s*<-\s*\w|'           
        r'=>|->'                    
        r')',
        re.IGNORECASE
    )

    clean_lines = []
    i = 0
    lines = text.split('\n')
    while i < len(lines):
        line = lines[i]
        if line.startswith('    ') or line.startswith('\t'):
            if CODE_INDICATORS.search(line):
                while i < len(lines) and (
                    lines[i].startswith('    ') or
                    lines[i].startswith('\t') or
                    lines[i].strip() == ''
                ):
                    i += 1
                continue
        clean_lines.append(line)
        i += 1
    text = '\n'.join(clean_lines)
    log.info(f"Paper cleaned: {len(text)} chars remaining")
    return text.strip()


def chunk_text_by_section(text: str, max_tokens: int = 3000) -> list[str]:
    max_chars = max_tokens * 4

    section_pattern = re.compile(
        r'\n(?=(?:Abstract|Introduction|Background|Related Work|'
        r'Methodology|Method|Approach|System Model|'
        r'Results|Experiments|Evaluation|Discussion|'
        r'Conclusion|References)\b)',
        re.IGNORECASE
    )
    sections = section_pattern.split(text)
    sections = [s.strip() for s in sections if s.strip()]

    chunks = []
    current_chunk = ""

    for section in sections:
        if len(current_chunk) + len(section) <= max_chars:
            current_chunk += "\n\n" + section
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            if len(section) > max_chars:
                paragraphs = section.split("\n\n")
                para_chunk = ""
                for para in paragraphs:
                    if len(para_chunk) + len(para) <= max_chars:
                        para_chunk += "\n\n" + para
                    else:
                        if para_chunk:
                            chunks.append(para_chunk.strip())
                        para_chunk = para
                if para_chunk:
                    chunks.append(para_chunk.strip())
            else:
                current_chunk = section
    if current_chunk:
        chunks.append(current_chunk.strip())

    return [c for c in chunks if c]


def call_claude(system: str, user: str, max_tokens: int = 4096) -> str:
    for attempt in range(3):
        try:
            response = anthropic_client.messages.create(
                model=GENERATORS["claude-sonnet"],
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}]
            )
            return response.content[0].text
        except Exception as e:
            log.warning(f"Claude attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)
    return ""


def call_openai_gpt(system: str, user: str, max_tokens: int = 4096) -> str:
    for attempt in range(3):
        try:
            response = openai_client.chat.completions.create(
                model=GENERATORS["gpt-4o-mini"],
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            log.warning(f"GPT attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)
    return ""


def call_llama(system: str, user: str, max_tokens: int = 4096) -> str:
    for attempt in range(3):
        try:
            response = llama_client.chat.completions.create(
                model=GENERATORS["llama-3.1-70b"],
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            log.warning(f"Llama attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)
    return ""


def call_generator(generator_key: str, system: str, user: str, max_tokens: int = 4096) -> str:
    if generator_key == "claude-sonnet":
        return call_claude(system, user, max_tokens)
    elif generator_key == "gpt-4o-mini":
        return call_openai_gpt(system, user, max_tokens)
    elif generator_key == "llama-3.1-70b":
        return call_llama(system, user, max_tokens)
    else:
        raise ValueError(f"Unknown generator: {generator_key}")

def generate_questions(
    chunk        : str,
    paper_id     : str,
    generator_key: str,
    difficulty   : str,
    target       : int,
    start_id     : int = 1,
    source       : str = "arxiv"
) -> list[dict]:
    system = GENERATOR_PROMPTS[difficulty] \
        .replace("{{GENERATOR_MODEL}}", generator_key) \
        .replace("{{PAPER_ID}}", paper_id) \
        .replace("{{SOURCE}}", source) \
        .replace("{{TARGET}}", str(target))

    user = f"Extract {target} {difficulty.upper()} questions from the following research paper text:\n\n{chunk}"
    raw  = call_generator(generator_key, system, user)

    if not raw:
        log.warning(f"Empty response from {generator_key} for {paper_id} [{difficulty}]")
        return []

    parsed = parse_json_response(raw)
    if not parsed or "questions" not in parsed:
        log.warning(f"Invalid response from {generator_key} for {paper_id} [{difficulty}]")
        return []

    questions = parsed["questions"]

    # Assign numeric question IDs and metadata
    for i, q in enumerate(questions):
        q["question_id"]     = start_id + i
        q["generator_model"] = generator_key
        q["paper_id"]        = paper_id
        q["source"]          = source
        q["difficulty"]      = difficulty
        q["category"]        = DIFFICULTY_CATEGORY[difficulty]

    log.info(f"Generated {len(questions)} {difficulty} questions | {paper_id} | {generator_key}")
    return questions

def process_paper(
    pdf_path     : str,
    paper_id     : str,
    generator_key: str,
    start_id     : int = 1,
    source       : str = "arxiv"
) -> list[dict]:

    log.info(f"{'='*60}")
    log.info(f"Paper: {paper_id} | Generator: {generator_key} | Source: {source} | Start ID: {start_id}")

    text = extract_text_from_pdf(pdf_path)
    if not text:
        log.error(f"No text extracted from {pdf_path}")
        return []

    text   = clean_paper_text(text)
    chunks = chunk_text_by_section(text)
    log.info(f"{paper_id}: {len(chunks)} chunks")

    full_text = "\n\n".join(chunks)

    all_questions = []
    current_id    = start_id

    for difficulty in DIFFICULTY_LEVELS:
        log.info(f"  Generating {QUESTIONS_PER_LEVEL} {difficulty} questions for {paper_id}")

        questions = generate_questions(
            chunk         = full_text,
            paper_id      = paper_id,
            generator_key = generator_key,
            difficulty    = difficulty,
            target        = QUESTIONS_PER_LEVEL,
            start_id      = current_id,
            source        = source
        )

        questions = questions[:QUESTIONS_PER_LEVEL]

        all_questions.extend(questions)
        current_id += len(questions)

        log.info(f"  {difficulty}: {len(questions)} questions (IDs {current_id - len(questions)}–{current_id - 1})")

    log.info(f"{paper_id}: {len(all_questions)} total questions generated ({QUESTIONS_PER_LEVEL} easy + {QUESTIONS_PER_LEVEL} medium + {QUESTIONS_PER_LEVEL} hard)")
    return all_questions


def assign_papers_to_generators(pdf_files: list[str]) -> dict[str, list[str]]:
    import random
    assignment = {k: [] for k in GENERATORS}
    keys = list(GENERATORS.keys())
    random.shuffle(keys)   # randomize starting generator

    for i, pdf in enumerate(pdf_files):
        assignment[keys[i % len(keys)]].append(pdf)

    for gen, papers in assignment.items():
        if papers:
            log.info(f"  {gen} -> {len(papers)} paper(s)")

    return assignment


LEXICON_DIR  = "./papers/lexicon"
LEXICON_FILE = "./papers/lexicon/tsn_lexicon.json"

LEXICON_GENERATOR_PROMPT = """You are an expert question generator specializing in time-sensitive networking (TSN) terminology.

Your task is to generate multiple-choice questions that test knowledge of TSN terms and acronyms.

You will receive a batch of TSN terms with their definitions. Generate one MCQ per term.

QUESTION TYPES — mix these:
- "What does [ACRONYM] stand for?"
- "Which term describes [definition paraphrase]?"
- "What is the correct definition of [TERM]?"
- "Which acronym refers to [concept]?"

OPTION RULES:
- Generate 4 answer options per question
- Distractors must be plausible TSN terms — not random words
- Use 5 options when strong plausible distractors exist in the content
- Use 4 options when content only supports 4 strong distractors
- Never force a weak 5th distractor just to reach 5
- Only one option should be clearly correct

QUESTION RULES:
- Questions must be fully self-contained
- Do NOT reference any paper or document
- Expand acronyms in the question text where needed

OUTPUT FORMAT:
Return a valid JSON object exactly as follows — no extra text, no markdown:

{
  "questions": [
    {
      "question": "...",
      "option_1": "...",
      "option_2": "...",
      "option_3": "...",
      "option_4": "...",
      "correct_option": 2,
      "answer": "option_2: ...",
      "explanation": "...",
      "category": "TSN Lexicon",
      "generator_model": "{{GENERATOR_MODEL}}"
    }
  ]
}

NOTE: correct_option is an integer (1-4).
IMPORTANT: Return only the JSON object. Nothing else."""


def load_lexicon() -> dict:
    if not os.path.exists(LEXICON_FILE):
        log.warning(f"Lexicon file not found: {LEXICON_FILE}")
        log.warning("Run lexicon_extractor.py first to generate the lexicon")
        return {}
    with open(LEXICON_FILE, "r", encoding="utf-8") as f:
        lexicon = json.load(f)
    log.info(f"Loaded lexicon with {len(lexicon)} terms")
    return lexicon


def generate_lexicon_questions(
    batch: list[dict],
    generator_key: str,
    start_id: int = 1
) -> list[dict]:
    terms_text = ""
    for item in batch:
        terms_text += f"- {item['acronym']}: {item['full_name']} — {item['definition']}"

    system = LEXICON_GENERATOR_PROMPT.replace("{{GENERATOR_MODEL}}", generator_key)
    user   = f"Generate one MCQ per term for these TSN terms:\n\n{terms_text}"

    raw = call_generator(generator_key, system, user)
    if not raw:
        return []

    parsed = parse_json_response(raw)
    if not parsed or "questions" not in parsed:
        log.warning(f"Invalid lexicon generator response")
        return []

    questions = parsed["questions"]
    for i, q in enumerate(questions):
        q["question_id"]     = start_id + i
        q["generator_model"] = generator_key
        q["paper_id"]        = "lexicon"
        q["source"]          = "lexicon"
        q["difficulty"]      = "lexicon"

    log.info(f"Generated {len(questions)} lexicon questions")
    return questions


def process_lexicon(
    lexicon      : dict,
    generator_key: str,
    start_id     : int = 1,
    batch_size   : int = 10
) -> list[dict]:
    terms = [
        {"acronym": k, "full_name": v["full_name"], "definition": v["definition"]}
        for k, v in lexicon.items()
    ]

    log.info(f"Generating lexicon MCQs for {len(terms)} terms in batches of {batch_size}")

    all_questions = []
    current_id    = start_id

    for i in range(0, len(terms), batch_size):
        batch = terms[i:i + batch_size]
        log.info(f"  Lexicon batch {i//batch_size + 1}/{(len(terms)-1)//batch_size + 1}")
        questions = generate_lexicon_questions(batch, generator_key, start_id=current_id)
        all_questions.extend(questions)
        current_id += len(questions)

    log.info(f"Total lexicon questions generated: {len(all_questions)}")
    return all_questions


def main():
    list_of_papers    = "./papers/list_of_papers"
    output = "./output/dataset"
    lexicon_questions = False   #True     # set True to include lexicon MCQs

    os.makedirs(output, exist_ok=True)
    all_pdf_files = []
    for folder, source_tag in [(list_of_papers, "list_of_papers")]:
        if os.path.exists(folder):
            pdfs = sorted(Path(folder).glob("*.pdf"))
            for p in pdfs:
                all_pdf_files.append((str(p), source_tag))
            log.info(f"Found {len(pdfs)} PDFs in {folder} (source: {source_tag})")
        else:
            log.warning(f"Folder not found, skipping: {folder}")

    if not all_pdf_files:
        log.error("No PDF files found in the folder")
        return

    log.info(f"Total PDFs to process: {len(all_pdf_files)}")
    log.info(f"Questions per paper: {QUESTIONS_PER_LEVEL} easy + {QUESTIONS_PER_LEVEL} medium + {QUESTIONS_PER_LEVEL} hard = {QUESTIONS_PER_LEVEL * 3} total")

    pdf_paths_only = [p for p, _ in all_pdf_files]
    source_map     = {p: s for p, s in all_pdf_files}
    assignment     = assign_papers_to_generators(pdf_paths_only)

    for gen, papers in assignment.items():
        log.info(f"  {gen}: {len(papers)} papers")

    all_questions = []
    global_id     = 1

    for generator_key, paper_paths in assignment.items():
        for pdf_path in paper_paths:
            paper_id = Path(pdf_path).stem
            source   = source_map.get(pdf_path, "arxiv")

            try:
                questions = process_paper(
                    pdf_path      = pdf_path,
                    paper_id      = paper_id,
                    generator_key = generator_key,
                    start_id      = global_id,
                    source        = source
                )
                all_questions.extend(questions)
                global_id += len(questions)

                # Save checkpoint per paper
                checkpoint_path = os.path.join(output, f"checkpoint_{paper_id}.json")
                with open(checkpoint_path, "w", encoding="utf-8") as f:
                    json.dump(questions, f, indent=2, ensure_ascii=False)
                log.info(f"Checkpoint: {checkpoint_path}")

            except Exception as e:
                log.error(f"Failed: {paper_id} | {e}", exc_info=True)

    if lexicon_questions:
        lexicon = load_lexicon()
        if lexicon:
            lex_generator = list(GENERATORS.keys())[0]
            log.info(f"Generating lexicon MCQs using {lex_generator}")
            lex_questions = process_lexicon(
                lexicon       = lexicon,
                generator_key = lex_generator,
                start_id      = global_id
            )
            all_questions.extend(lex_questions)
            global_id += len(lex_questions)

            lex_checkpoint = os.path.join(output, "checkpoint_lexicon.json")
            with open(lex_checkpoint, "w", encoding="utf-8") as f:
                json.dump(lex_questions, f, indent=2, ensure_ascii=False)
            log.info(f"Lexicon checkpoint saved: {lex_checkpoint}")

    # Save all raw questions
    output_path = os.path.join(output, "raw_questions.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, indent=2, ensure_ascii=False)

    log.info(f"{'='*60}")
    log.info(f"Total questions generated: {len(all_questions)}")
    log.info(f"  - Research paper questions: {len(all_questions) - (len(lex_questions) if lexicon_questions and lexicon else 0)}")
    log.info(f"  - Lexicon questions: {len(lex_questions) if lexicon_questions and 'lex_questions' in dir() else 0}")
    log.info(f"Raw questions saved: {output_path}")


if __name__ == "__main__":
    main()