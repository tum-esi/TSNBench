import os
import re
import json
import time
import logging
from pathlib import Path
from typing import Optional
import logging
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pdfplumber").setLevel(logging.ERROR)

from dotenv import load_dotenv
load_dotenv()

import pdfplumber
from openai import OpenAI
import anthropic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("lexicon_extractor.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


PAPER_DIR = "./papers/list_of_papers"
OUTPUT_DIR = "./output/lexicon"
OUTPUT_FILE = "./output/lexicon/tsn_lexicon.json"


anthropic_client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)


LEXICON_EXTRACTION_PROMPT = """You are a Time-Sensitive Networking (TSN) terminology expert.

Your task is to extract ALL acronyms and technical terms from the provided text.

EXTRACTION RULES:
- Extract acronyms (2-6 uppercase letters) that are explicitly defined in the text
- Extract technical terms that have domain-specific meaning in TSN
- Only extract terms that are DEFINED or EXPLAINED in the provided text
- Do NOT invent definitions — only use what is stated in the text
- Do NOT extract common English words or generic computing terms
- Focus on TSN-specific terms: protocols, standards, mechanisms, parameters
- Do not extract the terms which are proposed by the text

OUTPUT FORMAT:
Return a valid JSON object exactly as follows — no extra text, no markdown:

{
  "terms": [
    {
      "acronym": "TSN",
      "full_name": "Time-Sensitive Networking",
      "definition": "A set of IEEE 802.1 standards enabling deterministic communication over Ethernet networks"
    },
    {
      "acronym": "GCL",
      "full_name": "Gate Control List",
      "definition": "A schedule that controls the opening and closing of transmission gates in TSN"
    }
  ]
}

NOTE: Only use acronym term.
NOTE: Keep definitions concise — 1-2 sentences maximum.
IMPORTANT: Return only the JSON object. Nothing else."""

def extract_clean_text(pdf_path: str) -> str:
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        log.error(f"Failed to extract: {pdf_path}: {e}")
        return ""

    abstract_match = re.search(r'\bAbstract\b', text, re.IGNORECASE)
    if abstract_match:
        text = text[abstract_match.start():]
    else:
        intro_match = re.search(r'\bIntroduction\b', text, re.IGNORECASE)
        if intro_match:
            text = text[intro_match.start():]

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
    split_lines = text.split('\n')
    while i < len(split_lines):
        line = split_lines[i]
        if line.startswith('    ') or line.startswith('\t'):
            if CODE_INDICATORS.search(line):
                while i < len(split_lines) and (
                        split_lines[i].startswith('    ') or
                        split_lines[i].startswith('\t') or
                        split_lines[i].strip() == ''
                ):
                    i += 1
                continue
        clean_lines.append(line)
        i += 1
    text = '\n'.join(clean_lines)

    return text.strip()


def chunk_for_lexicon(text: str, max_chars: int = 8000) -> list[str]:
    paragraphs = text.split('\n\n')
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) <= max_chars:
            current += "\n\n" + para
        else:
            if current.strip():
                chunks.append(current.strip())
            current = para
    if current.strip():
        chunks.append(current.strip())
    return chunks


def extract_terms_from_chunk(chunk: str) -> list[dict]:
    for attempt in range(3):
        try:
            response = anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=LEXICON_EXTRACTION_PROMPT,
                messages=[{"role": "user", "content": f"Extract all TSN terms from this text:\n\n{chunk}"}]
            )
            raw = response.content[0].text.strip()
            raw = re.sub(r"^```(?:json)?", "", raw).strip()
            raw = re.sub(r"```$", "", raw).strip()

            parsed = json.loads(raw)
            return parsed.get("terms", [])

        except json.JSONDecodeError as e:
            log.warning(f"JSON parse error on attempt {attempt + 1}: {e}")
        except Exception as e:
            log.warning(f"API error on attempt {attempt + 1}: {e}")
            time.sleep(2 ** attempt)
    return []


def extract_lexicon_from_paper(pdf_path: str, paper_id: str) -> dict[str, dict]:
    log.info(f"Extracting lexicon from: {paper_id}")
    text = extract_clean_text(pdf_path)
    if not text:
        log.warning(f"No text extracted from {pdf_path}")
        return {}

    chunks = chunk_for_lexicon(text)
    log.info(f"  {len(chunks)} chunks to process")

    paper_lexicon = {}

    for i, chunk in enumerate(chunks):
        log.info(f"  Chunk {i + 1}/{len(chunks)}")
        terms = extract_terms_from_chunk(chunk)

        for term in terms:
            acronym = term.get("acronym", "").strip().upper()
            full_name = term.get("full_name", "").strip()
            definition = term.get("definition", "").strip()

            if not acronym or not full_name or not definition:
                continue

            if acronym not in paper_lexicon:
                paper_lexicon[acronym] = {
                    "full_name": full_name,
                    "definition": definition,
                    "source_paper": paper_id
                }
            else:
                if len(definition) > len(paper_lexicon[acronym]["definition"]):
                    paper_lexicon[acronym]["definition"] = definition
                    paper_lexicon[acronym]["source_paper"] = paper_id

    log.info(f"  Extracted {len(paper_lexicon)} terms from {paper_id}")
    return paper_lexicon


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_pdfs = []
    for folder in [PAPER_DIR]:
        if os.path.exists(folder):
            pdfs = sorted(Path(folder).glob("*.pdf"))
            all_pdfs.extend(pdfs)
            log.info(f"Found {len(pdfs)} PDFs in {folder}")
        else:
            log.warning(f"Folder not found: {folder}")

    if not all_pdfs:
        log.error("No PDFs found — check ARXIV_DIR and IEEE_DIR paths and EXTRA_DIR")
        return

    log.info(f"Total PDFs to process: {len(all_pdfs)}")

    master_lexicon = {}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            master_lexicon = json.load(f)
        log.info(f"Loaded existing lexicon with {len(master_lexicon)} terms")

    for pdf_path in all_pdfs:
        paper_id = pdf_path.stem

        try:
            paper_lexicon = extract_lexicon_from_paper(str(pdf_path), paper_id)

            for acronym, data in paper_lexicon.items():
                if acronym not in master_lexicon:
                    master_lexicon[acronym] = data
                else:
                    if len(data["definition"]) > len(master_lexicon[acronym]["definition"]):
                        master_lexicon[acronym] = data
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(master_lexicon, f, indent=2, ensure_ascii=False)
            log.info(f"Lexicon saved: {len(master_lexicon)} total terms")

        except Exception as e:
            log.error(f"Failed: {paper_id} | {e}", exc_info=True)

    sorted_lexicon = dict(sorted(master_lexicon.items()))
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted_lexicon, f, indent=2, ensure_ascii=False)

    log.info(f"{'=' * 60}")
    log.info(f"Lexicon extraction complete")
    log.info(f"Total unique terms: {len(sorted_lexicon)}")
    log.info(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()