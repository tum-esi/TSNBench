"""Reads the WCD values from a model answer."""

import json
import math
import re

# read text values like "3 * 5 = 15" as WCD_US lines
TEXT_VALUE_RULES = True

# larger values are treated as missing
MAX_WCD_US = 1e9

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
INF_RE = re.compile(
    r"(∞|\b(?:inf|infinity|infinite|unbounded|unbound|nan)\b|\bnot\s+bounded\b|\bno\s+bound\b)",
    re.IGNORECASE)

_SIGNED_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_NUMBER_ONLY_RE = re.compile(r"\s*[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\s*")
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_COMMENT_RE = re.compile(r"\s*//.*$", re.DOTALL)
_RESULT_SPLIT_RE = re.compile(r"[=≈]")

# calculations, e.g. "2 * 3", "4400 - 120", "2 *"
_EXPR_RE = re.compile(r"[\d)]\s*[*/×+]\s*[\d(]")
_SUB_RE = re.compile(r"[\d)]\s+-\s*[\d(]|[\d)]-\s+[\d(]|\)\s*-\s*[\d(]")
_DANGLING_RE = re.compile(r"[*/×+\-]\s*$")

_US_UNITS = r"µs|μs|us|usecs?|microsec(?:ond)?s?"
_OTHER_UNITS = (r"ns|nsecs?|nanosec(?:ond)?s?|ps|ms|msecs?|millisec(?:ond)?s?"
                r"|s|secs?|seconds?|min|minutes?|bits?|bytes?|[kmg]?bps|[kmg]?bits?/s|cycles?")
_UNIT_RE = re.compile(r"\s*(" + _US_UNITS + "|" + _OTHER_UNITS + r")(?![A-Za-z])", re.IGNORECASE)
_US_UNIT_RE = re.compile(r"^(?:" + _US_UNITS + r")$", re.IGNORECASE)

_NUM = r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
_PLAIN_TEXT_PATTERNS = [
    r"['\"]?(F\d+)['\"]?\s*[:=]\s*" + _NUM,
    r"(?:WCD|delay)\s+(?:of|for)?\s*(F\d+)\s+is\s+" + _NUM,
    r"(F\d+)[^:=\d]+" + _NUM + r"\s*(?:µs|us|microseconds)",
]

REFUSAL_PHRASES = [
    "i cannot", "i can't", "i am not able", "i'm not able", "unable to",
    "cannot answer", "not enough information", "insufficient information",
    "cannot calculate", "not possible to determine",
]


def _usable(f):
    return f if math.isfinite(f) and abs(f) <= MAX_WCD_US else None


def _to_float(val):
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        return _usable(float(val))
    if isinstance(val, dict):
        for v in val.values():
            f = _to_float(v)
            if f is not None:
                return f
        return None
    if isinstance(val, str):
        if INF_RE.search(val):
            return None
        m = NUM_RE.search(val)
        return _usable(float(m.group())) if m else None
    return None


def _norm_fid(fid):
    m = re.match(r"^\s*(?:f(?:low)?)?[\s_\-#:]*0*(\d+)\s*$", str(fid), re.IGNORECASE)
    return f"F{int(m.group(1))}" if m else str(fid).strip().upper()


def _outside_brackets(s):
    out, depth = [], 0
    for ch in s:
        if ch in "([":
            if depth == 0:
                out.append("(")
            depth += 1
        elif ch in ")]":
            if depth > 0:
                depth -= 1
                if depth == 0:
                    out.append(")")
        elif depth == 0:
            out.append(ch)
    if depth > 0:
        out.append(")")
    return "".join(out)


def _is_calculation(s):
    return bool(_EXPR_RE.search(s) or _SUB_RE.search(s) or _DANGLING_RE.search(s))


def _us_value(s):
    if INF_RE.search(s):
        return None
    m = _SIGNED_NUM_RE.search(s)
    if not m:
        return None
    u = _UNIT_RE.match(s, m.end())
    if u and not _US_UNIT_RE.match(u.group(1)):
        return None
    return _usable(float(m.group()))


def _wcd_line_value(val):
    v = _THOUSANDS_RE.sub("", val.replace("**", "").strip())
    tail = _RESULT_SPLIT_RE.split(_outside_brackets(v))[-1]
    if _is_calculation(tail):
        return None
    return _us_value(tail)


def _json_text_value(val):
    if TEXT_VALUE_RULES and isinstance(val, str):
        return _wcd_line_value(_COMMENT_RE.sub("", val))
    return _to_float(val)


def _is_text_missing(val):
    return (TEXT_VALUE_RULES and isinstance(val, str) and not INF_RE.search(val)
            and _NUMBER_ONLY_RE.fullmatch(val) is None
            and _wcd_line_value(_COMMENT_RE.sub("", val)) is None)


def _find_wcd_json(text):
    decoder = json.JSONDecoder()
    found, pos = None, 0
    while True:
        idx = text.find("{", pos)
        if idx == -1:
            return found
        try:
            obj, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            pos = idx + 1
            continue
        if isinstance(obj, dict) and "WCD_us" in obj:
            found, pos = obj, end
        else:
            pos = idx + 1


def _find_wcd_block(text, info=None):
    blocks = list(re.finditer(r'"WCD_us"\s*:\s*\{([^{}]*)\}', text))
    if not blocks:
        return {}
    out, text_missing = {}, 0
    for fid, val in re.findall(r'"?((?:F|f|[Ff]low[\s_]?)?\d+)"?\s*:\s*([^,}\n]+)', blocks[-1].group(1)):
        v = val.strip().strip('"\'')
        f = _json_text_value(v)
        if f is not None:
            out[_norm_fid(fid)] = f
        elif _is_text_missing(v):
            text_missing += 1
    if info is not None:
        info["text_missing"] = text_missing
    return out


def _find_flow_blocks(text):
    out = {}
    for block in re.split(r"(?im)^(?=\s*[*#>\-\s]*FLOW\s*:)", text):
        f = re.search(r"(?im)^\s*[*#>\-\s]*FLOW\s*:\s*\**\s*([A-Za-z_ ]*\d+)", block)
        w = re.search(r"(?im)^\s*[*#>\-\s]*WCD_US\s*:\s*\**\s*([^\n]+)", block)
        if f and w:
            out[_norm_fid(f.group(1))] = _wcd_line_value(w.group(1))
    return {k: v for k, v in out.items() if v is not None}


def parse_response(raw):
    empty = {"wcd_values": None, "confidence": None, "flow_profile": None,
             "reason": None, "invalid": True, "refusal": False, "parse_note": ""}
    if not raw:
        return {**empty, "parse_note": "empty_response"}
    clean = raw.strip()

    parsed = _find_wcd_json(clean)
    if parsed is not None:
        wcd_raw = parsed.get("WCD_us") or {}
        if not isinstance(wcd_raw, dict):
            wcd_raw = {}
        wcd = {_norm_fid(fid): _json_text_value(val) for fid, val in wcd_raw.items()}
        bad = [fid for fid, val in wcd_raw.items()
               if not isinstance(val, (int, float)) or isinstance(val, bool)]
        if any(v is not None for v in wcd.values()):
            conf = parsed.get("confidence")
            if isinstance(conf, (int, float)) and not isinstance(conf, bool):
                conf = round(max(0.0, min(1.0, float(conf))), 4)
            else:
                conf = None
            return {"wcd_values": wcd, "confidence": conf,
                    "flow_profile": parsed.get("flow_profile"), "reason": parsed.get("reason"),
                    "invalid": False, "refusal": False,
                    "parse_note": f"wcd_not_numeric:{bad}" if bad else "ok"}
        text_missing = sum(1 for val in wcd_raw.values() if _is_text_missing(val))
        if text_missing:
            return {**empty, "parse_note": f"wcd_text_no_value:{text_missing}"}

    info = {}
    wcd = _find_wcd_block(clean, info)
    if wcd:
        cm = re.search(r'"confidence"\s*:\s*(-?\d+(?:\.\d+)?)', clean)
        return {**empty, "wcd_values": wcd,
                "confidence": float(cm.group(1)) if cm else None,
                "invalid": False, "parse_note": "wcd_block_only"}
    if info.get("text_missing"):
        return {**empty, "parse_note": f"wcd_text_no_value:{info['text_missing']}"}

    blocks = _find_flow_blocks(clean)
    if blocks:
        return {**empty, "wcd_values": blocks, "invalid": False, "parse_note": "flow_blocks"}

    for pattern in _PLAIN_TEXT_PATTERNS:
        for m in re.finditer(pattern, clean, re.IGNORECASE):
            f = _usable(float(m.group(2)))
            if f is not None:
                wcd[m.group(1).upper()] = f
    if wcd:
        return {**empty, "wcd_values": wcd, "invalid": False,
                "parse_note": "extracted_from_plain_text"}

    if any(p in clean.lower() for p in REFUSAL_PHRASES):
        return {**empty, "invalid": False, "refusal": True, "parse_note": "refusal_detected"}

    if parsed is not None:
        return {**empty, "parse_note": "json_parsed_no_wcd_field"}
    return {**empty, "parse_note": "parse_failed_no_wcd_extractable"}
