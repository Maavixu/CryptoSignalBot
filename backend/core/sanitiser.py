"""
sanitiser.py — Prompt injection defence.
Every piece of external text (RSS, Reddit, Twitter, news) passes
through clean() before it touches any NLP model or log output.
This prevents adversarial content from hijacking the analysis pipeline.
"""
import re
from typing import List

# Patterns that look like instructions to an AI / LLM
_INJECTION_PATTERNS: List[re.Pattern] = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|context)", re.I),
    re.compile(r"you\s+are\s+now\s+a?n?\s+\w+", re.I),
    re.compile(r"(act|behave|respond)\s+as\s+(if\s+)?(you\s+are|an?)\s+", re.I),
    re.compile(r"system\s*prompt", re.I),
    re.compile(r"<\s*(system|user|assistant)\s*>", re.I),
    re.compile(r"\[INST\]|\[\/INST\]|<<SYS>>|<</SYS>>", re.I),
    re.compile(r"jailbreak|dan\s+mode|developer\s+mode", re.I),
    re.compile(r"disregard\s+(your\s+)?(training|guidelines|rules)", re.I),
    re.compile(r"new\s+instruction[s]?\s*:", re.I),
    re.compile(r"---+\s*(end|begin)\s*(of\s+)?(prompt|instruction)", re.I),
    re.compile(r"###\s*(instruction|prompt|system)", re.I),
]

# Max length to prevent extremely long content from overwhelming analysis
_MAX_TEXT_LENGTH = 2000
_MAX_TITLE_LENGTH = 300


def clean(text: str, source: str = "unknown") -> str:
    """
    Sanitise a piece of external text.
    - Strips injection-like patterns (replaced with [REDACTED])
    - Truncates to safe length
    - Strips null bytes and control characters
    Returns sanitised string. Never raises.
    """
    if not text:
        return ""

    try:
        # Remove null bytes and most control chars (keep \n \t)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        # Replace injection patterns
        for pattern in _INJECTION_PATTERNS:
            text = pattern.sub("[REDACTED]", text)

        # Truncate
        text = text[:_MAX_TEXT_LENGTH]

        # Strip excessive whitespace
        text = re.sub(r"\s{3,}", "  ", text).strip()

        return text

    except Exception:
        return ""


def clean_title(title: str) -> str:
    result = clean(title)
    return result[:_MAX_TITLE_LENGTH]


def is_suspicious(text: str) -> bool:
    """Returns True if the text contains injection-like patterns."""
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return True
    return False


def clean_batch(items: List[str], source: str = "unknown") -> List[str]:
    return [clean(t, source) for t in items if t]
