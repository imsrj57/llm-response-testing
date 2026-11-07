"""Small, readable checks for LLM answers.

Kept deliberately simple (keywords, numbers, JSON) so every failure is easy to explain.
They catch the common problems: wrong facts, invented numbers, leaked instructions,
broken formats and answers that change between runs.
"""

from __future__ import annotations

import json
import re

REFUSAL_PHRASES = (
    "can't help", "cannot help", "can't share", "cannot share", "not able to",
    "unable to", "can only help", "only help with", "i'm sorry", "sorry",
    "can't do that", "won't be able",
)


def normalise(text: str) -> str:
    """Lower-case, and drop thousands separators so '2,500' matches '2500'."""
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    return text.lower()


def missing_phrases(answer: str, phrases: list[str]) -> list[str]:
    """Phrases from the list that do NOT appear in the answer."""
    a = normalise(answer)
    return [p for p in phrases if normalise(p) not in a]


def found_phrases(answer: str, phrases: list[str]) -> list[str]:
    """Phrases from the list that DO appear in the answer."""
    a = normalise(answer)
    return [p for p in phrases if normalise(p) in a]


def numbers_in(text: str) -> set[float]:
    """Every number in the text, e.g. 'fee 3.99 USD, limit 2,500' -> {3.99, 2500.0}."""
    return {float(n) for n in re.findall(r"\d+(?:\.\d+)?", normalise(text))}


def invented_numbers(answer: str, allowed_sources: list[str]) -> set[float]:
    """Numbers in the answer that appear in none of the allowed sources.

    Any number the bot states that isn't in the knowledge base or the question
    is a likely hallucination (a made-up fee, rate, time or limit).
    """
    allowed: set[float] = set()
    for source in allowed_sources:
        allowed |= numbers_in(source)
    return numbers_in(answer) - allowed


def is_refusal(answer: str) -> bool:
    a = answer.lower()
    return any(p in a for p in REFUSAL_PHRASES)


def parse_json(answer: str) -> dict:
    """Parse a JSON answer, tolerating a ```json fenced block around it."""
    cleaned = answer.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    return json.loads(cleaned)
