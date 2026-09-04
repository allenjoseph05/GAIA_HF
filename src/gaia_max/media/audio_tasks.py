"""Conservative ingredient and spoken-page extraction from timestamped evidence."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field

from gaia_max.media.asr import Transcript


class AudioExtractionError(ValueError):
    """Audio evidence cannot support an exact task answer."""


class IngredientResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    ingredients: tuple[str, ...] = Field(min_length=1)
    evidence_timestamps: tuple[float, ...] = Field(min_length=1)
    section_start_seconds: float = Field(ge=0)
    section_end_seconds: float = Field(gt=0)


class NumericPageResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    pages: tuple[int, ...] = Field(min_length=1)
    evidence_timestamps: tuple[float, ...] = Field(min_length=1)
    ambiguous_spans: tuple[str, ...] = ()


_MEASUREMENT = re.compile(
    r"\b(?:\d+(?:[./]\d+)?|one|two|three|four|five|six|seven|eight|nine|half|quarter)\s*(?:cups?|tablespoons?|tbsp|teaspoons?|tsp|grams?|g|kg|ounces?|oz|pounds?|lb|ml|liters?)\b",
    re.I,
)
_PREPARATION = re.compile(
    r"\b(?:finely|roughly|freshly|thinly|chopped|diced|minced|sliced|grated|"
    r"melted|softened|divided|to taste)\b",
    re.I,
)
_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_SPOKEN_NUMBER = (
    r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)"
    r"(?:[- ](?:and[- ])?(?:one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
    r"nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred))*"
)
_PAGE_NUMBER = re.compile(
    rf"\bpages?\s+(?P<after>\d{{1,4}}|{_SPOKEN_NUMBER})\b|"
    rf"\b(?P<before>\d{{1,4}}|{_SPOKEN_NUMBER})\s+pages?\b"
)


def extract_ingredients(
    transcript: Transcript,
    *,
    section_start: str,
    section_end: str,
    excluded_sections: tuple[str, ...] = ("crust", "topping", "garnish"),
) -> IngredientResult:
    """Extract only explicitly spoken items inside the requested recipe section."""

    start_index = _find_anchor(transcript, section_start, 0)
    end_index = _find_anchor(transcript, section_end, start_index + 1)
    if end_index <= start_index:
        raise AudioExtractionError("ingredient section boundaries are invalid")
    items: list[str] = []
    timestamps: list[float] = []
    excluded = False
    for segment in transcript.segments[start_index + 1 : end_index]:
        text = segment.raw_text.strip()
        if any(name.casefold() in text.casefold() for name in excluded_sections):
            excluded = True
            continue
        if excluded:
            if re.search(r"\b(?:filling|sauce|main|ingredients)\b", text, re.I):
                excluded = False
            else:
                continue
        for candidate in re.split(r"[,;]|\band\b", text, flags=re.I):
            cleaned = _clean_ingredient(candidate)
            if cleaned and cleaned.casefold() not in {item.casefold() for item in items}:
                items.append(cleaned)
                timestamps.append(segment.start_seconds)
    if not items:
        raise AudioExtractionError("no explicit ingredients found in the requested section")
    ordered = sorted(zip(items, timestamps, strict=True), key=lambda pair: pair[0].casefold())
    return IngredientResult(
        ingredients=tuple(item for item, _ in ordered),
        evidence_timestamps=tuple(timestamp for _, timestamp in ordered),
        section_start_seconds=transcript.segments[start_index].start_seconds,
        section_end_seconds=transcript.segments[end_index].end_seconds,
    )


def extract_numeric_pages(transcript: Transcript) -> NumericPageResult:
    """Extract page-context numbers, preserving ambiguity for second-pass review."""

    pages: list[int] = []
    timestamps: list[float] = []
    ambiguous: list[str] = []
    for segment in transcript.segments:
        text = segment.normalized_text
        matches = _PAGE_NUMBER.finditer(text)
        for match in matches:
            token = next(group for group in match.groups() if group is not None)
            try:
                value = _parse_number(token)
            except AudioExtractionError:
                ambiguous.append(segment.raw_text)
                continue
            if value <= 0:
                ambiguous.append(segment.raw_text)
                continue
            if value not in pages:
                pages.append(value)
                timestamps.append(segment.start_seconds)
        if "page" in text and re.search(r"\b(?:-teen|-ty|teen|ty)\b", text):
            ambiguous.append(segment.raw_text)
    if not pages:
        raise AudioExtractionError("no unambiguous page-context numbers found")
    ordered = sorted(zip(pages, timestamps, strict=True))
    return NumericPageResult(
        pages=tuple(page for page, _ in ordered),
        evidence_timestamps=tuple(timestamp for _, timestamp in ordered),
        ambiguous_spans=tuple(dict.fromkeys(ambiguous)),
    )


def _find_anchor(transcript: Transcript, anchor: str, begin: int) -> int:
    normalized = " ".join(anchor.casefold().split())
    for index in range(begin, len(transcript.segments)):
        if normalized in transcript.segments[index].normalized_text:
            return index
    raise AudioExtractionError(f"section anchor not found: {anchor}")


def _clean_ingredient(candidate: str) -> str | None:
    cleaned = _MEASUREMENT.sub("", candidate)
    cleaned = _PREPARATION.sub("", cleaned)
    cleaned = re.sub(r"^[\s\d./-]+|[.\s]+$", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"^(?:add|use|then|with|some|the|of)\s+", "", cleaned, flags=re.I)
    if not cleaned or cleaned.casefold() in {"ingredients", "filling", "sauce"}:
        return None
    return cleaned.casefold()


def _parse_number(token: str) -> int:
    compact = token.casefold().replace("-", " ").strip()
    if compact.isdigit():
        return int(compact)
    words = [word for word in compact.split() if word != "and"]
    if not words or any(word not in _NUMBER_WORDS and word != "hundred" for word in words):
        raise AudioExtractionError("spoken number is ambiguous")
    total = 0
    current = 0
    for word in words:
        if word == "hundred":
            if current == 0:
                raise AudioExtractionError("spoken hundred lacks a multiplier")
            current *= 100
        else:
            value = _NUMBER_WORDS[word]
            if current and value >= 20 and current < 20:
                raise AudioExtractionError("spoken number order is invalid")
            current += value
    total += current
    return total


__all__ = [
    "AudioExtractionError",
    "IngredientResult",
    "NumericPageResult",
    "extract_ingredients",
    "extract_numeric_pages",
]
