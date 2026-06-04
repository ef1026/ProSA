from __future__ import annotations

import re
import string
import unicodedata


_PUNCT_TRANSLATION = str.maketrans("", "", string.punctuation + "，。！？；：、（）【】《》“”‘’—…")


def normalize_unicode(text: str | None) -> str:
    """Normalize Unicode with NFKC; None becomes an empty string."""
    if text is None:
        return ""
    return unicodedata.normalize("NFKC", str(text))


def normalize_whitespace(text: str | None) -> str:
    """Collapse repeated whitespace and trim leading/trailing whitespace."""
    value = "" if text is None else str(text)
    return re.sub(r"\s+", " ", value).strip()


def normalize_punctuation(text: str | None) -> str:
    """Remove common ASCII and CJK punctuation without aggressive stemming."""
    value = normalize_unicode(text)
    return value.translate(_PUNCT_TRANSLATION)


def normalize_number_text(text: str | None) -> str:
    """Normalize simple numeric formatting such as `1,000` to `1000`."""
    value = "" if text is None else str(text)
    return re.sub(r"(?<=\d),(?=\d{3}\b)", "", value)


def normalize_answer(text: str | None) -> str:
    """Normalize answer text for conservative matching.

    The normalization lowercases, applies full-width/half-width normalization,
    removes common punctuation, collapses whitespace, trims edges, and removes
    simple thousands separators from numbers.
    """
    value = normalize_unicode(text).lower()
    value = normalize_number_text(value)
    value = normalize_punctuation(value)
    return normalize_whitespace(value)
