"""Quote-first provenance: recover source_page from source_quote against a Docling index.

The LLM copies a verbatim quote. Page numbers are assigned in code by locating that
quote in page-indexed elements (sections, tables, figures, prose).
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional

FUZZY_THRESHOLD = 0.6
_WHITESPACE = re.compile(r"\s+")


def normalize_text(value: str) -> str:
    """Collapse whitespace and lowercase for quote matching."""
    if not value:
        return ""
    return _WHITESPACE.sub(" ", value).strip().lower()


def normalize_index(index: Optional[Iterable[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Accept element_index ({page, type, text}) or page_index ({page, text})."""
    if not index:
        return []
    out: List[Dict[str, Any]] = []
    for item in index:
        if not isinstance(item, dict):
            continue
        text = item.get("text") or ""
        if not str(text).strip():
            continue
        page = item.get("page", item.get("page_no"))
        try:
            page = int(page) if page is not None else None
        except (TypeError, ValueError):
            page = None
        out.append({
            "page": page,
            "type": item.get("type") or "text",
            "text": str(text),
        })
    return out


def locate_quote(
    quote: str,
    index: Optional[Iterable[Dict[str, Any]]],
    fuzzy_threshold: float = FUZZY_THRESHOLD,
) -> Optional[Dict[str, Any]]:
    """Return the best matching index element for a quote, or None."""
    needle = normalize_text(quote)
    if not needle:
        return None
    elements = normalize_index(index)
    best: Optional[Dict[str, Any]] = None
    best_score = 0.0
    needle_tokens = [tok for tok in needle.split() if tok]
    for el in elements:
        hay = normalize_text(el["text"])
        if not hay:
            continue
        if needle in hay or hay in needle:
            return {**el, "score": 1.0}
        if needle_tokens and all(tok in hay for tok in needle_tokens):
            score = 0.95
        else:
            score = _windowed_ratio(needle, hay)
        if score > best_score:
            best_score = score
            best = el
    if best is not None and best_score >= fuzzy_threshold:
        return {**best, "score": best_score}
    return None


def _windowed_ratio(needle: str, hay: str) -> float:
    """Ratio of needle against hay, using a sliding window when hay is longer."""
    if not needle or not hay:
        return 0.0
    if len(hay) <= len(needle) + 16:
        return SequenceMatcher(None, needle, hay).ratio()
    win = len(needle) + 16
    step = max(1, len(needle) // 3)
    best = 0.0
    for start in range(0, len(hay) - win + 1, step):
        best = max(best, SequenceMatcher(None, needle, hay[start:start + win]).ratio())
        if best >= 0.99:
            break
    return best


def apply_quote_provenance(
    obj: Any,
    index: Optional[Iterable[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Any:
    """Walk a parsed extraction dict and fill source_page from source_quote.

    If ``index`` is omitted, ``metadata['element_index']`` or
    ``metadata['page_index']`` is used. Unmatched quotes get
    ``quote_verified=False`` and ``source_page=None`` (do not keep a guessed page).
    """
    if index is None and metadata:
        index = metadata.get("element_index") or metadata.get("page_index")
    if isinstance(obj, dict):
        new = {key: apply_quote_provenance(value, index=index) for key, value in obj.items()}
        quote = new.get("source_quote")
        if isinstance(quote, str) and quote.strip():
            hit = locate_quote(quote, index)
            if hit and hit.get("page") is not None:
                new["source_page"] = hit["page"]
                new["quote_verified"] = True
                if hit.get("type"):
                    new["source_type"] = hit["type"]
            else:
                new["source_page"] = None
                new["quote_verified"] = False
        return new
    if isinstance(obj, list):
        return [apply_quote_provenance(item, index=index) for item in obj]
    return obj
