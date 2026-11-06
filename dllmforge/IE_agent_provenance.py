"""Quote-first provenance: recover source_page from source_quote against a Docling index.

The LLM copies a verbatim quote. Page numbers are assigned in code by locating that
quote in page-indexed elements (sections, tables, figures, prose).
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional

FUZZY_THRESHOLD = 0.55
_WHITESPACE = re.compile(r"\s+")
_DIGIT_LETTER = re.compile(r"(\d)([a-zA-Z])")
_INTERNAL_META_KEYS = {"element_index", "page_index"}


def public_metadata(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Copy processor metadata without the bulky locator index."""
    if not isinstance(meta, dict):
        return {}
    return {key: value for key, value in meta.items() if key not in _INTERNAL_META_KEYS}


def normalize_text(value: str) -> str:
    """Collapse whitespace, fix OCR units (50m -> 50 m), lowercase."""
    if not value:
        return ""
    value = value.replace("\ufffd", "x").replace("�", "x")
    value = value.replace("×", "x").replace("·", "x")
    value = value.replace("^", " ")
    value = re.sub(r"(\d):(\d)", r"\1.\2", value)
    value = _WHITESPACE.sub(" ", value).strip().lower()
    return _DIGIT_LETTER.sub(r"\1 \2", value)


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


def _quote_fragments(quote: str) -> List[str]:
    """Split a pasted quote into paragraphs, table rows, or sentences."""
    chunks = [part.strip() for part in re.split(r"\n\s*\n", quote.strip()) if part.strip()]
    fragments: List[str] = []
    for chunk in chunks:
        fragments.append(chunk)
        if "|" in chunk:
            for line in chunk.splitlines():
                line = line.strip()
                if not line.startswith("|") or "---" in line:
                    continue
                cells = [cell.strip() for cell in line.strip("|").split("|") if cell.strip()]
                if cells:
                    fragments.append(line)
            continue
        fragments.extend(
            sent.strip() for sent in re.split(r"(?<=[.!?])\s+", chunk) if len(sent.strip()) >= 24
        )
    return fragments or [quote]


def _locate_one(
    needle: str,
    elements: List[Dict[str, Any]],
    fuzzy_threshold: float,
) -> Optional[Dict[str, Any]]:
    if not needle:
        return None
    needle_tokens = [tok for tok in needle.split() if tok]
    min_contain = min(80, max(24, int(len(needle) * 0.25)))
    best: Optional[Dict[str, Any]] = None
    best_score = 0.0
    for el in elements:
        hay = normalize_text(el["text"])
        if not hay:
            continue
        if needle in hay:
            return {**el, "score": 1.0}
        if hay in needle and len(hay) >= min_contain:
            return {**el, "score": 0.99}
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


def locate_quote(
    quote: str,
    index: Optional[Iterable[Dict[str, Any]]],
    fuzzy_threshold: float = FUZZY_THRESHOLD,
) -> Optional[Dict[str, Any]]:
    """Return the best matching index element for a quote, or None.

    Long LLM quotes often concatenate a sentence + a table + more prose. Match
    the whole string first, then each paragraph, and prefer a table hit when
    the quote contains a markdown table.
    """
    elements = normalize_index(index)
    if not elements or not quote or not quote.strip():
        return None
    hits: List[Dict[str, Any]] = []
    whole = _locate_one(normalize_text(quote), elements, fuzzy_threshold)
    if whole:
        hits.append(whole)
    for fragment in _quote_fragments(quote):
        hit = _locate_one(normalize_text(fragment), elements, fuzzy_threshold)
        if hit:
            hits.append(hit)
    if not hits:
        return None
    if "|" in quote:
        table_hits = [hit for hit in hits if hit.get("type") == "table"]
        if table_hits:
            return max(table_hits, key=lambda hit: hit.get("score", 0.0))
    return max(hits, key=lambda hit: (hit.get("score", 0.0), len(hit.get("text") or "")))


def _windowed_ratio(needle: str, hay: str) -> float:
    """Ratio of needle against hay, using a sliding window when hay is longer."""
    if not needle or not hay:
        return 0.0
    if len(hay) <= len(needle) + 16:
        return SequenceMatcher(None, needle, hay).ratio()
    win = min(len(hay), len(needle) + 16)
    step = max(1, len(needle) // 3)
    best = 0.0
    for start in range(0, max(1, len(hay) - win + 1), step):
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
