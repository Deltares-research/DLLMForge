"""
Keyword document search (BM25) over project documents: a simple, dependency-free retrieval capability.

Documents (PDF, Markdown, plain text) are split into passages with the existing
:class:`~dllmforge.rag_preprocess_documents.TextChunker` and ranked with Okapi BM25. The capability returned by
:func:`document_search_capability` is a plain function, so it works with ``SimpleAgent`` tools and MCP alike
(e.g. via ``DatabaseProject.extra_capabilities``).

Limitations (by design, for now):

- **Lexical only**: passages must share words with the question; synonyms, paraphrases and other languages are
  not matched. A later embedding implementation (as in BOIRQ ``src/rag``) can keep the same tool interface.
- **In memory, rebuilt at start**: the index is built when the project is loaded; suitable for tens to a few
  hundred documents, not for large collections.
- **Text extraction**: PDF quality depends on ``pypdf``; scanned PDFs without a text layer yield nothing.
  Markdown and text files have no page numbers.
"""
import fnmatch
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

__all__ = ["Passage", "DocumentIndex", "document_search_capability", "DEFAULT_INCLUDE"]

DEFAULT_INCLUDE = ("*.pdf", "*.md", "*.txt")
_TOKEN = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


@dataclass(frozen=True)
class Passage:
    """A searchable piece of a document."""
    text: str
    source: str
    page: Optional[int] = None

    @property
    def citation(self) -> str:
        return f"[{self.source} p.{self.page}]" if self.page else f"[{self.source}]"


def _matches(path: Path, root: Path, patterns: Sequence[str]) -> bool:
    relative = path.relative_to(root).as_posix() if path.is_relative_to(root) else path.name
    return any(fnmatch.fnmatch(path.name, p) or fnmatch.fnmatch(relative, p) for p in patterns)


def _collect_files(paths: Iterable[Path], include: Sequence[str], exclude: Sequence[str]) -> List[Tuple[Path, Path]]:
    """(file, root) pairs: files given directly, plus matching files found recursively in directories."""
    files: Dict[Path, Path] = {}
    for path in paths:
        path = Path(path)
        if path.is_file():
            files.setdefault(path, path.parent)
        elif path.is_dir():
            for candidate in sorted(path.rglob("*")):
                if candidate.is_file() and _matches(candidate, path, include):
                    files.setdefault(candidate, path)
        else:
            raise ValueError(f"document path not found: {path}")
    return [(file, root) for file, root in files.items() if not _matches(file, root, exclude)]


def _pages(path: Path) -> Tuple[List[Tuple[int, str]], bool]:
    """(page_number, text) pairs and whether the numbers are real pages."""
    if path.suffix.lower() == ".pdf":
        from .rag_preprocess_documents import PDFLoader
        pages, _, _ = PDFLoader().load(path)
        return pages, True
    return [(1, path.read_text(encoding="utf-8", errors="replace"))], False


class DocumentIndex:
    """In-memory BM25 index over passages."""

    def __init__(self, passages: Sequence[Passage], k1: float = 1.5, b: float = 0.75):
        self.passages = list(passages)
        self.k1, self.b = k1, b
        self._tf = [Counter(_tokens(p.text)) for p in self.passages]
        self._lengths = [sum(tf.values()) for tf in self._tf]
        self._avg_length = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        df: Counter = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(self.passages)
        self._idf = {term: math.log(1 + (n - count + 0.5) / (count + 0.5)) for term, count in df.items()}

    def __len__(self) -> int:
        return len(self.passages)

    @property
    def sources(self) -> List[str]:
        return sorted({p.source for p in self.passages})

    @classmethod
    def from_paths(cls,
                   paths: Iterable[Union[str, Path]],
                   include: Sequence[str] = DEFAULT_INCLUDE,
                   exclude: Sequence[str] = (),
                   chunk_size: int = 1000,
                   chunk_overlap: int = 200) -> "DocumentIndex":
        """Index files and directories (searched recursively for ``include`` patterns, minus ``exclude``).

        Raises ``ValueError`` when a path does not exist or no passages were found.
        """
        from .rag_preprocess_documents import TextChunker
        chunker = TextChunker(chunk_size=chunk_size, overlap_size=chunk_overlap)
        passages: List[Passage] = []
        files = _collect_files([Path(p) for p in paths], include, exclude)
        for file, root in files:
            pages, paged = _pages(file)
            source = file.relative_to(root).as_posix()
            for chunk in chunker.chunk_text(pages, source):
                passages.append(Passage(chunk["text"], source, chunk["page_number"] if paged else None))
        if not passages:
            raise ValueError(f"no searchable text found in {[str(p) for p in paths]} "
                             f"(include={list(include)}, exclude={list(exclude)}, files={len(files)})")
        return cls(passages)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Passage, float]]:
        """Best ``top_k`` passages for ``query`` with their BM25 scores (only passages sharing a word)."""
        terms = [t for t in dict.fromkeys(_tokens(query)) if t in self._idf]
        scored = []
        for index, tf in enumerate(self._tf):
            score = 0.0
            norm = self.k1 * (1 - self.b + self.b * self._lengths[index] / (self._avg_length or 1))
            for term in terms:
                freq = tf.get(term, 0)
                if freq:
                    score += self._idf[term] * freq * (self.k1 + 1) / (freq + norm)
            if score > 0:
                scored.append((score, index))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [(self.passages[index], score) for score, index in scored[:max(1, top_k)]]


def document_search_capability(index: DocumentIndex, top_k: int = 5, max_top_k: int = 20) -> Callable[..., str]:
    """Build the ``search_documents(query, top_k)`` capability over ``index``."""

    def search_documents(query: str, top_k: Optional[int] = None) -> str:
        if not query or not query.strip():
            return "ERROR: Empty search query."
        k = min(max(1, top_k or default_top_k), max_top_k)
        results = index.search(query, k)
        if not results:
            return (f"No passages in the project documents match '{query}'. Try other keywords (the search "
                    "matches words, not meaning); if nothing matches, say the documents do not cover this.")
        parts = [f"{len(results)} passage(s) for '{query}' (keyword search; cite as shown):"]
        for number, (passage, score) in enumerate(results, 1):
            parts.append(f"[{number}] {passage.citation} (score {score:.2f})\n{passage.text}")
        return "\n\n".join(parts)

    default_top_k = top_k
    search_documents.__doc__ = (
        "Search the project documents (methodology, manuals, reports) by keywords and return the most relevant "
        "passages with their source as [file p.N]. Use it for definitions, methods and background that are not "
        "stored in the database; cite the sources in your answer. It matches words, not meaning: use the terms "
        f"the documents are likely to use, and retry with synonyms if nothing is found. Default top_k "
        f"{top_k}, maximum {max_top_k}.\n\nDocuments: {', '.join(index.sources)}.")
    return search_documents
