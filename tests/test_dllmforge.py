from dllmforge.__version__ import version
import pytest
from pathlib import Path
from dllmforge.rag_preprocess_documents import (
    DocumentLoader,
    PDFLoader,
    TextChunker
)


class TestVersion:

    def test_version(self):
        assert version == "0.1.0"

# Mock document loader for testing
class MockLoader(DocumentLoader):
    def load(self, file_path: Path):
        # Return some test data
        return [
            (1, "This is the first page. It has two sentences."),
            (2, "This is page two. Another sentence here. And one more.")
        ]

def test_text_chunker():
    """Test the TextChunker class."""
    chunker = TextChunker(chunk_size=50, overlap_size=10)
    test_pages = [
        (1, "This is a test sentence. Another test sentence."),
        (2, "Page two sentence one. Page two sentence two.")
    ]
    
    chunks = chunker.chunk_text(test_pages)
    
    assert len(chunks) > 0
    assert isinstance(chunks[0], dict)
    assert 'text' in chunks[0]
    assert 'page_number' in chunks[0]
    assert 'chunk_index' in chunks[0]
    assert 'total_chunks' in chunks[0]
    assert chunks[0]['chunk_index'] == 0
    assert all(chunk['total_chunks'] == len(chunks) for chunk in chunks)

def test_mock_loader():
    """Test the mock document loader."""
    loader = MockLoader()
    pages = loader.load(Path("dummy.pdf"))
    
    assert len(pages) == 2
    assert all(isinstance(page, tuple) for page in pages)
    assert all(len(page) == 2 for page in pages)
    assert all(isinstance(page[0], int) for page in pages)
    assert all(isinstance(page[1], str) for page in pages)

def test_chunker_empty_input():
    """Test TextChunker with empty input."""
    chunker = TextChunker()
    chunks = chunker.chunk_text([])
    assert len(chunks) == 0

def test_chunker_single_short_text():
    """Test TextChunker with a single short text that fits in one chunk."""
    chunker = TextChunker(chunk_size=1000)
    test_pages = [(1, "Short text.")]
    chunks = chunker.chunk_text(test_pages)
    
    assert len(chunks) == 1
    assert chunks[0]['text'] == "Short text."
    assert chunks[0]['page_number'] == 1
    assert chunks[0]['chunk_index'] == 0
    assert chunks[0]['total_chunks'] == 1

def test_chunker_overlap():
    """Test that chunks properly overlap."""
    chunker = TextChunker(chunk_size=20, overlap_size=10)
    test_pages = [(1, "First chunk text. Second chunk text.")]
    chunks = chunker.chunk_text(test_pages)
    
    assert len(chunks) > 1
    # Check that the end of first chunk overlaps with start of second chunk
    first_chunk_end = chunks[0]['text'][-10:]
    second_chunk_start = chunks[1]['text'][:10]
    assert first_chunk_end.strip() in second_chunk_start or second_chunk_start in first_chunk_end.strip()
