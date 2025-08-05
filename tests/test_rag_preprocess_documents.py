import pytest
from pathlib import Path
from dllmforge.rag_preprocess_documents import PDFLoader, TextChunker
from unittest.mock import patch, MagicMock, mock_open


def test_pdf_loader_reads_pages(monkeypatch):
    # Mock PyPDF2 PdfReader and its pages
    mock_page1 = MagicMock()
    mock_page1.extract_text.return_value = "Page 1 text."
    mock_page2 = MagicMock()
    mock_page2.extract_text.return_value = "Page 2 text."
    mock_reader = MagicMock()
    mock_reader.pages = [mock_page1, mock_page2]

    with patch("dllmforge.rag_preprocess_documents.PdfReader", return_value=mock_reader):
        with patch("builtins.open", mock_open(read_data=b"dummy")):
            loader = PDFLoader()
            # The file path is irrelevant due to mocking
            pages, file_name, metadata = loader.load(Path("dummy.pdf"))
            assert pages == [(1, "Page 1 text."), (2, "Page 2 text.")]
            assert file_name == "dummy.pdf"


def test_pdf_loader_skips_empty_pages():
    mock_page1 = MagicMock()
    mock_page1.extract_text.return_value = "   "  # Empty after strip
    mock_page2 = MagicMock()
    mock_page2.extract_text.return_value = "Not empty."
    mock_reader = MagicMock()
    mock_reader.pages = [mock_page1, mock_page2]

    with patch("dllmforge.rag_preprocess_documents.PdfReader", return_value=mock_reader):
        with patch("builtins.open", mock_open(read_data=b"dummy")):
            loader = PDFLoader()
            pages, file_name, metadata = loader.load(Path("dummy.pdf"))
            assert pages == [(2, "Not empty.")]
            assert file_name == "dummy.pdf"


def test_text_chunker_long_sentence():
    chunker = TextChunker(chunk_size=10, overlap_size=2)
    # One sentence longer than chunk_size
    test_pages = [(1, "A verylongsentencewithoutspaces.")]
    chunks = chunker.chunk_text(test_pages, "test.pdf")
    assert len(chunks) >= 1
    assert all(isinstance(chunk["text"], str) for chunk in chunks)
    assert all(chunk["file_name"] == "test.pdf" for chunk in chunks)


def test_text_chunker_multiple_pages():
    chunker = TextChunker(chunk_size=20, overlap_size=5)
    test_pages = [(1, "Page one. More text."), (2, "Page two. Even more text here.")]
    chunks = chunker.chunk_text(test_pages, "test.pdf")
    assert all("page_number" in chunk for chunk in chunks)
    assert all("file_name" in chunk for chunk in chunks)
    assert set(chunk["page_number"] for chunk in chunks) == {1, 2}
    assert all(chunk["file_name"] == "test.pdf" for chunk in chunks)


def test_text_chunker_without_file_name():
    chunker = TextChunker(chunk_size=20, overlap_size=5)
    test_pages = [(1, "Page one. More text.")]
    chunks = chunker.chunk_text(test_pages)  # No file_name provided
    assert len(chunks) >= 1
    assert all("page_number" in chunk for chunk in chunks)
    assert all("file_name" in chunk for chunk in chunks)
    assert all(chunk["file_name"] is None for chunk in chunks)
