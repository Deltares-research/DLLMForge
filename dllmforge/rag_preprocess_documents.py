"""
This module provides document preprocessing functionality for RAG (Retrieval-Augmented Generation) pipelines.
It includes document loading and text chunking for PDF files.
"""
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any
from abc import ABC, abstractmethod
import re

from pypdf import PdfReader


class DocumentLoader(ABC):
    """Abstract base class for document loaders."""

    @abstractmethod
    def load(self, file_path: Path) -> List[Tuple[int, str]]:
        """
        Load a document and return its contents as a list of (page_number, text) tuples.
        Args:
            file_path: Path to the document file
        Returns:
            List of tuples containing (page_number, text) pairs
        """
        pass


class PDFLoader(DocumentLoader):
    """Loader for PDF documents using PyPDF2."""

    def load(self, file_path: Path) -> Tuple[List[Tuple[int, str]], str]:
        """
        Load a PDF document and extract text from its pages.
        Args:
            file_path: Path to the PDF file
        Returns:
            Tuple containing (pages_with_text, file_name) where pages_with_text is a list of (page_number, text) pairs
        """
        file_name = os.path.basename(file_path)

        pages_with_text = []
        with open(file_path, "rb") as pdf_file:
            pdf_reader = PdfReader(pdf_file)
            metadata = pdf_reader.metadata
            for page_number, page in enumerate(pdf_reader.pages, start=1):
                text = page.extract_text()
                if text.strip():  # Skip empty pages
                    pages_with_text.append((page_number, text))
        return pages_with_text, file_name, metadata


class TextChunker:
    """Class for chunking text into smaller segments with overlap.
    For detailed information about chunking strategies in RAG applications, including:
    - Why chunking is important
    - How to choose chunk size and overlap
    - Different splitting techniques
    - Evaluation methods
    See: https://www.mongodb.com/developer/products/atlas/choosing-chunking-strategy-rag/
    """

    def __init__(self, chunk_size: int = 1000, overlap_size: int = 200):
        """
        Initialize the TextChunker.
        Args:
            chunk_size: Maximum size of each chunk in characters
            overlap_size: Number of characters to overlap between chunks (recommended: 5-20% of chunk_size)
        """
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size

    def chunk_text(self, pages_with_text: List[Tuple[int, str]], file_name: str = None, metadata: dict = None) -> List[Dict[str, Any]]:
        """
        Split text into chunks while preserving sentence boundaries.
        Args:
            pages_with_text: List of tuples containing (page_number, text) pairs
            file_name: Name of the source file (optional)
            metadata: Metadata information extracted from the document (optional)
        Returns:
            List of dictionaries containing chunks with metadata:
            {
                'text': str,           # The chunk text
                'page_number': int,    # Source page number
                'chunk_index': int,    # Index of the chunk
                'total_chunks': int,   # Total number of chunks from this document
                'file_name': str       # Name of the source file
            }
        """
        chunks: List[Dict[str, Any]] = []

        for page_number, text in pages_with_text:
            sentences = re.split(r'(?<=[.!?]) +', text)
            current_chunk = ""

            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= self.chunk_size:
                    current_chunk += sentence + " "
                else:
                    # Add the current chunk
                    if current_chunk.strip():
                        chunks.append({
                            'text': current_chunk.strip(),
                            'page_number': page_number,
                            'chunk_index': len(chunks),
                            'total_chunks': None,  # Will be updated after all chunks are created
                            'file_name': file_name,
                            'metadata': metadata if metadata else None
                        })

                    # Start a new chunk with overlap
                    current_chunk = current_chunk[-self.overlap_size:].strip() + " " + sentence + " "

            # Add any remaining text as a chunk
            if current_chunk.strip():
                chunks.append({
                    'text': current_chunk.strip(),
                    'page_number': page_number,
                    'chunk_index': len(chunks),
                    'total_chunks': None,
                    'file_name': file_name,
                    'metadata': metadata if metadata else None
                })

        # Update total_chunks in all chunks
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk['total_chunks'] = total_chunks

        return chunks


if __name__ == "__main__":
    # Example usage
    data_dir = Path(r'c:\Users\deng_jg\work\16centralized_agents\test_data')
    pdf_path = data_dir / "lstm_low_flow.pdf"

    # Load the PDF document
    loader = PDFLoader()
    pages, file_name, metadata = loader.load(pdf_path)

    # Create chunks with custom settings
    chunker = TextChunker(chunk_size=1000, overlap_size=200)
    chunks = chunker.chunk_text(pages, file_name)

    # Print some information about the chunks
    print(f"Generated {len(chunks)} chunks from file: {file_name}")
    for i, chunk in enumerate(chunks[:2]):  # Print first two chunks as example
        print(f"\nChunk {i+1}:")
        print(f"File: {chunk['file_name']}")
        print(f"Page: {chunk['page_number']}")
        print(f"Index: {chunk['chunk_index']} of {chunk['total_chunks']}")
        print(f"Text preview: {chunk['text'][:100]}...")
