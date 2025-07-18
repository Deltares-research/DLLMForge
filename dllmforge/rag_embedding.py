"""
This module provides embedding functionality for RAG (Retrieval-Augmented Generation) pipelines.
It can be used to 1) vectorize document chunks, and 2) vectorize user queries.
The module uses Azure OpenAI embeddings model as an example of using hosted embedding APIs. Note you need
Azure OpenAI service and a deployed embedding model on Azure to use this module.
"""

from typing import List, Any
import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings

class AzureOpenAIEmbeddingModel:
    """Class for embedding queries and document chunks using Azure OpenAI Embeddings."""

    def __init__(self, model: str = "text-embedding-3-large"):
        """
        Initialize the embedding model using environment variables for Azure OpenAI.
        Args:
            model: Name of the embedding model to use 
            (default: "text-embedding-3-large", check Azure OpenAI for available models)
        """
        load_dotenv()
        api_base = os.getenv('AZURE_OPENAI_API_BASE')
        deployment_name_embeddings = os.getenv('AZURE_OPENAI_DEPLOYMENT_EMBEDDINGS')
        api_key = os.getenv('AZURE_OPENAI_API_KEY')
        api_version = os.getenv('AZURE_OPENAI_API_VERSION')

        self.embeddings = AzureOpenAIEmbeddings(
            model=model,
            azure_endpoint=api_base,
            azure_deployment=deployment_name_embeddings,
            api_key=api_key,
            openai_api_version=api_version
        )

    def embed(self, query_or_chunks: Any) -> Any:
        """
        Embed a single query string or a list of document chunks.
        Args:
            query_or_chunks: A string (query) or a list of strings (document chunks)
        Returns:
            Embedding vector(s): list of floats for a single string, or list of list of floats for a list of strings
        """
        if isinstance(query_or_chunks, str):
            query_text = query_or_chunks
            vectorized_query = self.embeddings.embed_query(query_text)
            return vectorized_query
        elif isinstance(query_or_chunks, list) and all(isinstance(t, str) for t in query_or_chunks):
            chunks = query_or_chunks
            vectorized_chunks = []
            for chunk in chunks:
                embedding = self.embeddings.embed_query(chunk["text"])
                vectorized_chunks.append({
                    "chunk_id": f"{chunk['file_name']}_p{chunk['page_number']}",
                    "chunk": chunk["text"],
                    "page_number": chunk["page_number"],
                    "file_name": chunk["file_name"],
                    "text_vector": embedding
                })
            return vectorized_chunks
        else:
            raise ValueError("Input must be a string or a list of strings.")


if __name__ == "__main__":
    # Example usage
    model = AzureOpenAIEmbeddingModel()

    # Example: Embedding a query
    query = "What is the capital of France?"
    query_embedding = model.embed(query)
    print("Query embedding (first 5 values):", query_embedding[:5])

    # Example: Embedding document chunks
    from rag_preprocess_documents import *
    from pathlib import Path
    
    data_dir = Path(r'c:\Users\deng_jg\work\16centralized_agents\test_data')
    pdf_path = data_dir / "lstm_low_flow.pdf"

    # Load the PDF document
    loader = PDFLoader()
    pages, file_name = loader.load(pdf_path)

    # Create chunks with custom settings
    chunker = TextChunker(chunk_size=1000, overlap_size=200)
    chunks = chunker.chunk_text(pages, file_name)

    # Embed the document chunks
    chunk_embeddings = model.embed(chunks)
    print(f"Generated {len(chunk_embeddings)} embeddings for document chunks.")
    for i, emb in enumerate(chunk_embeddings):
        print(f"Chunk {i+1} - File: {emb['file_name']}, Page: {emb['page_number']}")
        print(f"  Text preview: {emb['chunk'][:100]}...")
        print(f"  Embedding (first 5 values): {emb['text_vector'][:5]}")
        print()
