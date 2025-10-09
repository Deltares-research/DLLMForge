"""
This module provides embedding functionality for RAG (Retrieval-Augmented Generation) pipelines.
It can be used to 1) vectorize document chunks, and 2) vectorize user queries.
The module uses Azure OpenAI embeddings model as an example of using hosted embedding APIs. Note you need
Azure OpenAI service and a deployed embedding model on Azure to use this module.
"""

from typing import List, Any, Union, Dict
import base64
import re
import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings


class AzureOpenAIEmbeddingModel:
    """Class for embedding queries and document chunks using Azure OpenAI Embeddings."""

    def __init__(self,
                 model: str = "text-embedding-3-large",
                 api_base: str = None,
                 deployment_name_embeddings: str = None,
                 api_key: str = None,
                 api_version: str = None):
        """
        Initialize the embedding model using provided arguments or environment variables for Azure OpenAI.
        Args:
            model: Name of the embedding model to use
            api_base: Azure OpenAI API base URL
            deployment_name_embeddings: Azure OpenAI deployment name for embeddings
            api_key: Azure OpenAI API key
            api_version: Azure OpenAI API version
        """
        load_dotenv()
        api_base = api_base or os.getenv('AZURE_OPENAI_API_BASE')
        deployment_name_embeddings = deployment_name_embeddings or os.getenv('AZURE_OPENAI_DEPLOYMENT_EMBEDDINGS')
        api_key = api_key or os.getenv('AZURE_OPENAI_API_KEY')
        api_version = api_version or os.getenv('AZURE_OPENAI_API_VERSION')

        self.embeddings = AzureOpenAIEmbeddings(model=model,
                                                azure_endpoint=api_base,
                                                azure_deployment=deployment_name_embeddings,
                                                api_key=api_key,
                                                openai_api_version=api_version)

    @staticmethod
    def validate_embedding(embedding: List[float]) -> bool:
        """Validate that the embedding is not empty."""
        if not embedding:
            return False
        if not all(isinstance(x, (int, float)) for x in embedding):
            return False
        return True

    @staticmethod
    def encode_filename(filename: str) -> str:
        """Encode filename to be safe for Azure Cognitive Search document keys."""
        # Remove file extension
        name_without_ext = os.path.splitext(filename)[0]
        # Replace spaces and special characters with underscores
        safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', name_without_ext)
        # Encode in base64 and make it URL-safe
        encoded = base64.urlsafe_b64encode(safe_name.encode()).decode()
        # Remove padding
        encoded = encoded.rstrip('=')
        return encoded

    def embed(self, query_or_chunks: Union[str, List[Dict[str, Any]]]) -> Union[List[float], List[Dict[str, Any]]]:
        """
        Embed a single query string or a list of document chunks.
        Args:
            query_or_chunks: A string (query) or a list of dictionaries (document chunks)
                            Each dictionary should have keys: "text", "file_name", "page_number"
        Returns:
            For a string query: list of floats (embedding vector)
            For document chunks: list of dictionaries with keys: "chunk_id", "chunk", "page_number",
                                "file_name", "text_vector"
        """
        if isinstance(query_or_chunks, str):
            query_text = query_or_chunks
            vectorized_query = self.embeddings.embed_query(query_text)
            if not self.validate_embedding(vectorized_query):
                raise ValueError("Invalid embedding generated for query.")
            return vectorized_query
        elif isinstance(query_or_chunks, list) and all(isinstance(t, dict) for t in query_or_chunks):
            chunks = query_or_chunks
            vectorized_chunks = []
            for chunk in chunks:
                embedding = self.embeddings.embed_query(chunk["text"])
                if not self.validate_embedding(embedding):
                    raise ValueError(f"Invalid embedding generated for chunk: {chunk}")

                # encode file_name to be safe for Azure Cognitive Search document keys.
                safe_filename = self.encode_filename(chunk["file_name"])

                vectorized_chunks.append({
                    "chunk_id": f"{safe_filename}_i{chunk['chunk_index']}",
                    "chunk": chunk["text"],
                    "page_number": chunk["page_number"],
                    "file_name": chunk["file_name"],
                    "text_vector": embedding
                })
            return vectorized_chunks
        else:
            raise ValueError("Input must be a string or a list of dictionaries.")


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
        if i < 4:
            print(f"Chunk {i+1} - File: {emb['file_name']}, Page: {emb['page_number']}")
            print(f"  Text preview: {emb['chunk'][:100]}...")
            print(f"  Embedding (first 5 values): {emb['text_vector'][:5]}")
            print()
        else:
            break
