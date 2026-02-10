"""
This module provides embedding functionality for RAG (Retrieval-Augmented Generation) pipelines.
It can be used to 1) vectorize document chunks, and 2) vectorize user queries.
The module uses Azure OpenAI embeddings model as an example of using hosted embedding APIs. Note you need
Azure OpenAI service and a deployed embedding model on Azure to use this module.
"""

from typing import List, Any, Union, Dict

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    HuggingFaceEmbeddings = None


class LangchainHFEmbeddingModel:
    """Class for embedding queries and document chunks using LangChain's HuggingFaceEmbeddings."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the HuggingFaceEmbeddings from LangChain.

        Args:
            model_name: Name or path of the Hugging Face model (default: "sentence-transformers/all-MiniLM-L6-v2").
        """
        # kwargs for encoder; adjust as needed
        encode_kwargs = {"normalize_embeddings": False}  #, "trust_remote_code": True}
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name, encode_kwargs=encode_kwargs)

    @staticmethod
    def validate_embedding(embedding: List[float]) -> bool:
        """Validate that the embedding vector is non-empty and numeric."""
        if not embedding:
            return False
        if not all(isinstance(x, (int, float)) for x in embedding):
            return False
        return True

    def embed(self, query_or_chunks: Union[str, List[Dict[str, Any]]]) -> Union[List[float], List[Dict[str, Any]]]:
        """
        Embed a single query string or a list of document chunks.

        Args:
            query_or_chunks: A string (query) or list of dicts with keys:
                            "text", "file_name", "page_number".

        Returns:
            - For a string query: list of floats (vector embedding).
            - For document chunks: list of dicts with keys:
              "chunk_id", "chunk", "page_number", "file_name", "text_vector".
        """
        # Single query
        if isinstance(query_or_chunks, str):
            vectorized_query = self.embeddings.embed_query(query_or_chunks)
            if not self.validate_embedding(vectorized_query):
                raise ValueError("Invalid embedding generated for query.")
            return vectorized_query

        # List of chunks
        if isinstance(query_or_chunks, list) and all(isinstance(c, dict) for c in query_or_chunks):
            texts = [chunk["text"] for chunk in query_or_chunks]
            embeddings = self.embeddings.embed_documents(texts)
            vectorized_chunks = []
            for chunk, emb in zip(query_or_chunks, embeddings):
                if not self.validate_embedding(emb):
                    raise ValueError(f"Invalid embedding for chunk: {chunk}")
                vectorized_chunks.append({
                    "chunk_id": f"{chunk['file_name']}_i{chunk['chunk_index']}",
                    "chunk": chunk["text"],
                    "page_number": chunk["page_number"],
                    "file_name": chunk["file_name"],
                    "text_vector": emb
                })
            return vectorized_chunks

        raise ValueError("Input must be a string or a list of dictionaries.")
