"""
This module provides embedding functionality for RAG (Retrieval-Augmented Generation) pipelines.
It can be used to 1) vectorize document chunks, and 2) vectorize user queries.
The module uses Azure OpenAI embeddings model as an example of using hosted embedding APIs. Note you need
Azure OpenAI service and a deployed embedding model on Azure to use this module.
"""

from typing import List, Any, Union, Dict
import os
from langchain_huggingface import HuggingFaceEmbeddings
from dllmforge.LLMs.Deltares_LLMs import DeltaresOllamaLLM
from langchain.schema import HumanMessage, SystemMessage

class LangchainHFEmbeddingModel:
    """Class for embedding queries and document chunks using LangChain's HuggingFaceEmbeddings."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the HuggingFaceEmbeddings from LangChain.

        Args:
            model_name: Name or path of the Hugging Face model (default: "sentence-transformers/all-MiniLM-L6-v2").
        """
        # kwargs for encoder; adjust as needed
        encode_kwargs = {"normalize_embeddings": False}
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs=encode_kwargs
        )

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


if __name__ == "__main__":
    # Example usage
    model = LangchainHFEmbeddingModel(model_name="Linq-AI-Research/Linq-Embed-Mistral") # Qwen/Qwen3-Embedding-8B


    # Example: Embedding document chunks
    from dllmforge.rag_preprocess_documents import *
    from pathlib import Path
    
    data_dir = Path(r'D:\\LLMs\\DLLMForge\\tests\\test_input\\piping_documents')
    # find all PDF files in the directory
    pdfs = list(data_dir.glob("*.pdf"))
    # Load the PDF document
    loader = PDFLoader()
    # Create chunks with custom settings
    chunker = TextChunker(chunk_size=1000, overlap_size=200)
    global_embeddings = []
    metadatas = []
    for pdf_path in pdfs:
        pages, file_name, metadata = loader.load(pdf_path)
        # Create chunks with custom settings
        chunks = chunker.chunk_text(pages, file_name, metadata)
        # Embed the document chunks
        chunk_embeddings = model.embed(chunks)
        global_embeddings.extend(chunk_embeddings)
        metadatas.extend([chunk["metadata"] for chunk in chunks])
        print(f"Embedded {len(chunk_embeddings)} chunks from {file_name}.")
    print(f"Total embeddings generated: {len(global_embeddings)}")
    # now create the vector store
    import faiss
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from langchain_community.vectorstores import FAISS


    # Dimension of embeddings
    index = faiss.IndexFlatL2(len(global_embeddings[0]["text_vector"]))

    vector_store = FAISS(
        embedding_function=model.embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    # Add embeddings to the vector store
    for chunk, meta in zip(global_embeddings, metadatas):

        vector_store.add_texts(
            texts=[chunk["chunk"]],
            metadatas=[meta],
            ids=[chunk["chunk_id"]],
            embeddings=[chunk["text_vector"]]
        )

    # query the vector store directly to check wat is achterland in piping?
    query_embedding = vector_store.similarity_search_with_score(
        query="kritisch stijghoogteverschil piping",
        k=5
    )
    print("Query result:", query_embedding)

    # now create the LLM
    llm = DeltaresOllamaLLM(base_url="https://chat-api.directory.intra", model_name="qwen3:latest", temperature=0.8)

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.5, "k": 10},
    )

    chat_result = llm.ask_with_retriever("Wat is de kritisch stijghoogteverschil piping?", retriever)
    print("Answer:", chat_result.generations[0].message.content)



