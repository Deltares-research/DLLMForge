"""
This module provides embedding functionality for RAG (Retrieval-Augmented Generation) pipelines.
It can be used to 1) vectorize document chunks, and 2) vectorize user queries.
The module uses Azure OpenAI embeddings model as an example of using hosted embedding APIs. Note you need
Azure OpenAI service and a deployed embedding model on Azure to use this module.
"""

from anyio import mkdtemp
from dllmforge.LLMs.Deltares_LLMs import DeltaresOllamaLLM
from dllmforge.rag_preprocess_documents import PDFLoader, TextChunker
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from dllmforge.rag_embedding_open_source import LangchainHFEmbeddingModel
# Example: Embedding document chunks
from dllmforge.rag_preprocess_documents import *
from pathlib import Path
import faiss
from dllmforge.rag_evaluation import evaluate_rag_response, RAGEvaluator

from langchain_docling import DoclingLoader
from docling.chunking import HybridChunker
from langchain_docling.loader import ExportType

TEST_QUESTIONS = [{
    "question": "Size of images produced by schemaGAN?",
    "ground_truth": "The images produced by schemaGAN have a size of **512 × 32 pixels**."
}, {
    "question": "What is the network architecture based on?",
    "ground_truth": "the pix2pix method from Isola et al. (2017)"
}]

if __name__ == "__main__":
    # Example usage
    model = LangchainHFEmbeddingModel("intfloat/multilingual-e5-large")

    data_dir = Path(r'D:\\LLMs\\DLLMForge\\tests\\test_input\\piping_documents')
    # find all PDF files in the directory
    pdfs = list(data_dir.glob("*.pdf"))
    global_chunks = []
    for pdf_path in pdfs:
        loader = DoclingLoader(
            file_path=pdf_path,
            export_type=ExportType.DOC_CHUNKS,
            chunker=HybridChunker(
                tokenizer=model.embeddings.model_name,
                chunk_size=512,  # Max length supported by MiniLM
                chunk_overlap=50  # Some overlap for better context
            ))
        docs = loader.load()
        global_chunks.extend(docs)

    print(f"Total embeddings generated: {len(global_chunks)}")
    # now create the vector store

    # create faiss vector store
    index = faiss.IndexFlatL2(len(model.embed("test")))
    vector_store = FAISS(
        embedding_function=model.embeddings,
        index=index,
        docstore=InMemoryDocstore({}),
        index_to_docstore_id={},
    )
    vector_store.add_documents(global_chunks)
    # query the vector store directly to check wat is achterland in piping?
    query_embedding = vector_store.similarity_search_with_score(query="Size of images for schema GAN in pixels", k=5)
    print("Query result:", query_embedding)

    # now create the LLM
    llm = DeltaresOllamaLLM(base_url="https://chat-api.directory.intra", model_name="qwen3:latest", temperature=0.8)

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": 0.1,
            "k": 10
        },
    )

    chat_result = llm.ask_with_retriever("What is the size of images produced by schemaGAN?",
                                         retriever,
                                         max_tokens=5024,
                                         temperature=0.8)
    print("Answer:", chat_result.generations[0].message.content)

    # Now let's evaluate the RAG system
    evaluator = RAGEvaluator(llm_provider="deltares", deltares_llm=llm)
    results = []
    for q_data in TEST_QUESTIONS:
        question = q_data["question"]
        ground_truth = q_data["ground_truth"]
        chat_result = llm.ask_with_retriever(question, retriever)
        answer = chat_result.generations[0].message.content
        answer = answer.split("</think>")[-1].strip()  # Clean up the answer
        retrieved_contexts = retriever.invoke(question)
        evaluation = evaluator.evaluate_rag_pipeline(question=question,
                                                     generated_answer=answer,
                                                     retrieved_contexts=retrieved_contexts,
                                                     ground_truth_answer=ground_truth)
        # Store results
        result = {
            'question': question,
            'ground_truth': ground_truth,
            'response': answer,
            'context': retrieved_contexts,
            'evaluation': evaluation
        }
        results.append(result)
        print(f"Question: {question}")
        print(f"Ground Truth: {ground_truth}")
        print(f"Ragas score: {evaluation.ragas_score}")
        print(f"Relevancy score: {evaluation.answer_relevancy.score}")
        print(f"Faithfulness score: {evaluation.faithfulness.score}")
        print(f"Context recall score: {evaluation.context_recall.score}")
        print(f"Context relevancy score: {evaluation.context_relevancy.score}")
