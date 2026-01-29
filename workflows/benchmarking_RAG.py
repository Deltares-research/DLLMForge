"""
This module provides embedding functionality for RAG (Retrieval-Augmented Generation) pipelines.
It can be used to 1) vectorize document chunks, and 2) vectorize user queries.
The module uses Azure OpenAI embeddings model as an example of using hosted embedding APIs. Note you need
Azure OpenAI service and a deployed embedding model on Azure to use this module.
"""

from dllmforge.LLMs.Deltares_LLMs import DeltaresOllamaLLM
from dllmforge.rag_preprocess_documents import PDFLoader, TextChunker
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from dllmforge.rag_embedding_open_source import LangchainHFEmbeddingModel
from dllmforge.rag_search_and_response import AzureOpenAIEmbeddingModel, IndexManager, Retriever, LLMResponder
from dllmforge.langchain_api import LangchainAPI

# Example: Embedding document chunks
from dllmforge.rag_preprocess_documents import *
from pathlib import Path
import pathlib
import faiss
from dllmforge.rag_evaluation import evaluate_rag_response, RAGEvaluator

from langchain_docling import DoclingLoader
from docling.chunking import HybridChunker
from langchain_docling.loader import ExportType
import pandas as pd

CHUNKER = "text"  # options are 'docling' or 'text'


def docling_load_and_chunk(pdfs, model_embed):
    global_chunks = []
    for pdf_path in pdfs:
        loader = DoclingLoader(
            file_path=pdf_path,
            export_type=ExportType.DOC_CHUNKS,
            chunker=HybridChunker(
                tokenizer=model_embed.embeddings.model_name,
                chunk_size=512,  # Max length supported by MiniLM
                chunk_overlap=50,  # Some overlap for better context
            ),
        )
        docs = loader.load()
        global_chunks.extend(docs)
    return global_chunks


def text_load_and_chunk(pdfs, model_embed):
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
        chunk_embeddings = model_embed.embed(chunks)
        global_embeddings.extend(chunk_embeddings)
        metadatas.extend([chunk["metadata"] for chunk in chunks])
    return global_embeddings, metadatas


def set_up_RAG_AZURE(
    data_dir,
    index_name,
    index_exists=False,
    chunk_size=1000,
    overlap_size=200,
    embedding_model="text-embedding-3-large",
):
    # initialize the embedding model
    model = AzureOpenAIEmbeddingModel(
        model=embedding_model)  # here there is a default model set, you can customize it if needed
    if not (index_exists):
        pdfs = list(data_dir.glob("*.pdf"))  # find all PDF files in the directory
        loader = PDFLoader()  # Load the PDF document
        chunker = TextChunker(chunk_size=chunk_size, overlap_size=overlap_size)  # Create chunks with custom settings
        embedding_dim = 3072  # Adjust if your embedding model uses a different dimension
        index_manager = IndexManager(index_name=index_name, embedding_dim=embedding_dim)
        index_manager.create_index()
        # embed the chunks
        global_embeddings = []
        for pdf_path in pdfs:
            pages_with_text, file_name, metadata = loader.load(pdf_path)
            # Create chunks with custom settings
            chunks = chunker.chunk_text(pages_with_text, file_name)
            # Embed the document chunks
            chunk_embeddings = model.embed(chunks)
            global_embeddings.extend(chunk_embeddings)
            print(f"Embedded {len(chunk_embeddings)} chunks from {file_name}.")
            index_manager.upload_documents(chunk_embeddings)
    # Retrieval phase
    retriever = Retriever(embedding_model=model, index_name=index_name)
    return retriever


def set_up_RAG_local(data_dir):
    # Example usage
    model = LangchainHFEmbeddingModel("intfloat/multilingual-e5-large")
    # now create or read the vector store
    temp_dir = r"temp_vector_store_minikennisbank"
    # if it exists load it
    if pathlib.Path(temp_dir).exists():
        vector_store = FAISS.load_local(temp_dir, model.embeddings, allow_dangerous_deserialization=True)
        print(f"Loaded vector store from {temp_dir}")
    else:
        # find all PDF files in the directory
        pdfs = list(data_dir.glob("*.pdf"))
        if CHUNKER == "docling":
            global_chunks = docling_load_and_chunk(pdfs, model)
        else:
            global_chunks, metadatas = text_load_and_chunk(pdfs, model)
        print(f"Total embeddings generated: {len(global_chunks)}")
        # create faiss vector store
        index = faiss.IndexFlatL2(len(model.embed("test")))
        vector_store = FAISS(
            embedding_function=model.embeddings,
            index=index,
            docstore=InMemoryDocstore({}),
            index_to_docstore_id={},
        )
        if CHUNKER == "docling":
            vector_store.add_documents(global_chunks)
        else:
            for chunk, meta in zip(global_chunks, metadatas):
                vector_store.add_texts(texts=[chunk["chunk"]],
                                       metadatas=[meta],
                                       ids=[chunk["chunk_id"]],
                                       embeddings=[chunk["text_vector"]])
        # create the directory if it doesn't exist
        pathlib.Path(temp_dir).mkdir(parents=True, exist_ok=True)
        vector_store.save_local(temp_dir)
    print(f"Vector store saved to {temp_dir}")
    # now create the LLM
    llm = DeltaresOllamaLLM(base_url="https://chat-api.directory.intra", model_name="qwen3:latest", temperature=0.4)

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": 0.1,
            "k": 10
        },
    )
    return llm, retriever


def set_up_test_questions_from_csv(csv_file):
    df_questions = pd.read_csv(csv_file, delimiter=";")
    test_questions = []
    for _, row in df_questions.iterrows():
        test_questions.append({
            "Question_ID": row["Question_ID"],
            "question": row["Question_Text"],
            "ground_truth": row["Ground_truth"],
            "Document_name": row["Document_name"],
        })
    return test_questions


if __name__ == "__main__":
    data_dir = Path(r"minikennisbank")
    retriever = set_up_RAG_AZURE(data_dir, index_exists=True, index_name="minikennisbank-index-test")
    llm = LangchainAPI(model_provider="azure-openai", temperature=0.4).llm
    # Now let's evaluate the RAG system
    evaluator = RAGEvaluator(llm_provider="azure-openai")
    test_questions = set_up_test_questions_from_csv(r"Benchmarks.csv")
    results = []
    for q_data in test_questions:
        question = q_data["question"]
        ground_truth = q_data["ground_truth"]

        retrieved_contexts = retriever.invoke(question)
        responder = LLMResponder(llm)
        answer = responder.generate(question, retrieved_contexts)

        evaluation = evaluator.evaluate_rag_pipeline(
            question=question,
            generated_answer=answer,
            retrieved_contexts=retrieved_contexts,
            ground_truth_answer=ground_truth,
        )
        # Store results
        result = {
            "question": question,
            "ground_truth": ground_truth,
            "response": answer,
            "context": retrieved_contexts,
            "evaluation": evaluation,
        }
        results.append(result)
        print(f"Question: {question}")
        print(f"Ground Truth: {ground_truth}")
        print(f"Response: {answer}")
        print(f"Ragas score: {evaluation.ragas_score}")
        print(f"Relevancy score: {evaluation.answer_relevancy.score}")
        print(f"Faithfulness score: {evaluation.faithfulness.score}")
        print(f"Context recall score: {evaluation.context_recall.score}")
        print(f"Context precision score: {evaluation.context_precision.score}")
        # save the results into the directory
        output_dir = Path(r"results_open_ai")
        output_dir.mkdir(parents=True, exist_ok=True)
        evaluator.save_evaluation_results(evaluation, str(output_dir / f"evaluation_{q_data['Question_ID']}.json"))
