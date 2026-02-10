=========================================
Open Source RAG Pipeline Tutorial
=========================================

This tutorial demonstrates how to build a complete Retrieval-Augmented Generation (RAG) pipeline using open source components from the DLLMForge library. The pipeline includes document preprocessing, open source embeddings, vector storage, and RAG evaluation.

Overview
========

The RAG pipeline consists of several key components:

1. **Document Loading**: Load PDF documents and extract text
2. **Text Chunking**: Split documents into manageable chunks
3. **Embedding Generation**: Create vector embeddings using open source models
4. **Vector Storage**: Store embeddings in a FAISS vector database
5. **Retrieval**: Find relevant document chunks for queries
6. **Generation**: Generate answers using an open source LLM
7. **Evaluation**: Assess the quality of the RAG system

Step-by-Step Implementation
===========================

1. Import Required Modules
--------------------------

Start by importing all necessary components:

.. code-block:: python
    
    from dllmforge.rag_embedding_open_source import LangchainHFEmbeddingModel
    from dllmforge.rag_evaluation import RAGEvaluator
    from dllmforge.LLMs.Deltares_LLMs import DeltaresOllamaLLM
    from dllmforge.rag_preprocess_documents import PDFLoader, TextChunker
    
    from langchain_community.vectorstores import FAISS
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from pathlib import Path
    import faiss

2. Initialize the Embedding Model
---------------------------------

Create an embedding model using open source HuggingFace transformers:

.. code-block:: python

    # Initialize the embedding model
    # Default model: "sentence-transformers/all-MiniLM-L6-v2"
    model = LangchainHFEmbeddingModel("intfloat/multilingual-e5-large")
    

The ``LangchainHFEmbeddingModel`` class supports any HuggingFace sentence transformer model and provides:

- Automatic model downloading and caching
- Batch embedding for efficient processing
- Input validation for embeddings

We recommend starting with the default model and experimenting with others based on your use case.
The ranked list of popular models can be found at: https://huggingface.co/spaces/mteb/leaderboard.
Note that the bigger the number of parameters, the better the performance, but also the higher the resource requirements.
Which implies that the model might not fit into memory on smaller machines and take a long time to download.

3. Load and Process Documents
-----------------------------

First, you will need to download some PDF documents to use as your knowledge base.
In this example, we use the schemaGAN paper from science direct (https://www.sciencedirect.com/science/article/pii/S0266352X25001260).
Download this pdf into a local directory and update the path below.
In our case, we copy the pdf into a folder named "documents" in the root of the repository.

Two important parameters to consider when chunking documents are ``chunk_size`` and ``overlap_size``.

- ``chunk_size``: The maximum size of each text chunk (in characters). Smaller chunks may improve retrieval performance but increase the number of chunks.
- ``overlap_size``: The number of overlapping characters between chunks. Overlapping chunks can help preserve context but may increase redundancy.

Load PDF documents and create text chunks:

.. code-block:: python

    # Define the directory containing PDF documents
    data_dir = Path(r'documents')
    pdfs = list(data_dir.glob("*.pdf"))
    
    # Initialize document loader and chunker
    loader = PDFLoader()
    chunker = TextChunker(chunk_size=1000, overlap_size=200)
    
    global_embeddings = []
    metadatas = []
    
    # Process each PDF file
    for pdf_path in pdfs:
        # Load the PDF document
        pages, file_name, metadata = loader.load(pdf_path)
        
        # Create chunks with overlap for better context preservation
        chunks = chunker.chunk_text(pages, file_name, metadata)
        
        # Generate embeddings for chunks
        chunk_embeddings = model.embed(chunks)
        
        # Store embeddings and metadata
        global_embeddings.extend(chunk_embeddings)
        metadatas.extend([chunk["metadata"] for chunk in chunks])
        
        print(f"Embedded {len(chunk_embeddings)} chunks from {file_name}.")
    
    print(f"Total embeddings generated: {len(global_embeddings)}")

After running this code, you should see output indicating the number of chunks embedded from each PDF and the total embeddings generated.
In this example, you will see the following output:

    Embedded 107 chunks from 1-s2.0-S0266352X25001260-main.pdf.
    Total embeddings generated: 107


4. Create Vector Store
----------------------

Set up a FAISS vector store for efficient similarity search:
In this example, the FAISS index was used but other index types can be used as well, like MongoDB or Weaviate.

.. code-block:: python

    # Get embedding dimension
    embedding_dim = len(global_embeddings[0]["text_vector"])
    
    # Create FAISS index for L2 (Euclidean) distance
    index = faiss.IndexFlatL2(embedding_dim)
    
    # Initialize vector store
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

**Alternative Index Types:**

- ``IndexFlatL2``: Exact L2 distance (slower but accurate)
- ``IndexFlatIP``: Inner product similarity
- ``IndexIVFFlat``: Faster approximate search for large datasets

5. Test Retrieval
------------------

Test the vector store with a sample query:

.. code-block:: python

    # Query the vector store directly
    query_embedding = vector_store.similarity_search_with_score(
        query="Size of images for schema GAN", 
        k=5
    )
    
    print("Query result:", query_embedding)
    
    # Each result contains (Document, similarity_score)
    for doc, score in query_embedding:
        print(f"Score: {score}")
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Metadata: {doc.metadata}")
        print("---")

6. Initialize the LLM
---------------------

Set up the open source language model using Ollama, in this case the Qwen3 model:

.. code-block:: python

    # Initialize Ollama LLM
    llm = DeltaresOllamaLLM(
        base_url="https://chat-api.directory.intra",  
        model_name="qwen3:latest",  # Or another available model
        temperature=0.8
    )

7. Create Retriever and Generate Answers
-----------------------------------------

Set up the retriever and generate answers to questions:

.. code-block:: python

    # Create retriever with similarity threshold
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": 0.1,  # Minimum similarity score
            "k": 10  # Maximum number of documents to retrieve
        },
    )
    
    # Generate answer using RAG
    question = "Size of images produced by schemaGAN? give me the answer in axb format"
    chat_result = llm.ask_with_retriever(question, retriever)
    answer = chat_result.generations[0].message.content
    
    print("Answer:", answer)

The answer should be relevant to the question based on the retrieved documents.
If the answer is not satisfactory, consider refining the query or adjusting the retriever and/or embedding model.

8. Evaluate the RAG System
---------------------------

Use the built-in evaluation framework to assess RAG performance:

.. code-block:: python

    # Define test questions with ground truth answers
    TEST_QUESTIONS = [{
        "question": "Size of images produced by schemaGAN?",
        "ground_truth": "The images produced by schemaGAN have a size of **512 × 32 pixels**."
    }, {
        "question": "What is the network architecture based on?",
        "ground_truth": "the pix2pix method from Isola et al. (2017)"
    }]
    
    # Initialize evaluator
    evaluator = RAGEvaluator(llm_provider="deltares", deltares_llm=llm)
    
    results = []
    for q_data in TEST_QUESTIONS:
        question = q_data["question"]
        ground_truth = q_data["ground_truth"]
        
        # Generate answer
        chat_result = llm.ask_with_retriever(question, retriever)
        answer = chat_result.generations[0].message.content
        answer = answer.split("</think>")[-1].strip()  # Clean up response
        
        # Get retrieved contexts
        retrieved_contexts = retriever.invoke(question)
        
        # Evaluate the RAG pipeline
        evaluation = evaluator.evaluate_rag_pipeline(
            question=question,
            generated_answer=answer,
            retrieved_contexts=retrieved_contexts,
            ground_truth_answer=ground_truth
        )
        
        # Store results
        result = {
            'question': question,
            'ground_truth': ground_truth,
            'response': answer,
            'context': retrieved_contexts,
            'evaluation': evaluation
        }
        results.append(result)
        
        # Print evaluation metrics
        print(f"Question: {question}")
        print(f"RAGAS Score: {evaluation.ragas_score:.3f}")
        print(f"Answer Relevancy: {evaluation.answer_relevancy.score:.3f}")
        print(f"Faithfulness: {evaluation.faithfulness.score:.3f}")
        print(f"Context Recall: {evaluation.context_recall.score:.3f}")
        print(f"Context Relevancy: {evaluation.context_relevancy.score:.3f}")
        print("---")

Evaluation Metrics Explained
=============================

The RAG evaluation provides four key metrics:

**1. Context Relevancy (0-1)**
   Measures how relevant the retrieved documents are to the question. Higher scores indicate better retrieval.
   If the score is low, it indicates a potential issue with the embedding model or vector store.

**2. Context Recall (0-1)**
   Measures whether all necessary information was retrieved. Compares retrieved context with ground truth.
   Low scores suggest that important documents may be missing from the retrieval.
   There is possible room for improvement by adjusting chunk size, overlap, or using a different embedding model.

**3. Faithfulness (0-1)**
   Measures factual accuracy and absence of hallucinations. Checks if the answer is grounded in the retrieved context.
   Low scores indicate that the LLM may be generating information not supported by the context.
   Adjust the LLM temperature or try a different model to improve faithfulness.

**4. Answer Relevancy (0-1)**
   Measures how directly the answer addresses the question. Penalizes verbose or off-topic responses.
   Low scores suggest that the LLM may not be effectively using the retrieved context.
   Experiment with prompt engineering or different LLMs to enhance answer relevancy.

**RAGAS Score**
   Overall score combining all metrics, providing a single measure of RAG system quality.

Advanced Configuration
======================

Optimizing Chunk Size
----------------------

Experiment with different chunk sizes based on your documents:

.. code-block:: python

    # For technical documents with detailed information
    chunker = TextChunker(chunk_size=1500, overlap_size=300)
    
    # For shorter, conversational content
    chunker = TextChunker(chunk_size=500, overlap_size=100)
    
    # For very long documents
    chunker = TextChunker(chunk_size=2000, overlap_size=400)

Using Different Embedding Models
---------------------------------

Try different embedding models for better performance:

.. code-block:: python

    # More capable but larger model
    model = LangchainHFEmbeddingModel(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    
    # Multilingual model
    model = LangchainHFEmbeddingModel(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # Domain-specific model (if available)
    model = LangchainHFEmbeddingModel(
        model_name="sentence-transformers/allenai-specter"
    )

Improving Retrieval
-------------------

Fine-tune retrieval parameters:

.. code-block:: python

    # For high precision (fewer but more relevant results)
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": 0.3,  # Higher threshold
            "k": 5  # Fewer results
        },
    )
    
    # For high recall (more results, potentially less relevant)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 20  # More results
        },
    )
    
    # For diverse results using Maximum Marginal Relevance
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 10,
            "lambda_mult": 0.7  # Balance between relevance and diversity
        },
    )

Docling Automatic chunker
--------------------------

If the documents are not very well structured, you can use the automatic chunker from docling.
This chunker uses a language model to create chunks based on the content and structure of the document.
Docling chunks are typically more semantically meaningful than fixed-size chunks, and can improve retrieval performance.

The following code snippet shows how to use the docling chunker. You should replace the previous chunking code with this code.

.. code-block:: python

    from langchain_docling import DoclingLoader
    from docling.chunking import HybridChunker
    from langchain_docling.loader import ExportType

    # find all PDF files in the directory
    pdfs = list(data_dir.glob("*.pdf"))
    global_chunks = []
    for pdf_path in pdfs:
        loader = DoclingLoader(
            file_path=pdf_path,
            export_type=ExportType.DOC_CHUNKS,
            chunker=HybridChunker(tokenizer=model.embeddings.model_name,
                chunk_size=512,        # Max length supported by MiniLM
                chunk_overlap=50      # Some overlap for better context
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

    
