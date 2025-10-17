DLLMForge Documentation
=======================

Welcome to DLLMForge
--------------------

DLLMForge is a repository for LLM (Large Language Model) tools developed at Deltares. 
It provides simple open and closed source tools to interact with various with D-LLMForge
you can :

- Use an simple LLM to ask questions.
- Build your RAG with HuggingFace or AZURE embeddings and vector stores.

Features
--------

DLLMForge provides a modular toolkit for:

- **Multi-LLM Support**: Integration with OpenAI, Anthropic, and open-source Deltares hosted models
- **RAG Pipeline**: Complete document ingestion, embedding, and retrieval system
- **Agent Framework**: Simple but extensible agent architecture with tool support
- **Evaluation Tools**: Comprehensive RAG system evaluation using various metrics
- **Flexible Backends**: Support for both cloud (Azure, OpenAI) and local deployments

Repository Structure
--------------------

DLLMForge is organized into several key components that work together to provide a comprehensive LLM toolkit:

Core Package (``dllmforge/``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The main package contains the following modules:

**Core Agent Framework**
  - ``agent_core.py`` - Simple agent infrastructure with tool support
    
    * :class:`~dllmforge.agent_core.SimpleAgent` - Basic agentic workflows
    * :func:`~dllmforge.agent_core.create_basic_agent` - Agent factory function
    * :func:`~dllmforge.agent_core.create_basic_tools` - Tool creation utilities

**LLM API Integrations**
  - ``openai_api.py`` - OpenAI API integration
    
    * :class:`~dllmforge.openai_api.OpenAIAPI` - OpenAI API wrapper
    
  - ``anthropic_api.py`` - Anthropic Claude API integration
    
    * :class:`~dllmforge.anthropic_api.AnthropicAPI` - Anthropic API wrapper
    
  - ``langchain_api.py`` - LangChain framework integration
  - ``llamaindex_api.py`` - LlamaIndex framework integration
    
    * :class:`~dllmforge.llamaindex_api.LlamaIndexAPI` - LlamaIndex API wrapper

**RAG (Retrieval-Augmented Generation) Components**
  - ``rag_preprocess_documents.py`` - Document loading and chunking
    
    * :class:`~dllmforge.rag_preprocess_documents.DocumentLoader` - Abstract document loader
    * :class:`~dllmforge.rag_preprocess_documents.PDFLoader` - Load PDF documents
    * :class:`~dllmforge.rag_preprocess_documents.TextChunker` - Split text into manageable chunks with overlap
    
  - ``rag_embedding.py`` - Azure OpenAI embedding models
    
    * :class:`~dllmforge.rag_embedding.AzureOpenAIEmbeddingModel` - Generate embeddings for text
    
  - ``rag_embedding_open_source.py`` - Open-source embedding models
    
    * :class:`~dllmforge.rag_embedding_open_source.LangchainHFEmbeddingModel` - HuggingFace embeddings via LangChain
    
  - ``rag_search_and_response.py`` - Search and response generation
    
    * :class:`~dllmforge.rag_search_and_response.IndexManager` - Manage vector indices
    * :class:`~dllmforge.rag_search_and_response.Retriever` - Retrieve relevant documents
    * :class:`~dllmforge.rag_search_and_response.LLMResponder` - Generate responses using LLMs
    
  - ``rag_evaluation.py`` - RAG system evaluation
    
    * :class:`~dllmforge.rag_evaluation.RAGEvaluator` - Evaluate RAG system performance
    * :class:`~dllmforge.rag_evaluation.EvaluationResult` - Store individual evaluation metrics
    * :class:`~dllmforge.rag_evaluation.RAGEvaluationResult` - Store comprehensive RAG evaluation results

**Specialized Components**
  - ``LLMs/Deltares_LLMs.py`` - Deltares-specific LLM implementations
  - ``utils/`` - Utility functions and helpers



Workflows (``workflows/``)
~~~~~~~~~~~~~~~~~~~~~~~~~~

  - ``open_source_RAG.py`` - Example workflow for open-source RAG implementation

Example Streamlit-based Application (``streamlit_apps/``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   - ``app.py`` - Streamlit-based RAG application
   - ``streamlit_water_management_app.py`` - Streamlit-based water management application


Quick Start
-----------

Installation
~~~~~~~~~~~~

To install DLLMForge, you can use pip:

.. code-block:: bash

   pip install git+https://github.com/Deltares-research/DLLMForge


Tutorials
~~~~~~~~~
The following tutorials are available:

- :doc:`tutorials/LLM_tutorial`
- :doc:`tutorials/RAG_tutorial`

Background Information
----------------------

For more information on LLMs and RAG systems, see:

- :doc:`background/LLM_explained`
- :doc:`background/RAGS_explained`


API Reference
-------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   tutorials/LLM_tutorial
   tutorials/RAG_tutorial
   modules
   background/LLM_explained
   background/RAGS_explained

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`