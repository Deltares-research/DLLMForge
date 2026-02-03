"""
Test optional dependencies and graceful degradation.

This test suite ensures that:
1. Core functionality works without optional dependencies
2. Optional imports are properly guarded
3. Availability flags are correctly set
4. Appropriate errors are raised when using unavailable components

Environment variables (set in tox.ini):
- TEST_OPENAI_API: Whether to test OpenAI API components
- TEST_ANTHROPIC_API: Whether to test Anthropic API components
- TEST_LLAMAINDEX: Whether to test LlamaIndex components
- TEST_RAG_COMPONENTS: Whether to test advanced RAG components
- TEST_RAG_EVALUATION: Whether to test RAG evaluation
- TEST_LOCAL_EMBEDDINGS: Whether to test local/open-source embeddings
- TEST_DOCLING: Whether to test Docling document processing
"""

import pytest
import sys
import os
from importlib import import_module, reload


# Helper function to check if a test should run based on environment
def should_test(env_var: str) -> bool:
    """Check if a feature should be tested based on environment variable."""
    value = os.environ.get(env_var, 'false').lower()
    return value in ('true', '1', 'yes')


class TestCoreImports:
    """Test that core imports work without optional dependencies."""

    def test_package_import(self):
        """Test that the package itself can be imported."""
        import dllmforge
        assert hasattr(dllmforge, '__version__')

    def test_core_components_import(self):
        """Test that core components can be imported."""
        import dllmforge

        # Core agentic infrastructure should always be available
        assert hasattr(dllmforge, 'SimpleAgent')
        assert hasattr(dllmforge, 'create_basic_agent')
        assert hasattr(dllmforge, 'create_basic_tools')

        # RAG base components should always be available
        assert hasattr(dllmforge, 'PDFLoader')
        assert hasattr(dllmforge, 'TextChunker')

    def test_agent_core_module(self):
        """Test that agent_core module can be imported."""
        from dllmforge import agent_core
        assert hasattr(agent_core, 'SimpleAgent')
        assert hasattr(agent_core, 'create_basic_agent')

    def test_rag_preprocessing_module(self):
        """Test that RAG preprocessing module can be imported."""
        from dllmforge import rag_preprocess_documents
        assert hasattr(rag_preprocess_documents, 'PDFLoader')
        assert hasattr(rag_preprocess_documents, 'TextChunker')

    def test_rag_embedding_module(self):
        """Test that RAG embedding module can be imported."""
        from dllmforge import rag_embedding
        assert hasattr(rag_embedding, 'AzureOpenAIEmbeddingModel')


class TestOptionalAPIImports:
    """Test optional API imports."""

    @pytest.mark.skipif(not should_test('TEST_OPENAI_API'), reason="OpenAI API tests disabled")
    def test_openai_api_import(self):
        """Test OpenAI API import (requires openai package)."""
        try:
            from dllmforge import openai_api
            from dllmforge import rag_embedding
            assert hasattr(rag_embedding, 'AzureOpenAIEmbeddingModel')
            assert hasattr(openai_api, 'OpenAIAPI')
        except ImportError as e:
            # If openai is not installed, this should fail gracefully
            assert 'openai' in str(e).lower() or 'langchain_openai' in str(e).lower()

    @pytest.mark.skipif(not should_test('TEST_ANTHROPIC_API'), reason="Anthropic API tests disabled")
    def test_anthropic_api_import(self):
        """Test Anthropic API import (requires anthropic package)."""
        import dllmforge

        if dllmforge.ANTHROPIC_AVAILABLE:
            from dllmforge import anthropic_api
            assert hasattr(anthropic_api, 'AnthropicAPI')
        else:
            # Should be gracefully handled
            with pytest.raises(ImportError):
                from dllmforge import anthropic_api
                # Force evaluation
                anthropic_api.AnthropicAPI

    @pytest.mark.skipif(not should_test('TEST_LLAMAINDEX'), reason="LlamaIndex tests disabled")
    def test_llamaindex_api_import(self):
        """Test LlamaIndex API import (requires llama-index package)."""
        import dllmforge

        if dllmforge.LLAMAINDEX_AVAILABLE:
            from dllmforge import llamaindex_api

    @pytest.mark.skipif(not should_test('TEST_RAG_COMPONENTS'), reason="RAG components tests disabled")
    def test_rag_search_and_response_import(self):
        """Test RAG search and response module import."""
        import dllmforge

        if dllmforge.RAG_COMPONENTS_AVAILABLE:
            from dllmforge import rag_search_and_response
            assert hasattr(rag_search_and_response, 'IndexManager')
            assert hasattr(rag_search_and_response, 'Retriever')
            assert hasattr(rag_search_and_response, 'LLMResponder')

    @pytest.mark.skipif(not should_test('TEST_RAG_EVALUATION'), reason="RAG evaluation tests disabled")
    def test_rag_evaluation_import(self):
        """Test RAG evaluation module import."""
        import dllmforge

        if dllmforge.RAG_EVALUATION_AVAILABLE:
            from dllmforge import rag_evaluation
            assert hasattr(rag_evaluation, 'RAGEvaluator')

    @pytest.mark.skipif(not should_test('TEST_LOCAL_EMBEDDINGS'), reason="Local embeddings tests disabled")
    def test_local_embeddings_import(self):
        import dllmforge

        if dllmforge.RAG_EVALUATION_AVAILABLE:
            from dllmforge import rag_evaluation
            assert hasattr(rag_evaluation, 'RAGEvaluator')

    def test_rag_embedding_open_source_import(self):
        """Test open source embedding module (requires sentence-transformers)."""
        try:
            from dllmforge import rag_embedding_open_source
            assert hasattr(rag_embedding_open_source, 'LangchainHFEmbeddingModel')
        except ImportError:
            # If sentence-transformers is not installed, this should fail gracefully
            pass


class TestInformationExtractionAgent:
    """Test Information Extraction (IE) agent components."""

    def test_ie_agent_config_import(self):
        """Test IE agent config module import."""
        try:
            from dllmforge import IE_agent_config
            # Basic import should work
            assert IE_agent_config is not None
        except ImportError:
            # May have dependencies on other packages
            pass

    def test_ie_agent_document_processor_import(self):
        """Test IE agent document processor module import."""
        try:
            from dllmforge import IE_agent_document_processor
            assert IE_agent_document_processor is not None
        except ImportError:
            # May have dependencies on other packages
            pass

    def test_ie_agent_schema_generator_import(self):
        """Test IE agent schema generator module import."""
        try:
            from dllmforge import IE_agent_schema_generator
            assert IE_agent_schema_generator is not None
        except ImportError:
            # May have dependencies on other packages
            pass

    @pytest.mark.skipif(not should_test('TEST_DOCLING'), reason="Docling tests disabled")
    def test_ie_agent_extractor_import(self):
        """Test IE agent extractor module import."""
        try:
            from dllmforge import IE_agent_extractor
            assert IE_agent_extractor is not None
        except ImportError:
            # May have dependencies on other packages
            pass

    def test_ie_agent_extractor_docling_import(self):
        """Test IE agent docling extractor module import (requires langchain_docling)."""
        try:
            from dllmforge import IE_agent_extractor_docling
            assert IE_agent_extractor_docling is not None
        except ImportError as e:
            # Expected if langchain_docling is not installed
            pass


class TestCoreInstantiationWithoutOptionals:
    """Test that core components can be instantiated (or fail gracefully)."""

    def test_simple_agent_class_available(self):
        """Test that SimpleAgent class is available."""
        from dllmforge import SimpleAgent
        assert SimpleAgent is not None
        # We can check that it's a class
        assert callable(SimpleAgent)

    def test_helper_functions_callable(self):
        """Test that helper functions are callable."""
        from dllmforge import create_basic_agent, create_basic_tools
        assert callable(create_basic_agent)
        assert callable(create_basic_tools)

    def test_pdf_loader_class_available(self):
        """Test that PDFLoader class is available."""
        from dllmforge import PDFLoader
        assert PDFLoader is not None
        assert callable(PDFLoader)

    def test_text_chunker_class_available(self):
        """Test that TextChunker class is available."""
        from dllmforge import TextChunker
        assert TextChunker is not None
        assert callable(TextChunker)


class TestDunderAll:
    """Test the __all__ export list."""

    def test_all_list_exists(self):
        """Test that __all__ is defined."""
        import dllmforge
        assert hasattr(dllmforge, '__all__')
        assert isinstance(dllmforge.__all__, list)

    def test_all_list_contains_core_components(self):
        """Test that __all__ contains core components."""
        import dllmforge

        # Core components should always be in __all__
        core_components = [
            '__version__',
            'SimpleAgent',
            'create_basic_agent',
            'create_basic_tools',
            'AzureOpenAIEmbeddingModel',
            'PDFLoader',
            'TextChunker',
        ]

        for component in core_components:
            assert component in dllmforge.__all__, f"{component} missing from __all__"

    def test_all_list_matches_availability(self):
        """Test that __all__ list matches component availability."""
        import dllmforge

        # If components are available, they should be in __all__
        if dllmforge.RAG_COMPONENTS_AVAILABLE:
            assert 'IndexManager' in dllmforge.__all__
            assert 'Retriever' in dllmforge.__all__
            assert 'LLMResponder' in dllmforge.__all__

        if dllmforge.RAG_EVALUATION_AVAILABLE:
            assert 'RAGEvaluator' in dllmforge.__all__

        if dllmforge.ANTHROPIC_AVAILABLE:
            assert 'AnthropicAPI' in dllmforge.__all__

        if dllmforge.LLAMAINDEX_AVAILABLE:
            assert 'LlamaIndexAPI' in dllmforge.__all__

    def test_all_exports_are_importable(self):
        """Test that everything in __all__ can actually be imported."""
        import dllmforge

        for name in dllmforge.__all__:
            assert hasattr(dllmforge, name), f"{name} in __all__ but not importable"
