"""
DLLMForge - Deltares LLM Forge Toolkit

A comprehensive toolkit for building and deploying LLM-based applications
with RAG capabilities, agentic workflows, and enterprise-grade features.
"""

# Version information
from .__version__ import __version__

# Core agentic infrastructure (always available)
from .agent_core import (SimpleAgent, create_basic_agent, create_basic_tools)

# RAG components (always available)
from .rag_embedding import AzureOpenAIEmbeddingModel
from .rag_preprocess_documents import PDFLoader, TextChunker

# Optional components - import only if available
try:
    from .rag_search_and_response import IndexManager, Retriever, LLMResponder
    RAG_COMPONENTS_AVAILABLE = True
except ImportError:
    RAG_COMPONENTS_AVAILABLE = False
    IndexManager = None
    Retriever = None
    LLMResponder = None

try:
    from .rag_evaluation import RAGEvaluator
    RAG_EVALUATION_AVAILABLE = True
except ImportError:
    RAG_EVALUATION_AVAILABLE = False
    RAGEvaluator = None

try:
    from .anthropic_api import AnthropicAPI
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    AnthropicAPI = None

try:
    from .llamaindex_api import LlamaIndexAPI
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False
    LlamaIndexAPI = None

try:
    from .ollama_api import OllamaAPI
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    OllamaAPI = None

try:
    from .multi_tool_agent import MultiToolAgent
    MULTI_TOOL_AVAILABLE = True
except ImportError:
    MULTI_TOOL_AVAILABLE = False
    MultiToolAgent = None

# Build the list of available exports
__all__ = [
    # Version
    "__version__",

    # Core agentic infrastructure (always available)
    "SimpleAgent",
    "create_basic_agent",
    "create_basic_tools",

    # RAG components (always available)
    "AzureOpenAIEmbeddingModel",
    "PDFLoader",
    "TextChunker",
]

# Add optional components if available
if RAG_COMPONENTS_AVAILABLE:
    __all__.extend(["IndexManager", "Retriever", "LLMResponder"])

if RAG_EVALUATION_AVAILABLE:
    __all__.append("RAGEvaluator")

if ANTHROPIC_AVAILABLE:
    __all__.append("AnthropicAPI")

if LLAMAINDEX_AVAILABLE:
    __all__.append("LlamaIndexAPI")

if OLLAMA_AVAILABLE:
    __all__.append("OllamaAPI")

if MULTI_TOOL_AVAILABLE:
    __all__.append("MultiToolAgent")
