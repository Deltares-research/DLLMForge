"""
One factory for LangChain chat models across the providers used with SimpleAgent.

- ``azure-openai``, ``openai``, ``mistral``: delegated to :class:`dllmforge.langchain_api.LangchainAPI`.
- ``deltares``: :class:`dllmforge.LLMs.Deltares_LLMs.DeltaresOllamaLLM`; base URL from ``DELTARES_LLM_BASE_URL``.
- ``ollama``: ``langchain_ollama.ChatOllama`` (optional dependency, imported lazily).

Providers in :data:`TEXT_ROUTING_PROVIDERS` lack reliable native tool calling and should run
SimpleAgent with ``enable_text_tool_routing=True`` (see :func:`uses_text_routing`).
"""
import os
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel

__all__ = [
    "TEXT_ROUTING_PROVIDERS", "SUPPORTED_PROVIDERS", "DEFAULT_DELTARES_BASE_URL", "uses_text_routing",
    "default_model_name", "create_chat_model"
]

TEXT_ROUTING_PROVIDERS = frozenset({"deltares", "ollama"})
SUPPORTED_PROVIDERS = ("azure-openai", "openai", "mistral", "deltares", "ollama")
DEFAULT_DELTARES_BASE_URL = "https://chat-api.directory.intra"
_DEFAULT_LOCAL_MODEL = "gemma3:latest"


def uses_text_routing(provider: str) -> bool:
    """Whether SimpleAgent should use text-based JSON tool routing for ``provider``."""
    return provider.lower() in TEXT_ROUTING_PROVIDERS


def default_model_name(provider: str) -> Optional[str]:
    """Model or deployment name configured in the environment for ``provider``."""
    provider = provider.lower()
    if provider == "azure-openai":
        return os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    if provider == "openai":
        return os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
    if provider == "mistral":
        return os.getenv("MISTRAL_MODEL_NAME")
    if provider == "deltares":
        return os.getenv("DELTARES_MODEL_NAME", _DEFAULT_LOCAL_MODEL)
    if provider == "ollama":
        return os.getenv("OLLAMA_MODEL", _DEFAULT_LOCAL_MODEL)
    return None


def create_chat_model(provider: str, model_name: Optional[str] = None, temperature: float = 0.1) -> BaseChatModel:
    """Create a chat model.

    Args:
        provider: One of :data:`SUPPORTED_PROVIDERS`.
        model_name: Model or (Azure) deployment name; defaults to :func:`default_model_name`.
        temperature: Sampling temperature.
    """
    provider = provider.lower()
    name = model_name or default_model_name(provider)
    if provider == "deltares":
        from .LLMs.Deltares_LLMs import DeltaresOllamaLLM
        llm: BaseChatModel = DeltaresOllamaLLM(base_url=os.getenv("DELTARES_LLM_BASE_URL", DEFAULT_DELTARES_BASE_URL),
                                               model_name=name)
    elif provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise ImportError("The 'ollama' provider needs langchain-ollama: pip install langchain-ollama") from exc
        llm = ChatOllama(model=name, temperature=temperature)
    elif provider in ("azure-openai", "openai", "mistral"):
        from .langchain_api import LangchainAPI
        if provider == "azure-openai":
            api = LangchainAPI(model_provider=provider, temperature=temperature, deployment_name=name)
        else:
            api = LangchainAPI(model_provider=provider, temperature=temperature, model_name=name)
        llm = api.llm
    else:
        raise ValueError(f"Unknown provider {provider!r}; choose from {', '.join(SUPPORTED_PROVIDERS)}")
    return llm
