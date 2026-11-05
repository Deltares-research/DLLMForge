"""
Create LLM object and api calls from langchain, including Azure and non-Azure models.
We use openai and mistral models for examples.
An overview of available langchain chat models: https://python.langchain.com/docs/integrations/chat/
"""
import os
from dotenv import load_dotenv
from typing import Any, List, Optional
import requests
import json

try:
    from langchain_openai import AzureChatOpenAI, ChatOpenAI
    LANGCHAIN_OPENAI_AVAILABLE = True
except ImportError:
    LANGCHAIN_OPENAI_AVAILABLE = False
    AzureChatOpenAI = None
    ChatOpenAI = None

try:
    from langchain_mistralai import ChatMistralAI
    LANGCHAIN_MISTRALAI_AVAILABLE = True
except ImportError:
    LANGCHAIN_MISTRALAI_AVAILABLE = False
    ChatMistralAI = None

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.outputs import LLMResult

# GPT-5 / o-series reject max_tokens (including JSON null) and custom temperature.
REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def is_reasoning_model(name: Optional[str]) -> bool:
    """Return True for OpenAI/Azure names that reject max_tokens and temperature."""
    if not name:
        return False
    n = name.strip().lower()
    return any(n.startswith(prefix) for prefix in REASONING_MODEL_PREFIXES)


class LangchainAPI:
    """Class to interact with various LLM providers using Langchain."""

    def __init__(self,
                 model_provider: str = "azure-openai",
                 temperature: float = 0.1,
                 api_key=None,
                 api_base=None,
                 api_version=None,
                 deployment_name=None,
                 model_name=None,
                 reasoning_model=None):
        """
        Initialize the Langchain API client with specified configuration.

        Args:
            model_provider (str): Provider of model to use. Options are:
                - "azure-openai": Use Azure OpenAI
                - "openai": Use OpenAI
                - "mistral": Use Mistral
            temperature (float): Temperature setting for the model (0.0 to 1.0)
            api_key (str): API key for the provider
            api_base (str): API base URL (for Azure)
            api_version (str): API version (for Azure)
            deployment_name (str): Deployment name (for Azure)
            model_name (str): Model name (for OpenAI/Mistral)
            reasoning_model (bool): Force reasoning-model request params. None infers
                from the Azure deployment or OpenAI model name (gpt-5 / o1 / o3 / o4).
        """
        # Load environment variables if not provided
        load_dotenv()

        self.model_provider = model_provider.lower()
        self.temperature = temperature
        self.deployment_name = deployment_name or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        self.model_name = model_name or os.getenv("OPENAI_MODEL_NAME") or os.getenv("AZURE_OPENAI_MODEL_NAME")
        if reasoning_model is None:
            self.reasoning_model = is_reasoning_model(self.deployment_name) or is_reasoning_model(self.model_name)
        else:
            self.reasoning_model = bool(reasoning_model)

        # Initialize the appropriate LLM based on model_provider
        if self.model_provider == "azure-openai":
            azure_kwargs = {
                "azure_endpoint": api_base or os.getenv("AZURE_OPENAI_ENDPOINT"),
                "api_key": api_key or os.getenv("AZURE_OPENAI_API_KEY"),
                "azure_deployment": self.deployment_name,
                "api_version": api_version or os.getenv("AZURE_OPENAI_API_VERSION"),
            }
            if not self.reasoning_model:
                azure_kwargs["temperature"] = self.temperature
            self.llm = AzureChatOpenAI(**azure_kwargs)
        elif self.model_provider == "openai":
            openai_kwargs = {
                "api_key": api_key or os.getenv("OPENAI_API_KEY"),
                "model": self.model_name,
            }
            if not self.reasoning_model:
                openai_kwargs["temperature"] = self.temperature
            self.llm = ChatOpenAI(**openai_kwargs)
        elif self.model_provider == "mistral":
            self.llm = ChatMistralAI(api_key=api_key or os.getenv("MISTRAL_API_KEY"),
                                     model=model_name or os.getenv("MISTRAL_MODEL_NAME"),
                                     temperature=self.temperature)
        else:
            raise ValueError(
                f"Unsupported model provider: {model_provider}. Choose from 'azure-openai', 'openai', or 'mistral'")

    def check_server_status(self):
        """Check if the LLM service is accessible."""
        try:
            response = self.send_test_message("Hello")
            if response:
                print(f"{self.model_provider} service is up!")
                return True
            else:
                print(f"{self.model_provider} service is down!")
                return False
        except Exception as e:
            print(f"Error checking server status: {e}")
            return False

    def send_test_message(self, prompt="Hello, how are you?"):
        """
        Send a test message to the model and get a response.

        Args:
            prompt (str): The prompt string to send.

        Returns:
            dict: Dictionary containing the response and metadata.
        """
        try:
            messages = [("system", "You are a helpful assistant."), ("human", prompt)]
            response = self.llm.invoke(messages)
            return {
                "response": response.content,
                "model": self.model_provider,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if hasattr(response, 'usage') else None,
                    "completion_tokens": response.usage.completion_tokens if hasattr(response, 'usage') else None,
                    "total_tokens": response.usage.total_tokens if hasattr(response, 'usage') else None
                }
            }
        except Exception as e:
            print(f"Error sending test message: {e}")
            return None

    def _completion_invoke_kwargs(self, temperature=None, max_tokens=None):
        """Build invoke kwargs, omitting values Azure/OpenAI reasoning models reject."""
        invoke_kwargs = {}
        if max_tokens is not None:
            token_key = "max_completion_tokens" if self.reasoning_model else "max_tokens"
            invoke_kwargs[token_key] = max_tokens
        if not self.reasoning_model:
            invoke_kwargs["temperature"] = self.temperature if temperature is None else temperature
        return invoke_kwargs

    def chat_completion(self, messages, temperature=None, max_tokens=None):
        """
        Get a chat completion from the model.

        Args:
            messages (list): List of message tuples (role, content)
            temperature (float): Optional temperature override
            max_tokens (int): Optional max tokens override
        Returns:
            dict: Dictionary containing the response and metadata.
        """
        try:
            invoke_kwargs = self._completion_invoke_kwargs(temperature=temperature, max_tokens=max_tokens)
            response = self.llm.invoke(messages, **invoke_kwargs)
            return {
                "response": response.content,
                "model": self.model_provider,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if hasattr(response, 'usage') else None,
                    "completion_tokens": response.usage.completion_tokens if hasattr(response, 'usage') else None,
                    "total_tokens": response.usage.total_tokens if hasattr(response, 'usage') else None
                }
            }
        except Exception as e:
            print(f"Error getting chat completion: {e}")
            return None

    def ask_with_retriever(self, question: str, retriever):
        """
        Ask a question using the retriever to get context.

        Args:
            question (str): The question to ask.
            retriever: A rag retriever object that can retrieve relevant context.
            **kwargs: Additional keyword arguments to pass to the LLM (e.g., temperature, max_tokens).

        Returns:
            The response from the LLM.
        """
        context = retriever.invoke(question)
        prompt = [
            SystemMessage(
                content=
                "You are a helpful assistant that answers questions based on the provided context. As you are thinking reflect on the question."
            ),
            HumanMessage(content=f"Question: {question} \nContext: {context}"),
        ]
        response = self.llm(prompt)
        return response.content.strip()
