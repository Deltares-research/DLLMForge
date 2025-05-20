"""
Create LLM object and API calls using llama_index, including Azure and non-Azure models.
We use openai and mistral models for examples.
An overview of available llama_index LLMs: https://docs.llamaindex.ai/en/stable/module_guides/models/llms/modules/ 
"""
import os
from dotenv import load_dotenv

# LlamaIndex LLM imports
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.llms.openai import OpenAI
from llama_index.llms.mistralai import MistralAI
from llama_index.core.llms import ChatMessage


class LlamaIndexAPI:
    """Class to interact with various LLM providers using LlamaIndex."""

    def __init__(self,
                 model_provider: str = "azure-openai",
                 temperature: float = 0.0,
                 api_key=None,
                 api_base=None,
                 api_version=None,
                 deployment_name=None,
                 model_name=None):
        """
        Initialize the LlamaIndex API client with specified configuration.
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
        """
        load_dotenv()
        self.model_provider = model_provider.lower()
        self.temperature = temperature

        if self.model_provider == "azure-openai":
            self.llm = AzureOpenAI(
                engine=deployment_name or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                model=model_name or os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-35-turbo"),
                api_key=api_key or os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=api_base or os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version=api_version or os.getenv("AZURE_OPENAI_API_VERSION"),
                temperature=self.temperature
            )
        elif self.model_provider == "openai":
            self.llm = OpenAI(
                api_key=api_key or os.getenv("OPENAI_API_KEY"),
                model=model_name or os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo"),
                temperature=self.temperature
            )
        elif self.model_provider == "mistral":
            self.llm = MistralAI(
                api_key=api_key or os.getenv("MISTRAL_API_KEY"),
                model=model_name or os.getenv("MISTRAL_MODEL_NAME", "mistral-medium"),
                temperature=self.temperature
            )
        else:
            raise ValueError(f"Unsupported model provider: {model_provider}. Choose from 'azure-openai', 'openai', or 'mistral'")

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
            messages = [
                ChatMessage(role="system", content="You are a helpful assistant."),
                ChatMessage(role="user", content=prompt)
            ]
            response = self.llm.chat(messages)
            return {
                "response": response.message.content if hasattr(response, 'message') else str(response),
                "model": self.model_provider,
                "usage": getattr(response, 'usage', None)
            }
        except Exception as e:
            print(f"Error sending test message: {e}")
            return None

    def chat_completion(self, messages, temperature=None, max_tokens=None):
        """
        Get a chat completion from the model.
        Args:
            messages (list): List of message dicts or tuples (role, content)
            temperature (float): Optional temperature override
            max_tokens (int): Optional max tokens override
        Returns:
            dict: Dictionary containing the response and metadata.
        """
        try:
            # Convert messages to ChatMessage objects if needed
            chat_messages = []
            for m in messages:
                if isinstance(m, ChatMessage):
                    chat_messages.append(m)
                elif isinstance(m, dict):
                    chat_messages.append(ChatMessage(role=m["role"], content=m["content"]))
                elif isinstance(m, (list, tuple)) and len(m) == 2:
                    chat_messages.append(ChatMessage(role=m[0], content=m[1]))
                else:
                    raise ValueError("Invalid message format")
            response = self.llm.chat(
                chat_messages,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens
            )
            return {
                "response": response.message.content if hasattr(response, 'message') else str(response),
                "model": self.model_provider,
                "usage": getattr(response, 'usage', None)
            }
        except Exception as e:
            print(f"Error getting chat completion: {e}")
            return None 