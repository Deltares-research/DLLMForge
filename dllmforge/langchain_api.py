"""
Create LLM object and api calls from langchain, including Azure and non-Azure models.
We use openai and mistral models for examples.
An overview of available langchain chat models: https://python.langchain.com/docs/integrations/chat/
"""
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_mistralai import ChatMistralAI


class LangchainAPI:
    """Class to interact with various LLM providers using Langchain."""

    def __init__(self,
                 model_provider: str = "azure-openai",
                 temperature: float = 0.1,
                 api_key=None,
                 api_base=None,
                 api_version=None,
                 deployment_name=None,
                 model_name=None):
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
        """
        # Load environment variables if not provided
        load_dotenv()

        self.model_provider = model_provider.lower()
        self.temperature = temperature

        # Initialize the appropriate LLM based on model_provider
        if self.model_provider == "azure-openai":
            self.llm = AzureChatOpenAI(azure_endpoint=api_base or os.getenv("AZURE_OPENAI_ENDPOINT"),
                                       api_key=api_key or os.getenv("AZURE_OPENAI_API_KEY"),
                                       azure_deployment=deployment_name or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                                       api_version=api_version or os.getenv("AZURE_OPENAI_API_VERSION"),
                                       temperature=self.temperature)
        elif self.model_provider == "openai":
            self.llm = ChatOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"),
                                  model=model_name or os.getenv("OPENAI_MODEL_NAME"),
                                  temperature=self.temperature)
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
            response = self.llm.invoke(messages, temperature=temperature or self.temperature, max_tokens=max_tokens)
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
