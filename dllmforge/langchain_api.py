"""
Create LLM object and api calls from langchain, including Azure and non-Azure models.
We use openai and mistral models for examples.
An overview of available langchain chat models: https://python.langchain.com/docs/integrations/chat/
"""
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_mistralai import ChatMistralAI


load_dotenv()

class LangchainLLMCreator():
    def __init__(self, model_provider: str = "openai", temperature: float = 0.0):
        """
        Initialize the LLM creator using langchain with specified model provider and temperature.
        
        Args:
            model_provider (str): Provider of model to use. Options are:
                - "azure" (default): Use Azure OpenAI
                - "openai": Use OpenAI
                - "mistral": Use Mistral
            temperature (float): Temperature setting for the model (0.0 to 1.0).
                               Higher values make the output more random/creative.
                               Default is 0.0 for most deterministic output.
        """
        self.model_provider = model_provider.lower()
        self.temperature = temperature
        
        # Initialize the appropriate LLM based on model_provider
        if self.model_provider == "azure-openai":
            self.llm = AzureChatOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                temperature=self.temperature
            )
        elif self.model_provider == "openai":
            self.llm = ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                model=os.getenv("OPENAI_MODEL_NAME"),
                temperature=self.temperature
            )
        elif self.model_provider == "mistral":
            self.llm = ChatMistralAI(
                api_key=os.getenv("MISTRAL_API_KEY"),
                model=os.getenv("MISTRAL_MODEL_NAME"),
                temperature=self.temperature
            )
        else:
            raise ValueError(f"Unsupported model provider: {model_provider}. Choose from 'azure', 'openai', or 'mistral'")


    def send_test_message(self, prompt="Hello, how are you?"):
        """
        Send a simple message to the model using the .invoke() method.
        Args:
            prompt (str): The prompt string to send.
        Returns:
            str: The content of the AI's response.
        """
        try:
            messages = [
                ("system", "You are a helpful assistant."),
                ("human", prompt)
            ]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            print(f"Error sending test message: {e}")
            return None

