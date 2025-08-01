import os
from dotenv import load_dotenv
from openai import AzureOpenAI
import json

class OpenAIAPI:
    """Class to interact with Azure OpenAI API."""

    def __init__(self, 
                 api_key=None,
                 api_base=None,
                 api_version=None,
                 deployment_name="gpt-4o",
                 embedding_deployment="text-embedding-3-large"):
        """Initialize the OpenAI API client with Azure configuration."""
        # Load environment variables if not provided
        load_dotenv()
        
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.api_base = api_base or os.getenv("AZURE_OPENAI_API_BASE")
        self.api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION")
        self.deployment_name = deployment_name
        self.embedding_deployment = embedding_deployment
        
        # Initialize the Azure OpenAI client
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.api_base
        )

    def check_server_status(self):
        """Check if the Azure OpenAI service is accessible."""
        try:
            # Try to list models as a health check
            response = self.client.models.list()
            if response:
                print("Azure OpenAI service is up!")
                return True
            else:
                print("Azure OpenAI service is down!")
                return False
        except Exception as e:
            print(f"Error checking server status: {e}")
            return False

    def list_available_models(self):
        """List available models from Azure OpenAI."""
        try:
            response = self.client.models.list()
            return [model.id for model in response]
        except Exception as e:
            print(f"Error listing models: {e}")
            return None

    def send_test_message(self, prompt="Hello, how are you?"):
        """Send a test message to the model and get a response."""
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            return {
                "response": response.choices[0].message.content,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
        except Exception as e:
            print(f"Error sending test message: {e}")
            return None

    def get_embeddings(self, text):
        """Get embeddings for the given text using Azure OpenAI."""
        try:
            response = self.client.embeddings.create(
                model=self.embedding_deployment,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            return None

    def chat_completion(self, messages, temperature=0.7, max_tokens=800):
        """Get a chat completion from the model."""
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return {
                "response": response.choices[0].message.content,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
        except Exception as e:
            print(f"Error getting chat completion: {e}")
            return None 