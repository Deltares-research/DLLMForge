import os
from dotenv import load_dotenv
from anthropic import Anthropic
import json

class AnthropicAPI:
    """Class to interact with Anthropic's Claude API."""

    def __init__(self, 
                 api_key=None,
                 model="claude-3-7-sonnet-20250219",  # Changed default to your available model
                 deployment_claude37=None,
                 deployment_claude35=None):
        """Initialize the Anthropic API client with configuration."""
        # Load environment variables if not provided
        load_dotenv()
        
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.deployment_claude37 = deployment_claude37 or os.getenv("ANTHROPIC_DEPLOYMENT_CLAUDE37")
        self.deployment_claude35 = deployment_claude35 or os.getenv("ANTHROPIC_DEPLOYMENT_CLAUDE35")
        
        # Initialize the Anthropic client
        self.client = Anthropic(api_key=self.api_key)

    def check_server_status(self):
        """Check if the Anthropic API service is accessible."""
        try:
            # Try to send a simple message as a health check
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1,
                messages=[{"role": "user", "content": "Hello"}]
            )
            if response:
                print("Anthropic API service is up!")
                return True
            else:
                print("Anthropic API service is down!")
                return False
        except Exception as e:
            print(f"Error checking server status: {e}")
            return False

    def list_available_models(self):
        """List available models from Anthropic."""
        # Only include the deployment models that are available
        models = []
        
        if self.deployment_claude37:
            models.append(self.deployment_claude37)
        if self.deployment_claude35:
            models.append(self.deployment_claude35)
            
        return models

    def send_test_message(self, prompt="Hello, how are you?"):
        """Send a test message to the model and get a response."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            return {
                "response": response.content[0].text,
                "model": response.model,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                }
            }
        except Exception as e:
            print(f"Error sending test message: {e}")
            return None

    def chat_completion(self, messages, temperature=0.7, max_tokens=1000):
        """Get a chat completion from the model."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages
            )
            return {
                "response": response.content[0].text,
                "model": response.model,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                }
            }
        except Exception as e:
            print(f"Error getting chat completion: {e}")
            return None 