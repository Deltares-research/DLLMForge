import requests
import json


class OllamaAPI:
    """Class to interact with the Ollama API through a reverse proxy."""

    def __init__(self, url="https://chat-api.directory.intra", model="qwq:latest"):
        self.url = url
        self.model = model

    def check_server_status(self):
        """Check if the server is up and running."""
        try:
            response = requests.head(self.url, verify=False)
            if response.status_code == 200:
                print("Server is up!")
                return True
            else:
                print("Server is down!")
                return False
        except requests.RequestException as e:
            print(f"Error checking server status: {e}")
            return False

    def list_available_models(self):
        """List available models from the API."""
        try:
            response = requests.get(f"{self.url}/api/tags", verify=False)
            if response.status_code == 200:
                return [model['name'] for model in json.loads((response.text))['models']]
            else:
                print(f"Failed to list models: {response.status_code}")
                return None
        except requests.RequestException as e:
            print(f"Error listing models: {e}")
            return None

    def send_test_message(self, prompt="Hello, how are you?"):
        """Send a test message to the model and get a response."""
        try:
            data = {"model": self.model, "prompt": prompt, "stream": False}
            response = requests.post(f"{self.url}/api/generate", json=data, verify=False)
            if response.status_code == 200:
                return json.loads((response.text))
            else:
                print(f"Failed to send test message: {response.status_code}")
                return None
        except requests.RequestException as e:
            print(f"Error sending test message: {e}")
            return None
