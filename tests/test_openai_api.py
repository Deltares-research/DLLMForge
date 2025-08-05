import pytest
from unittest.mock import patch, MagicMock
from dllmforge.openai_api import OpenAIAPI
import json
import os


class TestOpenAIAPI:

    # skip if the test is run on github
    @pytest.mark.skip(reason="Skip test on GitHub Actions, you can run it locally.")
    def test_local(self):
        """Test the API locally with actual Azure OpenAI service."""
        api = OpenAIAPI()

        # Test server status
        assert api.check_server_status() is True, "Azure OpenAI service should be accessible"

        # Test model listing
        models = api.list_available_models()
        assert models is not None, "No models found. Check if the API is running."
        assert isinstance(models, list), "Models should be a list."
        assert len(models) > 0, "No models found. Check if the API is running."

        # Test chat completion with multiple models
        test_prompt = "Create a simple HTML webpage with a greeting message and a background color of your choice. Give me only the HTML code so I can save it in a file immediately. No other text is needed."

        # Create output directory if it doesn't exist
        output_dir = "tests/test_output"
        os.makedirs(output_dir, exist_ok=True)

        # Test each model
        for model in models:
            print(f"\nTesting model: {model}")
            api.deployment_name = model
            response = api.send_test_message(prompt=test_prompt)
            assert response is not None, f"Failed to get response from model {model}"
            assert "response" in response, f"Response should contain 'response' field for model {model}"

            # Save the HTML response to a file
            output_file = os.path.join(output_dir, f"response_{model}.html")
            with open(output_file, "w") as f:
                f.write(response["response"])
            print(f"Output saved to: {output_file}")

        # Test embeddings
        test_text = "This is a test text for embeddings"
        embeddings = api.get_embeddings(test_text)
        assert embeddings is not None, "Failed to get embeddings"
        assert isinstance(embeddings, list), "Embeddings should be a list"
        assert len(embeddings) > 0, "Embeddings list should not be empty"

    @patch("dllmforge.openai_api.AzureOpenAI")
    def test_check_server_status(self, mock_client):
        """Test server status check with mocked client."""
        # Mock successful server status check
        mock_instance = MagicMock()
        mock_instance.models.list.return_value = ["model1", "model2"]
        mock_client.return_value = mock_instance

        api = OpenAIAPI()
        assert api.check_server_status() is True

        # Mock failed server status check
        mock_instance.models.list.side_effect = Exception("API Error")
        assert api.check_server_status() is False

    @patch("dllmforge.openai_api.AzureOpenAI")
    def test_list_available_models(self, mock_client):
        """Test model listing with mocked client."""
        # Mock successful model listing
        mock_instance = MagicMock()
        mock_model1 = MagicMock()
        mock_model1.id = "model1"
        mock_model2 = MagicMock()
        mock_model2.id = "model2"
        mock_instance.models.list.return_value = [mock_model1, mock_model2]
        mock_client.return_value = mock_instance

        api = OpenAIAPI()
        models = api.list_available_models()
        assert models == ["model1", "model2"]

        # Mock failed model listing
        mock_instance.models.list.side_effect = Exception("API Error")
        assert api.list_available_models() is None

    @patch("dllmforge.openai_api.AzureOpenAI")
    def test_send_test_message(self, mock_client):
        """Test sending test message with mocked client."""
        # Mock successful message sending
        mock_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.model = "gpt-4"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        mock_instance.chat.completions.create.return_value = mock_response
        mock_client.return_value = mock_instance

        api = OpenAIAPI()
        response = api.send_test_message(prompt="Test prompt")
        assert response["response"] == "Test response"
        assert response["model"] == "gpt-4"
        assert response["usage"]["prompt_tokens"] == 10
        assert response["usage"]["completion_tokens"] == 20
        assert response["usage"]["total_tokens"] == 30

        # Mock failed message sending
        mock_instance.chat.completions.create.side_effect = Exception("API Error")
        assert api.send_test_message(prompt="Test prompt") is None

    @patch("dllmforge.openai_api.AzureOpenAI")
    def test_get_embeddings(self, mock_client):
        """Test getting embeddings with mocked client."""
        # Mock successful embeddings request
        mock_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock()]
        mock_response.data[0].embedding = [0.1, 0.2, 0.3]
        mock_instance.embeddings.create.return_value = mock_response
        mock_client.return_value = mock_instance

        api = OpenAIAPI()
        embeddings = api.get_embeddings("Test text")
        assert embeddings == [0.1, 0.2, 0.3]

        # Mock failed embeddings request
        mock_instance.embeddings.create.side_effect = Exception("API Error")
        assert api.get_embeddings("Test text") is None

    @patch("dllmforge.openai_api.AzureOpenAI")
    def test_chat_completion(self, mock_client):
        """Test chat completion with mocked client."""
        # Mock successful chat completion
        mock_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.model = "gpt-4"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        mock_instance.chat.completions.create.return_value = mock_response
        mock_client.return_value = mock_instance

        api = OpenAIAPI()
        messages = [{
            "role": "system",
            "content": "You are a helpful assistant."
        }, {
            "role": "user",
            "content": "Test prompt"
        }]
        response = api.chat_completion(messages)
        assert response["response"] == "Test response"
        assert response["model"] == "gpt-4"
        assert response["usage"]["prompt_tokens"] == 10
        assert response["usage"]["completion_tokens"] == 20
        assert response["usage"]["total_tokens"] == 30

        # Mock failed chat completion
        mock_instance.chat.completions.create.side_effect = Exception("API Error")
        result = api.chat_completion(messages)
        print(f"Returned value from chat_completion: {result}")
        assert result is None
