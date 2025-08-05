import pytest
from unittest.mock import patch, MagicMock
from dllmforge.anthropic_api import AnthropicAPI
import json
import os


class TestAnthropicAPI:

    # skip if the test is run on github
    @pytest.mark.skip(reason="Skip test on GitHub Actions, you can run it locally.")
    def test_local(self):
        """Test the API locally with actual Anthropic service."""
        api = AnthropicAPI()
        
        # Test server status
        assert api.check_server_status() is True, "Anthropic API service should be accessible"
        
        # Test model listing
        models = api.list_available_models()
        assert models is not None, "No models found"
        assert isinstance(models, list), "Models should be a list"
        assert len(models) > 0, "No models found"
        
        # Verify deployment models are included
        assert api.deployment_claude37 in models, "Claude 3.7 deployment model should be in the list"
        assert api.deployment_claude35 in models, "Claude 3.5 deployment model should be in the list"
        
        # Test chat completion with multiple models
        test_prompt = "Create a simple HTML webpage with a greeting message and a background color of your choice. Give me only the HTML code so I can save it in a file immediately. No other text is needed."
        
        # Create output directory if it doesn't exist
        output_dir = "tests/test_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Test each model
        for model in models:
            print(f"\nTesting model: {model}")
            api.model = model
            response = api.send_test_message(prompt=test_prompt)
            assert response is not None, f"Failed to get response from model {model}"
            assert "response" in response, f"Response should contain 'response' field for model {model}"
            
            # Save the HTML response to a file
            output_file = os.path.join(output_dir, f"response_{model}.html")
            with open(output_file, "w") as f:
                f.write(response["response"])
            print(f"Output saved to: {output_file}")

    @patch("dllmforge.anthropic_api.Anthropic")
    def test_check_server_status(self, mock_client):
        """Test server status check with mocked client."""
        # Mock successful server status check
        mock_instance = MagicMock()
        mock_response = MagicMock()
        mock_instance.messages.create.return_value = mock_response
        mock_client.return_value = mock_instance
        
        api = AnthropicAPI()
        assert api.check_server_status() is True

        # Mock failed server status check
        mock_instance.messages.create.side_effect = Exception("API Error")
        assert api.check_server_status() is False


    @patch("dllmforge.anthropic_api.Anthropic")
    def test_send_test_message(self, mock_client):
        """Test sending test message with mocked client."""
        # Mock successful message sending
        mock_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "Test response"
        mock_response.model = "claude-3-opus-20240229"
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 20
        mock_instance.messages.create.return_value = mock_response
        mock_client.return_value = mock_instance
        
        api = AnthropicAPI()
        response = api.send_test_message(prompt="Test prompt")
        assert response["response"] == "Test response"
        assert response["model"] == "claude-3-opus-20240229"
        assert response["usage"]["input_tokens"] == 10
        assert response["usage"]["output_tokens"] == 20
        assert response["usage"]["total_tokens"] == 30

        # Mock failed message sending
        mock_instance.messages.create.side_effect = Exception("API Error")
        assert api.send_test_message(prompt="Test prompt") is None 