import pytest
from unittest.mock import patch, MagicMock
import os
from dllmforge.llamaindex_api import LlamaIndexAPI


class TestLlamaIndexAPI:

    # skip if the test is run on github
    @pytest.mark.skip(reason="Skip test on GitHub Actions, you can run it locally.")
    def test_local(self):
        """Test the API locally with actual LLM services."""
        # Test Azure OpenAI
        api = LlamaIndexAPI(model_provider="azure-openai")
        assert api.check_server_status() is True, "Azure OpenAI service should be accessible"
        
        # Test OpenAI
        api = LlamaIndexAPI(model_provider="openai")
        assert api.check_server_status() is True, "OpenAI service should be accessible"
        
        # Test Mistral
        api = LlamaIndexAPI(model_provider="mistral")
        assert api.check_server_status() is True, "Mistral service should be accessible"
        
        # Test chat completion with Azure OpenAI
        test_prompt = "Create a simple HTML webpage with a greeting message and a background color of your choice. Give me only the HTML code so I can save it in a file immediately. No other text is needed."
        
        # Create output directory if it doesn't exist
        output_dir = "tests/test_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Test each provider
        providers = ["azure-openai", "openai", "mistral"]
        for provider in providers:
            print(f"\nTesting provider: {provider}")
            api = LlamaIndexAPI(model_provider=provider)
            response = api.send_test_message(prompt=test_prompt)
            assert response is not None, f"Failed to get response from {provider}"
            assert "response" in response, f"Response should contain 'response' field for {provider}"
            
            # Save the HTML response to a file
            output_file = os.path.join(output_dir, f"response_{provider}_llamaindex.html")
            with open(output_file, "w") as f:
                f.write(response["response"])
            print(f"Output saved to: {output_file}")

    def test_check_server_status_azure(self):
        api = LlamaIndexAPI(model_provider="azure-openai", deployment_name="test", api_key="test_key")
        assert api.check_server_status() is False

        # check that all is set up correctly
        assert api.llm is not None
        assert api.llm.engine == "test"
        assert api.llm.api_key == "test_key"
        assert type(api.llm).__name__ == "AzureOpenAI"


    def test_check_server_status_openai(self):
        """Test server status check with mocked OpenAI client."""
    
        api = LlamaIndexAPI(model_provider="openai")
        assert api.check_server_status() is False

        assert api.llm is not None
        assert type(api.llm).__name__ == "OpenAI"


    def test_check_server_status_mistral(self):
        """Test server status check with mocked Mistral client."""
        api = LlamaIndexAPI(model_provider="mistral", api_key="test_key", model_name="test_model")
        assert api.check_server_status() is False

        # check that all is set up correctly
        assert api.llm is not None
        assert type(api.llm).__name__ == "MistralAI"

    @patch("llama_index.llms.azure_openai.AzureOpenAI")
    def test_send_test_message_azure(self, mock_azure):
        """Test sending test message with mocked Azure client."""
        # Mock successful message sending
        mock_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.message = MagicMock(content="Test response")
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        mock_instance.chat.return_value = mock_response
        mock_azure.return_value = mock_instance
        
        # call the mocked API
        api = LlamaIndexAPI(model_provider="azure-openai", deployment_name="test", api_key="test_key")
        api.llm = mock_instance  # Set the mocked instance
        response = api.send_test_message(prompt="Test prompt")
        assert response["response"] == "Test response"
        assert response["model"] == "azure-openai"
        assert response["usage"].prompt_tokens == 10
        assert response["usage"].completion_tokens == 20
        assert response["usage"].total_tokens == 30

        # Mock failed message sending
        mock_instance.chat.side_effect = Exception("API Error")
        assert api.send_test_message(prompt="Test prompt") is None

    @patch("llama_index.llms.openai.OpenAI")
    def test_send_test_message_openai(self, mock_openai):
        """Test sending test message with mocked OpenAI client."""
        # Mock successful message sending
        mock_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.message = MagicMock(content="Test response")
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        mock_instance.chat.return_value = mock_response
        mock_openai.return_value = mock_instance
        
        # call the mocked API
        api = LlamaIndexAPI(model_provider="openai")
        api.llm = mock_instance  # Set the mocked instance
        response = api.send_test_message(prompt="Test prompt")
        assert response["response"] == "Test response"
        assert response["model"] == "openai"
        assert response["usage"].prompt_tokens == 10
        assert response["usage"].completion_tokens == 20
        assert response["usage"].total_tokens == 30

        # Mock failed message sending
        mock_instance.chat.side_effect = Exception("API Error")
        assert api.send_test_message(prompt="Test prompt") is None

    @patch("llama_index.llms.mistralai.MistralAI")
    def test_send_test_message_mistral(self, mock_mistral):
        """Test sending test message with mocked Mistral client."""
        # Mock successful message sending
        mock_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.message = MagicMock(content="Test response")
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        mock_instance.chat.return_value = mock_response
        mock_mistral.return_value = mock_instance
        
        # call the mocked API
        api = LlamaIndexAPI(model_provider="mistral", api_key="test_key", model_name="test_model")
        api.llm = mock_instance  # Set the mocked instance
        response = api.send_test_message(prompt="Test prompt")
        assert response["response"] == "Test response"
        assert response["model"] == "mistral"
        assert response["usage"].prompt_tokens == 10
        assert response["usage"].completion_tokens == 20
        assert response["usage"].total_tokens == 30

        # Mock failed message sending
        mock_instance.chat.side_effect = Exception("API Error")
        assert api.send_test_message(prompt="Test prompt") is None

    @patch("llama_index.llms.azure_openai.AzureOpenAI")
    def test_chat_completion_azure(self, mock_azure):
        """Test chat completion with mocked Azure client."""
        # Mock successful chat completion
        mock_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.message = MagicMock(content="Test response")
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        mock_instance.chat.return_value = mock_response
        mock_azure.return_value = mock_instance
        
        # call the mocked API
        api = LlamaIndexAPI(model_provider="azure-openai", deployment_name="test", api_key="test_key")  
        api.llm = mock_instance
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Test prompt"}
        ]
        response = api.chat_completion(messages)
        assert response["response"] == "Test response"
        assert response["model"] == "azure-openai"
        assert response["usage"].prompt_tokens == 10
        assert response["usage"].completion_tokens == 20
        assert response["usage"].total_tokens == 30

        # Mock failed chat completion
        mock_instance.chat.side_effect = Exception("API Error")
        assert api.chat_completion(messages) is None

    def test_init_invalid_provider(self):
        """Test initialization with invalid provider"""
        with pytest.raises(ValueError, match="Unsupported model provider"):
            LlamaIndexAPI(model_provider="invalid") 