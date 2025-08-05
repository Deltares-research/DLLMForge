import pytest
from unittest.mock import patch, MagicMock
import os
from dllmforge.langchain_api import LangchainAPI


class TestLangchainAPI:

    # skip if the test is run on github
    @pytest.mark.skip(reason="Skip test on GitHub Actions, you can run it locally.")
    def test_local(self):
        """Test the API locally with actual LLM services."""
        # Test Azure OpenAI
        api = LangchainAPI(model_provider="azure-openai")
        assert api.check_server_status() is True, "Azure OpenAI service should be accessible"

        # Test OpenAI
        api = LangchainAPI(model_provider="openai")
        assert api.check_server_status() is True, "OpenAI service should be accessible"

        # Test Mistral
        api = LangchainAPI(model_provider="mistral")
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
            api = LangchainAPI(model_provider=provider)
            response = api.send_test_message(prompt=test_prompt)
            assert response is not None, f"Failed to get response from {provider}"
            assert "response" in response, f"Response should contain 'response' field for {provider}"

            # Save the HTML response to a file
            output_file = os.path.join(output_dir, f"response_{provider}.html")
            with open(output_file, "w") as f:
                f.write(response["response"])
            print(f"Output saved to: {output_file}")

    @patch("dllmforge.langchain_api.AzureChatOpenAI")
    def test_check_server_status_azure(self, mock_azure):
        """Test server status check with mocked Azure client."""
        # Mock successful server status check
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = MagicMock(content="Test response")
        mock_azure.return_value = mock_instance

        api = LangchainAPI(model_provider="azure-openai")
        assert api.check_server_status() is True

        # Mock failed server status check
        mock_instance.invoke.side_effect = Exception("API Error")
        assert api.check_server_status() is False

    @patch("dllmforge.langchain_api.ChatOpenAI")
    def test_check_server_status_openai(self, mock_openai):
        """Test server status check with mocked OpenAI client."""
        # Mock successful server status check
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = MagicMock(content="Test response")
        mock_openai.return_value = mock_instance

        api = LangchainAPI(model_provider="openai")
        assert api.check_server_status() is True

        # Mock failed server status check
        mock_instance.invoke.side_effect = Exception("API Error")
        assert api.check_server_status() is False

    @patch("dllmforge.langchain_api.ChatMistralAI")
    def test_check_server_status_mistral(self, mock_mistral):
        """Test server status check with mocked Mistral client."""
        # Mock successful server status check
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = MagicMock(content="Test response")
        mock_mistral.return_value = mock_instance

        api = LangchainAPI(model_provider="mistral")
        assert api.check_server_status() is True

        # Mock failed server status check
        mock_instance.invoke.side_effect = Exception("API Error")
        assert api.check_server_status() is False

    @patch("dllmforge.langchain_api.AzureChatOpenAI")
    def test_send_test_message_azure(self, mock_azure):
        """Test sending test message with mocked Azure client."""
        # Mock successful message sending
        mock_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Test response"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        mock_instance.invoke.return_value = mock_response
        mock_azure.return_value = mock_instance

        api = LangchainAPI(model_provider="azure-openai")
        response = api.send_test_message(prompt="Test prompt")
        assert response["response"] == "Test response"
        assert response["model"] == "azure-openai"
        assert response["usage"]["prompt_tokens"] == 10
        assert response["usage"]["completion_tokens"] == 20
        assert response["usage"]["total_tokens"] == 30

        # Mock failed message sending
        mock_instance.invoke.side_effect = Exception("API Error")
        assert api.send_test_message(prompt="Test prompt") is None

    @patch("dllmforge.langchain_api.ChatOpenAI")
    def test_send_test_message_openai(self, mock_openai):
        """Test sending test message with mocked OpenAI client."""
        # Mock successful message sending
        mock_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Test response"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        mock_instance.invoke.return_value = mock_response
        mock_openai.return_value = mock_instance

        api = LangchainAPI(model_provider="openai")
        response = api.send_test_message(prompt="Test prompt")
        assert response["response"] == "Test response"
        assert response["model"] == "openai"
        assert response["usage"]["prompt_tokens"] == 10
        assert response["usage"]["completion_tokens"] == 20
        assert response["usage"]["total_tokens"] == 30

        # Mock failed message sending
        mock_instance.invoke.side_effect = Exception("API Error")
        assert api.send_test_message(prompt="Test prompt") is None

    @patch("dllmforge.langchain_api.ChatMistralAI")
    def test_send_test_message_mistral(self, mock_mistral):
        """Test sending test message with mocked Mistral client."""
        # Mock successful message sending
        mock_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Test response"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        mock_instance.invoke.return_value = mock_response
        mock_mistral.return_value = mock_instance

        api = LangchainAPI(model_provider="mistral")
        response = api.send_test_message(prompt="Test prompt")
        assert response["response"] == "Test response"
        assert response["model"] == "mistral"
        assert response["usage"]["prompt_tokens"] == 10
        assert response["usage"]["completion_tokens"] == 20
        assert response["usage"]["total_tokens"] == 30

        # Mock failed message sending
        mock_instance.invoke.side_effect = Exception("API Error")
        assert api.send_test_message(prompt="Test prompt") is None

    @patch("dllmforge.langchain_api.AzureChatOpenAI")
    def test_chat_completion_azure(self, mock_azure):
        """Test chat completion with mocked Azure client."""
        # Mock successful chat completion
        mock_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Test response"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        mock_instance.invoke.return_value = mock_response
        mock_azure.return_value = mock_instance

        api = LangchainAPI(model_provider="azure-openai")
        messages = [("system", "You are a helpful assistant."), ("human", "Test prompt")]
        response = api.chat_completion(messages)
        assert response["response"] == "Test response"
        assert response["model"] == "azure-openai"
        assert response["usage"]["prompt_tokens"] == 10
        assert response["usage"]["completion_tokens"] == 20
        assert response["usage"]["total_tokens"] == 30

        # Mock failed chat completion
        mock_instance.invoke.side_effect = Exception("API Error")
        assert api.chat_completion(messages) is None

    def test_init_invalid_provider(self):
        """Test initialization with invalid provider"""
        with pytest.raises(ValueError, match="Unsupported model provider"):
            LangchainAPI(model_provider="invalid")
