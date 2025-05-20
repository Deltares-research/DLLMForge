import pytest
from unittest.mock import patch, MagicMock
import os
from dllmforge.langchain_api import LangchainLLMCreator

@pytest.fixture
def mock_env_vars():
    """Fixture to set up mock environment variables"""
    env_vars = {
        "AZURE_OPENAI_ENDPOINT": "https://test-azure-endpoint",
        "AZURE_OPENAI_API_KEY": "test-azure-key",
        "AZURE_OPENAI_DEPLOYMENT_NAME": "test-deployment",
        "AZURE_OPENAI_API_VERSION": "2023-05-15",
        "OPENAI_API_KEY": "test-openai-key",
        "OPENAI_MODEL_NAME": "gpt-3.5-turbo",
        "MISTRAL_API_KEY": "test-mistral-key",
        "MISTRAL_MODEL_NAME": "mistral-small"
    }
    with patch.dict(os.environ, env_vars):
        yield env_vars

@pytest.fixture
def mock_llm_response():
    """Fixture to create a mock LLM response"""
    mock_response = MagicMock()
    mock_response.content = "I am doing well, thank you for asking!"
    return mock_response

def test_init_azure_openai(mock_env_vars):
    """Test initialization with Azure OpenAI provider"""
    creator = LangchainLLMCreator(model_provider="azure-openai")
    assert creator.model_provider == "azure-openai"
    assert creator.temperature == 0.0

def test_init_openai(mock_env_vars):
    """Test initialization with OpenAI provider"""
    creator = LangchainLLMCreator(model_provider="openai", temperature=0.5)
    assert creator.model_provider == "openai"
    assert creator.temperature == 0.5

def test_init_mistral(mock_env_vars):
    """Test initialization with Mistral provider"""
    creator = LangchainLLMCreator(model_provider="mistral")
    assert creator.model_provider == "mistral"
    assert creator.temperature == 0.0

def test_init_invalid_provider():
    """Test initialization with invalid provider"""
    with pytest.raises(ValueError, match="Unsupported model provider"):
        LangchainLLMCreator(model_provider="invalid")

@patch('dllmforge.langchain_api.AzureChatOpenAI')
def test_send_test_message_azure(mock_azure_chat, mock_env_vars, mock_llm_response):
    """Test sending test message with Azure OpenAI"""
    mock_azure_chat.return_value.invoke.return_value = mock_llm_response
    creator = LangchainLLMCreator(model_provider="azure-openai")
    response = creator.send_test_message("Hello, how are you?")
    assert response == "I am doing well, thank you for asking!"

@patch('dllmforge.langchain_api.ChatOpenAI')
def test_send_test_message_openai(mock_openai_chat, mock_env_vars, mock_llm_response):
    """Test sending test message with OpenAI"""
    mock_openai_chat.return_value.invoke.return_value = mock_llm_response
    creator = LangchainLLMCreator(model_provider="openai")
    response = creator.send_test_message("Hello, how are you?")
    assert response == "I am doing well, thank you for asking!"

@patch('dllmforge.langchain_api.ChatMistralAI')
def test_send_test_message_mistral(mock_mistral_chat, mock_env_vars, mock_llm_response):
    """Test sending test message with Mistral"""
    mock_mistral_chat.return_value.invoke.return_value = mock_llm_response
    creator = LangchainLLMCreator(model_provider="mistral")
    response = creator.send_test_message("Hello, how are you?")
    assert response == "I am doing well, thank you for asking!"

@patch('dllmforge.langchain_api.ChatOpenAI')
def test_send_test_message_error(mock_openai_chat, mock_env_vars):
    """Test error handling in send_test_message"""
    # Set up the mock to raise an exception on invoke
    mock_openai_chat.return_value.invoke.side_effect = Exception("API Error")
    creator = LangchainLLMCreator(model_provider="openai")
    response = creator.send_test_message("Hello")
    assert response is None 