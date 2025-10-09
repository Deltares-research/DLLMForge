import json
import pytest
from unittest.mock import patch, MagicMock, Mock
import requests
from langchain.schema import (
    AIMessage,
    ChatGeneration,
    ChatResult,
    HumanMessage,
    SystemMessage,
)

from dllmforge.LLMs.Deltares_LLMs import DeltaresOllamaLLM, Metadata


class TestMetadata:
    """Test cases for the Metadata model."""

    def test_metadata_default_values(self):
        """Test that Metadata has correct default values."""
        metadata = Metadata()
        assert metadata.context_window == 4096

    def test_metadata_custom_values(self):
        """Test that Metadata can be initialized with custom values."""
        metadata = Metadata(context_window=8192)
        assert metadata.context_window == 8192

    def test_metadata_validation(self):
        """Test that Metadata validates input types."""
        # Valid input
        metadata = Metadata(context_window=2048)
        assert isinstance(metadata.context_window, int)

        # Test with string that can be converted to int
        metadata = Metadata(context_window="4096")
        assert metadata.context_window == 4096


class TestDeltaresOllamaLLM:
    """Test cases for the DeltaresOllamaLLM class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.base_url = "http://localhost:11434"
        self.model_name = "test-model"
        self.llm = DeltaresOllamaLLM(base_url=self.base_url, model_name=self.model_name)

    def test_initialization_required_params(self):
        """Test that DeltaresOllamaLLM can be initialized with required parameters."""
        llm = DeltaresOllamaLLM(base_url="http://localhost:11434", model_name="llama2")
        assert llm.base_url == "http://localhost:11434"
        assert llm.model_name == "llama2"
        assert llm.metadata.context_window == 4096
        assert llm.headers is None
        assert "helpful assistant" in llm.system_message

    def test_initialization_all_params(self):
        """Test that DeltaresOllamaLLM can be initialized with all parameters."""
        metadata = Metadata(context_window=8192)
        headers = {"Authorization": "Bearer token123"}
        system_message = "Custom system message"

        llm = DeltaresOllamaLLM(base_url="http://localhost:11434",
                                model_name="llama2",
                                metadata=metadata,
                                headers=headers,
                                system_message=system_message)

        assert llm.base_url == "http://localhost:11434"
        assert llm.model_name == "llama2"
        assert llm.metadata.context_window == 8192
        assert llm.headers == headers
        assert llm.system_message == system_message

    def test_identifying_params(self):
        """Test the _identifying_params property."""
        params = self.llm._identifying_params
        assert params == {"model_name": self.model_name, "base_url": self.base_url}

    def test_llm_type_property(self):
        """Test the llm_type property."""
        assert self.llm.llm_type == "custom_chat"

    def test_internal_llm_type_property(self):
        """Test the _llm_type property."""
        assert self.llm._llm_type == "ollama"

    @patch('requests.post')
    def test_generate_success(self, mock_post):
        """Test successful _generate method call."""
        # Mock response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.text = json.dumps({"response": "Test response"})
        mock_post.return_value = mock_response

        # Test messages
        messages = [SystemMessage(content="You are a helpful assistant"), HumanMessage(content="Hello")]

        # Call _generate
        result = self.llm._generate(messages)

        # Assertions
        assert isinstance(result, ChatResult)
        assert len(result.generations) == 1
        assert isinstance(result.generations[0], ChatGeneration)
        assert isinstance(result.generations[0].message, AIMessage)
        assert result.generations[0].message.content == "Test response"

        # Verify request was made correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]['json']['model'] == self.model_name
        assert call_args[1]['json']['prompt'] == "You are a helpful assistantHello"
        assert call_args[1]['json']['stream'] is False
        assert call_args[0][0] == f"{self.base_url}/api/generate"

    @patch('requests.post')
    def test_generate_with_kwargs(self, mock_post):
        """Test _generate method with additional kwargs."""
        # Mock response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.text = json.dumps({"response": "Test response"})
        mock_post.return_value = mock_response

        # Test messages
        messages = [HumanMessage(content="Hello")]

        # Call _generate with kwargs
        result = self.llm._generate(messages, temperature=0.7, max_tokens=2048)

        # Verify request payload includes kwargs
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        assert payload['options']['temperature'] == 0.7
        assert payload['options']['max_tokens'] == 2048

    @patch('requests.post')
    def test_generate_http_error(self, mock_post):
        """Test _generate method with HTTP error."""
        # Mock response with error
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_post.return_value = mock_response

        messages = [HumanMessage(content="Hello")]

        # Should raise HTTPError
        with pytest.raises(requests.HTTPError):
            self.llm._generate(messages)

    @patch('requests.post')
    def test_ask_with_retriever(self, mock_post):
        """Test ask_with_retriever method."""
        # Mock response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.text = json.dumps({"response": "Test response with context"})
        mock_post.return_value = mock_response

        # Mock retriever
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = "Retrieved context"

        # Call method
        result = self.llm.ask_with_retriever("What is the weather?", mock_retriever)

        # Assertions
        mock_retriever.invoke.assert_called_once_with("What is the weather?")
        assert isinstance(result, ChatResult)
        assert result.generations[0].message.content == "Test response with context"

    @patch('requests.post')
    def test_chat_completion_with_parameters(self, mock_post):
        """Test chat_completion method with custom parameters."""
        # Mock response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.text = json.dumps({"response": "Test response"})
        mock_post.return_value = mock_response

        messages = [{"role": "user", "content": "Hello"}]

        # Call with custom parameters
        result = self.llm.chat_completion(messages, temperature=0.8, max_tokens=1024, top_p=0.9)

        # Verify parameters were passed correctly
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        assert payload['options']['temperature'] == 0.8
        assert payload['options']['num_predict'] == 1024
        assert payload['options']['top_p'] == 0.9

    @patch('requests.post')
    def test_chat_completion_with_headers(self, mock_post):
        """Test chat_completion method with custom headers."""
        # Setup LLM with headers
        headers = {"Authorization": "Bearer token123"}
        llm = DeltaresOllamaLLM(base_url=self.base_url, model_name=self.model_name, headers=headers)

        # Mock response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.text = json.dumps({"response": "Test response"})
        mock_post.return_value = mock_response

        messages = [{"role": "user", "content": "Hello"}]

        # Call method
        result = llm.chat_completion(messages)

        # Verify headers were included
        call_args = mock_post.call_args
        assert call_args[1]['headers'] == headers

    @patch('requests.post')
    def test_chat_completion_streaming(self, mock_post):
        """Test chat_completion method with streaming enabled."""
        # Mock response object for streaming
        mock_response = MagicMock()
        mock_post.return_value = mock_response

        messages = [{"role": "user", "content": "Hello"}]

        # Call with streaming
        result = self.llm.chat_completion(messages, stream=True)

        # Should return the response object for streaming
        assert result == mock_response

        # Verify stream parameter was set
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        assert payload['stream'] is True
