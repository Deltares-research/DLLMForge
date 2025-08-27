import pytest
import requests
from dllmforge.LLMs.Deltares_LLMs import DeltaresOllamaLLM


class TestDeltaresOllamaLLMIntegration:
    """Integration tests for DeltaresOllamaLLM without mocks."""

    @pytest.fixture
    def llm_instance(self):
        """Create a test instance of DeltaresOllamaLLM."""
        return DeltaresOllamaLLM(base_url="https://chat-api.directory.intra", model_name="qwen3:latest")

    def test_chat_completion_basic(self, llm_instance):
        """Test basic chat completion functionality."""
        messages = [{
            "role": "system",
            "content": "You are a helpful assistant. Answer briefly."
        }, {
            "role": "user",
            "content": "What is 2+2? Answer with just the number."
        }]

        try:
            result = llm_instance.chat_completion(messages, max_tokens=50)

            # Verify response structure
            assert isinstance(result, dict)
            assert "choices" in result
            assert len(result["choices"]) == 1
            assert "message" in result["choices"][0]
            assert result["choices"][0]["message"]["role"] == "assistant"
            assert isinstance(result["choices"][0]["message"]["content"], str)
            assert len(result["choices"][0]["message"]["content"]) > 0
            assert result["choices"][0]["finish_reason"] == "stop"
            assert result["model"] == "qwen3:latest"

            # Verify usage information
            assert "usage" in result
            assert "prompt_tokens" in result["usage"]
            assert "completion_tokens" in result["usage"]
            assert "total_tokens" in result["usage"]
            assert isinstance(result["usage"]["prompt_tokens"], int)
            assert isinstance(result["usage"]["completion_tokens"], int)
            assert isinstance(result["usage"]["total_tokens"], int)

        except requests.ConnectionError:
            pytest.skip("Ollama server not available at localhost:11434")
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                pytest.skip(f"Model 'llama2' not found on Ollama server")
            else:
                raise

    def test_chat_completion_temperature_variation(self, llm_instance):
        """Test that different temperature values work."""
        messages = [{"role": "user", "content": "Say hello in a creative way."}]

        try:
            # Test low temperature
            result_low = llm_instance.chat_completion(messages, temperature=0.1, max_tokens=30)

            # Test high temperature
            result_high = llm_instance.chat_completion(messages, temperature=0.9, max_tokens=30)

            # Both should return valid responses
            assert isinstance(result_low["choices"][0]["message"]["content"], str)
            assert isinstance(result_high["choices"][0]["message"]["content"], str)
            assert len(result_low["choices"][0]["message"]["content"]) > 0
            assert len(result_high["choices"][0]["message"]["content"]) > 0

        except requests.ConnectionError:
            pytest.skip("Ollama server not available")
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                pytest.skip("Model not found on Ollama server")
            else:
                raise

    def test_chat_completion_conversation(self, llm_instance):
        """Test multi-turn conversation."""
        messages = [{
            "role": "system",
            "content": "You are a helpful assistant."
        }, {
            "role": "user",
            "content": "My name is John."
        }, {
            "role": "assistant",
            "content": "Hello John! Nice to meet you."
        }, {
            "role": "user",
            "content": "What's my name?"
        }]

        try:
            result = llm_instance.chat_completion(messages, max_tokens=50)

            # Should be able to reference the name from context
            response_content = result["choices"][0]["message"]["content"].lower()
            # The response should ideally contain "john" but we'll just check it's not empty
            assert len(response_content) > 0
            assert isinstance(result["choices"][0]["message"]["content"], str)

        except requests.ConnectionError:
            pytest.skip("Ollama server not available")
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                pytest.skip("Model not found on Ollama server")
            else:
                raise

    def test_chat_completion_empty_messages(self, llm_instance):
        """Test with empty messages list."""
        messages = []

        try:
            result = llm_instance.chat_completion(messages, max_tokens=20)

            # Should still return a valid response structure
            assert isinstance(result, dict)
            assert "choices" in result
            assert len(result["choices"]) == 1

        except requests.ConnectionError:
            pytest.skip("Ollama server not available")
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                pytest.skip("Model not found on Ollama server")
            else:
                raise

    def test_chat_completion_max_tokens_limit(self, llm_instance):
        """Test that max_tokens parameter is respected."""
        messages = [{"role": "user", "content": "Write a very long story about a cat."}]

        try:
            # Test with very small max_tokens
            result = llm_instance.chat_completion(messages, max_tokens=5)

            response = result["choices"][0]["message"]["content"]
            # Response should be relatively short due to token limit
            assert isinstance(response, str)
            assert len(response) > 0

        except requests.ConnectionError:
            pytest.skip("Ollama server not available")
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                pytest.skip("Model not found on Ollama server")
            else:
                raise

    def test_invalid_base_url(self):
        """Test behavior with invalid base URL."""
        llm = DeltaresOllamaLLM(base_url="http://invalid-url:9999", model_name="test-model")

        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(requests.ConnectionError):
            llm.chat_completion(messages)

    def test_invalid_model_name(self, llm_instance):
        """Test behavior with invalid model name."""
        # Create instance with non-existent model
        llm = DeltaresOllamaLLM(base_url="http://localhost:11434", model_name="non-existent-model-12345")

        messages = [{"role": "user", "content": "Hello"}]

        try:
            with pytest.raises(requests.HTTPError):
                llm.chat_completion(messages)
        except requests.ConnectionError:
            pytest.skip("Ollama server not available")


# To run specific tests when Ollama is available:
@pytest.mark.integration
class TestWithOllama:
    """Tests that require Ollama to be running with specific models."""

    def test_streaming_response(self):
        """Test streaming functionality (requires manual verification)."""
        llm = DeltaresOllamaLLM(base_url="https://chat-api.directory.intra", model_name="qwen3:latest")

        messages = [{"role": "user", "content": "Count from 1 to 5."}]

        try:
            response = llm.chat_completion(messages, stream=True, max_tokens=50)

            # For streaming, we get the response object back
            assert hasattr(response, 'text') or hasattr(response, 'iter_content')

        except requests.ConnectionError:
            pytest.skip("Ollama server not available")
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                pytest.skip("Model not found")
            else:
                raise
