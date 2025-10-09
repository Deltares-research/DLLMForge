import json
import pytest
import os
from unittest.mock import patch, MagicMock
import requests
from langchain.schema import (AIMessage, ChatResult, HumanMessage, SystemMessage)
import base64

from dllmforge.LLMs.Deltares_LLMs import DeltaresOllamaLLM, get_deltares_models

#  see if we are running in GitHub Actions
IN_GITHUB_ACTIONS = os.getenv('SKIP_INTEGRATION_TESTS', 'false').lower() == 'true'


class TestDeltaresOllamaLLMIntegration:
    """Integration and edge case tests for DeltaresOllamaLLM."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.base_url = "https://chat-api.directory.intra"
        self.model_name = "test-model"
        self.llm = DeltaresOllamaLLM(base_url=self.base_url, model_name=self.model_name)

    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
    def test_integration_generate(self):
        """Integration test with real Ollama server (skipped by default)."""
        # This test would require a real Ollama server running
        llm = DeltaresOllamaLLM(
            base_url="https://chat-api.directory.intra",
            model_name="qwen3:latest"  # Assumes qwen3 is available
        )

        messages = [HumanMessage(content="Hello, how are you?")]

        result = llm._generate(messages)
        assert isinstance(result, ChatResult)
        assert len(result.generations) > 0
        assert isinstance(result.generations[0].message, AIMessage)
        assert len(result.generations[0].message.content) > 0

    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
    def test_integration_chat_completion(self):
        """Integration test with real Ollama server (skipped by default)."""
        # This test would require a real Ollama server running
        llm = DeltaresOllamaLLM(
            base_url="https://chat-api.directory.intra",
            model_name="llama3.1:70b"  # Assumes llama3.1:70b is available
        )

        messages = [{
            "role": "system",
            "content": "You are a helpful assistant"
        }, {
            "role": "user",
            "content": "Hello, how are you?"
        }]

        result = llm.chat_completion(messages, temperature=0.7, max_tokens=500)
        assert result['model'] == "llama3.1:70b"
        assert 'choices' in result
        assert len(result['choices']) > 0
        assert 'message' in result['choices'][0]
        assert 'content' in result['choices'][0]['message']
        assert len(result['choices'][0]['message']['content']) > 0

    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
    def test_get_available_models(self):
        """Test fetching available models from the Ollama server."""
        llm = DeltaresOllamaLLM(base_url="https://chat-api.directory.intra", model_name="test-model")

        models = llm.get_available_models()
        assert isinstance(models, list)
        assert len(models) > 0
        assert all(isinstance(model, str) for model in models)

    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
    def test_get_available_models_wrong_url(self):
        """Test fetching models from an incorrect URL."""
        llm = DeltaresOllamaLLM(
            base_url="http://localhost:9999",  # Assuming nothing is running here
            model_name="test-model")

        with pytest.raises(requests.exceptions.RequestException):
            llm.get_available_models()

    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
    def test_validate_model(self):
        """Test model validation against available models."""
        llm = DeltaresOllamaLLM(
            base_url="https://chat-api.directory.intra",
            model_name="qwen3:latest"  # Assumes qwen3 is available
        )

        # Should not raise an error
        assert llm.validate_model()

        llm_invalid = DeltaresOllamaLLM(base_url="https://chat-api.directory.intra", model_name="nonexistent-model")
        assert not llm_invalid.validate_model()

    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
    def test_integration_get_deltares_models(self):
        """Test the get_deltares_models utility function."""
        models = get_deltares_models(base_url="https://chat-api.directory.intra")
        assert isinstance(models, list)
        assert len(models) > 0
        assert all(isinstance(model, str) for model in models)

    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
    def test_integration_vision_model_with_image(self):
        """Integration test with vision model and image input."""
        llm = DeltaresOllamaLLM(base_url="https://chat-api.directory.intra", model_name="llava-phi3:latest")

        # Load and encode image as base64
        image_path = "tests/test_input/cat.jpg"  # Replace with actual image path
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

        messages = [{
            "role": "user",
            "content": "What do you see in this image?",
            "images": [image_base64]  # Move the image data to the images field
        }]

        result = llm.chat_completion(messages, temperature=0.7, max_tokens=500)
        assert result['model'] == "llava-phi3:latest"
        assert 'choices' in result
        assert len(result['choices']) > 0
        assert 'message' in result['choices'][0]
        assert 'content' in result['choices'][0]['message']
        assert len(result['choices'][0]['message']['content']) > 0
        assert "cat" in result['choices'][0]['message']['content'].lower()
