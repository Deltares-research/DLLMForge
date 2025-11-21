import json
from typing import Any, List, Optional

import requests
from langchain.chat_models.base import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel
import urllib3

urllib3.disable_warnings()


class Metadata(BaseModel):
    context_window: int = 4096


class DeltaresOllamaLLM(BaseChatModel):
    """A LangChain wrapper around your own HTTP‐serving chat model."""

    base_url: str  # e.g. "https://api.mycompany.ai"
    model_name: str  # e.g. "my‐finetuned‐gpt"
    metadata: Metadata = Metadata()
    headers: Optional[dict] = None  # e.g. {"Authorization": "Bearer ..."}
    system_message: Optional[
        str] = "You are a helpful assistant that answers questions based on the provided context. As you are thinking reflect on the question."

    def _generate(
        self,
        messages: List[List[SystemMessage | HumanMessage | AIMessage]],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # 1. Build the payload in your server’s expected format
        # messages
        payload = {
            "model": self.model_name,
            "prompt": "".join([message.content for message in messages]),
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.2),
                "max_tokens": kwargs.get("max_tokens", 4096),
            },
        }

        # 2. Call your server
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            verify=False,
        )
        resp.raise_for_status()
        data = resp.text
        if data.startswith("{") or data.startswith("["):
            data = json.loads(data)
        else:
            raise ValueError(f"Unexpected response format: {data}")

        # 3. Build a ChatResult
        generations: List[ChatGeneration] = []
        content = data.get("response", "")
        msg = AIMessage(content=content)
        generations.append(ChatGeneration(message=msg))

        return ChatResult(generations=generations)

    @property
    def _identifying_params(self) -> dict:
        return {"model_name": self.model_name, "base_url": self.base_url}

    @property
    def llm_type(self) -> str:
        return "custom_chat"

    @property
    def _llm_type(self) -> str:
        # this is the string by which LangChain will label/log your model
        return "ollama"

    def ask_with_retriever(self, question: str, retriever, **kwargs) -> str:
        """Ask a question using the retriever to get context."""
        context = retriever.invoke(question)
        prompt = [
            SystemMessage(content=self.system_message),
            HumanMessage(content=f"Question: {question} \nContext: {context}"),
        ]
        chat_result = self._generate(prompt, **kwargs)
        return chat_result

    def chat_completion(
        self,
        messages: List[dict],
        temperature: float = 0.2,
        max_tokens: int = 512,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict:
        """
        Direct chat completion method that accepts OpenAI-style message format.
        Args:
            messages: List of message dictionaries in OpenAI format.
            temperature: Sampling temperature for the model.
            max_tokens: Maximum number of tokens to generate.
            stream: Whether to stream the response.
            **kwargs: Additional parameters for the request.

        Returns:
            Dict with completion response
        """
        has_images = any(msg.get("images") for msg in messages)

        if has_images:
            # For vision models, use the chat API format
            # Convert to Ollama chat format
            ollama_messages = []
            for msg in messages:
                ollama_msg = {"role": msg.get("role", "user"), "content": msg.get("content", "")}
                if msg.get("images"):
                    ollama_msg["images"] = msg["images"]
                ollama_messages.append(ollama_msg)

            payload = {
                "model": self.model_name,
                "messages": ollama_messages,
                "stream": stream,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    **kwargs
                },
            }
            endpoint = f"{self.base_url}/api/chat"
        else:
            # Convert messages to prompt format expected by Ollama
            prompt_parts = []
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")

                if role == "system":
                    prompt_parts.append(f"System: {content}")
                elif role == "user":
                    prompt_parts.append(f"Human: {content}")
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {content}")

            prompt = "\n".join(prompt_parts) + "\nAssistant:"

            # Build payload for Ollama API
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": stream,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    **kwargs
                },
            }
            endpoint = f"{self.base_url}/api/generate"
        headers = self.headers or {}
        # Make request to Ollama API
        resp = requests.post(
            endpoint,
            json=payload,
            headers=headers,
            verify=False,
        )
        resp.raise_for_status()

        # Parse response
        if stream:
            return resp  # Return response object for streaming
        else:
            data = resp.text
            if data.startswith("{") or data.startswith("["):
                data = json.loads(data)
            else:
                raise ValueError(f"Unexpected response format: {data}")
            # Chat API response format
            content = data.get("message", {}).get("content", "")
            if content == "":
                content = data.get("response", "")
            # Return OpenAI-style response format
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": content
                    },
                    "finish_reason": data.get("done_reason", "stop"),
                    "done": data.get("done", True)
                }],
                "model": self.model_name,
                "usage": {
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                    "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
                }
            }

    def get_available_models(self) -> List[str]:
        """
        Fetch available models from the Ollama API.
        
        Returns:
            List of available model names, sorted alphabetically.
            
        Raises:
            requests.RequestException: If the API request fails.
            ValueError: If the response format is unexpected.
        """
        try:
            resp = requests.get(f"{self.base_url}/api/tags", headers=self.headers or {}, verify=False, timeout=10)
            resp.raise_for_status()

            data = resp.json()

            # Extract model names from the response
            models = []
            if "models" in data:
                for model in data["models"]:
                    if "name" in model:
                        models.append(model["name"])

            # Return sorted list of model names
            return sorted(models)

        except requests.RequestException as e:
            raise requests.RequestException(f"Failed to fetch models from {self.base_url}/api/tags: {e}")
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Unexpected response format from models API: {e}")

    def validate_model(self) -> bool:
        """
        Check if the current model_name is available on the server.
        
        Returns:
            True if the model is available, False otherwise.
        """
        try:
            available_models = self.get_available_models()
            return self.model_name in available_models
        except (requests.RequestException, ValueError):
            return False

    @classmethod
    def list_available_models(cls, base_url: str, headers: Optional[dict] = None) -> List[str]:
        """
        Class method to fetch available models without instantiating the class.
        
        Args:
            base_url: The base URL of the Ollama API
            headers: Optional headers for the request
            
        Returns:
            List of available model names, sorted alphabetically.
        """
        try:
            resp = requests.get(f"{base_url}/api/tags", headers=headers or {}, verify=False, timeout=10)
            resp.raise_for_status()

            data = resp.json()

            # Extract model names from the response
            models = []
            if "models" in data:
                for model in data["models"]:
                    if "name" in model:
                        models.append(model["name"])

            # Return sorted list of model names
            return sorted(models)

        except requests.RequestException as e:
            raise requests.RequestException(f"Failed to fetch models from {base_url}/api/tags: {e}")


def get_deltares_models(base_url: str = "https://chat-api.deltares.nl") -> List[str]:
    """
    Convenience function to get available models from Deltares API.

    Args:
        base_url: The base URL of the Deltares API (default: https://chat-api.deltares.nl)

    Returns:
        List of available model names, sorted alphabetically.
    """
    return DeltaresOllamaLLM.list_available_models(base_url)
