import json
from typing import Any, List, Optional

import requests
from langchain.chat_models.base import BaseChatModel
from langchain.schema import (
    AIMessage,
    ChatGeneration,
    ChatResult,
    HumanMessage,
    SystemMessage,
)
from pydantic import BaseModel


class Metadata(BaseModel):
    context_window: int = 4096


class DeltaresOllamaLLM(BaseChatModel):
    """A LangChain wrapper around your own HTTP‐serving chat model."""

    base_url: str  # e.g. "https://api.mycompany.ai"
    model_name: str  # e.g. "my‐finetuned‐gpt"
    metadata: Metadata = Metadata()
    headers: Optional[dict] = None  # e.g. {"Authorization": "Bearer ..."}
    system_message: Optional[str] = "You are a helpful assistant that answers questions based on the provided context."

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
                "max_tokens": kwargs.get("max_tokens", 512),
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
    def _identifying_params(self) -> dict:
        return {"model_name": self.model_name}

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
            SystemMessage(content="Please provide a concise answer.")
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
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional parameters
        
        Returns:
            Dict with completion response
        """
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
        
        # Add headers if provided
        headers = self.headers or {}
        
        # Make request to Ollama API
        resp = requests.post(
            f"{self.base_url}/api/generate",
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
            
            # Return OpenAI-style response format
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": data.get("response", "")
                    },
                    "finish_reason": "stop"
                }],
                "model": self.model_name,
                "usage": {
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                    "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
                }
            }
    