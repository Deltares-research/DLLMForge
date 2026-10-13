"""Offline fake chat models and non-database tools for agent tests (no network, no API keys)."""
from typing import Any, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


class ScriptedChatModel(BaseChatModel):
    """Chat model that replays scripted responses and records every prompt it receives.

    After the script is exhausted the last response is repeated.
    ``bind_tools`` returns the model itself so it works with SimpleAgent's native tool path.
    """

    responses: List[BaseMessage]
    calls: List[List[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def _generate(self,
                  messages: List[BaseMessage],
                  stop: Optional[List[str]] = None,
                  run_manager: Any = None,
                  **kwargs: Any) -> ChatResult:
        self.calls.append(list(messages))
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return ChatResult(generations=[ChatGeneration(message=self.responses[index].model_copy())])

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        return self


def tool_call(name: str, args: dict, call_id: str = "call-1") -> AIMessage:
    """AIMessage requesting a native tool call."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


def text(content: str) -> AIMessage:
    """Plain AIMessage (final answer or text-routing JSON)."""
    return AIMessage(content=content)


def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First number.
        b: Second number.
    """
    return a + b


def echo(text: str) -> str:
    """Echo the given text back.

    Args:
        text: Text to echo.
    """
    return f"Echo: {text}"
