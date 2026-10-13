"""
Extract the final answer and the tool steps from SimpleAgent output (no database knowledge).

SimpleAgent produces two message shapes:

- native tool calling: ``AIMessage(tool_calls=[...])`` followed by ``ToolMessage(name=..., content=...)``
- text routing: ``HumanMessage("Tool '<name>' result: <content>")`` and a final
  ``AIMessage('{"final_answer": "..."}')``
"""
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = ["ToolStep", "extract_answer_and_steps", "parse_json_object"]

_TEXT_RESULT = re.compile(r"^Tool '([^']+)' result:\s?(.*)\Z", re.DOTALL)


@dataclass
class ToolStep:
    """One executed tool call."""

    tool: str
    content: str
    args: Dict[str, Any] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)


def parse_json_object(text: str) -> Optional[dict]:
    """Parse the outermost ``{...}`` in ``text``; tolerate raw newlines inside strings."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    candidate = text[start:end + 1]
    for attempt in (candidate, candidate.replace("\n", "\\n").replace("\r", "\\r")):
        try:
            value = json.loads(attempt)
            return value if isinstance(value, dict) else None
        except ValueError:
            continue
    return None


def _content(message: Any) -> str:
    content = getattr(message, "content", "") or ""
    if isinstance(content, list):  # content blocks
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content)


def extract_answer_and_steps(
    messages: Sequence[Any],
    text_routing: bool,
    step_parsers: Optional[Dict[str, Callable[[str], Dict[str, Any]]]] = None,
) -> Tuple[str, List[ToolStep]]:
    """Return ``(answer, steps)`` from SimpleAgent messages.

    Args:
        messages: The ``messages`` list returned by ``SimpleAgent.app.invoke``.
        text_routing: Whether the agent used text-based tool routing.
        step_parsers: Optional ``{tool_name: parser}``; ``parser(content)`` fills ``ToolStep.details``.
    """
    step_parsers = step_parsers or {}
    call_args: Dict[str, Dict[str, Any]] = {}
    steps: List[ToolStep] = []

    for message in messages:
        kind = getattr(message, "type", None)
        if kind == "ai":
            for call in getattr(message, "tool_calls", None) or []:
                call_args[call.get("id") or ""] = call.get("args") or {}
        elif kind == "tool":
            name = getattr(message, "name", None) or "unknown"
            steps.append(ToolStep(name, _content(message), call_args.get(getattr(message, "tool_call_id", ""), {})))
        elif kind == "human":
            match = _TEXT_RESULT.match(_content(message))
            if match:
                steps.append(ToolStep(match.group(1), match.group(2).strip()))

    for step in steps:
        parser = step_parsers.get(step.tool)
        if parser:
            step.details = parser(step.content)

    answer = ""
    for message in reversed(messages):
        content = _content(message)
        if getattr(message, "type", None) != "ai" or not content.strip():
            continue
        if not text_routing:
            answer = content
            break
        obj = parse_json_object(content)
        if obj and "final_answer" in obj:
            answer = str(obj["final_answer"])
            break
        if obj and "tool" in obj:
            continue
        answer = content
        break
    return answer, steps
