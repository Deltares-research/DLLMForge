"""
General helpers for giving tools to an agent. Nothing here is specific to databases; the database tools
(``execute_<source>_sql`` etc.) live in ``database/capabilities.py``.

- :func:`think`: a tool that does nothing except record the model's plan before it acts, so the plan can be
  read afterwards. Copied from BOIRQ.
- :func:`to_langchain_tools`: turns plain Python functions (type hints + docstring) into the tool format
  SimpleAgent expects, so the same functions can also be given, unchanged, to the MCP server.
"""
from typing import Callable, Iterable, List, Optional

from langchain_core.tools import BaseTool, StructuredTool

__all__ = ["think", "to_langchain_tools"]


def think(plan: Optional[str] = None, thought: Optional[str] = None) -> str:
    """Record your plan before acting: which tools you will call, with what inputs, and why.

    Keep it to 2-4 sentences. This tool has no side effects; it only returns your plan.

    Args:
        plan: The plan to record.
        thought: Alias of plan.
    """
    text = plan if plan is not None else thought
    return f"Plan recorded: {text}" if text else "Plan recorded."


def to_langchain_tools(functions: Iterable[Callable]) -> List[BaseTool]:
    """Wrap plain functions (type hints + docstring) as LangChain tools; existing tools pass through."""
    tools: List[BaseTool] = []
    for func in functions:
        if isinstance(func, BaseTool):
            tools.append(func)
            continue
        tools.append(
            StructuredTool.from_function(func=func,
                                         name=func.__name__,
                                         description=(func.__doc__ or func.__name__).strip(),
                                         parse_docstring=False))
    return tools
