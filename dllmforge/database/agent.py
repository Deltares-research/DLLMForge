"""
Native database agent: configures DLLMForge's :class:`~dllmforge.agent_core.SimpleAgent` with the
database capabilities and instructions of a :class:`~dllmforge.database.project.DatabaseProject`.

The agent loop itself is SimpleAgent's; this module only wires the database pieces into it.
"""
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage

from ..agent_core import SimpleAgent
from ..agent_messages import ToolStep, extract_answer_and_steps
from ..agent_tools import think, to_langchain_tools
from ..llm_factory import create_chat_model, uses_text_routing
from .capabilities import DatabaseCapabilities, split_executed_sql
from .context import build_database_instructions
from .project import DatabaseProject

__all__ = ["DatabaseAgent"]

History = Sequence[Tuple[str, str]]


class DatabaseAgent:
    """Answer natural-language questions about a database project with SimpleAgent.

    Args:
        project: The database project.
        model_provider: LLM provider; defaults to ``<ENV_PREFIX>_MODEL_PROVIDER`` or ``azure-openai``.
        model_name: Model/deployment; defaults to ``<ENV_PREFIX>_MODEL_NAME`` or the provider default.
        temperature: Sampling temperature.
        max_tool_iterations: Tool-call budget per question in text-routing mode.
        llm: Use this chat model instead of creating one (``model_provider`` still selects the routing mode).
        text_routing: Override the routing mode derived from the provider.
    """

    def __init__(self,
                 project: DatabaseProject,
                 model_provider: Optional[str] = None,
                 model_name: Optional[str] = None,
                 temperature: float = 0.1,
                 max_tool_iterations: int = 8,
                 llm: Optional[BaseChatModel] = None,
                 text_routing: Optional[bool] = None):
        self.project = project
        prefix = project.env_prefix
        self.model_provider = (model_provider or os.getenv(f"{prefix}_MODEL_PROVIDER") or "azure-openai").lower()
        self.model_name = model_name or os.getenv(f"{prefix}_MODEL_NAME")
        self.text_routing = uses_text_routing(self.model_provider) if text_routing is None else text_routing

        self.capabilities = DatabaseCapabilities(project)
        self.functions = self.capabilities.functions() + [think]
        self.system_message = build_database_instructions(project, self.capabilities)
        # Every capability may run SQL through run_sql, so all results are parsed for "Executed SQL:".
        self._step_parsers = {func.__name__: split_executed_sql for func in self.functions if func is not think}

        if llm is None:
            llm = create_chat_model(self.model_provider, self.model_name, temperature)
        self._agent = SimpleAgent(system_message=self.system_message,
                                  temperature=temperature,
                                  llm=llm,
                                  enable_text_tool_routing=self.text_routing,
                                  max_tool_iterations=max_tool_iterations)
        for tool in to_langchain_tools(self.functions):
            self._agent.add_tool(tool)
        self._agent.compile()

    @property
    def tool_names(self) -> List[str]:
        return [tool.name for tool in self._agent.tools]

    @property
    def simple_agent(self) -> SimpleAgent:
        return self._agent

    def _input(self, question: str, history: Optional[History]) -> Dict[str, Any]:
        messages: List[Any] = []
        if not self.text_routing:
            # SimpleAgent's native path prepends its system message on every model call unless the
            # conversation already starts with one; supplying it once avoids repeated system messages.
            messages.append(SystemMessage(content=self.system_message))
        for previous_question, previous_answer in history or []:
            messages += [("user", previous_question), ("assistant", previous_answer)]
        messages.append(("user", question))
        return {"messages": messages}

    def _result(self, messages: Sequence[Any]) -> Tuple[str, List[ToolStep]]:
        return extract_answer_and_steps(messages, self.text_routing, self._step_parsers)

    def ask(self, question: str, history: Optional[History] = None) -> Tuple[str, List[ToolStep]]:
        """Answer ``question``; returns ``(answer, steps)``. SQL steps have ``details["sql"]``."""
        result = self._agent.app.invoke(self._input(question, history))
        return self._result(result["messages"])

    def stream(self,
               question: str,
               history: Optional[History] = None,
               verbose: bool = True) -> Tuple[str, List[ToolStep]]:
        """Like :meth:`ask`, printing each new message as it arrives when ``verbose``."""
        final: Optional[Sequence[Any]] = None
        for chunk in self._agent.app.stream(self._input(question, history), stream_mode="values"):
            final = chunk["messages"]
            if verbose:
                final[-1].pretty_print()
        return self._result(final) if final else ("", [])

    def run_interactive(self) -> None:
        """Simple terminal loop with conversation history; type ``exit`` to stop."""
        history: List[Tuple[str, str]] = []
        print(f"{self.project.title} — ask a question (type 'exit' to quit)")
        while True:
            try:
                question = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if question.lower() in {"exit", "quit", "q"}:
                break
            if not question:
                continue
            answer, steps = self.ask(question, history)
            for step in steps:
                for sql in step.details.get("sql", []):
                    print(f"\n[{step.tool}]\n{sql}")
            print(f"\n{answer}")
            history.append((question, answer))
