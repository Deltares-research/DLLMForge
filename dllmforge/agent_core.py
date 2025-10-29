"""
Simple agent core for DLLMForge - Clean LangGraph utilities.

This module provides simple, elegant utilities for creating LangGraph agents
following the pattern established in water_management_agent_simple.py.
"""

import logging
from typing import List, Callable, Literal
from dotenv import load_dotenv

# LangChain and LangGraph imports
from langchain_core.tools import tool as langchain_tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState, START, END

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def tool(func):
    """
    DLLMForge wrapper around LangChain's @tool decorator.
    
    This decorator provides a consistent interface for creating tools
    within the DLLMForge ecosystem while maintaining compatibility
    with LangChain's tool system.
    
    Args:
        func: Function to be converted into a tool
        
    Returns:
        Tool function that can be used with SimpleAgent
    """
    logger.info(f"Registering DLLMForge tool: {func.__name__}")

    # Apply the original LangChain decorator
    return langchain_tool(func)


class SimpleAgent:
    """Simple agent class for LangGraph workflows."""

    def __init__(self,
                 system_message: str = None,
                 temperature: float = 0.1,
                 model_provider: str = "azure-openai",
                 llm=None,
                 enable_text_tool_routing: bool = False,
                 max_tool_iterations: int = 3):
        """
        Initialize a simple LangGraph agent.
        
        Args:
            system_message: System message for the agent
            temperature: LLM temperature setting
            model_provider: LLM provider ("azure-openai", "openai", "mistral")
        """
        # Load environment variables
        load_dotenv()

        # Initialize LLM (allow custom LLM override)
        if llm is not None:
            self.llm = llm
        else:
            from .langchain_api import LangchainAPI
            llm_api = LangchainAPI(model_provider=model_provider, temperature=temperature)
            self.llm = llm_api.llm

        # Store system message
        self.system_message = system_message or "You are a helpful AI assistant."

        # Initialize tools and workflow components
        self.tools = []
        self.workflow = StateGraph(MessagesState)
        self.app = None
        self.enable_text_tool_routing = enable_text_tool_routing
        self.max_tool_iterations = max_tool_iterations

        logger.info("Simple agent initialized")

    def add_tool(self, tool_func: Callable) -> None:
        """
        Add a tool to the agent.
        
        Args:
            tool_func: Function decorated with @tool
        """
        self.tools.append(tool_func)
        logger.info(f"Added tool: {tool_func.name}")

    def add_node(self, name: str, func: Callable) -> None:
        """
        Add a node to the workflow.
        
        Args:
            name: Node name
            func: Node function
        """
        self.workflow.add_node(name, func)
        logger.info(f"Added node: {name}")

    def add_edge(self, from_node: str, to_node: str) -> None:
        """
        Add a simple edge between nodes.
        
        Args:
            from_node: Source node
            to_node: Target node
        """
        self.workflow.add_edge(from_node, to_node)
        logger.info(f"Added edge: {from_node} -> {to_node}")

    def add_conditional_edge(self, from_node: str, condition_func: Callable) -> None:
        """
        Add a conditional edge.
        
        Args:
            from_node: Source node
            condition_func: Function that determines routing
        """
        self.workflow.add_conditional_edges(from_node, condition_func)
        logger.info(f"Added conditional edge from: {from_node}")

    def create_simple_workflow(self) -> None:
        """Create a simple agent -> tools workflow with optional text-based tool routing."""

        if self.tools and not self.enable_text_tool_routing:
            tool_node = ToolNode(self.tools)
            llm_with_tools = self.llm.bind_tools(self.tools)

            def call_model(state: MessagesState):
                messages = state["messages"]
                if not messages or messages[0].type != "system":
                    from langchain_core.messages import SystemMessage
                    messages = [SystemMessage(content=self.system_message)] + messages
                response = llm_with_tools.invoke(messages)
                return {"messages": messages + [response]}

            def should_continue(state: MessagesState) -> Literal["tools", END]:
                messages = state["messages"]
                last_message = messages[-1]
                if self.tools and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                    return "tools"
                return END

            self.add_node("agent", call_model)
            self.add_node("tools", tool_node)
            self.add_edge(START, "agent")
            self.add_conditional_edge("agent", should_continue)
            self.add_edge("tools", "agent")
        else:
            def call_model(state: MessagesState):
                import json
                from langchain_core.messages import SystemMessage, HumanMessage

                def parse_tool_directive(text: str):
                    try:
                        start = text.find("{")
                        end = text.rfind("}")
                        if start == -1 or end == -1 or end <= start:
                            return None
                        obj = json.loads(text[start:end + 1])
                        if isinstance(obj, dict) and ("tool" in obj or "final_answer" in obj):
                            return obj
                    except Exception:
                        return None
                    return None

                messages = state["messages"]
                available_tools = [getattr(t, "name", "") for t in self.tools]
                tool_list_str = "\n".join(f"- {name}" for name in available_tools if name)
                if not messages or messages[0].type != "system":
                    routing_instructions = (
                        "You can call tools by responding ONLY with a JSON object.\n"
                        "- To call a tool: {\"tool\": \"<tool_name>\", \"args\": { ... }}\n"
                        "- To answer: {\"final_answer\": \"...\"}\n"
                        "Use EXACT tool names. Do not invent names like 'text' or 'pizza_prices'.\n"
                        f"Available tools (exact names):\n{tool_list_str}\n"
                        "Example: {\"tool\": \"add\", \"args\": {\"a\": 2, \"b\": 3}}"
                    )
                    messages = [SystemMessage(content=f"{self.system_message}\n\n{routing_instructions}")] + messages

                loop_messages = list(messages)
                for _ in range(max(1, int(self.max_tool_iterations))):
                    response = self.llm.invoke(loop_messages)
                    content = getattr(response, "content", "") or ""
                    directive = parse_tool_directive(content)
                    if directive and "tool" in directive and "args" in directive:
                        tool_name = directive["tool"]
                        tool_name_l = str(tool_name).lower()
                        resolved_name = tool_name
                        if tool_name_l in {"text", "summary", "summarise", "summarize"}:
                            summary_tool = next((n for n in available_tools if n and ("summary" in n.lower() or "summar" in n.lower())), None)
                            if summary_tool:
                                resolved_name = summary_tool
                        args = directive.get("args", {})
                        tool_func = next((t for t in self.tools if getattr(t, "name", None) == resolved_name), None)
                        if tool_func is None:
                            avail_str = ", ".join(available_tools)
                            loop_messages.append(HumanMessage(content=f"Tool '{tool_name}' not found. Available tools: {avail_str}. Use exact names."))
                            continue
                        try:
                            result = tool_func.invoke(args)
                        except Exception as e:
                            result = f"Tool error: {e}"
                        loop_messages.append(HumanMessage(content=f"Tool '{resolved_name}' result: {result}"))
                        continue
                    loop_messages.append(response)
                    break
                return {"messages": loop_messages}

            self.add_node("agent", call_model)
            self.add_edge(START, "agent")
            self.add_edge("agent", END)

        logger.info("Simple workflow created with human interaction support")

    def compile(self, checkpointer=None) -> None:
        """Compile the workflow."""
        # Automatically create simple workflow if not already created
        if self.app is None:
            self.create_simple_workflow()

        if checkpointer:
            self.app = self.workflow.compile(checkpointer=checkpointer)
        else:
            self.app = self.workflow.compile()
        logger.info("Workflow compiled successfully")

    def process_query(self, query: str, stream: bool = True) -> None:
        """
        Process a query with the agent.
        
        Args:
            query: User query
            stream: Whether to stream the response
        """
        if not self.app:
            raise RuntimeError("Workflow not compiled. Call compile() first.")

        print(f"\n{'='*60}")
        print(f"PROCESSING: {query}")
        print(f"{'='*60}\n")

        try:
            if stream:
                def _parse_tool_directive(text: str):
                    import json
                    try:
                        start = text.find("{")
                        end = text.rfind("}")
                        if start == -1 or end == -1 or end <= start:
                            return None
                        obj = json.loads(text[start:end + 1])
                        if isinstance(obj, dict) and ("tool" in obj or "final_answer" in obj):
                            return obj
                    except Exception:
                        return None
                    return None

                for chunk in self.app.stream({"messages": [("user", query)]}, stream_mode="values"):
                    last_msg = chunk["messages"][-1]
                    # Enhanced display for text-based tool routing (Deltares path)
                    if self.enable_text_tool_routing:
                        content = getattr(last_msg, "content", "") or ""
                        msg_type = getattr(last_msg, "type", None)
                        # If the model proposed a tool call as JSON, render a rich section
                        if msg_type == "ai":
                            directive = _parse_tool_directive(content)
                            if directive and "tool" in directive and "args" in directive:
                                tool_name = directive["tool"]
                                args = directive.get("args", {})
                                print("================================== Ai Message ==================================")
                                print("Tool Calls:")
                                print(f"  {tool_name} (text_routing)")
                                print("  Args:")
                                for k, v in args.items():
                                    print(f"    {k}: {v}")
                                print()
                                continue
                        # If a tool result was appended as a human message, render like a Tool Message
                        if msg_type == "human" and isinstance(content, str) and content.startswith("Tool '") and "' result: " in content:
                            # Expected format: Tool '<name>' result: <payload>
                            try:
                                name_part, result_part = content.split("' result: ", 1)
                                tool_name = name_part.replace("Tool '", "").rstrip("'")
                            except Exception:
                                tool_name = "unknown"
                                result_part = content
                            print("================================= Tool Message =================================")
                            print(f"Name: {tool_name}")
                            print()
                            print(result_part)
                            print()
                            continue
                    # Default pretty print for all other cases (including Azure/tool-node path)
                    last_msg.pretty_print()
                    print()
            else:
                result = self.app.invoke({"messages": [("user", query)]})
                result["messages"][-1].pretty_print()

        except Exception as e:
            print(f"Error processing query: {e}")

    def run_interactive(self) -> None:
        """Run the agent in interactive mode."""
        print(f"🤖 Agent Ready - Type 'quit' to exit")
        print("=" * 50)

        while True:
            try:
                user_input = input("\n💬 Your question: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                if user_input:
                    self.process_query(user_input)
            except KeyboardInterrupt:
                break

        print("\nGoodbye! 🤖")


def create_basic_agent(system_message: str = None,
                       temperature: float = 0.1,
                       model_provider: str = "azure-openai") -> SimpleAgent:
    """
    Create a basic agent with standard setup.
    
    Args:
        system_message: System message for the agent
        temperature: LLM temperature
        model_provider: LLM provider ("azure-openai", "openai", "mistral")
        
    Returns:
        SimpleAgent: Configured agent instance
    """
    agent = SimpleAgent(system_message, temperature, model_provider)
    agent.create_simple_workflow()
    agent.compile()
    return agent


def create_echo_tool():
    """Create a simple echo tool for testing."""

    @tool
    def echo(text: str) -> str:
        """Echo the input text back to the user.
        
        Args:
            text: Text to echo back
        """
        return f"Echo: {text}"

    return echo


def create_basic_tools() -> List[Callable]:
    """
    Create basic utility tools for testing.
    
    Returns:
        List of tool functions
    """

    @tool
    def echo(text: str) -> str:
        """Echo the input text back to the user.
        
        Args:
            text: Text to echo back
        """
        return f"Echo: {text}"

    @tool
    def timestamp() -> str:
        """Get the current timestamp."""
        from datetime import datetime
        return f"Current time: {datetime.now().isoformat()}"

    @tool
    def calculator(expression: str) -> str:
        """Safely evaluate simple math expressions.
        
        Args:
            expression: Math expression to evaluate (e.g., "2 + 3 * 4")
        """
        try:
            # Only allow safe characters
            allowed_chars = set('0123456789+-*/.() ')
            if not all(c in allowed_chars for c in expression):
                return "Error: Only basic math operations allowed"

            result = eval(expression)
            return f"{expression} = {result}"
        except Exception as e:
            return f"Error calculating '{expression}': {str(e)}"

    return [echo, timestamp, calculator]


# Simple usage example
if __name__ == "__main__":
    try:
        # Create agent with basic tools
        agent = SimpleAgent("You are a helpful assistant with basic tools.")

        # Add tools
        basic_tools = create_basic_tools()
        for tool in basic_tools:
            agent.add_tool(tool)

        # Create and compile workflow
        agent.create_simple_workflow()
        agent.compile()

        # Test query
        agent.process_query("What's 5 + 3 * 2?")

        # Interactive mode
        # agent.run_interactive()

    except Exception as e:
        print(f"Error: {e}")
