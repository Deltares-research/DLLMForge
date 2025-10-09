"""
Simple agent core for DLLMForge - Clean LangGraph utilities.

This module provides simple, elegant utilities for creating LangGraph agents
following the pattern established in water_management_agent_simple.py.
"""

import os
import logging
from typing import List, Dict, Any, Callable, Literal
from dotenv import load_dotenv

# LangChain and LangGraph imports
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState, START, END

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleAgent:
    """Simple agent class for LangGraph workflows."""

    def __init__(self, system_message: str = None, temperature: float = 0.1):
        """
        Initialize a simple LangGraph agent.
        
        Args:
            system_message: System message for the agent
            temperature: LLM temperature setting
        """
        # Load environment variables
        load_dotenv()

        # Initialize Azure OpenAI
        self.llm = AzureChatOpenAI(azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT_GPT41'),
                                   api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
                                   azure_endpoint=os.getenv('AZURE_OPENAI_API_BASE'),
                                   api_key=os.getenv('AZURE_OPENAI_API_KEY'),
                                   temperature=temperature)

        # Store system message
        self.system_message = system_message or "You are a helpful AI assistant."

        # Initialize tools and workflow components
        self.tools = []
        self.workflow = StateGraph(MessagesState)
        self.app = None

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
        """Create a simple agent -> tools workflow pattern with proper human interaction handling."""

        # Create tool node if we have tools
        if self.tools:
            tool_node = ToolNode(self.tools)
            llm_with_tools = self.llm.bind_tools(self.tools)
        else:
            tool_node = None
            llm_with_tools = self.llm

        def call_model(state: MessagesState):
            """Call the LLM with current messages."""
            messages = state["messages"]

            # Add system message if not present
            if not messages or messages[0].type != "system":
                from langchain_core.messages import SystemMessage
                messages = [SystemMessage(content=self.system_message)] + messages

            response = llm_with_tools.invoke(messages)
            return {"messages": messages + [response]}

        def should_continue(state: MessagesState) -> Literal["tools", "agent", END]:
            """Determine if we should continue to tools, back to agent, or end."""
            messages = state["messages"]
            last_message = messages[-1]

            # If tools available and LLM made tool calls, continue to tools
            if self.tools and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"

            # Check if we need to continue processing after human interaction
            # Look for workflow continuation signals in recent messages
            for msg in reversed(messages[-3:]):  # Check last 3 messages
                if hasattr(msg, 'content') and msg.content:
                    content = str(msg.content).lower()
                    # If ask_human was called, we need to wait for human input
                    if "workflow_paused" in content or "human input requested" in content:
                        return END  # Stop here and wait for human response
                    # If workflow is continuing after feedback, go back to agent
                    elif "workflow_continuing" in content or "feedback received" in content:
                        return "agent"
                    # If final approval granted, end the workflow
                    elif "final_approval_granted" in content or "user is satisfied" in content:
                        return END

            # Default: if no special signals, end the workflow
            return END

        # Add nodes
        self.add_node("agent", call_model)
        if tool_node:
            self.add_node("tools", tool_node)

        # Add edges with proper routing
        self.add_edge(START, "agent")
        if tool_node:
            # Agent can go to tools or end
            self.add_conditional_edge("agent", should_continue)
            # Tools always go back to agent for continued processing
            self.add_edge("tools", "agent")
        else:
            # If no tools, agent can still continue processing
            self.add_conditional_edge("agent", should_continue)

        logger.info("Simple workflow created with human interaction support")

    def compile(self, checkpointer=None) -> None:
        """Compile the workflow."""
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
                for chunk in self.app.stream({"messages": [("user", query)]}, stream_mode="values"):
                    chunk["messages"][-1].pretty_print()
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


def create_basic_agent(system_message: str = None, temperature: float = 0.1) -> SimpleAgent:
    """
    Create a basic agent with standard setup.
    
    Args:
        system_message: System message for the agent
        temperature: LLM temperature
        
    Returns:
        SimpleAgent: Configured agent instance
    """
    agent = SimpleAgent(system_message, temperature)
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
