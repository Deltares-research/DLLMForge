"""
Tutorial: Advanced Water Management Agent
========================================
Demonstrates core LangGraph concepts for hydrologists and groundwater experts:
- Nodes: Different types of processing units (calculations, data lookup, analysis)
- Edges: Connections between nodes (workflow paths)
- Conditional Edges: AI decides which path to take based on the water problem
- Tools: Grouped by domain (calculations, aquifer info, watershed data)

Fun and practical examples for water professionals!
"""

# Load environment variables from .env file for API keys and endpoints
from dotenv import load_dotenv

load_dotenv()

# Import dllmforge simple agent and tool decorator
from dllmforge.agent_core import SimpleAgent, tool
import logging
from dllmforge.openai_api import OpenAIAPI

from langgraph.prebuilt import ToolNode
from langgraph.graph import START, END

# ============================================================================
# 1. DEFINE TOOLS - Water-focused tools for hydrologists
# ============================================================================


# Water calculation tools
@tool
def calculate_flow_rate(area: float, velocity: float) -> float:
    """Calculate flow rate using Q = A × V (discharge = area × velocity)."""
    return area * velocity


@tool
def calculate_groundwater_storage(aquifer_area: float, thickness: float, porosity: float) -> str:
    """Calculate groundwater storage volume using V = A × h × n and return a readable result."""
    volume = aquifer_area * thickness * porosity
    return f"Groundwater storage V = A×h×n = {aquifer_area} × {thickness} × {porosity} = {volume}"


@tool
def calculate_water_balance(precipitation: float, evapotranspiration: float, runoff: float) -> float:
    """Calculate water balance: P - ET - R = ΔS (change in storage)."""
    return precipitation - evapotranspiration - runoff


@tool
def calculate_darcy_velocity(hydraulic_conductivity: float, hydraulic_gradient: float) -> float:
    """Calculate Darcy velocity using v = K × i."""
    return hydraulic_conductivity * hydraulic_gradient


# Aquifer and watershed information tools - placeholder for connecting a Retreival Augmented Generation (RAG) system
@tool
def get_aquifer_info(aquifer_name: str) -> str:
    """Get information about a specific aquifer."""
    aquifer_data = {
        "ogallala": "The Ogallala Aquifer is a major water source for the Great Plains, spanning 8 states. It's a confined aquifer with high-quality water but faces depletion concerns.",
        "floridan": "The Floridan Aquifer System is one of the world's most productive aquifers, supplying water to Florida, Georgia, and Alabama. It's a karst limestone aquifer.",
        "edwards": "The Edwards Aquifer in Texas is a karst limestone aquifer known for its unique ecosystem and recharge characteristics. It supplies water to San Antonio.",
        "high plains": "The High Plains Aquifer (Ogallala) is the largest aquifer in the US, providing irrigation water for agriculture across the Great Plains region."
    }
    return aquifer_data.get(aquifer_name.lower(),
                            f"Sorry, I don't have specific information about the {aquifer_name} aquifer.")


@tool  #placeholder for connecting a Retreival Augmented Generation (RAG) system
def get_watershed_info(watershed_name: str) -> str:
    """Get information about a specific watershed."""
    watershed_data = {
        "mississippi": "The Mississippi River watershed drains 41% of the continental US, covering 1.2 million square miles. It's the 4th largest watershed in the world.",
        "colorado": "The Colorado River watershed spans 7 US states and 2 Mexican states. It's heavily managed with dams and faces water scarcity challenges.",
        "amazon": "The Amazon River watershed is the largest in the world, covering 2.7 million square miles across 9 countries in South America.",
        "nile": "The Nile River watershed spans 11 countries in Africa, providing water for 300 million people. It's the longest river in the world."
    }
    return watershed_data.get(watershed_name.lower(),
                              f"Sorry, I don't have specific information about the {watershed_name} watershed.")


# ============================================================================
# 2. CREATE AGENT
# ============================================================================

agent = SimpleAgent("""You are a helpful water management assistant for hydrologists and groundwater experts.

When users ask for water calculations, use the calculation tools:
- "What's the flow rate for area 10 m² and velocity 2 m/s?" → use calculate_flow_rate(10, 2)
- "Calculate groundwater storage for area 1000 km², thickness 50 m, porosity 0.3" → use calculate_groundwater_storage(1000, 50, 0.3)
- "What's the water balance for P=1000mm, ET=600mm, R=200mm?" → use calculate_water_balance(1000, 600, 200)
- "Calculate Darcy velocity for K=0.01 m/s and gradient 0.05" → use calculate_darcy_velocity(0.01, 0.05)

When users ask about aquifers or watersheds, use the information tools:
- "Tell me about the Ogallala aquifer" → use get_aquifer_info("ogallala")
- "What's the Mississippi watershed like?" → use get_watershed_info("mississippi")

Always use the appropriate tools for water-related questions and provide context about the calculations.""")

# ============================================================================
# 3. CREATE NODES - Water management workflow components
# ============================================================================

# Group tools by domain
calculation_tools = [
    calculate_flow_rate, calculate_groundwater_storage, calculate_water_balance, calculate_darcy_velocity
]
info_tools = [get_aquifer_info, get_watershed_info]
all_tools = calculation_tools + info_tools

# Create specialized tool nodes
calculation_node = ToolNode(calculation_tools)
info_node = ToolNode(info_tools)
unified_node = ToolNode(all_tools)

logger = logging.getLogger(__name__)


def concise_summary(state):
    """Create a concise LLM summary of the latest agent output (single-input summarizer)."""
    from langchain_core.messages import AIMessage

    messages = state["messages"]

    tool_msgs = [m for m in messages if getattr(m, "type", "") == "tool"]
    tools_text = "\n".join([getattr(tm, "content", "") for tm in tool_msgs]) or ""

    # Prefer summarizing tool outputs so calculated values are included; fallback to latest AI text
    if tools_text:
        text_to_summarize = tools_text
    else:
        text_to_summarize = ""
        for m in reversed(messages):
            if getattr(m, "type", "") == "ai" and getattr(m, "content", ""):
                text_to_summarize = m.content
                break

    try:
        from dllmforge.langchain_api import LangchainAPI

        llm_api = LangchainAPI()
        prompt_messages = [
            ("system",
             "You are a helpful assistant. Summarize the following tool results concisely (1-2 sentences). Include any numeric values verbatim; do not omit calculated numbers."
             ),
            ("human", text_to_summarize),
        ]
        response = llm_api.llm.invoke(prompt_messages)
        summary_text = getattr(response, "content", str(response))
    except Exception as e:
        summary_text = f"Could not generate summary: {e}"

    summary_message = AIMessage(content=f"Summary: {summary_text}")
    return {"messages": messages + [summary_message]}


def extended_summary(state):
    """Create an extended LLM summary of the latest agent output (single-input summarizer)."""
    from langchain_core.messages import AIMessage

    messages = state["messages"]

    tool_msgs = [m for m in messages if getattr(m, "type", "") == "tool"]
    tools_text = "\n".join([getattr(tm, "content", "") for tm in tool_msgs]) or ""

    # Prefer summarizing tool outputs so calculated values are included; fallback to latest AI text
    if tools_text:
        text_to_summarize = tools_text
    else:
        text_to_summarize = ""
        for m in reversed(messages):
            if getattr(m, "type", "") == "ai" and getattr(m, "content", ""):
                text_to_summarize = m.content
                break

    try:
        from dllmforge.langchain_api import LangchainAPI

        llm_api = LangchainAPI()
        prompt_messages = [
            ("system",
             "You are a helpful assistant. Summarize the following tool convering all possible details. Max 1 paragraph. Include any numeric values verbatim; do not omit calculated numbers."
             ),
            ("human", text_to_summarize),
        ]
        response = llm_api.llm.invoke(prompt_messages)
        summary_text = getattr(response, "content", str(response))
    except Exception as e:
        summary_text = f"Could not generate summary: {e}"

    summary_message = AIMessage(content=f"Summary: {summary_text}")
    return {"messages": messages + [summary_message]}


def steps_taken(state):
    """Summarize steps taken (which tools were invoked) along with their outputs, and log them."""
    messages = state["messages"]
    from langchain_core.messages import AIMessage

    steps_lines = []
    tool_names_used = set()
    # Collect tool invocations from agent messages
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for call in tool_calls:
                name = call.get("name", "unknown_tool")
                args = call.get("args", {})
                line = f"Called {name} with args: {args}"
                steps_lines.append(line)
                logger.info(line)
                if name:
                    tool_names_used.add(name)

    # Collect corresponding tool results
    tool_result_lines = []
    for tm in [m for m in messages if getattr(m, "type", "") == "tool"]:
        tool_name = getattr(tm, "name", getattr(tm, "tool", "tool"))
        tool_content = getattr(tm, "content", "")
        tool_result_lines.append(f"Result from {tool_name}: {tool_content}")
        if tool_name:
            tool_names_used.add(tool_name)

    # Infer deterministic routing steps (non-agent decisions)
    calculation_tool_names = {
        'calculate_flow_rate', 'calculate_groundwater_storage', 'calculate_water_balance', 'calculate_darcy_velocity'
    }
    info_tool_names = {'get_aquifer_info', 'get_watershed_info'}

    used_calc = any(name in calculation_tool_names for name in tool_names_used)
    used_info = any(name in info_tool_names for name in tool_names_used)

    route_lines = []
    if used_calc and used_info:
        route_lines.append("Routed via unified_node (agentic)")
        route_lines.append("Routed to extended_summary (deterministic)")
    elif used_calc:
        route_lines.append("Routed to calculation_node (agentic)")
        route_lines.append("Routed to concise_summary (deterministic)")
    elif used_info:
        route_lines.append("Routed to info_node (agentic)")
        route_lines.append("Routed to concise_summary (deterministic)")
    else:
        # No tools used -> direct concise_summary per routing function
        route_lines.append("Routed to concise_summary (deterministic)")

    combined = []
    if route_lines:
        combined.append("Routing decisions:")
        combined.extend(route_lines)
        # If unified path, also enumerate the unified toolset
        if used_calc and used_info:
            unified_tool_names = []
            for t in (calculation_tools + info_tools):
                unified_tool_names.append(getattr(t, "name", getattr(t, "__name__", "tool")))
            if unified_tool_names:
                combined.append("Unified toolset:")
                combined.extend(unified_tool_names)
    if steps_lines:
        combined.append("Tool calls:")
        combined.extend(steps_lines)
    if tool_result_lines:
        combined.append("Tool results:")
        combined.extend(tool_result_lines)

    if not combined:
        steps_text = "No tools used"
    else:
        steps_text = "\n• ".join([""] + combined)

    logger.info("Steps summary complete")
    steps_message = AIMessage(content=f"🔧 Agentic Steps Taken:{steps_text}")
    return {"messages": messages + [steps_message]}


# Add nodes to the workflow
agent.add_node("calculation_node", calculation_node)
agent.add_node("info_node", info_node)
agent.add_node("unified_node", unified_node)
agent.add_node("concise_summary", concise_summary)
agent.add_node("extended_summary", extended_summary)
agent.add_node("steps_taken", steps_taken)

# ============================================================================
# 4. CREATE CONDITIONAL EDGE - Let AI decide the path
# ============================================================================


def route_to_node(state):
    """Conditional edge: AI decides which node to use based on the water problem."""
    messages = state["messages"]
    last_message = messages[-1]

    # Check if the agent wants to call tools
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        tool_calls = last_message.tool_calls
        tool_names = [tool_call['name'] for tool_call in tool_calls]

        # Check which type of tools are needed
        calculation_tools_needed = any(tool in [
            'calculate_flow_rate', 'calculate_groundwater_storage', 'calculate_water_balance',
            'calculate_darcy_velocity'
        ] for tool in tool_names)
        info_tools_needed = any(tool in ['get_aquifer_info', 'get_watershed_info'] for tool in tool_names)

        # Route based on tool types
        if calculation_tools_needed and info_tools_needed:
            return "unified_node"  # Both calculation and info needed
        elif calculation_tools_needed:
            return "calculation_node"  # Only calculations needed
        elif info_tools_needed:
            return "info_node"  # Only info needed
        else:
            return "unified_node"  # Default fallback
    else:
        # No tools needed, route to concise summary
        return "concise_summary"


# Add the conditional edge
agent.add_conditional_edge("agent", route_to_node)

# ============================================================================
# 5. CREATE REGULAR EDGES
# ============================================================================


# Define the agent node
def call_model(state):
    """Agent node: The AI that makes decisions."""
    messages = state["messages"]

    # Add system message if not present
    if not messages or messages[0].type != "system":
        from langchain_core.messages import SystemMessage
        messages = [SystemMessage(content=agent.system_message)] + messages

    # Bind all tools to the agent
    llm_with_tools = agent.llm.bind_tools(all_tools)
    response = llm_with_tools.invoke(messages)
    return {"messages": messages + [response]}


agent.add_node("agent", call_model)

# Add edges to create the workflow
agent.add_edge(START, "agent")  # Start → Agent
agent.add_edge("calculation_node", "concise_summary")  # Calculation Node → Concise Summary
agent.add_edge("info_node", "concise_summary")  # Info Node → Concise Summary
agent.add_edge("unified_node", "extended_summary")  # Unified Node → Extended Summary
agent.add_edge("concise_summary", "steps_taken")  # Concise Summary → Steps Taken
agent.add_edge("extended_summary", "steps_taken")  # Extended Summary → Steps Taken
agent.add_edge("steps_taken", END)  # Steps Taken → End

# ============================================================================
# 6. COMPILE THE WORKFLOW
# ============================================================================

agent.app = agent.workflow.compile()

# ============================================================================
# 7. DISPLAY WORKFLOW INFORMATION
# ============================================================================

print("💧 Advanced Water Management Agent Tutorial")
print("=" * 60)
print("Core concepts demonstrated:")
print("• NODES: agent, calculation_node, info_node, unified_node, concise_summary, extended_summary, steps_taken")
print("• TOOLS: calculate_flow_rate, calculate_groundwater_storage, calculate_water_balance, calculate_darcy_velocity")
print("• INFO TOOLS: get_aquifer_info, get_watershed_info")
print("• ROUTING (agentic): agent decides tool usage → calculation_node | info_node | unified_node")
print("• ROUTING (deterministic): calculation/info → concise_summary; unified → extended_summary → steps_taken → END")
print("• WORKFLOW: START → agent → tool_node(s) → summary_node → steps_taken → END")
print("=" * 60)

# ============================================================================
# 8. TEST THE WORKFLOW
# ============================================================================

if __name__ == "__main__":
    # Test cases that demonstrate different routing paths for water professionals
    test_queries = [
        "What's the flow rate for a channel with area 15 m² and velocity 1.5 m/s?. Calculate groundwater storage for area 2000 km², thickness 30 m, porosity 0.25. Tell me about the Ogallala aquifer",  # Should route to calculation_node
        "Tell me about the Ogallala aquifer",  # Should route to info_node
        "Calculate groundwater storage for area 2000 km², thickness 30 m, porosity 0.25 and tell me about the Floridan aquifer"  # Should route to unified_node
    ]

    print("\n🧪 Testing the water management workflow:")
    print("=" * 60)

    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}: {query}")
        print("-" * 40)
        try:
            agent.process_query(query, stream=True)
        except Exception as e:
            print(f"Error: {e}")

    print("\n" + "=" * 60)
    print("✅ Water management concepts demonstrated!")
    print("• Nodes contain calculation and information tools")
    print("• Edges connect nodes in logical workflow paths")
    print("• Agentic vs deterministic routing clearly demonstrated")
    print("=" * 60)
