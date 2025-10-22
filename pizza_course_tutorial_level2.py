"""
Pizza Course Tutorial - Level 2: Core Concepts
=============================================
Demonstrates: nodes, tools, edges, conditional edges
Simple and focused on the fundamentals
"""

from dllmforge.agent_core import SimpleAgent, tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, END

# ============================================================================
# 1. DEFINE TOOLS - Grouped by domain for clarity
# ============================================================================

# Math tools
@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b

@tool
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

# Pizza tools
@tool
def get_pizza_price(pizza_type: str) -> float:
    """Get the price of a pizza type."""
    prices = {
        "margherita": 12.99,
        "pepperoni": 15.99,
        "vegetarian": 14.99,
        "supreme": 17.99
    }
    return prices.get(pizza_type.lower(), 10.99)

@tool
def get_pizza_ingredients(pizza_type: str) -> str:
    """Get the ingredients of a pizza type."""
    ingredients = {
        "margherita": "Tomato sauce, mozzarella, basil",
        "pepperoni": "Tomato sauce, mozzarella, pepperoni",
        "vegetarian": "Tomato sauce, mozzarella, bell peppers, mushrooms, onions",
        "supreme": "Tomato sauce, mozzarella, pepperoni, sausage, bell peppers, mushrooms"
    }
    return ingredients.get(pizza_type.lower(), "Unknown pizza type")

@tool
def draw_ascii_pizza(pizza_type: str) -> str:
    """Draw a custom pizza in ASCII art based on the pizza type."""
    # Let the LLM generate custom ASCII art
    from langchain_core.messages import HumanMessage
    
    prompt = f"""
    Create a beautiful ASCII art drawing of a {pizza_type} pizza.
    
    Requirements:
    - Use ASCII characters (no emojis)
    - Make it visually appealing
    - Include the pizza name at the top
    - Show toppings appropriate for {pizza_type}
    - Keep it reasonably sized (not too big)
    - Use characters like: ┌ ┐ └ ┘ │ ╱ ╲ █ ▓ ░ ▒
    
    Return only the ASCII art, no explanations.
    """
    
    # Use the agent's LLM to generate the ASCII art
    response = agent.llm.invoke([HumanMessage(content=prompt)])
    return response.content

# ============================================================================
# 2. CREATE AGENT - Same as Level 1!
# ============================================================================

agent = SimpleAgent("You are a helpful assistant that can do math and tell you about pizza.")

# ============================================================================
# 3. CREATE SPECIALIZED NODES - Math and Pizza!
# ============================================================================

# Group tools by domain for nodes
math_tools = [multiply, add]
pizza_tools = [get_pizza_price, get_pizza_ingredients]
drawing_tools = [draw_ascii_pizza]

# Create specialized tool nodes
math_node = ToolNode(math_tools)
pizza_node = ToolNode(pizza_tools)
drawing_node = ToolNode(drawing_tools)

# Create summary node (non-tool node)
def create_summary(state):
    """
    SUMMARY NODE: Create a conversational summary of what the user requested.
    This demonstrates NON-CONDITIONAL edges - always flows here before ending.
    """
    messages = state["messages"]
    user_message = messages[0].content  # Original user request
    
    summary_prompt = f"""
    The user asked: "{user_message}"
    
    Create a brief, friendly summary of what you accomplished.
    Keep it simple and conversational.
    
    Examples:
    - "I calculated 5 * 3 for you, which equals 15!"
    - "I found the price of margherita pizza - it's $12.99!"
    - "I drew you a pepperoni pizza in ASCII art!"
    
    Keep it short and friendly.
    """
    
    from langchain_core.messages import HumanMessage
    summary_response = agent.llm.invoke([HumanMessage(content=summary_prompt)])
    
    from langchain_core.messages import AIMessage
    summary_message = AIMessage(content=f"📝 **Summary:** {summary_response.content}")
    
    return {"messages": messages + [summary_message]}

# Add nodes to workflow
agent.add_node("math_node", math_node)
agent.add_node("pizza_node", pizza_node)
agent.add_node("drawing_node", drawing_node)
agent.add_node("summary", create_summary)

# ============================================================================
# 4. ADD CONDITIONAL EDGE - Let agent decide!
# ============================================================================

def route_to_node(state):  
    """  
    CONDITIONAL EDGE: Let the agent decide which node to use.  
    """  
    messages = state["messages"]  
    last_message = messages[-1]  
    
    # If agent wants to call tools, check which type
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        # Check which tools the agent wants to call
        tool_calls = last_message.tool_calls
        for tool_call in tool_calls:
            tool_name = tool_call['name']
            if tool_name in ['multiply', 'add']:
                return "math_node"
            elif tool_name in ['get_pizza_price', 'get_pizza_ingredients']:
                return "pizza_node"
            elif tool_name == 'draw_ascii_pizza':
                return "drawing_node"
        return "math_node"  # Default to math if unclear
    else:  
        # If no tools needed, go to summary
        return "summary"  

# Add conditional edge from agent
agent.add_conditional_edge("agent", route_to_node)

# ============================================================================
# 5. ADD REGULAR EDGES 
# ============================================================================

# Define the agent node
def call_model(state):
    """Agent node - the AI that makes decisions."""
    messages = state["messages"]
    
    # Add system message
    if not messages or messages[0].type != "system":
        from langchain_core.messages import SystemMessage
        messages = [SystemMessage(content=agent.system_message)] + messages
    
    # Bind all tools to the agent - let it choose which ones to use
    all_tools = math_tools + pizza_tools + drawing_tools
    llm_with_tools = agent.llm.bind_tools(all_tools)
    response = llm_with_tools.invoke(messages)
    return {"messages": messages + [response]}

agent.add_node("agent", call_model)

# Add edges to create workflow
agent.add_edge(START, "agent")           # Start goes to agent
agent.add_edge("math_node", "summary")   # Math node goes to summary (NON-CONDITIONAL)
agent.add_edge("pizza_node", "summary")  # Pizza node goes to summary (NON-CONDITIONAL)
agent.add_edge("drawing_node", "summary") # Drawing node goes to summary (NON-CONDITIONAL)
agent.add_edge("summary", END)           # Summary goes to end (NON-CONDITIONAL)

# ============================================================================
# 6. COMPILE - Create the workflow!
# ============================================================================

agent.app = agent.workflow.compile()

# ============================================================================
# 7. THAT'S IT! Ready to test!
# ============================================================================

print("🍕 Pizza Course Tutorial - Level 2: Core Concepts")
print("=" * 50)
print("Core concepts demonstrated:")
print("• NODES: math_node, pizza_node, drawing_node, summary, agent")
print("• TOOLS: multiply, add, get_pizza_price, get_pizza_ingredients, draw_ascii_pizza")
print("• CONDITIONAL EDGE: agent decides which tool node to go to")
print("• NON-CONDITIONAL EDGES: all tool nodes → summary → END")
print("=" * 50)

# Display the graph
print("\nGraph Visualization:")
try:
    # Method 1: Use LangGraph's native visualization
    graph = agent.app.get_graph()
    
    # Print the graph structure as text first
    print("Graph Structure:")
    print("=" * 40)
    print(f"Nodes: {list(graph.nodes.keys())}")
    print(f"Edges: {list(graph.edges)}")
    
    # Try different ways to access conditional edges
    try:
        print(f"Conditional Edges: {list(graph.conditional_edges)}")
    except AttributeError:
        try:
            print(f"Conditional Edges: {list(graph.conditional_edges_map)}")
        except AttributeError:
            try:
                print(f"Conditional Edges: {list(graph.conditional_edges_map.keys())}")
            except AttributeError:
                print("Conditional Edges: Unable to access (API may have changed)")
    
    print("=" * 40)
    
    # Debug: Show all available attributes
    print("Available Graph Attributes:")
    attrs = [attr for attr in dir(graph) if not attr.startswith('_')]
    print(f"Graph has {len(attrs)} attributes: {attrs[:10]}...")  # Show first 10
    print("=" * 40)
    
    # Try Mermaid visualization
    try:
        print("\nAttempting Mermaid visualization...")
        graph_image = graph.draw_mermaid_png(
            background_color="white",
            padding=20
        )
        
        # Try to display in Jupyter/IPython
        try:
            from IPython.display import Image, display
            display(Image(graph_image))
            print("✅ Graph displayed in Jupyter")
        except ImportError:
            print("ℹ️ IPython not available, saving image only")
        
        # Save the image
        with open("pizza_course_level2_core.png", "wb") as f:
            f.write(graph_image)
        print("✅ Graph saved as 'pizza_course_level2_core.png'")
        
    except Exception as viz_error:
        print(f"❌ Mermaid visualization failed: {viz_error}")
        print("\nTrying alternative visualization methods...")
        
        # Alternative: Try to get Mermaid syntax
        try:
            mermaid_syntax = graph.draw_mermaid()
            print("Mermaid Syntax:")
            print("=" * 40)
            print(mermaid_syntax)
            print("=" * 40)
            print("You can copy this syntax to https://mermaid.live/ to visualize")
        except Exception as syntax_error:
            print(f"Could not get Mermaid syntax: {syntax_error}")
        
        # Fallback: Simple text representation
        print("\nText Workflow Representation:")
        print("START → agent")
        print("agent → [CONDITIONAL] → math_node | pizza_node | drawing_node | summary")
        print("math_node → summary")
        print("pizza_node → summary")
        print("drawing_node → summary")
        print("summary → END")
        
except Exception as e:
    print(f"❌ Could not access graph: {e}")
    print("The workflow may not have compiled correctly")

# ============================================================================
# 8. TEST IT!
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*50)
    print("TESTING CORE CONCEPTS")
    print("="*50)
    
    # Simple test cases
    test_queries = [
        "What is 5 * 3?",                    # Should route to math_node
        "What's the price of margherita?",   # Should route to pizza_node
        "Add 10 and 7",                      # Should route to math_node
        "What ingredients are in pepperoni?", # Should route to pizza_node
        "Draw me a pepperoni pizza",         # Should route to drawing_node
        "Hello, how are you?"                # Should go to general_response
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🧪 TEST {i}: {query}")
        print("-" * 30)
        try:
            agent.process_query(query, stream=True)
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "="*50)
    print("CORE CONCEPTS DEMONSTRATED!")
    print("• Nodes contain tools")
    print("• Edges connect nodes")
    print("• Conditional edges let AI choose paths")
    print("="*50)