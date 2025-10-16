"""
Pizza Course Tutorial - Ultra Simple DLLMForge Version
======================================================
Super high-level and simple - demonstrates the power of DLLMForge abstraction
"""

from dllmforge.agent_core import SimpleAgent
from langchain_core.tools import tool

# ============================================================================
# 1. DEFINE TOOLS - That's it!
# ============================================================================

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b

@tool
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

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

# ============================================================================
# 2. CREATE AGENT - One line!
# ============================================================================

agent = SimpleAgent("You are a helpful assistant that can do math and tell you about pizza prices.")

# ============================================================================
# 3. ADD TOOLS - One line each!
# ============================================================================

agent.add_tool(multiply)
agent.add_tool(add)
agent.add_tool(get_pizza_price)

# ============================================================================
# 4. COMPILE - One line!
# ============================================================================

agent.compile()

# ============================================================================
# 5. THAT'S IT! Ready to use!
# ============================================================================

print("🍕 Pizza Course Tutorial - Ultra Simple DLLMForge Version")
# Display the graph
print("\nGraph Visualization:")
try:
    from IPython.display import Image, display
    # Access the compiled graph, not the raw workflow
    graph_image = agent.app.get_graph().draw_mermaid_png()
    display(Image(graph_image))
    
    # Save the image
    with open("pizza_course_ultra_simple.png", "wb") as f:
        f.write(graph_image)
    print("✅ Graph saved as 'pizza_course_ultra_simple.png'")
    
except Exception as e:
    print(f"Could not display graph: {e}")
    print("Note: Graph visualization requires additional dependencies")

# ============================================================================
# 6. TEST IT!
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING THE ULTRA SIMPLE AGENT")
    print("="*60)
    
    # Test cases
    test_queries = [
        "What is 5 * 3?",
        "What's the price of margherita pizza?",
        "Add 10 and 7",
        "Tell me about pepperoni pizza",
        "Hello, how are you?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🧪 TEST {i}: {query}")
        print("-" * 40)
        try:
            # Use stream=True to see tool calls
            agent.process_query(query, stream=True)
        except Exception as e:
            print(f"Error: {e}")
    
