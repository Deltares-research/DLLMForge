Tutorial: Advanced Water Management Agent
=========================================

This tutorial demonstrates how to build an advanced LangGraph/DLLMForge agent with hydrological related examples. The agent showcases core LangGraph concepts including nodes, edges, and conditional edges using practical water management examples.

Learning Objectives
------------------

After completing this tutorial, you will understand:

- How to create specialized nodes for different types of water analysis
- How to implement conditional edges that route queries based on water problem types
- How to group tools by domain (calculations vs. information retrieval)
- How to build a workflow that intelligently handles hydrological questions

Core Concepts Demonstrated
--------------------------

**Nodes**: Different types of processing units
   - Agent node: Makes routing decisions
   - Calculation node: Handles water calculations
   - Info node: Retrieves aquifer and watershed information
   - Unified node: Handles complex queries requiring both calculations and information

**Edges**: Connections between nodes, this control is what makes it more advanced than the simple agent, as it allows explicit routing rather than relying on the prompt only.
   - Regular edges: Direct connections (calculation_node → summary)
   - Conditional edges: AI decides which path to take based on the query

**Tools**: Grouped by domain for clarity
   - Water calculation tools: Flow rate, groundwater storage, water balance, Darcy velocity
   - Information tools: Aquifer and watershed data (with RAG system placeholders)

Workflow Overview
-----------------

The water management agent uses intelligent routing to determine the appropriate processing path (copy the code into mermaid.live to visualize the workflow to view the grap):

.. mermaid::
   :align: center

   graph TD
       START([START]) --> agent[Agent Decision<br/>💧 Water Management AI]
       
       agent -->|"Water calculations needed<br/>(flow rate, storage, balance, Darcy)"| calculation_node[Calculation Node<br/>🔢 calculate_flow_rate<br/>calculate_groundwater_storage<br/>calculate_water_balance<br/>calculate_darcy_velocity]
       
       agent -->|"Aquifer/watershed info needed<br/>(RAG system placeholders)"| info_node[Info Node<br/>📚 get_aquifer_info<br/>get_watershed_info]
       
       agent -->|"Both calculations & info needed"| unified_node[Unified Node<br/>🔄 All Tools Available<br/>Calculations + Information]
       
       agent -->|"No tools needed"| summary[Summary<br/>💧 Water Analysis Complete]
       
       calculation_node --> summary
       info_node --> summary
       unified_node --> summary
       
       summary --> END([END])
       
       classDef startEnd fill:#e1f5fe,stroke:#01579b,stroke-width:3px
       classDef agent fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
       classDef calculation fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
       classDef info fill:#fff3e0,stroke:#e65100,stroke-width:2px
       classDef unified fill:#fff8e1,stroke:#f57c00,stroke-width:2px
       classDef summary fill:#fce4ec,stroke:#880e4f,stroke-width:2px
       
       class START,END startEnd
       class agent agent
       class calculation_node calculation
       class info_node info
       class unified_node unified
       class summary summary

Water Calculation Tools
-----------------------

The agent includes four essential water calculation tools based on fundamental hydrological equations:

**Flow Rate Calculation**
   .. code-block:: python

      @tool
      def calculate_flow_rate(area: float, velocity: float) -> float:
          """Calculate flow rate using Q = A × V (discharge = area × velocity)."""
          return area * velocity

**Groundwater Storage Calculation**
   .. code-block:: python

      @tool
      def calculate_groundwater_storage(aquifer_area: float, thickness: float, porosity: float) -> float:
          """Calculate groundwater storage volume using V = A × h × n."""
          return aquifer_area * thickness * porosity

**Water Balance Calculation**
   .. code-block:: python

      @tool
      def calculate_water_balance(precipitation: float, evapotranspiration: float, runoff: float) -> float:
          """Calculate water balance: P - ET - R = ΔS (change in storage)."""
          return precipitation - evapotranspiration - runoff

**Darcy Velocity Calculation**
   .. code-block:: python

      @tool
      def calculate_darcy_velocity(hydraulic_conductivity: float, hydraulic_gradient: float) -> float:
          """Calculate Darcy velocity using v = K × i."""
          return hydraulic_conductivity * hydraulic_gradient

Information Retrieval Tools
---------------------------

The agent includes information tools with placeholders for connecting to RAG (Retrieval Augmented Generation) systems:

**Aquifer Information Tool**
   .. code-block:: python

      @tool  # placeholder for connecting a Retrieval Augmented Generation (RAG) system
      def get_aquifer_info(aquifer_name: str) -> str:
          """Get information about a specific aquifer."""
          # This would connect to your RAG system for real-time aquifer data
          aquifer_data = {
              "ogallala": "The Ogallala Aquifer is a major water source...",
              "floridan": "The Floridan Aquifer System is one of the world's most productive aquifers...",
              # Additional aquifer data
          }
          return aquifer_data.get(aquifer_name.lower(), f"Sorry, I don't have specific information about the {aquifer_name} aquifer.")

**Watershed Information Tool**
   .. code-block:: python

      @tool  # placeholder for connecting a Retrieval Augmented Generation (RAG) system
      def get_watershed_info(watershed_name: str) -> str:
          """Get information about a specific watershed."""
          # This would connect to your RAG system for real-time watershed data
          watershed_data = {
              "mississippi": "The Mississippi River watershed drains 41% of the continental US...",
              "colorado": "The Colorado River watershed spans 7 US states...",
              # Additional watershed data
          }
          return watershed_data.get(watershed_name.lower(), f"Sorry, I don't have specific information about the {watershed_name} watershed.")

Conditional Edge Implementation
------------------------------

The conditional edge function determines which node to route to based on the tools the agent wants to call:

.. code-block:: python

   def route_to_node(state):
       """Conditional edge: AI decides which node to use based on the water problem."""
       messages = state["messages"]
       last_message = messages[-1]
       
       # Check if the agent wants to call tools
       if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
           tool_calls = last_message.tool_calls
           tool_names = [tool_call['name'] for tool_call in tool_calls]
           
           # Check which type of tools are needed
           calculation_tools_needed = any(tool in ['calculate_flow_rate', 'calculate_groundwater_storage', 'calculate_water_balance', 'calculate_darcy_velocity'] for tool in tool_names)
           info_tools_needed = any(tool in ['get_aquifer_info', 'get_watershed_info'] for tool in tool_names)
           
           # Route based on tool types
           if calculation_tools_needed and info_tools_needed:
               return "unified_node"  # Both calculation and info needed
           elif calculation_tools_needed:
               return "calculation_node"      # Only calculations needed
           elif info_tools_needed:
               return "info_node"      # Only info needed
           else:
               return "unified_node"   # Default fallback
       else:
           # No tools needed, go directly to summary
           return "summary"

Workflow Assembly
-----------------

The workflow is assembled by adding nodes and edges:

.. code-block:: python

   # Group tools by domain
   calculation_tools = [calculate_flow_rate, calculate_groundwater_storage, calculate_water_balance, calculate_darcy_velocity]
   info_tools = [get_aquifer_info, get_watershed_info]
   all_tools = calculation_tools + info_tools

   # Create specialized tool nodes
   calculation_node = ToolNode(calculation_tools)
   info_node = ToolNode(info_tools)
   unified_node = ToolNode(all_tools)

   # Add nodes to the workflow
   agent.add_node("calculation_node", calculation_node)
   agent.add_node("info_node", info_node)
   agent.add_node("unified_node", unified_node)
   agent.add_node("summary", create_summary)

   # Add conditional edge
   agent.add_conditional_edge("agent", route_to_node)

   # Add regular edges
   agent.add_edge(START, "agent")
   agent.add_edge("calculation_node", "summary")
   agent.add_edge("info_node", "summary")
   agent.add_edge("unified_node", "summary")
   agent.add_edge("summary", END)

Testing the Workflow
--------------------

The tutorial includes test cases that demonstrate different routing scenarios:

**Test Case 1: Flow Rate Calculation**
   - Query: "What's the flow rate for a channel with area 15 m² and velocity 1.5 m/s?"
   - Route: Agent → Calculation Node → Summary → End
   - Tools Used: calculate_flow_rate

**Test Case 2: Aquifer Information**
   - Query: "Tell me about the Ogallala aquifer"
   - Route: Agent → Info Node → Summary → End
   - Tools Used: get_aquifer_info

**Test Case 3: Complex Analysis**
   - Query: "Calculate groundwater storage for area 2000 km², thickness 30 m, porosity 0.25 and tell me about the Floridan aquifer"
   - Route: Agent → Unified Node → Summary → End
   - Tools Used: calculate_groundwater_storage + get_aquifer_info

Running the Tutorial
--------------------

To run the complete tutorial:

.. code-block:: python

   python tutorial_advanced_agent.py

This will:
1. Display the workflow structure
2. Run the test cases
3. Show the routing decisions for each query type

Testing with Custom Queries
----------------------------

Once the agent is compiled, you can test it with your own water management queries:

.. code-block:: python

   # Test flow rate calculation
   agent.process_query("What's the flow rate for a channel with area 15 m² and velocity 1.5 m/s?", stream=True)
   
   # Test aquifer information
   agent.process_query("Tell me about the Ogallala aquifer", stream=True)
   
   # Test complex analysis requiring both calculations and information
   agent.process_query("Calculate groundwater storage for area 2000 km², thickness 30 m, porosity 0.25 and tell me about the Floridan aquifer", stream=True)
   
   # Test water balance calculation
   agent.process_query("Calculate the water balance for precipitation 1000 mm, evapotranspiration 600 mm, and runoff 200 mm", stream=True)
   
   # Test Darcy velocity
   agent.process_query("What's the Darcy velocity for hydraulic conductivity 0.01 m/s and hydraulic gradient 0.05?", stream=True)

The agent will automatically:
1. **Analyze the query** to determine which tools are needed
2. **Route to the appropriate node** (calculation, info, or unified)
3. **Execute the required tools** for calculations or information retrieval
4. **Generate a comprehensive summary** of the results

Key Benefits for Water Professionals
------------------------------------

**Domain-Specific Tools**: All tools are designed for hydrological calculations and information retrieval

**Intelligent Routing**: The agent automatically determines whether calculations, information lookup, or both are needed

**RAG Integration Ready**: Information tools include placeholders for connecting to real-time databases

**Scalable Architecture**: Easy to add new calculation tools or information sources

**Professional Context**: Uses proper hydrological terminology and real-world examples

Next Steps
----------

- Connect the information tools to your RAG system for real-time aquifer and watershed data
- Add more specialized calculation tools for your specific hydrological needs
- Implement additional nodes for data visualization or report generation
- Extend the conditional routing logic for more complex decision trees

This tutorial provides a solid foundation for building sophisticated water management agents using LangGraph's powerful workflow capabilities.
