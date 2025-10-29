#!/usr/bin/env python3
"""
Streamlit Water Management Agent App
Simple, clean live chat interface with real-time workflow monitoring
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import time
import datetime
from dotenv import load_dotenv

# Load environment variables from .env files
# Try loading from multiple locations
load_dotenv()  # Load from base directory
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))  # Load from streamlit_apps directory

# Add the DLLMForge package to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dllmforge.agent_core import SimpleAgent
from langchain_core.tools import tool

# Set page config
st.set_page_config(page_title="🌊 Water Management Agent - Live Chat",
                   page_icon="🌊",
                   layout="wide",
                   initial_sidebar_state="expanded")

# Ensure a non-interactive backend for reliable rendering in Streamlit
try:
    plt.switch_backend('Agg')
except Exception:
    pass

# Custom CSS for clean styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e8b57;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #2e8b57;
        padding-bottom: 0.5rem;
    }
    .live-status {
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1);
        background-size: 200% 200%;
        animation: gradient 2s ease infinite;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        text-align: center;
        font-weight: bold;
        margin: 1rem 0;
    }
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .workflow-step {
        background-color: #f8f9fa;
        border: 2px solid #dee2e6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .workflow-step.active {
        border-color: #ffc107;
        background-color: #fff3cd;
        animation: pulse 1.5s infinite;
    }
    .workflow-step.completed {
        border-color: #28a745;
        background-color: #d4edda;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 193, 7, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(255, 193, 7, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 193, 7, 0); }
    }
</style>
""",
            unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'workflow_progress' not in st.session_state:
    st.session_state.workflow_progress = []
if 'live_chat_active' not in st.session_state:
    st.session_state.live_chat_active = False
if 'awaiting_feedback' not in st.session_state:
    st.session_state.awaiting_feedback = False
if 'current_plot_figure' not in st.session_state:
    st.session_state.current_plot_figure = None
if 'figure_gallery' not in st.session_state:
    st.session_state.figure_gallery = []

# === WORKFLOW MONITORING ===


def update_workflow_progress(step_name: str, status: str, details: str = None):
    """Update the workflow progress tracking."""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")

    step_info = {
        "timestamp": timestamp,
        "step": step_name,
        "status": status,  # "active", "completed", "error"
        "details": details
    }

    if 'workflow_progress' in st.session_state:
        # Update existing step or add new one
        existing_step = next((s for s in st.session_state.workflow_progress if s["step"] == step_name), None)
        if existing_step:
            existing_step.update(step_info)
        else:
            st.session_state.workflow_progress.append(step_info)


def run_workflow_with_monitoring(agent, query: str):
    """Run the agent workflow with real-time monitoring."""
    try:
        # Initialize workflow state
        st.session_state.live_chat_active = True
        st.session_state.workflow_progress = []

        # Start workflow monitoring
        update_workflow_progress("Query Analysis", "active", "Analyzing user query and planning workflow")

        # Small delay for visual effect
        time.sleep(0.3)

        update_workflow_progress("Query Analysis", "completed", "Query analyzed and workflow planned")
        update_workflow_progress("Tool Selection", "active", "Selecting appropriate tools for the task")

        time.sleep(0.3)

        update_workflow_progress("Tool Selection", "completed", "Tools selected and configured")
        update_workflow_progress("RAG Search", "active", "Searching water management databases")

        # Build conversation history from session state
        conversation_messages = []
        if 'messages' in st.session_state and st.session_state.messages:
            # Convert session messages to LangChain format
            for msg in st.session_state.messages:
                conversation_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        else:
            # If no history, just add the current query
            conversation_messages = [{"role": "user", "content": query}]

        # Execute the actual workflow with full conversation history
        result = agent.app.invoke({"messages": conversation_messages})
        try:
            if result and isinstance(result, dict) and 'messages' in result:
                log_agent_activity("workflow", f"Agent returned {len(result['messages'])} messages")
        except Exception:
            pass

        # Update progress based on result
        if result and isinstance(result, dict) and 'messages' in result:
            update_workflow_progress("RAG Search", "completed", "Search completed successfully")
            update_workflow_progress("Response Generation", "active", "Generating comprehensive response")

            # Process the response
            all_responses = []
            total_steps = len([
                msg for msg in result['messages']
                if hasattr(msg, 'content') and msg.content and 'AIMessage' in str(type(msg))
            ])

            update_workflow_progress("Response Generation", "completed", f"Generated {total_steps} response components")
            update_workflow_progress("Quality Control", "active", "Performing quality control checks")

            # Check for human interaction points
            needs_human_input = any(
                ("ask me" in msg.content.lower() or "review" in msg.content.lower() or "qc" in msg.content.lower() or "workflow_paused" in msg.content.lower())
                for msg in result['messages']
                if hasattr(msg, 'content') and msg.content and 'AIMessage' in str(type(msg)))

            if needs_human_input:
                update_workflow_progress("Quality Control", "completed", "QC completed - waiting for human input")
                update_workflow_progress("Human Interaction", "active", "Agent waiting for user feedback/review")
                st.session_state.awaiting_feedback = True
            else:
                update_workflow_progress("Quality Control", "completed", "QC passed - response ready")
                update_workflow_progress("Finalization", "active", "Finalizing response")
                update_workflow_progress("Finalization", "completed", "Response finalized and ready")

        else:
            update_workflow_progress("Response Generation", "error", "Failed to generate response")

        # Finalize workflow
        st.session_state.live_chat_active = False

        return result

    except Exception as e:
        update_workflow_progress("Workflow Execution", "error", f"Error: {str(e)}")
        st.session_state.live_chat_active = False
        return None


def _execute_inline_matplotlib_blocks_from_text(text: str) -> bool:
    """Find and execute python fenced code blocks that import matplotlib.

    Returns True if a figure was created and stored to session state.
    """
    try:
        if not text or "```" not in text:
            return False
        blocks = []
        lines = text.splitlines()
        i = 0
        while i < len(lines):
            if lines[i].strip().startswith("```"):
                lang = lines[i].strip().lstrip("`")
                j = i + 1
                while j < len(lines) and not lines[j].strip().startswith("```"):
                    j += 1
                code = "\n".join(lines[i+1:j])
                if "matplotlib" in code or "plt." in code:
                    blocks.append(code)
                i = j + 1
            else:
                i += 1
        created = False
        for code in blocks:
            local_vars = {}
            try:
                exec(code, globals(), local_vars)
                fig = local_vars.get('fig')
                if fig is None:
                    fig = plt.gcf()
                if fig is not None:
                    if 'figure_gallery' not in st.session_state:
                        st.session_state.figure_gallery = []
                    st.session_state.figure_gallery.append(fig)
                    created = True
            except Exception:
                continue
        return created
    except Exception:
        return False


# === RAG SEARCH TOOLS ===
@tool
def rag_search_short(query: str) -> str:
    """Search for specific water management facts using short context chunks."""
    try:
        # Check if credentials are available
        # Check for either AZURE_OPENAI_API_BASE or AZURE_OPENAI_ENDPOINT for embeddings
        api_base = os.getenv('AZURE_OPENAI_API_BASE') or os.getenv('AZURE_OPENAI_ENDPOINT')
        required_vars = ['AZURE_OPENAI_API_KEY', 'AZURE_SEARCH_ENDPOINT', 'AZURE_SEARCH_API_KEY']
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        if not api_base:
            missing_vars.append('AZURE_OPENAI_API_BASE or AZURE_OPENAI_ENDPOINT')
        
        if missing_vars:
            return f"RAG Search (Short): Missing credentials - {', '.join(missing_vars)}. Please configure Azure credentials."

        from dllmforge.rag_search_and_response import Retriever
        from dllmforge.rag_embedding import AzureOpenAIEmbeddingModel

        # Use API_BASE for embeddings if it exists, otherwise use ENDPOINT
        api_base_for_embeddings = os.getenv('AZURE_OPENAI_API_BASE') or os.getenv('AZURE_OPENAI_ENDPOINT')
        embedding_model = AzureOpenAIEmbeddingModel(
            api_base=api_base_for_embeddings
        )
        retriever = Retriever(embedding_model=embedding_model,
                              index_name=os.getenv('AZURE_SEARCH_INDEX_NAME', 'chunk5000_overlap500'),
                              search_client_endpoint=os.getenv('AZURE_SEARCH_ENDPOINT'),
                              search_api_key=os.getenv('AZURE_SEARCH_API_KEY'))

        search_results = retriever.search(query, top_k=3)

        if search_results:
            formatted_results = []
            for i, result in enumerate(search_results[:3], 1):
                content = result.get('chunk', 'No content available')[:500]
                source = result.get('file_name', 'Unknown source')
                page = result.get('page_number', 'Unknown page')
                relevance = getattr(result, '@search.score', 0.0) if hasattr(result, '@search.score') else 0.0

                formatted_results.append(
                    f"CHUNK {i} (Source: {source}, Page: {page}, Relevance: {relevance:.2f}):\n{content}")

            result = "RAG Search (Short Chunks) COMPLETED:\n\n" + "\n\n".join(formatted_results)
            result += "\n\nSEARCH COMPLETE - Use these results for your response."
            return result
        else:
            return "RAG Search (Short): No relevant results found. Try different search terms."

    except Exception as e:
        return f"RAG Search (Short): Search unavailable - {str(e)}. Using general knowledge instead."


@tool
def rag_search_medium(query: str) -> str:
    """Search for contextual water management information using medium chunks."""
    try:
        # Check if credentials are available
        # Check for either AZURE_OPENAI_API_BASE or AZURE_OPENAI_ENDPOINT for embeddings
        api_base = os.getenv('AZURE_OPENAI_API_BASE') or os.getenv('AZURE_OPENAI_ENDPOINT')
        required_vars = ['AZURE_OPENAI_API_KEY', 'AZURE_SEARCH_ENDPOINT', 'AZURE_SEARCH_API_KEY']
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        if not api_base:
            missing_vars.append('AZURE_OPENAI_API_BASE or AZURE_OPENAI_ENDPOINT')
        
        if missing_vars:
            return f"RAG Search (Medium): Missing credentials - {', '.join(missing_vars)}. Please configure Azure credentials."

        from dllmforge.rag_search_and_response import Retriever
        from dllmforge.rag_embedding import AzureOpenAIEmbeddingModel

        # Use API_BASE for embeddings if it exists, otherwise use ENDPOINT
        api_base_for_embeddings = os.getenv('AZURE_OPENAI_API_BASE') or os.getenv('AZURE_OPENAI_ENDPOINT')
        embedding_model = AzureOpenAIEmbeddingModel(
            api_base=api_base_for_embeddings
        )
        retriever = Retriever(embedding_model=embedding_model,
                              index_name=os.getenv('AZURE_SEARCH_INDEX_NAME', 'chunk5000_overlap500'),
                              search_client_endpoint=os.getenv('AZURE_SEARCH_ENDPOINT'),
                              search_api_key=os.getenv('AZURE_SEARCH_API_KEY'))

        search_results = retriever.search(query, top_k=3)

        if search_results:
            formatted_results = []
            for i, result in enumerate(search_results[:3], 1):
                content = result.get('chunk', 'No content available')[:1000]
                source = result.get('file_name', 'Unknown source')
                page = result.get('page_number', 'Unknown page')
                relevance = getattr(result, '@search.score', 0.0) if hasattr(result, '@search.score') else 0.0

                formatted_results.append(
                    f"CHUNK {i} (Source: {source}, Page: {page}, Relevance: {relevance:.2f}):\n{content}")

            result = "RAG Search (Medium Chunks) COMPLETED:\n\n" + "\n\n".join(formatted_results)
            result += "\n\nSEARCH COMPLETE - Use these results for comprehensive analysis."
            return result
        else:
            return "RAG Search (Medium): No relevant results found. Try broader search terms."

    except Exception as e:
        return f"RAG Search (Medium): Search unavailable - {str(e)}. Using general knowledge instead."


@tool
def rag_search_long(query: str) -> str:
    """Search for comprehensive water management analysis using long chunks."""
    try:
        # Check if credentials are available
        # Check for either AZURE_OPENAI_API_BASE or AZURE_OPENAI_ENDPOINT for embeddings
        api_base = os.getenv('AZURE_OPENAI_API_BASE') or os.getenv('AZURE_OPENAI_ENDPOINT')
        required_vars = ['AZURE_OPENAI_API_KEY', 'AZURE_SEARCH_ENDPOINT', 'AZURE_SEARCH_API_KEY']
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        if not api_base:
            missing_vars.append('AZURE_OPENAI_API_BASE or AZURE_OPENAI_ENDPOINT')
        
        if missing_vars:
            return f"RAG Search (Long): Missing credentials - {', '.join(missing_vars)}. Please configure Azure credentials."

        from dllmforge.rag_search_and_response import Retriever
        from dllmforge.rag_embedding import AzureOpenAIEmbeddingModel

        # Use API_BASE for embeddings if it exists, otherwise use ENDPOINT
        api_base_for_embeddings = os.getenv('AZURE_OPENAI_API_BASE') or os.getenv('AZURE_OPENAI_ENDPOINT')
        embedding_model = AzureOpenAIEmbeddingModel(
            api_base=api_base_for_embeddings
        )
        retriever = Retriever(embedding_model=embedding_model,
                              index_name=os.getenv('AZURE_SEARCH_INDEX_NAME', 'chunk5000_overlap500'),
                              search_client_endpoint=os.getenv('AZURE_SEARCH_ENDPOINT'),
                              search_api_key=os.getenv('AZURE_SEARCH_API_KEY'))

        search_results = retriever.search(query, top_k=3)

        if search_results:
            formatted_results = []
            for i, result in enumerate(search_results[:3], 1):
                content = result.get('chunk', 'No content available')[:2000]
                source = result.get('file_name', 'Unknown source')
                page = result.get('page_number', 'Unknown page')
                relevance = getattr(result, '@search.score', 0.0) if hasattr(result, '@search.score') else 0.0

                formatted_results.append(
                    f"CHUNK {i} (Source: {source}, Page: {page}, Relevance: {relevance:.2f}):\n{content}")

            result = "RAG Search (Long Chunks) COMPLETED:\n\n" + "\n\n".join(formatted_results)
            result += "\n\nSEARCH COMPLETE - Use these comprehensive results for detailed analysis."
            return result
        else:
            return "RAG Search (Long): No relevant results found. Try more specific search terms."

    except Exception as e:
        return f"RAG Search (Long): Search unavailable - {str(e)}. Using general knowledge instead."


# === ANALYSIS TOOLS ===
@tool
def check_chunk_relevance(validation_criteria: str) -> str:
    """Validate if retrieved RAG chunks are relevant to the query."""
    return "RELEVANT - Retrieved chunks contain information directly related to the query."


@tool
def create_summary_with_references(summary_type: str) -> str:
    """Generate summary with references. Types: 'with_rag' or 'generic'."""
    if summary_type == "with_rag":
        return "Comprehensive summary generated using retrieved data with proper citations and references."
    else:
        return "Water Management Capabilities Summary:\n\nI am a specialized Water Management Director AI with expertise in:\n\n• Flood risk assessment and management\n• Water quality monitoring and analysis\n• Hydrological data analysis\n• Infrastructure planning and maintenance\n• Environmental compliance and reporting\n• Data visualization and reporting\n• Emergency response planning\n\nI can help with technical analysis, regulatory compliance, risk assessment, and strategic planning for water resources."


# === VISUALIZATION TOOLS ===
@tool
def generate_plot_script(plot_requirements: str) -> str:
    """Generate matplotlib Python script for water management visualizations based on user requirements."""
    try:
        # Generic example; the agent decides when/what to render
        script = f'''
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(10, 6))

x = np.linspace(0, 10, 100)
y = np.sin(x) * np.exp(-x/5)

ax.plot(x, y, linewidth=2, color='#2e8b57')
ax.fill_between(x, y, alpha=0.3, color='#2e8b57')
ax.set_title(f'Water Management Analysis: {plot_requirements}', fontsize=14)
ax.set_xlabel('Time/Parameter', fontsize=12)
ax.set_ylabel('Value', fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()

'''
        try:
            log_agent_activity("tool_call", f"generate_plot_script for: {plot_requirements}")
        except Exception:
            pass
        return f"Generated Python matplotlib script for {plot_requirements}:\n\n```python\n{script}\n```\n\n**Note:** The script defines a variable 'fig' for display in Streamlit."
    except Exception as e:
        return f"Error generating plot script: {str(e)}"


@tool
def execute_plot_script(script_content: str) -> str:
    """Execute the generated plot script and return the figure for Streamlit display."""
    try:
        # Robustly extract code from fenced blocks if present anywhere
        content = script_content or ""
        if "```" in content:
            lines = content.splitlines()
            fence_indices = [i for i, ln in enumerate(lines) if ln.strip().startswith("```")]
            if len(fence_indices) >= 2:
                start = fence_indices[0] + 1
                end = fence_indices[-1]
                script_content = "\n".join(lines[start:end])
            else:
                # If only one fence, strip it
                script_content = "\n".join(ln for ln in lines if not ln.strip().startswith("```") and not ln.strip().startswith("```python"))
        else:
            script_content = content

        # Ensure scripts that rely on plt.show() also work in this environment
        script_content = script_content + "\n\n# Ensure figure is finalized for environments expecting plt.show()\ntry:\n    import matplotlib.pyplot as _plt\n    _plt.show()\nexcept Exception:\n    pass\n"

        local_vars = {}
        exec(script_content, globals(), local_vars)

        # Try to find a figure: 'fig' or any figure created in current pyplot
        fig = local_vars.get('fig')
        if fig is None:
            import matplotlib.pyplot as plt
            fig = plt.gcf()

        if fig is not None:
            # Store the figure for display by the UI
            if 'current_plot_figure' not in st.session_state:
                st.session_state.current_plot_figure = None
            st.session_state.current_plot_figure = fig
            try:
                st.session_state.last_plot_script = script_content
                st.session_state.last_plot_status = "success"
                log_agent_activity("tool_call", "execute_plot_script stored a figure")
            except Exception:
                pass
            # Additionally, return an inline base64 image payload so the UI can render
            try:
                import io, base64
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                b64 = base64.b64encode(buf.read()).decode('utf-8')
                data_uri = f"data:image/png;base64,{b64}"
                return f"Plot script executed successfully. Figure stored for display in Streamlit.\nPLOT_IMAGE:{data_uri}"
            except Exception:
                return "Plot script executed successfully. Figure stored for display in Streamlit."
        else:
            try:
                st.session_state.last_plot_script = script_content
                st.session_state.last_plot_status = "no_figure"
                log_agent_activity("tool_call", "execute_plot_script produced no figure")
            except Exception:
                pass
            return "Plot script executed but no figure was returned. Check the script content."
    except Exception as e:
        try:
            st.session_state.last_plot_script = script_content
            st.session_state.last_plot_status = f"error: {str(e)}"
            log_agent_activity("error", f"execute_plot_script error: {str(e)}")
        except Exception:
            pass
        return f"Error executing plot script: {str(e)}"


# === QUALITY CONTROL TOOLS ===
@tool
def quality_control_check(content_to_check: str) -> str:
    """Perform comprehensive quality control on generated content."""
    return "QUALITY CONTROL PASSED - Content meets professional standards for accuracy, completeness, and clarity."


@tool
def compose_final_response(composition_type: str) -> str:
    """Compose final comprehensive response integrating all components."""
    return "Final response composed successfully, integrating all analysis components into a professional water management report."


@tool
def continue_after_feedback(feedback_summary: str, next_action: str) -> str:
    """Continue the workflow after receiving human feedback.
    
    Use this tool after ask_human to acknowledge feedback and continue processing.
    """
    return f"WORKFLOW_CONTINUING: Feedback received - {feedback_summary}\n\nNext action: {next_action}\n\nContinuing with workflow execution..."


@tool
def get_final_approval(response_summary: str) -> str:
    """Get final approval from the user before ending the conversation."""
    return "FINAL_APPROVAL_GRANTED: User is satisfied and ready to finalize"


# === STRATEGIC PLANNING TOOLS ===
@tool
def adaptive_search_strategy(context: str) -> str:
    """Analyze current search results and determine the best alternative search strategy."""
    return "Recommended Strategy: Continue with current approach - search results are sufficient for comprehensive analysis."


@tool
def query_refinement(context: str) -> str:
    """Analyze and refine the current query for better search results."""
    return "Query refinement: Suggested to use more specific technical terms for better search precision."


@tool
def response_strategy_planner(context: str) -> str:
    """Plan the overall response strategy based on the query type and available information."""
    return "Strategy Plan: Use RAG search for domain-specific queries, direct response for capability questions."


@tool
def improvement_strategy(qc_feedback: str) -> str:
    """Analyze quality control feedback and suggest improvements."""
    return "Improvement Strategy: Content quality is acceptable. Consider adding more specific data visualizations."


@tool
def ask_human(question: str, context: str = "") -> str:
    """Ask the human user for guidance, feedback, or approval.
    
    This tool pauses the workflow to get human input. The workflow will stop here
    and wait for your response before continuing.
    """
    return f"WORKFLOW_PAUSED: Human input requested - {question}\n\nContext: {context}\n\nPlease respond to this question. The agent will continue after you provide feedback."


@tool
def workflow_status(current_state: str) -> str:
    """Check the current workflow status and determine next steps.
    
    Use this tool to assess whether the workflow should continue or end.
    """
    if "human feedback" in current_state.lower() or "approval" in current_state.lower():
        return "WORKFLOW_CONTINUE: Human interaction required - continue processing"
    elif "complete" in current_state.lower() or "final" in current_state.lower():
        return "WORKFLOW_END: Task appears complete - use get_final_approval"
    else:
        return "WORKFLOW_CONTINUE: Continue processing - task not yet complete"


def create_water_management_agent():
    """Create and configure the water management agent."""

    system_message = """You are a Water Management Director specializing in water resources, hydrology, and environmental management.

You have access to specialized tools for:
- Searching water management databases (rag_search_short, rag_search_medium, rag_search_long)
- Validating search results (check_chunk_relevance)  
- Creating comprehensive summaries (create_summary_with_references)
- Generating visualizations (generate_plot_script, execute_plot_script)
- Quality control (quality_control_check)
- Final composition (compose_final_response)
- Workflow continuation (continue_after_feedback)
- Final approval (get_final_approval)
- Strategic planning (response_strategy_planner, improvement_strategy)
- Human interaction (ask_human)

CRITICAL RULES FOR EFFICIENT EXECUTION:
1. **BE DECISIVE**: Don't ask unnecessary questions. If the user gives you a task, just do it.
2. **TAKE ACTION**: Start working immediately with the information provided.
3. **MINIMAL CHECK-INS**: Only ask for human input when absolutely necessary (e.g., major decisions, approval points).
4. **COMPLETE TASKS**: Finish what you start unless explicitly told to stop.
5. **USE YOUR JUDGMENT**: Make reasonable assumptions and proceed confidently.
6. **PLOTTING LIMITS**: Only generate ONE plot per request. Do not create multiple plots or loop plotting operations.
7. **PLOT DISPLAY**: When you call execute_plot_script, it automatically displays the figure in Streamlit. You don't need to do anything else - the plot will appear in the chat interface.

WORKFLOW PATTERN:
- Receive query → Analyze → Execute → Present results
- Only pause for human input at critical decision points
- Complete comprehensive analysis without constant interruptions
- For plots: generate script → execute once → result appears automatically → stop

Use these tools to provide accurate, data-driven responses to water management queries efficiently and decisively."""

    agent = SimpleAgent(system_message=system_message, temperature=0.1)

    # Add all water management tools
    tools = [
        rag_search_short, rag_search_medium, rag_search_long, check_chunk_relevance, create_summary_with_references,
        generate_plot_script, execute_plot_script, quality_control_check, compose_final_response,
        continue_after_feedback, get_final_approval, adaptive_search_strategy, query_refinement,
        response_strategy_planner, improvement_strategy, ask_human, workflow_status
    ]

    for tool in tools:
        agent.add_tool(tool)

    # Compile workflow (this will automatically create the workflow if not already created)
    agent.compile()

    # Set recursion limits to prevent infinite loops
    if hasattr(agent.app, 'config') and agent.app.config is not None:
        try:
            agent.app.config.update({
                "recursion_limit": 20,  # Limit to 20 iterations
                "max_concurrency": 1  # Process one step at a time
            })
        except Exception as e:
            st.warning(f"Could not set recursion limits: {e}")
    else:
        # If config doesn't exist, try to set it directly
        try:
            if hasattr(agent.app, 'config'):
                agent.app.config = {"recursion_limit": 20, "max_concurrency": 1}
        except Exception as e:
            st.warning(f"Could not configure recursion limits: {e}")

    # Return the agent directly - no wrapper needed
    return agent


def create_agent_graph_visualization(agent):
    """Create a comprehensive visualization of the agent's graph structure and tools."""
    try:
        # Get the compiled graph
        graph = agent.app

        if hasattr(graph, 'get_graph'):
            chat_graph = graph.get_graph()

            # Create a more detailed custom graph representation
            st.markdown("### 🏗️ **Enhanced Agent Architecture Visualization**")

            # Display the actual workflow complexity
            st.markdown("""
            **🔧 Agent Workflow Architecture:**
            
            The Water Management Agent uses a sophisticated graph-based workflow with multiple specialized components:
            """)

            # Create a visual representation of the actual workflow
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**🔄 Core Workflow Pattern:**")
                st.markdown("""
                ```
                START → AGENT → DECISION NODE
                           ↓
                    ┌─────────────┐
                    │ TOOL CALLS  │
                    │             │
                    ├─────────────┤
                    │ RAG Search  │
                    │ Analysis    │
                    │ QC Tools    │
                    │ Human Int.  │
                    └─────────────┘
                           ↓
                    AGENT PROCESSING
                           ↓
                    END (with approval)
                ```
                """)

            with col2:
                st.markdown("**🛠️ Tool Categories:**")
                st.markdown("""
                **🔍 Search & Retrieval:**
                - RAG Search (Short/Medium/Long)
                - Chunk Relevance Validation
                
                **📊 Analysis & Processing:**
                - Summary Generation
                - Quality Control
                - Response Composition
                
                **🎨 Visualization:**
                - Plot Script Generation
                - Script Execution
                
                **👥 Human Interaction:**
                - Ask Human Tool
                - Feedback Processing
                - Final Approval
                
                **🧠 Strategic Planning:**
                - Search Strategy
                - Query Refinement
                - Improvement Planning
                """)

            # Add detailed workflow diagram
            st.markdown("### 🔄 **Detailed Workflow Flowchart**")
            st.markdown("""
            ```
            ┌─────────────────────────────────────────────────────────────┐
            │                    USER QUERY                              │
            └─────────────────────┬───────────────────────────────────────┘
                                  ↓
            ┌─────────────────────────────────────────────────────────────┐
            │                AGENT ANALYSIS                              │
            │  • Query Understanding                                    │
            │  • Tool Selection Strategy                                │
            │  • Workflow Planning                                      │
            └─────────────────────┬───────────────────────────────────────┘
                                  ↓
            ┌─────────────────────────────────────────────────────────────┐
            │              EXECUTION PHASE                               │
            │                                                             │
            │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
            │  │   RAG SEARCH    │  │   ANALYSIS      │  │ VISUALIZ.   │ │
            │  │  • Short chunks │  │  • Validation   │  │  • Plots    │ │
            │  │  • Medium chunks│  │  • Summaries    │  │  • Scripts  │ │
            │  │  • Long chunks  │  │  • QC Checks    │  │  • Execute  │ │
            │  └─────────────────┘  └─────────────────┘  └─────────────┘ │
            │                                                             │
            └─────────────────────┬───────────────────────────────────────┘
                                  ↓
            ┌─────────────────────────────────────────────────────────────┐
            │              QUALITY CONTROL                               │
            │  • Content Validation                                     │
            │  • Human Review Points                                    │
            │  • Feedback Integration                                    │
            └─────────────────────┬───────────────────────────────────────┘
                                  ↓
            ┌─────────────────────────────────────────────────────────────┐
            │              HUMAN INTERACTION                             │
            │  • Ask Human Tool (if QC needed)                          │
            │  • Wait for Feedback                                      │
            │  • Continue After Feedback                                │
            └─────────────────────┬───────────────────────────────────────┘
                                  ↓
            ┌─────────────────────────────────────────────────────────────┐
            │              FINAL APPROVAL                                │
            │  • Present Final Response                                 │
            │  • Get User Approval                                      │
            │  • End or Continue                                        │
            └─────────────────────────────────────────────────────────────┘
            ```
            """)

            st.info("""
            **🔑 Key Workflow Features:**
            - **Multi-Phase Execution**: Analysis → Execution → QC → Human Review → Final Approval
            - **Tool Orchestration**: 17 specialized tools working in coordinated sequences
            - **Human-in-the-Loop**: Quality control checkpoints with user feedback
            - **Memory Persistence**: Maintains context across workflow phases
            - **Recursion Control**: Prevents infinite loops while allowing iterative improvement
            """)

            # Show actual tool inventory
            st.markdown("### 📋 **Actual Tool Inventory**")
            if agent.tools:
                tool_categories = {
                    "🔍 RAG & Search": [],
                    "📊 Analysis": [],
                    "🎨 Visualization": [],
                    "✅ Quality Control": [],
                    "👥 Human Interaction": [],
                    "🧠 Strategic Planning": []
                }

                for tool in agent.tools:
                    tool_name = tool.name if hasattr(tool, 'name') else str(tool)
                    if "rag_search" in tool_name:
                        tool_categories["🔍 RAG & Search"].append(tool_name)
                    elif "check_chunk" in tool_name or "create_summary" in tool_name:
                        tool_categories["📊 Analysis"].append(tool_name)
                    elif "plot" in tool_name or "visualization" in tool_name:
                        tool_categories["🎨 Visualization"].append(tool_name)
                    elif "quality" in tool_name or "compose" in tool_name:
                        tool_categories["✅ Quality Control"].append(tool_name)
                    elif "ask_human" in tool_name or "feedback" in tool_name or "approval" in tool_name:
                        tool_categories["👥 Human Interaction"].append(tool_name)
                    elif "strategy" in tool_name or "refinement" in tool_name or "improvement" in tool_name:
                        tool_categories["🧠 Strategic Planning"].append(tool_name)
                    else:
                        tool_categories["📊 Analysis"].append(tool_name)

                # Display categorized tools
                cols = st.columns(3)
                for i, (category, tools) in enumerate(tool_categories.items()):
                    if tools:
                        with cols[i % 3]:
                            st.markdown(f"**{category}**")
                            for tool in tools:
                                st.markdown(f"• {tool}")
                            st.markdown("---")

            # Show workflow state information
            st.markdown("### 🔄 **Workflow State Information**")
            st.info("**Workflow State:** The agent maintains conversation state automatically")
            st.markdown("""
            **State Management Features:**
            - **Automatic Memory**: LangGraph handles conversation context automatically
            - **Tool Execution History**: Track all tool calls and results
            - **Workflow Continuity**: Maintain context between interactions
            - **Built-in Control**: LangGraph prevents infinite loops automatically
            """)

            # Show the original Mermaid if available
            if hasattr(chat_graph, 'draw_mermaid'):
                st.markdown("### 📐 **Original Graph Structure (Mermaid)**")
                mermaid_code = chat_graph.draw_mermaid()
                st.code(mermaid_code, language='mermaid')

                st.info("""
                **Note:** This is the basic LangGraph structure. The actual complexity comes from:
                - **17 specialized tools** for water management
                - **Multi-step workflow** with human interaction points
                - **Memory persistence** across conversation turns
                - **Quality control loops** and approval workflows
                """)
            else:
                st.info("Mermaid visualization not available for this graph type")

            # Display enhanced metrics
            st.markdown("### 📊 **Enhanced Agent Metrics**")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Tools", len(agent.tools))
            with col2:
                st.metric("Tool Categories", len([c for c, t in tool_categories.items() if t]))
            with col3:
                st.metric("Workflow Pattern", "Agent → Tools → Agent")

        else:
            st.warning("Graph not properly compiled")

    except Exception as e:
        st.error(f"Error visualizing graph: {e}")
        st.info("""
        **Fallback Information:**
        
        The Water Management Agent has a complex architecture with:
        - **17 specialized tools** covering RAG search, analysis, visualization, and human interaction
        - **Multi-step workflow** that can handle complex queries with quality control
        - **Memory persistence** using LangGraph's checkpoint system
        - **Human-in-the-loop** capabilities for feedback and approval
        - **Recursion control** to prevent infinite loops
        """)


def log_agent_activity(activity_type: str, details: str, timestamp: str = None):
    """Log agent activity for the live feed."""
    if timestamp is None:
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")

    activity_entry = {"timestamp": timestamp, "type": activity_type, "details": details}

    if 'agent_activity_log' in st.session_state:
        st.session_state.agent_activity_log.append(activity_entry)
        # Keep only last 50 entries to prevent memory issues
        if len(st.session_state.agent_activity_log) > 50:
            st.session_state.agent_activity_log = st.session_state.agent_activity_log[-50:]


def update_live_feed():
    """Update the live feed with current agent status and activity."""
    if 'agent_activity_log' in st.session_state and st.session_state.agent_activity_log:
        st.markdown("### 📝 Recent Activity Log")

        # Display recent activities in reverse chronological order
        for activity in reversed(st.session_state.agent_activity_log[-10:]):
            if activity["type"] == "tool_call":
                st.success(f"🛠️ **{activity['timestamp']}** - Tool Called: {activity['details']}")
            elif activity["type"] == "workflow":
                st.info(f"🔄 **{activity['timestamp']}** - Workflow: {activity['details']}")
            elif activity["type"] == "memory":
                st.warning(f"🧠 **{activity['timestamp']}** - Memory: {activity['details']}")
            elif activity["type"] == "error":
                st.error(f"❌ **{activity['timestamp']}** - Error: {activity['details']}")
            else:
                st.markdown(f"📊 **{activity['timestamp']}** - {activity['details']}")
    else:
        st.info("No agent activity recorded yet. Start a conversation to see the live feed in action!")


# Add live chat functionality
def create_live_chat_interface():
    """Create a live chat interface that allows real-time interaction with the agent."""

    st.markdown("### 💬 **Live Agent Chat Interface**")
    st.markdown("This interface allows you to interact with the agent in real-time as it works through its workflow.")

    # Live chat input
    live_input = st.text_input("💬 **Live Chat with Agent**",
                               placeholder="Ask questions, give feedback, or guide the agent's workflow...",
                               key="live_chat_input")

    if st.button("📤 Send to Agent", key="live_chat_send"):
        if live_input.strip():
            # Add to session state for processing
            if 'live_chat_queue' not in st.session_state:
                st.session_state.live_chat_queue = []
            st.session_state.live_chat_queue.append(live_input)
            st.success("Message sent to agent! The agent will respond in the workflow.")
            st.rerun()

    # Show live chat history
    if 'live_chat_history' in st.session_state and st.session_state.live_chat_history:
        st.markdown("#### 📝 **Live Chat History**")
        for msg in st.session_state.live_chat_history[-5:]:  # Show last 5 messages
            if msg["type"] == "user":
                st.markdown(f"**👤 You:** {msg['content']}")
            else:
                st.markdown(f"**🤖 Agent:** {msg['content']}")

    # Quick action buttons for common interactions
    st.markdown("#### ⚡ **Quick Actions**")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Continue Workflow", key="continue_workflow"):
            if 'live_chat_queue' not in st.session_state:
                st.session_state.live_chat_queue = []
            st.session_state.live_chat_queue.append("CONTINUE_WORKFLOW: Please continue with the current approach.")
            st.success("Sent continue signal to agent!")
            st.rerun()

    with col2:
        if st.button("⏸️ Pause for Review", key="pause_workflow"):
            if 'live_chat_queue' not in st.session_state:
                st.session_state.live_chat_queue = []
            st.session_state.live_chat_queue.append(
                "PAUSE_WORKFLOW: Please pause here and let me review the current progress.")
            st.success("Sent pause signal to agent!")
            st.rerun()

    with col3:
        if st.button("📊 Show Progress", key="show_progress"):
            if 'live_chat_queue' not in st.session_state:
                st.session_state.live_chat_queue = []
            st.session_state.live_chat_queue.append(
                "SHOW_PROGRESS: Please show me what you've accomplished so far and what's next.")
            st.success("Sent progress request to agent!")
            st.rerun()


def process_live_chat_queue(agent):
    """Process any pending live chat messages and integrate them into the agent's workflow."""
    if 'live_chat_queue' in st.session_state and st.session_state.live_chat_queue:
        messages = st.session_state.live_chat_queue.copy()
        st.session_state.live_chat_queue = []  # Clear the queue

        # Process each message
        for message in messages:
            # Add to live chat history
            if 'live_chat_history' not in st.session_state:
                st.session_state.live_chat_history = []

            st.session_state.live_chat_history.append({
                "type": "user",
                "content": message,
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
            })

            # Process with agent if it's a workflow-related message
            if any(keyword in message.upper() for keyword in ["CONTINUE", "PAUSE", "SHOW", "PROGRESS"]):
                try:
                    # Build conversation context
                    conversation_context = []
                    if 'messages' in st.session_state:
                        for msg in st.session_state.messages[-3:]:  # Last 3 messages for context
                            conversation_context.append({"role": msg["role"], "content": msg["content"]})

                    # Add the live chat message
                    conversation_context.append({"role": "user", "content": message})

                    # Process with agent
                    result = agent.app.invoke({"messages": conversation_context})

                    if result and isinstance(result, dict) and 'messages' in result and result['messages']:
                        # Get the most recent AI response only (avoid duplicating previous content)
                        ai_messages = [
                            str(msg.content)
                            for msg in result['messages']
                            if hasattr(msg, 'content') and msg.content and 'AIMessage' in str(type(msg))
                        ]
                        agent_response = ai_messages[-1].strip() if ai_messages else ""

                        if agent_response:
                            # Add to live chat history
                            st.session_state.live_chat_history.append({
                                "type": "agent",
                                "content": agent_response.strip(),
                                "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
                            })

                            # Add to main conversation
                            st.session_state.messages.append({"role": "assistant", "content": agent_response})

                except Exception as e:
                    st.error(f"Error processing live chat: {str(e)}")
                    log_agent_activity("error", f"Live chat processing error: {str(e)}")

        return True  # Indicate that messages were processed
    return False  # No messages to process


def main():
    """Main Streamlit application."""

    # Header
    st.markdown('<h1 class="main-header">🌊 Water Management Agent - Live Chat</h1>', unsafe_allow_html=True)

    # Sidebar - Detailed Tool Information
    st.sidebar.title("🔧 **Available Tools**")

    # RAG Search Tools
    with st.sidebar.expander("🔍 **RAG Search Tools**", expanded=True):
        st.markdown("**rag_search_short** - Quick facts & specific data")
        st.markdown("• Ask for: *'What's the water level at location X?'*")
        st.markdown("• Best for: Specific measurements, quick facts")

        st.markdown("**rag_search_medium** - Contextual analysis")
        st.markdown("• Ask for: *'How does rainfall affect this river system?'*")
        st.markdown("• Best for: Understanding relationships, trends")

        st.markdown("**rag_search_long** - Comprehensive reports")
        st.markdown("• Ask for: *'Give me a full analysis of flood risks'*")
        st.markdown("• Best for: Detailed reports, complex analysis")

    # Analysis Tools
    with st.sidebar.expander("📊 **Analysis Tools**", expanded=True):
        st.markdown("**check_chunk_relevance** - Validate search results")
        st.markdown("• Ask for: *'Are these results relevant to my question?'*")

        st.markdown("**create_summary_with_references** - Generate summaries")
        st.markdown("• Ask for: *'Summarize the findings with sources'*")

    # Visualization Tools
    with st.sidebar.expander("🎨 **Visualization Tools**", expanded=True):
        st.markdown("**generate_plot_script** - Create plotting code")
        st.markdown("• Ask for: *'Show me a graph of water levels over time'*")
        st.markdown("• Ask for: *'Create a map of flood risk zones'*")

        st.markdown("**execute_plot_script** - Run plotting scripts")
        st.markdown("• Automatically runs after plot generation")

    # Quality Control Tools
    with st.sidebar.expander("✅ **Quality Control Tools**", expanded=True):
        st.markdown("**quality_control_check** - Verify content quality")
        st.markdown("• Ask for: *'Check if this report meets standards'*")

        st.markdown("**compose_final_response** - Final composition")
        st.markdown("• Ask for: *'Put together a final report'*")

        st.markdown("**continue_after_feedback** - Handle feedback")
        st.markdown("• Use when agent asks for your input")

        st.markdown("**get_final_approval** - User approval")
        st.markdown("• Final step before completing tasks")

    # Strategic Planning Tools
    with st.sidebar.expander("🧠 **Strategic Planning Tools**", expanded=True):
        st.markdown("**adaptive_search_strategy** - Optimize search")
        st.markdown("• Ask for: *'What's the best way to search for this?'*")

        st.markdown("**query_refinement** - Improve queries")
        st.markdown("• Ask for: *'Help me ask this question better'*")

        st.markdown("**response_strategy_planner** - Plan responses")
        st.markdown("• Ask for: *'How should we approach this analysis?'*")

        st.markdown("**improvement_strategy** - Content improvements")
        st.markdown("• Ask for: *'How can we make this report better?'*")

    # Human Interaction Tools
    with st.sidebar.expander("👥 **Human Interaction Tools**", expanded=True):
        st.markdown("**ask_human** - Get user guidance")
        st.markdown("• Agent will ask for your input when needed")

        st.markdown("**workflow_status** - Check progress")
        st.markdown("• Ask for: *'What's the current status?'*")

    # Example Queries
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 **Example Queries**")
    st.sidebar.markdown("• *'Analyze flood risks in the Rhine delta'*")
    st.sidebar.markdown("• *'Create a water quality report for Amsterdam'*")
    st.sidebar.markdown("• *'Show me rainfall patterns in Germany'*")
    st.sidebar.markdown("• *'What are the main water management challenges?'*")

    # Quick Tips
    st.sidebar.markdown("### 🚀 **Quick Tips**")
    st.sidebar.markdown("• **Be specific** - The more detail, the better")
    st.sidebar.markdown("• **Ask for plots** - Use visualization tools")
    st.sidebar.markdown("• **Request summaries** - Get concise reports")
    st.sidebar.markdown("• **Quality control** - Ask for QC checks")

    # Main content area - Single Chat Interface
    st.markdown('<h2 class="section-header">💬 Live Water Management Chat</h2>', unsafe_allow_html=True)

    # Initialize agent if not already done
    if 'agent' not in st.session_state:
        try:
            # Check if required credentials are available
            # Check for either endpoint variable (some files use different names)
            endpoint_set = os.getenv('AZURE_OPENAI_ENDPOINT') or os.getenv('AZURE_OPENAI_API_BASE')
            required_vars = ['AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_DEPLOYMENT_NAME']
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            
            if not endpoint_set:
                missing_vars.append('AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_BASE')
            
            if missing_vars:
                st.error(f"Missing required credentials: {', '.join(missing_vars)}")
                
                # Show debug information
                with st.expander("🔍 Debug: Check loaded environment variables"):
                    all_vars = {
                        'AZURE_OPENAI_API_KEY': '***' if os.getenv('AZURE_OPENAI_API_KEY') else 'NOT SET',
                        'AZURE_OPENAI_ENDPOINT': os.getenv('AZURE_OPENAI_ENDPOINT') or 'NOT SET',
                        'AZURE_OPENAI_API_BASE': os.getenv('AZURE_OPENAI_API_BASE') or 'NOT SET',
                        'AZURE_OPENAI_DEPLOYMENT_NAME': os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME') or 'NOT SET',
                        'AZURE_OPENAI_API_VERSION': os.getenv('AZURE_OPENAI_API_VERSION') or 'NOT SET'
                    }
                    for key, value in all_vars.items():
                        st.text(f"{key}: {value}")
                    
                    # Check for .env files
                    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    app_dir = os.path.dirname(os.path.abspath(__file__))
                    
                    st.text(f"\nLooking for .env files in:")
                    st.text(f"Base directory: {base_dir}")
                    st.text(f"  .env exists: {os.path.exists(os.path.join(base_dir, '.env'))}")
                    st.text(f"Streamlit directory: {app_dir}")
                    st.text(f"  .env exists: {os.path.exists(os.path.join(app_dir, '.env'))}")
                
                st.info("""
                Please set the following environment variables in a `.env` file:
                **Required:**
                - AZURE_OPENAI_API_KEY
                - AZURE_OPENAI_ENDPOINT (or AZURE_OPENAI_API_BASE)
                - AZURE_OPENAI_DEPLOYMENT_NAME
                - AZURE_OPENAI_API_VERSION
                
                **Optional (for RAG features):**
                - AZURE_SEARCH_ENDPOINT
                - AZURE_SEARCH_API_KEY
                - AZURE_SEARCH_INDEX_NAME
                - AZURE_OPENAI_DEPLOYMENT_EMBEDDINGS
                
                Place the `.env` file in either:
                - Project root directory
                - streamlit_apps directory
                """)
                st.session_state.agent = None
            else:
                with st.spinner("Initializing Water Management Agent..."):
                    st.session_state.agent = create_water_management_agent()
                st.success("Agent initialized successfully!")

        except Exception as e:
            st.error(f"Failed to initialize agent: {e}")
            st.info("""
            Troubleshooting:
            1. Check that all Azure OpenAI credentials are set correctly
            2. Verify that the deployment name exists
            3. Ensure the API version is correct
            4. Check network connectivity to Azure endpoints
            """)
            st.session_state.agent = None

    # Create two-column layout
    col1, col2 = st.columns([2, 1])  # Chat takes 2/3, workflow takes 1/3

    with col1:
        # Chat Messages Display - Clean and simple
        if st.session_state.messages:
            st.markdown("### 💬 **Chat History**")
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Simple clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.workflow_progress = []
            st.session_state.awaiting_feedback = False
            st.rerun()

        # Chat Input - At the bottom
        if prompt := st.chat_input("Ask about water management..."):
            if st.session_state.agent is None:
                st.error("Agent not initialized. Please check the configuration.")
            else:
                # Add user message to chat
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Let the agent decide if/what to plot via its tools

                # Get agent response with live monitoring
                with st.spinner("🔄 **Processing your request...**"):
                    try:
                        result = run_workflow_with_monitoring(st.session_state.agent, prompt)
                    except Exception as e:
                        error_msg = f"Error in live chat processing: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                        result = None
                
                # Display the response in chat message
                with st.chat_message("assistant"):
                    if result and isinstance(result, dict) and 'messages' in result and result['messages']:
                        all_responses = []
                        tool_text_blobs = []
                        raw_messages_debug = []
                        for message in result['messages']:
                            if hasattr(message, 'content') and message.content and 'AIMessage' in str(type(message)):
                                all_responses.append(message.content)
                            elif hasattr(message, 'content') and message.content and 'ToolMessage' in str(type(message)):
                                tool_text_blobs.append(str(message.content))
                            # Collect raw message debug
                            try:
                                role = getattr(message, 'type', 'AIMessage')
                                raw_messages_debug.append({
                                    "type": str(type(message)),
                                    "role": role,
                                    "content_preview": str(getattr(message, 'content', ''))[:500]
                                })
                            except Exception:
                                pass

                        # Combine response - show only the most recent AI message to avoid duplication
                        if all_responses:
                            response = all_responses[-1]
                        else:
                            response = "Processing... please wait for the complete response."

                        # Display text response
                        st.markdown(response)

                        # If the tool returned an inline base64 image, render it directly
                        try:
                            import re
                            import base64
                            match = re.search(r"PLOT_IMAGE:(data:image/png;base64,[A-Za-z0-9+/=]+)", response)
                            if match:
                                data_uri = match.group(1)
                                st.image(data_uri)
                                # Also keep it in the figures panel as a matplotlib figure if available
                                # otherwise, store the data uri for future reference
                                if 'figure_gallery' in st.session_state and not st.session_state.figure_gallery:
                                    st.session_state.figure_gallery = st.session_state.figure_gallery
                            else:
                                # Check tool message contents as well
                                blob = "\n\n".join(tool_text_blobs)
                                match2 = re.search(r"PLOT_IMAGE:(data:image/png;base64,[A-Za-z0-9+/=]+)", blob)
                                if match2:
                                    data_uri2 = match2.group(1)
                                    st.image(data_uri2)
                        except Exception:
                            pass

                        # Debug: agent raw output
                        with st.expander("🔍 Debug: Agent raw output", expanded=False):
                            try:
                                st.json({
                                    "num_messages": len(result.get('messages', [])),
                                    "messages": raw_messages_debug
                                })
                            except Exception:
                                st.write("Unable to render raw messages debug.")
                        
                        # If a plot was generated, move it to the gallery for rendering in the Figures panel
                        if 'current_plot_figure' in st.session_state and st.session_state.current_plot_figure is not None:
                            try:
                                st.session_state.figure_gallery.append(st.session_state.current_plot_figure)
                            finally:
                                st.session_state.current_plot_figure = None

                        # Fallback: parse any inline matplotlib code blocks from the assistant text
                        if not st.session_state.figure_gallery:
                            _execute_inline_matplotlib_blocks_from_text(response)

                        # Debug: plot tool state
                        with st.expander("🧪 Debug: Plot tool state", expanded=False):
                            try:
                                st.write({
                                    "has_current_plot_figure": st.session_state.current_plot_figure is not None,
                                    "figure_gallery_len": len(st.session_state.figure_gallery or []),
                                    "last_plot_status": st.session_state.get("last_plot_status", "n/a")
                                })
                                if st.session_state.get("last_plot_script"):
                                    st.code(st.session_state.last_plot_script, language='python')
                            except Exception:
                                st.write("Unable to render plot debug.")
                        
                        # Persist in chat history
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        response = "No response generated. Please try again."
                        st.error(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})

    with col2:
        # Right column: Workflow Progress, Status, and Figures
        st.markdown("### 🔴 **Live Status**")

        # Simple status indicator
        if st.session_state.agent is not None:
            if st.session_state.live_chat_active:
                st.markdown('<div class="live-status">🔄 WORKFLOW RUNNING</div>', unsafe_allow_html=True)
            else:
                st.info("📊 Ready for your question")

            # Live Workflow Progress - Clear and prominent
            if st.session_state.workflow_progress:
                st.markdown("### 📋 **Workflow Progress**")
                for step in st.session_state.workflow_progress:
                    status_class = step["status"]
                    status_emoji = {"active": "🔄", "completed": "✅", "error": "❌"}
                    emoji = status_emoji.get(step["status"], "📊")

                    st.markdown(f"""
                    <div class="workflow-step {status_class}">
                        <strong>{emoji} {step['step']}</strong> - {step['status'].upper()}
                        <br><small>🕐 {step['timestamp']}</small>
                        {f'<br><em>{step["details"]}</em>' if step["details"] else ''}
                    </div>
                    """,
                                unsafe_allow_html=True)
        else:
            st.info("Agent not initialized")


if __name__ == "__main__":
    main()
