Tutorial: LLM capabilities of DLLMForge
======================================

This tutorial shows how to use DLLMForge to build a tiny tool-using agent and ask questions via different LLM providers (Azure OpenAI, OpenAI, Mistral, or Deltares-hosted models). You will learn how to configure providers with environment variables, run the ultra-simple pizza agent, and switch providers without changing your business logic.

What you will use
------------------

- :class:`~dllmforge.langchain_api.LangchainAPI` — LangChain integration for Azure OpenAI, OpenAI, and Mistral.
- :class:`~dllmforge.llamaindex_api.LlamaIndexAPI` — LlamaIndex integration for the same providers.
- :class:`~dllmforge.LLMs.Deltares_LLMs.DeltaresOllamaLLM` — Deltares-hosted models (no external API key required; Deltares network/VPN needed).

Prerequisites
-------------

1. Python environment with the project requirements installed.
2. A ``.env`` file in your project root with provider credentials (see below).
3. Optional: IPython/Jupyter if you want to display the LangGraph diagram.

Environment setup
-----------------

Create or update your ``.env`` with the variables for the providers you plan to use.

Azure OpenAI (default in DLLMForge examples)::

    AZURE_OPENAI_ENDPOINT=https://your-azure-endpoint
    AZURE_OPENAI_API_KEY=your_azure_openai_api_key
    AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
    AZURE_OPENAI_API_VERSION=2024-12-01-preview

OpenAI::

    OPENAI_API_KEY=your_openai_api_key
    OPENAI_MODEL_NAME=gpt-4.1  # or gpt-4o, etc.

Mistral::

    MISTRAL_API_KEY=your_mistral_api_key
    MISTRAL_MODEL_NAME=mistral-large-latest

Deltares-hosted (no API key; requires Deltares network/VPN)::

    # No keys required; you will specify base_url and model at runtime

Quick start: the ultra-simple pizza agent
----------------------------------------

The file ``pizza_course_tutorial_ultra_simple.py`` defines a minimal agent with a few tools (arithmetic, pizza pricing, and a summary generator that calls the configured LLM). It uses :class:`~dllmforge.agent_core.SimpleAgent` under the hood, which wires up a small LangGraph with optional tool routing.

Key elements
~~~~~~~~~~~~

- Tools are created with the DLLMForge ``@tool`` decorator (a thin wrapper over LangChain tools).
- The agent is created once and tools are attached.
- Provider selection is controlled by the ``model_provider`` argument and your ``.env`` file.

Example (excerpt)::

    from dllmforge.agent_core import SimpleAgent, tool

    # Basic math tools
    @tool
    def add(a: float, b: float) -> float:
        """Add two numbers together."""
        return a + b

    @tool
    def subtract(a: float, b: float) -> float:
        """Subtract two numbers from each other."""
        return a - b

    @tool
    def multiply(a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b

    @tool
    def divide(a: float, b: float) -> float:
        """Divide two numbers."""
        return a / b

    # Pizza pricing tool
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

    # LLM-powered summary tool
    @tool
    def make_summary(question: str, result: str) -> str:
        """Use the configured LLM to create a concise, conversational summary."""
        from dllmforge.langchain_api import LangchainAPI
        llm_api = LangchainAPI()  # honors your .env + default provider
        messages = [
            ("system", "You are a helpful assistant. Create a concise, friendly summary of the provided result in the context of the question. Mention all of the tools that you used"),
            ("human", f"Question:\n{question}\n\nResult:\n{result}\n\nPlease return a brief, conversational summary (1-3 sentences).")
        ]
        response = llm_api.llm.invoke(messages)
        return getattr(response, "content", str(response))

    agent = SimpleAgent(
        "You are a helpful assistant that can do math and tell you about pizza prices. Only use the tools, do not try to do maths in your head.",
        model_provider="azure-openai",
        temperature=0.1,
    )
    agent.add_tool(make_summary)
    agent.add_tool(divide)
    agent.add_tool(multiply)
    agent.add_tool(add)
    agent.add_tool(subtract)
    agent.add_tool(get_pizza_price)
    agent.compile()

Run it::

    python pizza_course_tutorial_ultra_simple.py

Testing the agent
~~~~~~~~~~~~~~~~~

Once your agent is compiled, you can test it with various queries. Here's an example that demonstrates the agent's ability to handle complex calculations involving pizza prices and fractions::

    agent.process_query("If I had half a pepperoni pizza and 1/4 of a margherita pizza and I paid for both pizzas. How much should I Tikkie my friend?", stream=True)

This query will:
1. Calculate the price of a pepperoni pizza (15.99)
2. Calculate the price of a margherita pizza (12.99)
3. Determine half of the pepperoni price (7.995)
4. Determine 1/4 of the margherita price (3.2475)
5. Calculate the total amount your friend owes (11.2425)
6. Generate a friendly summary using the LLM

The agent uses its maths and pizza tools for calculations and the summary tool to provide a conversational response.

Switching providers
-------------------

To change providers, keep your code the same and switch the agent argument and/or your ``.env``.

Azure OpenAI (default)::

    agent = SimpleAgent(system_message, model_provider="azure-openai", temperature=0.1)

OpenAI::

    agent = SimpleAgent(system_message, model_provider="openai", temperature=0.1)
    # Requires OPENAI_API_KEY and OPENAI_MODEL_NAME

Mistral::

    agent = SimpleAgent(system_message, model_provider="mistral", temperature=0.1)
    # Requires MISTRAL_API_KEY and MISTRAL_MODEL_NAME

Using alternative integration layers
------------------------------------

You can also work directly with the integration classes if you prefer those patterns.

LlamaIndex API::

    from dllmforge.llamaindex_api import LlamaIndexAPI
    api_llama_openai = LlamaIndexAPI(model_provider="openai")
    api_llama_mistral = LlamaIndexAPI(model_provider="mistral")
    api_llama_azure   = LlamaIndexAPI(model_provider="azure-openai")

LangChain API::

    from dllmforge.langchain_api import LangchainAPI
    api_lc_openai = LangchainAPI(model_provider="openai")
    api_lc_mistral = LangchainAPI(model_provider="mistral")
    api_lc_azure   = LangchainAPI(model_provider="azure-openai")

Deltares-hosted models::

    from dllmforge.LLMs.Deltares_LLMs import DeltaresOllamaLLM
    base_url = "https://chat-api.directory.intra"
    model_name = "llama3.1:70b"
    llm = DeltaresOllamaLLM(base_url=base_url, model_name=model_name)

Asking questions (integration classes)
--------------------------------------

All integration classes accept message lists and return responses from the model.

Messages example::

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]

Examples::

    # LlamaIndex
    response_llama = api_llama_openai.chat_completions(messages)
    print("LlamaIndex OpenAI Response:", response_llama)

    # LangChain
    response_lc = api_lc_openai.chat_completions(messages)
    print("LangChain OpenAI Response:", response_lc)

    # Deltares-hosted
    response_deltares = llm.chat_completions(messages)
    print("Deltares Model Response:", response_deltares)

Notes on temperature and max tokens (Deltares-hosted)
-----------------------------------------------------

For Deltares-hosted models you may pass parameters like ``temperature`` and ``max_tokens`` to control creativity and output length. A lower temperature (e.g., 0.0–0.3) yields more deterministic outputs; higher values produce more diverse outputs. Use ``max_tokens`` to bound response size for summaries and UI constraints.

Tip: To disable extended “thinking” modes where supported, pass a ``/no_think`` flag in the user content convention used by your deployment.

Next steps
----------

- Add more domain tools and observe how the agent routes between model replies and tool calls.
- Swap providers to compare cost, latency, and quality for your workload.
- Explore the richer examples in the repo for RAG and multi-tool workflows.


