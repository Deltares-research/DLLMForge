Tutorial LLM capabilities of DLLMForge
======================================

This tutorial demonstrates how to use a simple LLM to ask questions using DLLMForge.
There are different API's available to use LLMs from OpenAI, Mistral, AZUREOpenAI or Deltares hosted models.
There are three different ways to use LLMs using:

- :class:`~dllmforge.llamaindex_api.LlamaIndexAPI` - Containing LlamaIndex framework integration with OpenAI, AZUREOpenAI and Mistral models.
- :class:`~dllmforge.langchain_api.LangchainAPI` - Containing LangChain framework integration with OpenAI, AZUREOpenAI and Mistral models.
- :class:`~dllmforge.LLMs.Deltares_LLMs.DeltaresOllamaLLM` - Containing Deltares hosted models.

For the OpenAI, Mistral a .env file is needed with the API keys, these are specified in the following section.
For the Deltares hosted models, no API key is needed, but you need to be on the Deltares network or VPN.

Environment Setup
-----------------

To use the OpenAI model the following environment variables are needed in a .env file:

.. code-block:: text

    OPENAI_API_KEY=your_openai_api_key
    OPENAI_MODEL_NAME=gpt-4 # or other available models

.. code-block:: text

    AZURE_OPENAI_API_KEY=your_azure_openai_api_key
    AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
    AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
    AZURE_OPENAI_API_VERSION=your_api_version
    AZURE_OPENAI_MODEL_NAME=gpt-4 # or other available models

.. code-block:: text

    MISTRAL_API_KEY=your_mistral_api_key
    MISTRAL_MODEL_NAME=mistral-7b-instruct-v0.1 # or other available models


Initialize LLM
--------------

First, we initialize the LLM using one of the available API's.

Using LlamaIndex API with OpenAIAPI

.. code-block:: python

    from dllmforge.llamaindex_api import LlamaIndexAPI

    # Initialize OpenAI API
    api_llama_openai = LlamaIndexAPI(model_provider="openai")
    api_llama_mistral = LlamaIndexAPI(model_provider="mistral")
    api_llama_azure = LlamaIndexAPI(model_provider="azure-openai")

Using LangChain API with OpenAIAPI

.. code-block:: python

    from dllmforge.langchain_api import LangchainAPI

    # Initialize OpenAI API
    api_langchain_openai = LangchainAPI(model_provider="openai")
    api_langchain_mistral = LangchainAPI(model_provider="mistral")
    api_langchain_azure = LangchainAPI(model_provider="azure-openai")

Using Deltares hosted models

.. code-block:: python

    from dllmforge.LLMs.Deltares_LLMs import DeltaresOllamaLLM

    # Initialize Deltares hosted model
    base_url = "https://chat-api.directory.intra"
    model_name = "llama3.1:70b"  # or other available models
    llm = DeltaresOllamaLLM(base_url=base_url, model_name=model_name)

Using the LLM to ask questions
------------------------------

Now we can use the initialized LLM to ask questions.
All of the classes have a method `chat_completions` to ask a question to the LLM.

The method takes a dictionary of messages as input and returns the response from the LLM.
The dictionary of messages should contain a list of messages with the role and content.

.. code-block:: python

    # Define the messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]

Then the messages can be passed to the `chat_completions` method of the different API's.

.. code-block:: python

    # Using LlamaIndex API with OpenAI
    response_llama_openai = api_llama_openai.chat_completion(messages)
    print("LlamaIndex OpenAI Response:", response_llama_openai)

    # Using LangChain API with OpenAI
    response_langchain_openai = api_langchain_openai.chat_completion(messages)
    print("LangChain OpenAI Response:", response_langchain_openai)

    # Using Deltares hosted model
    response_deltares = llm.chat_completion(messages)
    print("Deltares Model Response:", response_deltares)

This tutorial demonstrates how to use a simple LLM to ask questions using DLLMForge.
There are different API's available to use LLMs from OpenAI, Mistral, AZUREOpenAI or Deltares hosted models.

Footnote about Deltares hosted models
-------------------------------------

Note that for the Deltares hosted models you can also define the temperature and max_tokens parameters in the `chat_completions` method.

The temperature parameter (between 0 and 1) controls the randomness of the output.
A low temperature (closer to 0) makes the output more focused, deterministic, and repetitive, as the model sticks to the most probable words, 
making it ideal for tasks like factual summaries. A high temperature (above 1) makes the output more random, creative, and varied, as the model 
is more likely to choose less likely words, which is better for creative writing or brainstorming.  

The max_tokens parameter controls the maximum length of the output in tokens. It can be used to limit the response length and ensure that the 
output fits within specific constraints.

To turn off the thinking process of the model, you can add the "/no_think" flag to the messages.



