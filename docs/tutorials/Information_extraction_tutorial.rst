=========================================
Information Extraction with LLMs Tutorial
=========================================

This tutorial demonstrates how to build a complete information extraction (IE) pipeline using Large Language Models (LLMs) from the DLLMForge library. The pipeline uses LLMs to automatically extract structured data from unstructured documents like PDFs, research papers, and technical reports.

For a full end-to-end code example, see the Jupyter notebook: `information_extration.ipynb <../../workflows/information_extration.ipynb>`_

Overview
========

LLM-based information extraction transforms unstructured documents into structured, queryable data. Instead of manually reading through hundreds of pages to collect specific information, you can use LLMs to automatically identify and extract the data you need into a predefined schema.

The Information Extraction Workflow
====================================

The DLLMForge IE pipeline consists of three main stages, each supported by specialized components:

**Stage 1: Schema Definition**
   Define what information you want to extract using Pydantic models. You can either define your own schema manually or use :class:`~dllmforge.IE_agent_schema_generator.SchemaGenerator` to automatically generate one based on your task description. The schema defines the structure of data you want to extract, including field names, types, and descriptions. If you already have a schema, save it as a .py file and skip to Stage 2.

**Stage 2: Document Processing**
   Convert documents (currently supported: PDFs, docx, csv, images) into LLM-readable format. The :class:`~dllmforge.IE_agent_document_processor.DocumentProcessor` handles this automatically within the extraction pipeline:
   
   - Text extraction: Converts PDFs and documents to plain text (faster, lower cost)
   - Image extraction: Converts document pages to images (better for complex layouts, diagrams). Note: Requires multimodal LLM.
   
   **Note:** Document processing is handled automatically by the :class:`~dllmforge.IE_agent_extractor.InfoExtractor` - you configure it but don't need to run it separately.

**Stage 3: LLM Extraction**
   Extract structured data matching your schema using :class:`~dllmforge.IE_agent_extractor.InfoExtractor`. The extractor orchestrates the entire process: it processes documents using the configured :class:`~dllmforge.IE_agent_document_processor.DocumentProcessor`, manages document chunking for large files, and coordinates LLM interactions. For large documents, text is split into chunks and processed separately, with results aggregated at the end.

**When to Use Text vs. Image Extraction:**

- **Use text extraction when:**
  
  - Documents have simple, linear text structure
  - You're extracting primarily textual information
  - Documents are text-based PDFs or docx files

- **Use image extraction when:**
  
  - Documents contain complex tables or diagrams
  - You're working with scanned documents
  - Text extraction produces poor results

Step-by-Step Implementation
============================

1. Import Required Modules
---------------------------

Start by importing all necessary components:

.. code-block:: python
    
    from dllmforge.IE_agent_schema_generator import SchemaGenerator
    from dllmforge.IE_agent_document_processor import DocumentProcessor
    from dllmforge.IE_agent_extractor import InfoExtractor
    from dllmforge.langchain_api import LangchainAPI
    from pathlib import Path
    import importlib.util
    import re
    import json

2. Define Output Schema
-----------------------

The first step is to define what information you want to extract. You have two options:

**Option A: Define Your Own Schema**

If you know exactly what structure you need, create a Pydantic schema manually and save it as a .py file.

Here's an example schema for extracting machine learning model hyperparameters from research papers. Each field is optional and includes a description:

.. code-block:: python

    # Save this as: my_custom_schema.py
    from pydantic import BaseModel, Field
    from typing import Optional, List
    
    class ModelHyperparameters(BaseModel):
        model_type: Optional[str] = Field(None, description="Type of model")
        num_layers: Optional[int] = Field(None, description="Number of layers")
        learning_rate: Optional[float] = Field(None, description="Learning rate")
        batch_size: Optional[int] = Field(None, description="Batch size")
        epochs: Optional[int] = Field(None, description="Number of epochs")

Once you've saved your schema, import and use it directly in your extraction code:

.. code-block:: python

    from my_custom_schema import ModelHyperparameters
    SchemaClass = ModelHyperparameters

If you use this approach, skip to step 3 (Understanding Document Processing).

**Option B: Automated Schema Generation**

Let the LLM automatically generate a schema based on your task description. This is useful when you're not sure about the exact structure or want to explore what fields to extract.

In this example, we'll use the LLM to generate a schema for extracting machine learning model hyperparameters from research papers:

.. code-block:: python

    # Define your extraction task
    schema_task_description = (
        "Generate a Pydantic schema class named ModelHyperparameters to extract machine learning model hyperparameters from research papers and documentation. "
        "The schema should capture: model architecture details (type, layers, neurons, etc.), "
        "training parameters (learning rate, batch size, epochs), "
        "optimization settings (optimizer, loss function), regularization techniques (dropout, etc.)."
    )
    
    # Prepare output directory for the schema
    schema_dir = Path("generated_schemas")
    schema_dir.mkdir(exist_ok=True)
    schema_file = schema_dir / "model_hyperparameters.py"
    
    # Create schema generator with direct arguments (no config object needed)
    schema_generator = SchemaGenerator(
        task_description=schema_task_description,
        output_path=str(schema_file)
    )
    
    # Generate the schema using LLM
    schema_code = schema_generator.generate_schema()
    
    print("Generated Schema:")
    print(schema_code)
    
    # Save the schema to file
    schema_generator.save_schema(schema_code)

The generated schema will look something like this:

.. code-block:: python

    from pydantic import BaseModel, Field
    from typing import Optional, List
    
    class ModelHyperparameters(BaseModel):
        model_type: Optional[str] = Field(None, description="Type of model (e.g., LSTM, CNN, Transformer)")
        num_layers: Optional[int] = Field(None, description="Number of layers in the model")
        hidden_units: Optional[int] = Field(None, description="Number of hidden units/neurons")
        learning_rate: Optional[float] = Field(None, description="Learning rate for training")
        batch_size: Optional[int] = Field(None, description="Batch size for training")
        epochs: Optional[int] = Field(None, description="Number of training epochs")
        optimizer: Optional[str] = Field(None, description="Optimization algorithm used")
        loss_function: Optional[str] = Field(None, description="Loss function used")
        dropout_rate: Optional[float] = Field(None, description="Dropout rate for regularization")

**Load the Generated Schema Dynamically**

After generating the schema, we need to load it as a Python class. This code extracts the class name from the generated schema and dynamically imports it:

.. code-block:: python

    # Extract the class name from the generated code
    class_matches = re.finditer(r"class\s+(\w+)\s*\(", schema_code)
    class_names = [match.group(1) for match in class_matches]
    
    if not class_names:
        raise ValueError("Could not find any class names in generated schema")
    
    schema_class_name = class_names[-1]  # Get the last class (usually the main one)
    
    # Dynamically import the schema
    spec = importlib.util.spec_from_file_location("model_hyperparameters", schema_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Get the schema class
    if not hasattr(module, schema_class_name):
        raise ValueError(f"Generated schema does not contain class {schema_class_name}")
    
    SchemaClass = getattr(module, schema_class_name)
    
    print(f"Successfully loaded schema class: {schema_class_name}")

**Alternative: Using Example Documents**

You can improve schema generation by providing an example document. The LLM will analyze the document's content and structure to create a more tailored schema:

.. code-block:: python

    # With example document (text or file path)
    example_text = """
    We trained an LSTM model with 3 layers and 128 hidden units.
    The learning rate was set to 0.001 with a batch size of 32.
    Training ran for 50 epochs using the Adam optimizer.
    """
    
    # Create schema generator with example document (no config object needed)
    schema_generator = SchemaGenerator(
        task_description="Extract model hyperparameters from this paper",
        example_doc=example_text,  # Can also be a file path to PDF
        output_path="generated_schemas/hyperparameters_from_example.py"
    )
    
    schema_code = schema_generator.generate_schema()
    schema_generator.save_schema(schema_code)

3. Understanding Document Processing
-------------------------------------

Before extraction can begin, documents must be converted into a format that LLMs can process. The ``DocumentProcessor`` component handles this conversion, supporting both text and image extraction modes.

**Important:** The ``InfoExtractor`` (covered in Section 5) handles document processing automatically. You configure the ``DocumentProcessor`` and pass it to the ``InfoExtractor``, which then uses it internally. You typically don't need to call document processing methods directly.

**How Document Processing Works**

The ``DocumentProcessor`` can convert documents in two ways:

1. **Text extraction**: Extracts text from PDFs, docx, csv files (faster, lower cost)
2. **Image extraction**: Converts document pages to images (better for complex layouts, requires multimodal LLM)

**Configuring Document Processing**

You configure the ``DocumentProcessor`` by passing parameters directly (no config object needed):

.. code-block:: python

    # Define your document directories
    document_input_dir = r"path/to/your/documents"
    document_output_dir = r"path/to/output"
    
    # Configure document processor for text or image extraction (direct arguments)
    doc_processor = DocumentProcessor(
        input_dir=document_input_dir,
        file_pattern="*.pdf",              # Pattern to match files
        output_type="text",                # "text" or "image" extraction, default is "text"
        output_dir=document_output_dir
    )

You'll pass this configured processor to the ``InfoExtractor`` in Section 5, where it will be used automatically during extraction.

1. Initialize the LLM
---------------------

Configure the LLM that will perform the extraction. DLLMForge supports multiple providers through a unified API:

.. code-block:: python

    # Example using Azure OpenAI
    llm_api = LangchainAPI(
        model_provider="azure-openai",
        temperature=0.1  # Low temperature for consistent, factual extraction
    )
    
    # Example: Using OpenAI
    # llm_api = LangchainAPI(
    #     model_provider="openai",
    #     model_name="gpt-4",
    #     temperature=0.1
    # )

**Temperature Settings:**

- ``0.0-0.2``: Most deterministic, best for factual extraction
- ``0.3-0.5``: Balanced, allows some interpretation
- ``0.6-1.0``: More creative, may introduce inconsistencies (not recommended for IE)

1. Extract Information
----------------------

Now we can create the ``InfoExtractor`` to orchestrate the entire extraction pipeline. The ``InfoExtractor`` will use the ``DocumentProcessor`` we configured in Step 3 to automatically handle document processing, then perform the LLM-based extraction.

**Create InfoExtractor**

.. code-block:: python

    # Define extraction parameters
    system_prompt = "Extract model hyperparameters from research paper."
    chunk_size = 80000      # Maximum characters per chunk
    chunk_overlap = 10000   # Overlap between chunks to preserve context
    
    # Create InfoExtractor - it will use doc_processor internally
    extractor = InfoExtractor(
        output_schema=SchemaClass,              # The Pydantic schema we generated
        llm_api=llm_api,                        # The LLM instance
        system_prompt=system_prompt,            # Instruction for the LLM
        chunk_size=chunk_size,                  # For splitting large documents
        chunk_overlap=chunk_overlap,            # Context preservation
        doc_processor=doc_processor,            # Document processor from Step 3
        document_output_type="text"             # Must match doc_processor output_type
    )

The ``InfoExtractor`` now has everything it needs: a schema to structure the data, an LLM to do the extraction, and a document processor to convert files into LLM-readable format.

**Understanding Chunk Parameters:**

When documents exceed the LLM's context window, they're split into chunks:

- ``chunk_size``: Maximum characters per chunk (80,000 is typical for GPT-4. For other models, check with LLM provider)
- ``chunk_overlap``: Characters shared between consecutive chunks (preserves context across boundaries)

**Extract from Single Document**

Now you can extract information with a single call. The ``InfoExtractor`` handles document processing automatically:

.. code-block:: python

    # Define the document path
    single_doc_path = Path(document_input_dir) / "lstm_low_flow.pdf"
    
    # Process and extract in one step - InfoExtractor handles everything
    doc = extractor.doc_processor.process_file(single_doc_path)  # Converts PDF to text
    results = extractor.process_document(doc)                     # Extracts information
    
    # View results
    print(f"\nExtracted {len(results)} result(s)")
    
    if results:
        print("\nFirst result:")
        print(json.dumps(results[0], indent=2))
    
    # Save results to JSON
    output_path = Path(document_output_dir) / "lstm_low_flow_extracted.json"
    extractor.save_results(results, output_path)
    
    print(f"\nResults saved to {output_path}")

Behind the scenes, the ``InfoExtractor``:

1. Uses ``doc_processor.process_file()`` to convert the PDF to text
2. Chunks the text if it's too large for the LLM
3. Sends each chunk to the LLM with the extraction prompt
4. Validates results against your Pydantic schema
5. Aggregates and returns all extracted data

**Expected Output:**

.. code-block:: json

    {
      "model_type": "LSTM",
      "num_layers": 3,
      "hidden_units": 128,
      "learning_rate": 0.001,
      "batch_size": 32,
      "epochs": 100,
      "optimizer": "Adam",
      "loss_function": "MSE",
      "dropout_rate": 0.2
    }

**Batch Processing Multiple Documents**

To extract from all documents in a directory:

.. code-block:: python

    # Process all documents in the directory
    extractor.process_all()
    
    print(f"Batch extraction complete! Check output directory: {document_output_dir}")

The ``process_all()`` method automatically:

1. Finds all files matching the pattern
2. Processes each document
3. Extracts information using the schema
4. Saves individual JSON files for each document

Understanding the Results
=========================

**Result Structure**

Each extraction result is a dictionary that matches your Pydantic schema. The structure includes:

- **Field names**: As defined in your schema
- **Values**: Extracted from the document
- **None values**: For optional fields not found in the document

**Handling Missing or Optional Fields**

The LLM will return ``None`` for optional fields it cannot find:

.. code-block:: python

    # Check if a field was found
    if results[0].get('dropout_rate') is not None:
        print(f"Dropout rate: {results[0]['dropout_rate']}")
    else:
        print("Dropout rate not mentioned in document")

**Validation and Error Handling**

Pydantic automatically validates the extracted data against your schema:

.. code-block:: python

    try:
        # Results are already validated against the schema
        for result in results:
            # Access fields with confidence they match the schema
            model_type = result.get('model_type', 'Unknown')
            learning_rate = result.get('learning_rate', 0.0)
            print(f"Model: {model_type}, LR: {learning_rate}")
    except Exception as e:
        print(f"Validation error: {e}")

**Exporting to Different Formats**

Convert results to CSV for analysis:

.. code-block:: python

    import pandas as pd
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    csv_path = Path(document_output_dir) / "extracted_hyperparameters.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"Results exported to CSV: {csv_path}")
    print(df.head())

Advanced Configuration
======================

Manual Document Processing (Advanced)
--------------------------------------

In most cases, you'll let ``InfoExtractor`` handle document processing automatically. However, for advanced use cases, you can process documents manually for more control.

**When to Use Manual Processing:**

- You need to inspect processed documents before extraction
- You want to cache processed documents for multiple extraction runs
- You're building a custom pipeline with additional processing steps

**Manual Text Processing Example**

.. code-block:: python

    from dllmforge.IE_agent_document_processor import DocumentProcessor
    from pathlib import Path
    
    # Create document processor (direct arguments, no config object needed)
    doc_processor = DocumentProcessor(
        input_dir=document_input_dir,
        file_pattern="*.pdf",
        output_type="text",
        output_dir=document_output_dir
    )
    
    # Process a single document manually
    single_doc_path = Path(document_input_dir) / "research_paper.pdf"
    processed_doc = doc_processor.process_file(single_doc_path)
    
    # Inspect the processed document
    print(f"Document processed: {processed_doc.metadata['source_file']}")
    print(f"Content type: {processed_doc.content_type}")
    print(f"Text length: {processed_doc.metadata['text_length']} characters")
    print(f"\nFirst 500 characters:")
    print(processed_doc.content[:500])
    
    # Now use it with InfoExtractor
    results = extractor.process_document(processed_doc)

**Process Multiple Documents Manually**

.. code-block:: python

    # Process all PDFs in the directory
    all_processed_docs = doc_processor.process_directory()
    
    print(f"Processed {len(all_processed_docs)} documents")
    
    # Inspect each document
    for doc in all_processed_docs:
        print(f"- {doc.metadata['source_file']}: {doc.metadata['text_length']} chars")
    
    # Extract from each document
    for doc in all_processed_docs:
        results = extractor.process_document(doc)
        # Save results...

**Manual Image Processing**

For documents with complex layouts, you can process to images manually:

.. code-block:: python

    # Create image processor (direct arguments, no config object needed)
    image_processor = DocumentProcessor(
        input_dir=document_input_dir,
        file_pattern="*.pdf",
        output_type="image",
        output_dir=document_output_dir
    )
    
    # Process PDF to images (one image per page)
    processed_images = image_processor.process_file(single_doc_path)
    
    print(f"Number of pages: {len(processed_images)}")
    
    # Inspect each page
    for img_doc in processed_images:
        page_num = img_doc.metadata['page_number']
        size_kb = img_doc.metadata['image_size_bytes'] / 1024
        print(f"Page {page_num}: {size_kb:.1f} KB")
    
    # Extract from images using multimodal LLM
    multimodal_extractor = InfoExtractor(
        output_schema=SchemaClass,
        llm_api=multimodal_llm,  # Must support vision
        system_prompt="Extract information from document images.",
        doc_processor=image_processor,
        document_output_type="image"
    )
    
    results = multimodal_extractor.process_document(processed_images)

**Understanding Document Metadata**

Each processed document includes metadata for tracking:

- ``source_file``: Original file path
- ``file_type``: File extension (.pdf, .docx, etc.)
- ``text_length``: Number of characters extracted (text mode)
- ``page_number``: Page number (image mode)
- ``image_size_bytes``: Image size in bytes (image mode)

Improving Extraction Quality
-----------------------------

**Crafting Effective System Prompts**

The system prompt guides the LLM's extraction behavior. Be specific and clear:

.. code-block:: python

    # Basic prompt
    basic_prompt = "Extract model hyperparameters."
    
    # Better: More specific prompt
    better_prompt = """Extract machine learning model hyperparameters from the research paper.
    Focus on: model architecture, training parameters, and optimization settings.
    Only extract explicitly stated values, do not infer or estimate."""
    
    # Best: Very detailed prompt with examples
    detailed_prompt = """You are extracting model hyperparameters from a research paper.
    
    Extract the following information:
    - Model architecture: type, number of layers, hidden units
    - Training: learning rate, batch size, number of epochs
    - Optimization: optimizer name, loss function
    - Regularization: dropout rate, weight decay
    
    Rules:
    1. Only extract values explicitly mentioned in the text
    2. Use None for fields not found
    3. Preserve exact numeric values and units
    4. Use standard naming conventions (e.g., "Adam" not "adam optimizer")
    """
    
    extractor = InfoExtractor(
        output_schema=SchemaClass,
        llm_api=llm_api,
        system_prompt=detailed_prompt,  # Use the detailed prompt
        chunk_size=80000,
        chunk_overlap=10000,
        doc_processor=doc_processor,
        document_output_type="text"
    )

**Using Example Documents for Schema Generation**

Provide example documents to help the LLM understand the data structure:

.. code-block:: python

    # Load an example document
    example_pdf_path = Path("examples/sample_paper.pdf")
    
    # Create schema generator with example PDF (direct arguments, no config object needed)
    schema_generator = SchemaGenerator(
        task_description="Extract model hyperparameters from ML research papers",
        example_doc=str(example_pdf_path),  # Provide example PDF
        output_path="generated_schemas/ml_hyperparameters.py"
    )
    
    schema_code = schema_generator.generate_schema()

The LLM will analyze the example document's structure and create a more tailored schema.

**Adjusting LLM Temperature**

Temperature affects extraction consistency:

.. code-block:: python

    # Very deterministic (recommended for most IE tasks)
    strict_llm = LangchainAPI(
        model_provider="azure-openai",
        temperature=0.0
    )
    
    # Slightly more flexible (for nuanced interpretation)
    flexible_llm = LangchainAPI(
        model_provider="azure-openai",
        temperature=0.3
    )

Choosing Between Text and Image Extraction
-------------------------------------------

The choice between text and image extraction depends on your documents and what information you need to extract. You configure this choice in Step 3 when setting up the ``DocumentProcessor``.

**Text Extraction Workflow**

Best for documents with linear text structure:

.. code-block:: python

    # Step 3: Configure for text extraction (direct arguments, no config object needed)
    text_processor = DocumentProcessor(
        input_dir="documents/research_papers",
        file_pattern="*.pdf",
        output_type="text",              # Key configuration
        output_dir="output/text"
    )
    
    # Step 4: Initialize LLM
    llm_api = LangchainAPI(
        model_provider="azure-openai",
        temperature=0.1
    )
    
    # Step 5: Create extractor
    text_extractor = InfoExtractor(
        output_schema=SchemaClass,
        llm_api=llm_api,
        system_prompt="Extract information from the text.",
        chunk_size=80000,
        chunk_overlap=10000,
        doc_processor=text_processor,
        document_output_type="text"      # Must match processor
    )

**Use text extraction for:**

- Research papers with standard formatting
- Technical reports without complex layouts
- Documents where text is the primary content
- When you need faster, lower-cost processing

**Image Extraction Workflow**

Best for documents with complex visual elements:

.. code-block:: python

    # Step 3: Configure for image extraction (direct arguments, no config object needed)
    image_processor = DocumentProcessor(
        input_dir="documents/forms_and_tables",
        file_pattern="*.pdf",
        output_type="image",             # Key configuration
        output_dir="output/images"
    )
    
    # Step 4: Initialize multimodal LLM (REQUIRED for images)
    multimodal_llm = LangchainAPI(
        model_provider="openai",
        model_name="gpt-4",
        temperature=0.1
    )
    
    # Step 5: Create extractor
    image_extractor = InfoExtractor(
        output_schema=SchemaClass,
        llm_api=multimodal_llm,
        system_prompt="Extract information from the document images.",
        doc_processor=image_processor,
        document_output_type="image"     # Must match processor
    )

**Use image extraction for:**

- Documents with complex tables and charts
- Scanned documents or poor-quality PDFs
- Forms with specific visual layouts
- When text extraction misses critical information

**Important:** Image extraction requires a multimodal LLM that supports vision and incurs higher API costs.

**Hybrid Approach: Try Text First, Then Images**

For challenging documents, you can try text extraction first and fall back to images if needed:

.. code-block:: python

    # First attempt: text extraction (faster, cheaper)
    text_doc = text_processor.process_file(document_path)
    text_results = text_extractor.process_document(text_doc)
    
    # Check if results are satisfactory
    def results_complete(results):
        """Check if critical fields were extracted"""
        critical_fields = ['model_type', 'learning_rate', 'batch_size']
        return all(results[0].get(field) is not None for field in critical_fields)
    
    # If incomplete, try image extraction
    if not results_complete(text_results):
        print("Text extraction incomplete, trying image extraction...")
        image_doc = image_processor.process_file(document_path)
        image_results = image_extractor.process_document(image_doc)
        
        # Use image results if better
        final_results = image_results if results_complete(image_results) else text_results
    else:
        final_results = text_results

