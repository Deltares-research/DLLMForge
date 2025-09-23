#TODO: add async version of this module
"""
Synchronous Information Extractor module for extracting structured information from documents using LLM with Docling.
"""
import os
import json
from typing import List, Dict, Any, Optional, Union, Generator
from pathlib import Path
from pydantic import BaseModel
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers.json import parse_json_markdown
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_api import LangchainAPI
from IE_agent_config import IEAgentConfig, ExtractorConfig
import base64

# Docling imports
from docling.document_converter import DocumentConverter


class DoclingProcessedDocument:
    """Class representing a document processed by Docling"""

    def __init__(self,
                 content: Union[str, bytes],
                 content_type: str,
                 metadata: Optional[Dict[str, Any]] = None,
                 docling_result=None):
        self.content = content
        self.content_type = content_type
        self.metadata = metadata or {}
        self.docling_result = docling_result  # Store the full Docling result for advanced features


class DocumentChunk:
    """Class representing a chunk of document content"""

    def __init__(self,
                 content: Union[str, bytes],
                 content_type: str,
                 metadata: Optional[Dict[str, Any]] = None,
                 docling_elements: Optional[List] = None):
        self.content = content
        self.content_type = content_type
        self.metadata = metadata or {}
        self.docling_elements = docling_elements or []  # Store Docling elements for structure awareness


class DoclingDocumentProcessor:
    """Document processor using Docling for advanced PDF processing"""

    def __init__(self, config):
        self.config = config

        # Use simple DocumentConverter without problematic pipeline options
        # The default configuration should work fine for most use cases
        self.converter = DocumentConverter()

    def encode_image_base64(self, image_data: bytes) -> str:
        """Encode image data to base64 string"""
        return base64.b64encode(image_data).decode('utf-8')

    def process_document(self, file_path: Path) -> Optional[DoclingProcessedDocument]:
        """Process a single document using Docling"""
        try:
            # Convert document using Docling
            result = self.converter.convert(str(file_path))
            # Create metadata from Docling result
            metadata = {
                'source_file': str(file_path),
                'file_name': file_path.name,
                'file_type': file_path.suffix.lower(),
                'num_pages': len(result.document.pages) if hasattr(result.document, 'pages') else 1,
                'num_tables': len(result.document.tables),
                'num_figures': len(result.document.pictures),
                'processing_method': 'docling'
            }

            # Extract structured content
            content_parts = []
            for text_value in result.document.texts:
                sub_metadata = metadata.copy()
                sub_metadata['page_no'] = text_value.prov[0].page_no
                doc_chunk = DocumentChunk(content=text_value.text, content_type="text", metadata=sub_metadata)
                content_parts.append(doc_chunk)

            # Extract tables if any
            for table in result.document.tables:
                table_md = table.export_to_markdown()
                sub_metadata = metadata.copy()
                sub_metadata['page_no'] = table.prov[0].page_no
                content_parts.append(
                    DocumentChunk(content=f"\n\n## Table\n{table_md}", content_type="text", metadata=sub_metadata))
                content_parts.append(f"\n\n## Table\n{table_md}")

            # Combine all content
            full_content = content_parts

            # Add document metadata if available
            if hasattr(result.document, 'meta') and result.document.meta:
                metadata.update({
                    'title': getattr(result.document.meta, 'title', ''),
                    'author': getattr(result.document.meta, 'author', ''),
                    'subject': getattr(result.document.meta, 'subject', ''),
                })

            return DoclingProcessedDocument(content=full_content,
                                            content_type='chunks',
                                            metadata=metadata,
                                            docling_result=result)

        except Exception as e:
            print(f"Error processing document {file_path}: {e}")
            return None

    def process_directory(self) -> List[DoclingProcessedDocument]:
        """Process all documents in the configured directory"""
        input_dir = Path(self.config.document.input_dir)
        if not input_dir.exists():
            print(f"Input directory does not exist: {input_dir}")
            return []

        pattern = self.config.document.file_pattern
        files = list(input_dir.glob(pattern))

        if not files:
            print(f"No files found matching pattern {pattern} in {input_dir}")
            return []

        processed_docs = []
        for file_path in files:
            print(f"Processing: {file_path}")
            doc = self.process_document(file_path)
            if doc:
                processed_docs.append(doc)

        return processed_docs


class DoclingInfoExtractor:
    """Class for extracting information from documents using LLM with Docling preprocessing"""

    def __init__(self, config: IEAgentConfig, output_schema: type[BaseModel], llm_api: Optional[LangchainAPI] = None):
        """Initialize the information extractor"""
        self.config = config
        self.output_schema = output_schema
        self.llm_api = llm_api or LangchainAPI()
        self.output_parser = PydanticOutputParser(pydantic_object=output_schema)
        self.doc_processor = DoclingDocumentProcessor(config)
        self.system_prompt = self.refine_system_prompt(config.schema.task_description)

    def refine_system_prompt(self, task_description: str) -> str:
        """Use LLM to refine user's task description into a proper system prompt"""
        system_template = """You are an expert at creating clear and effective system prompts for LLMs.
        Your task is to refine a user's task description into a well-structured system prompt.
        
        Guidelines for prompt refinement:
        1. Maintain the core objective of the task
        2. Add clear instructions and constraints
        3. Include relevant context and examples if needed
        4. Structure the prompt in a logical order
        5. Use clear, unambiguous language
        6. Consider that the input will be processed by Docling for better structure recognition
        """

        human_template = """Please refine this task description into a proper system prompt:
        
        {task_description}
        
        Create a well-structured system prompt that will guide the LLM in extracting information
        according to the task requirements. The input will be preprocessed by Docling, which means
        tables, figures, and document structure will be well-preserved in markdown format.
        Be thorough but concise.
        """

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])

        messages = prompt.format_messages(task_description=task_description)
        response = self.llm_api.chat_completion(messages)
        refined_prompt = response["response"] if response else task_description
        print(f"Refined system prompt: {refined_prompt}")
        return refined_prompt

    def chunk_document(self, doc: DoclingProcessedDocument) -> Generator[DocumentChunk, None, None]:
        """Split document into chunks based on Docling structure if needed"""
        content = doc.content
        chunk_size = self.config.extractor.chunk_size
        overlap = self.config.extractor.chunk_overlap

        # For Docling-processed documents, we can be smarter about chunking
        # by respecting document structure (sections, tables, etc.)

        if len(content) <= chunk_size:
            # Document is small enough, return as single chunk
            yield DocumentChunk(
                content=content,
                content_type='text',
                metadata=doc.metadata,
                docling_elements=[]  # Could extract specific elements here
            )
            return

        # Smart chunking based on markdown sections
        lines = content.split('\n')
        current_chunk = []
        current_size = 0

        for line in lines:
            line_size = len(line) + 1  # +1 for newline

            # If adding this line would exceed chunk size and we have content
            if current_size + line_size > chunk_size and current_chunk:
                # Yield current chunk
                chunk_content = '\n'.join(current_chunk)
                yield DocumentChunk(content=chunk_content,
                                    content_type='text',
                                    metadata={
                                        **doc.metadata, 'chunk_size': len(chunk_content),
                                        'chunk_type': 'docling_smart'
                                    })

                # Start new chunk with overlap
                overlap_lines = current_chunk[-overlap // 50:] if overlap > 0 else []
                current_chunk = overlap_lines + [line]
                current_size = sum(len(l) + 1 for l in current_chunk)
            else:
                current_chunk.append(line)
                current_size += line_size

        # Yield final chunk if any content remains
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            yield DocumentChunk(content=chunk_content,
                                content_type='text',
                                metadata={
                                    **doc.metadata, 'chunk_size': len(chunk_content),
                                    'chunk_type': 'docling_smart'
                                })

    def create_text_extraction_prompt(self) -> ChatPromptTemplate:
        """/no_think Create prompt template for text-based information extraction with Docling awareness"""
        system_message_prompt = SystemMessagePromptTemplate.from_template(self.system_prompt + """
            
            Additional context: The input text has been processed by Docling, which means:
            - Tables are properly formatted in markdown
            - Document structure is preserved
            - Figures and captions are identified
            - Text quality is enhanced through advanced PDF processing
            
            Pay special attention to structured elements like tables and figures when extracting information.
            """)

        human_template = """Please extract the required information from the following Docling-processed text:
        
        {content}
        
        Extract the information according to this schema:
        {format_instructions}
        
        The text has been processed by Docling for better structure recognition. 
        Pay attention to tables (marked with ## Table), figures (marked with ## Figure), 
        and other structured elements in the markdown format.
        
        Return the extracted information in the specified JSON format.
        """

        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        return ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    def process_text_chunk(self, chunk: DocumentChunk) -> Optional[Dict[str, Any]]:
        """Process a text document chunk with Docling enhancements"""
        try:
            prompt = self.create_text_extraction_prompt()

            messages = prompt.format_messages(content=chunk.content,
                                              format_instructions="/no_think " +
                                              self.output_parser.get_format_instructions())

            response = self.llm_api.chat_completion(messages)
            if not response:
                return None

            parsed_json = parse_json_markdown(response["response"])
            # Validate against schema
            validated_response = self.output_schema(**parsed_json)
            return validated_response

        except Exception as e:
            print(f"Error processing text chunk: {e}")
            return None

    def create_multimodal_extraction_prompt(self) -> ChatPromptTemplate:
        """Create prompt template for multimodal extraction with Docling structure"""
        system_message_prompt = SystemMessagePromptTemplate.from_template(self.system_prompt + """
            
            Additional context: You are analyzing a document that has been processed by Docling,
            which provides enhanced structure recognition. The content includes both text and 
            visual elements that have been identified and structured.
            """)

        human_template = """Please extract the required information from the provided document content.
        The document has been processed with Docling for enhanced structure recognition.
        
        Extract the information according to this schema:
        {format_instructions}
        
        Return the extracted information in the specified JSON format.
        """

        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        return ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    def process_multimodal_chunk(self, chunk: DocumentChunk, doc: DoclingProcessedDocument) -> Optional[Dict[str, Any]]:
        """Process chunk with access to original Docling result for multimodal content"""
        try:
            prompt = self.create_multimodal_extraction_prompt()

            # Prepare content - combine text with image references
            content_parts = [{"type": "text", "text": chunk.content}]

            # If we have the Docling result, we can extract images
            if doc.docling_result and hasattr(doc.docling_result.document, 'pictures'):
                for i, picture in enumerate(doc.docling_result.document.pictures):
                    # Try to get image data if available
                    if hasattr(picture, 'image') and picture.image:
                        try:
                            # Convert image to base64
                            image_data = picture.image
                            if isinstance(image_data, bytes):
                                image_b64 = self.doc_processor.encode_image_base64(image_data)
                                content_parts.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_b64}"
                                    }
                                })
                        except Exception as e:
                            print(f"Could not process image {i}: {e}")

            messages = prompt.format_messages(format_instructions=self.output_parser.get_format_instructions())

            # Update the human message with multimodal content
            if len(content_parts) > 1:
                messages[1].content = content_parts

            response = self.llm_api.chat_completion(messages)
            if not response:
                return None

            parsed_json = parse_json_markdown(response["response"])
            # Validate against schema
            validated_response = self.output_schema(**parsed_json)
            return validated_response

        except Exception as e:
            print(f"Error processing multimodal chunk: {e}")
            # Fallback to text-only processing
            return self.process_text_chunk(chunk)

    def process_chunk(self, chunk: DocumentChunk, doc: DoclingProcessedDocument) -> Optional[Dict[str, Any]]:
        """Process a document chunk with Docling context"""
        # Check if we can do multimodal processing
        if (hasattr(self.llm_api, 'supports_vision') and self.llm_api.supports_vision() and doc.docling_result
                and hasattr(doc.docling_result.document, 'pictures') and doc.docling_result.document.pictures):
            return self.process_multimodal_chunk(chunk, doc)
        else:
            return self.process_text_chunk(chunk)

    def process_document(self, doc: Union[DoclingProcessedDocument,
                                          List[DoclingProcessedDocument]]) -> List[Dict[str, Any]]:
        """Process document and extract information"""
        # Handle both single documents and lists
        docs = [doc] if isinstance(doc, DoclingProcessedDocument) else doc

        # Process each document
        all_results = []
        for d in docs:
            if d.content_type != 'chunks':
                # Create chunks for the document
                chunks = list(self.chunk_document(d))
            else:
                chunks = d.content
            # Process chunks with document context
            doc_results = []
            for chunk in chunks:
                result = self.process_chunk(chunk, d)
                if result is not None:
                    doc_results.append(result)

            all_results.extend(doc_results)

        return all_results

    def save_results(self, results: List[Any], output_path: Path) -> None:
        """Save extraction results to JSON file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert Pydantic models to dictionaries
        json_results = []
        for result in results:
            if hasattr(result, 'model_dump'):  # Pydantic v2
                json_results.append(result.model_dump())
            elif hasattr(result, 'dict'):  # Pydantic v1
                json_results.append(result.dict())
            else:
                json_results.append(result)  # Already a dict or other JSON-serializable object

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)

        print(f"Results saved to {output_path}")

    def process_all(self) -> None:
        """Process all documents in configured directory"""
        # Process documents using Docling
        processed_docs = self.doc_processor.process_directory()

        if not processed_docs:
            print("No documents to process")
            return

        # Process each document
        for doc in processed_docs:
            try:
                results = self.process_document(doc)

                # Generate output path from source file name
                source_file = Path(doc.metadata['source_file']).stem
                output_path = Path(self.config.document.output_dir) / f"{source_file}_extracted_docling.json"
                self.save_results(results, output_path)

            except Exception as e:
                print(f"Error processing document: {e}")
                continue


if __name__ == "__main__":
    import os
    import importlib.util
    from pathlib import Path
    from IE_agent_config import IEAgentConfig, ExtractorConfig, DocumentConfig, SchemaConfig
    from IE_agent_schema_generator import SchemaGenerator
    from dllmforge.LLMs.Deltares_LLMs import DeltaresOllamaLLM

    # Setup paths
    current_dir = Path(__file__).parent
    schema_dir = current_dir / "generated_schemas"
    schema_dir.mkdir(exist_ok=True)
    schema_file = schema_dir / "model_hyperparameters_docling.py"

    # Create schema configuration for model hyperparameters
    schema_config = SchemaConfig(
        task_description=
        """/no_think Generate a Pydantic schema class named ModelHyperparameters to extract machine learning model hyperparameters from research papers and documentation.
        The schema should capture: model architecture details (type, layers, neurons, etc.), training parameters (learning rate, batch size, epochs), 
        optimization settings (optimizer, loss function), regularization techniques (dropout, etc.). 
        Note: Input will be processed by Docling for enhanced structure recognition including tables and figures.""",
        output_path=str(schema_file))

    # Generate and save the schema
    llm = DeltaresOllamaLLM(base_url="https://chat-api.directory.intra", model_name="qwen3:latest", temperature=0.8)
    schema_generator = SchemaGenerator(schema_config, llm_api=llm)
    schema_code = schema_generator.generate_schema()

    # Find all class names in the generated code
    import re
    class_matches = re.finditer(r"class\s+(\w+)\s*\(", schema_code)
    class_names = [match.group(1) for match in class_matches]

    if not class_names:
        raise ValueError("Could not find any class names in generated schema")

    # Get the last class as it's typically the main schema
    schema_class_name = class_names[-1]
    print(f"\nFound schema classes: {', '.join(class_names)}")
    print(f"Using main schema class: {schema_class_name}")

    # Save the schema
    schema_generator.save_schema(schema_code)

    # Import the generated schema module
    spec = importlib.util.spec_from_file_location("model_hyperparameters_docling", schema_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get the schema class dynamically
    if not hasattr(module, schema_class_name):
        raise ValueError(f"Generated schema does not contain class {schema_class_name}")
    SchemaClass = getattr(module, schema_class_name)

    # Create configuration for the extractor
    config = IEAgentConfig(
        schema=schema_config,  # Reuse the same schema config
        document=DocumentConfig(
            input_dir=r"tests\test_input",
            output_dir=r"tests\test_output",
            file_pattern="*.pdf",  # Process PDF files
            output_type="text"  # Extract as text
        ),
        extractor=ExtractorConfig())

    # Example 1: Process single document with Docling
    print("\nExample 1: Processing single document with Docling...")
    single_doc_path = Path(
        r"tests\test_input\piping_documents\Campos Montero et al. - 2025 - SchemaGAN A conditional Generative Adversarial Network for geotechnical subsurface schematisation.pdf"
    )

    # Create extractor with Docling support
    extractor = DoclingInfoExtractor(config=config, output_schema=SchemaClass, llm_api=llm)

    # Process the document
    doc = extractor.doc_processor.process_document(single_doc_path)
    if doc:
        results = extractor.process_document(doc)
        output_path = Path(config.document.output_dir) / f"{single_doc_path.stem}_extracted_docling.json"
        extractor.save_results(results, output_path)

    # Example 2: Process entire directory with Docling
    #print("\nExample 2: Processing entire directory with Docling...")
    ## Create new extractor instance with the same schema
    #extractor = DoclingInfoExtractor(config=config, output_schema=SchemaClass, llm_api=llm)
    #extractor.process_all()
