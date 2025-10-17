#TODO: add async version of this module

"""
Synchronous Information Extractor module for extracting structured information from documents using LLM.
"""
import os
import json
from typing import List, Dict, Any, Optional, Union, Generator
from pathlib import Path
from pydantic import BaseModel
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers.json import parse_json_markdown
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from dllmforge.langchain_api import LangchainAPI
from dllmforge.IE_agent_config import IEAgentConfig, ExtractorConfig
from dllmforge.IE_agent_document_processor import ProcessedDocument, DocumentProcessor

class DocumentChunk:
    """Class representing a chunk of document content"""
    def __init__(self, 
                 content: Union[str, bytes],
                 content_type: str,
                 metadata: Optional[Dict[str, Any]] = None):
        self.content = content
        self.content_type = content_type
        self.metadata = metadata or {}

class InfoExtractor:
    """Class for extracting information from documents using LLM"""
    def __init__(self, 
                 config: Optional[IEAgentConfig] = None,
                 output_schema: Optional[type[BaseModel]] = None,
                 llm_api: Optional[LangchainAPI] = None,
                 # Plain-argument mode:
                 system_prompt: Optional[str] = None,
                 chunk_size: Optional[int] = None,
                 chunk_overlap: Optional[int] = None,
                 doc_processor: Optional[DocumentProcessor] = None,
                 document_output_type: str = 'text',
                 ):
        """Initialize the information extractor.

        You can use either `config` (IEAgentConfig), or pass the individual parameters directly.
        """
        if config:
            self.config = config
            self.output_schema = output_schema or None
            self.llm_api = llm_api or LangchainAPI()
            self.doc_processor = doc_processor or DocumentProcessor(config.document)
            self.output_parser = PydanticOutputParser(pydantic_object=output_schema)
            self.chunk_size = config.extractor.chunk_size
            self.chunk_overlap = config.extractor.chunk_overlap
            self.system_prompt = self.refine_system_prompt(config.schema.task_description)
        else:
            if output_schema is None:
                raise ValueError('output_schema is required if config is not given')
            self.config = None
            self.output_schema = output_schema
            self.llm_api = llm_api or LangchainAPI()
            self.output_parser = PydanticOutputParser(pydantic_object=output_schema)
            self.chunk_size = chunk_size or 80000
            self.chunk_overlap = chunk_overlap or 10000
            # NOTE: direct mode must require plain prompt string
            self.system_prompt = system_prompt or "You are an information extraction LLM."
            if doc_processor:
                self.doc_processor = doc_processor
            else:
                # create a very basic DocumentProcessor (assume user will provide method input)
                self.doc_processor = DocumentProcessor(DocumentConfig(
                    input_dir=Path('.'), file_pattern="*.pdf", output_type=document_output_type
                ))

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
        """
        
        human_template = """Please refine this task description into a proper system prompt:
        
        {task_description}
        
        Create a well-structured system prompt that will guide the LLM in extracting information
        according to the task requirements. Be thorough but concise.
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

    def chunk_document(self, doc: ProcessedDocument) -> Generator[DocumentChunk, None, None]:
        """Split document into chunks if needed based on thresholds"""
        if doc.content_type == 'text':
            text = doc.content
            chunk_size = self.chunk_size if hasattr(self, 'chunk_size') else self.config.extractor.chunk_size
            overlap = self.chunk_overlap if hasattr(self, 'chunk_overlap') else self.config.extractor.chunk_overlap
            start = 0
            while start < len(text):
                end = start + chunk_size
                if end < len(text):
                    # Try to find a space to break at
                    while end < len(text) and text[end] != ' ':
                        end -= 1
                    if end == start:  # No space found
                        end = start + chunk_size
                yield DocumentChunk(
                    content=text[start:end],
                    content_type='text',
                    metadata={
                        **doc.metadata,
                        'chunk_start': start,
                        'chunk_end': end
                    }
                )
                start = end - overlap
        elif doc.content_type == 'image':
            yield DocumentChunk(
                content=doc.content,
                content_type='image',
                metadata=doc.metadata
            )
        else:
            yield DocumentChunk(
                content=doc.content,
                content_type=doc.content_type,
                metadata=doc.metadata
            )

    def create_text_extraction_prompt(self) -> ChatPromptTemplate:
        """Create prompt template for text-based information extraction"""
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            self.system_prompt
        )
        
        human_template = """Please extract the required information from the following text:
        
        {content}
        
        Extract the information according to this schema:
        {format_instructions}
        
        Return the extracted information in the specified JSON format.
        """
        
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        
        return ChatPromptTemplate.from_messages([
            system_message_prompt,
            human_message_prompt
        ])

    def process_text_chunk(self, chunk: DocumentChunk) -> Optional[Dict[str, Any]]:
        """Process a text document chunk"""
        try:
            prompt = self.create_text_extraction_prompt()
            messages = prompt.format_messages(
                content=chunk.content,
                format_instructions=self.output_parser.get_format_instructions()
            )
            response = self.llm_api.chat_completion(messages)
            if not response:
                return None
            parsed_json = parse_json_markdown(response["response"])
            # print("PARSED JSON TO VALIDATE:", parsed_json)  # DEBUG LINE
            # Validate against schema
            validated_response = self.output_schema(**parsed_json)
            return validated_response
        except Exception as e:
            print(f"Error processing text chunk: {e}")
            return None

    def create_image_extraction_prompt(self) -> ChatPromptTemplate:
        """Create prompt template for image-based information extraction"""
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            self.system_prompt + "\nNote: The input will include images that you should analyze."
        )
        
        human_template = """Please extract the required information from the provided image.
        
        Extract the information according to this schema:
        {format_instructions}
        
        Return the extracted information in the specified JSON format as above.
        """
        
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        
        return ChatPromptTemplate.from_messages([
            system_message_prompt,
            human_message_prompt
        ])

    def process_image_chunk(self, chunk: DocumentChunk) -> Optional[Dict[str, Any]]:
        """Process an image document chunk"""
        try:
            prompt = self.create_image_extraction_prompt()
            content = f"data:image/jpeg;base64,{self.doc_processor.encode_image_base64(chunk.content)}"
            
            messages = prompt.format_messages(
                format_instructions=self.output_parser.get_format_instructions()
            )
            
            # Convert to multimodal format
            messages[1].content = [
                {"type": "text", "text": messages[1].content},
                {
                    "type": "image_url",
                    "image_url": {"url": content}
                }
            ]
            
            response = self.llm_api.chat_completion(messages)
            if not response:
                return None
            
            parsed_json = parse_json_markdown(response["response"])
            # Validate against schema
            validated_response = self.output_schema(**parsed_json)
            return validated_response
            
        except Exception as e:
            print(f"Error processing image chunk: {e}")
            return None

    def process_chunk(self, chunk: DocumentChunk) -> Optional[Dict[str, Any]]:
        """Process a document chunk based on its type"""
        if chunk.content_type == 'text':
            return self.process_text_chunk(chunk)
        else:  # image
            return self.process_image_chunk(chunk)

    def process_document(self, doc: Union[ProcessedDocument, List[ProcessedDocument]]) -> List[Dict[str, Any]]:
        """Process document and extract information"""
        # Patch: robustly wrap non-list docs
        if not isinstance(doc, list):
            docs = [doc]
        else:
            docs = doc
        # Create chunks for all documents
        chunks = []
        for d in docs:
            chunks.extend(list(self.chunk_document(d)))
        # Process chunks sequentially
        results = []
        for chunk in chunks:
            result = self.process_chunk(chunk)
            if result is not None:
                results.append(result)
        return results

    def save_results(self, 
                    results: List[Any], 
                    output_path: Union[str, Path]) -> None:
        """Save extraction results to JSON file"""
        output_path = Path(output_path)
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
        # Process documents
        processed_docs = self.doc_processor.process_directory()
        
        if not processed_docs:
            print("No documents to process")
            return
        
        # Process each document
        for doc in processed_docs:
            try:
                results = self.process_document(doc)
                
                # Generate output path from source file name
                if isinstance(doc, list):
                    source_file = Path(doc[0].metadata['source_file']).stem
                else:
                    source_file = Path(doc.metadata['source_file']).stem
                    
                output_path = Path(self.config.document.output_dir) / f"{source_file}_extracted.json"
                self.save_results(results, output_path)
                
            except Exception as e:
                print(f"Error processing document: {e}")
                continue


if __name__ == "__main__":
    # ------- Explicit Example: Using config objects (full control) -------
    import os
    import importlib.util
    from pathlib import Path
    from dllmforge.IE_agent_config import IEAgentConfig, ExtractorConfig, DocumentConfig, SchemaConfig
    from dllmforge.IE_agent_schema_generator import SchemaGenerator
    from dllmforge.IE_agent_document_processor import DocumentProcessor
    from dllmforge.langchain_api import LangchainAPI
    from glob import glob
    import json
    import re

    # 1. PREPARE SCHEMA (always required)
    current_dir = Path(__file__).parent
    schema_dir = current_dir / "generated_schemas"
    schema_dir.mkdir(exist_ok=True)
    schema_file = schema_dir / "model_hyperparameters.py"
    schema_task_description = (
        "Generate a Pydantic schema class named ModelHyperparameters to extract machine learning model hyperparameters from research papers and documentation. "
        "The schema should capture: model architecture details (type, layers, neurons, etc.), "
        "training parameters (learning rate, batch size, epochs), "
        "optimization settings (optimizer, loss function), regularization techniques (dropout, etc.)."
    )
    schema_config = SchemaConfig(
        task_description=schema_task_description,  # REQUIRED
        example_doc=None,                         # optional
        user_schema_path=None,                    # optional
        output_path=str(schema_file)              # optional for saving schema
    )
    schema_generator = SchemaGenerator(schema_config)
    schema_code = schema_generator.generate_schema()
    class_matches = re.finditer(r"class\s+(\w+)\s*\(", schema_code)
    class_names = [match.group(1) for match in class_matches]
    if not class_names:
        raise ValueError("Could not find any class names in generated schema")
    schema_class_name = class_names[-1]
    schema_generator.save_schema(schema_code)
    spec = importlib.util.spec_from_file_location("model_hyperparameters", schema_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, schema_class_name):
        raise ValueError(f"Generated schema does not contain class {schema_class_name}")
    SchemaClass = getattr(module, schema_class_name)

    # ----- Specify ALL config arguments explicitly
    document_input_dir = r"c:/Users/deng_jg/work/16centralized_agents/test_data/test"
    document_file_pattern = "*.pdf"
    document_output_type = "text"
    document_output_dir = r"c:/Users/deng_jg/work/16centralized_agents/test_data/output"

    chunk_size = 80000      # how large (chars) each text chunk should be
    chunk_overlap = 10000   # how much chunks overlap (chars)

    output_schema = SchemaClass  # REQUIRED
    llm_api = LangchainAPI(model_provider="azure-openai", temperature=0.1)  # OPTIONAL, or None for default


#%%    # 2. CONFIG-BASED (FULL) USAGE
    # Build ALL config objects with all fields
    extractor_config = ExtractorConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    document_config = DocumentConfig(
        input_dir=document_input_dir,
        file_pattern=document_file_pattern,
        output_type=document_output_type,
        output_dir=document_output_dir
    )
    config = IEAgentConfig(
        schema=schema_config,
        document=document_config,
        extractor=extractor_config
    )

    extractor = InfoExtractor(
        config=config,                  # REQUIRED (when using config route)
        output_schema=output_schema,    # REQUIRED
        llm_api=llm_api                 # Optional
    )
    
    # --- 2a. Process single file (with all InfoExtractor vars shown)
    single_doc_path = os.path.join(document_input_dir, "lstm_low_flow.pdf")

    doc = extractor.doc_processor.process_file(single_doc_path)  # Uses DocumentProcessor config
    if doc:
        results = extractor.process_document(doc)
        output_path = Path(document_output_dir) / "lstm_low_flow_extracted.json"  # fix: Path object
        extractor.save_results(results, output_path)
        print(f"[CONFIG] Single file results saved to {output_path}")

    # --- 2b. Directory mode (all config-driven)
    extractor.process_all()
    print(f"[CONFIG] Directory batch complete! Check {document_output_dir}")

#%%
    # ------- Explicit Example: Direct/no-config (all args shown) -----------
    print("\nExample 2: Direct, no config objects (all params explicit)")
    # Define for direct mode:
    direct_system_prompt = "Extract model hyperparameters from research paper."
    direct_doc_processor = DocumentProcessor(
        input_dir=document_input_dir,
        file_pattern=document_file_pattern,
        output_type=document_output_type,
        output_dir=document_output_dir
    )
    # Create InfoExtractor using keyword arguments (no config)
    direct_extractor = InfoExtractor(
        output_schema=output_schema,
        llm_api=llm_api,
        system_prompt=direct_system_prompt,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        doc_processor=direct_doc_processor,
        document_output_type=document_output_type
    )

    # --- Process single file ---
    single_doc_path = os.path.join(document_input_dir, "lstm_low_flow.pdf")
    doc = direct_extractor.doc_processor.process_file(single_doc_path)
    results = direct_extractor.process_document(doc)
    print(f"[DIRECT] Single-file direct results (first result): {results[0] if results else None}")
    output_path = os.path.join(document_output_dir, "lstm_low_flow_extracted.json")
    direct_extractor.save_results(results, output_path)
    
    # --- Directory mode (loop) ---
    direct_extractor.process_all()
    direct_extractor.save_results(results, output_path)
