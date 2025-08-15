"""
Async Information Extractor module for extracting structured information from documents using LLM.
"""
import os
import json
import asyncio
from typing import List, Dict, Any, Optional, Union, Generator
from pathlib import Path
from pydantic import BaseModel
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from .langchain_api import LangchainAPI
from .IE_agent_config import IEAgentConfig, ExtractorConfig
from .IE_agent_document_processor import ProcessedDocument, DocumentProcessor

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
                 config: IEAgentConfig,
                 output_schema: type[BaseModel],
                 llm_api: Optional[LangchainAPI] = None):
        """Initialize the information extractor"""
        self.config = config
        self.output_schema = output_schema
        self.llm_api = llm_api or LangchainAPI(
            model_provider=config.llm.model_provider,
            temperature=config.llm.temperature,
            api_key=config.llm.api_key,
            api_base=config.llm.api_base,
            api_version=config.llm.api_version,
            deployment_name=config.llm.deployment_name,
            model_name=config.llm.model_name
        )
        self.output_parser = PydanticOutputParser(pydantic_object=output_schema)
        self.doc_processor = DocumentProcessor(config.document)
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
        """
        
        human_template = """Please refine this task description into a proper system prompt:
        
        {task_description}
        
        Create a well-structured system prompt that will guide the LLM in extracting information
        according to the task requirements.
        """
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
        
        messages = prompt.format_messages(task_description=task_description)
        response = self.llm_api.chat_completion(messages)
        
        return response["response"] if response else task_description

    def chunk_document(self, doc: ProcessedDocument) -> Generator[DocumentChunk, None, None]:
        """Split document into chunks if needed based on thresholds"""
        if doc.content_type == 'text' and doc.content_length > self.config.document.text_chunk_threshold:
            # Split text into chunks
            text = doc.content
            chunk_size = self.config.extractor.chunk_size
            overlap = self.config.extractor.chunk_overlap
            
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
                
        elif doc.content_type == 'image' and doc.content_length > self.config.document.image_chunk_threshold:
            # For images that are too large, we might want to compress or split them
            # For now, we'll just yield the original as one chunk
            yield DocumentChunk(
                content=doc.content,
                content_type='image',
                metadata=doc.metadata
            )
        else:
            # Document is under threshold, yield as single chunk
            yield DocumentChunk(
                content=doc.content,
                content_type=doc.content_type,
                metadata=doc.metadata
            )

    def create_extraction_prompt(self) -> ChatPromptTemplate:
        """Create prompt template for information extraction"""
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            self.system_prompt
        )
        
        human_template = """Please extract the required information from the following {content_type}:
        
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

    async def process_chunk(self, 
                          chunk: DocumentChunk,
                          semaphore: asyncio.Semaphore) -> Optional[Dict[str, Any]]:
        """Process a single document chunk asynchronously"""
        async with semaphore:
            try:
                prompt = self.create_extraction_prompt()
                
                # Prepare content based on chunk type
                if chunk.content_type == 'text':
                    content = chunk.content
                    content_type = 'text'
                else:  # image
                    content = f"data:image/jpeg;base64,{self.doc_processor.encode_image_base64(chunk.content)}"
                    content_type = 'image'
                
                # Format messages
                messages = prompt.format_messages(
                    content=content,
                    content_type=content_type,
                    format_instructions=self.output_parser.get_format_instructions()
                )
                
                if chunk.content_type == 'image':
                    messages[1].content = [
                        {"type": "text", "text": messages[1].content},
                        {
                            "type": "image_url",
                            "image_url": {"url": content}
                        }
                    ]
                
                # Call LLM and parse response
                response = await self.llm_api.chat_completion(messages)
                if not response:
                    return None
                    
                parsed_response = self.output_parser.parse(response["response"])
                
                # Add chunk metadata to response
                result = parsed_response.dict()
                result['_chunk_metadata'] = chunk.metadata
                
                return result
                
            except Exception as e:
                print(f"Error processing chunk: {e}")
                return None

    async def process_document(self, doc: Union[ProcessedDocument, List[ProcessedDocument]]) -> List[Dict[str, Any]]:
        """Process document and extract information asynchronously"""
        # Handle both single documents and lists
        docs = [doc] if isinstance(doc, ProcessedDocument) else doc
        
        # Create chunks for all documents
        chunks = []
        for d in docs:
            chunks.extend(list(self.chunk_document(d)))
        
        # Create semaphore for limiting concurrent tasks
        semaphore = asyncio.Semaphore(self.config.extractor.max_concurrent_tasks)
        
        # Process chunks concurrently
        tasks = [
            self.process_chunk(chunk, semaphore)
            for chunk in chunks
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Filter out None results from failed chunks
        return [r for r in results if r is not None]

    def save_results(self, 
                    results: List[Dict[str, Any]], 
                    output_path: Path) -> None:
        """Save extraction results to JSON file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {output_path}")

    async def process_all(self) -> None:
        """Process all documents in configured directory"""
        # Process documents
        processed_docs = self.doc_processor.process_directory()
        
        if not processed_docs:
            print("No documents to process")
            return
        
        # Process each document
        for doc in processed_docs:
            try:
                results = await self.process_document(doc)
                
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

async def main():
    """Example usage of the Information Extraction Agent"""
    # Example configuration
    config = IEAgentConfig(
        schema=SchemaConfig(
            task_description="""
            Extract technical specifications from engineering documents. The information should include:
            1. Document metadata (title, date, author)
            2. Technical parameters (measurements, materials, specifications)
            3. Key findings or conclusions
            4. Any safety considerations or warnings
            """
        ),
        document=DocumentConfig(
            input_dir="input_docs",
            output_dir="output",
            file_pattern="*.pdf",
            output_type="text"
        )
    )
    
    # Example output schema
    class TechnicalInfo(BaseModel):
        """Schema for technical information extraction"""
        document_metadata: dict = Field(
            description="Document metadata including title, date, and author"
        )
        technical_parameters: dict = Field(
            description="Technical measurements and specifications"
        )
        key_findings: List[str] = Field(
            description="Main conclusions or findings"
        )
        safety_considerations: Optional[List[str]] = Field(
            description="Safety warnings or considerations",
            default=None
        )
    
    # Initialize extractor
    extractor = InfoExtractor(
        config=config,
        output_schema=TechnicalInfo
    )
    
    # Process all documents
    await extractor.process_all()

if __name__ == "__main__":
    asyncio.run(main())
