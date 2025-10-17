"""
Schema Generator module for automatically generating Pydantic models based on user descriptions
and example documents using LLM.
"""
import json
from typing import Optional, Dict, Any, Union
from pathlib import Path
from pydantic import BaseModel
from langchain_core.output_parsers import BaseOutputParser
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from dllmforge.langchain_api import LangchainAPI
from dllmforge.IE_agent_config import SchemaConfig
from dllmforge.utils.document_loader import DocumentLoader
import re


class PythonCodeOutputParser(BaseOutputParser[str]):
    """Parse Python code from LLM responses that may contain markdown."""

    def parse(self, text: str) -> str:
        """Parse the output of an LLM call to extract Python code."""
        # Try to find code within ```python ... ``` blocks
        python_pattern = r'```python\s*(.*?)\s*```'
        match = re.search(python_pattern, text, re.DOTALL)

        if match:
            return match.group(1).strip()

        # Fallback: try any ``` blocks
        general_pattern = r'```\s*(.*?)\s*```'
        match = re.search(general_pattern, text, re.DOTALL)

        if match:
            return match.group(1).strip()

        # If no markdown blocks, return the whole text
        return text.strip()

    def get_format_instructions(self) -> str:
        return "Wrap your Python code in ```python ... ``` markdown blocks."


class SchemaGenerator:
    """Class for generating Pydantic schemas using LLM"""

    def __init__(self, config: SchemaConfig, llm_api: Optional[LangchainAPI] = None):
        """Initialize the schema generator
        
        Args:
            config: Schema generation configuration
            llm_api: Optional pre-configured LangchainAPI instance
        """
        self.config = config
        self.llm_api = llm_api or LangchainAPI()
        self.document_loader = DocumentLoader()
        self.setup_parser()

    def setup_parser(self):
        """Setup the Pydantic output parser for structured verification results"""
        self.output_parser = PythonCodeOutputParser()

    def _load_example_doc(self) -> Optional[str]:
        """Load and convert example document to text if provided as a file path"""
        if not self.config.example_doc:
            print("No example document provided")
            return None

        # If example_doc is already a string of text, return it
        if not any(self.config.example_doc.endswith(ext) for ext in ['.pdf', '.docx', '.xlsx', '.csv']):
            print("Example document is already a string of text")
            return self.config.example_doc

        try:
            # If it's a file path, try to load and convert to text
            example_path = Path(self.config.example_doc)
            if example_path.exists():
                print(f"Loading example document from {example_path}")
                return self.document_loader.load_document(example_path)
        except Exception as e:
            print(f"Warning: Failed to load example document: {e}")
            return None

    def create_schema_generation_prompt(self) -> ChatPromptTemplate:
        """Create prompt template for generating Pydantic schema"""
        system_template = """You are a specialized assistant for generating Pydantic data models.
        Your task is to analyze the user's description and example document (if provided) to create
        an appropriate Pydantic schema for information extraction.
        
        Guidelines for schema generation:
        1. Use clear, descriptive field names
        2. Include appropriate type hints and field descriptions
        3. Use nested models when dealing with structured data
        4. Add validation rules where appropriate
        5. Make fields optional when their presence is not guaranteed
        
        The output should be a valid Pydantic model definition in Python code format.
        """

        human_template = """Based on the following information, create a Pydantic schema:
        
        Task Description: {task_description}
        
        Example Document (if provided):
        {example_doc}
        
        Generate a Pydantic model that captures all relevant information fields.
        Include field descriptions and appropriate type hints. See instructions below:
        {format_instructions}
        """

        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        return ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    def _load_user_schema(self, schema_path: Path) -> Optional[str]:
        """Load user-provided schema from Python file"""
        try:
            if not schema_path.exists():
                print(f"Warning: Schema file not found: {schema_path}")
                return None

            with open(schema_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error loading schema file: {e}")
            return None

    def generate_schema(self) -> str:
        """Generate Pydantic schema based on task description and optional example document"""
        # If user provided a schema file, load and return it
        if self.config.user_schema_path:
            schema_code = self._load_user_schema(self.config.user_schema_path)
            if schema_code:
                print(f"Loaded user schema from {self.config.user_schema_path}")
                return schema_code
            # If loading fails, fall back to generation
            print("Failed to load user schema, falling back to generation")

        # Load and convert example document if provided
        example_doc_text = self._load_example_doc()

        # Generate schema using LLM
        prompt = self.create_schema_generation_prompt()
        format_instructions = self.output_parser.get_format_instructions()
        messages = prompt.format_messages(task_description=self.config.task_description,
                                          example_doc=example_doc_text if example_doc_text else "No example provided",
                                          format_instructions=format_instructions)

        response = self.llm_api.chat_completion(messages)
        try:
            parsed_result = self.output_parser.parse(response["response"])
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return None
        return parsed_result

    def save_schema(self, schema_code: str) -> None:
        """Save generated schema to a Python file"""
        if not self.config.output_path:
            return

        output_path = Path(self.config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Add imports and any necessary wrapper code
        full_code = f'''"""
Generated Pydantic schema for information extraction
"""
{schema_code}
'''

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_code)
        print(f"Schema saved to {output_path}")


if __name__ == "__main__":
    # Example 1: Generate schema from task description only
    #-----------------------------------------------------------------------------------------
    config_simple = SchemaConfig(task_description="""
        Extract flood event information from reports. We need to capture:
        1. Event dates (start and end)
        2. Location details
        3. Rainfall measurements
        4. Damage assessment
        5. Response actions taken
        """,
                                 output_path="generated_flood_schema.py")

    generator_simple = SchemaGenerator(config_simple)
    schema_code = generator_simple.generate_schema()
    generator_simple.save_schema(schema_code)
    print("\nExample 1: Schema generated from task description")
    print("-" * 50)
    print(schema_code)

    # Example 2: Generate schema with example document as text
    #------------------------------------------------------------------------------------------
    config_with_example = SchemaConfig(task_description="Extract technical specifications from engineering reports",
                                       example_doc="""
        Technical Report: Bridge Assessment
        Date: 2024-02-15
        Author: John Smith
        
        Structural Parameters:
        - Load capacity: 200 tons
        - Span length: 150 meters
        - Material: Reinforced concrete
        
        Safety Assessment:
        1. Current condition: Good
        2. Maintenance needed: Minor repairs
        3. Weight restrictions: None
        
        Recommendations:
        - Schedule routine inspection
        - Update load monitoring system
        - Replace worn expansion joints
        """,
                                       output_path="generated_technical_schema.py")

    generator_with_example = SchemaGenerator(config_with_example)
    schema_code = generator_with_example.generate_schema()
    generator_with_example.save_schema(schema_code)
    print("\nExample 2: Schema generated with example text")
    print("-" * 50)
    print(schema_code)

    # Example 3: Generate schema from PDF example
    #-----------------------------------------------------------------------------------------
    try:
        config_from_pdf = SchemaConfig(
            task_description="Extract rainfall event information from the following document",
            example_doc=
            r"c:\Users\deng_jg\work\12LLMs_ARPAL_flash_flood\llms_flash_flood\data\external\REM_20241008_rossaC_vers20241125.pdf",  # Replace with actual PDF path
            output_path="generated_rainfall_event_schema.py")

        generator_from_pdf = SchemaGenerator(config_from_pdf)
        schema_code = generator_from_pdf.generate_schema()
        generator_from_pdf.save_schema(schema_code)
        print("\nExample 3: Schema generated from PDF example")
        print("-" * 50)
        print(schema_code)
    except Exception as e:
        print(f"\nExample 3 failed: {e}")

    # Example 4: Use pre-defined schema
    #-----------------------------------------------------------------------------------------
    predefined_schema = r'c:\Users\deng_jg\work\16centralized_agents\DLLMForge\dllmforge\weather_schema.py'

    config_predefined = SchemaConfig(task_description="Extract weather event information",
                                     user_schema_path=Path(predefined_schema),
                                     output_path="weather_schema_from_file.py")

    generator_predefined = SchemaGenerator(config_predefined)
    schema_code = generator_predefined.generate_schema()
    generator_predefined.save_schema(schema_code)
    print("\nExample 4: Using pre-defined schema")
    print("-" * 50)
    print(schema_code)
