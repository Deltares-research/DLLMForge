"""
Schema Generator module for automatically generating Pydantic models based on user descriptions
and example documents using LLM.
"""
import json
from typing import Optional, Dict, Any
from pathlib import Path
from pydantic import BaseModel
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from .langchain_api import LangchainAPI
from .IE_agent_config import SchemaConfig

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
        Include field descriptions and appropriate type hints.
        """
        
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        
        return ChatPromptTemplate.from_messages([
            system_message_prompt,
            human_message_prompt
        ])

    def generate_schema(self) -> str:
        """Generate Pydantic schema based on task description and optional example document"""
        # If user provided a schema, return it
        if self.config.user_schema:
            return self.config.user_schema
        
        # Otherwise, generate schema using LLM
        prompt = self.create_schema_generation_prompt()
        messages = prompt.format_messages(
            task_description=self.config.task_description,
            example_doc=self.config.example_doc if self.config.example_doc else "No example provided"
        )
        
        response = self.llm_api.chat_completion(messages)
        return response["response"] if response else ""

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
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

{schema_code}
'''
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_code)
        print(f"Schema saved to {output_path}")

    def load_schema(self, schema_path: Path) -> type[BaseModel]:
        """Load a previously generated schema from a Python file"""
        # This is a placeholder - in practice, you'd need to implement
        # proper Python module loading
        raise NotImplementedError("Schema loading from file not yet implemented")
