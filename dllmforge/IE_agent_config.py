"""
Configuration module for Information Extraction Agent.
Defines configuration classes and utilities for managing user inputs.
"""
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
from pydantic import BaseModel, Field

class LLMConfig(BaseModel):
    """Configuration for LLM settings"""
    model_provider: str = Field(
        default="azure-openai",
        description="Provider of model to use (azure-openai, openai, or mistral)"
    )
    temperature: float = Field(
        default=0.1,
        description="Temperature setting for the model (0.0 to 1.0)"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for the provider"
    )
    api_base: Optional[str] = Field(
        default=None,
        description="API base URL (for Azure)"
    )
    api_version: Optional[str] = Field(
        default=None,
        description="API version (for Azure)"
    )
    deployment_name: Optional[str] = Field(
        default=None,
        description="Deployment name (for Azure)"
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Model name (for OpenAI/Mistral)"
    )

class SchemaConfig(BaseModel):
    """Configuration for schema generation"""
    task_description: str = Field(
        description="User's description of the information extraction task"
    )
    example_doc: Optional[str] = Field(
        default=None,
        description="Example document to help with schema generation"
    )
    user_schema: Optional[str] = Field(
        default=None,
        description="User-provided schema code (if not using auto-generation)"
    )
    output_path: Optional[Path] = Field(
        default=None,
        description="Path to save generated schema"
    )

class DocumentConfig(BaseModel):
    """Configuration for document processing"""
    input_dir: Path = Field(
        description="Directory containing input documents"
    )
    output_dir: Path = Field(
        description="Directory for output files"
    )
    file_pattern: str = Field(
        default="*.*",
        description="Glob pattern for matching input files"
    )
    output_type: str = Field(
        default="text",
        description="Type of processing ('text' or 'image')"
    )
    text_chunk_threshold: int = Field(
        default=100000,
        description="Character threshold for text chunking"
    )
    image_chunk_threshold: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        description="Size threshold for image chunking in bytes"
    )

class ExtractorConfig(BaseModel):
    """Configuration for information extraction"""
    max_concurrent_tasks: int = Field(
        default=5,
        description="Maximum number of concurrent LLM calls"
    )
    chunk_size: int = Field(
        default=2000,
        description="Size of text chunks when splitting is needed"
    )
    chunk_overlap: int = Field(
        default=200,
        description="Overlap between text chunks"
    )

class IEAgentConfig(BaseModel):
    """Main configuration class for Information Extraction Agent"""
    llm: LLMConfig = Field(
        default_factory=LLMConfig,
        description="LLM configuration"
    )
    schema: SchemaConfig = Field(
        description="Schema generation configuration"
    )
    document: DocumentConfig = Field(
        description="Document processing configuration"
    )
    extractor: ExtractorConfig = Field(
        default_factory=ExtractorConfig,
        description="Information extraction configuration"
    )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "IEAgentConfig":
        """Create config from dictionary"""
        return cls(**config_dict)

    @classmethod
    def load_from_file(cls, config_path: Union[str, Path]) -> "IEAgentConfig":
        """Load config from JSON or YAML file"""
        config_path = Path(config_path)
        if config_path.suffix == '.json':
            import json
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        elif config_path.suffix in ['.yaml', '.yml']:
            import yaml
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        return cls.from_dict(config_dict)

    def save_to_file(self, config_path: Union[str, Path]) -> None:
        """Save config to JSON or YAML file"""
        config_path = Path(config_path)
        config_dict = self.dict()
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        if config_path.suffix == '.json':
            import json
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif config_path.suffix in ['.yaml', '.yml']:
            import yaml
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

# Example configuration
example_config = {
    "llm": {
        "model_provider": "azure-openai",
        "temperature": 0.1
    },
    "schema": {
        "task_description": "Extract technical specifications from engineering documents",
        "example_doc": None,
        "output_path": "generated_schema.py"
    },
    "document": {
        "input_dir": "input_docs",
        "output_dir": "output",
        "file_pattern": "*.pdf",
        "output_type": "text"
    },
    "extractor": {
        "max_concurrent_tasks": 5,
        "chunk_size": 2000,
        "chunk_overlap": 200
    }
}
