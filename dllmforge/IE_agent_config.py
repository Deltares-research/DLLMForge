"""
Configuration module for Information Extraction Agent.
Defines configuration classes and utilities for managing user inputs.
"""
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
from pydantic import BaseModel, Field


class SchemaConfig(BaseModel):
    """Configuration for schema generation"""
    task_description: str = Field(description="User's description of the information extraction task")
    example_doc: Optional[str] = Field(default=None, description="Example document to help with schema generation")
    user_schema_path: Optional[Path] = Field(default=None, description="Path to user-provided schema Python file")
    output_path: Optional[Path] = Field(default=None, description="Path to save generated schema")


class DocumentConfig(BaseModel):
    """Configuration for document processing"""
    input_dir: Path = Field(description="Directory containing input documents")
    file_pattern: str = Field(default="*.pdf*", description="Glob pattern for matching input files")
    output_type: str = Field(default="text", description="Type of processing ('text' or 'image')")
    output_dir: Optional[Path] = Field(default=None, description="Directory for output files")


class ExtractorConfig(BaseModel):
    """Configuration for information extraction"""
    # max_concurrent_tasks: int = Field(
    #     default=5,
    #     description="Maximum number of concurrent LLM calls"
    # )
    chunk_size: int = Field(default=80000, description="Size of text chunks when splitting is needed")
    chunk_overlap: int = Field(default=10000, description="Overlap between text chunks")


class IEAgentConfig(BaseModel):
    """Main configuration class for Information Extraction Agent"""
    schema: SchemaConfig = Field(description="Schema generation configuration")
    document: DocumentConfig = Field(description="Document processing configuration")
    extractor: ExtractorConfig = Field(default_factory=ExtractorConfig,
                                       description="Information extraction configuration")

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


if __name__ == "__main__":
    config = IEAgentConfig.load_from_file(r"c:\Users\deng_jg\work\16centralized_agents\test_data\example_config.json")
    print(config)
