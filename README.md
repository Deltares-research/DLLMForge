# DLLMForge

A comprehensive toolkit for building and deploying LLM-based applications with RAG (Retrieval-Augmented Generation) capabilities, agentic workflows, and enterprise-grade features.

## Features

- **Multi-Provider LLM Support**: OpenAI, Azure OpenAI, Anthropic, Mistral, and Deltares hosted models
- **RAG Pipeline**: Complete document processing, embedding, indexing, and retrieval
- **Advanced Document Processing**: PDF, text chunking, and Docling integration
- **Agentic Workflows**: Build autonomous agents with LangChain and LangGraph
- **Local Embeddings**: Run models locally with HuggingFace and Sentence Transformers
- **Evaluation Tools**: Built-in RAG evaluation capabilities
- **Web Apps**: Ready-to-use Streamlit applications
- **Flexible Installation**: Install only what you need with modular dependencies

## Requirements

- Python >= 3.10
- Windows or Linux/Unix operating system

## Installation

### Basic Installation

Install the core package with essential dependencies:

```bash
pip install git+https://github.com/Deltares-research/DLLMForge.git
```

Or clone and install locally:

```bash
git clone https://github.com/Deltares-research/DLLMForge.git
cd DLLMForge
pip install -e .
```

### Installation Options

DLLMForge offers modular installation to minimize dependencies:

#### Core Only (Minimal)
Basic functionality without optional components:
```bash
pip install -e ".[core]"
```

or 
```bash
pip install "git+https://github.com/Deltares-research/DLLMForge.git@main#egg=dllmforge[core]"
```

#### API Integration
For cloud-based LLM providers (OpenAI, Anthropic, Mistral, Azure):
```bash
pip install -e ".[api]"
```

or 

```bash
pip install "git+https://github.com/Deltares-research/DLLMForge.git@main#egg=dllmforge[api]"
```

#### Local Models
For running models locally with HuggingFace and advanced document processing:
```bash
pip install -e ".[local]"
```

or 
```bash
pip install "git+https://github.com/Deltares-research/DLLMForge.git@main#egg=dllmforge[local]"
```


#### Development
Includes testing, linting, and documentation tools:
```bash
pip install -e ".[dev]"
```

or 
```bash
pip install "git+https://github.com/Deltares-research/DLLMForge.git@main#egg=dllmforge[dev]"
```

#### Web Applications
For running FastAPI and Streamlit apps:
```bash
pip install -e ".[web]"
```

or 
```bash
pip install "git+https://github.com/Deltares-research/DLLMForge.git@main#egg=dllmforge[web]"
```

#### Complete Installation
Install everything:
```bash
pip install -e ".[all]"
```
or 
```bash
pip install "git+https://github.com/Deltares-research/DLLMForge.git@main#egg=dllmforge[all]"
```

## Environment Variables

Create a `.env` file in your project root:

```env
# OpenAI
OPENAI_API_KEY=your-openai-key

# Azure OpenAI
AZURE_OPENAI_API_KEY=your-azure-key
AZURE_OPENAI_ENDPOINT=your-endpoint
AZURE_OPENAI_DEPLOYMENT=your-deployment

# Anthropic
ANTHROPIC_API_KEY=your-anthropic-key

# Mistral
MISTRAL_API_KEY=your-mistral-key
```

## Development Setup

### Clone and Install Development Dependencies

```bash
git clone https://github.com/Deltares-research/DLLMForge.git
cd DLLMForge
pip install -e ".[all]"
```

### Pre-commit Hooks

Set up pre-commit hooks for code quality:

```bash
pre-commit install
```

### Running Tests

Run tests with pytest:

```bash
# All tests
pytest

# With coverage
pytest --cov=dllmforge --cov-report=html

```

### Testing with Tox

Test across multiple Python versions:

```bash
# Run all environments
tox

# Specific Python version
tox -e py311

# Test minimal core installation
tox -e core

# Test API dependencies
tox -e api

# Test local dependencies
tox -e local
```

### Code Formatting

Format code with yapf:

```bash
yapf -i -r dllmforge tests
```

Check for linting issues:

```bash
flake8 dllmforge tests
```

## Testing Different Dependency Configurations

The package supports testing various installation configurations:

- **`tox -e core`**: Tests minimal installation without optional dependencies
- **`tox -e api`**: Tests with cloud API dependencies only
- **`tox -e local`**: Tests with local model dependencies
- **`tox -e py310,py311,py312`**: Tests across Python versions with all dependencies

## Project Structure

```
DLLMForge/
├── dllmforge/                 # Main package
│   ├── agent_core.py         # Core agentic functionality
│   ├── anthropic_api.py      # Anthropic Claude integration
│   ├── openai_api.py         # OpenAI integration
│   ├── llamaindex_api.py     # LlamaIndex integration
│   ├── rag_*.py              # RAG pipeline components
│   ├── IE_agent_*.py         # Information extraction agents
│   └── utils/                # Utility modules
├── tests/                     # Test suite
├── streamlit_apps/           # Web applications
├── docs/                     # Documentation
├── workflows/                # Example workflows
└── pyproject.toml           # Package configuration
```

## Documentation

Build documentation with Sphinx:

```bash
cd docs
make html
```

View documentation in `docs/_build/html/index.html`

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting (`tox`, `pre-commit run --all-files`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Continuous Integration

The project uses GitHub Actions for CI/CD:

- Tests run on Ubuntu and Windows
- Python versions: 3.10, 3.11, 3.12
- Multiple dependency configurations tested
- Code coverage reporting

## License

See [LICENSE](LICENSE) file for details.

## Support

For issues, questions, or contributions, please visit:
https://github.com/Deltares-research/DLLMForge/issues

## Authors

LLM Team @ Deltares

---

**Note**: This is a research project by Deltares. Some features may be experimental.
