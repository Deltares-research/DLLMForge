#TODO: add async version of this module
"""
Synchronous Information Extractor using Docling as a structured document front-end.

Docling serializes PDFs in reading order (sections, markdown tables, figure captions)
with [PAGE N] tags. The LLM extracts against a Pydantic schema; source_page is recovered
from source_quote via IE_agent_provenance.
"""
import json
from typing import List, Dict, Any, Optional, Union, Generator, Tuple
from pathlib import Path
from pydantic import BaseModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.utils.json import parse_json_markdown
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from dllmforge.langchain_api import LangchainAPI
from dllmforge.IE_agent_config import IEAgentConfig, ExtractorConfig, DocumentConfig
from dllmforge.IE_agent_provenance import apply_quote_provenance
import base64

try:
    from docling.document_converter import DocumentConverter
    DOCLING_AVAILABLE = True
except Exception:
    DOCLING_AVAILABLE = False

    class DocumentConverter:
        """Fallback stub used when the real `docling` package isn't available."""

        def __init__(self, *args, **kwargs):
            return None

        def convert(self, *args, **kwargs):
            raise RuntimeError("docling is not available in this environment")


DOCLING_PROMPT_ADDENDUM = """
The input was preprocessed by Docling:
- Blocks are tagged [PAGE N], [SECTION: ...], [TABLE: ...], [FIGURE: ...].
- Tables are markdown grids. Prefer table values over rounded prose when they conflict.
- Copy source_quote verbatim from the table row or sentence that supports the field.
- Do not set source_page; it is filled after extraction from the quote.
"""

_SKIP_LABELS = {"page_header", "page_footer"}
_SECTION_LABELS = {"section_header", "title"}
_TABLE_LABELS = {"table"}
_FIGURE_LABELS = {"picture", "figure"}
_CAPTION_LABELS = {"caption"}
_STRUCTURE_SPLIT = ("[PAGE ", "[SECTION:", "[TABLE:", "[FIGURE:")


def _label_of(item: Any) -> str:
    label = getattr(item, "label", None)
    if label is None:
        return ""
    value = getattr(label, "value", label)
    return str(value).lower()


def _page_no(item: Any) -> Optional[int]:
    prov = getattr(item, "prov", None) or []
    if not prov:
        return None
    page = getattr(prov[0], "page_no", None)
    try:
        return int(page) if page is not None else None
    except (TypeError, ValueError):
        return None


def _item_kind(item: Any) -> str:
    label = _label_of(item)
    name = type(item).__name__.lower()
    if label in _SKIP_LABELS:
        return "skip"
    if label in _TABLE_LABELS or "table" in name or (
            hasattr(item, "export_to_markdown") and "table" in name):
        return "table"
    if label in _FIGURE_LABELS or "picture" in name or "figure" in name:
        return "figure"
    if label in _SECTION_LABELS or "sectionheader" in name or name in {"titleitem", "sectionheaderitem"}:
        return "section"
    if label in _CAPTION_LABELS:
        return "caption"
    return "text"


def _item_text(item: Any, kind: str, document: Any = None) -> str:
    if kind == "table" and hasattr(item, "export_to_markdown"):
        try:
            if document is not None:
                return (item.export_to_markdown(doc=document) or "").strip()
            return (item.export_to_markdown() or "").strip()
        except TypeError:
            return (item.export_to_markdown() or "").strip()
        except Exception:
            pass
    for attr in ("text", "caption"):
        value = getattr(item, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value is not None and not isinstance(value, str):
            inner = getattr(value, "text", None)
            if isinstance(inner, str) and inner.strip():
                return inner.strip()
    captions = getattr(item, "captions", None) or []
    bits = []
    for cap in captions:
        text = cap if isinstance(cap, str) else getattr(cap, "text", "")
        if text:
            bits.append(str(text).strip())
    return " ".join(bits).strip()


def _iter_docling_items(document: Any) -> List[Any]:
    if hasattr(document, "iterate_items"):
        items = []
        for entry in document.iterate_items():
            item = entry[0] if isinstance(entry, (tuple, list)) else entry
            items.append(item)
        if items:
            return items
    items = []
    for attr in ("texts", "tables", "pictures"):
        items.extend(list(getattr(document, attr, None) or []))
    return items


def serialize_docling_document(document: Any) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """Turn a Docling document into tagged markdown plus an element index.

    Returns (markdown, element_index, stats).
    """
    parts: List[str] = []
    index: List[Dict[str, Any]] = []
    current_page: Optional[int] = None
    pending_caption: Optional[str] = None
    n_tables = 0
    n_figures = 0

    for item in _iter_docling_items(document):
        kind = _item_kind(item)
        if kind == "skip":
            continue
        page = _page_no(item)
        text = _item_text(item, kind, document)
        if kind == "caption":
            pending_caption = text
            continue
        if not text:
            continue
        if page is not None and page != current_page:
            current_page = page
            parts.append(f"[PAGE {page}]")
        if kind == "section":
            block = f"[SECTION: {text}]"
        elif kind == "table":
            n_tables += 1
            caption = (pending_caption or "").strip()
            if not caption:
                raw_cap = getattr(item, "caption", None)
                if isinstance(raw_cap, str) and raw_cap.strip() and not raw_cap.lstrip().startswith("|"):
                    caption = raw_cap.strip()
            header = f"[TABLE: {caption}]" if caption else "[TABLE]"
            block = f"{header}\n{text}"
            pending_caption = None
        elif kind == "figure":
            n_figures += 1
            caption = pending_caption or text
            block = f"[FIGURE: {caption}]\nCaption: {caption}"
            pending_caption = None
        else:
            block = text
            pending_caption = None
        parts.append(block)
        index.append({"page": page, "type": kind, "text": text if kind != "figure" else caption})

    content = "\n\n".join(parts).strip()
    if content:
        content += "\n"
    pages = getattr(document, "pages", None)
    page_count = len(pages) if pages is not None else len({el["page"] for el in index if el.get("page")})
    stats = {
        "page_count": page_count,
        "num_tables": n_tables or len(getattr(document, "tables", None) or []),
        "num_figures": n_figures or len(getattr(document, "pictures", None) or []),
    }
    return content, index, stats


def _as_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        bits = []
        for item in content:
            if hasattr(item, "content"):
                bits.append(str(item.content))
            else:
                bits.append(str(item))
        return "\n\n".join(bits)
    return str(content or "")


def _split_structure_blocks(text: str) -> List[str]:
    """Split markdown on Docling structure tags, keeping the tag with the block."""
    if not text:
        return []
    lines = text.split("\n")
    blocks: List[str] = []
    current: List[str] = []
    for line in lines:
        stripped = line.lstrip()
        if current and any(stripped.startswith(tag) for tag in _STRUCTURE_SPLIT):
            blocks.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        joined = "\n".join(current).strip()
        if joined:
            blocks.append(joined)
    return blocks or [text]


class DoclingProcessedDocument:
    """Class representing a document processed by Docling"""

    def __init__(self,
                 content: Union[str, bytes, list],
                 content_type: str,
                 metadata: Optional[Dict[str, Any]] = None,
                 docling_result=None):
        self.content = content
        self.content_type = content_type
        self.metadata = metadata or {}
        self.docling_result = docling_result


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
        self.docling_elements = docling_elements or []


class _PlainDocumentConfig:
    """Minimal config wrapper so .document.input_dir works without IEAgentConfig."""

    def __init__(self, document: DocumentConfig):
        self.document = document


class DoclingDocumentProcessor:
    """Document processor using Docling for advanced PDF processing"""

    def __init__(self,
                 config: Any = None,
                 input_dir: Optional[Union[str, Path]] = None,
                 file_pattern: Optional[str] = None,
                 output_dir: Optional[Union[str, Path]] = None):
        if config is not None:
            self.config = config
            if hasattr(config, "document"):
                doc_cfg = config.document
                if input_dir is not None:
                    doc_cfg.input_dir = Path(input_dir)
                if file_pattern is not None:
                    doc_cfg.file_pattern = file_pattern
                if output_dir is not None:
                    doc_cfg.output_dir = Path(output_dir)
            elif hasattr(config, "input_dir"):
                self.config = _PlainDocumentConfig(config if isinstance(config, DocumentConfig) else DocumentConfig(
                    input_dir=Path(input_dir or config.input_dir),
                    file_pattern=file_pattern or getattr(config, "file_pattern", "*.pdf"),
                    output_type="text",
                    output_dir=Path(output_dir) if output_dir is not None else getattr(config, "output_dir", None),
                ))
        else:
            self.config = _PlainDocumentConfig(
                DocumentConfig(
                    input_dir=Path(input_dir or "."),
                    file_pattern=file_pattern or "*.pdf",
                    output_type="text",
                    output_dir=Path(output_dir) if output_dir is not None else None,
                ))
        self.converter = DocumentConverter()

    def encode_image_base64(self, image_data: bytes) -> str:
        """Encode image data to base64 string"""
        return base64.b64encode(image_data).decode("utf-8")

    def process_document(self, file_path: Path) -> Optional[DoclingProcessedDocument]:
        """Process a single document using Docling into tagged markdown."""
        try:
            file_path = Path(file_path)
            result = self.converter.convert(str(file_path))
            document = result.document
            content, element_index, stats = serialize_docling_document(document)
            metadata = {
                "source_file": str(file_path),
                "file_name": file_path.name,
                "file_type": file_path.suffix.lower(),
                "text_length": len(content),
                "page_count": stats["page_count"],
                "num_pages": stats["page_count"],
                "num_tables": stats["num_tables"],
                "num_figures": stats["num_figures"],
                "processing_method": "docling",
                "element_index": element_index,
            }
            if hasattr(document, "meta") and document.meta:
                metadata.update({
                    "title": getattr(document.meta, "title", ""),
                    "author": getattr(document.meta, "author", ""),
                    "subject": getattr(document.meta, "subject", ""),
                })
            return DoclingProcessedDocument(
                content=content,
                content_type="text",
                metadata=metadata,
                docling_result=result,
            )
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
    """Extract structured information from Docling-preprocessed documents using an LLM."""

    def __init__(
        self,
        config: Optional[IEAgentConfig] = None,
        output_schema: Optional[type[BaseModel]] = None,
        llm_api: Optional[LangchainAPI] = None,
        system_prompt: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        doc_processor: Optional[DoclingDocumentProcessor] = None,
        use_vision: bool = False,
    ):
        """Initialize the information extractor.

        Pass ``config`` (IEAgentConfig) for the original path, or the same
        plain arguments as InfoExtractor. Direct mode does not call
        ``refine_system_prompt``.
        """
        self.llm_api = llm_api or LangchainAPI()
        self.use_vision = use_vision
        if config:
            if output_schema is None:
                raise ValueError("output_schema is required")
            self.config = config
            self.output_schema = output_schema
            self.output_parser = PydanticOutputParser(pydantic_object=output_schema)
            self.doc_processor = doc_processor or DoclingDocumentProcessor(config)
            self.chunk_size = config.extractor.chunk_size
            self.chunk_overlap = config.extractor.chunk_overlap
            self.system_prompt = self.refine_system_prompt(config.schema.task_description)
        else:
            if output_schema is None:
                raise ValueError("output_schema is required if config is not given")
            self.config = None
            self.output_schema = output_schema
            self.output_parser = PydanticOutputParser(pydantic_object=output_schema)
            self.chunk_size = chunk_size or 80000
            self.chunk_overlap = chunk_overlap or 10000
            base = system_prompt or "You are an information extraction LLM."
            if DOCLING_PROMPT_ADDENDUM.strip() not in base:
                self.system_prompt = base.rstrip() + "\n" + DOCLING_PROMPT_ADDENDUM
            else:
                self.system_prompt = base
            self.doc_processor = doc_processor or DoclingDocumentProcessor(input_dir=".", file_pattern="*.pdf")

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

    def _chunk_limits(self) -> Tuple[int, int]:
        if self.config is not None:
            return self.config.extractor.chunk_size, self.config.extractor.chunk_overlap
        return self.chunk_size, self.chunk_overlap

    def chunk_document(self, doc: DoclingProcessedDocument) -> Generator[DocumentChunk, None, None]:
        """Split on section/table/page tags when the document exceeds chunk_size."""
        content = _as_text_content(doc.content)
        chunk_size, overlap = self._chunk_limits()
        metadata = dict(doc.metadata or {})

        if len(content) <= chunk_size:
            yield DocumentChunk(content=content, content_type="text", metadata=metadata)
            return

        blocks = _split_structure_blocks(content)
        current: List[str] = []
        current_size = 0

        def emit(parts: List[str]) -> DocumentChunk:
            chunk_content = "\n\n".join(parts)
            return DocumentChunk(
                content=chunk_content,
                content_type="text",
                metadata={**metadata, "chunk_size": len(chunk_content), "chunk_type": "docling_smart"},
            )

        for block in blocks:
            block_size = len(block) + 2
            if current and current_size + block_size > chunk_size:
                yield emit(current)
                overlap_text = "\n\n".join(current)
                if overlap > 0 and overlap_text:
                    current = [overlap_text[-overlap:].lstrip(), block]
                    current_size = sum(len(p) + 2 for p in current)
                else:
                    current = [block]
                    current_size = block_size
            else:
                current.append(block)
                current_size += block_size

        if current:
            yield emit(current)

    def create_text_extraction_prompt(self) -> ChatPromptTemplate:
        """Create prompt template for text-based information extraction with Docling awareness"""
        system_message_prompt = SystemMessagePromptTemplate.from_template(self.system_prompt)

        human_template = """Please extract the required information from the following Docling-processed text:
        
        {content}
        
        Extract the information according to this schema:
        {format_instructions}
        
        The text has been processed by Docling. Prefer markdown table values over rounded prose.
        Copy source_quote verbatim. Do not set source_page.
        
        Return the extracted information in the specified JSON format.
        """

        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        return ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    def process_text_chunk(self, chunk: DocumentChunk) -> Optional[Dict[str, Any]]:
        """Process a text document chunk with Docling enhancements"""
        try:
            prompt = self.create_text_extraction_prompt()
            content = chunk.content if isinstance(chunk.content, str) else _as_text_content(chunk.content)
            messages = prompt.format_messages(
                content=content,
                format_instructions="/no_think " + self.output_parser.get_format_instructions(),
            )
            response = self.llm_api.chat_completion(messages)
            if not response:
                return None
            parsed_json = parse_json_markdown(response["response"])
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
        """Attach only figures whose page appears in this chunk's [PAGE N] tags."""
        try:
            prompt = self.create_multimodal_extraction_prompt()
            content_text = chunk.content if isinstance(chunk.content, str) else _as_text_content(chunk.content)
            content_parts = [{"type": "text", "text": content_text}]
            chunk_pages = {int(p) for p in __import__("re").findall(r"\[PAGE (\d+)\]", content_text)}

            if doc.docling_result and hasattr(doc.docling_result.document, "pictures"):
                for i, picture in enumerate(doc.docling_result.document.pictures):
                    page = _page_no(picture)
                    if chunk_pages and page is not None and page not in chunk_pages:
                        continue
                    if hasattr(picture, "image") and picture.image:
                        try:
                            image_data = picture.image
                            if isinstance(image_data, bytes):
                                image_b64 = self.doc_processor.encode_image_base64(image_data)
                                content_parts.append({
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                                })
                        except Exception as e:
                            print(f"Could not process image {i}: {e}")

            messages = prompt.format_messages(format_instructions=self.output_parser.get_format_instructions())
            if len(content_parts) > 1:
                messages[1].content = content_parts

            response = self.llm_api.chat_completion(messages)
            if not response:
                return None
            parsed_json = parse_json_markdown(response["response"])
            return self.output_schema(**parsed_json)
        except Exception as e:
            print(f"Error processing multimodal chunk: {e}")
            return self.process_text_chunk(chunk)

    def process_chunk(self, chunk: DocumentChunk, doc: DoclingProcessedDocument) -> Optional[Dict[str, Any]]:
        """Process a document chunk with Docling context (text by default)."""
        if (self.use_vision and hasattr(self.llm_api, "supports_vision") and self.llm_api.supports_vision()
                and doc.docling_result and hasattr(doc.docling_result.document, "pictures")
                and doc.docling_result.document.pictures):
            return self.process_multimodal_chunk(chunk, doc)
        return self.process_text_chunk(chunk)

    def _result_to_dict(self, result: Any) -> Dict[str, Any]:
        if hasattr(result, "model_dump"):
            return result.model_dump()
        if hasattr(result, "dict"):
            return result.dict()
        if isinstance(result, dict):
            return dict(result)
        return {"value": result}

    def process_document(self, doc: Union[DoclingProcessedDocument,
                                          List[DoclingProcessedDocument]]) -> List[Dict[str, Any]]:
        """Process document, recover source_page from quotes, merge processor metadata."""
        docs = [doc] if isinstance(doc, DoclingProcessedDocument) else doc
        all_results = []
        for d in docs:
            if getattr(d, "content_type", None) == "chunks":
                chunks = d.content
            else:
                chunks = list(self.chunk_document(d))
            for chunk in chunks:
                result = self.process_chunk(chunk, d)
                if result is None:
                    continue
                schema_dict = self._result_to_dict(result)
                meta = dict(getattr(d, "metadata", None) or {})
                chunk_meta = dict(getattr(chunk, "metadata", None) or {})
                schema_dict = apply_quote_provenance(schema_dict, metadata={**meta, **chunk_meta})
                all_results.append({**schema_dict, "metadata": {**meta, **chunk_meta}})
        return all_results

    def save_results(self, results: List[Any], output_path: Path) -> None:
        """Save extraction results to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        json_results = []
        for result in results:
            if isinstance(result, dict):
                json_results.append(result)
            elif hasattr(result, "model_dump"):
                json_results.append(result.model_dump())
            elif hasattr(result, "dict"):
                json_results.append(result.dict())
            else:
                json_results.append(result)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_path}")

    def process_all(self) -> None:
        """Process all documents in configured directory"""
        processed_docs = self.doc_processor.process_directory()
        if not processed_docs:
            print("No documents to process")
            return
        output_dir = None
        if self.config is not None:
            output_dir = Path(self.config.document.output_dir)
        elif getattr(self.doc_processor.config, "document", None) is not None:
            output_dir = Path(self.doc_processor.config.document.output_dir or ".")
        else:
            output_dir = Path(".")
        for doc in processed_docs:
            try:
                results = self.process_document(doc)
                source_file = Path(doc.metadata["source_file"]).stem
                output_path = output_dir / f"{source_file}_extracted_docling.json"
                self.save_results(results, output_path)
            except Exception as e:
                print(f"Error processing document: {e}")
                continue


if __name__ == "__main__":
    import importlib.util
    import re
    from dllmforge.IE_agent_config import SchemaConfig
    from dllmforge.IE_agent_schema_generator import SchemaGenerator
    from dllmforge.LLMs.Deltares_LLMs import DeltaresOllamaLLM

    current_dir = Path(__file__).parent
    schema_dir = current_dir / "generated_schemas"
    schema_dir.mkdir(exist_ok=True)
    schema_file = schema_dir / "model_hyperparameters_docling.py"

    schema_config = SchemaConfig(
        task_description=
        """/no_think Generate a Pydantic schema class named ModelHyperparameters to extract machine learning model hyperparameters from research papers and documentation.
        The schema should capture: model architecture details (type, layers, neurons, etc.), training parameters (learning rate, batch size, epochs), 
        optimization settings (optimizer, loss function), regularization techniques (dropout, etc.). 
        Note: Input will be processed by Docling for enhanced structure recognition including tables and figures.""",
        output_path=str(schema_file))

    llm = DeltaresOllamaLLM(base_url="https://chat-api.directory.intra", model_name="qwen3:latest", temperature=0.8)
    schema_generator = SchemaGenerator(schema_config, llm_api=llm)
    schema_code = schema_generator.generate_schema()

    class_matches = re.finditer(r"class\s+(\w+)\s*\(", schema_code)
    class_names = [match.group(1) for match in class_matches]

    if not class_names:
        raise ValueError("Could not find any class names in generated schema")

    schema_class_name = class_names[-1]
    print(f"\nFound schema classes: {', '.join(class_names)}")
    print(f"Using main schema class: {schema_class_name}")

    schema_generator.save_schema(schema_code)

    spec = importlib.util.spec_from_file_location("model_hyperparameters_docling", schema_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, schema_class_name):
        raise ValueError(f"Generated schema does not contain class {schema_class_name}")
    SchemaClass = getattr(module, schema_class_name)

    config = IEAgentConfig(
        schema=schema_config,
        document=DocumentConfig(
            input_dir=r"tests\test_input",
            output_dir=r"tests\test_output",
            file_pattern="*.pdf",
            output_type="text",
        ),
        extractor=ExtractorConfig())

    print("\nExample 1: Processing single document with Docling...")
    single_doc_path = Path(
        r"tests\test_input\piping_documents\Campos Montero et al. - 2025 - SchemaGAN A conditional Generative Adversarial Network for geotechnical subsurface schematisation.pdf"
    )

    extractor = DoclingInfoExtractor(config=config, output_schema=SchemaClass, llm_api=llm)

    doc = extractor.doc_processor.process_document(single_doc_path)
    if doc:
        results = extractor.process_document(doc)
        output_path = Path(config.document.output_dir) / f"{single_doc_path.stem}_extracted_docling.json"
        extractor.save_results(results, output_path)
