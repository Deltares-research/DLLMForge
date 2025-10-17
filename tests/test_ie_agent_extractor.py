import pytest
from unittest.mock import patch, MagicMock, mock_open
from dllmforge.IE_agent_extractor import InfoExtractor, DocumentChunk

import types


class DummySchema:

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def dict(self):
        return self.__dict__

    def model_dump(self):
        # Simulate pydantic v2 API
        return self.__dict__


@pytest.fixture
def mock_config():
    mock = MagicMock()
    mock.schema.task_description = "Extract info."
    mock.extractor.chunk_size = 10
    mock.extractor.chunk_overlap = 2
    mock.document.output_dir = "/outputs"
    return mock


@pytest.fixture
def mock_doc_processor():
    # Returns a fake ProcessedDocument
    doc = MagicMock()
    doc.content = "some document text"
    doc.content_type = 'text'
    doc.metadata = {"source_file": "file1.pdf"}
    processor = MagicMock()
    processor.process_directory.return_value = [doc]
    processor.process_file.return_value = doc
    processor.encode_image_base64.return_value = "encodedstring"
    return processor


@pytest.fixture
def mock_llm_api():
    mock = MagicMock()
    mock.chat_completion.return_value = {"response": '{"foo": "bar"}'}
    return mock


@patch("dllmforge.IE_agent_extractor.PydanticOutputParser", autospec=True)
def test_info_extractor_constructor(MockParser, mock_config, mock_doc_processor, mock_llm_api):
    info = InfoExtractor(config=mock_config,
                         output_schema=DummySchema,
                         llm_api=mock_llm_api,
                         doc_processor=mock_doc_processor)
    assert info.config == mock_config
    assert info.llm_api == mock_llm_api
    assert info.output_schema == DummySchema
    assert info.doc_processor == mock_doc_processor
    assert hasattr(info, "process_document")


@patch("dllmforge.IE_agent_extractor.PydanticOutputParser", autospec=True)
def test_process_document_simple(MockParser, mock_config, mock_doc_processor, mock_llm_api):
    info = InfoExtractor(config=mock_config,
                         output_schema=DummySchema,
                         llm_api=mock_llm_api,
                         doc_processor=mock_doc_processor)

    # Patch chunk_document and process_chunk to simulate extraction
    info.chunk_document = MagicMock(return_value=[DocumentChunk("abc", "text")])
    info.process_chunk = MagicMock(return_value={"foo": "bar"})
    doc = MagicMock()
    doc.content_type = 'text'
    doc.content = 'abc abc abc'
    results = info.process_document(doc)
    assert results == [{"foo": "bar"}]
    info.chunk_document.assert_called_with(doc)


@patch("dllmforge.IE_agent_extractor.PydanticOutputParser", autospec=True)
def test_chunk_document_yields_chunks(MockParser, mock_config, mock_doc_processor, mock_llm_api):
    info = InfoExtractor(config=mock_config,
                         output_schema=DummySchema,
                         llm_api=mock_llm_api,
                         doc_processor=mock_doc_processor)
    doc = MagicMock()
    doc.content_type = 'text'
    doc.content = "The quick brown fox jumps over the lazy dog and runs away"
    doc.metadata = {}
    chunks = list(info.chunk_document(doc))
    assert all(isinstance(c, DocumentChunk) for c in chunks)
    assert chunks


@patch("builtins.open", new_callable=mock_open)
def test_save_results_json(mock_file, mock_config, mock_doc_processor, mock_llm_api):
    info = InfoExtractor(config=mock_config,
                         output_schema=DummySchema,
                         llm_api=mock_llm_api,
                         doc_processor=mock_doc_processor)
    data = [DummySchema(a=1, b=2), DummySchema(x=3)]
    info.save_results(data, "/tmp/out.json")
    mock_file.assert_called_once()
    handle = mock_file()
    handle.write.assert_called()


@patch("dllmforge.IE_agent_extractor.PydanticOutputParser", autospec=True)
def test_process_all(mock_PydanticOutputParser, mock_config, mock_doc_processor, mock_llm_api):
    info = InfoExtractor(config=mock_config,
                         output_schema=DummySchema,
                         llm_api=mock_llm_api,
                         doc_processor=mock_doc_processor)
    info.process_document = MagicMock(return_value=[{"foo": "bar"}])
    info.save_results = MagicMock()
    mock_doc_processor.process_directory.return_value = [MagicMock(metadata={"source_file": "testfile.pdf"})]
    info.config.document.output_dir = "/out"
    info.process_all()
    info.process_document.assert_called()
    info.save_results.assert_called()


@patch("dllmforge.IE_agent_extractor.PydanticOutputParser", autospec=True)
def test_error_handling_text_chunk(MockParser, mock_config, mock_doc_processor, mock_llm_api):
    # Simulate LLM returning None
    mock_llm_api.chat_completion.return_value = None
    info = InfoExtractor(config=mock_config,
                         output_schema=DummySchema,
                         llm_api=mock_llm_api,
                         doc_processor=mock_doc_processor)
    chunk = DocumentChunk("broken", "text")
    # Force output_parser to raise error on get_format_instructions
    info.output_parser.get_format_instructions.side_effect = Exception("fail")
    result = info.process_text_chunk(chunk)
    assert result is None
