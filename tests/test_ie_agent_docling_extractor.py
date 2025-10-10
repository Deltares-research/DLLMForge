import pytest
from unittest.mock import MagicMock, patch, mock_open
from dllmforge.IE_agent_extractor_docling import DoclingDocumentProcessor, DoclingInfoExtractor, DoclingProcessedDocument, DocumentChunk

@pytest.fixture
def dummy_config(tmp_path):
    class DummyConfig:
        class schema:
            task_description = "Extract info."
        class extractor:
            chunk_size = 100
            chunk_overlap = 10
        class document:
            input_dir = tmp_path
            file_pattern = "*.pdf"
            output_dir = tmp_path
    return DummyConfig()

@patch("dllmforge.IE_agent_extractor_docling.DocumentConverter")
def test_docling_document_processor_process_document(mock_converter_class, dummy_config, tmp_path):
    mock_converter = MagicMock()
    doc = MagicMock()
    doc.texts = [MagicMock(text="abc", prov=[MagicMock(page_no=1)])]
    doc.tables = []
    doc.pictures = []
    doc.meta = None
    mock_result = MagicMock(document=doc)
    mock_converter.convert.return_value = mock_result
    mock_converter_class.return_value = mock_converter
    proc = DoclingDocumentProcessor(dummy_config)
    out = proc.process_document(tmp_path / "foo.pdf")
    assert isinstance(out, DoclingProcessedDocument)
    assert out.content_type == "chunks"
    assert isinstance(out.content, list)

@patch("dllmforge.IE_agent_extractor_docling.base64.b64encode", return_value=b"ZmFrZQ==")
def test_encode_image_base64(mock_b64, dummy_config):
    proc = DoclingDocumentProcessor(dummy_config)
    out = proc.encode_image_base64(b"fake")
    assert out == "ZmFrZQ=="
    mock_b64.assert_called_once()

@patch("dllmforge.IE_agent_extractor_docling.DocumentConverter")
def test_process_directory(mock_converter_class, dummy_config, tmp_path):
    # Setup fake dir with two pdfs
    (tmp_path / "a.pdf").write_text("x")
    (tmp_path / "b.pdf").write_text("y")
    doc_conv = MagicMock()
    doc = MagicMock()
    doc.texts = [MagicMock(text="abc", prov=[MagicMock(page_no=1)])]
    doc.tables = []
    doc.pictures = []
    doc.meta = None
    doc_conv.convert.return_value = MagicMock(document=doc)
    mock_converter_class.return_value = doc_conv
    proc = DoclingDocumentProcessor(dummy_config)
    results = proc.process_directory()
    assert len(results) == 2
    assert all(isinstance(d, DoclingProcessedDocument) for d in results)

@patch("dllmforge.IE_agent_extractor_docling.PydanticOutputParser", autospec=True)
def test_docling_info_extractor_construction(mock_parser, dummy_config):
    dummy_schema = MagicMock()
    mock_llm = MagicMock()
    info = DoclingInfoExtractor(
        config=dummy_config,
        output_schema=dummy_schema,
        llm_api=mock_llm,
    )
    assert info.llm_api == mock_llm
    assert info.config == dummy_config
    assert info.output_schema == dummy_schema

@patch("dllmforge.IE_agent_extractor_docling.PydanticOutputParser", autospec=True)
def test_docling_info_extractor_chunk_document(mock_parser, dummy_config):
    info = DoclingInfoExtractor(config=dummy_config, output_schema=MagicMock(), llm_api=MagicMock())
    doc = MagicMock()
    # doc.content is a list of DocumentChunk with dummy content
    doc.content = [DocumentChunk("txt", "text", {}) for _ in range(2)]
    doc.metadata = {}
    info.config.extractor.chunk_size = 10
    # Should return single chunk if content is small
    out = list(info.chunk_document(doc))
    assert out

@patch("dllmforge.IE_agent_extractor_docling.PydanticOutputParser", autospec=True)
def test_docling_info_extractor_process_text_chunk_success(mock_parser, dummy_config):
    dummy_schema = MagicMock(return_value={"foo": 1})
    mock_llm = MagicMock()
    mock_llm.chat_completion.return_value = {"response": '{"a": 1}'}
    parser = MagicMock()
    parser.get_format_instructions.return_value = "INST"
    parser.parse_json_markdown = MagicMock()
    mock_parser.return_value = parser
    info = DoclingInfoExtractor(config=dummy_config, output_schema=dummy_schema, llm_api=mock_llm)
    chunk = DocumentChunk("doc", "text", {})
    out = info.process_text_chunk(chunk)
    assert isinstance(out, dict) or out is None

@patch("dllmforge.IE_agent_extractor_docling.PydanticOutputParser", autospec=True)
def test_docling_info_extractor_process_document_list(mock_parser, dummy_config):
    info = DoclingInfoExtractor(config=dummy_config, output_schema=MagicMock(), llm_api=MagicMock())
    d1, d2 = MagicMock(content_type='text'), MagicMock(content_type='chunks')
    d1.content = [DocumentChunk("doc", "text", {})]
    d2.content = [DocumentChunk("doc", "text", {})]
    info.chunk_document = MagicMock(return_value=d1.content)
    info.process_chunk = MagicMock(return_value={"foo": 1})
    out = info.process_document([d1, d2])
    assert out == [{"foo": 1}, {"foo": 1}]

@patch("builtins.open", new_callable=mock_open)
def test_docling_info_extractor_save_results(mock_file, dummy_config):
    info = DoclingInfoExtractor(config=dummy_config, output_schema=MagicMock(), llm_api=MagicMock())
    class Dummy:
        def dict(self): return {"a": 1}
        def model_dump(self): return {"b": 1}
    data = [Dummy()]
    info.save_results(data, dummy_config.document.input_dir / "x.json")
    mock_file.assert_called_once()
    handle = mock_file()
    handle.write.assert_called()

@patch("dllmforge.IE_agent_extractor_docling.PydanticOutputParser", autospec=True)
def test_docling_info_extractor_process_all(mock_parser, dummy_config):
    info = DoclingInfoExtractor(config=dummy_config, output_schema=MagicMock(), llm_api=MagicMock())
    d = MagicMock(metadata={"source_file": "f.pdf"})
    info.doc_processor.process_directory = MagicMock(return_value=[d])
    info.process_document = MagicMock(return_value=[{"ok": 1}])
    info.save_results = MagicMock()
    info.config.document.output_dir = dummy_config.document.input_dir
    info.process_all()
    info.process_document.assert_called()
    info.save_results.assert_called()
