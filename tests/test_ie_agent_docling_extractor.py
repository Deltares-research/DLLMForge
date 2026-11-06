import pytest
from unittest.mock import MagicMock, patch, mock_open
from dllmforge.IE_agent_extractor_docling import (
    DoclingDocumentProcessor,
    DoclingInfoExtractor,
    DoclingProcessedDocument,
    DocumentChunk,
    serialize_docling_document,
    DOCLING_PROMPT_ADDENDUM,
    _clean_table_markdown,
)


class FakeProv:

    def __init__(self, page_no):
        self.page_no = page_no


class FakeText:

    def __init__(self, text, page, label="paragraph"):
        self.text = text
        self.label = label
        self.prov = [FakeProv(page)]


class FakeTable:

    def __init__(self, markdown, page, caption="Table 1. Summary"):
        self._markdown = markdown
        self.label = "table"
        self.prov = [FakeProv(page)]
        self.caption = caption

    def export_to_markdown(self):
        return self._markdown


class FakeDocument:

    def __init__(self, items, n_pages=4):
        self._items = items
        self.pages = list(range(n_pages))
        self.tables = [i for i in items if hasattr(i, "export_to_markdown")]
        self.pictures = []
        self.texts = [i for i in items if not hasattr(i, "export_to_markdown")]
        self.meta = None

    def iterate_items(self):
        for item in self._items:
            yield item, 0


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


def test_serialize_docling_keeps_table_in_reading_order():
    table_md = "| | Eupen | La Gileppe |\n| Dam height | 66 m | 68 m |"
    doc = FakeDocument([
        FakeText("2 Case study", 2, label="section_header"),
        FakeText("Two 50 m high dams are located in the upper part of the catchment.", 2),
        FakeTable(table_md, 4),
    ])
    markdown, index, stats = serialize_docling_document(doc)
    assert markdown.index("[PAGE 2]") < markdown.index("[PAGE 4]")
    assert markdown.index("[SECTION: 2 Case study]") < markdown.index("[TABLE:")
    assert "[TABLE: Table 1. Summary]" in markdown
    assert "66 m" in markdown
    assert index[0]["type"] == "section"
    assert index[-1]["type"] == "table"
    assert index[-1]["page"] == 4
    assert stats["num_tables"] == 1


@patch("dllmforge.IE_agent_extractor_docling.DocumentConverter")
def test_docling_document_processor_process_document(mock_converter_class, dummy_config, tmp_path):
    mock_converter = MagicMock()
    document = FakeDocument([FakeText("abc", 1)])
    mock_converter.convert.return_value = MagicMock(document=document)
    mock_converter_class.return_value = mock_converter
    proc = DoclingDocumentProcessor(dummy_config)
    out = proc.process_document(tmp_path / "foo.pdf")
    assert isinstance(out, DoclingProcessedDocument)
    assert out.content_type == "text"
    assert isinstance(out.content, str)
    assert "[PAGE 1]" in out.content
    assert "abc" in out.content
    assert out.metadata["element_index"][0]["page"] == 1
    assert out.metadata["processing_method"] == "docling"


@patch("dllmforge.IE_agent_extractor_docling.base64.b64encode", return_value=b"ZmFrZQ==")
def test_encode_image_base64(mock_b64, dummy_config):
    proc = DoclingDocumentProcessor(dummy_config)
    out = proc.encode_image_base64(b"fake")
    assert out == "ZmFrZQ=="
    mock_b64.assert_called_once()


@patch("dllmforge.IE_agent_extractor_docling.DocumentConverter")
def test_process_directory(mock_converter_class, dummy_config, tmp_path):
    (tmp_path / "a.pdf").write_text("x")
    (tmp_path / "b.pdf").write_text("y")
    doc_conv = MagicMock()
    doc_conv.convert.return_value = MagicMock(document=FakeDocument([FakeText("abc", 1)]))
    mock_converter_class.return_value = doc_conv
    proc = DoclingDocumentProcessor(dummy_config)
    results = proc.process_directory()
    assert len(results) == 2
    assert all(isinstance(d, DoclingProcessedDocument) for d in results)


@patch("dllmforge.IE_agent_extractor_docling.DocumentConverter")
def test_plain_arg_processor_constructor(mock_converter_class, tmp_path):
    proc = DoclingDocumentProcessor(input_dir=tmp_path, file_pattern="*.pdf", output_dir=tmp_path)
    assert proc.config.document.file_pattern == "*.pdf"


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
def test_direct_arg_skips_refine_system_prompt(mock_parser):
    with patch.object(DoclingInfoExtractor, "refine_system_prompt") as mock_refine:
        info = DoclingInfoExtractor(
            output_schema=MagicMock(),
            llm_api=MagicMock(),
            system_prompt="Extract basins.",
        )
        mock_refine.assert_not_called()
    assert "Extract basins." in info.system_prompt
    assert "[TABLE:" in DOCLING_PROMPT_ADDENDUM
    assert "Prefer table values" in info.system_prompt or "table" in info.system_prompt.lower()


@patch("dllmforge.IE_agent_extractor_docling.PydanticOutputParser", autospec=True)
def test_docling_info_extractor_chunk_document(mock_parser, dummy_config):
    info = DoclingInfoExtractor(config=dummy_config, output_schema=MagicMock(), llm_api=MagicMock())
    doc = MagicMock()
    doc.content = [DocumentChunk("txt", "text", {}) for _ in range(2)]
    doc.metadata = {}
    info.config.extractor.chunk_size = 10
    out = list(info.chunk_document(doc))
    assert out


@patch("dllmforge.IE_agent_extractor_docling.PydanticOutputParser", autospec=True)
def test_docling_info_extractor_process_text_chunk_success(mock_parser, dummy_config):
    dummy_schema = MagicMock(return_value={"foo": 1})
    mock_llm = MagicMock()
    mock_llm.chat_completion.return_value = {"response": '{"a": 1}'}
    parser = MagicMock()
    parser.get_format_instructions.return_value = "INST"
    mock_parser.return_value = parser
    info = DoclingInfoExtractor(config=dummy_config, output_schema=dummy_schema, llm_api=mock_llm)
    chunk = DocumentChunk("doc", "text", {})
    out = info.process_text_chunk(chunk)
    assert isinstance(out, dict) or out is None


@patch("dllmforge.IE_agent_extractor_docling.PydanticOutputParser", autospec=True)
def test_docling_info_extractor_process_document_list(mock_parser, dummy_config):
    info = DoclingInfoExtractor(config=dummy_config, output_schema=MagicMock(), llm_api=MagicMock())
    d1 = MagicMock(content_type="text", metadata={})
    d2 = MagicMock(content_type="chunks", metadata={})
    d1.content = [DocumentChunk("doc", "text", {})]
    d2.content = [DocumentChunk("doc", "text", {})]
    info.chunk_document = MagicMock(return_value=d1.content)
    info.process_chunk = MagicMock(return_value={"foo": 1})
    out = info.process_document([d1, d2])
    assert [item["foo"] for item in out] == [1, 1]
    assert all("metadata" in item for item in out)


@patch("dllmforge.IE_agent_extractor_docling.PydanticOutputParser", autospec=True)
def test_process_document_recovers_source_page(mock_parser, dummy_config):
    info = DoclingInfoExtractor(config=dummy_config, output_schema=MagicMock(), llm_api=MagicMock())
    doc = DoclingProcessedDocument(
        content="[PAGE 2]\n\nTwo 50 m high dams are located in the upper part of the catchment.",
        content_type="text",
        metadata={
            "element_index": [{
                "page": 2,
                "type": "text",
                "text": "Two 50 m high dams are located in the upper part of the catchment.",
            }]
        },
    )
    info.chunk_document = MagicMock(return_value=[
        DocumentChunk(doc.content, "text", doc.metadata)
    ])
    info.process_chunk = MagicMock(return_value={
        "source_page": 4,
        "source_quote": "Two 50 m high dams are located in the upper part of the catchment.",
    })
    out = info.process_document(doc)
    assert out[0]["source_page"] == 2
    assert out[0]["quote_verified"] is True
    assert "element_index" not in out[0]["metadata"]
    assert "page_index" not in out[0]["metadata"]


def test_clean_table_markdown_repairs_scientific_notation():
    raw = (
        "| Capacity | 25 � 10 6 hm 3 | 26:4 � 10 6 hm 3 |\n"
        "| Dam height | 66 m | 68 m |"
    )
    cleaned = _clean_table_markdown(raw)
    assert "25 × 10^6" in cleaned
    assert "26.4 × 10^6" in cleaned
    assert "�" not in cleaned


def test_serialize_docling_cleans_garbled_table_cells():
    table_md = "| | Eupen | La Gileppe |\n| Capacity | 25 � 10 6 hm 3 | 26:4 � 10 6 hm 3 |"
    doc = FakeDocument([FakeTable(table_md, 4)])
    markdown, index, stats = serialize_docling_document(doc)
    assert "25 × 10^6" in markdown
    assert "26.4 × 10^6" in index[-1]["text"]
    assert stats["num_tables"] == 1


@patch("builtins.open", new_callable=mock_open)
def test_docling_info_extractor_save_results(mock_file, dummy_config):
    info = DoclingInfoExtractor(config=dummy_config, output_schema=MagicMock(), llm_api=MagicMock())

    class Dummy:

        def dict(self):
            return {"a": 1}

        def model_dump(self):
            return {"b": 1}

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
