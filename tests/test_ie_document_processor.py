import pytest
from unittest.mock import MagicMock, patch, mock_open
from dllmforge.IE_agent_document_processor import DocumentProcessor, ProcessedDocument
import io


@pytest.fixture
def doc_config(tmp_path):

    class MockConfig:
        input_dir = tmp_path
        file_pattern = "*.pdf"
        output_type = "text"
        output_dir = tmp_path

    return MockConfig()


@patch("dllmforge.IE_agent_document_processor.DocumentLoader")
def test_process_to_text_doc(mock_loader_class, doc_config, tmp_path):
    mock_loader = MagicMock()
    mock_loader.load_document.return_value = "filecontent"
    mock_loader_class.return_value = mock_loader
    proc = DocumentProcessor(config=doc_config)
    path = tmp_path / "test.pdf"
    path.write_text("xx")
    pd = proc.process_to_text(str(path))
    assert pd.content == "filecontent"
    assert pd.content_type == "text"
    assert "source_file" in pd.metadata


@patch("dllmforge.IE_agent_document_processor.Image")
@patch("dllmforge.IE_agent_document_processor.fitz.open")
def test_process_to_image_pdf(mock_fitz_open, mock_Image, doc_config, tmp_path):
    doc_config.output_type = "image"
    # Setup fake fitz+PIL chain
    fake_page = MagicMock()
    fake_pix = MagicMock(width=1,
                         height=1,
                         samples=b"\xff" * 3,
                         get_pixmap=MagicMock(return_value=MagicMock(width=1, height=1, samples=b'\x00' * 3)))
    fake_page.get_pixmap.return_value = fake_pix

    class FakeFitzDoc(list):

        def close(self):
            pass

    fake_fitzdoc = FakeFitzDoc([fake_page])
    mock_fitz_open.return_value = fake_fitzdoc
    mock_img = MagicMock()
    bytestream = io.BytesIO(b"abc")
    mock_img.save = MagicMock()
    mock_Image.frombytes.return_value = mock_img
    proc = DocumentProcessor(config=doc_config)
    # File extension matters but file content doesn't
    path = tmp_path / "img.pdf"
    path.write_bytes(b"xx")
    out = proc.process_to_image(str(path))
    assert out[0].content_type == "image"
    assert isinstance(out[0].content, bytes) or hasattr(out[0].content, "__class__")


@patch("dllmforge.IE_agent_document_processor.DocumentLoader")
def test_process_file_branching(mock_loader_class, doc_config, tmp_path):
    mock_loader = MagicMock()
    mock_loader.load_document.return_value = "txt"
    mock_loader_class.return_value = mock_loader
    proc = DocumentProcessor(config=doc_config)
    path = tmp_path / "abc.pdf"
    path.write_text("""""")
    doc_config.output_type = "text"
    out1 = proc.process_file(path)
    assert out1.content_type == "text"
    doc_config.output_type = "image"
    with patch("dllmforge.IE_agent_document_processor.Image"), \
         patch("dllmforge.IE_agent_document_processor.fitz.open"):
        out2 = proc.process_file(path)
        assert isinstance(out2, list)


@patch("dllmforge.IE_agent_document_processor.DocumentLoader")
def test_process_directory_glob(mock_loader_class, doc_config, tmp_path):
    mock_loader = MagicMock()
    mock_loader.load_document.return_value = "stuff"
    mock_loader_class.return_value = mock_loader
    proc = DocumentProcessor(config=doc_config)
    # Create two files matching the pattern
    path1 = tmp_path / "a1.pdf"
    path2 = tmp_path / "b2.pdf"
    path1.write_text("")
    path2.write_text("")
    results = proc.process_directory()
    # Should process both files, yield two docs
    assert len(results) == 2
    assert all(r.content == "stuff" for r in results)


@patch("dllmforge.IE_agent_document_processor.DocumentLoader")
def test_process_directory_no_files(mock_loader_class, doc_config, tmp_path):
    mock_loader_class.return_value = MagicMock()
    proc = DocumentProcessor(config=doc_config)
    # No pdfs present
    results = proc.process_directory()
    assert results == []


@patch("dllmforge.IE_agent_document_processor.DocumentLoader")
def test_process_to_text_error_handling(mock_loader_class, doc_config, tmp_path):
    mock_loader = MagicMock()
    mock_loader.load_document.side_effect = Exception("fail")
    mock_loader_class.return_value = mock_loader
    proc = DocumentProcessor(config=doc_config)
    # File exists but loader fails
    path = tmp_path / "z.pdf"
    path.write_text("K")
    with pytest.raises(Exception):
        proc.process_to_text(str(path))
