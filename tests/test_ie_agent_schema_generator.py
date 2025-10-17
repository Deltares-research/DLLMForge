import pytest
from unittest.mock import patch, MagicMock, mock_open
from dllmforge.IE_agent_schema_generator import SchemaGenerator

@pytest.fixture
def schema_config(tmp_path):
    class DummyConfig:
        task_description = "Extract widgets."
        example_doc = "foo.pdf"
        user_schema_path = tmp_path / "user.py"
        output_path = tmp_path / "file.py"
    return DummyConfig()

@patch("dllmforge.IE_agent_schema_generator.LangchainAPI")
def test_schema_generator_init(mock_lc, schema_config):
    s = SchemaGenerator(schema_config, llm_api=MagicMock())
    assert s.config == schema_config
    assert s.llm_api is not None
    assert hasattr(s, "output_parser")

@patch("dllmforge.IE_agent_schema_generator.LangchainAPI")
def test_generate_schema_user_schema(mock_lc, schema_config):
    # User schema path exists, should trigger _load_user_schema
    schema_code = "class Foo: ..."
    with patch("dllmforge.IE_agent_schema_generator.SchemaGenerator._load_user_schema", return_value=schema_code):
        g = SchemaGenerator(schema_config, llm_api=MagicMock())
        out = g.generate_schema()
        assert out == schema_code

@patch("dllmforge.IE_agent_schema_generator.LangchainAPI")
def test_generate_schema_fallback_to_llm(mock_lc, schema_config):
    config = schema_config
    # Simulate user_schema_path yields None, so LLM path is used
    with patch("dllmforge.IE_agent_schema_generator.SchemaGenerator._load_user_schema", return_value=None), \
         patch("dllmforge.IE_agent_schema_generator.SchemaGenerator._load_example_doc", return_value=None), \
         patch.object(SchemaGenerator, "create_schema_generation_prompt") as mock_prompt:
        g = SchemaGenerator(config, llm_api=MagicMock())
        mock_parser = MagicMock()
        g.output_parser = mock_parser  # Patch instance, not class
        prompt = MagicMock()
        mock_prompt.return_value = prompt
        messages = ["msg"]
        prompt.format_messages.return_value = messages
        response = {"response": "class Foo: pass"}
        g.llm_api.chat_completion.return_value = response
        mock_parser.get_format_instructions.return_value = "instructions"
        mock_parser.parse.return_value = "class Foo: pass"
        out = g.generate_schema()
        assert "class Foo" in out

@patch("dllmforge.IE_agent_schema_generator.DocumentLoader")
def test_load_example_doc_file(mock_doc_loader, schema_config, tmp_path):
    # Simulate .pdf exists, loader works
    p = tmp_path / "x.pdf"
    p.write_text("Hello")
    schema_config.example_doc = str(p)
    mock_doc_loader().load_document.return_value = "some text"
    g = SchemaGenerator(schema_config, llm_api=MagicMock())
    result = g._load_example_doc()
    assert result == "some text"

@patch("builtins.open", new_callable=mock_open)
def test_save_schema_opens_and_writes(mock_file, schema_config):
    g = SchemaGenerator(schema_config, llm_api=MagicMock())
    data = "class Foo: ..."
    schema_config.output_path = schema_config.output_path.parent / "outschema.py"
    g.save_schema(data)
    mock_file.assert_called_once()
    handle = mock_file()
    handle.write.assert_called()

def test__load_user_schema_reads(schema_config, tmp_path):
    g = SchemaGenerator(schema_config, llm_api=MagicMock())
    p = tmp_path / "abc.py"
    p.write_text("class Foo: ...")
    out = g._load_user_schema(p)
    assert out == "class Foo: ..."

@patch("builtins.open", new_callable=mock_open)
def test__load_user_schema_file_missing(mock_file, schema_config, tmp_path):
    g = SchemaGenerator(schema_config, llm_api=MagicMock())
    # File does not exist
    not_here = tmp_path / "does_not_exist.py"
    out = g._load_user_schema(not_here)
    assert out is None
