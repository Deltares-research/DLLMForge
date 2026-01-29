import pytest
from unittest.mock import MagicMock, patch
from dllmforge import rag_search_and_response as rag


@pytest.fixture
def mock_index_client():
    with patch('dllmforge.rag_search_and_response.SearchIndexClient') as mock:
        yield mock


@pytest.fixture
def mock_search_client():
    with patch('dllmforge.rag_search_and_response.SearchClient') as mock:
        yield mock


@pytest.fixture
def mock_embedding_model():
    model = MagicMock()
    model.embed.return_value = [0.1, 0.2, 0.3]
    return model


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.invoke.return_value.content.strip.return_value = 'Mocked LLM response.'
    return llm


def test_index_manager_create_index(mock_index_client):
    manager = rag.IndexManager('endpoint', 'key', 'test_index', 3)
    # Should call create_or_update_index without raising
    manager.create_index()
    assert mock_index_client.return_value.create_or_update_index.called


def test_index_manager_upload_documents(mock_search_client):
    manager = rag.IndexManager('endpoint', 'key', 'test_index', 3)
    docs = [{"chunk_id": "id1", "chunk": "text", "page_number": 1, "file_name": "file", "text_vector": [0.1, 0.2, 0.3]}]
    manager.upload_documents(docs)
    assert mock_search_client.return_value.upload_documents.called


def test_retriever_search(mock_search_client, mock_embedding_model):
    # Mock search returns a list of dicts
    mock_search_client.return_value.search.return_value = [{
        "chunk_id": "id1",
        "chunk": "text",
        "page_number": 1,
        "file_name": "file"
    }]
    retriever = rag.Retriever(mock_embedding_model, 'test_index', 'endpoint', 'key')
    results = retriever.invoke('query', top_k=1)
    assert isinstance(results, list)
    assert results[0]["chunk_id"] == "id1"
    assert mock_embedding_model.embed.called
    assert mock_search_client.return_value.search.called


def test_llm_responder_generate(mock_llm):
    responder = rag.LLMResponder(mock_llm)
    chunks = ["This is a chunk."]
    query = "What is this?"
    response = responder.generate(query, chunks)
    assert response == 'Mocked LLM response.'
    assert mock_llm.invoke.called
