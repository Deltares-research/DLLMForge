import pytest
from unittest.mock import patch, MagicMock
from dllmforge.rag_embedding import AzureOpenAIEmbeddingModel


class TestAzureOpenAIEmbeddingModel:
    """Test class for AzureOpenAIEmbeddingModel functionality."""

    @patch('dllmforge.rag_embedding.load_dotenv')
    @patch('dllmforge.rag_embedding.AzureOpenAIEmbeddings')
    @patch.dict(
        'os.environ', {
            'AZURE_OPENAI_API_BASE': 'https://test.openai.azure.com/',
            'AZURE_OPENAI_DEPLOYMENT_EMBEDDINGS': 'test-embedding-deployment',
            'AZURE_OPENAI_API_KEY': 'test-api-key',
            'AZURE_OPENAI_API_VERSION': '2024-02-15-preview'
        })
    def test_initialization_with_default_model(self, mock_embeddings, mock_load_dotenv):
        """Test that the model initializes correctly with default parameters."""
        mock_embeddings_instance = MagicMock()
        mock_embeddings.return_value = mock_embeddings_instance

        model = AzureOpenAIEmbeddingModel()

        mock_load_dotenv.assert_called_once()
        mock_embeddings.assert_called_once_with(model="text-embedding-3-large",
                                                azure_endpoint='https://test.openai.azure.com/',
                                                azure_deployment='test-embedding-deployment',
                                                api_key='test-api-key',
                                                openai_api_version='2024-02-15-preview')
        assert model.embeddings == mock_embeddings_instance

    @patch('dllmforge.rag_embedding.load_dotenv')
    @patch('dllmforge.rag_embedding.AzureOpenAIEmbeddings')
    @patch.dict(
        'os.environ', {
            'AZURE_OPENAI_API_BASE': 'https://test.openai.azure.com/',
            'AZURE_OPENAI_DEPLOYMENT_EMBEDDINGS': 'test-embedding-deployment',
            'AZURE_OPENAI_API_KEY': 'test-api-key',
            'AZURE_OPENAI_API_VERSION': '2024-02-15-preview'
        })
    def test_initialization_with_custom_model(self, mock_embeddings, mock_load_dotenv):
        """Test that the model initializes correctly with custom model name."""
        mock_embeddings_instance = MagicMock()
        mock_embeddings.return_value = mock_embeddings_instance

        model = AzureOpenAIEmbeddingModel(model="text-embedding-ada-002")

        mock_embeddings.assert_called_once_with(model="text-embedding-ada-002",
                                                azure_endpoint='https://test.openai.azure.com/',
                                                azure_deployment='test-embedding-deployment',
                                                api_key='test-api-key',
                                                openai_api_version='2024-02-15-preview')

    def test_validate_embedding_valid(self):
        """Test that validate_embedding returns True for valid embeddings."""
        valid_embedding = [0.1, 0.2, 0.3, -0.5, 0.8]
        assert AzureOpenAIEmbeddingModel.validate_embedding(valid_embedding) is True

    def test_validate_embedding_empty(self):
        """Test that validate_embedding returns False for empty embeddings."""
        assert AzureOpenAIEmbeddingModel.validate_embedding([]) is False

    def test_validate_embedding_invalid_types(self):
        """Test that validate_embedding returns False for embeddings with invalid types."""
        invalid_embedding = [0.1, "string", 0.3, None, 0.8]
        assert AzureOpenAIEmbeddingModel.validate_embedding(invalid_embedding) is False

    def test_validate_embedding_none(self):
        """Test that validate_embedding returns False for None."""
        assert AzureOpenAIEmbeddingModel.validate_embedding(None) is False

    @patch('dllmforge.rag_embedding.load_dotenv')
    @patch('dllmforge.rag_embedding.AzureOpenAIEmbeddings')
    @patch.dict(
        'os.environ', {
            'AZURE_OPENAI_API_BASE': 'https://test.openai.azure.com/',
            'AZURE_OPENAI_DEPLOYMENT_EMBEDDINGS': 'test-embedding-deployment',
            'AZURE_OPENAI_API_KEY': 'test-api-key',
            'AZURE_OPENAI_API_VERSION': '2024-02-15-preview'
        })
    def test_embed_query_string(self, mock_embeddings, mock_load_dotenv):
        """Test embedding a single query string."""
        mock_embeddings_instance = MagicMock()
        mock_embeddings_instance.embed_query.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_embeddings.return_value = mock_embeddings_instance

        model = AzureOpenAIEmbeddingModel()
        query = "What is the capital of France?"
        result = model.embed(query)

        mock_embeddings_instance.embed_query.assert_called_once_with(query)
        assert result == [0.1, 0.2, 0.3, 0.4, 0.5]

    @patch('dllmforge.rag_embedding.load_dotenv')
    @patch('dllmforge.rag_embedding.AzureOpenAIEmbeddings')
    @patch.dict(
        'os.environ', {
            'AZURE_OPENAI_API_BASE': 'https://test.openai.azure.com/',
            'AZURE_OPENAI_DEPLOYMENT_EMBEDDINGS': 'test-embedding-deployment',
            'AZURE_OPENAI_API_KEY': 'test-api-key',
            'AZURE_OPENAI_API_VERSION': '2024-02-15-preview'
        })
    def test_embed_document_chunks(self, mock_embeddings, mock_load_dotenv):
        """Test embedding a list of document chunks."""
        mock_embeddings_instance = MagicMock()
        mock_embeddings_instance.embed_query.side_effect = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_embeddings.return_value = mock_embeddings_instance

        model = AzureOpenAIEmbeddingModel()
        chunks = [{
            "text": "First chunk text",
            "file_name": "test.pdf",
            "page_number": 1,
            "chunk_index": 0
        }, {
            "text": "Second chunk text",
            "file_name": "test.pdf",
            "page_number": 2,
            "chunk_index": 1
        }]

        result = model.embed(chunks)

        assert len(result) == 2
        assert result[0]["chunk_id"] == "dGVzdA_i0"
        assert result[0]["chunk"] == "First chunk text"
        assert result[0]["page_number"] == 1
        assert result[0]["file_name"] == "test.pdf"
        assert result[0]["text_vector"] == [0.1, 0.2, 0.3]

        assert result[1]["chunk_id"] == "dGVzdA_i1"
        assert result[1]["chunk"] == "Second chunk text"
        assert result[1]["page_number"] == 2
        assert result[1]["file_name"] == "test.pdf"
        assert result[1]["text_vector"] == [0.4, 0.5, 0.6]

    @patch('dllmforge.rag_embedding.load_dotenv')
    @patch('dllmforge.rag_embedding.AzureOpenAIEmbeddings')
    @patch.dict(
        'os.environ', {
            'AZURE_OPENAI_API_BASE': 'https://test.openai.azure.com/',
            'AZURE_OPENAI_DEPLOYMENT_EMBEDDINGS': 'test-embedding-deployment',
            'AZURE_OPENAI_API_KEY': 'test-api-key',
            'AZURE_OPENAI_API_VERSION': '2024-02-15-preview'
        })
    def test_embed_query_invalid_embedding(self, mock_embeddings, mock_load_dotenv):
        """Test that embed raises ValueError when invalid embedding is generated for query."""
        mock_embeddings_instance = MagicMock()
        mock_embeddings_instance.embed_query.return_value = []  # Invalid empty embedding
        mock_embeddings.return_value = mock_embeddings_instance

        model = AzureOpenAIEmbeddingModel()
        query = "What is the capital of France?"

        with pytest.raises(ValueError, match="Invalid embedding generated for query."):
            model.embed(query)

    @patch('dllmforge.rag_embedding.load_dotenv')
    @patch('dllmforge.rag_embedding.AzureOpenAIEmbeddings')
    @patch.dict(
        'os.environ', {
            'AZURE_OPENAI_API_BASE': 'https://test.openai.azure.com/',
            'AZURE_OPENAI_DEPLOYMENT_EMBEDDINGS': 'test-embedding-deployment',
            'AZURE_OPENAI_API_KEY': 'test-api-key',
            'AZURE_OPENAI_API_VERSION': '2024-02-15-preview'
        })
    def test_embed_chunks_invalid_embedding(self, mock_embeddings, mock_load_dotenv):
        """Test that embed raises ValueError when invalid embedding is generated for chunks."""
        mock_embeddings_instance = MagicMock()
        mock_embeddings_instance.embed_query.side_effect = [
            [0.1, 0.2, 0.3],  # Valid first embedding
            []  # Invalid empty embedding for second chunk
        ]
        mock_embeddings.return_value = mock_embeddings_instance

        model = AzureOpenAIEmbeddingModel()
        chunks = [{
            "text": "First chunk text",
            "file_name": "test.pdf",
            "page_number": 1,
            "chunk_index": 0
        }, {
            "text": "Second chunk text",
            "file_name": "test.pdf",
            "page_number": 2,
            "chunk_index": 1
        }]

        with pytest.raises(ValueError, match="Invalid embedding generated for chunk:"):
            model.embed(chunks)

    @patch('dllmforge.rag_embedding.load_dotenv')
    @patch('dllmforge.rag_embedding.AzureOpenAIEmbeddings')
    @patch.dict(
        'os.environ', {
            'AZURE_OPENAI_API_BASE': 'https://test.openai.azure.com/',
            'AZURE_OPENAI_DEPLOYMENT_EMBEDDINGS': 'test-embedding-deployment',
            'AZURE_OPENAI_API_KEY': 'test-api-key',
            'AZURE_OPENAI_API_VERSION': '2024-02-15-preview'
        })
    def test_embed_invalid_input_type(self, mock_embeddings, mock_load_dotenv):
        """Test that embed raises ValueError for invalid input types."""
        mock_embeddings_instance = MagicMock()
        mock_embeddings.return_value = mock_embeddings_instance

        model = AzureOpenAIEmbeddingModel()

        # Test with integer
        with pytest.raises(ValueError, match="Input must be a string or a list of dictionaries."):
            model.embed(123)

        # Test with list containing non-dict items
        with pytest.raises(ValueError, match="Input must be a string or a list of dictionaries."):
            model.embed(["string", 123, {"text": "valid"}])

    @patch('dllmforge.rag_embedding.load_dotenv')
    @patch('dllmforge.rag_embedding.AzureOpenAIEmbeddings')
    @patch.dict(
        'os.environ', {
            'AZURE_OPENAI_API_BASE': 'https://test.openai.azure.com/',
            'AZURE_OPENAI_DEPLOYMENT_EMBEDDINGS': 'test-embedding-deployment',
            'AZURE_OPENAI_API_KEY': 'test-api-key',
            'AZURE_OPENAI_API_VERSION': '2024-02-15-preview'
        })
    def test_embed_empty_chunks_list(self, mock_embeddings, mock_load_dotenv):
        """Test embedding an empty list of chunks."""
        mock_embeddings_instance = MagicMock()
        mock_embeddings.return_value = mock_embeddings_instance

        model = AzureOpenAIEmbeddingModel()
        chunks = []

        result = model.embed(chunks)
        assert result == []
        # Should not call embed_query for empty list
        mock_embeddings_instance.embed_query.assert_not_called()

    @patch('dllmforge.rag_embedding.load_dotenv')
    @patch('dllmforge.rag_embedding.AzureOpenAIEmbeddings')
    @patch.dict(
        'os.environ', {
            'AZURE_OPENAI_API_BASE': 'https://test.openai.azure.com/',
            'AZURE_OPENAI_DEPLOYMENT_EMBEDDINGS': 'test-embedding-deployment',
            'AZURE_OPENAI_API_KEY': 'test-api-key',
            'AZURE_OPENAI_API_VERSION': '2024-02-15-preview'
        })
    def test_embed_chunks_with_special_characters(self, mock_embeddings, mock_load_dotenv):
        """Test embedding chunks with special characters and unicode."""
        mock_embeddings_instance = MagicMock()
        mock_embeddings_instance.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_embeddings.return_value = mock_embeddings_instance

        model = AzureOpenAIEmbeddingModel()
        chunks = [{
            "text": "Special chars: !@#$%^&*()",
            "file_name": "test.pdf",
            "page_number": 1,
            "chunk_index": 0
        }, {
            "text": "Unicode: 你好世界 🌍",
            "file_name": "test.pdf",
            "page_number": 2,
            "chunk_index": 1
        }]

        result = model.embed(chunks)

        assert len(result) == 2
        assert result[0]["chunk"] == "Special chars: !@#$%^&*()"
        assert result[1]["chunk"] == "Unicode: 你好世界 🌍"
        assert mock_embeddings_instance.embed_query.call_count == 2
