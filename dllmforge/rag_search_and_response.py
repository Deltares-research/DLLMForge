"""
This module provides "create index/vector-database", "search" and "response"
functionality for RAG (Retrieval-Augmented Generation) pipelines.
Three steps are involved:
1. Create index/vector-database: create an index/vector-database on Azure AI search service.
1. Search: use Azure AI search service to retrieve relevant chunks from the vector database .
2. Response: use LLMs to generate a response to the user query based on the retrieved chunks.
The module uses Azure AI search service and Azure OpenAI service as an example of using hosted search 
APIs and LLMs APIs. Note you need
Azure AI search service, Azure OpenAI service and a deployed LLM model on Azure to use this module.

The example demonstrates the whole pipeline of RAG, including:
1. Preprocess the documents to chunks.
2. Vectorize the chunks.
3. Create vector index and store the chunks in the vector database.
4. Search the vector database for relevant chunks.
5. Generate a response to the user query based on the retrieved chunks.
"""
import os
from dotenv import load_dotenv

try:
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents.models import VectorizedQuery
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient
    from azure.search.documents.indexes.models import (SearchField, SearchFieldDataType, VectorSearch,
                                                       HnswAlgorithmConfiguration, VectorSearchProfile,
                                                       AzureOpenAIVectorizer, AzureOpenAIVectorizerParameters,
                                                       SearchIndex)
    AZURE_SEARCH_AVAILABLE = True
except ImportError:
    AZURE_SEARCH_AVAILABLE = False
    AzureKeyCredential = None
    VectorizedQuery = None
    SearchClient = None
    SearchIndexClient = None
    SearchField = None
    SearchFieldDataType = None
    VectorSearch = None
    HnswAlgorithmConfiguration = None
    VectorSearchProfile = None
    AzureOpenAIVectorizer = None
    AzureOpenAIVectorizerParameters = None
    SearchIndex = None

from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from .rag_embedding import AzureOpenAIEmbeddingModel

# Optionally load environment variables from a .env file
load_dotenv()

# Remove top-level config variables, use function/class arguments with os.getenv fallback


class IndexManager:

    def __init__(self, search_client_endpoint=None, search_api_key=None, index_name=None, embedding_dim=None):
        self.endpoint = search_client_endpoint or os.getenv('AZURE_SEARCH_ENDPOINT')
        self.search_api_key = search_api_key or os.getenv('AZURE_SEARCH_API_KEY')
        self.index_name = index_name or "dllmforge_index"
        self.embedding_dim = embedding_dim
        self.index_client = SearchIndexClient(endpoint=self.endpoint,
                                              credential=AzureKeyCredential(self.search_api_key))

    def create_index(self, api_base=None, deployment_name_embeddings=None, api_key=None):
        api_base = api_base or os.getenv('AZURE_OPENAI_API_BASE')
        deployment_name_embeddings = deployment_name_embeddings or os.getenv('AZURE_OPENAI_DEPLOYMENT_EMBEDDINGS')
        api_key = api_key or os.getenv('AZURE_OPENAI_API_KEY')
        fields = [
            SearchField(name="chunk_id", type=SearchFieldDataType.String, key=True, sortable=True, filterable=True),
            SearchField(name="chunk", type=SearchFieldDataType.String),
            SearchField(name="page_number", type=SearchFieldDataType.Int32),
            SearchField(name="file_name", type=SearchFieldDataType.String, filterable=True, searchable=True),
            SearchField(name="text_vector",
                        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                        vector_search_dimensions=self.embedding_dim,
                        vector_search_profile_name="myHnswProfile")
        ]
        # Configure vector search with Azure OpenAI credentials
        vector_search = VectorSearch(algorithms=[HnswAlgorithmConfiguration(name="myHnsw")],
                                     profiles=[
                                         VectorSearchProfile(name="myHnswProfile",
                                                             algorithm_configuration_name="myHnsw",
                                                             vectorizer_name="myOpenAI")
                                     ],
                                     vectorizers=[
                                         AzureOpenAIVectorizer(vectorizer_name="myOpenAI",
                                                               kind="azureOpenAI",
                                                               parameters=AzureOpenAIVectorizerParameters(
                                                                   resource_url=api_base,
                                                                   deployment_name=deployment_name_embeddings,
                                                                   model_name=deployment_name_embeddings,
                                                                   api_key=api_key))
                                     ])
        index = SearchIndex(name=self.index_name, fields=fields, vector_search=vector_search)
        try:
            self.index_client.create_or_update_index(index)
            print(f"Index '{self.index_name}' created.")
        except Exception as e:
            print(f"Index creation error: {e}")

    def upload_documents(self, vectorized_chunks):
        search_client = SearchClient(endpoint=self.endpoint,
                                     index_name=self.index_name,
                                     credential=AzureKeyCredential(self.search_api_key))
        try:
            search_client.upload_documents(documents=vectorized_chunks)
            print(f"Uploaded {len(vectorized_chunks)}.")
        except Exception as e:
            print(f"Upload error: {e}")


class Retriever:

    def __init__(self, embedding_model, index_name=None, search_client_endpoint=None, search_api_key=None):
        self.embedding_model = embedding_model
        self.index_name = index_name or os.getenv('AZURE_SEARCH_INDEX_NAME') or "dllmforge_index"
        self.endpoint = search_client_endpoint or os.getenv('AZURE_SEARCH_ENDPOINT')
        self.search_api_key = search_api_key or os.getenv('AZURE_SEARCH_API_KEY')
        self.search_client = SearchClient(endpoint=self.endpoint,
                                          index_name=self.index_name,
                                          credential=AzureKeyCredential(self.search_api_key))

    def get_embeddings(self, text):
        text_vectorized = self.embedding_model.embed(text)
        return text_vectorized

    def invoke(self, query_text, top_k=5):
        query_vectorized = self.get_embeddings(query_text)
        vector_query = VectorizedQuery(vector=query_vectorized, k_nearest_neighbors=top_k, fields="text_vector")
        results = self.search_client.search(
            search_text=
            None,  # pure vector search, no text search. If you want to do text search, set search_text=query_text.
            vector_queries=[vector_query],
            select=["chunk_id", "chunk", "page_number", "file_name"],  # The list of fields to retrieve.
            top=top_k  # The number of auto-completed terms to retrieve.
            # This must be a value between 1 and 100. The default is 5.
        )
        return list(results)


class LLMResponder:

    def __init__(self, llm):
        self.llm = llm

    def augment_prompt_with_context(self, query_text, chunks):
        chunks_text = "\n\n".join(f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(chunks))
        sys_prompt = SystemMessagePromptTemplate.from_template(
            "You are a helpful assistant that answers questions based on the provided context.")
        human_prompt = HumanMessagePromptTemplate.from_template('Question: "{query_text}"\n\nContext:\n{chunks_text}')
        full_prompt = ChatPromptTemplate.from_messages([sys_prompt, human_prompt])
        complete_prompt = full_prompt.format_prompt(query_text=query_text, chunks_text=chunks_text).to_messages()
        return complete_prompt

    def generate(self, query_text, retrieved_chunks):
        prompt = self.augment_prompt_with_context(query_text, retrieved_chunks)
        response = self.llm.invoke(prompt)
        return response.content.strip()
