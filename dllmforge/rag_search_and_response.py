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
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (SearchField, SearchFieldDataType, VectorSearch,
                                                   HnswAlgorithmConfiguration, VectorSearchProfile,
                                                   AzureOpenAIVectorizer, AzureOpenAIVectorizerParameters, SearchIndex)
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
import os
from dotenv import load_dotenv
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

    def search(self, query_text, top_k=5):
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
        response = self.llm(prompt)
        return response.content.strip()


if __name__ == "__main__":
    # Example demonstration RAGpipeline
    # step 1: preprocess the documents to chunks and embed the chunks
    from pathlib import Path
    from rag_preprocess_documents import *
    data_dir = Path(r'c:\Users\deng_jg\work\16centralized_agents\test_data')
    pdfs = list(data_dir.glob("*.pdf"))  # find all PDF files in the directory
    loader = PDFLoader()  # Load the PDF document
    chunker = TextChunker(chunk_size=1000, overlap_size=200)  # Create chunks with custom settings

    # initialize the embedding model
    model = AzureOpenAIEmbeddingModel()

    # embed the chunks
    global_embeddings = []
    for pdf_path in pdfs:
        pages, file_name = loader.load(pdf_path)
        # Create chunks with custom settings
        chunks = chunker.chunk_text(pages, file_name)
        # Embed the document chunks
        chunk_embeddings = model.embed(chunks)
        global_embeddings.extend(chunk_embeddings)
        print(f"Embedded {len(chunk_embeddings)} chunks from {file_name}.")
    print(f"Total embeddings generated: {len(global_embeddings)}")

    # Index and upload phase
    embedding_dim = 3072  # Adjust if your embedding model uses a different dimension
    index_name = "dllmforge_index"
    index_manager = IndexManager(search_client_endpoint, search_api_key, index_name, embedding_dim)
    index_manager.create_index()
    index_manager.upload_documents(global_embeddings)

    # Retrieval phase
    retriever = Retriever(model, index_name, search_client_endpoint, search_api_key)
    query = "What is the area of the Rhine basin?"
    top_k = 5
    results = retriever.search(query, top_k=top_k)
    print(results)

    from langchain_openai import AzureChatOpenAI
    llm = AzureChatOpenAI(azure_endpoint=api_base,
                          api_key=api_key,
                          azure_deployment=deployment_name_gpt4o,
                          api_version=api_version,
                          temperature=0.1)
    responder = LLMResponder(llm)
    answer = responder.generate(query, results)
    print(answer)
