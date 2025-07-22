"""
This module provides "search" and "response" functionality for RAG (Retrieval-Augmented Generation) pipelines.
Two steps are involved:
1. Search: use search service to retrieve relevant chunks from the vector database.
2. Response: use LLMs to generate a response to the user query based on the retrieved chunks.

The search module uses Azure AI search service as an example of using hosted search APIs. Note you need
Azure AI search service and a deployed search index (vector database) on Azure to use this module.

The response module uses Azure OpenAI service as an example of using hosted LLMs APIs. Note you need
Azure OpenAI service and a deployed LLM model on Azure to use this module.

The example demonstrates the whole pipeline of RAG, including:
1. Preprocess the documents to chunks.
2. Vectorize the chunks.
3. Store the chunks in the vector database.
4. Search the vector database for relevant chunks.
5. Generate a response to the user query based on the retrieved chunks.
"""

from azure.search.documents.models import VectorizedQuery  
from azure.search.documents import SearchClient
from rag_embedding import AzureOpenAIEmbeddingModel
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate  
import os
from dotenv import load_dotenv

# Optionally load environment variables from a .env file
load_dotenv()

# Configuration from environment variables  
api_key = os.getenv('AZURE_OPENAI_API_KEY')  
deployment_name_embeddings = os.getenv('AZURE_OPENAI_DEPLOYMENT_EMBEDDINGS')  
api_base = os.getenv('AZURE_OPENAI_API_BASE')  
api_version = os.getenv('AZURE_OPENAI_API_VERSION')  
search_client_endpoint = os.getenv('AZURE_SEARCH_ENDPOINT')  
search_api_key = os.getenv('AZURE_SEARCH_API_KEY')  
deployment_name_gpt4o = os.getenv('AZURE_OPENAI_DEPLOYMENT_GPT4')  
  
# Validate required environment variables  
required_vars = [  
    'AZURE_OPENAI_API_KEY',  
    'AZURE_OPENAI_API_BASE',  
    'AZURE_SEARCH_ENDPOINT',  
    'AZURE_SEARCH_API_KEY'  
]  
for var in required_vars:  
    if not os.getenv(var):  
        raise ValueError(f"Missing required environment variable: {var}")  
  

class Retriever:
    def __init__(self, search_client, embedding_model):
        self.search_client = search_client
        self.embedding_model = embedding_model

    def get_embeddings(self, text):
        text_vectorized = self.embedding_model.embed(text)
        return text_vectorized

    def search(self, query_text, top_k=5):
        query_vectorized = self.get_embeddings(query_text)
        vector_query = VectorizedQuery(vector=query_vectorized, k_nearest_neighbors=top_k, fields="text_vector")
        results = self.search_client.search(
            search_text=None,  # pure vector search, no text search. If you want to do text search, set search_text=query_text.
            vector_queries=[vector_query],
            select=["chunk_id", "chunk", "page_number", "file_name"], #The list of fields to retrieve. 
            top=top_k #The number of auto-completed terms to retrieve. This must be a value between 1 and 100. The default is 5.
        )
        return list(results)

class LLMResponder:
    def __init__(self, llm):
        self.llm = llm

    def augment_prompt_with_context(self, query_text, chunks):
        chunks_text = "\n\n".join(f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(chunks))
        sys_prompt = SystemMessagePromptTemplate.from_template("You are a helpful assistant that answers questions based on the provided context.")
        human_prompt = HumanMessagePromptTemplate.from_template('Question: "{query_text}"\n\nContext:\n{chunks_text}')
        full_prompt = ChatPromptTemplate.from_messages([sys_prompt, human_prompt])
        complete_prompt = full_prompt.format_prompt(query_text=query_text, chunks_text=chunks_text).to_messages()
        return complete_prompt

    def generate(self, query_text, retrieved_chunks):
        prompt = self.augment_prompt_with_context(query_text, retrieved_chunks)
        response = self.llm(prompt)
        return response

if __name__ == "__main__":
    # Example demonstration RAGpipeline
    # 

    
    # Initialize the embeddings object  
    embedding_model_name = "text-embedding-3-large"  
    embedding_model = AzureOpenAIEmbeddingModel(model=embedding_model_name)
    # 1. Initialize search client
    search_client = SearchClient(
        endpoint=search_client_endpoint,
        index_name="your-index-name",  # Replace with your actual index name
        credential=search_api_key
    )
    # 2. Initialize retriever and responder
    retriever = Retriever(search_client, embedding_model)
    from langchain.chat_models import AzureChatOpenAI
    llm = AzureChatOpenAI(
        azure_endpoint=api_base,
        api_key=api_key,
        api_version=api_version,
        azure_deployment=deployment_name_gpt4o,
        temperature=0.2
    )
    llm = None  # Replace with actual LLM instance
    responder = LLMResponder(llm)
    # 3. Run a query
    query = "Your user query here"
    top_k = 5
    results = retriever.search(query, top_k=top_k)
    answer = responder.generate(query, results)
    print("Generated response:")
    print(answer)

     
