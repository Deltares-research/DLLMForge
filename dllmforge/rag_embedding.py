"""
This module provides embedding functionality for RAG (Retrieval-Augmented Generation) pipelines.
It can be used to 1)vectorize document chunks, and 2)vectorize user queries.
The module uses Azure OpenAI embeddings model as an example of using hosted embedding APIs. Note you need
to have an Azure OpenAI service and deploy the openai embedding model on Azure to use this module.
"""
import os
from dotenv import load_dotenv
from typing import List, Dict, Any
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()

# Initialize the embeddings object  
embeddings_model = "text-embedding-3-large"  
embeddings_3_large = AzureOpenAIEmbeddings(  
    model=embeddings_model,
    azure_endpoint=api_base,
    azure_deployment=deployment_name_embeddings,
    api_key=api_key,  
    openai_api_version=api_version  
)  
  
def get_embeddings(text):  
    response = embeddings_3_large.embed_query(text)  
    return response  