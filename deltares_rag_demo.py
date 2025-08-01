#!/usr/bin/env python3
"""
Deltares RAG Demo - Fun and Educational RAG Evaluation

This script demonstrates RAG performance across different configurations:
- Uses existing Deltares indexes in Azure AI Search
- Compares 6 different configurations (3 chunking strategies × 2 embedding models)
- Allows interactive querying with RAGAS evaluation

Perfect for showing colleagues how different configurations affect RAG performance!
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# Import DLLMForge modules
from dllmforge.rag_embedding import AzureOpenAIEmbeddingModel
from dllmforge.rag_search_and_response import Retriever, LLMResponder
from dllmforge.openai_api import OpenAIAPI
from dllmforge.rag_evaluation import RAGEvaluator, evaluate_rag_response

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

# Base index name (should match what was used in build_deltares_indexes.py)
BASE_INDEX_NAME = "deltares_rag_demo"

# Expected index configurations
EXPECTED_INDEXES = [
    "small_chunks_ada002",
    "small_chunks_text_embedding_3_large", 
    "medium_chunks_ada002",
    "medium_chunks_text_embedding_3_large",
    "large_chunks_ada002",
    "large_chunks_text_embedding_3_large"
]

# Embedding models (for creating retrievers)
EMBEDDING_MODELS = {
    "ada002": {
        "deployment": "AZURE_OPENAI_DEPLOYMENT_EMBEDDINGS_ADA002",
        "model_name": "text-embedding-ada-002",
        "dimensions": 1536
    },
    "text_embedding_3_large": {
        "deployment": "AZURE_OPENAI_DEPLOYMENT_EMBEDDINGS", 
        "model_name": "text-embedding-3-large",
        "dimensions": 3072
    }
}

# =============================================================================


class DeltaresRAGDemo:
    """Main class for the Deltares RAG demonstration."""
    
    def __init__(self, index_info_file: Optional[str] = None):
        """Initialize the RAG demo."""
        # Check environment variables
        self._check_environment()
        
        # Initialize components
        self.openai_api = OpenAIAPI()
        
        # Azure configuration
        self.search_endpoint = os.getenv('AZURE_SEARCH_ENDPOINT')
        self.search_api_key = os.getenv('AZURE_SEARCH_API_KEY')
        
        if not all([self.search_endpoint, self.search_api_key]):
            raise ValueError("Azure Search credentials not found in environment variables")
        
        # Load index information
        self.index_info = self._load_index_info(index_info_file)
        
        print("🚀 Deltares RAG Demo initialized")
    
    def _check_environment(self):
        """Check that required environment variables are set."""
        required_vars = [
            'AZURE_OPENAI_API_KEY',
            'AZURE_OPENAI_API_BASE',
            'AZURE_SEARCH_ENDPOINT',
            'AZURE_SEARCH_API_KEY'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")
        
        # Check embedding model deployments
        embedding_deployments = []
        for model_config in EMBEDDING_MODELS.values():
            deployment_var = model_config["deployment"]
            if os.getenv(deployment_var):
                embedding_deployments.append(deployment_var)
        
        if not embedding_deployments:
            raise ValueError("No embedding model deployments found in environment variables")
        
        print(f"✅ Found {len(embedding_deployments)} embedding model deployments")
    
    def _load_index_info(self, index_info_file: Optional[str] = None) -> Dict[str, Any]:
        """Load index information from file or use defaults."""
        if index_info_file and os.path.exists(index_info_file):
            print(f"📂 Loading index info from: {index_info_file}")
            with open(index_info_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            print("📋 Using default index configuration")
            return {
                "base_index_name": BASE_INDEX_NAME,
                "indexes": {
                    index_key: {
                        "name": f"{BASE_INDEX_NAME}_{index_key}",
                        "status": "expected",
                        "strategy": index_key.split('_')[0] + '_' + index_key.split('_')[1],
                        "model": '_'.join(index_key.split('_')[2:])
                    }
                    for index_key in EXPECTED_INDEXES
                }
            }
    
    def get_available_indexes(self) -> Dict[str, Any]:
        """Get list of available indexes."""
        available_indexes = {}
        
        for index_key, index_config in self.index_info["indexes"].items():
            index_name = index_config["name"]
            
            # Check if index exists by trying to create a retriever
            try:
                model_key = index_config["model"]
                embedding_model = self.create_embedding_model(model_key)
                
                retriever = Retriever(
                    embedding_model,
                    index_name,
                    self.search_endpoint,
                    self.search_api_key
                )
                
                # Test connection
                retriever.search("test", top_k=1)
                
                available_indexes[index_key] = {
                    "name": index_name,
                    "retriever": retriever,
                    "config": index_config
                }
                
                print(f"  ✅ Found index: {index_name}")
                
            except Exception as e:
                print(f"  ❌ Index not found: {index_name} ({e})")
                continue
        
        return available_indexes
        """Prepare scraped content for chunking."""
        print("\n📝 Preparing content for chunking...")
        
        prepared_content = []
        
        for page in scraped_content:
            # Combine title, description, and content
            full_text = f"Title: {page['title']}\n\n"
            if page['description']:
                full_text += f"Description: {page['description']}\n\n"
            full_text += f"Content: {page['content']}"
            
            prepared_content.append({
                "text": full_text,
                "file_name": f"deltares_{page['url'].split('/')[-2] if page['url'].split('/')[-2] else 'home'}",
                "page_number": 1,
                "url": page['url'],
                "title": page['title']
            })
        
        print(f"✅ Prepared {len(prepared_content)} documents for chunking")
        return prepared_content
    
    def create_embedding_model(self, model_key: str) -> AzureOpenAIEmbeddingModel:
        """Create an embedding model instance."""
        model_config = EMBEDDING_MODELS[model_key]
        deployment_var = model_config["deployment"]
        
        # Set the deployment environment variable temporarily
        original_deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT_EMBEDDINGS')
        os.environ['AZURE_OPENAI_DEPLOYMENT_EMBEDDINGS'] = os.getenv(deployment_var)
        
        try:
            embedding_model = AzureOpenAIEmbeddingModel()
            return embedding_model
        finally:
            # Restore original deployment
            if original_deployment:
                os.environ['AZURE_OPENAI_DEPLOYMENT_EMBEDDINGS'] = original_deployment
            else:
                os.environ.pop('AZURE_OPENAI_DEPLOYMENT_EMBEDDINGS', None)
    
    def interactive_query(self, available_indexes: Dict[str, Any]):
        """Allow user to interactively query all indexes."""
        print("\n" + "=" * 60)
        print("🎯 INTERACTIVE QUERY MODE")
        print("=" * 60)
        
        llm_responder = LLMResponder(llm=self.openai_api)
        evaluator = RAGEvaluator(llm_provider="auto")
        
        while True:
            print("\n📝 Enter your question about Deltares (or 'quit' to exit):")
            query = input("> ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            print(f"\n🔍 Searching across all {len(available_indexes)} configurations...")
            print("=" * 60)
            
            results = {}
            
            # Query all indexes
            for index_key, index_data in available_indexes.items():
                print(f"\n📊 Testing: {index_key}")
                
                try:
                    retriever = index_data["retriever"]
                    
                    # Search
                    search_results = retriever.search(query, top_k=5)
                    
                    if not search_results:
                        print("  ⚠️  No results found")
                        continue
                    
                    # Generate response
                    llm_response = llm_responder.generate(query, search_results)
                    
                    # Extract contexts for evaluation
                    contexts = [result["chunk"] for result in search_results]
                    
                    # Evaluate with RAGAS (no ground truth for interactive mode)
                    evaluation_result = evaluator.evaluate_rag_pipeline(
                        question=query,
                        generated_answer=llm_response,
                        retrieved_contexts=contexts
                    )
                    
                    results[index_key] = {
                        "response": llm_response,
                        "contexts": contexts,
                        "evaluation": evaluation_result,
                        "search_results": search_results
                    }
                    
                    print(f"  🎯 RAGAS Score: {evaluation_result.ragas_score:.3f}")
                    print(f"  📄 Retrieved {len(search_results)} chunks")
                    print(f"  💬 Response: {llm_response[:100]}...")
                    
                except Exception as e:
                    print(f"  ❌ Error: {e}")
                    continue
            
            # Compare results
            if results:
                self.compare_results(query, results)
            
            # Ask if user wants to continue
            print(f"\n🔄 Query another question? (y/n): ", end="")
            if input().lower() not in ['y', 'yes']:
                break
    
    def compare_results(self, query: str, results: Dict[str, Any]):
        """Compare results across all configurations."""
        print(f"\n📊 COMPARISON RESULTS FOR: '{query}'")
        print("=" * 80)
        
        # Sort by RAGAS score
        sorted_results = sorted(
            results.items(), 
            key=lambda x: x[1]["evaluation"].ragas_score, 
            reverse=True
        )
        
        print(f"{'Configuration':<25} {'RAGAS':<8} {'Context Rel':<12} {'Faithfulness':<12} {'Answer Rel':<12}")
        print("-" * 80)
        
        for index_key, result in sorted_results:
            eval_result = result["evaluation"]
            print(f"{index_key:<25} {eval_result.ragas_score:<8.3f} "
                  f"{eval_result.context_relevancy.score:<12.3f} "
                  f"{eval_result.faithfulness.score:<12.3f} "
                  f"{eval_result.answer_relevancy.score:<12.3f}")
        
        # Show best configuration
        best_config, best_result = sorted_results[0]
        print(f"\n🏆 Best Configuration: {best_config}")
        print(f"   RAGAS Score: {best_result['evaluation'].ragas_score:.3f}")
        print(f"   Response: {best_result['response'][:200]}...")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"deltares_query_results_{timestamp}.json"
        
        # Prepare results for saving
        save_data = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "results": {}
        }
        
        for index_key, result in results.items():
            save_data["results"][index_key] = {
                "response": result["response"],
                "ragas_score": result["evaluation"].ragas_score,
                "context_relevancy": result["evaluation"].context_relevancy.score,
                "faithfulness": result["evaluation"].faithfulness.score,
                "answer_relevancy": result["evaluation"].answer_relevancy.score,
                "context_count": len(result["contexts"])
            }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {filename}")
    



def main():
    """Main function to run the Deltares RAG demo."""
    parser = argparse.ArgumentParser(description="Deltares RAG Demo - Fun RAG Evaluation")
    parser.add_argument("--index-info", type=str,
                       help="Path to index info JSON file (from build_deltares_indexes.py)")
    
    args = parser.parse_args()
    
    print("🚀 DELTARES RAG DEMO")
    print("=" * 60)
    print("🎯 Fun and educational RAG evaluation across different configurations!")
    print("=" * 60)
    
    try:
        # Initialize demo
        demo = DeltaresRAGDemo(index_info_file=args.index_info)
        
        # Get available indexes
        print("\n🔍 Checking available indexes...")
        available_indexes = demo.get_available_indexes()
        
        if not available_indexes:
            print("❌ No indexes found. Please run build_deltares_indexes.py first.")
            return
        
        print(f"\n✅ Found {len(available_indexes)} indexes ready for demo!")
        
        # Interactive query mode
        demo.interactive_query(available_indexes)
        
        print("\n🎉 Deltares RAG Demo completed!")
        print("   Hope your colleagues enjoyed the demonstration! 😄")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 