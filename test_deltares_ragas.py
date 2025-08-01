#!/usr/bin/env python3
"""
Deltares RAGAS Test - Focused Evaluation

This script runs RAGAS evaluation on the completed Deltares indexes:
- deltares_rag_demo_chunks_100_overlap_10_ada002
- deltares_rag_demo_chunks_100_overlap_10_text_embedding_3_large  
- deltares_rag_demo_chunks_300_overlap_30_ada002

Tests retrieval and generation quality using RAGAS metrics.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List
from datetime import datetime
from dotenv import load_dotenv

# Import DLLMForge modules
from dllmforge.rag_embedding import AzureOpenAIEmbeddingModel
from dllmforge.rag_search_and_response import Retriever, LLMResponder
from dllmforge.rag_evaluation import RAGEvaluator, evaluate_rag_response

# Import LangChain for LLM
from langchain_openai import AzureChatOpenAI

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

# Test indexes (the ones you mentioned are ready)
TEST_INDEXES = [
    "deltares_rag_demo_chunks_100_overlap_10_ada002",
    "deltares_rag_demo_chunks_100_overlap_10_text_embedding_3_large",
    "deltares_rag_demo_chunks_300_overlap_30_ada002"
]

# Test questions for Deltares content
TEST_QUESTIONS = [
    {
        "question": "What does Deltares do?",
        "ground_truth": "Deltares is a leading research institute that focuses on water management, subsurface and infrastructure. They provide expertise and innovative solutions for challenges related to water, soil and subsurface."
    },
    {
        "question": "What are Deltares' main research areas?",
        "ground_truth": "Deltares focuses on water management, subsurface and infrastructure research. They work on flood risk management, coastal protection, groundwater management, and sustainable infrastructure development."
    },
    {
        "question": "How does Deltares help with climate adaptation?",
        "ground_truth": "Deltares helps with climate adaptation through research on flood risk management, coastal protection, water management strategies, and developing tools and models for climate resilience."
    },
    {
        "question": "What is Deltares' approach to sustainable development?",
        "ground_truth": "Deltares promotes sustainable development through research on water management, environmental protection, infrastructure resilience, and developing solutions that balance economic, social and environmental needs."
    },
    {
        "question": "Who is Gualbert Oude Essink?",
        "ground_truth": "Gualbert Oude Essink, PhD (Civil Engineering, Delft University of Technology) is senior hydrogeologist at Deltares and associate professor at the Utrecht University."
    }
]

# Embedding models configuration
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


class DeltaresRAGASTest:
    """RAGAS evaluation for Deltares indexes."""
    
    def __init__(self):
        """Initialize the RAGAS test."""
        # Check environment variables
        self._check_environment()
        
        # Initialize components
        self.llm = AzureChatOpenAI(
            azure_endpoint=os.getenv('AZURE_OPENAI_API_BASE'),
            api_key=os.getenv('AZURE_OPENAI_API_KEY'),
            azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT_GPT4o'),  # Use the correct deployment
            api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
            temperature=0.1
        )
        
        # Azure configuration
        self.search_endpoint = os.getenv('AZURE_SEARCH_ENDPOINT')
        self.search_api_key = os.getenv('AZURE_SEARCH_API_KEY')
        
        if not all([self.search_endpoint, self.search_api_key]):
            raise ValueError("Azure Search credentials not found in environment variables")
        
        # Initialize RAGAS evaluator
        self.evaluator = RAGEvaluator()
        
        print("🚀 Deltares RAGAS Test initialized")
    
    def _check_environment(self):
        """Check that required environment variables are set."""
        required_vars = [
            'AZURE_OPENAI_API_KEY',
            'AZURE_OPENAI_API_BASE',
            'AZURE_OPENAI_DEPLOYMENT_GPT4o',  # Use the correct deployment
            'AZURE_SEARCH_ENDPOINT',
            'AZURE_SEARCH_API_KEY'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")
        
        # Debug: Print the deployment name being used
        deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_GPT4o')
        print(f"✅ Environment variables checked")
        print(f"🔧 Using LLM deployment: {deployment_name}")
        
        # Check embedding model deployments
        embedding_deployments = []
        for model_config in EMBEDDING_MODELS.values():
            deployment_var = model_config["deployment"]
            deployment_value = os.getenv(deployment_var)
            if deployment_value:
                embedding_deployments.append(f"{deployment_var}={deployment_value}")
            else:
                print(f"⚠️  Missing embedding deployment: {deployment_var}")
        
        if not embedding_deployments:
            raise ValueError("No embedding model deployments found in environment variables")
        
        print(f"✅ Found embedding deployments: {', '.join(embedding_deployments)}")
    
    def get_index_model(self, index_name: str) -> str:
        """Extract the embedding model from index name."""
        if "ada002" in index_name:
            return "ada002"
        elif "text_embedding_3_large" in index_name:
            return "text_embedding_3_large"
        else:
            raise ValueError(f"Unknown model in index name: {index_name}")
    
    def create_retriever(self, index_name: str) -> Retriever:
        """Create a retriever for a specific index."""
        model_key = self.get_index_model(index_name)
        model_config = EMBEDDING_MODELS[model_key]
        
        # Set the deployment environment variable temporarily
        original_deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT_EMBEDDINGS')
        os.environ['AZURE_OPENAI_DEPLOYMENT_EMBEDDINGS'] = os.getenv(model_config["deployment"])
        
        try:
            # Create embedding model with the correct model name
            embedding_model = AzureOpenAIEmbeddingModel(model=model_config["model_name"])
            
            # Create retriever (correct parameter order)
            retriever = Retriever(
                embedding_model,
                index_name,
                self.search_endpoint,
                self.search_api_key
            )
            
            return retriever
        finally:
            # Restore original deployment
            if original_deployment:
                os.environ['AZURE_OPENAI_DEPLOYMENT_EMBEDDINGS'] = original_deployment
            else:
                os.environ.pop('AZURE_OPENAI_DEPLOYMENT_EMBEDDINGS', None)
    
    def create_llm_responder(self) -> LLMResponder:
        """Create LLM responder for generation."""
        return LLMResponder(self.llm)
    
    def test_single_index(self, index_name: str, questions: List[Dict[str, str]]) -> Dict[str, Any]:
        """Test a single index with all questions."""
        print(f"\n🔍 Testing index: {index_name}")
        print("-" * 60)
        
        # Create components
        retriever = self.create_retriever(index_name)
        llm_responder = self.create_llm_responder()
        
        results = []
        total_scores = {
            'context_relevancy': 0.0,
            'context_recall': 0.0,
            'faithfulness': 0.0,
            'answer_relevancy': 0.0,
            'ragas_score': 0.0
        }
        
        for i, q_data in enumerate(questions, 1):
            question = q_data["question"]
            ground_truth = q_data["ground_truth"]
            
            print(f"\n📝 Question {i}/{len(questions)}: {question}")
            
            try:
                # Retrieve relevant documents
                retrieved_docs = retriever.search(question, top_k=5)
                
                # Generate response
                response = llm_responder.generate(question, retrieved_docs)
                
                # Convert retrieved docs to list of strings for evaluation
                retrieved_contexts = [doc['chunk'] for doc in retrieved_docs]
                
                # Evaluate with RAGAS
                evaluation = evaluate_rag_response(
                    question=question,
                    generated_answer=response,
                    retrieved_contexts=retrieved_contexts,
                    ground_truth_answer=ground_truth
                )
                
                # Store results
                result = {
                    'question': question,
                    'ground_truth': ground_truth,
                    'response': response,
                    'context': retrieved_docs,
                    'evaluation': evaluation
                }
                results.append(result)
                
                # Accumulate scores (extract score values from EvaluationResult objects)
                for metric in total_scores.keys():
                    if hasattr(evaluation, metric):
                        metric_value = getattr(evaluation, metric)
                        if hasattr(metric_value, 'score'):  # It's an EvaluationResult object
                            total_scores[metric] += metric_value.score
                        else:  # It's a direct float value
                            total_scores[metric] += metric_value
                
                # Print results (extract score values from EvaluationResult objects)
                def get_score(metric_value):
                    if hasattr(metric_value, 'score'):
                        return metric_value.score
                    return metric_value
                
                print(f"  🎯 RAGAS Score: {evaluation.ragas_score:.3f}")
                print(f"  📊 Context Relevancy: {get_score(evaluation.context_relevancy):.3f}")
                print(f"  📊 Context Recall: {get_score(evaluation.context_recall):.3f}")
                print(f"  📊 Faithfulness: {get_score(evaluation.faithfulness):.3f}")
                print(f"  📊 Answer Relevancy: {get_score(evaluation.answer_relevancy):.3f}")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                continue
        
        # Calculate averages
        num_questions = len(results)
        if num_questions > 0:
            for metric in total_scores.keys():
                total_scores[metric] /= num_questions
        
        return {
            'index_name': index_name,
            'results': results,
            'average_scores': total_scores,
            'num_questions': num_questions
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run RAGAS tests on all specified indexes."""
        print("🚀 DELTARES RAGAS TEST")
        print("=" * 60)
        print(f"🎯 Testing {len(TEST_INDEXES)} indexes with {len(TEST_QUESTIONS)} questions")
        print("=" * 60)
        
        all_results = {}
        
        for index_name in TEST_INDEXES:
            try:
                result = self.test_single_index(index_name, TEST_QUESTIONS)
                all_results[index_name] = result
            except Exception as e:
                print(f"❌ Failed to test {index_name}: {e}")
                continue
        
        return all_results
    
    def print_comparison(self, results: Dict[str, Any]):
        """Print comprehensive comparison of all results."""
        print("\n" + "=" * 100)
        print("📊 DELTARES RAGAS EVALUATION - COMPREHENSIVE RESULTS")
        print("=" * 100)
        
        # Parse index configurations
        index_configs = {}
        for index_name in results.keys():
            if "chunks_100_overlap_10_ada002" in index_name:
                index_configs[index_name] = {"chunk_size": 100, "overlap": 10, "model": "ada002", "dimensions": 1536}
            elif "chunks_100_overlap_10_text_embedding_3_large" in index_name:
                index_configs[index_name] = {"chunk_size": 100, "overlap": 10, "model": "text-embedding-3-large", "dimensions": 3072}
            elif "chunks_300_overlap_30_ada002" in index_name:
                index_configs[index_name] = {"chunk_size": 300, "overlap": 30, "model": "ada002", "dimensions": 1536}
            elif "chunks_500_overlap_50_ada002" in index_name:
                index_configs[index_name] = {"chunk_size": 500, "overlap": 50, "model": "ada002", "dimensions": 1536}
            elif "chunks_300_overlap_30_text_embedding_3_large" in index_name:
                index_configs[index_name] = {"chunk_size": 300, "overlap": 30, "model": "text-embedding-3-large", "dimensions": 3072}
            elif "chunks_500_overlap_50_text_embedding_3_large" in index_name:
                index_configs[index_name] = {"chunk_size": 500, "overlap": 50, "model": "text-embedding-3-large", "dimensions": 3072}
        
        # Detailed header
        print(f"{'Configuration':<25} {'Chunk':<8} {'Overlap':<8} {'Model':<20} {'RAGAS':<8} {'Ctx Rel':<8} {'Ctx Recall':<10} {'Faith':<8} {'Ans Rel':<8}")
        print("-" * 100)
        
        # Results with configuration details
        for index_name, result in results.items():
            scores = result['average_scores']
            config = index_configs.get(index_name, {})
            
            # Create readable configuration string
            config_str = f"{config.get('chunk_size', 'N/A')} chars, {config.get('overlap', 'N/A')} overlap"
            model_str = config.get('model', 'unknown')
            
            print(f"{config_str:<25} {config.get('chunk_size', 'N/A'):<8} {config.get('overlap', 'N/A'):<8} "
                  f"{model_str:<20} {scores['ragas_score']:<8.3f} {scores['context_relevancy']:<8.3f} "
                  f"{scores['context_recall']:<10.3f} {scores['faithfulness']:<8.3f} {scores['answer_relevancy']:<8.3f}")
        
        # Find best performer
        best_index = max(results.keys(), 
                        key=lambda x: results[x]['average_scores']['ragas_score'])
        best_score = results[best_index]['average_scores']['ragas_score']
        best_config = index_configs.get(best_index, {})
        
        print("-" * 100)
        print(f"🏆 BEST PERFORMER: {best_index}")
        print(f"   Configuration: {best_config.get('chunk_size', 'N/A')} characters, {best_config.get('overlap', 'N/A')} overlap, {best_config.get('model', 'unknown')} model")
        print(f"   RAGAS Score: {best_score:.3f}")
        
        # Detailed analysis
        print("\n" + "=" * 100)
        print("📈 DETAILED ANALYSIS")
        print("=" * 100)
        
        # Sort by RAGAS score
        sorted_results = sorted(results.items(), key=lambda x: x[1]['average_scores']['ragas_score'], reverse=True)
        
        print("🎯 PERFORMANCE RANKING:")
        for i, (index_name, result) in enumerate(sorted_results, 1):
            scores = result['average_scores']
            config = index_configs.get(index_name, {})
            print(f"   {i}. {config.get('chunk_size', 'N/A')} chars, {config.get('overlap', 'N/A')} overlap, {config.get('model', 'unknown')} - RAGAS: {scores['ragas_score']:.3f}")
        
        # Key insights
        print("\n🔍 KEY INSIGHTS:")
        
        # Chunk size analysis
        chunk_scores = {}
        for index_name, result in results.items():
            config = index_configs.get(index_name, {})
            chunk_size = config.get('chunk_size')
            if chunk_size:
                if chunk_size not in chunk_scores:
                    chunk_scores[chunk_size] = []
                chunk_scores[chunk_size].append(result['average_scores']['ragas_score'])
        
        if chunk_scores:
            print("   📏 Chunk Size Impact:")
            for chunk_size, scores_list in sorted(chunk_scores.items()):
                avg_score = sum(scores_list) / len(scores_list)
                print(f"      • {chunk_size} characters: Average RAGAS {avg_score:.3f}")
        
        # Model comparison
        model_scores = {}
        for index_name, result in results.items():
            config = index_configs.get(index_name, {})
            model = config.get('model')
            if model:
                if model not in model_scores:
                    model_scores[model] = []
                model_scores[model].append(result['average_scores']['ragas_score'])
        
        if model_scores:
            print("   🤖 Embedding Model Comparison:")
            for model, scores_list in sorted(model_scores.items()):
                avg_score = sum(scores_list) / len(scores_list)
                print(f"      • {model}: Average RAGAS {avg_score:.3f}")
        
        # Metric analysis
        print("   📊 Metric Analysis:")
        best_result = results[best_index]
        best_scores = best_result['average_scores']
        print(f"      • Context Relevancy ({best_scores['context_relevancy']:.3f}): How relevant retrieved documents are")
        print(f"      • Context Recall ({best_scores['context_recall']:.3f}): How much necessary information was retrieved")
        print(f"      • Faithfulness ({best_scores['faithfulness']:.3f}): Accuracy of answer vs retrieved context")
        print(f"      • Answer Relevancy ({best_scores['answer_relevancy']:.3f}): How well answer addresses the question")
        
        print("\n" + "=" * 100)
        print("🚀 RECOMMENDATIONS")
        print("=" * 100)
        print(f"   ✅ Use {best_config.get('chunk_size', 'N/A')}-character chunks with {best_config.get('overlap', 'N/A')}-character overlap")
        print(f"   ✅ Continue using {best_config.get('model', 'unknown')} embedding model for this domain")
        print(f"   ✅ System shows excellent faithfulness ({best_scores['faithfulness']:.3f}) - focus on improving retrieval")
        print(f"   ✅ Consider testing larger chunks if available for even better context recall")
        print("   ✅ Monitor context relevancy for potential retrieval improvements")
    
    def save_results(self, results: Dict[str, Any]):
        """Save detailed results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"deltares_ragas_results_{timestamp}.json"
        
        # Prepare data for JSON serialization
        serializable_results = {}
        for index_name, result in results.items():
            serializable_results[index_name] = {
                'index_name': result['index_name'],
                'average_scores': result['average_scores'],
                'num_questions': result['num_questions'],
                'results': []
            }
            
            # Convert results to serializable format
            for r in result['results']:
                serializable_results[index_name]['results'].append({
                    'question': r['question'],
                    'ground_truth': r['ground_truth'],
                    'response': r['response'],
                    'context': [str(doc) for doc in r['context']],  # Convert documents to strings
                    'evaluation': {
                        'ragas_score': r['evaluation'].ragas_score,
                        'context_relevancy': r['evaluation'].context_relevancy,
                        'context_recall': r['evaluation'].context_recall,
                        'faithfulness': r['evaluation'].faithfulness,
                        'answer_relevancy': r['evaluation'].answer_relevancy
                    }
                })
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Detailed results saved to: {filename}")


def main():
    """Main function to run RAGAS tests."""
    parser = argparse.ArgumentParser(description="Deltares RAGAS Test")
    parser.add_argument("--save-results", action="store_true",
                       help="Save detailed results to JSON file")
    
    args = parser.parse_args()
    
    try:
        # Initialize test
        tester = DeltaresRAGASTest()
        
        # Run tests
        results = tester.run_all_tests()
        
        if not results:
            print("❌ No tests completed successfully")
            return
        
        # Print comparison
        tester.print_comparison(results)
        
        # Save results if requested
        if args.save_results:
            tester.save_results(results)
        
        print("\n🎉 RAGAS testing completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 