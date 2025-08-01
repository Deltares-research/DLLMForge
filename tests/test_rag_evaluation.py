#!/usr/bin/env python3
"""
Tests for the RAGAS evaluation module.

This module tests the RAG evaluation functionality including:
- Individual metric evaluation
- Complete pipeline evaluation
- Error handling
- Result formatting
"""

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch
from typing import List, Dict, Any

from dllmforge.rag_evaluation import (
    RAGEvaluator, 
    evaluate_rag_response, 
    RAGEvaluationResult,
    EvaluationResult
)


class TestRAGEvaluator:
    """Test cases for the RAGEvaluator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock the LLM APIs to avoid actual API calls during testing
        self.mock_openai_api = Mock()
        self.mock_anthropic_api = Mock()
        
        # Sample responses for different metrics
        self.sample_context_relevancy_response = {
            "relevant_sentences": ["Machine learning is a subset of AI"],
            "total_sentences": 3,
            "relevant_count": 1,
            "ratio": 0.333,
            "explanation": "Only one sentence is relevant to the question"
        }
        
        self.sample_faithfulness_response = {
            "statements": ["Tokyo has 14 million people"],
            "supported_statements": ["Tokyo has 14 million people"],
            "unsupported_statements": [],
            "total_statements": 1,
            "supported_count": 1,
            "ratio": 1.0,
            "explanation": "All statements are supported by the context"
        }
        
        self.sample_answer_relevancy_response = {
            "probable_questions": ["What is the capital of France?"],
            "relevancy_score": 0.9,
            "explanation": "The answer directly addresses the question",
            "strengths": ["Direct answer"],
            "weaknesses": []
        }
    
    @patch('dllmforge.rag_evaluation.OpenAIAPI')
    @patch('dllmforge.rag_evaluation.AnthropicAPI')
    def test_initialization(self, mock_anthropic, mock_openai):
        """Test evaluator initialization."""
        # Mock environment variables
        with patch.dict(os.environ, {'AZURE_OPENAI_API_KEY': 'test_key'}):
            evaluator = RAGEvaluator(llm_provider="openai")
            assert evaluator.llm_provider == "openai"
    
    @patch('dllmforge.rag_evaluation.OpenAIAPI')
    @patch('dllmforge.rag_evaluation.AnthropicAPI')
    def test_auto_provider_selection(self, mock_anthropic, mock_openai):
        """Test automatic provider selection."""
        # Test OpenAI selection
        with patch.dict(os.environ, {'AZURE_OPENAI_API_KEY': 'test_key'}):
            evaluator = RAGEvaluator(llm_provider="auto")
            assert evaluator.llm_provider == "openai"
        
        # Test Anthropic selection
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_key'}, clear=True):
            evaluator = RAGEvaluator(llm_provider="auto")
            assert evaluator.llm_provider == "anthropic"
    
    @patch('dllmforge.rag_evaluation.OpenAIAPI')
    @patch('dllmforge.rag_evaluation.AnthropicAPI')
    def test_context_relevancy_evaluation(self, mock_anthropic, mock_openai):
        """Test context relevancy evaluation."""
        # Setup mock
        mock_openai_instance = Mock()
        mock_openai_instance.chat_completion.return_value = {
            "response": json.dumps(self.sample_context_relevancy_response)
        }
        mock_openai.return_value = mock_openai_instance
        
        with patch.dict(os.environ, {'AZURE_OPENAI_API_KEY': 'test_key'}):
            evaluator = RAGEvaluator(llm_provider="openai")
            
            question = "What is machine learning?"
            contexts = [
                "Machine learning is a subset of artificial intelligence.",
                "The weather today is sunny.",
                "Cooking requires patience and skill."
            ]
            
            result = evaluator.evaluate_context_relevancy(question, contexts)
            
            assert isinstance(result, EvaluationResult)
            assert result.metric_name == "context_relevancy"
            assert result.score == 0.333
            assert "Only one sentence is relevant" in result.explanation
    
    @patch('dllmforge.rag_evaluation.OpenAIAPI')
    @patch('dllmforge.rag_evaluation.AnthropicAPI')
    def test_faithfulness_evaluation(self, mock_anthropic, mock_openai):
        """Test faithfulness evaluation."""
        # Setup mock
        mock_openai_instance = Mock()
        mock_openai_instance.chat_completion.return_value = {
            "response": json.dumps(self.sample_faithfulness_response)
        }
        mock_openai.return_value = mock_openai_instance
        
        with patch.dict(os.environ, {'AZURE_OPENAI_API_KEY': 'test_key'}):
            evaluator = RAGEvaluator(llm_provider="openai")
            
            question = "What is the population of Tokyo?"
            answer = "Tokyo has 14 million people."
            contexts = ["Tokyo's population is 14 million according to census data."]
            
            result = evaluator.evaluate_faithfulness(question, answer, contexts)
            
            assert isinstance(result, EvaluationResult)
            assert result.metric_name == "faithfulness"
            assert result.score == 1.0
            assert "All statements are supported" in result.explanation
    
    @patch('dllmforge.rag_evaluation.OpenAIAPI')
    @patch('dllmforge.rag_evaluation.AnthropicAPI')
    def test_answer_relevancy_evaluation(self, mock_anthropic, mock_openai):
        """Test answer relevancy evaluation."""
        # Setup mock
        mock_openai_instance = Mock()
        mock_openai_instance.chat_completion.return_value = {
            "response": json.dumps(self.sample_answer_relevancy_response)
        }
        mock_openai.return_value = mock_openai_instance
        
        with patch.dict(os.environ, {'AZURE_OPENAI_API_KEY': 'test_key'}):
            evaluator = RAGEvaluator(llm_provider="openai")
            
            question = "What is the capital of France?"
            answer = "The capital of France is Paris."
            
            result = evaluator.evaluate_answer_relevancy(question, answer)
            
            assert isinstance(result, EvaluationResult)
            assert result.metric_name == "answer_relevancy"
            assert result.score == 0.9
            assert "directly addresses the question" in result.explanation
    
    @patch('dllmforge.rag_evaluation.OpenAIAPI')
    @patch('dllmforge.rag_evaluation.AnthropicAPI')
    def test_context_recall_evaluation(self, mock_anthropic, mock_openai):
        """Test context recall evaluation."""
        # Setup mock
        mock_openai_instance = Mock()
        mock_openai_instance.chat_completion.return_value = {
            "response": json.dumps({
                "statements": ["Paris is the capital of France"],
                "supported_statements": ["Paris is the capital of France"],
                "total_statements": 1,
                "supported_count": 1,
                "ratio": 1.0,
                "explanation": "All statements are supported"
            })
        }
        mock_openai.return_value = mock_openai_instance
        
        with patch.dict(os.environ, {'AZURE_OPENAI_API_KEY': 'test_key'}):
            evaluator = RAGEvaluator(llm_provider="openai")
            
            question = "What is the capital of France?"
            contexts = ["Paris is the capital and largest city of France."]
            ground_truth = "Paris is the capital of France."
            
            result = evaluator.evaluate_context_recall(question, contexts, ground_truth)
            
            assert isinstance(result, EvaluationResult)
            assert result.metric_name == "context_recall"
            assert result.score == 1.0
    
    @patch('dllmforge.rag_evaluation.OpenAIAPI')
    @patch('dllmforge.rag_evaluation.AnthropicAPI')
    def test_complete_pipeline_evaluation(self, mock_anthropic, mock_openai):
        """Test complete pipeline evaluation."""
        # Setup mock responses
        mock_openai_instance = Mock()
        mock_openai_instance.chat_completion.side_effect = [
            {"response": json.dumps(self.sample_context_relevancy_response)},
            {"response": json.dumps(self.sample_faithfulness_response)},
            {"response": json.dumps(self.sample_answer_relevancy_response)},
            {"response": json.dumps({
                "statements": ["Paris is the capital"],
                "supported_statements": ["Paris is the capital"],
                "total_statements": 1,
                "supported_count": 1,
                "ratio": 1.0,
                "explanation": "All statements supported"
            })}
        ]
        mock_openai.return_value = mock_openai_instance
        
        with patch.dict(os.environ, {'AZURE_OPENAI_API_KEY': 'test_key'}):
            evaluator = RAGEvaluator(llm_provider="openai")
            
            result = evaluator.evaluate_rag_pipeline(
                question="What is the capital of France?",
                generated_answer="The capital of France is Paris.",
                retrieved_contexts=["Paris is the capital of France."],
                ground_truth_answer="Paris is the capital of France."
            )
            
            assert isinstance(result, RAGEvaluationResult)
            assert result.ragas_score > 0
            assert result.evaluation_time > 0
            assert result.metadata["question"] == "What is the capital of France?"
            assert result.metadata["has_ground_truth"] is True
    
    def test_ragas_score_calculation(self):
        """Test RAGAS score calculation."""
        evaluator = RAGEvaluator(llm_provider="openai")
        
        # Test with all scores equal
        score = evaluator.calculate_ragas_score(0.8, 0.8, 0.8, 0.8)
        assert score == 0.8
        
        # Test with different scores
        score = evaluator.calculate_ragas_score(1.0, 0.5, 0.5, 1.0)
        assert score == 0.667  # Harmonic mean of [1.0, 0.5, 0.5, 1.0]
        
        # Test with zero scores
        score = evaluator.calculate_ragas_score(0.0, 0.5, 0.5, 0.5)
        assert score == 0.5  # Should handle zero scores gracefully
    
    @patch('dllmforge.rag_evaluation.OpenAIAPI')
    @patch('dllmforge.rag_evaluation.AnthropicAPI')
    def test_save_evaluation_results(self, mock_anthropic, mock_openai):
        """Test saving evaluation results to JSON."""
        # Create a mock evaluation result
        context_relevancy = EvaluationResult(
            metric_name="context_relevancy",
            score=0.8,
            explanation="Good relevancy",
            details={"test": "data"}
        )
        
        context_recall = EvaluationResult(
            metric_name="context_recall",
            score=0.9,
            explanation="Good recall",
            details={"test": "data"}
        )
        
        faithfulness = EvaluationResult(
            metric_name="faithfulness",
            score=0.85,
            explanation="Good faithfulness",
            details={"test": "data"}
        )
        
        answer_relevancy = EvaluationResult(
            metric_name="answer_relevancy",
            score=0.75,
            explanation="Good relevancy",
            details={"test": "data"}
        )
        
        result = RAGEvaluationResult(
            context_relevancy=context_relevancy,
            context_recall=context_recall,
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            ragas_score=0.825,
            evaluation_time=5.0,
            metadata={"test": "metadata"}
        )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            evaluator = RAGEvaluator(llm_provider="openai")
            evaluator.save_evaluation_results(result, temp_file)
            
            # Verify file was created and contains expected data
            with open(temp_file, 'r') as f:
                saved_data = json.load(f)
            
            assert saved_data["ragas_score"] == 0.825
            assert saved_data["evaluation_time"] == 5.0
            assert "metrics" in saved_data
            assert "context_relevancy" in saved_data["metrics"]
            assert saved_data["metrics"]["context_relevancy"]["score"] == 0.8
            
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    @patch('dllmforge.rag_evaluation.OpenAIAPI')
    @patch('dllmforge.rag_evaluation.AnthropicAPI')
    def test_error_handling(self, mock_anthropic, mock_openai):
        """Test error handling in evaluation."""
        # Setup mock to raise exception
        mock_openai_instance = Mock()
        mock_openai_instance.chat_completion.side_effect = Exception("API Error")
        mock_openai.return_value = mock_openai_instance
        
        with patch.dict(os.environ, {'AZURE_OPENAI_API_KEY': 'test_key'}):
            evaluator = RAGEvaluator(llm_provider="openai")
            
            # Should handle API errors gracefully
            result = evaluator.evaluate_context_relevancy("test", ["test"])
            assert result.score == 0.5  # Default fallback score
            assert "Could not parse LLM response" in result.explanation


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    @patch('dllmforge.rag_evaluation.RAGEvaluator')
    def test_evaluate_rag_response(self, mock_evaluator_class):
        """Test the evaluate_rag_response convenience function."""
        # Setup mock
        mock_evaluator = Mock()
        mock_evaluator_class.return_value = mock_evaluator
        
        mock_result = Mock()
        mock_result.ragas_score = 0.85
        mock_evaluator.evaluate_rag_pipeline.return_value = mock_result
        
        # Test the convenience function
        result = evaluate_rag_response(
            question="What is the capital of France?",
            generated_answer="Paris is the capital of France.",
            retrieved_contexts=["Paris is the capital of France."],
            ground_truth_answer="Paris is the capital of France.",
            llm_provider="openai",
            save_results=False
        )
        
        assert result.ragas_score == 0.85
        mock_evaluator.evaluate_rag_pipeline.assert_called_once()
        mock_evaluator.print_evaluation_summary.assert_called_once()


class TestIntegration:
    """Integration tests with actual DLLMForge components."""
    
    def test_import_compatibility(self):
        """Test that the module can be imported with existing DLLMForge components."""
        try:
            from dllmforge.rag_evaluation import RAGEvaluator
            from dllmforge.rag_preprocess_documents import PDFLoader
            from dllmforge.rag_embedding import AzureOpenAIEmbeddingModel
            from dllmforge.rag_search_and_response import IndexManager, Retriever, LLMResponder
            
            # If we get here, imports are successful
            assert True
        except ImportError as e:
            pytest.fail(f"Import failed: {e}")
    
    def test_dataclass_serialization(self):
        """Test that dataclasses can be properly serialized."""
        from dllmforge.rag_evaluation import EvaluationResult, RAGEvaluationResult
        
        # Test EvaluationResult serialization
        eval_result = EvaluationResult(
            metric_name="test",
            score=0.8,
            explanation="test explanation",
            details={"key": "value"}
        )
        
        # Should be able to access all attributes
        assert eval_result.metric_name == "test"
        assert eval_result.score == 0.8
        assert eval_result.explanation == "test explanation"
        assert eval_result.details["key"] == "value"
        
        # Test RAGEvaluationResult creation
        context_relevancy = EvaluationResult("context_relevancy", 0.8, "test", {})
        context_recall = EvaluationResult("context_recall", 0.9, "test", {})
        faithfulness = EvaluationResult("faithfulness", 0.85, "test", {})
        answer_relevancy = EvaluationResult("answer_relevancy", 0.75, "test", {})
        
        rag_result = RAGEvaluationResult(
            context_relevancy=context_relevancy,
            context_recall=context_recall,
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            ragas_score=0.825,
            evaluation_time=5.0,
            metadata={"test": "metadata"}
        )
        
        assert rag_result.ragas_score == 0.825
        assert rag_result.context_relevancy.score == 0.8
        assert rag_result.context_recall.score == 0.9


if __name__ == "__main__":
    pytest.main([__file__]) 