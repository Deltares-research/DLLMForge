"""
RAGAS Evaluation Module for DLLMForge

This module provides comprehensive evaluation metrics for RAG (Retrieval-Augmented Generation)
pipelines using RAGAS-inspired metrics without requiring external dashboards or services.

The module evaluates four key aspects of RAG systems:
1. Context Relevancy - measures the signal-to-noise ratio in retrieved contexts
2. Context Recall - measures the ability to retrieve all necessary information
3. Faithfulness - measures factual accuracy and absence of hallucinations
4. Answer Relevancy - measures how relevant and to-the-point answers are

All evaluations are performed using LLMs to provide human-like assessment without requiring
annotated datasets.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

from .openai_api import OpenAIAPI
from .anthropic_api import AnthropicAPI
from dllmforge.LLMs.Deltares_LLMs import DeltaresOllamaLLM
from .langchain_api import LangchainAPI

# Load environment variables
load_dotenv()


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    metric_name: str
    score: float
    explanation: str
    details: Dict[str, Any]


@dataclass
class RAGEvaluationResult:
    """Container for complete RAG evaluation results."""
    context_relevancy: EvaluationResult
    context_recall: EvaluationResult
    faithfulness: EvaluationResult
    answer_relevancy: EvaluationResult
    ragas_score: float
    evaluation_time: float
    metadata: Dict[str, Any]


class RAGEvaluator:
    """
    RAGAS-inspired evaluator for RAG pipelines.

    This evaluator provides four key metrics:
    - Context Relevancy: Measures the signal-to-noise ratio in retrieved contexts
    - Context Recall: Measures the ability to retrieve all necessary information
    - Faithfulness: Measures factual accuracy and absence of hallucinations
    - Answer Relevancy: Measures how relevant and to-the-point answers are
    """

    def __init__(self, llm_provider: str = "auto", deltares_llm: Optional[DeltaresOllamaLLM] = None):
        """
        Initialize the RAG evaluator.

        Args:
            llm_provider: LLM provider to use ("openai", "anthropic", "deltares" or "auto")
        """
        self.llm_provider = llm_provider

        # Initialize LLM APIs
        if self.llm_provider == "openai":
            self.openai_api = LangchainAPI(model_provider="openai")
        elif self.llm_provider == "anthropic":
            self.anthropic_api = AnthropicAPI()
        elif self.llm_provider == "azure-openai":
            self.azure_openai_api = LangchainAPI(model_provider="azure-openai")
        elif self.llm_provider == "deltares":
            if deltares_llm is None:
                raise ValueError("Deltares LLM must be provided when using 'deltares' provider")
            self.deltares_llm = deltares_llm
        elif self.llm_provider == "auto":
            # Automatically determine which LLM to use based on available credentials
            self._setup_llm()

    def _setup_llm(self):
        """Setup the LLM provider based on available credentials."""
        if self.llm_provider == "auto":
            # Check for available APIs
            if os.getenv('OPENAI_API_KEY'):
                self.llm_provider = "openai"
            elif os.getenv('ANTHROPIC_API_KEY'):
                self.llm_provider = "anthropic"
            elif os.getenv('AZURE_OPENAI_API_KEY'):
                self.llm_provider = "azure-openai"
            elif self.deltares_llm is not None:
                self.llm_provider = "deltares"
            else:
                raise ValueError("No LLM API credentials found. Please set up OpenAI or Anthropic API keys.")

        print(f"Using LLM provider: {self.llm_provider}")

    def _call_llm(self, messages: List[Dict[str, str]], temperature: float = 0.1) -> str:
        """
        Call the LLM with the specified messages.

        Args:
            messages: List of message dictionaries
            temperature: Temperature for generation
        Returns:
            LLM response text
        """
        try:
            if self.llm_provider == "openai":
                response = self.openai_api.chat_completion(messages=messages, temperature=temperature, max_tokens=1000)
                return response.get("response", "")
            elif self.llm_provider == "anthropic":
                response = self.anthropic_api.chat_completion(messages=messages,
                                                              temperature=temperature,
                                                              max_tokens=1000)
                return response.get("response", "")
            elif self.llm_provider == "azure-openai":
                response = self.azure_openai_api.chat_completion(messages=messages, temperature=temperature, max_tokens=1000)
                return response.get("response", "")
            elif self.llm_provider == "deltares":
                return self.deltares_llm.chat_completion(messages=messages, temperature=temperature, max_tokens=1000)
            else:
                raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return ""

    def evaluate_context_relevancy(self, question: str, retrieved_contexts: List[str]) -> EvaluationResult:
        """
        Evaluate the relevancy of retrieved contexts to the question.

        This metric measures the signal-to-noise ratio in the retrieved contexts.
        It identifies which sentences from the context are actually needed to answer the question.

        Args:
            question: The user's question
            retrieved_contexts: List of retrieved context chunks
        Returns:
            EvaluationResult with score and explanation
        """
        context_text = "\n\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(retrieved_contexts)])

        prompt = f"""/no_think You are evaluating the relevancy of retrieved contexts for a question-answering system.

Question: {question}

Retrieved Contexts:
{context_text}

Your task is to:
1. Identify which sentences from the retrieved contexts are actually needed to answer the question
2. Calculate the ratio: (number of relevant sentences) / (total number of sentences)

Instructions:
- A sentence is relevant if it contains information that directly helps answer the question
- Ignore sentences that are just background information or don't contribute to answering the question
- Count sentences carefully and provide the exact ratio

Please respond in the following JSON format:
{{
    "relevant_sentences": ["sentence 1", "sentence 2", ...],
    "total_sentences": number,
    "relevant_count": number,
    "ratio": float,
    "explanation": "Brief explanation of your reasoning"
}}"""

        messages = [{
            "role": "system",
            "content": "You are a helpful assistant that evaluates the relevancy of text contexts."
        }, {
            "role": "user",
            "content": prompt
        }]

        response = self._call_llm(messages)

        try:
            # Try to parse JSON response
            # check if this is arleady type dict
            if isinstance(response, dict):
                response = response['choices'][0]['message']['content']
                # now remove the empty think
                response = response.replace("<think>\n\n</think>\n\n", "")
            result = json.loads(response)
            score = result.get("ratio", 0.0)
            explanation = result.get("explanation", "No explanation provided")
            details = {
                "relevant_sentences": result.get("relevant_sentences", []),
                "total_sentences": result.get("total_sentences", 0),
                "relevant_count": result.get("relevant_count", 0)
            }
        except json.JSONDecodeError:
            # Try to extract JSON from the response using regex
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    score = result.get("ratio", 0.0)
                    explanation = result.get("explanation", "Extracted from response")
                    details = {
                        "relevant_sentences": result.get("relevant_sentences", []),
                        "total_sentences": result.get("total_sentences", 0),
                        "relevant_count": result.get("relevant_count", 0)
                    }
                # not bare except:
                except json.JSONDecodeError:
                    # Fallback: try to extract score from text
                    score = 0.5  # Default score
                    explanation = "Could not parse LLM response (JSON extraction failed)"
                    details = {"raw_response": response}
            else:
                # Fallback: try to extract score from text
                score = 0.5  # Default score
                explanation = "Could not parse LLM response (no JSON found)"
                details = {"raw_response": response}

        return EvaluationResult(metric_name="context_relevancy", score=score, explanation=explanation, details=details)

    def evaluate_context_recall(self, question: str, retrieved_contexts: List[str],
                                ground_truth_answer: str) -> EvaluationResult:
        """
        Evaluate the recall of retrieved contexts against a ground truth answer.
        This metric measures the ability of the retriever to retrieve all necessary information
        needed to answer the question by checking if each statement from the ground truth
        can be found in the retrieved context.

        Args:
            question: The user's question
            retrieved_contexts: List of retrieved context chunks
            ground_truth_answer: The reference answer to compare against
        Returns:
            EvaluationResult with score and explanation
        """
        context_text = "\n\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(retrieved_contexts)])

        prompt = f""" /no_think You are evaluating the recall of retrieved contexts for a question-answering system.

Question: {question}

Ground Truth Answer: {ground_truth_answer}

Retrieved Contexts:
{context_text}

Your task is to:
1. Break down the ground truth answer into individual factual statements
2. Check if each statement can be supported by information in the retrieved contexts
3. Calculate the ratio: (number of supported statements) / (total number of statements)

Instructions:
- A statement is supported if the same information appears in the retrieved contexts
- Consider paraphrasing and different ways of expressing the same fact
- Be strict about factual accuracy - the context must contain the actual information

Please respond in the following JSON format:
{{
    "statements": ["statement 1", "statement 2", ...],
    "supported_statements": ["statement 1", "statement 3", ...],
    "total_statements": number,
    "supported_count": number,
    "ratio": float,
    "explanation": "Brief explanation of your reasoning"
}}"""

        messages = [{
            "role": "system",
            "content": "You are a helpful assistant that evaluates the recall of text contexts."
        }, {
            "role": "user",
            "content": prompt
        }]

        response = self._call_llm(messages)

        try:
            if isinstance(response, dict):
                response = response['choices'][0]['message']['content']
                # now remove the empty think
                response = response.replace("<think>\n\n</think>\n\n", "")
            result = json.loads(response)
            score = result.get("ratio", 0.0)
            explanation = result.get("explanation", "No explanation provided")
            details = {
                "statements": result.get("statements", []),
                "supported_statements": result.get("supported_statements", []),
                "total_statements": result.get("total_statements", 0),
                "supported_count": result.get("supported_count", 0)
            }
        except json.JSONDecodeError:
            # Try to extract JSON from the response using regex
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    score = result.get("ratio", 0.0)
                    explanation = result.get("explanation", "Extracted from response")
                    details = {
                        "statements": result.get("statements", []),
                        "supported_statements": result.get("supported_statements", []),
                        "total_statements": result.get("total_statements", 0),
                        "supported_count": result.get("supported_count", 0)
                    }
                except json.JSONDecodeError:
                    score = 0.5
                    explanation = "Could not parse LLM response (JSON extraction failed)"
                    details = {"raw_response": response}
            else:
                score = 0.5
                explanation = "Could not parse LLM response (no JSON found)"
                details = {"raw_response": response}

        return EvaluationResult(metric_name="context_recall", score=score, explanation=explanation, details=details)

    def evaluate_faithfulness(self, question: str, generated_answer: str,
                              retrieved_contexts: List[str]) -> EvaluationResult:
        """
        Evaluate the faithfulness of the generated answer to the retrieved contexts.

        This metric measures the factual accuracy of the generated answer by checking
        if all statements in the answer are supported by the retrieved contexts.

        Args:
            question: The user's question
            generated_answer: The answer generated by the RAG system
            retrieved_contexts: List of retrieved context chunks
        Returns:
            EvaluationResult with score and explanation
        """
        context_text = "\n\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(retrieved_contexts)])

        prompt = f"""/no_think You are evaluating the faithfulness of a generated answer to the provided contexts.

Question: {question}

Generated Answer: {generated_answer}

Retrieved Contexts:
{context_text}

Your task is to:
1. Identify all factual statements made in the generated answer
2. Check if each statement is supported by the retrieved contexts
3. Calculate the ratio: (number of supported statements) / (total number of statements)

Instructions:
- A statement is supported if the same information appears in the retrieved contexts
- Consider paraphrasing and different ways of expressing the same fact
- Be strict about factual accuracy - the context must contain the actual information
- Ignore statements that are just common knowledge or reasonable inferences

Please respond in the following JSON format:
{{
    "statements": ["statement 1", "statement 2", ...],
    "supported_statements": ["statement 1", "statement 3", ...],
    "unsupported_statements": ["statement 2", ...],
    "total_statements": number,
    "supported_count": number,
    "ratio": float,
    "explanation": "Brief explanation of your reasoning"
}}"""

        messages = [{
            "role": "system",
            "content": "You are a helpful assistant that evaluates the faithfulness of generated answers."
        }, {
            "role": "user",
            "content": prompt
        }]

        response = self._call_llm(messages)

        try:
            if isinstance(response, dict):
                response = response['choices'][0]['message']['content']
                # now remove the empty think
                response = response.replace("<think>\n\n</think>\n\n", "")
            result = json.loads(response)
            score = result.get("ratio", 0.0)
            explanation = result.get("explanation", "No explanation provided")
            details = {
                "statements": result.get("statements", []),
                "supported_statements": result.get("supported_statements", []),
                "unsupported_statements": result.get("unsupported_statements", []),
                "total_statements": result.get("total_statements", 0),
                "supported_count": result.get("supported_count", 0)
            }
        except json.JSONDecodeError:
            # Try to extract JSON from the response using regex
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    score = result.get("ratio", 0.0)
                    explanation = result.get("explanation", "Extracted from response")
                    details = {
                        "statements": result.get("statements", []),
                        "supported_statements": result.get("supported_statements", []),
                        "unsupported_statements": result.get("unsupported_statements", []),
                        "total_statements": result.get("total_statements", 0),
                        "supported_count": result.get("supported_count", 0)
                    }
                except json.JSONDecodeError:
                    score = 0.5
                    explanation = "Could not parse LLM response (JSON extraction failed)"
                    details = {"raw_response": response}
            else:
                score = 0.5
                explanation = "Could not parse LLM response (no JSON found)"
                details = {"raw_response": response}

        return EvaluationResult(metric_name="faithfulness", score=score, explanation=explanation, details=details)

    def evaluate_answer_relevancy(self, question: str, generated_answer: str) -> EvaluationResult:
        """
        Evaluate the relevancy of the generated answer to the question.
        This metric measures how relevant and to-the-point the answer is by generating
        probable questions that the answer could answer and computing similarity to the actual question.
        Args:
            question: The user's question
            generated_answer: The answer generated by the RAG system
        Returns:
            EvaluationResult with score and explanation
        """
        prompt = f"""/no_think You are evaluating the relevancy of a generated answer to a question.

Original Question: {question}

Generated Answer: {generated_answer}

Your task is to:
1. Generate 3-5 probable questions that this answer could reasonably answer
2. Rate how well the generated answer addresses the original question on a scale of 0.0 to 1.0
3. Consider factors like:
   - Does the answer directly address the question?
   - Is the answer complete and comprehensive?
   - Is the answer focused and not overly verbose?
   - Does the answer provide the information the question is asking for?

Please respond in the following JSON format:
{{
    "probable_questions": ["question 1", "question 2", "question 3"],
    "relevancy_score": float,
    "explanation": "Brief explanation of your reasoning",
    "strengths": ["strength 1", "strength 2"],
    "weaknesses": ["weakness 1", "weakness 2"]
}}"""

        messages = [{
            "role": "system",
            "content": "You are a helpful assistant that evaluates the relevancy of answers."
        }, {
            "role": "user",
            "content": prompt
        }]

        response = self._call_llm(messages)

        try:
            if isinstance(response, dict):
                response = response['choices'][0]['message']['content']
                # now remove the empty think
                response = response.replace("<think>\n\n</think>\n\n", "")
            result = json.loads(response)
            score = result.get("relevancy_score", 0.5)
            explanation = result.get("explanation", "No explanation provided")
            details = {
                "probable_questions": result.get("probable_questions", []),
                "strengths": result.get("strengths", []),
                "weaknesses": result.get("weaknesses", [])
            }
        except json.JSONDecodeError:
            # Try to extract JSON from the response using regex
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    score = result.get("relevancy_score", 0.5)
                    explanation = result.get("explanation", "Extracted from response")
                    details = {
                        "probable_questions": result.get("probable_questions", []),
                        "strengths": result.get("strengths", []),
                        "weaknesses": result.get("weaknesses", [])
                    }
                except json.JSONDecodeError:
                    score = 0.5
                    explanation = "Could not parse LLM response (JSON extraction failed)"
                    details = {"raw_response": response}
            else:
                score = 0.5
                explanation = "Could not parse LLM response (no JSON found)"
                details = {"raw_response": response}

        return EvaluationResult(metric_name="answer_relevancy", score=score, explanation=explanation, details=details)

    def calculate_ragas_score(self, context_relevancy: float, context_recall: float, faithfulness: float,
                              answer_relevancy: float) -> float:
        """
        Calculate the RAGAS score as the harmonic mean of all four metrics.
        Args:
            context_relevancy: Context relevancy score
            context_recall: Context recall score
            faithfulness: Faithfulness score
            answer_relevancy: Answer relevancy score
        Returns:
            RAGAS score (harmonic mean)
        """
        scores = [context_relevancy, context_recall, faithfulness, answer_relevancy]
        # Filter out zero scores to avoid division by zero
        non_zero_scores = [score for score in scores if score > 0]

        if not non_zero_scores:
            return 0.0

        # Calculate harmonic mean
        harmonic_mean = len(non_zero_scores) / sum(1 / score for score in non_zero_scores)
        return harmonic_mean

    def evaluate_rag_pipeline(self,
                              question: str,
                              generated_answer: str,
                              retrieved_contexts: List[str],
                              ground_truth_answer: Optional[str] = None) -> RAGEvaluationResult:
        """
        Evaluate a complete RAG pipeline using all four metrics.

        Args:
            question: The user's question
            generated_answer: The answer generated by the RAG system
            retrieved_contexts: List of retrieved context chunks
            ground_truth_answer: Optional ground truth answer for context recall evaluation
        Returns:
            Complete evaluation results
        """
        start_time = time.time()

        print("🔍 Starting RAG evaluation...")

        # Evaluate context relevancy
        print("  📊 Evaluating context relevancy...")
        context_relevancy = self.evaluate_context_relevancy(question, retrieved_contexts)

        # Evaluate faithfulness
        print("  📊 Evaluating faithfulness...")
        faithfulness = self.evaluate_faithfulness(question, generated_answer, retrieved_contexts)

        # Evaluate answer relevancy
        print("  📊 Evaluating answer relevancy...")
        answer_relevancy = self.evaluate_answer_relevancy(question, generated_answer)

        # Evaluate context recall (if ground truth is provided)
        if ground_truth_answer:
            print("  📊 Evaluating context recall...")
            context_recall = self.evaluate_context_recall(question, retrieved_contexts, ground_truth_answer)
        else:
            # Use context relevancy as a proxy for context recall
            context_recall = EvaluationResult(metric_name="context_recall",
                                              score=context_relevancy.score,
                                              explanation="Using context relevancy as proxy (no ground truth provided)",
                                              details={"note": "Ground truth answer not provided"})

        # Calculate RAGAS score
        ragas_score = self.calculate_ragas_score(context_relevancy.score, context_recall.score, faithfulness.score,
                                                 answer_relevancy.score)

        evaluation_time = time.time() - start_time

        # Compile metadata
        metadata = {
            "llm_provider": self.llm_provider,
            "question": question,
            "generated_answer": generated_answer,
            "context_count": len(retrieved_contexts),
            "has_ground_truth": ground_truth_answer is not None
        }

        return RAGEvaluationResult(context_relevancy=context_relevancy,
                                   context_recall=context_recall,
                                   faithfulness=faithfulness,
                                   answer_relevancy=answer_relevancy,
                                   ragas_score=ragas_score,
                                   evaluation_time=evaluation_time,
                                   metadata=metadata)

    def print_evaluation_summary(self, result: RAGEvaluationResult):
        """
        Print a formatted summary of the evaluation results.
        Args:
            result: The evaluation results to summarize
        """
        print("\n" + "=" * 60)
        print("📊 RAG EVALUATION SUMMARY")
        print("=" * 60)

        print(f"🎯 Overall RAGAS Score: {result.ragas_score:.3f}")
        print(f"⏱️  Evaluation Time: {result.evaluation_time:.2f} seconds")
        print()

        print("📈 Individual Metrics:")
        print(f"  • Context Relevancy: {result.context_relevancy.score:.3f}")
        print(f"    {result.context_relevancy.explanation}")

        print(f"  • Context Recall: {result.context_recall.score:.3f}")
        print(f"    {result.context_recall.explanation}")

        print(f"  • Faithfulness: {result.faithfulness.score:.3f}")
        print(f"    {result.faithfulness.explanation}")

        print(f"  • Answer Relevancy: {result.answer_relevancy.score:.3f}")
        print(f"    {result.answer_relevancy.explanation}")

        print("\n" + "=" * 60)

    def save_evaluation_results(self, result: RAGEvaluationResult, output_file: str):
        """
        Save evaluation results to a JSON file.
        Args:
            result: The evaluation results to save
            output_file: Path to the output JSON file
        """
        # Convert dataclass to dictionary
        result_dict = {
            "ragas_score": result.ragas_score,
            "evaluation_time": result.evaluation_time,
            "metadata": result.metadata,
            "metrics": {
                "context_relevancy": {
                    "score": result.context_relevancy.score,
                    "explanation": result.context_relevancy.explanation,
                    "details": result.context_relevancy.details
                },
                "context_recall": {
                    "score": result.context_recall.score,
                    "explanation": result.context_recall.explanation,
                    "details": result.context_recall.details
                },
                "faithfulness": {
                    "score": result.faithfulness.score,
                    "explanation": result.faithfulness.explanation,
                    "details": result.faithfulness.details
                },
                "answer_relevancy": {
                    "score": result.answer_relevancy.score,
                    "explanation": result.answer_relevancy.explanation,
                    "details": result.answer_relevancy.details
                }
            }
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)

        print(f"💾 Evaluation results saved to: {output_file}")


def evaluate_rag_response(question: str,
                          generated_answer: str,
                          retrieved_contexts: List[str],
                          ground_truth_answer: Optional[str] = None,
                          llm_provider: str = "auto",
                          save_results: bool = True,
                          output_file: Optional[str] = None) -> RAGEvaluationResult:
    """
    Convenience function to evaluate a RAG response.
    Args:
        question: The user's question
        generated_answer: The answer generated by the RAG system
        retrieved_contexts: List of retrieved context chunks
        ground_truth_answer: Optional ground truth answer for context recall evaluation
        llm_provider: LLM provider to use ("openai", "anthropic", or "auto")
        save_results: Whether to save results to a file
        output_file: Optional output file path
    Returns:
        Complete evaluation results
    """
    evaluator = RAGEvaluator(llm_provider=llm_provider)

    result = evaluator.evaluate_rag_pipeline(question=question,
                                             generated_answer=generated_answer,
                                             retrieved_contexts=retrieved_contexts,
                                             ground_truth_answer=ground_truth_answer)

    evaluator.print_evaluation_summary(result)

    if save_results:
        if output_file is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"rag_evaluation_{timestamp}.json"

        evaluator.save_evaluation_results(result, output_file)

    return result
