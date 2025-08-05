import pytest
import os
import time
from dllmforge.openai_api import OpenAIAPI
from dllmforge.anthropic_api import AnthropicAPI


class TestLLMCapabilities:

    def __init__(self):
        """Initialize the test class and create output directory."""
        self.output_dir = "model_outputs"
        os.makedirs(self.output_dir, exist_ok=True)

    @pytest.mark.skip(reason="Skip test on GitHub Actions, you can run it locally.")
    def test_code_generation(self):
        """Test code generation capabilities with different models."""
        openai_api = OpenAIAPI()
        anthropic_api = AnthropicAPI()

        openai_models = ["gpt-4o", "o1", "o4-mini", "gpt-4", "gpt-4.1-mini"]
        anthropic_models = ["claude-3-7-sonnet-20250219", "claude-3-5-sonnet-20240620"]

        prompt = """Write a Python function that calculates the water balance of a reservoir.
        The function should:
        - Take inputs for inflow, outflow, precipitation, evaporation, and initial storage
        - Calculate the final storage after a given time period
        - Include type hints and docstring
        - Handle edge cases (e.g., negative storage)
        Return only the function code."""

        timing_results = {}

        # Test OpenAI models
        for model in openai_models:
            print(f"\nTesting code generation with OpenAI model: {model}")
            openai_api.model = model

            start_time = time.time()
            response = openai_api.send_test_message(prompt=prompt)
            end_time = time.time()

            timing_results[f"openai_{model}"] = end_time - start_time

            assert response is not None
            assert "response" in response
            code = response["response"]

            # Basic validation of generated code
            assert "def" in code
            assert any(term in code.lower() for term in ["water", "balance", "storage", "reservoir"])
            assert "return" in code
            assert any(param in code.lower() for param in ["inflow", "outflow", "precipitation", "evaporation"])

            # Save the generated code
            output_file = os.path.join(self.output_dir, f"water_balance_openai_{model}.py")
            with open(output_file, "w") as f:
                f.write(code)
            print(f"Code saved to: {output_file}")

            print(f"Model {model} completed in {timing_results[f'openai_{model}']:.2f} seconds")

        # Test Anthropic models
        for model in anthropic_models:
            print(f"\nTesting code generation with Anthropic model: {model}")
            anthropic_api.model = model

            start_time = time.time()
            response = anthropic_api.send_test_message(prompt=prompt)
            end_time = time.time()

            timing_results[f"anthropic_{model}"] = end_time - start_time

            assert response is not None
            assert "response" in response
            code = response["response"]

            # Basic validation of generated code
            assert "def" in code
            assert any(term in code.lower() for term in ["water", "balance", "storage", "reservoir"])
            assert "return" in code
            assert any(param in code.lower() for param in ["inflow", "outflow", "precipitation", "evaporation"])

            # Save the generated code
            output_file = os.path.join(self.output_dir, f"water_balance_anthropic_{model}.py")
            with open(output_file, "w") as f:
                f.write(code)
            print(f"Code saved to: {output_file}")

            print(f"Model {model} completed in {timing_results[f'anthropic_{model}']:.2f} seconds")

        self._print_timing_summary("Code Generation", timing_results)

    @pytest.mark.skip(reason="Skip test on GitHub Actions, you can run it locally.")
    def test_math_reasoning(self):
        """Test mathematical reasoning capabilities."""
        openai_api = OpenAIAPI()
        anthropic_api = AnthropicAPI()

        openai_models = ["gpt-4o", "o1", "o4-mini", "gpt-4", "gpt-4.1-mini"]
        anthropic_models = ["claude-3-7-sonnet-20250219", "claude-3-5-sonnet-20240620"]

        prompt = """Solve this water management problem step by step:
        A water treatment plant processes 2.5 million liters of water per day. 
        The plant has three treatment stages:
        - Stage 1 removes 60% of contaminants
        - Stage 2 removes 75% of remaining contaminants
        - Stage 3 removes 90% of remaining contaminants
        If the incoming water has 1000 mg/L of contaminants, what is the final concentration after all three stages?
        Show your work and explain each step."""

        timing_results = {}

        # Test OpenAI models
        for model in openai_models:
            print(f"\nTesting math reasoning with OpenAI model: {model}")
            openai_api.model = model

            start_time = time.time()
            response = openai_api.send_test_message(prompt=prompt)
            end_time = time.time()

            timing_results[f"openai_{model}"] = end_time - start_time

            assert response is not None
            assert "response" in response
            solution = response["response"]

            # Basic validation of solution
            assert any(str(num) in solution for num in [2.5, 60, 75, 90, 1000])  # Check if numbers are mentioned
            assert any(word in solution.lower() for word in ["stage", "contaminant", "concentration", "treatment"])

            # Save the solution
            output_file = os.path.join(self.output_dir, f"water_treatment_solution_openai_{model}.txt")
            with open(output_file, "w") as f:
                f.write(solution)
            print(f"Solution saved to: {output_file}")

            print(f"Model {model} completed in {timing_results[f'openai_{model}']:.2f} seconds")

        # Test Anthropic models
        for model in anthropic_models:
            print(f"\nTesting math reasoning with Anthropic model: {model}")
            anthropic_api.model = model

            start_time = time.time()
            response = anthropic_api.send_test_message(prompt=prompt)
            end_time = time.time()

            timing_results[f"anthropic_{model}"] = end_time - start_time

            assert response is not None
            assert "response" in response
            solution = response["response"]

            # Basic validation of solution
            assert any(str(num) in solution for num in [2.5, 60, 75, 90, 1000])  # Check if numbers are mentioned
            assert any(word in solution.lower() for word in ["stage", "contaminant", "concentration", "treatment"])

            # Save the solution
            output_file = os.path.join(self.output_dir, f"water_treatment_solution_anthropic_{model}.txt")
            with open(output_file, "w") as f:
                f.write(solution)
            print(f"Solution saved to: {output_file}")

            print(f"Model {model} completed in {timing_results[f'anthropic_{model}']:.2f} seconds")

        self._print_timing_summary("Math Reasoning", timing_results)

    @pytest.mark.skip(reason="Skip test on GitHub Actions, you can run it locally.")
    def test_technical_writing(self):
        """Test technical documentation writing capabilities."""
        openai_api = OpenAIAPI()
        anthropic_api = AnthropicAPI()

        openai_models = ["gpt-4o", "o1", "o4-mini", "gpt-4", "gpt-4.1-mini"]
        anthropic_models = ["claude-3-7-sonnet-20250219", "claude-3-5-sonnet-20240620"]

        prompt = """Write a brief technical documentation for a water level forecasting API endpoint.
        The API should predict water levels for a given location and time period.
        Include:
        - Endpoint description and purpose
        - Required input parameters (location, time period, model type)
        - Response format with example values
        - Error handling scenarios
        - Example usage with curl command
        Keep it concise and professional, focusing on practical implementation details."""

        timing_results = {}

        # Test OpenAI models
        for model in openai_models:
            print(f"\nTesting technical writing with OpenAI model: {model}")
            openai_api.model = model

            start_time = time.time()
            response = openai_api.send_test_message(prompt=prompt)
            end_time = time.time()

            timing_results[f"openai_{model}"] = end_time - start_time

            assert response is not None
            assert "response" in response
            doc = response["response"]

            # Basic validation of documentation
            assert any(word in doc.lower() for word in ["water level", "forecast", "prediction", "api"])
            assert any(word in doc.lower() for word in ["parameter", "response", "error", "example"])
            assert "curl" in doc.lower() or "http" in doc.lower()

            # Save the documentation
            output_file = os.path.join(self.output_dir, f"water_level_api_docs_openai_{model}.md")
            with open(output_file, "w") as f:
                f.write(doc)
            print(f"Documentation saved to: {output_file}")

            print(f"Model {model} completed in {timing_results[f'openai_{model}']:.2f} seconds")

        # Test Anthropic models
        for model in anthropic_models:
            print(f"\nTesting technical writing with Anthropic model: {model}")
            anthropic_api.model = model

            start_time = time.time()
            response = anthropic_api.send_test_message(prompt=prompt)
            end_time = time.time()

            timing_results[f"anthropic_{model}"] = end_time - start_time

            assert response is not None
            assert "response" in response
            doc = response["response"]

            # Basic validation of documentation
            assert any(word in doc.lower() for word in ["water level", "forecast", "prediction", "api"])
            assert any(word in doc.lower() for word in ["parameter", "response", "error", "example"])
            assert "curl" in doc.lower() or "http" in doc.lower()

            # Save the documentation
            output_file = os.path.join(self.output_dir, f"water_level_api_docs_anthropic_{model}.md")
            with open(output_file, "w") as f:
                f.write(doc)
            print(f"Documentation saved to: {output_file}")

            print(f"Model {model} completed in {timing_results[f'anthropic_{model}']:.2f} seconds")

        self._print_timing_summary("Technical Writing", timing_results)

    def _print_timing_summary(self, test_name, timing_results):
        """Helper method to print timing summaries."""
        print(f"\n{test_name} Timing Summary:")
        print("-" * 40)
        for model, time_taken in timing_results.items():
            print(f"{model}: {time_taken:.2f} seconds")
        print("-" * 40)


if __name__ == "__main__":
    # This allows running the tests directly with python
    test = TestLLMCapabilities()
    test.test_code_generation()
    test.test_math_reasoning()
    test.test_technical_writing()
