import pytest
import os
import time
from dllmforge.langchain_api import LangchainAPI

class TestLangchainCapabilities:
    def __init__(self):
        """Initialize the test class and create output directory."""
        self.output_dir = "model_outputs"
        os.makedirs(self.output_dir, exist_ok=True)

    @pytest.mark.skip(reason="Skip test on GitHub Actions, you can run it locally.")
    def test_code_generation(self):
        """Test code generation capabilities with different Langchain models."""
        # Initialize different model providers
        azure_api = LangchainAPI(model_provider="azure-openai")
        openai_api = LangchainAPI(model_provider="openai")
        mistral_api = LangchainAPI(model_provider="mistral")
        
        prompt = """Write a Python function that calculates the water balance of a reservoir.
        The function should:
        - Take inputs for inflow, outflow, precipitation, evaporation, and initial storage
        - Calculate the final storage after a given time period
        - Include type hints and docstring
        - Handle edge cases (e.g., negative storage)
        Return only the function code."""
        
        timing_results = {}
        
        # Test Azure OpenAI
        print("\nTesting code generation with Azure OpenAI")
        start_time = time.time()
        response = azure_api.send_test_message(prompt=prompt)
        end_time = time.time()
        
        timing_results["azure_openai"] = end_time - start_time
        
        assert response is not None
        assert "response" in response
        code = response["response"]
        
        # Basic validation of generated code
        assert "def" in code
        assert any(term in code.lower() for term in ["water", "balance", "storage", "reservoir"])
        assert "return" in code
        assert any(param in code.lower() for param in ["inflow", "outflow", "precipitation", "evaporation"])
        
        # Save the generated code
        output_file = os.path.join(self.output_dir, "water_balance_azure_openai.py")
        with open(output_file, "w") as f:
            f.write(code)
        print(f"Code saved to: {output_file}")
        
        # Test OpenAI
        print("\nTesting code generation with OpenAI")
        start_time = time.time()
        response = openai_api.send_test_message(prompt=prompt)
        end_time = time.time()
        
        timing_results["openai"] = end_time - start_time
        
        assert response is not None
        assert "response" in response
        code = response["response"]
        
        # Basic validation and save
        assert "def" in code
        output_file = os.path.join(self.output_dir, "water_balance_openai.py")
        with open(output_file, "w") as f:
            f.write(code)
        print(f"Code saved to: {output_file}")
        
        # Test Mistral
        print("\nTesting code generation with Mistral")
        start_time = time.time()
        response = mistral_api.send_test_message(prompt=prompt)
        end_time = time.time()
        
        timing_results["mistral"] = end_time - start_time
        
        assert response is not None
        assert "response" in response
        code = response["response"]
        
        # Basic validation and save
        assert "def" in code
        output_file = os.path.join(self.output_dir, "water_balance_mistral.py")
        with open(output_file, "w") as f:
            f.write(code)
        print(f"Code saved to: {output_file}")
        
        self._print_timing_summary("Code Generation", timing_results)

    @pytest.mark.skip(reason="Skip test on GitHub Actions, you can run it locally.")
    def test_math_reasoning(self):
        """Test mathematical reasoning capabilities with different Langchain models."""
        azure_api = LangchainAPI(model_provider="azure-openai")
        openai_api = LangchainAPI(model_provider="openai")
        mistral_api = LangchainAPI(model_provider="mistral")
        
        prompt = """Solve this water management problem step by step:
        A water treatment plant processes 2.5 million liters of water per day. 
        The plant has three treatment stages:
        - Stage 1 removes 60% of contaminants
        - Stage 2 removes 75% of remaining contaminants
        - Stage 3 removes 90% of remaining contaminants
        If the incoming water has 1000 mg/L of contaminants, what is the final concentration after all three stages?
        Show your work and explain each step."""
        
        timing_results = {}
        
        # Test each model provider
        for api, provider in [(azure_api, "azure_openai"), (openai_api, "openai"), (mistral_api, "mistral")]:
            print(f"\nTesting math reasoning with {provider}")
            start_time = time.time()
            response = api.send_test_message(prompt=prompt)
            end_time = time.time()
            
            timing_results[provider] = end_time - start_time
            
            assert response is not None
            assert "response" in response
            solution = response["response"]
            
            # Basic validation of solution
            assert any(str(num) in solution for num in [2.5, 60, 75, 90, 1000])
            assert any(word in solution.lower() for word in ["stage", "contaminant", "concentration", "treatment"])
            
            # Save the solution
            output_file = os.path.join(self.output_dir, f"water_treatment_solution_{provider}.txt")
            with open(output_file, "w") as f:
                f.write(solution)
            print(f"Solution saved to: {output_file}")
        
        self._print_timing_summary("Math Reasoning", timing_results)

    @pytest.mark.skip(reason="Skip test on GitHub Actions, you can run it locally.")
    def test_technical_writing(self):
        """Test technical documentation writing capabilities with different Langchain models."""
        azure_api = LangchainAPI(model_provider="azure-openai")
        openai_api = LangchainAPI(model_provider="openai")
        mistral_api = LangchainAPI(model_provider="mistral")
        
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
        
        # Test each model provider
        for api, provider in [(azure_api, "azure_openai"), (openai_api, "openai"), (mistral_api, "mistral")]:
            print(f"\nTesting technical writing with {provider}")
            start_time = time.time()
            response = api.send_test_message(prompt=prompt)
            end_time = time.time()
            
            timing_results[provider] = end_time - start_time
            
            assert response is not None
            assert "response" in response
            doc = response["response"]
            
            # Basic validation of documentation
            assert any(word in doc.lower() for word in ["water level", "forecast", "prediction", "api"])
            assert any(word in doc.lower() for word in ["parameter", "response", "error", "example"])
            assert "curl" in doc.lower() or "http" in doc.lower()
            
            # Save the documentation
            output_file = os.path.join(self.output_dir, f"water_level_api_docs_{provider}.md")
            with open(output_file, "w") as f:
                f.write(doc)
            print(f"Documentation saved to: {output_file}")
        
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
    test = TestLangchainCapabilities()
    test.test_code_generation()
    test.test_math_reasoning()
    test.test_technical_writing() 