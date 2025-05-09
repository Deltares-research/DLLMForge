import pytest
import os
import time
from dllmforge.openai_api import OpenAIAPI

class TestMultipleModels:
    @pytest.mark.skip(reason="Skip test on GitHub Actions, you can run it locally.")
    def test_html_generation(self):
        """Test HTML generation with multiple models and save outputs."""
        api = OpenAIAPI()
        
        # Define the models to test
        models = [
            "gpt-4o",
            "o1",
            "o4-mini",
            "gpt-4",
            "gpt-4.1-mini"
        ]
        
        # Common prompt for all models
        prompt = """Create a simple HTML webpage with a greeting message and a background color of your choice. 
        Give me only the HTML code so I can save it in a file immediately. 
        No other text is needed. I want text that is inspiring and water management related."""
        
        # Create output directory if it doesn't exist
        output_dir = "model_outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        # Dictionary to store timing results
        timing_results = {}
        
        # Test each model
        for model in models:
            print(f"\nTesting model: {model}")
            
            # Set the model
            api.model = model
            
            # Start timing
            start_time = time.time()
            
            # Get the response
            response = api.send_test_message(prompt=prompt)
            
            # End timing
            end_time = time.time()
            response_time = end_time - start_time
            timing_results[model] = response_time
            
            assert response is not None, f"Failed to get response from model {model}"
            assert "response" in response, f"Response should contain 'response' field for model {model}"
            
            # Extract the HTML content
            html_content = response["response"]
            
            # Save to file
            output_file = os.path.join(output_dir, f"output_{model}.html")
            with open(output_file, "w") as f:
                f.write(html_content)
            
            print(f"Output saved to: {output_file}")
            print(f"Response time: {response_time:.2f} seconds")
            
            # Verify the file was created
            assert os.path.exists(output_file), f"Output file for model {model} was not created"
            
            # # Basic content verification
            # assert "Hello World" in html_content, f"HTML content for model {model} should contain 'Hello World'"
            # assert "background-color: red" in html_content.lower() or "background: red" in html_content.lower(), f"HTML content for model {model} should have red background"
            # assert "color: white" in html_content.lower(), f"HTML content for model {model} should have white text"
            
            print(f"Model {model} test completed successfully")
        
        # Print timing summary
        print("\nTiming Summary:")
        print("-" * 40)
        for model, time_taken in timing_results.items():
            print(f"{model}: {time_taken:.2f} seconds")
        print("-" * 40)

if __name__ == "__main__":
    # This allows running the test directly with python
    test = TestMultipleModels()
    test.test_html_generation() 