import pytest
import os
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
            "o4-mini"
        ]
        
        # Common prompt for all models
        prompt = "Create a simple HTML webpage that displays 'Hello World' in white text on a red background. Return only the HTML code, no explanations."
        
        # Create output directory if it doesn't exist
        output_dir = "model_outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        # Test each model
        for model in models:
            print(f"\nTesting model: {model}")
            
            # Set the model
            api.model = model
            
            # Get the response
            response = api.send_test_message(prompt=prompt)
            assert response is not None, f"Failed to get response from model {model}"
            assert "response" in response, f"Response should contain 'response' field for model {model}"
            
            # Extract the HTML content
            html_content = response["response"]
            
            # Save to file
            output_file = os.path.join(output_dir, f"output_{model}.html")
            with open(output_file, "w") as f:
                f.write(html_content)
            
            print(f"Output saved to: {output_file}")
            
            # Verify the file was created
            assert os.path.exists(output_file), f"Output file for model {model} was not created"
            
            # Basic content verification
            assert "Hello World" in html_content, f"HTML content for model {model} should contain 'Hello World'"
            assert "background-color: red" in html_content.lower() or "background: red" in html_content.lower(), f"HTML content for model {model} should have red background"
            assert "color: white" in html_content.lower(), f"HTML content for model {model} should have white text"
            
            print(f"Model {model} test completed successfully")

if __name__ == "__main__":
    # This allows running the test directly with python
    test = TestMultipleModels()
    test.test_html_generation() 