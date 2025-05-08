import pytest
from unittest.mock import patch
from dllmforge.ollama_api import OllamaAPI
import json


class TestOllamaAPI:

    # skip if the test is run on github
    @pytest.mark.skip(reason="Skip test on GitHub Actions, you can run it locally.")
    def test_localy(self):
        # check in your local Deltares machine if the API is running with the VPN on
        api = OllamaAPI()
        models = api.list_available_models()
        assert models is not None, "No models found. Check if the API is running."
        assert isinstance(models, list), "Models should be a list."
        assert len(models) > 0, "No models found. Check if the API is running."
        # define test prompt for all models
        test_prompt = "Create a simple HTML webpage with a greeting message and a background color of your choice. Give me only the HTML code so I can save it in a file immidietly. No other text is needed."
        duration = []
        for model in models[1:]:
            assert isinstance(model, str), "Model names should be strings."
            api.model = model
            response = api.send_test_message(prompt=test_prompt)
            # write the response to a file
            model = model.split(":")[0]
            with open(f"tests/test_output/response_{model}.html", "w") as f:
                f.write(response['response'])
            duration.append(response['prompt_eval_duration'])

    @patch("requests.head")
    def test_check_server_status(self, mock_head):
        # Mock successful server status check
        mock_head.return_value.status_code = 200
        api = OllamaAPI()
        assert api.check_server_status() is True

        # Mock failed server status check
        mock_head.return_value.status_code = 500
        assert api.check_server_status() is False

    @patch("requests.get")
    def test_list_available_models(self, mock_get):
        # Mock successful model listing
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = json.dumps({"models": [{"name": "model1"}, {"name": "model2"}]})
        api = OllamaAPI()
        models = api.list_available_models()
        assert models == ["model1", "model2"]

        # Mock failed model listing
        mock_get.return_value.status_code = 500
        assert api.list_available_models() is None

    @patch("requests.post")
    def test_send_test_message(self, mock_post):
        # Mock successful message sending
        mock_post.return_value.status_code = 200
        mock_post.return_value.text = json.dumps({"response": "Test response"})
        api = OllamaAPI()
        response = api.send_test_message(prompt="Test prompt")
        assert response == {"response": "Test response"}

        # Mock failed message sending
        mock_post.return_value.status_code = 500
        assert api.send_test_message(prompt="Test prompt") is None