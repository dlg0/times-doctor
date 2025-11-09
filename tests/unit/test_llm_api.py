"""Unit tests for LLM API calls."""

import pytest
from unittest.mock import Mock, patch, MagicMock


class TestOpenAIResponsesAPI:
    """Test OpenAI GPT-5 Responses API integration."""
    
    @pytest.fixture
    def mock_gpt5_response(self):
        """Mock response from GPT-5 Responses API with correct output structure."""
        return {
            "id": "resp_123",
            "object": "response",
            "created_at": 1234567890,
            "model": "gpt-5-nano",
            "output": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "This is the compressed output from GPT-5."
                        }
                    ]
                }
            ],
            "usage": {
                "input_tokens": 1000,
                "output_tokens": 50,
                "total_tokens": 1050
            },
            "status": "completed"
        }
    
    @patch('times_doctor.core.llm.os.environ.get')
    @patch('httpx.post')
    def test_successful_response_parsing(self, mock_post, mock_env, mock_gpt5_response):
        """Test that we correctly parse a successful GPT-5 response."""
        from times_doctor.core.llm import _call_openai_responses_api
        
        # Setup environment
        mock_env.return_value = "sk-test-key"
        
        # Setup HTTP mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_gpt5_response
        mock_response.headers = {}
        mock_post.return_value = mock_response
        
        # Make API call
        text, metadata = _call_openai_responses_api("test prompt", model="gpt-5-nano")
        
        # Verify text extraction
        assert text == "This is the compressed output from GPT-5.", f"Got: {repr(text)}"
        assert metadata["model"] == "gpt-5-nano"
        assert metadata["input_tokens"] == 1000
        assert metadata["output_tokens"] == 50
        assert metadata["provider"] == "openai"
        assert "duration_seconds" in metadata
        assert "cost_usd" in metadata
    
    @patch('times_doctor.core.llm.os.environ.get')
    @patch('httpx.post')
    def test_empty_output_handling(self, mock_post, mock_env):
        """Test handling of empty output array."""
        from times_doctor.core.llm import _call_openai_responses_api
        
        mock_env.return_value = "sk-test-key"
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "output": [],
            "usage": {"input_tokens": 100, "output_tokens": 0, "total_tokens": 100}
        }
        mock_response.headers = {}
        mock_post.return_value = mock_response
        
        text, metadata = _call_openai_responses_api("test prompt")
        
        assert text == ""
        assert metadata["output_tokens"] == 0
    
    @patch('times_doctor.core.llm.os.environ.get')
    @patch('httpx.post')
    def test_empty_content_array(self, mock_post, mock_env):
        """Test handling when output exists but content array is empty.
        
        This reproduces the issue where GPT-5 returns:
        output: [{"id": "...", "type": "...", "summary": "...", "content": []}]
        
        The content array is empty, leading to text_content being empty string.
        """
        from times_doctor.core.llm import _call_openai_responses_api
        
        mock_env.return_value = "sk-test-key"
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "resp_456",
            "object": "response",
            "created_at": 1234567890,
            "model": "gpt-5-nano",
            "output": [
                {
                    "id": "msg_123",
                    "type": "message",
                    "summary": "Some summary text here",
                    "content": []  # Empty content array!
                }
            ],
            "usage": {
                "input_tokens": 115096,
                "output_tokens": 420,
                "total_tokens": 115516
            },
            "status": "completed"
        }
        mock_response.headers = {}
        mock_post.return_value = mock_response
        
        text, metadata = _call_openai_responses_api("test prompt", model="gpt-5-nano")
        
        # Should fall back to summary when content is empty
        assert text == "Some summary text here", f"Got empty text when summary was available: {repr(text)}"
        assert metadata["input_tokens"] == 115096
        assert metadata["output_tokens"] == 420
    
    @patch('times_doctor.core.llm.os.environ.get')
    @patch('httpx.post')
    def test_cost_calculation(self, mock_post, mock_env, mock_gpt5_response):
        """Test that cost is calculated correctly for gpt-5-nano."""
        from times_doctor.core.llm import _call_openai_responses_api
        
        mock_env.return_value = "sk-test-key"
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_gpt5_response
        mock_response.headers = {}
        mock_post.return_value = mock_response
        
        text, metadata = _call_openai_responses_api("test prompt", model="gpt-5-nano")
        
        # gpt-5-nano: $0.0001/1k input, $0.0004/1k output
        expected_cost = (1000 / 1000 * 0.0001) + (50 / 1000 * 0.0004)
        assert abs(metadata["cost_usd"] - expected_cost) < 0.0001, f"Expected {expected_cost}, got {metadata['cost_usd']}"
