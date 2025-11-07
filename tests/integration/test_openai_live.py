"""Integration tests for OpenAI API calls with real data.

These tests make actual API calls and are skipped if OPENAI_API_KEY is not set.
"""

import pytest
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file for API keys
load_dotenv()


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set - skipping live API tests"
)
class TestOpenAILiveAPI:
    """Test real OpenAI API calls with actual QA_CHECK.LOG data."""
    
    def test_compress_qa_check_with_real_data(self, tmp_path):
        """Test compression of real QA_CHECK.LOG data using OpenAI API.
        
        This test:
        1. Loads a sample from the actual QA_CHECK.LOG file
        2. Uses the real compression prompt
        3. Sends it to OpenAI API
        4. Verifies that the response contains text
        5. Logs both request and response to disk for inspection
        """
        from times_doctor.llm import _call_openai_responses_api, log_llm_call
        from times_doctor.prompts import build_qa_check_compress_prompt
        
        # Find the QA_CHECK.LOG file
        data_dir = Path(__file__).parent.parent.parent / "data"
        qa_check_files = list(data_dir.rglob("QA_CHECK.LOG"))
        
        assert len(qa_check_files) > 0, "No QA_CHECK.LOG files found in data/ directory"
        
        qa_check_path = qa_check_files[0]
        
        # Read first 50 lines (enough for a meaningful test without using too many tokens)
        with open(qa_check_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [f.readline() for _ in range(50)]
        
        sample_content = ''.join(lines)
        
        # Build the compression prompt using the actual prompt builder
        prompt = build_qa_check_compress_prompt(sample_content)
        
        # Create a log directory for this test
        log_dir = tmp_path / "_llm_calls"
        log_dir.mkdir(exist_ok=True)
        
        # Make the actual API call
        text, metadata = _call_openai_responses_api(
            prompt, 
            model="gpt-5-nano", 
            reasoning_effort="minimal"
        )
        
        # Log the call for inspection
        log_llm_call("test_compress_qa_check", prompt, text, metadata, log_dir)
        
        # Verify the response
        assert text, "Response text should not be empty"
        assert len(text) > 0, "Response text should have content"
        assert metadata.get("model") == "gpt-5-nano", "Model should be gpt-5-nano"
        assert metadata.get("input_tokens", 0) > 0, "Should have input tokens"
        assert metadata.get("output_tokens", 0) > 0, "Should have output tokens"
        
        # Verify log file was created
        log_files = list(log_dir.glob("*.json"))
        assert len(log_files) > 0, "Should have created at least one log file"
        
        # Read and verify log file structure
        import json
        with open(log_files[0], 'r') as f:
            log_data = json.load(f)
        
        assert "timestamp" in log_data
        assert "call_type" in log_data
        assert "prompt" in log_data
        assert "response" in log_data
        assert "metadata" in log_data
        assert log_data["call_type"] == "test_compress_qa_check"
        assert log_data["response"] == text
        assert log_data["prompt"] == prompt
        
        # Print summary for visibility
        print(f"\n✅ API call successful")
        print(f"   Input tokens: {metadata.get('input_tokens')}")
        print(f"   Output tokens: {metadata.get('output_tokens')}")
        print(f"   Cost: ${metadata.get('cost_usd', 0):.4f}")
        print(f"   Response length: {len(text)} chars")
        print(f"   Log saved to: {log_files[0]}")
        print(f"\n📝 Response preview (first 200 chars):")
        print(f"   {text[:200]}...")
