"""Integration test for OpenAI API streaming functionality."""

import pytest
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file for API keys
load_dotenv()


class TestOpenAIStreaming:
    """Test OpenAI API streaming capabilities."""
    
    @pytest.fixture
    def check_openai_key(self):
        """Check if OpenAI API key is available."""
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not found in environment")
    
    def test_openai_streaming_basic(self, check_openai_key):
        """Test basic streaming response from OpenAI API.
        
        This test:
        1. Sends a simple prompt to OpenAI API with streaming enabled
        2. Collects streamed chunks via callback
        3. Verifies complete response is assembled correctly
        4. Validates metadata (tokens, cost, etc.)
        """
        from times_doctor.core.llm import _call_openai_api
        
        # Simple prompt for testing
        prompt = "Explain what linear programming is in one sentence."
        
        # Track streamed chunks
        chunks = []
        def stream_callback(content: str):
            chunks.append(content)
            # Show progress
            print(content, end="", flush=True)
        
        # Call with streaming
        result, metadata = _call_openai_api(
            prompt,
            model="gpt-5-nano",  # Fast, cheap model for testing
            stream_callback=stream_callback
        )
        
        # Verify result
        assert result, "Should return non-empty result"
        assert len(result) > 20, "Should return substantial response"
        
        # Verify chunks were received
        assert len(chunks) > 0, "Should receive streaming chunks"
        
        # Verify reassembly matches
        reassembled = "".join(chunks)
        assert reassembled.strip() == result, "Streamed chunks should match final result"
        
        # Verify metadata
        assert "model" in metadata
        assert "provider" in metadata
        assert metadata["provider"] == "openai"
        assert "input_tokens" in metadata
        assert "output_tokens" in metadata
        assert metadata["output_tokens"] > 0, "Should track output tokens"
        assert "cost_usd" in metadata
        assert metadata["cost_usd"] > 0, "Should calculate cost"
        
        # Print summary
        print(f"\n\n✅ Streaming test completed successfully")
        print(f"   Chunks received: {len(chunks)}")
        print(f"   Total response length: {len(result)} chars")
        print(f"   Input tokens: {metadata['input_tokens']}")
        print(f"   Output tokens: {metadata['output_tokens']}")
        print(f"   Cost: ${metadata['cost_usd']:.6f}")
    
    def test_openai_different_model_with_streaming(self, check_openai_key):
        """Test streaming with different model to verify it works across models.
        
        This test verifies that streaming works consistently with
        different GPT models.
        """
        from times_doctor.core.llm import _call_openai_api
        
        prompt = "Count from 1 to 3."
        
        chunks = []
        def stream_callback(content: str):
            chunks.append(content)
        
        # Call with gpt-5-mini for variety
        result, metadata = _call_openai_api(
            prompt,
            model="gpt-5-mini",
            stream_callback=stream_callback
        )
        
        # Verify result
        assert result, "Should return result"
        
        # Verify chunks were received
        assert len(chunks) > 0, "Should stream chunks"
        
        # Verify metadata
        assert "model" in metadata
        assert metadata["provider"] == "openai"
        assert metadata["output_tokens"] > 0
        
        print(f"\n✅ Different model streaming test passed")
        print(f"   Model: {metadata.get('model')}")
        print(f"   Chunks: {len(chunks)}")
    
    def test_openai_streaming_fallback_on_error(self, check_openai_key):
        """Test that streaming falls back to non-streaming on errors.
        
        If the org doesn't support streaming (requires verification),
        the API should gracefully fall back to non-streaming mode.
        """
        from times_doctor.core.llm import _call_openai_api
        
        prompt = "Say hello"
        
        # Try streaming (might fall back depending on org settings)
        result, metadata = _call_openai_api(
            prompt,
            model="gpt-5-nano",
            stream_callback=lambda x: None
        )
        
        # Should get result either way
        assert result, "Should return result (streaming or fallback)"
        assert "hello" in result.lower() or "hi" in result.lower()
        
        # Should have metadata
        assert "model" in metadata
        assert "provider" in metadata
        
        print(f"\n✅ Fallback handling test passed")
        print(f"   Result received: {result[:50]}...")
    
    def test_openai_streaming_long_response(self, check_openai_key):
        """Test streaming with a longer response to verify chunking works.
        
        This ensures that the streaming mechanism properly handles
        responses that come in multiple chunks.
        """
        from times_doctor.core.llm import _call_openai_api
        
        # Prompt that generates longer response
        prompt = "List the first 5 prime numbers and explain what a prime number is."
        
        chunks = []
        chunk_sizes = []
        
        def stream_callback(content: str):
            chunks.append(content)
            chunk_sizes.append(len(content))
        
        result, metadata = _call_openai_api(
            prompt,
            model="gpt-5-nano",
            stream_callback=stream_callback
        )
        
        # Verify result
        assert result, "Should return result"
        assert len(result) > 100, "Should be substantial response"
        
        # Verify multiple chunks received
        assert len(chunks) > 3, "Should receive multiple chunks for longer response"
        
        # Verify chunks combine correctly
        reassembled = "".join(chunks)
        assert reassembled.strip() == result
        
        print(f"\n✅ Long response streaming test passed")
        print(f"   Total chunks: {len(chunks)}")
        print(f"   Chunk sizes: {chunk_sizes}")
        print(f"   Total length: {len(result)} chars")
        print(f"   Average chunk size: {sum(chunk_sizes) / len(chunk_sizes):.1f} chars")
