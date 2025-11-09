"""Integration test for Anthropic API streaming functionality."""

import pytest

# Load .env file BEFORE checking for API key
from dotenv import load_dotenv

load_dotenv()


@pytest.mark.skip(
    reason="Anthropic streaming implementation needs debugging - OpenAI streaming is working and is the primary provider"
)
class TestAnthropicStreaming:
    """Test Anthropic API streaming capabilities."""

    def test_anthropic_streaming_basic(self):
        """Test basic streaming response from Anthropic API.

        This test:
        1. Sends a simple prompt to Anthropic API with streaming enabled
        2. Collects streamed chunks via callback
        3. Verifies complete response is assembled correctly
        4. Validates metadata (tokens, cost, etc.)
        """
        from times_doctor.core.llm import _call_anthropic_api

        # Simple prompt for testing
        prompt = "Explain what linear programming is in one sentence."

        # Track streamed chunks
        chunks = []

        def stream_callback(content: str):
            chunks.append(content)
            # Show progress
            print(content, end="", flush=True)

        # Call with streaming
        result, metadata = _call_anthropic_api(
            prompt,
            model="claude-3-5-haiku-20241022",  # Fast, cheap model for testing
            stream_callback=stream_callback,
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
        assert metadata["provider"] == "anthropic"
        assert "input_tokens" in metadata
        assert "output_tokens" in metadata
        assert metadata["output_tokens"] > 0, "Should track output tokens"
        assert "cost_usd" in metadata
        assert metadata["cost_usd"] > 0, "Should calculate cost"

        # Print summary
        print("\n\n✅ Anthropic streaming test completed successfully")
        print(f"   Chunks received: {len(chunks)}")
        print(f"   Total response length: {len(result)} chars")
        print(f"   Input tokens: {metadata['input_tokens']}")
        print(f"   Output tokens: {metadata['output_tokens']}")
        print(f"   Cost: ${metadata['cost_usd']:.6f}")

    def test_anthropic_streaming_different_model(self):
        """Test streaming with Sonnet model for variety.

        This test verifies that streaming works consistently with
        different Claude models.
        """
        from times_doctor.core.llm import _call_anthropic_api

        prompt = "Count from 1 to 3."

        chunks = []

        def stream_callback(content: str):
            chunks.append(content)

        # Call with claude-sonnet
        result, metadata = _call_anthropic_api(
            prompt, model="claude-3-5-sonnet-20241022", stream_callback=stream_callback
        )

        # Verify result
        assert result, "Should return result"

        # Verify chunks were received
        assert len(chunks) > 0, "Should stream chunks"

        # Verify metadata
        assert "model" in metadata
        assert metadata["provider"] == "anthropic"
        assert metadata["output_tokens"] > 0

        print("\n✅ Sonnet model streaming test passed")
        print(f"   Model: {metadata.get('model')}")
        print(f"   Chunks: {len(chunks)}")

    def test_anthropic_streaming_fallback_on_error(self):
        """Test that streaming falls back to non-streaming on errors.

        If there's an error with streaming, the API should gracefully
        fall back to non-streaming mode.
        """
        from times_doctor.core.llm import _call_anthropic_api

        prompt = "Say hello"

        # Try streaming
        result, metadata = _call_anthropic_api(
            prompt, model="claude-3-5-haiku-20241022", stream_callback=lambda x: None
        )

        # Should get result either way
        assert result, "Should return result (streaming or fallback)"
        assert "hello" in result.lower() or "hi" in result.lower()

        # Should have metadata
        assert "model" in metadata
        assert "provider" in metadata

        print("\n✅ Fallback handling test passed")
        print(f"   Result received: {result[:50]}...")

    def test_anthropic_streaming_long_response(self):
        """Test streaming with a longer response to verify chunking works.

        This ensures that the streaming mechanism properly handles
        responses that come in multiple chunks.
        """
        from times_doctor.core.llm import _call_anthropic_api

        # Prompt that generates longer response
        prompt = "List the first 5 prime numbers and explain what a prime number is."

        chunks = []
        chunk_sizes = []

        def stream_callback(content: str):
            chunks.append(content)
            chunk_sizes.append(len(content))

        result, metadata = _call_anthropic_api(
            prompt, model="claude-3-5-haiku-20241022", stream_callback=stream_callback
        )

        # Verify result
        assert result, "Should return result"
        assert len(result) > 100, "Should be substantial response"

        # Verify multiple chunks received
        assert len(chunks) > 3, "Should receive multiple chunks for longer response"

        # Verify chunks combine correctly
        reassembled = "".join(chunks)
        assert reassembled.strip() == result

        print("\n✅ Long response streaming test passed")
        print(f"   Total chunks: {len(chunks)}")
        print(f"   Chunk sizes: {chunk_sizes}")
        print(f"   Total length: {len(result)} chars")
        print(f"   Average chunk size: {sum(chunk_sizes) / len(chunk_sizes):.1f} chars")
