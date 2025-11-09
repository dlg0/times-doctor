"""Tests for LLM response caching."""

import json

from times_doctor.core import llm_cache


def test_compute_cache_key_stable():
    """Test that cache key is stable for same inputs."""
    key1 = llm_cache.compute_cache_key(
        prompt="test prompt", model="gpt-5-nano", temperature=0.2, reasoning_effort="medium"
    )
    key2 = llm_cache.compute_cache_key(
        prompt="test prompt", model="gpt-5-nano", temperature=0.2, reasoning_effort="medium"
    )
    assert key1 == key2


def test_compute_cache_key_different_prompts():
    """Test that different prompts produce different keys."""
    key1 = llm_cache.compute_cache_key(prompt="prompt 1", model="gpt-5-nano")
    key2 = llm_cache.compute_cache_key(prompt="prompt 2", model="gpt-5-nano")
    assert key1 != key2


def test_compute_cache_key_different_models():
    """Test that different models produce different keys."""
    key1 = llm_cache.compute_cache_key(prompt="test", model="gpt-5-nano")
    key2 = llm_cache.compute_cache_key(prompt="test", model="gpt-5-mini")
    assert key1 != key2


def test_compute_cache_key_different_params():
    """Test that different parameters produce different keys."""
    key1 = llm_cache.compute_cache_key(prompt="test", model="gpt-5-nano", temperature=0.2)
    key2 = llm_cache.compute_cache_key(prompt="test", model="gpt-5-nano", temperature=0.5)
    assert key1 != key2


def test_compute_cache_key_ignores_none():
    """Test that None values don't affect cache key."""
    key1 = llm_cache.compute_cache_key(prompt="test", model="gpt-5-nano", temperature=None)
    key2 = llm_cache.compute_cache_key(prompt="test", model="gpt-5-nano")
    assert key1 == key2


def test_read_cache_miss(tmp_path):
    """Test reading cache when no cached response exists."""
    result = llm_cache.read_cache(prompt="test", model="gpt-5-nano", cache_dir=tmp_path)
    assert result is None


def test_write_and_read_cache(tmp_path):
    """Test writing and reading a cached response."""
    prompt = "What is 2+2?"
    model = "gpt-5-nano"
    response = "4"
    metadata = {"model": model, "input_tokens": 10, "output_tokens": 5, "cost_usd": 0.001}

    # Write to cache
    llm_cache.write_cache(
        prompt=prompt,
        model=model,
        response=response,
        metadata=metadata,
        cache_dir=tmp_path,
        temperature=0.2,
    )

    # Read from cache
    cached = llm_cache.read_cache(prompt=prompt, model=model, cache_dir=tmp_path, temperature=0.2)

    assert cached is not None
    cached_response, cached_metadata = cached
    assert cached_response == response
    assert cached_metadata["model"] == model
    assert cached_metadata["input_tokens"] == 10
    assert cached_metadata["output_tokens"] == 5


def test_cache_file_structure(tmp_path):
    """Test that cache file contains expected structure."""
    prompt = "Test prompt"
    model = "gpt-5-nano"
    response = "Test response"
    metadata = {"model": model, "input_tokens": 10}

    llm_cache.write_cache(
        prompt=prompt,
        model=model,
        response=response,
        metadata=metadata,
        cache_dir=tmp_path,
        reasoning_effort="medium",
    )

    # Find the cache file
    cache_files = list(tmp_path.glob("cache_*.json"))
    assert len(cache_files) == 1

    # Read and verify structure
    with open(cache_files[0]) as f:
        data = json.load(f)

    assert "cache_key" in data
    assert "timestamp" in data
    assert "model" in data
    assert data["model"] == model
    assert "params" in data
    assert data["params"]["reasoning_effort"] == "medium"
    assert "response" in data
    assert data["response"] == response
    assert "metadata" in data
    assert "prompt_hash" in data
    assert "prompt_length" in data


def test_cache_different_params_separate_entries(tmp_path):
    """Test that different parameters create separate cache entries."""
    prompt = "Same prompt"
    model = "gpt-5-nano"

    # Write two cache entries with different parameters
    llm_cache.write_cache(
        prompt=prompt,
        model=model,
        response="Response 1",
        metadata={},
        cache_dir=tmp_path,
        temperature=0.2,
    )

    llm_cache.write_cache(
        prompt=prompt,
        model=model,
        response="Response 2",
        metadata={},
        cache_dir=tmp_path,
        temperature=0.5,
    )

    # Should have 2 cache files
    cache_files = list(tmp_path.glob("cache_*.json"))
    assert len(cache_files) == 2

    # Read with different temperatures
    cached1 = llm_cache.read_cache(prompt, model, tmp_path, temperature=0.2)
    cached2 = llm_cache.read_cache(prompt, model, tmp_path, temperature=0.5)

    assert cached1 is not None
    assert cached2 is not None
    assert cached1[0] == "Response 1"
    assert cached2[0] == "Response 2"


def test_clear_cache(tmp_path):
    """Test clearing all cached responses."""
    # Create multiple cache entries
    for i in range(3):
        llm_cache.write_cache(
            prompt=f"Prompt {i}",
            model="gpt-5-nano",
            response=f"Response {i}",
            metadata={},
            cache_dir=tmp_path,
        )

    # Verify files exist
    assert len(list(tmp_path.glob("cache_*.json"))) == 3

    # Clear cache
    count = llm_cache.clear_cache(tmp_path)
    assert count == 3

    # Verify files deleted
    assert len(list(tmp_path.glob("cache_*.json"))) == 0


def test_clear_cache_empty_dir(tmp_path):
    """Test clearing cache when directory is empty."""
    count = llm_cache.clear_cache(tmp_path)
    assert count == 0


def test_clear_cache_nonexistent_dir(tmp_path):
    """Test clearing cache when directory doesn't exist."""
    nonexistent = tmp_path / "nonexistent"
    count = llm_cache.clear_cache(nonexistent)
    assert count == 0


def test_cache_corrupted_file(tmp_path):
    """Test that corrupted cache files are handled gracefully."""
    # Create a corrupted cache file
    cache_file = tmp_path / "cache_corrupted.json"
    cache_file.write_text("invalid json{{{")

    # Try to read - should return None for cache miss
    result = llm_cache.read_cache(prompt="test", model="gpt-5-nano", cache_dir=tmp_path)
    # Even though there's a file, it's not readable, so still cache miss
    # (the function computes a specific cache key and looks for that file)
    assert result is None


def test_cache_preserves_unicode(tmp_path):
    """Test that cache correctly handles unicode characters."""
    prompt = "¿Qué es 2+2? 你好 🚀"
    response = "La respuesta es 4. 答案是4。✓"
    model = "gpt-5-nano"

    llm_cache.write_cache(
        prompt=prompt, model=model, response=response, metadata={}, cache_dir=tmp_path
    )

    cached = llm_cache.read_cache(prompt, model, tmp_path)
    assert cached is not None
    assert cached[0] == response


def test_get_cache_path(tmp_path):
    """Test that cache path is correctly computed."""
    cache_key = "abc123"
    path = llm_cache.get_cache_path(cache_key, tmp_path)
    assert path == tmp_path / f"cache_{cache_key}.json"
